"""
test_github_api.py
===================

Mocked unit tests for GitHubTools REST API read methods.

Only HTTP is mocked — no network access is required.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

from codebase_assistant.exceptions.model_exceptions import RateLimitError
from codebase_assistant.exceptions.tool_exceptions import (
    InvalidRepositoryURLError,
    ToolExecutionError,
    UnsupportedFileTypeError,
)
from codebase_assistant.tools.github_tools import GitHubTools


def _response(
    status: int,
    payload: Any = None,
    *,
    text: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> MagicMock:
    """Build a fake requests.Response-like object."""
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {}
    if text is not None:
        response.text = text
    elif payload is None:
        response.text = ""
    else:
        import json

        response.text = json.dumps(payload)
    response.json.return_value = payload
    return response


@pytest.fixture
def tools() -> GitHubTools:
    """Anonymous GitHubTools instance (no token)."""
    return GitHubTools(token=None, repo=None)


@pytest.fixture
def authed_tools() -> GitHubTools:
    """Authenticated GitHubTools instance."""
    return GitHubTools(token="ghp_test_token", repo=None)


def test_get_file_contents_success(tools: GitHubTools) -> None:
    """Decoded UTF-8 file contents are returned."""
    content = "print('hello')\n"
    payload = {
        "type": "file",
        "encoding": "base64",
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(200, payload),
    ) as mock_request:
        result = tools.get_file_contents("pallets/flask", "README.md", ref="main")

    assert result == content
    kwargs = mock_request.call_args.kwargs
    assert mock_request.call_args.args[0] == "GET"
    assert "pallets/flask/contents/README.md" in mock_request.call_args.args[1]
    assert kwargs["params"] == {"ref": "main"}
    assert "Authorization" not in kwargs["headers"]


def test_get_file_contents_with_token_sends_authorization(
    authed_tools: GitHubTools,
) -> None:
    """Token-backed calls include an Authorization bearer header."""
    payload = {
        "type": "file",
        "encoding": "base64",
        "content": base64.b64encode(b"ok").decode("ascii"),
    }
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(200, payload),
    ) as mock_request:
        authed_tools.get_file_contents("owner/repo", "a.py")

    headers = mock_request.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer ghp_test_token"


def test_list_issues_success_excludes_pull_requests(tools: GitHubTools) -> None:
    """Issues are mapped and PRs mixed into the issues API are dropped."""
    payload = [
        {
            "number": 1,
            "title": "Bug",
            "state": "open",
            "labels": [{"name": "bug"}],
            "user": {"login": "alice"},
            "html_url": "https://github.com/o/r/issues/1",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        },
        {
            "number": 2,
            "title": "A PR disguised as an issue",
            "state": "open",
            "labels": [],
            "user": {"login": "bob"},
            "html_url": "https://github.com/o/r/pull/2",
            "created_at": "2024-01-03T00:00:00Z",
            "updated_at": "2024-01-03T00:00:00Z",
            "pull_request": {"url": "https://api.github.com/repos/o/r/pulls/2"},
        },
    ]
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(200, payload),
    ):
        issues = tools.list_issues("o/r")

    assert len(issues) == 1
    assert issues[0] == {
        "number": 1,
        "title": "Bug",
        "state": "open",
        "labels": ["bug"],
        "author": "alice",
        "url": "https://github.com/o/r/issues/1",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
    }


def test_list_pull_requests_success(tools: GitHubTools) -> None:
    """Pull requests map source/target branches and author."""
    payload = [
        {
            "number": 7,
            "title": "Add feature",
            "state": "open",
            "user": {"login": "carol"},
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "html_url": "https://github.com/o/r/pull/7",
            "created_at": "2024-02-01T00:00:00Z",
        }
    ]
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(200, payload),
    ):
        pulls = tools.list_pull_requests("https://github.com/o/r")

    assert pulls == [
        {
            "number": 7,
            "title": "Add feature",
            "state": "open",
            "author": "carol",
            "source_branch": "feature",
            "target_branch": "main",
            "url": "https://github.com/o/r/pull/7",
            "created_at": "2024-02-01T00:00:00Z",
        }
    ]


def test_repository_not_found(tools: GitHubTools) -> None:
    """HTTP 404 becomes InvalidRepositoryURLError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(404, text="Not Found"),
    ):
        with pytest.raises(InvalidRepositoryURLError):
            tools.list_issues("missing/repo")


def test_file_not_found(tools: GitHubTools) -> None:
    """Missing file path raises InvalidRepositoryURLError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(404, text="Not Found"),
    ):
        with pytest.raises(InvalidRepositoryURLError):
            tools.get_file_contents("o/r", "nope.py")


def test_rate_limiting_403(tools: GitHubTools) -> None:
    """403 with exhausted rate-limit header raises RateLimitError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(
            403,
            text="API rate limit exceeded",
            headers={"X-RateLimit-Remaining": "0"},
        ),
    ):
        with pytest.raises(RateLimitError):
            tools.list_pull_requests("o/r")


def test_rate_limiting_429(tools: GitHubTools) -> None:
    """HTTP 429 raises RateLimitError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(429, text="Retry later"),
    ):
        with pytest.raises(RateLimitError):
            tools.get_file_contents("o/r", "a.py")


def test_authentication_failure(authed_tools: GitHubTools) -> None:
    """HTTP 401 raises ToolExecutionError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(401, text="Bad credentials"),
    ):
        with pytest.raises(ToolExecutionError, match="authentication"):
            authed_tools.list_issues("o/r")


def test_timeout(tools: GitHubTools) -> None:
    """requests.Timeout becomes ToolExecutionError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        side_effect=requests.Timeout("slow"),
    ):
        with pytest.raises(ToolExecutionError, match="timed out"):
            tools.get_file_contents("o/r", "a.py")


def test_connection_failure(tools: GitHubTools) -> None:
    """requests.ConnectionError becomes ToolExecutionError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        side_effect=requests.ConnectionError("offline"),
    ):
        with pytest.raises(ToolExecutionError, match="connection failed"):
            tools.list_issues("o/r")


def test_binary_file_handling(tools: GitHubTools) -> None:
    """Non-UTF-8 file bytes raise UnsupportedFileTypeError."""
    payload = {
        "type": "file",
        "encoding": "base64",
        "content": base64.b64encode(b"\xff\xfe\x00\x01").decode("ascii"),
    }
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(200, payload),
    ):
        with pytest.raises(UnsupportedFileTypeError, match="binary"):
            tools.get_file_contents("o/r", "blob.bin")


def test_anonymous_access_omits_authorization(tools: GitHubTools) -> None:
    """No token means no Authorization header."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(200, []),
    ) as mock_request:
        tools.list_issues("o/r")

    assert "Authorization" not in mock_request.call_args.kwargs["headers"]


def test_requests_exceptions_are_not_raised_raw(tools: GitHubTools) -> None:
    """Generic RequestException is wrapped, never leaked."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        side_effect=requests.RequestException("boom"),
    ):
        with pytest.raises(ToolExecutionError):
            tools.list_pull_requests("o/r")


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def test_create_branch_success(authed_tools: GitHubTools) -> None:
    """Resolve source SHA then create refs/heads/<branch>."""
    responses = [
        _response(
            200,
            {
                "ref": "refs/heads/main",
                "object": {"sha": "abc123", "type": "commit"},
            },
        ),
        _response(
            201,
            {
                "ref": "refs/heads/feature",
                "object": {"sha": "abc123"},
                "url": "https://api.github.com/repos/o/r/git/refs/heads/feature",
            },
        ),
    ]
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        side_effect=responses,
    ) as mock_request:
        result = authed_tools.create_branch("o/r", "feature", source_branch="main")

    assert result == {
        "branch": "feature",
        "sha": "abc123",
        "url": "https://api.github.com/repos/o/r/git/refs/heads/feature",
    }
    assert mock_request.call_count == 2
    assert mock_request.call_args_list[0].args[0] == "GET"
    assert mock_request.call_args_list[1].args[0] == "POST"
    assert mock_request.call_args_list[1].kwargs["json"] == {
        "ref": "refs/heads/feature",
        "sha": "abc123",
    }


def test_create_existing_branch(authed_tools: GitHubTools) -> None:
    """Creating a branch that already exists raises a validation error."""
    responses = [
        _response(
            200,
            {"ref": "refs/heads/main", "object": {"sha": "abc123"}},
        ),
        _response(422, text='{"message":"Reference already exists"}'),
    ]
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        side_effect=responses,
    ):
        with pytest.raises(ToolExecutionError, match="validation error"):
            authed_tools.create_branch("o/r", "feature")


def test_commit_new_file(authed_tools: GitHubTools) -> None:
    """Missing file → PUT create without sha; created=True."""
    responses = [
        _response(404, text="Not Found"),
        _response(
            201,
            {
                "content": {
                    "html_url": "https://github.com/o/r/blob/main/new.py",
                },
                "commit": {
                    "sha": "commit1",
                    "html_url": "https://github.com/o/r/commit/commit1",
                },
            },
        ),
    ]
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        side_effect=responses,
    ) as mock_request:
        result = authed_tools.commit_file(
            "o/r",
            "new.py",
            "print(1)\n",
            "add new.py",
            "main",
        )

    assert result["created"] is True
    assert result["commit_sha"] == "commit1"
    assert result["commit_url"] == "https://github.com/o/r/commit/commit1"
    assert result["content_url"] == "https://github.com/o/r/blob/main/new.py"
    put_body = mock_request.call_args_list[1].kwargs["json"]
    assert put_body["message"] == "add new.py"
    assert put_body["branch"] == "main"
    assert "sha" not in put_body
    assert base64.b64decode(put_body["content"]).decode("utf-8") == "print(1)\n"


def test_update_existing_file(authed_tools: GitHubTools) -> None:
    """Existing file → PUT update includes blob sha; created=False."""
    responses = [
        _response(
            200,
            {
                "type": "file",
                "sha": "blobsha",
                "encoding": "base64",
                "content": base64.b64encode(b"old").decode("ascii"),
            },
        ),
        _response(
            200,
            {
                "content": {"html_url": "https://github.com/o/r/blob/main/a.py"},
                "commit": {
                    "sha": "commit2",
                    "html_url": "https://github.com/o/r/commit/commit2",
                },
            },
        ),
    ]
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        side_effect=responses,
    ) as mock_request:
        result = authed_tools.commit_file(
            "o/r", "a.py", "new", "update a.py", "main"
        )

    assert result["created"] is False
    assert result["commit_sha"] == "commit2"
    put_body = mock_request.call_args_list[1].kwargs["json"]
    assert put_body["sha"] == "blobsha"


def test_create_pull_request_success(authed_tools: GitHubTools) -> None:
    """PR create maps number/title/url/state/head/base."""
    payload = {
        "number": 9,
        "title": "Ship it",
        "html_url": "https://github.com/o/r/pull/9",
        "state": "open",
        "head": {"ref": "feature"},
        "base": {"ref": "main"},
    }
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(201, payload),
    ) as mock_request:
        result = authed_tools.create_pull_request(
            "o/r",
            "Ship it",
            "details",
            "feature",
            base="main",
        )

    assert result == {
        "number": 9,
        "title": "Ship it",
        "url": "https://github.com/o/r/pull/9",
        "state": "open",
        "head": "feature",
        "base": "main",
    }
    assert mock_request.call_args.kwargs["json"] == {
        "title": "Ship it",
        "body": "details",
        "head": "feature",
        "base": "main",
    }


def test_write_authentication_failure(authed_tools: GitHubTools) -> None:
    """HTTP 401 on write raises ToolExecutionError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(401, text="Bad credentials"),
    ):
        with pytest.raises(ToolExecutionError, match="authentication"):
            authed_tools.create_pull_request("o/r", "t", "", "feature")


def test_write_validation_error(authed_tools: GitHubTools) -> None:
    """HTTP 422 on PR create raises ToolExecutionError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(422, text='{"message":"Validation Failed"}'),
    ):
        with pytest.raises(ToolExecutionError, match="validation error"):
            authed_tools.create_pull_request("o/r", "t", "", "feature")


def test_write_repository_not_found(authed_tools: GitHubTools) -> None:
    """HTTP 404 on write raises InvalidRepositoryURLError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        return_value=_response(404, text="Not Found"),
    ):
        with pytest.raises(InvalidRepositoryURLError):
            authed_tools.create_branch("missing/repo", "feature")


def test_write_timeout(authed_tools: GitHubTools) -> None:
    """Timeouts are wrapped as ToolExecutionError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        side_effect=requests.Timeout("slow"),
    ):
        with pytest.raises(ToolExecutionError, match="timed out"):
            authed_tools.commit_file("o/r", "a.py", "x", "m", "main")


def test_write_connection_error(authed_tools: GitHubTools) -> None:
    """Connection failures are wrapped as ToolExecutionError."""
    with patch(
        "codebase_assistant.tools.github_tools.requests.request",
        side_effect=requests.ConnectionError("offline"),
    ):
        with pytest.raises(ToolExecutionError, match="connection failed"):
            authed_tools.create_pull_request("o/r", "t", "", "feature")


def test_write_operations_reuse_api_helper(authed_tools: GitHubTools) -> None:
    """Write methods go through _api_request rather than raw HTTP."""
    with patch.object(
        authed_tools,
        "_api_request",
        return_value={
            "number": 1,
            "title": "t",
            "html_url": "https://github.com/o/r/pull/1",
            "state": "open",
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
        },
    ) as mock_helper:
        authed_tools.create_pull_request("o/r", "t", "b", "feature")

    mock_helper.assert_called_once()
    assert mock_helper.call_args.args[0] == "POST"
    assert mock_helper.call_args.args[1] == "/repos/o/r/pulls"
