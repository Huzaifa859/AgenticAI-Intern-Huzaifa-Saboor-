"""
github_tools.py
================

Acquiring the repository to be analyzed, plus read-only GitHub REST API
helpers for file contents, issues, and pull requests.

Public HTTPS GitHub URLs can be cloned without authentication. API
reads use an optional GitHub token from Config when present, and fall
back to anonymous access for public repositories.
"""

from __future__ import annotations

import base64
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

import requests

from ..config import Config
from ..exceptions.model_exceptions import RateLimitError
from ..exceptions.tool_exceptions import (
    InvalidRepositoryURLError,
    RepositoryCloneError,
    ToolExecutionError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)

# Hosts accepted as GitHub. Anything else is rejected rather than
# silently attempted.
GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})

# Schemes recognized as remote. Single-character schemes are excluded so
# a Windows drive letter ("C:\repo") is treated as a local path.
REMOTE_SCHEMES = frozenset({"http", "https", "git", "ssh"})

# Ceiling on how long a clone may run before being abandoned.
# TODO: move onto Config alongside the other operational limits.
CLONE_TIMEOUT_SECONDS = 300

#: Default timeout for GitHub REST API calls (seconds).
API_TIMEOUT_SECONDS = 30.0

#: GitHub REST API root.
GITHUB_API_BASE = "https://api.github.com"


class GitHubTools:
    """
    Repository acquisition for the ingestion pipeline.

    Validates a repository reference and makes it available on the local
    filesystem, which is all the MVP requires: analysis then proceeds
    through FilesystemTools.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        repo: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize GitHubTools.

        Args:
            token: GitHub API token. When set, API reads send an
                Authorization header; when absent, public repositories
                remain readable anonymously.
            repo: Default repository in "owner/name" format.
            config: Optional Config instance. A default is loaded when
                not supplied.
        """
        self.config = config or Config.load()
        self.token = token if token is not None else self.config.github_token
        self.repo = repo

    # ------------------------------------------------------------------
    # Implemented
    # ------------------------------------------------------------------

    def validate_repository(self, repo_url: str) -> bool:
        """
        Validate that a repository reference is usable.

        Accepts an HTTPS GitHub URL (`https://github.com/owner/name`,
        with or without a `.git` suffix) or a path to an existing local
        directory.

        Args:
            repo_url: Repository URL or local path to validate.

        Returns:
            True when the reference is valid. Never returns False — an
            invalid reference raises instead, because the reason for
            rejection is more useful to the caller than a bare False.

        Raises:
            InvalidRepositoryURLError: If the reference is empty, uses an
                unsupported scheme or host, is missing owner/name, or
                points at a local path that does not exist.
        """
        if not repo_url or not str(repo_url).strip():
            raise InvalidRepositoryURLError(
                "Repository reference must be a non-empty string."
            )

        reference = str(repo_url).strip()

        if self.is_remote_reference(reference):
            self._validate_remote_url(reference)
        else:
            self._validate_local_path(reference)

        return True

    def clone_repository(self, repo_url: str, destination: str = ".") -> bool:
        """
        Make a repository available on the local filesystem.

        Remote HTTPS URLs are cloned into `destination`. A local
        repository is cloned only when `destination` names a different
        directory; when it resolves to the source itself, the code is
        already in place and nothing is copied.

        Args:
            repo_url: Repository URL or local path.
            destination: Local directory to clone into.

        Returns:
            True when the repository is available at the expected
            location.

        Raises:
            InvalidRepositoryURLError: If the reference fails validation.
            RepositoryCloneError: If the destination is already occupied,
                git is unavailable, or the clone itself fails.
        """
        self.validate_repository(repo_url)
        reference = str(repo_url).strip()

        target = Path(destination).expanduser().resolve()

        if not self.is_remote_reference(reference):
            source = Path(reference).expanduser().resolve()
            if source == target:
                # Already on disk at the requested location.
                return True

        self._ensure_destination_available(target)
        self._run_clone(reference, target)
        return True

    @staticmethod
    def is_remote_reference(repo_url: str) -> bool:
        """
        Report whether a reference is a remote URL rather than a path.

        Args:
            repo_url: Reference to classify.

        Returns:
            True for remote URLs, False for local paths. A single
            character scheme is treated as a Windows drive letter, not a
            URL scheme.
        """
        scheme = urlparse(str(repo_url)).scheme.lower()
        if len(scheme) <= 1:
            return False
        return scheme in REMOTE_SCHEMES

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_remote_url(self, repo_url: str) -> None:
        """
        Validate a remote repository URL.

        Args:
            repo_url: Remote URL to validate.

        Raises:
            InvalidRepositoryURLError: If the scheme is not HTTPS, the
                host is not GitHub, or owner/name is missing.
        """
        parsed = urlparse(repo_url)

        if parsed.scheme.lower() != "https":
            raise InvalidRepositoryURLError(
                f"Only HTTPS repository URLs are supported, got "
                f"{parsed.scheme!r} in {repo_url!r}."
            )

        if parsed.netloc.lower() not in GITHUB_HOSTS:
            raise InvalidRepositoryURLError(
                f"Only GitHub repositories are supported, got host "
                f"{parsed.netloc!r} in {repo_url!r}."
            )

        segments = [segment for segment in parsed.path.split("/") if segment]
        if len(segments) < 2:
            raise InvalidRepositoryURLError(
                f"Repository URL must include owner and name, e.g. "
                f"https://github.com/owner/name -- got {repo_url!r}."
            )

    @staticmethod
    def _validate_local_path(repo_path: str) -> None:
        """
        Validate a local repository path.

        Args:
            repo_path: Local path to validate.

        Raises:
            InvalidRepositoryURLError: If the path does not exist or is
                not a directory.
        """
        source = Path(repo_path).expanduser()

        if not source.exists():
            raise InvalidRepositoryURLError(
                f"Local repository path does not exist: {repo_path!r}"
            )
        if not source.is_dir():
            raise InvalidRepositoryURLError(
                f"Local repository path is not a directory: {repo_path!r}"
            )

    @staticmethod
    def _ensure_destination_available(destination: Path) -> None:
        """
        Confirm a clone destination is empty or absent.

        Args:
            destination: Directory the clone will be written to.

        Raises:
            RepositoryCloneError: If the destination exists and is not
                empty, or exists as a file.
        """
        if not destination.exists():
            return
        if destination.is_file():
            raise RepositoryCloneError(
                f"Clone destination {destination} exists and is a file."
            )
        if any(destination.iterdir()):
            raise RepositoryCloneError(
                f"Clone destination {destination} already exists and is not empty."
            )

    @staticmethod
    def _run_clone(source: str, destination: Path) -> None:
        """
        Clone a repository, preferring GitPython over the git binary.

        GitPython is the declared dependency, but falling back to the
        `git` executable keeps cloning working in environments where the
        package is not installed yet.

        Args:
            source: Repository URL or local path to clone from.
            destination: Directory to clone into.

        Raises:
            RepositoryCloneError: If the clone fails, times out, or no
                git implementation is available.
        """
        try:
            from git import Repo
            from git.exc import GitError
        except ImportError:
            GitHubTools._run_clone_with_git_binary(source, destination)
            return

        try:
            Repo.clone_from(source, str(destination))
        except GitError as exc:
            raise RepositoryCloneError(
                f"Failed to clone {source!r} into {destination}: {exc}"
            ) from exc

    @staticmethod
    def _run_clone_with_git_binary(source: str, destination: Path) -> None:
        """
        Clone by invoking the `git` executable.

        Args:
            source: Repository URL or local path to clone from.
            destination: Directory to clone into.

        Raises:
            RepositoryCloneError: If git is missing, times out, or exits
                non-zero.
        """
        try:
            completed = subprocess.run(
                ["git", "clone", source, str(destination)],
                capture_output=True,
                text=True,
                timeout=CLONE_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise RepositoryCloneError(
                "Cloning requires either GitPython or the `git` executable, "
                "and neither is available."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RepositoryCloneError(
                f"Cloning {source!r} timed out after {CLONE_TIMEOUT_SECONDS}s."
            ) from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RepositoryCloneError(
                f"Failed to clone {source!r} into {destination}: {detail}"
            )

    # ------------------------------------------------------------------
    # GitHub REST API reads
    # ------------------------------------------------------------------

    def get_file_contents(
        self,
        repository: str,
        path: str,
        ref: Optional[str] = None,
    ) -> str:
        """
        Retrieve a file's decoded contents from a GitHub repository.

        Args:
            repository: Repository as ``owner/name`` or an HTTPS GitHub
                URL.
            path: Path to the file within the repository.
            ref: Optional branch, tag, or commit SHA. When omitted, the
                repository default branch is used.

        Returns:
            The file contents as a UTF-8 string.

        Raises:
            InvalidRepositoryURLError: Malformed repository reference or
                repository/file not found (HTTP 404).
            ToolExecutionError: Authentication failure, timeouts,
                connection errors, or other API failures.
            RateLimitError: GitHub rate limiting (HTTP 403/429).
            UnsupportedFileTypeError: Binary (non-UTF-8) file content.
        """
        owner, name = self._parse_repository(repository)
        file_path = str(path or "").strip().lstrip("/")
        if not file_path:
            raise InvalidRepositoryURLError(
                "File path must be a non-empty repository-relative path."
            )

        api_path = (
            f"/repos/{owner}/{name}/contents/{quote(file_path, safe='/')}"
        )
        params: Dict[str, str] = {}
        if ref:
            params["ref"] = str(ref)

        payload = self._api_request("GET", api_path, params=params or None)

        if isinstance(payload, list):
            raise ToolExecutionError(
                f"Path {file_path!r} in {owner}/{name} is a directory, "
                f"not a file."
            )
        if not isinstance(payload, dict):
            raise ToolExecutionError(
                f"Unexpected GitHub contents response for {file_path!r}."
            )
        if payload.get("type") and payload.get("type") != "file":
            raise ToolExecutionError(
                f"Path {file_path!r} in {owner}/{name} is not a file "
                f"(type={payload.get('type')!r})."
            )

        encoding = str(payload.get("encoding") or "").lower()
        raw_content = payload.get("content")
        if raw_content is None:
            raise ToolExecutionError(
                f"GitHub did not return content for {file_path!r} in "
                f"{owner}/{name}. Large files may require the Git Blob API."
            )

        if encoding and encoding != "base64":
            # Rare plain-text responses; treat as already-decoded text.
            text = str(raw_content)
            if "\x00" in text:
                raise UnsupportedFileTypeError(
                    f"File {file_path!r} in {owner}/{name} appears binary."
                )
            return text

        try:
            decoded = base64.b64decode(str(raw_content), validate=False)
        except (ValueError, TypeError) as exc:
            raise ToolExecutionError(
                f"Could not decode GitHub content for {file_path!r}: {exc}"
            ) from exc

        try:
            return decoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedFileTypeError(
                f"File {file_path!r} in {owner}/{name} is binary and "
                f"cannot be returned as text."
            ) from exc

    def list_issues(
        self,
        repository: str,
        state: str = "open",
    ) -> List[Dict[str, Any]]:
        """
        List issues for a repository (pull requests excluded).

        Args:
            repository: Repository as ``owner/name`` or HTTPS URL.
            state: Issue state filter (``open``, ``closed``, ``all``).

        Returns:
            A list of issue dicts with number, title, state, labels,
            author, url, created_at, and updated_at.

        Raises:
            InvalidRepositoryURLError: Malformed repository or 404.
            ToolExecutionError: Auth, timeout, or connection failures.
            RateLimitError: GitHub rate limiting.
        """
        owner, name = self._parse_repository(repository)
        params = {
            "state": self._normalize_state(state),
            "per_page": "100",
        }
        payload = self._api_request(
            "GET",
            f"/repos/{owner}/{name}/issues",
            params=params,
        )
        if not isinstance(payload, list):
            raise ToolExecutionError(
                f"Unexpected GitHub issues response for {owner}/{name}."
            )

        issues: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            # The Issues API also returns pull requests; skip those.
            if "pull_request" in item:
                continue
            user = item.get("user") or {}
            labels_raw = item.get("labels") or []
            labels: List[str] = []
            for label in labels_raw:
                if isinstance(label, dict):
                    label_name = label.get("name")
                    if label_name:
                        labels.append(str(label_name))
                elif label:
                    labels.append(str(label))
            issues.append(
                {
                    "number": item.get("number"),
                    "title": item.get("title") or "",
                    "state": item.get("state") or "",
                    "labels": labels,
                    "author": (user.get("login") if isinstance(user, dict) else "")
                    or "",
                    "url": item.get("html_url") or item.get("url") or "",
                    "created_at": item.get("created_at") or "",
                    "updated_at": item.get("updated_at") or "",
                }
            )
        return issues

    def list_pull_requests(
        self,
        repository: str,
        state: str = "open",
    ) -> List[Dict[str, Any]]:
        """
        List pull requests for a repository.

        Args:
            repository: Repository as ``owner/name`` or HTTPS URL.
            state: PR state filter (``open``, ``closed``, ``all``).

        Returns:
            A list of PR dicts with number, title, state, author,
            source branch, target branch, url, and created_at.

        Raises:
            InvalidRepositoryURLError: Malformed repository or 404.
            ToolExecutionError: Auth, timeout, or connection failures.
            RateLimitError: GitHub rate limiting.
        """
        owner, name = self._parse_repository(repository)
        params = {
            "state": self._normalize_state(state),
            "per_page": "100",
        }
        payload = self._api_request(
            "GET",
            f"/repos/{owner}/{name}/pulls",
            params=params,
        )
        if not isinstance(payload, list):
            raise ToolExecutionError(
                f"Unexpected GitHub pull-requests response for {owner}/{name}."
            )

        pulls: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            user = item.get("user") or {}
            head = item.get("head") or {}
            base = item.get("base") or {}
            pulls.append(
                {
                    "number": item.get("number"),
                    "title": item.get("title") or "",
                    "state": item.get("state") or "",
                    "author": (user.get("login") if isinstance(user, dict) else "")
                    or "",
                    "source_branch": (
                        head.get("ref") if isinstance(head, dict) else ""
                    )
                    or "",
                    "target_branch": (
                        base.get("ref") if isinstance(base, dict) else ""
                    )
                    or "",
                    "url": item.get("html_url") or item.get("url") or "",
                    "created_at": item.get("created_at") or "",
                }
            )
        return pulls

    # ------------------------------------------------------------------
    # Shared API helpers
    # ------------------------------------------------------------------

    def _api_request(
        self,
        method: str,
        api_path: str,
        *,
        params: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Perform one GitHub REST API request with shared error mapping.

        Args:
            method: HTTP method (``GET``, ``POST``, ``PUT``, …).
            api_path: Path beginning with ``/repos/...``.
            params: Optional query parameters.
            json_body: Optional JSON request body for write operations.

        Returns:
            Parsed JSON payload.

        Raises:
            InvalidRepositoryURLError: HTTP 404.
            ToolExecutionError: HTTP 401/409/422, timeouts, connection
                errors, or other non-success responses.
            RateLimitError: HTTP 403/429 rate limiting.
        """
        url = f"{GITHUB_API_BASE}{api_path}"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "codebase-assistant",
        }
        token = (self.token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = requests.request(
                method.upper(),
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=API_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            raise ToolExecutionError(
                f"GitHub API request timed out after {API_TIMEOUT_SECONDS:.0f}s: "
                f"{api_path}"
            ) from exc
        except requests.ConnectionError as exc:
            raise ToolExecutionError(
                f"GitHub API connection failed for {api_path}: {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise ToolExecutionError(
                f"GitHub API request failed for {api_path}: {exc}"
            ) from exc

        return self._map_api_response(response, api_path)

    def _map_api_response(self, response: requests.Response, api_path: str) -> Any:
        """
        Convert an HTTP response into JSON or a project exception.

        Args:
            response: ``requests`` response object.
            api_path: Path used for error messages.

        Returns:
            Parsed JSON body on success.
        """
        status = int(response.status_code)
        body_text = ""
        try:
            body_text = (response.text or "").strip()
        except Exception:
            body_text = ""

        if status == 401:
            raise ToolExecutionError(
                "GitHub authentication failed. Check the configured "
                "GitHub token."
            )
        if status == 403:
            if self._is_rate_limit_response(response, body_text):
                raise RateLimitError(
                    "GitHub API rate limit exceeded. Retry later or "
                    "authenticate with a token."
                )
            raise ToolExecutionError(
                f"GitHub API permission denied for {api_path}: "
                f"{body_text or status}"
            )
        if status == 404:
            raise InvalidRepositoryURLError(
                f"GitHub resource not found: {api_path}"
            )
        if status == 409:
            raise ToolExecutionError(
                f"GitHub API conflict for {api_path}: "
                f"{body_text or 'resource conflict'}"
            )
        if status == 422:
            raise ToolExecutionError(
                f"GitHub API validation error for {api_path}: "
                f"{body_text or 'unprocessable entity'}"
            )
        if status == 429:
            raise RateLimitError(
                "GitHub API rate limit exceeded (HTTP 429). Retry later."
            )
        if status >= 400:
            raise ToolExecutionError(
                f"GitHub API error {status} for {api_path}: "
                f"{body_text or 'no details'}"
            )

        if status == 204 or not body_text:
            return {}

        try:
            return response.json()
        except ValueError as exc:
            raise ToolExecutionError(
                f"GitHub API returned non-JSON for {api_path}."
            ) from exc

    @staticmethod
    def _is_rate_limit_response(
        response: requests.Response, body_text: str
    ) -> bool:
        """Detect GitHub rate-limit responses among HTTP 403s."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            return True
        lowered = body_text.lower()
        return "rate limit" in lowered or "api rate limit" in lowered

    def _parse_repository(self, repository: str) -> Tuple[str, str]:
        """
        Normalize a repository reference to ``(owner, name)``.

        Args:
            repository: ``owner/name`` shorthand or HTTPS GitHub URL.
                When empty, falls back to ``self.repo``.

        Returns:
            Owner and repository name.

        Raises:
            InvalidRepositoryURLError: If the reference cannot be parsed.
        """
        reference = (repository or self.repo or "").strip()
        if not reference:
            raise InvalidRepositoryURLError(
                "Repository must be provided as 'owner/name' or a "
                "https://github.com/owner/name URL."
            )

        if self.is_remote_reference(reference):
            self._validate_remote_url(reference)
            parsed = urlparse(reference)
            segments = [segment for segment in parsed.path.split("/") if segment]
            owner, name = segments[0], segments[1]
        else:
            parts = [part for part in reference.replace("\\", "/").split("/") if part]
            if len(parts) != 2:
                raise InvalidRepositoryURLError(
                    f"Repository must be 'owner/name', got {reference!r}."
                )
            owner, name = parts[0], parts[1]

        if name.lower().endswith(".git"):
            name = name[: -len(".git")]
        if not owner or not name:
            raise InvalidRepositoryURLError(
                f"Repository must include owner and name, got {reference!r}."
            )
        return owner, name

    @staticmethod
    def _normalize_state(state: str) -> str:
        """Normalize an issues/PRs state filter to open|closed|all."""
        normalized = (state or "open").strip().lower()
        if normalized not in {"open", "closed", "all"}:
            raise ToolExecutionError(
                f"Invalid state {state!r}; expected 'open', 'closed', or 'all'."
            )
        return normalized

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create_branch(
        self,
        repository: str,
        branch_name: str,
        source_branch: str = "main",
    ) -> Dict[str, Any]:
        """
        Create a branch from the tip of ``source_branch``.

        Resolves the latest commit SHA of ``source_branch``, then creates
        ``refs/heads/<branch_name>`` pointing at that SHA.

        Args:
            repository: Repository as ``owner/name`` or HTTPS URL.
            branch_name: Name of the new branch.
            source_branch: Existing branch to branch from.

        Returns:
            Dict with ``branch``, ``sha``, and ``url``.

        Raises:
            InvalidRepositoryURLError: Repository or source branch not
                found.
            ToolExecutionError: Auth failure, permission denied, branch
                already exists (422), conflict (409), or transport errors.
            RateLimitError: GitHub rate limiting.
        """
        owner, name = self._parse_repository(repository)
        new_branch = str(branch_name or "").strip()
        source = str(source_branch or "main").strip() or "main"
        if not new_branch:
            raise ToolExecutionError("branch_name must be a non-empty string.")

        source_ref = self._api_request(
            "GET",
            f"/repos/{owner}/{name}/git/ref/heads/{quote(source, safe='')}",
        )
        if not isinstance(source_ref, dict):
            raise ToolExecutionError(
                f"Unexpected GitHub ref response for source branch {source!r}."
            )
        object_info = source_ref.get("object") or {}
        sha = ""
        if isinstance(object_info, dict):
            sha = str(object_info.get("sha") or "")
        if not sha:
            raise ToolExecutionError(
                f"Could not resolve commit SHA for source branch {source!r}."
            )

        created = self._api_request(
            "POST",
            f"/repos/{owner}/{name}/git/refs",
            json_body={
                "ref": f"refs/heads/{new_branch}",
                "sha": sha,
            },
        )
        if not isinstance(created, dict):
            raise ToolExecutionError(
                f"Unexpected GitHub response creating branch {new_branch!r}."
            )

        ref_url = str(created.get("url") or "")
        html_url = f"https://github.com/{owner}/{name}/tree/{new_branch}"
        return {
            "branch": new_branch,
            "sha": sha,
            "url": ref_url or html_url,
        }

    def commit_file(
        self,
        repository: str,
        path: str,
        content: str,
        message: str,
        branch: str,
    ) -> Dict[str, Any]:
        """
        Create or update a file via the GitHub Contents API.

        If the file already exists on ``branch``, its blob SHA is fetched
        and the file is updated. Otherwise a new file is created.
        Content is Base64-encoded automatically.

        Args:
            repository: Repository as ``owner/name`` or HTTPS URL.
            path: Repository-relative file path.
            content: File text content (UTF-8).
            message: Commit message.
            branch: Target branch name.

        Returns:
            Dict with ``commit_sha``, ``commit_url``, ``content_url``,
            and ``created`` (True when the file did not previously exist).

        Raises:
            InvalidRepositoryURLError: Repository not found.
            ToolExecutionError: Auth, permission, validation, or
                transport failures.
            RateLimitError: GitHub rate limiting.
        """
        owner, name = self._parse_repository(repository)
        file_path = str(path or "").strip().lstrip("/")
        branch_name = str(branch or "").strip()
        commit_message = str(message or "").strip()
        if not file_path:
            raise ToolExecutionError("path must be a non-empty repository-relative path.")
        if not branch_name:
            raise ToolExecutionError("branch must be a non-empty branch name.")
        if not commit_message:
            raise ToolExecutionError("message must be a non-empty commit message.")

        api_path = f"/repos/{owner}/{name}/contents/{quote(file_path, safe='/')}"
        existing_sha: Optional[str] = None
        created = True
        try:
            existing = self._api_request(
                "GET",
                api_path,
                params={"ref": branch_name},
            )
            if isinstance(existing, dict) and existing.get("sha"):
                existing_sha = str(existing["sha"])
                created = False
        except InvalidRepositoryURLError:
            # File (or path) not found on this branch → create.
            existing_sha = None
            created = True

        encoded = base64.b64encode(
            (content if content is not None else "").encode("utf-8")
        ).decode("ascii")
        body: Dict[str, Any] = {
            "message": commit_message,
            "content": encoded,
            "branch": branch_name,
        }
        if existing_sha:
            body["sha"] = existing_sha

        payload = self._api_request("PUT", api_path, json_body=body)
        if not isinstance(payload, dict):
            raise ToolExecutionError(
                f"Unexpected GitHub contents response committing {file_path!r}."
            )

        commit_info = payload.get("commit") or {}
        content_info = payload.get("content") or {}
        commit_sha = ""
        commit_url = ""
        content_url = ""
        if isinstance(commit_info, dict):
            commit_sha = str(commit_info.get("sha") or "")
            commit_url = str(
                commit_info.get("html_url") or commit_info.get("url") or ""
            )
        if isinstance(content_info, dict):
            content_url = str(
                content_info.get("html_url") or content_info.get("url") or ""
            )

        return {
            "commit_sha": commit_sha,
            "commit_url": commit_url,
            "content_url": content_url,
            "created": created,
        }

    def create_pull_request(
        self,
        repository: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> Dict[str, Any]:
        """
        Create a pull request on the repository.

        Args:
            repository: Repository as ``owner/name`` or HTTPS URL.
            title: Pull request title.
            body: Pull request description.
            head: Source branch (or ``owner:branch`` for forks).
            base: Target branch (default ``main``).

        Returns:
            Dict with ``number``, ``title``, ``url``, ``state``,
            ``head``, and ``base``.

        Raises:
            InvalidRepositoryURLError: Repository not found.
            ToolExecutionError: Auth, permission, validation (e.g. PR
                already exists), or transport failures.
            RateLimitError: GitHub rate limiting.
        """
        owner, name = self._parse_repository(repository)
        pr_title = str(title or "").strip()
        head_branch = str(head or "").strip()
        base_branch = str(base or "main").strip() or "main"
        if not pr_title:
            raise ToolExecutionError("title must be a non-empty string.")
        if not head_branch:
            raise ToolExecutionError("head must be a non-empty branch name.")

        payload = self._api_request(
            "POST",
            f"/repos/{owner}/{name}/pulls",
            json_body={
                "title": pr_title,
                "body": body if body is not None else "",
                "head": head_branch,
                "base": base_branch,
            },
        )
        if not isinstance(payload, dict):
            raise ToolExecutionError(
                f"Unexpected GitHub response creating pull request on "
                f"{owner}/{name}."
            )

        head_info = payload.get("head") or {}
        base_info = payload.get("base") or {}
        return {
            "number": payload.get("number"),
            "title": payload.get("title") or pr_title,
            "url": payload.get("html_url") or payload.get("url") or "",
            "state": payload.get("state") or "",
            "head": (
                head_info.get("ref") if isinstance(head_info, dict) else head_branch
            )
            or head_branch,
            "base": (
                base_info.get("ref") if isinstance(base_info, dict) else base_branch
            )
            or base_branch,
        }
