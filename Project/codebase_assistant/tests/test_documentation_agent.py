"""
test_documentation_agent.py
=============================

Unit tests for DocumentationAgent.

The LLM client and Retriever are mocked so tests do not require a live
model provider or a populated Chroma index. Filesystem reads use a
temporary repository on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from codebase_assistant.agents.documentation_agent import DocumentationAgent
from codebase_assistant.schemas.schemas import (
    AgentRequest,
    AgentType,
    DocumentationResult,
    ModelResponse,
    RetrievedChunk,
)


VALID_DOC_PAYLOAD = {
    "file_path": "math_utils.py",
    "function_name": "add",
    "summary": "Return the sum of two numbers.",
    "parameters": [
        {"name": "a", "type": "int", "description": "First addend."},
        {"name": "b", "type": "int", "description": "Second addend."},
    ],
    "returns": "The arithmetic sum of a and b.",
    "example_usage": "add(1, 2)",
}


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a tiny repository with one Python module."""
    module = tmp_path / "math_utils.py"
    module.write_text(
        '"""Tiny math helpers."""\n\n'
        "def add(a, b):\n"
        '    """Return a + b."""\n'
        "    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


def _mock_client(available: bool = True, content: str = "") -> MagicMock:
    """Build a mock LLMClient."""
    client = MagicMock()
    client.is_available.return_value = available
    client.generate.return_value = ModelResponse(content=content, usage={}, raw={})
    return client


def _mock_retriever(chunks: Optional[List[RetrievedChunk]] = None) -> MagicMock:
    """Build a mock Retriever."""
    retriever = MagicMock()
    retriever.retrieve.return_value = list(chunks or [])
    retriever.vector_store_path = "./.codebase_assistant/chroma"
    retriever.config = MagicMock()
    retriever.vector_db = MagicMock()
    return retriever


def _agent(
    client: MagicMock,
    retriever: MagicMock,
) -> DocumentationAgent:
    """Construct a DocumentationAgent with mocked collaborators."""
    return DocumentationAgent(
        model_client=client,
        retriever=retriever,
    )


def _readme_request(repo: Path) -> AgentRequest:
    return AgentRequest(
        task_id="doc-1",
        agent_type=AgentType.DOCUMENTATION,
        instruction="Generate a README summary.",
        context={"repo_path": str(repo), "doc_type": "readme"},
    )


def _docstring_request(repo: Path) -> AgentRequest:
    return AgentRequest(
        task_id="doc-2",
        agent_type=AgentType.DOCUMENTATION,
        instruction="Document the add function.",
        context={
            "repo_path": str(repo),
            "file_path": str(repo / "math_utils.py"),
            "function_name": "add",
            "doc_type": "docstring",
        },
    )


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_successful_documentation_generation(
    _mock_index: Any, sample_repo: Path
) -> None:
    """A valid LLM response should yield a successful AgentResponse."""
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    retriever = _mock_retriever()
    agent = _agent(client, retriever)

    response = agent.handle(_docstring_request(sample_repo))

    assert response.success is True
    assert response.errors == []
    assert isinstance(response.output, DocumentationResult)
    assert response.output.summary == VALID_DOC_PAYLOAD["summary"]
    client.generate.assert_called_once()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_retriever_returns_context(_mock_index: Any, sample_repo: Path) -> None:
    """Retrieved chunks should be used when building the LLM prompt."""
    chunk = RetrievedChunk(
        source="math_utils.py",
        content="def add(a, b):\n    return a + b",
        score=0.91,
        metadata={"file_path": "math_utils.py"},
    )
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    retriever = _mock_retriever([chunk])
    agent = _agent(client, retriever)

    response = agent.handle(_docstring_request(sample_repo))

    assert response.success is True
    retriever.retrieve.assert_called()
    prompt = client.generate.call_args.args[0][1].content
    assert "RETRIEVED CONTEXT" in prompt
    assert "def add(a, b)" in prompt
    assert "primary" in prompt.lower()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_retriever_returns_no_context(_mock_index: Any, sample_repo: Path) -> None:
    """Empty retrieval should still succeed using repository files."""
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    retriever = _mock_retriever([])
    agent = _agent(client, retriever)

    response = agent.handle(_readme_request(sample_repo))

    assert response.success is True
    retriever.retrieve.assert_called()
    prompt = client.generate.call_args.args[0][1].content
    assert "RETRIEVED CONTEXT" in prompt
    assert "(none)" in prompt
    assert "math_utils.py" in prompt


def test_model_unavailable(sample_repo: Path) -> None:
    """An unavailable model should fail gracefully without calling generate."""
    client = _mock_client(available=False)
    retriever = _mock_retriever()
    agent = _agent(client, retriever)

    response = agent.handle(_readme_request(sample_repo))

    assert response.success is False
    assert response.errors
    assert "unavailable" in response.errors[0].lower()
    assert isinstance(response.output, DocumentationResult)
    assert response.output.summary == ""
    client.generate.assert_not_called()
    retriever.retrieve.assert_not_called()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_malformed_llm_response(_mock_index: Any, sample_repo: Path) -> None:
    """Non-JSON model output should not raise and should report failure."""
    client = _mock_client(content="sorry, I cannot produce JSON today")
    retriever = _mock_retriever()
    agent = _agent(client, retriever)

    response = agent.handle(_docstring_request(sample_repo))

    assert response.success is False
    assert isinstance(response.output, DocumentationResult)
    assert response.output.summary == ""
    assert response.errors


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_documentation_result_populated_correctly(
    _mock_index: Any, sample_repo: Path
) -> None:
    """All DocumentationResult fields should be populated from the LLM JSON."""
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_docstring_request(sample_repo))
    result = response.output

    assert isinstance(result, DocumentationResult)
    assert result.file_path == "math_utils.py"
    assert result.function_name == "add"
    assert result.summary == "Return the sum of two numbers."
    assert len(result.parameters) == 2
    assert result.parameters[0]["name"] == "a"
    assert result.returns == "The arithmetic sum of a and b."
    assert result.example_usage == "add(1, 2)"


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_graceful_failure_when_generate_raises(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Transport/model errors during generate should not escape uncaught."""
    client = _mock_client(available=True)
    client.generate.side_effect = RuntimeError("connection reset")
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_readme_request(sample_repo))

    assert response.success is False
    assert isinstance(response.output, DocumentationResult)
    assert response.output.summary == ""
    # Failure may be reported via empty summary / error path without crash.
    assert response.errors or response.output.summary == ""


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_run_returns_success_dict(_mock_index: Any, sample_repo: Path) -> None:
    """run() should surface a success status when a README is generated."""
    payload = dict(VALID_DOC_PAYLOAD)
    payload["function_name"] = "README"
    payload["summary"] = "Tiny math helpers repository."
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    outcome = agent.run(str(sample_repo))

    assert outcome["status"] == "success"
    assert "Tiny math helpers" in outcome["message"]


@pytest.fixture
def project_repo(sample_repo: Path) -> Path:
    """Extend the sample repository with project metadata files."""
    (sample_repo / "requirements.txt").write_text(
        "pydantic>=2.0\nchromadb>=0.4\n", encoding="utf-8"
    )
    (sample_repo / "app").mkdir()
    (sample_repo / "app" / "main.py").write_text(
        "def main():\n    print('run')\n", encoding="utf-8"
    )
    return sample_repo


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_readme_prompt_is_symbol_scoped(
    _mock_index: Any, project_repo: Path
) -> None:
    """README mode documents each public symbol with a focused prompt."""
    from codebase_assistant.tracing.tracer import Tracer

    chunk = RetrievedChunk(
        source="math_utils.py",
        content="def add(a, b):\n    return a + b",
        score=0.9,
        metadata={"file_path": "math_utils.py"},
    )
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    agent = _agent(client, _mock_retriever([chunk]))
    agent.tracer = Tracer(run_id="readme-symbols")

    response = agent.handle(_readme_request(project_repo))

    assert response.success is True
    assert client.generate.call_count >= 2  # add + main
    prompts = [call.args[0][1].content for call in client.generate.call_args_list]
    assert any("Document function add()" in prompt for prompt in prompts)
    assert any("Document function main()" in prompt for prompt in prompts)
    assert any("math_utils.py" in prompt for prompt in prompts)
    assert any("app/main.py" in prompt for prompt in prompts)
    # Per-symbol prompts must not dump the whole repository inventory listing.
    assert all(
        "REPOSITORY INVENTORY (every path that exists" not in prompt
        for prompt in prompts
    )
    assert "documentation_merge_finished" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_readme_guidance_requests_structured_sections(
    _mock_index: Any, project_repo: Path
) -> None:
    """The README prompt should ask for full markdown documentation."""
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    agent.handle(_readme_request(project_repo))

    prompt = client.generate.call_args.args[0][1].content
    for heading in (
        "## Project Overview",
        "## Architecture Overview",
        "## Installation Requirements",
        "## How to Run",
        "## Limitations",
    ):
        assert heading in prompt


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_docstring_mode_skips_repository_wide_sections(
    _mock_index: Any, project_repo: Path
) -> None:
    """Single-function documentation should not pay for repo-wide context."""
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    agent.handle(_docstring_request(project_repo))

    prompt = client.generate.call_args.args[0][1].content
    assert "REPOSITORY INVENTORY" not in prompt
    assert "PROJECT FILES" not in prompt


def test_system_prompt_enforces_grounding_and_markdown() -> None:
    """The system prompt must keep the model grounded and markdown-formatted."""
    from codebase_assistant.agents.documentation_agent import _SYSTEM_PROMPT

    lowered = _SYSTEM_PROMPT.lower()
    assert "never invent" in lowered
    assert "markdown" in lowered
    assert "single paragraph" in lowered


def test_dedupe_chunks_removes_duplicate_content() -> None:
    """Duplicate retrieved chunks should collapse to one entry."""
    chunks = [
        RetrievedChunk(source="a.py", content="def add(a, b): return a + b", score=0.9),
        RetrievedChunk(
            source="a.py",
            content="def add(a, b): return a + b",
            score=0.8,
        ),
        RetrievedChunk(source="b.py", content="def sub(a, b): return a - b", score=0.7),
    ]
    unique = DocumentationAgent._dedupe_chunks(chunks)
    assert len(unique) == 2
    assert unique[0].content.startswith("def add")
    assert unique[1].content.startswith("def sub")


# ---------------------------------------------------------------------------
# JSON repair retry (exactly one attempt)
# ---------------------------------------------------------------------------

_MALFORMED_DOC_OUTPUT = (
    "Here is the documentation:\n"
    "```json\n"
    "{\n"
    '  "file_path": "math_utils.py",\n'
    '  "function_name": "add",\n'
    '  "summary": "Return the sum of two numbers.",\n'
    '  "parameters": [{"name": "a", "type": "int", "description": "First addend."},\n'
    '  "returns": "sum",\n'
    '  "example_usage": "add(1, 2)"\n'
    # Trailing comma / truncated fence makes this unparseable as-is.
)


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_malformed_json_repaired_successfully(
    _mock_index: Any, sample_repo: Path
) -> None:
    """One JSON-repair retry can recover a usable DocumentationResult."""
    from codebase_assistant.tracing.tracer import Tracer

    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=_MALFORMED_DOC_OUTPUT, usage={}, raw={}),
        ModelResponse(content=json.dumps(VALID_DOC_PAYLOAD), usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="doc-retry-ok")

    response = agent.handle(_docstring_request(sample_repo))

    assert response.success is True
    assert client.generate.call_count == 2
    assert response.output.summary == VALID_DOC_PAYLOAD["summary"]
    assert response.output.abstention is None
    retry_messages = client.generate.call_args_list[1].args[0]
    assert "ONLY valid DocumentationResult JSON" in retry_messages[0].content
    assert "PARSER / VALIDATION ERROR" in retry_messages[1].content
    assert "RAW MODEL OUTPUT" in retry_messages[1].content
    assert "ORIGINAL PROMPT" in retry_messages[1].content
    names = agent.tracer.event_names()
    assert "documentation_retry_started" in names
    assert "documentation_retry_success" in names
    assert "documentation_retry_failed" not in names


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_malformed_json_still_invalid_after_retry(
    _mock_index: Any, sample_repo: Path
) -> None:
    """If the repair output is still invalid, fall through to abstention."""
    from codebase_assistant.tracing.tracer import Tracer

    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content="not-json-at-all", usage={}, raw={}),
        ModelResponse(content="still-not-json {{{", usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="doc-retry-bad")

    response = agent.handle(_docstring_request(sample_repo))

    assert response.success is False
    assert client.generate.call_count == 2
    assert response.output.summary == ""
    assert response.output.abstention is not None
    assert response.output.abstention.reason == "LLM response could not be verified."
    names = agent.tracer.event_names()
    assert "documentation_retry_started" in names
    assert "documentation_retry_failed" in names
    assert "documentation_retry_success" not in names


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_provider_failure_during_retry(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Provider errors on the repair call must not crash; abstain instead."""
    from codebase_assistant.tracing.tracer import Tracer

    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content="not-json", usage={}, raw={}),
        RuntimeError("OpenRouter unavailable"),
    ]
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="doc-retry-provider")

    response = agent.handle(_docstring_request(sample_repo))

    assert response.success is False
    assert client.generate.call_count == 2
    assert response.output.summary == ""
    assert response.output.abstention is not None
    assert "documentation_retry_failed" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_json_retry_occurs_exactly_once(
    _mock_index: Any, sample_repo: Path
) -> None:
    """A failed repair must not trigger a second JSON retry."""
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content="broken-1", usage={}, raw={}),
        ModelResponse(content="broken-2", usage={}, raw={}),
        ModelResponse(content=json.dumps(VALID_DOC_PAYLOAD), usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())

    with patch.object(
        agent,
        "_retry_json_repair",
        wraps=agent._retry_json_repair,
    ) as retry_spy:
        response = agent.handle(_docstring_request(sample_repo))

    assert response.success is False
    assert client.generate.call_count == 2
    assert retry_spy.call_count == 1


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_valid_json_never_triggers_retry(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Well-formed DocumentationResult JSON must skip the repair path."""
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    with patch.object(
        agent,
        "_retry_json_repair",
        wraps=agent._retry_json_repair,
    ) as retry_spy:
        response = agent.handle(_docstring_request(sample_repo))

    assert response.success is True
    client.generate.assert_called_once()
    assert retry_spy.call_count == 0


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_abstention_still_occurs_after_unsuccessful_retry(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Existing abstention reason remains after an unsuccessful JSON retry."""
    client = _mock_client(content="sorry, I cannot produce JSON today")
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_docstring_request(sample_repo))

    assert response.success is False
    assert response.errors
    assert "could not be verified" in response.errors[0].lower()
    assert response.output.abstention is not None
    assert response.output.abstention.reason == "LLM response could not be verified."
    assert client.generate.call_count == 2


# ---------------------------------------------------------------------------
# Documentation grounding / hallucination filtering
# ---------------------------------------------------------------------------


@pytest.fixture
def grounded_repo(tmp_path: Path) -> Path:
    """Repository with a real module layout for grounding checks."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "app" / "auth.py").write_text(
        '"""Auth helpers."""\n\n'
        "class AuthHelper:\n"
        "    def login(self, user):\n"
        "        return user\n\n"
        "def authenticate(user):\n"
        "    return bool(user)\n",
        encoding="utf-8",
    )
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


def _grounding_payload(**overrides: Any) -> Dict[str, Any]:
    payload = {
        "file_path": "math_utils.py",
        "function_name": "add",
        "summary": "Return the sum of two numbers.",
        "parameters": [
            {"name": "a", "type": "int", "description": "First addend."},
            {"name": "b", "type": "int", "description": "Second addend."},
        ],
        "returns": "The arithmetic sum of a and b.",
        "example_usage": "add(1, 2)",
    }
    payload.update(overrides)
    return payload


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_grounding_all_references_valid(
    _mock_index: Any, grounded_repo: Path
) -> None:
    """Fully grounded documentation should succeed without removals."""
    from codebase_assistant.tracing.tracer import Tracer

    payload = _grounding_payload(
        file_path="app/auth.py",
        function_name="authenticate",
        summary=(
            "## Purpose\n"
            "Authenticate users via `authenticate` in `app/auth.py`.\n"
            "Also uses class AuthHelper."
        ),
        parameters=[
            {
                "name": "app.auth",
                "type": "module",
                "description": "Auth module.",
            }
        ],
        returns="bool",
        example_usage="authenticate('u')",
    )
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="ground-ok")

    response = agent.handle(
        AgentRequest(
            task_id="g1",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document authenticate.",
            context={
                "repo_path": str(grounded_repo),
                "file_path": str(grounded_repo / "app" / "auth.py"),
                "function_name": "authenticate",
                "doc_type": "docstring",
            },
        )
    )

    assert response.success is True
    assert response.output.file_path == "app/auth.py"
    assert response.output.function_name == "authenticate"
    assert "authenticate" in response.output.summary
    assert "app/auth.py" in response.output.summary
    names = agent.tracer.event_names()
    assert "documentation_grounding_started" in names
    assert "documentation_grounding_finished" in names
    assert "documentation_grounding_removed_claim" not in names


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_grounding_nonexistent_file_removed(
    _mock_index: Any, grounded_repo: Path
) -> None:
    """Hallucinated file paths must be removed or rewritten."""
    payload = _grounding_payload(
        file_path="app/services/auth.py",
        function_name="authenticate",
        summary="See app/services/auth.py for details.",
    )
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="g2",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document authenticate.",
            context={
                "repo_path": str(grounded_repo),
                "file_path": str(grounded_repo / "app" / "auth.py"),
                "function_name": "authenticate",
                "doc_type": "docstring",
            },
        )
    )

    assert response.success is True
    # Structured path is rewritten to a real inventory path (workspace-relative).
    assert response.output.file_path in {"app/auth.py", "auth.py"}
    assert "app/services/auth.py" not in response.output.summary
    assert "app/services/auth.py" not in response.output.file_path
    assert response.output.function_name == "authenticate"


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_grounding_nonexistent_module_removed(
    _mock_index: Any, grounded_repo: Path
) -> None:
    """Unknown module claims should be stripped from README-style output."""
    payload = _grounding_payload(
        file_path="app/auth.py",
        function_name="README",
        summary="The module app.services.billing handles invoices.",
        parameters=[
            {
                "name": "app.services.billing",
                "type": "module",
                "description": "Billing package.",
            },
            {
                "name": "app.auth",
                "type": "module",
                "description": "Auth package.",
            },
        ],
    )
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_readme_request(grounded_repo))

    assert response.success is True
    names = [item["name"] for item in response.output.parameters]
    assert "app.services.billing" not in names
    assert "app.auth" in names
    assert "app.services.billing" not in response.output.summary


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_grounding_nonexistent_class_removed(
    _mock_index: Any, grounded_repo: Path
) -> None:
    """Hallucinated class names must not survive grounding."""
    payload = _grounding_payload(
        file_path="app/auth.py",
        function_name="authenticate",
        summary="Uses AuthenticationService for login flows.",
        parameters=[
            {
                "name": "AuthenticationService",
                "type": "class",
                "description": "Missing class.",
            },
            {
                "name": "AuthHelper",
                "type": "class",
                "description": "Real helper.",
            },
        ],
    )
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="g4",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document auth.",
            context={
                "repo_path": str(grounded_repo),
                "file_path": str(grounded_repo / "app" / "auth.py"),
                "doc_type": "module",
            },
        )
    )

    assert response.success is True
    assert "AuthenticationService" not in response.output.summary
    names = [item["name"] for item in response.output.parameters]
    assert "AuthenticationService" not in names
    assert "AuthHelper" in names


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_grounding_nonexistent_function_removed(
    _mock_index: Any, grounded_repo: Path
) -> None:
    """Hallucinated function symbols must be removed."""
    payload = _grounding_payload(
        file_path="app/auth.py",
        function_name="ghost_login",
        summary="The function ghost_login validates tokens.",
        parameters=[
            {
                "name": "ghost_login",
                "type": "function",
                "description": "Missing.",
            },
            {
                "name": "authenticate",
                "type": "function",
                "description": "Real.",
            },
        ],
    )
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="g5",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document auth API.",
            context={
                "repo_path": str(grounded_repo),
                "file_path": str(grounded_repo / "app" / "auth.py"),
                "doc_type": "api",
            },
        )
    )

    assert response.success is True
    assert "ghost_login" not in (response.output.function_name or "")
    assert "ghost_login" not in response.output.summary
    names = [item["name"] for item in response.output.parameters]
    assert "ghost_login" not in names
    assert "authenticate" in names


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_grounding_mixed_valid_and_invalid_references(
    _mock_index: Any, grounded_repo: Path
) -> None:
    """Valid references remain while invalid ones are scrubbed."""
    payload = _grounding_payload(
        file_path="app/auth.py",
        function_name="authenticate",
        summary=(
            "Call `authenticate` from `app/auth.py`. "
            "Do not use `app/services/auth.py` or AuthenticationService."
        ),
        parameters=[
            {
                "name": "authenticate",
                "type": "function",
                "description": "Real function.",
            },
            {
                "name": "missing_fn",
                "type": "function",
                "description": "Fake.",
            },
        ],
    )
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="g6",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document authenticate.",
            context={
                "repo_path": str(grounded_repo),
                "file_path": str(grounded_repo / "app" / "auth.py"),
                "function_name": "authenticate",
                "doc_type": "module",
            },
        )
    )

    assert response.success is True
    assert "`authenticate`" in response.output.summary
    assert "app/auth.py" in response.output.summary
    assert "app/services/auth.py" not in response.output.summary
    assert "AuthenticationService" not in response.output.summary
    names = [item["name"] for item in response.output.parameters]
    assert names == ["authenticate"]


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_grounding_unchanged_when_everything_valid(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Grounding must be a no-op when every reference is already valid."""
    original = dict(VALID_DOC_PAYLOAD)
    client = _mock_client(content=json.dumps(original))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_docstring_request(sample_repo))

    assert response.success is True
    assert response.output.file_path == original["file_path"]
    assert response.output.function_name == original["function_name"]
    assert response.output.summary == original["summary"]
    assert response.output.returns == original["returns"]
    assert response.output.example_usage == original["example_usage"]
    assert response.output.parameters == original["parameters"]


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_grounding_abstains_when_nothing_can_be_verified(
    _mock_index: Any, grounded_repo: Path
) -> None:
    """If every claim is unsupported, abstain instead of returning fiction."""
    from codebase_assistant.tracing.tracer import Tracer

    payload = _grounding_payload(
        file_path="missing/service.py",
        function_name="totally_fake",
        summary="The function totally_fake lives in missing/service.py.",
        parameters=[
            {
                "name": "TotallyFakeService",
                "type": "class",
                "description": "Does not exist.",
            }
        ],
        returns="Uses TotallyFakeService.",
        example_usage="totally_fake()",
    )
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="ground-abstain")

    response = agent.handle(
        AgentRequest(
            task_id="g8",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document missing API.",
            context={
                "repo_path": str(grounded_repo),
                "file_path": str(grounded_repo / "app" / "auth.py"),
                "doc_type": "api",
            },
        )
    )

    assert response.success is False
    assert response.output.summary == ""
    assert response.output.abstention is not None
    assert "could not be grounded" in response.output.abstention.reason.lower()
    assert "documentation_grounding_abstained" in agent.tracer.event_names()


# ---------------------------------------------------------------------------
# Optional write-back (README + docstrings)
# ---------------------------------------------------------------------------


@pytest.fixture
def writeback_repo(tmp_path: Path) -> Path:
    """Repository with an undocumented public function for write-back tests."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_writeback_readme_written_successfully(
    _mock_index: Any, writeback_repo: Path
) -> None:
    """write_to_disk should create README.md from the generated summary."""
    from codebase_assistant.tracing.tracer import Tracer

    payload = dict(VALID_DOC_PAYLOAD)
    payload["function_name"] = "README"
    payload["summary"] = "## Project Overview\n\nTiny math helpers repository."
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="write-readme")

    response = agent.handle(
        AgentRequest(
            task_id="w1",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Generate a README summary.",
            context={
                "repo_path": str(writeback_repo),
                "doc_type": "readme",
                "write_to_disk": True,
            },
        )
    )

    readme = writeback_repo / "README.md"
    assert response.success is True
    assert readme.is_file()
    assert "Tiny math helpers repository." in readme.read_text(encoding="utf-8")
    assert "Write-back wrote README.md." in response.output.summary
    names = agent.tracer.event_names()
    assert "documentation_write_started" in names
    assert "documentation_write_finished" in names


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_writeback_preserves_existing_readme_unless_replace(
    _mock_index: Any, writeback_repo: Path
) -> None:
    """Existing README.md is preserved unless replace_existing=True."""
    existing = "# Existing README\nKeep me.\n"
    (writeback_repo / "README.md").write_text(existing, encoding="utf-8")
    payload = dict(VALID_DOC_PAYLOAD)
    payload["function_name"] = "README"
    payload["summary"] = "## Project Overview\n\nReplacement body."
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    preserved = agent.handle(
        AgentRequest(
            task_id="w2",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Generate a README summary.",
            context={
                "repo_path": str(writeback_repo),
                "doc_type": "readme",
                "write_to_disk": True,
                "replace_existing": False,
            },
        )
    )
    assert preserved.success is True
    assert (writeback_repo / "README.md").read_text(encoding="utf-8") == existing
    assert "already exists" in preserved.output.summary.lower()

    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())
    replaced = agent.handle(
        AgentRequest(
            task_id="w3",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Generate a README summary.",
            context={
                "repo_path": str(writeback_repo),
                "doc_type": "readme",
                "write_to_disk": True,
                "replace_existing": True,
            },
        )
    )
    text = (writeback_repo / "README.md").read_text(encoding="utf-8")
    assert replaced.success is True
    assert "Replacement body." in text
    assert "Keep me." not in text


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_writeback_inserts_function_docstring(
    _mock_index: Any, writeback_repo: Path
) -> None:
    """Missing function docstrings should be inserted with indentation preserved."""
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="w4",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document the add function.",
            context={
                "repo_path": str(writeback_repo),
                "file_path": str(writeback_repo / "math_utils.py"),
                "function_name": "add",
                "doc_type": "docstring",
                "write_to_disk": True,
            },
        )
    )

    source = (writeback_repo / "math_utils.py").read_text(encoding="utf-8")
    assert response.success is True
    assert 'def add(a, b):' in source
    assert '    """Return the sum of two numbers.' in source or (
        '    """\n    Return the sum of two numbers.' in source
    )
    assert "return a + b" in source
    assert "Write-back wrote docstring" in response.output.summary


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_writeback_leaves_existing_docstrings_untouched(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Existing function docstrings must not be overwritten by default."""
    before = (sample_repo / "math_utils.py").read_text(encoding="utf-8")
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="w5",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document the add function.",
            context={
                "repo_path": str(sample_repo),
                "file_path": str(sample_repo / "math_utils.py"),
                "function_name": "add",
                "doc_type": "docstring",
                "write_to_disk": True,
                "replace_existing": False,
            },
        )
    )

    after = (sample_repo / "math_utils.py").read_text(encoding="utf-8")
    assert response.success is True
    assert after == before
    assert '"""Return a + b."""' in after
    assert "skipped existing" in response.output.summary.lower()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_writeback_failures_are_graceful(
    _mock_index: Any, writeback_repo: Path
) -> None:
    """Filesystem write failures must not fail the agent response."""
    from codebase_assistant.tracing.tracer import Tracer

    payload = dict(VALID_DOC_PAYLOAD)
    payload["function_name"] = "README"
    payload["summary"] = "## Overview\n\nShould still return."
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="write-fail")

    with patch(
        "codebase_assistant.tools.filesystem_tools.FilesystemTools.write_file",
        side_effect=RuntimeError("disk full"),
    ):
        response = agent.handle(
            AgentRequest(
                task_id="w6",
                agent_type=AgentType.DOCUMENTATION,
                instruction="Generate a README summary.",
                context={
                    "repo_path": str(writeback_repo),
                    "doc_type": "readme",
                    "write_to_disk": True,
                },
            )
        )

    assert response.success is True
    assert "Should still return." in response.output.summary
    assert "Write-back warning:" in response.output.summary
    assert "documentation_write_failed" in agent.tracer.event_names()
    assert not (writeback_repo / "README.md").exists()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_writeback_disabled_by_default_is_noop(
    _mock_index: Any, writeback_repo: Path
) -> None:
    """Default write_to_disk=False must not create files."""
    payload = dict(VALID_DOC_PAYLOAD)
    payload["function_name"] = "README"
    payload["summary"] = "## Overview\n\nNo disk writes."
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_readme_request(writeback_repo))

    assert response.success is True
    assert not (writeback_repo / "README.md").exists()
    source = (writeback_repo / "math_utils.py").read_text(encoding="utf-8")
    assert '"""' not in source
    assert "Write-back" not in response.output.summary


def test_writeback_does_not_change_documentation_result_schema() -> None:
    """DocumentationResult must keep the public field set unchanged."""
    fields = set(DocumentationResult.model_fields)
    assert fields == {
        "file_path",
        "function_name",
        "summary",
        "parameters",
        "returns",
        "example_usage",
        "abstention",
    }


# ---------------------------------------------------------------------------
# Targeted documentation (file / function / class)
# ---------------------------------------------------------------------------


@pytest.fixture
def targeted_repo(tmp_path: Path) -> Path:
    """Repository with multiple modules for scoped documentation tests."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "app" / "auth.py").write_text(
        '"""Auth helpers."""\n\n'
        "class UserService:\n"
        "    def create(self, name):\n"
        "        return name\n\n"
        "def authenticate(user):\n"
        "    return bool(user)\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "billing.py").write_text(
        "def charge(amount):\n"
        "    return amount\n",
        encoding="utf-8",
    )
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_targeted_repository_documentation_uses_per_symbol_prompts(
    _mock_index: Any, targeted_repo: Path
) -> None:
    """Absent targeting fields documents each public symbol then merges."""
    from codebase_assistant.tracing.tracer import Tracer

    payload = dict(VALID_DOC_PAYLOAD)
    payload["summary"] = "## Project Overview\n\nRepository docs."
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="target-repo")

    response = agent.handle(_readme_request(targeted_repo))

    assert response.success is True
    assert client.generate.call_count >= 2
    prompts = [call.args[0][1].content for call in client.generate.call_args_list]
    assert any("Document function" in prompt for prompt in prompts)
    assert all(
        "REPOSITORY INVENTORY (every path that exists" not in prompt
        for prompt in prompts
    )
    names = agent.tracer.event_names()
    assert "documentation_target_selected" in names
    assert "documentation_symbol_started" in names
    assert "documentation_merge_finished" in names
    assert "documentation_target_grounded" in names


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_targeted_file_documentation(
    _mock_index: Any, targeted_repo: Path
) -> None:
    """file_path alone documents each public symbol in that file."""
    from codebase_assistant.tracing.tracer import Tracer

    payload = {
        "file_path": "auth.py",
        "function_name": "authenticate",
        "summary": "Auth helper symbol documentation.",
        "parameters": [],
        "returns": "",
        "example_usage": "",
    }
    chunks = [
        RetrievedChunk(
            source="auth.py",
            content="def authenticate(user): return bool(user)",
            score=0.9,
        ),
        RetrievedChunk(
            source="billing.py",
            content="def charge(amount): return amount",
            score=0.8,
        ),
    ]
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever(chunks))
    agent.tracer = Tracer(run_id="target-file")

    response = agent.handle(
        AgentRequest(
            task_id="t-file",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document app/auth.py",
            context={
                "repo_path": str(targeted_repo),
                "file_path": str(targeted_repo / "app" / "auth.py"),
                "doc_type": "module",
            },
        )
    )

    assert response.success is True
    assert client.generate.call_count >= 2
    prompts = [call.args[0][1].content for call in client.generate.call_args_list]
    assert any("Document function authenticate()" in prompt for prompt in prompts)
    assert any("Document class UserService." in prompt for prompt in prompts)
    assert all("def charge" not in prompt for prompt in prompts)
    assert all("REPOSITORY INVENTORY" not in prompt for prompt in prompts)
    assert "documentation_target_selected" in agent.tracer.event_names()
    assert "documentation_merge_finished" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_targeted_function_documentation(
    _mock_index: Any, targeted_repo: Path
) -> None:
    """function_name + file_path documents only that function."""
    from codebase_assistant.tracing.tracer import Tracer

    payload = {
        "file_path": "auth.py",
        "function_name": "authenticate",
        "summary": "Validate whether a user credential is present.",
        "parameters": [
            {"name": "user", "type": "Any", "description": "User identifier."}
        ],
        "returns": "True when user is truthy.",
        "example_usage": "authenticate('alice')",
    }
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="target-fn")

    response = agent.handle(
        AgentRequest(
            task_id="t-fn",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document function authenticate in app/auth.py",
            context={
                "repo_path": str(targeted_repo),
                "file_path": str(targeted_repo / "app" / "auth.py"),
                "function_name": "authenticate",
                "doc_type": "docstring",
            },
        )
    )

    assert response.success is True
    assert client.generate.call_count == 1
    prompt = client.generate.call_args.args[0][1].content
    assert "Document function authenticate()" in prompt
    assert "def authenticate" in prompt
    assert "class UserService" not in prompt
    assert "def charge" not in prompt
    assert "documentation_target_selected" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_targeted_class_documentation(
    _mock_index: Any, targeted_repo: Path
) -> None:
    """class_name can be resolved via AST inventory without file_path."""
    from codebase_assistant.tracing.tracer import Tracer

    payload = {
        "file_path": "auth.py",
        "function_name": "UserService",
        "summary": "Service that creates user records.",
        "parameters": [],
        "returns": "",
        "example_usage": "UserService().create('alice')",
    }
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="target-class")

    response = agent.handle(
        AgentRequest(
            task_id="t-class",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document class UserService",
            context={
                "repo_path": str(targeted_repo),
                "class_name": "UserService",
                "doc_type": "module",
            },
        )
    )

    assert response.success is True
    prompt = client.generate.call_args.args[0][1].content
    assert "Document class UserService." in prompt
    assert "CLASS NAME" in prompt
    assert "class UserService" in prompt
    assert "def charge" not in prompt
    assert "def add" not in prompt
    assert "documentation_target_selected" in agent.tracer.event_names()
    assert "documentation_target_grounded" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_targeted_nonexistent_file_abstains(
    _mock_index: Any, targeted_repo: Path
) -> None:
    """Missing files must abstain instead of inventing documentation."""
    from codebase_assistant.tracing.tracer import Tracer

    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="target-missing-file")

    response = agent.handle(
        AgentRequest(
            task_id="t-miss-file",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document missing module",
            context={
                "repo_path": str(targeted_repo),
                "file_path": str(targeted_repo / "app" / "missing.py"),
                "doc_type": "module",
            },
        )
    )

    assert response.success is False
    assert response.output.abstention is not None
    assert "target not found" in response.output.abstention.reason.lower()
    assert "missing.py" in response.output.abstention.reason
    assert client.generate.call_count == 0
    assert "documentation_target_not_found" in agent.tracer.event_names()
    assert "documentation_target_selected" not in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_targeted_nonexistent_function_abstains(
    _mock_index: Any, targeted_repo: Path
) -> None:
    """Missing functions must abstain with a searched-location hint."""
    from codebase_assistant.tracing.tracer import Tracer

    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="target-missing-fn")

    response = agent.handle(
        AgentRequest(
            task_id="t-miss-fn",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document missing function",
            context={
                "repo_path": str(targeted_repo),
                "file_path": str(targeted_repo / "app" / "auth.py"),
                "function_name": "does_not_exist",
                "doc_type": "docstring",
            },
        )
    )

    assert response.success is False
    assert response.output.abstention is not None
    reason = response.output.abstention.reason.lower()
    assert "target not found" in reason
    assert "does_not_exist" in reason
    assert "auth.py" in reason
    assert client.generate.call_count == 0
    assert "documentation_target_not_found" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_targeted_grounding_still_works(
    _mock_index: Any, targeted_repo: Path
) -> None:
    """Scoped documentation still strips hallucinated references."""
    from codebase_assistant.tracing.tracer import Tracer

    payload = {
        "file_path": "auth.py",
        "function_name": "authenticate",
        "summary": (
            "authenticate validates users. "
            "Also see FakeModule and the function ghost_helper."
        ),
        "parameters": [
            {"name": "user", "type": "Any", "description": "User identifier."},
            {
                "name": "ghost_helper",
                "type": "function",
                "description": "Missing helper.",
            },
        ],
        "returns": "True when user is truthy.",
        "example_usage": "authenticate('alice')",
    }
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="target-ground")

    response = agent.handle(
        AgentRequest(
            task_id="t-ground",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document function authenticate",
            context={
                "repo_path": str(targeted_repo),
                "file_path": str(targeted_repo / "app" / "auth.py"),
                "function_name": "authenticate",
                "doc_type": "docstring",
            },
        )
    )

    assert response.success is True
    assert "authenticate" in response.output.summary.lower()
    assert "ghost_helper" not in response.output.summary
    assert "FakeModule" not in response.output.summary
    param_names = [item["name"] for item in response.output.parameters]
    assert "ghost_helper" not in param_names
    assert "documentation_target_grounded" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_targeted_writeback_still_works(
    _mock_index: Any, targeted_repo: Path
) -> None:
    """Targeted function docs still support optional write-back."""
    payload = {
        "file_path": "auth.py",
        "function_name": "authenticate",
        "summary": "Return whether the user value is truthy.",
        "parameters": [
            {"name": "user", "type": "Any", "description": "User identifier."}
        ],
        "returns": "True when user is truthy.",
        "example_usage": "authenticate('alice')",
    }
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="t-write",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document authenticate",
            context={
                "repo_path": str(targeted_repo),
                "file_path": str(targeted_repo / "app" / "auth.py"),
                "function_name": "authenticate",
                "doc_type": "docstring",
                "write_to_disk": True,
            },
        )
    )

    source = (targeted_repo / "app" / "auth.py").read_text(encoding="utf-8")
    assert response.success is True
    assert "def authenticate" in source
    assert "Return whether the user value is truthy." in source


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_targeted_json_retry_still_works(
    _mock_index: Any, targeted_repo: Path
) -> None:
    """Malformed JSON still gets exactly one repair retry in targeted mode."""
    from codebase_assistant.tracing.tracer import Tracer

    good = {
        "file_path": "auth.py",
        "function_name": "authenticate",
        "summary": "Validate a user credential.",
        "parameters": [],
        "returns": "bool",
        "example_usage": "authenticate('alice')",
    }
    client = _mock_client(content="not-json")
    client.generate.side_effect = [
        ModelResponse(content="not-json {{{", usage={}, raw={}),
        ModelResponse(content=json.dumps(good), usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="target-retry")

    response = agent.handle(
        AgentRequest(
            task_id="t-retry",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document authenticate",
            context={
                "repo_path": str(targeted_repo),
                "file_path": str(targeted_repo / "app" / "auth.py"),
                "function_name": "authenticate",
                "doc_type": "docstring",
            },
        )
    )

    assert response.success is True
    assert client.generate.call_count == 2
    assert "documentation_retry_success" in agent.tracer.event_names()
    assert "documentation_target_selected" in agent.tracer.event_names()


# ---------------------------------------------------------------------------
# Per-symbol documentation generation + merge
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_symbol_repo(tmp_path: Path) -> Path:
    """Repository with multiple public functions and classes."""
    (tmp_path / "math.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n",
        encoding="utf-8",
    )
    (tmp_path / "auth.py").write_text(
        "class Gate:\n"
        "    def login(self, user):\n"
        "        return user\n\n"
        "    def logout(self):\n"
        "        return True\n\n"
        "def ping():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    return tmp_path


def _symbol_payload(name: str, summary: str) -> Dict[str, Any]:
    return {
        "file_path": "math.py",
        "function_name": name,
        "summary": summary,
        "parameters": [
            {
                "name": name,
                "type": "function",
                "description": f"Public function {name}.",
            }
        ],
        "returns": f"Result of {name}.",
        "example_usage": f"{name}(1, 2)",
    }


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_per_symbol_multiple_public_functions(
    _mock_index: Any, multi_symbol_repo: Path
) -> None:
    """Each public function should get its own LLM call."""
    from codebase_assistant.tracing.tracer import Tracer

    client = _mock_client(content=json.dumps(_symbol_payload("add", "Adds numbers.")))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="sym-fns")

    response = agent.handle(
        AgentRequest(
            task_id="s1",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document math.py",
            context={
                "repo_path": str(multi_symbol_repo),
                "file_path": str(multi_symbol_repo / "math.py"),
                "doc_type": "module",
            },
        )
    )

    assert response.success is True
    assert client.generate.call_count == 2
    prompts = [call.args[0][1].content for call in client.generate.call_args_list]
    assert any("Document function add()" in prompt for prompt in prompts)
    assert any("Document function subtract()" in prompt for prompt in prompts)
    assert "documentation_symbol_started" in agent.tracer.event_names()
    assert "documentation_symbol_finished" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_per_symbol_multiple_classes_and_methods(
    _mock_index: Any, multi_symbol_repo: Path
) -> None:
    """Classes and public methods are documented as independent symbols."""
    client = _mock_client(
        content=json.dumps(
            {
                "file_path": "auth.py",
                "function_name": "Gate",
                "summary": "Auth gate symbol.",
                "parameters": [],
                "returns": "",
                "example_usage": "Gate().login('a')",
            }
        )
    )
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="s2",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document auth.py",
            context={
                "repo_path": str(multi_symbol_repo),
                "file_path": str(multi_symbol_repo / "auth.py"),
                "doc_type": "module",
            },
        )
    )

    assert response.success is True
    prompts = [call.args[0][1].content for call in client.generate.call_args_list]
    assert any("Document class Gate." in prompt for prompt in prompts)
    assert any("Document method Gate.login()" in prompt for prompt in prompts)
    assert any("Document method Gate.logout()" in prompt for prompt in prompts)
    assert any("Document function ping()" in prompt for prompt in prompts)
    assert client.generate.call_count == 4


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_per_symbol_merge_correctness(
    _mock_index: Any, multi_symbol_repo: Path
) -> None:
    """Merged DocumentationResult keeps ordered symbol sections and fields."""
    from codebase_assistant.tracing.tracer import Tracer

    payloads = [
        _symbol_payload("add", "Adds two numbers."),
        _symbol_payload("subtract", "Subtracts two numbers."),
    ]
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(item), usage={}, raw={}) for item in payloads
    ]
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="sym-merge")

    response = agent.handle(
        AgentRequest(
            task_id="s3",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document math.py",
            context={
                "repo_path": str(multi_symbol_repo),
                "file_path": str(multi_symbol_repo / "math.py"),
                "doc_type": "module",
            },
        )
    )

    assert response.success is True
    summary = response.output.summary
    assert summary.index("## add") < summary.index("## subtract")
    assert "Adds two numbers." in summary
    assert "Subtracts two numbers." in summary
    names = [item["name"] for item in response.output.parameters]
    assert names == ["add", "subtract"]
    assert "add(1, 2)" in response.output.example_usage
    assert "subtract(1, 2)" in response.output.example_usage
    assert "Result of add." in response.output.returns
    assert "Result of subtract." in response.output.returns
    assert "documentation_merge_started" in agent.tracer.event_names()
    assert "documentation_merge_finished" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_per_symbol_duplicate_removal(
    _mock_index: Any, multi_symbol_repo: Path
) -> None:
    """Identical summaries/examples across symbols are deduplicated."""
    shared = _symbol_payload("add", "Shared summary text.")
    shared["example_usage"] = "shared_example()"
    shared["returns"] = "shared return"
    client = _mock_client(content=json.dumps(shared))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="s4",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document math.py",
            context={
                "repo_path": str(multi_symbol_repo),
                "file_path": str(multi_symbol_repo / "math.py"),
                "doc_type": "module",
            },
        )
    )

    assert response.success is True
    assert response.output.summary.count("Shared summary text.") == 1
    assert response.output.example_usage.count("shared_example()") == 1
    assert response.output.returns.count("shared return") == 1


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_per_symbol_one_failure_keeps_others(
    _mock_index: Any, multi_symbol_repo: Path
) -> None:
    """One failing symbol should not discard successful documentation."""
    from codebase_assistant.tracing.tracer import Tracer

    good = _symbol_payload("add", "Adds values.")
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(good), usage={}, raw={}),
        ModelResponse(content="not-json {", usage={}, raw={}),
        ModelResponse(content="still-bad", usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="sym-partial")

    response = agent.handle(
        AgentRequest(
            task_id="s5",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document math.py",
            context={
                "repo_path": str(multi_symbol_repo),
                "file_path": str(multi_symbol_repo / "math.py"),
                "doc_type": "module",
            },
        )
    )

    assert response.success is True
    assert "Adds values." in response.output.summary
    assert "Generation Warnings" in response.output.summary
    assert "documentation_symbol_failed" in agent.tracer.event_names()
    assert "documentation_merge_finished" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_per_symbol_all_failures_abstain(
    _mock_index: Any, multi_symbol_repo: Path
) -> None:
    """Abstain only when every symbol documentation attempt fails."""
    from codebase_assistant.tracing.tracer import Tracer

    client = _mock_client(content="not-json")
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="sym-all-fail")

    response = agent.handle(
        AgentRequest(
            task_id="s6",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document math.py",
            context={
                "repo_path": str(multi_symbol_repo),
                "file_path": str(multi_symbol_repo / "math.py"),
                "doc_type": "module",
            },
        )
    )

    assert response.success is False
    assert response.output.abstention is not None
    assert "every discovered public symbol" in response.output.abstention.reason.lower()
    assert "documentation_symbol_failed" in agent.tracer.event_names()
    assert "documentation_merge_started" not in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_per_symbol_grounding_still_works(
    _mock_index: Any, multi_symbol_repo: Path
) -> None:
    """Per-symbol grounding still strips hallucinated references."""
    payload = {
        "file_path": "math.py",
        "function_name": "add",
        "summary": "add sums values. Also see FakeModule and the function ghost_helper.",
        "parameters": [
            {"name": "a", "type": "int", "description": "First."},
            {
                "name": "ghost_helper",
                "type": "function",
                "description": "Missing.",
            },
        ],
        "returns": "sum",
        "example_usage": "add(1, 2)",
    }
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="s7",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document add",
            context={
                "repo_path": str(multi_symbol_repo),
                "file_path": str(multi_symbol_repo / "math.py"),
                "function_name": "add",
                "doc_type": "docstring",
            },
        )
    )

    assert response.success is True
    assert "ghost_helper" not in response.output.summary
    assert "FakeModule" not in response.output.summary


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_per_symbol_json_retry_still_works(
    _mock_index: Any, multi_symbol_repo: Path
) -> None:
    """Malformed JSON still triggers one repair retry per symbol."""
    from codebase_assistant.tracing.tracer import Tracer

    good = _symbol_payload("add", "Adds values.")
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content="not-json {{{", usage={}, raw={}),
        ModelResponse(content=json.dumps(good), usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="sym-retry")

    response = agent.handle(
        AgentRequest(
            task_id="s8",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document add",
            context={
                "repo_path": str(multi_symbol_repo),
                "file_path": str(multi_symbol_repo / "math.py"),
                "function_name": "add",
                "doc_type": "docstring",
            },
        )
    )

    assert response.success is True
    assert client.generate.call_count == 2
    assert "documentation_retry_success" in agent.tracer.event_names()


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_per_symbol_writeback_still_works(
    _mock_index: Any, multi_symbol_repo: Path
) -> None:
    """Write-back still inserts a docstring after per-symbol generation."""
    payload = _symbol_payload("add", "Return the arithmetic sum.")
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="s9",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document add",
            context={
                "repo_path": str(multi_symbol_repo),
                "file_path": str(multi_symbol_repo / "math.py"),
                "function_name": "add",
                "doc_type": "docstring",
                "write_to_disk": True,
            },
        )
    )

    source = (multi_symbol_repo / "math.py").read_text(encoding="utf-8")
    assert response.success is True
    assert "Return the arithmetic sum." in source
    assert "def subtract" in source
