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
def test_readme_prompt_includes_inventory_and_project_files(
    _mock_index: Any, project_repo: Path
) -> None:
    """README mode should ground layout and setup claims in real files."""
    chunk = RetrievedChunk(
        source="math_utils.py",
        content="def add(a, b):\n    return a + b",
        score=0.9,
        metadata={"file_path": "math_utils.py"},
    )
    client = _mock_client(content=json.dumps(VALID_DOC_PAYLOAD))
    agent = _agent(client, _mock_retriever([chunk]))

    agent.handle(_readme_request(project_repo))

    prompt = client.generate.call_args.args[0][1].content
    assert "REPOSITORY INVENTORY" in prompt
    assert "app/main.py" in prompt
    assert "PROJECT FILES" in prompt
    assert "chromadb>=0.4" in prompt


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
    assert response.output.function_name == ""
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
