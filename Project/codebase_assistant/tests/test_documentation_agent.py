"""
test_documentation_agent.py
=============================

Unit tests for DocumentationAgent.

The LLM client and Retriever are mocked so tests do not require a
running Ollama server or a populated Chroma index. Filesystem reads use
a temporary repository on disk.
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


def test_ollama_unavailable(sample_repo: Path) -> None:
    """Unavailable Ollama should fail gracefully without calling generate."""
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
