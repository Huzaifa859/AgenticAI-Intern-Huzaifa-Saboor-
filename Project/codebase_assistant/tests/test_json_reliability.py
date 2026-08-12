"""
test_json_reliability.py
=========================

Cross-agent coverage for structured JSON generation, fence cleanup,
and the single LLM JSON-repair attempt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from codebase_assistant.agents.code_analysis_agent import CodeAnalysisAgent
from codebase_assistant.agents.documentation_agent import DocumentationAgent
from codebase_assistant.agents.testing_agent import TestingAgent
from codebase_assistant.config import Config
from codebase_assistant.schemas.schemas import (
    AgentRequest,
    AgentType,
    ModelResponse,
    RetrievedChunk,
)
from codebase_assistant.utils.json_output import JSON_OBJECT_RESPONSE_FORMAT

VALID_ANALYSIS = {
    "answer": "No additional issues found.",
    "findings": [],
}

VALID_TEST = {
    "summary": "Pytest coverage for add.",
    "generated_tests": {
        "test_math_utils.py": (
            "from math_utils import add\n\n"
            "def test_add_happy_path():\n"
            "    assert add(1, 2) == 3\n"
        )
    },
    "coverage_estimate": 0.5,
}


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Tiny repository used by agent-level JSON tests."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


def _mock_client(content: str = "") -> MagicMock:
    client = MagicMock()
    client.is_available.return_value = True
    response = ModelResponse(content=content, usage={}, raw={})
    client.generate.return_value = response

    def _stream(_messages, on_chunk=None, **_kwargs):
        if on_chunk is not None and content:
            on_chunk(content)
        return response

    client.generate_stream.side_effect = _stream
    client.config = Config(openrouter_api_key=None)
    return client


def _mock_retriever(
    chunks: Optional[List[RetrievedChunk]] = None,
) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = list(chunks or [])
    retriever.vector_store_path = "./.codebase_assistant/chroma"
    retriever.config = MagicMock()
    retriever.vector_db = MagicMock()
    return retriever


def test_analysis_valid_structured_output_streams_without_repair(
    sample_repo: Path,
) -> None:
    """Valid Analysis JSON should stream once and skip the repair call."""
    client = _mock_client(content=json.dumps(VALID_ANALYSIS))
    agent = CodeAnalysisAgent(model_client=client, retriever=_mock_retriever())

    with patch.object(agent, "_sync_index", return_value=None):
        report = agent.analyze_repository(str(sample_repo), use_rag=True)

    assert report.model_used is True
    assert report.llm_parse_failed is False
    assert client.generate_stream.call_count == 1
    assert client.generate.call_count == 0


def test_analysis_fenced_json_cleaned_without_repair(sample_repo: Path) -> None:
    """Markdown-fenced JSON should parse after deterministic cleanup."""
    fenced = (
        "```json\n"
        + json.dumps(VALID_ANALYSIS)
        + "\n```"
    )
    client = _mock_client(content=fenced)
    agent = CodeAnalysisAgent(model_client=client, retriever=_mock_retriever())

    with patch.object(agent, "_sync_index", return_value=None):
        report = agent.analyze_repository(str(sample_repo), use_rag=True)

    assert report.llm_parse_failed is False
    assert client.generate_stream.call_count == 1
    assert client.generate.call_count == 0
    assert report.answer == VALID_ANALYSIS["answer"]


def test_analysis_invalid_json_repaired_once(sample_repo: Path) -> None:
    """Invalid Analysis JSON gets exactly one repair attempt."""
    client = _mock_client()
    client.generate_stream.side_effect = lambda *_a, **_k: ModelResponse(
        content="not-json {{{", usage={}, raw={}
    )
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(VALID_ANALYSIS), usage={}, raw={}),
    ]
    agent = CodeAnalysisAgent(model_client=client, retriever=_mock_retriever())

    with patch.object(agent, "_sync_index", return_value=None):
        report = agent.analyze_repository(str(sample_repo), use_rag=True)

    assert report.llm_parse_failed is False
    assert client.generate_stream.call_count == 1
    assert client.generate.call_count == 1
    assert report.answer == VALID_ANALYSIS["answer"]
    repair_system = client.generate.call_args_list[0].args[0][0].content
    assert "JSON repair" in repair_system
    assert "code fences" in repair_system.lower()


def test_analysis_invalid_json_after_single_repair(sample_repo: Path) -> None:
    """A failed repair must not trigger a second JSON repair call."""
    client = _mock_client()
    client.generate_stream.side_effect = lambda *_a, **_k: ModelResponse(
        content="broken-1", usage={}, raw={}
    )
    client.generate.side_effect = [
        ModelResponse(content="broken-2", usage={}, raw={}),
        ModelResponse(content=json.dumps(VALID_ANALYSIS), usage={}, raw={}),
    ]
    agent = CodeAnalysisAgent(model_client=client, retriever=_mock_retriever())

    with patch.object(agent, "_sync_index", return_value=None):
        report = agent.analyze_repository(str(sample_repo), use_rag=True)

    assert report.llm_parse_failed is True
    assert client.generate_stream.call_count == 1
    assert client.generate.call_count == 1
    assert any("could not be parsed" in note.lower() for note in report.notes)


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_documentation_streams_freeform_markdown(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Documentation streams freeform markdown (no JSON response_format)."""
    client = _mock_client(content="## Purpose\n\nAdds two numbers.")
    agent = DocumentationAgent(model_client=client, retriever=_mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="doc-json",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document add.",
            context={
                "repo_path": str(sample_repo),
                "doc_type": "docstring",
                "file_path": str(sample_repo / "math_utils.py"),
                "function_name": "add",
            },
        )
    )

    assert response.success is True
    assert client.generate_stream.call_count == 1
    assert client.generate.call_count == 0


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_documentation_fenced_markdown_streams_without_generate(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Fenced documentation markdown is accepted via the stream path."""
    fenced = "```markdown\n## Purpose\n\nAdds two numbers.\n```"
    client = _mock_client(content=fenced)
    agent = DocumentationAgent(model_client=client, retriever=_mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="doc-fence",
            agent_type=AgentType.DOCUMENTATION,
            instruction="Document add.",
            context={
                "repo_path": str(sample_repo),
                "doc_type": "docstring",
                "file_path": str(sample_repo / "math_utils.py"),
                "function_name": "add",
            },
        )
    )

    assert response.success is True
    assert client.generate_stream.call_count == 1
    assert client.generate.call_count == 0


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_testing_valid_structured_output_uses_response_format(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Valid Testing JSON should stream once and request json_object."""
    client = _mock_client(content=json.dumps(VALID_TEST))
    agent = TestingAgent(model_client=client, retriever=_mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="test-json",
            agent_type=AgentType.TESTING,
            instruction="Generate pytest unit tests.",
            context={"repo_path": str(sample_repo)},
        )
    )

    assert response.success is True
    assert client.generate_stream.call_count == 1
    assert (
        client.generate_stream.call_args_list[0].kwargs.get("response_format")
        == JSON_OBJECT_RESPONSE_FORMAT
    )


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_testing_invalid_json_repaired_once(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Invalid Testing JSON gets exactly one JSON repair before pytest."""
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content="not-json {{{", usage={}, raw={}),
        ModelResponse(content=json.dumps(VALID_TEST), usage={}, raw={}),
    ]

    def _stream(messages, on_chunk=None, **kwargs):
        result = client.generate(messages, **kwargs)
        text = getattr(result, "content", "") or ""
        if on_chunk is not None and text:
            on_chunk(text)
        return result

    client.generate_stream.side_effect = _stream
    agent = TestingAgent(model_client=client, retriever=_mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="test-repair-ok",
            agent_type=AgentType.TESTING,
            instruction="Generate pytest unit tests.",
            context={"repo_path": str(sample_repo)},
        )
    )

    assert response.success is True
    assert client.generate_stream.call_count == 1
    assert client.generate.call_count == 2
    repair_system = client.generate.call_args_list[1].args[0][0].content
    assert "JSON repair" in repair_system
    assert "test_math_utils.py" in response.output.generated_tests


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_testing_invalid_json_after_single_repair(
    _mock_index: Any, sample_repo: Path
) -> None:
    """A failed Testing JSON repair must not retry again."""
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content="broken-1", usage={}, raw={}),
        ModelResponse(content="broken-2", usage={}, raw={}),
        ModelResponse(content=json.dumps(VALID_TEST), usage={}, raw={}),
    ]

    def _stream(messages, on_chunk=None, **kwargs):
        result = client.generate(messages, **kwargs)
        text = getattr(result, "content", "") or ""
        if on_chunk is not None and text:
            on_chunk(text)
        return result

    client.generate_stream.side_effect = _stream
    agent = TestingAgent(model_client=client, retriever=_mock_retriever())

    with patch.object(
        agent, "_retry_json_repair", wraps=agent._retry_json_repair
    ) as repair_spy:
        response = agent.handle(
            AgentRequest(
                task_id="test-repair-fail",
                agent_type=AgentType.TESTING,
                instruction="Generate pytest unit tests.",
                context={"repo_path": str(sample_repo)},
            )
        )

    assert repair_spy.call_count == 1
    assert client.generate.call_count == 2
    assert response.success is False
    assert response.output.abstention is not None
