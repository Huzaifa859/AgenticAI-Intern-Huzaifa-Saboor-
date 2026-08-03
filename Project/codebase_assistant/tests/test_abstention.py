"""
test_abstention.py
===================

Explicit abstention when agents lack grounded evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from codebase_assistant.agents.code_analysis_agent import CodeAnalysisAgent
from codebase_assistant.agents.documentation_agent import DocumentationAgent
from codebase_assistant.agents.testing_agent import TestingAgent
from codebase_assistant.schemas.schemas import (
    AgentRequest,
    AgentType,
    ModelResponse,
    RetrievedChunk,
)


def _mock_client(available: bool = True, content: str = "") -> MagicMock:
    client = MagicMock()
    client.is_available.return_value = available
    client.generate.return_value = ModelResponse(content=content, usage={}, raw={})
    return client


def _mock_retriever(chunks: Optional[List[RetrievedChunk]] = None) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = list(chunks or [])
    retriever.vector_store_path = "./.codebase_assistant/chroma"
    retriever.config = MagicMock()
    retriever.vector_db = MagicMock()
    return retriever


def test_analysis_abstains_on_empty_repository(tmp_path: Path) -> None:
    """An empty repository has no supported Python files to analyze."""
    agent = CodeAnalysisAgent(model_client=None, retriever=None)
    report = agent.analyze_repository(str(tmp_path), use_rag=False)

    assert report.abstention is not None
    assert report.findings == []
    assert "no supported Python files" in report.abstention.reason
    assert report.abstention.recommended_next_steps


def test_analysis_abstains_on_unsupported_repository(tmp_path: Path) -> None:
    """Non-Python repositories are unsupported for bug detection."""
    (tmp_path / "notes.md").write_text("# docs only\n", encoding="utf-8")
    (tmp_path / "data.txt").write_text("hello\n", encoding="utf-8")
    agent = CodeAnalysisAgent(model_client=None, retriever=None)
    report = agent.analyze_repository(str(tmp_path), use_rag=False)

    assert report.abstention is not None
    assert report.findings == []
    assert "no supported Python files" in report.abstention.reason


def test_analysis_abstains_when_retrieval_returns_nothing(tmp_path: Path) -> None:
    """Empty retrieval plus an empty model response yields abstention."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    # Valid empty JSON so this is not a parse failure; retrieval itself
    # produced no grounded context and the model offered no answer.
    client = _mock_client(content='{"answer": "", "findings": []}')
    retriever = _mock_retriever([])
    agent = CodeAnalysisAgent(model_client=client, retriever=retriever)

    with patch.object(agent, "_sync_index", return_value=None):
        report = agent.analyze_repository(str(tmp_path), use_rag=True)

    assert report.abstention is not None
    assert report.findings == []
    assert report.abstention.reason == "No grounded evidence was found."
    assert report.context == []


def test_analysis_abstains_when_all_findings_rejected(tmp_path: Path) -> None:
    """When every LLM finding fails grounding, abstain instead of inventing."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    hallucinated = {
        "answer": "Found a bug.",
        "findings": [
            {
                "bug_type": "undefined_variable",
                "description": "Invented issue",
                "severity": "high",
                "confidence": 0.9,
                "file_path": "math_utils.py",
                "function_name": "add",
                "line_start": 1,
                "line_end": 1,
                "evidence": "this evidence does not exist in the file",
                "detection_method": "llm",
            }
        ],
    }
    client = _mock_client(content=json.dumps(hallucinated))
    chunk = RetrievedChunk(
        source="math_utils.py",
        content="def add(a, b):\n    return a + b\n",
        score=0.9,
        metadata={"file_path": "math_utils.py"},
    )
    retriever = _mock_retriever([chunk])
    agent = CodeAnalysisAgent(model_client=client, retriever=retriever)

    with patch.object(agent, "_sync_index", return_value=None):
        report = agent.analyze_repository(str(tmp_path), use_rag=True)

    assert report.abstention is not None
    assert report.findings == []
    assert report.abstention.reason == "LLM response could not be verified."
    assert report.llm_proposed_count >= 1
    assert report.llm_grounded_count == 0


def test_documentation_abstains_without_evidence(tmp_path: Path) -> None:
    """Documentation abstains when the repository has no usable source."""
    client = _mock_client(content=json.dumps({"summary": "should not be used"}))
    retriever = _mock_retriever([])
    agent = DocumentationAgent(model_client=client, retriever=retriever)

    with patch.object(agent, "_ensure_index"):
        response = agent.handle(
            AgentRequest(
                task_id="doc-abs",
                agent_type=AgentType.DOCUMENTATION,
                instruction="Generate a README summary.",
                context={"repo_path": str(tmp_path), "doc_type": "readme"},
            )
        )

    assert response.success is False
    assert response.output is not None
    assert response.output.abstention is not None
    assert "no supported python files" in response.output.abstention.reason.lower()
    assert client.generate.call_count == 0


def test_testing_abstains_without_evidence(tmp_path: Path) -> None:
    """Testing abstains when there is no grounded source to test."""
    client = _mock_client(
        content=json.dumps(
            {
                "summary": "should not be used",
                "generated_tests": {"test_x.py": "def test_x():\n    assert True\n"},
                "coverage_estimate": 0.1,
            }
        )
    )
    retriever = _mock_retriever([])
    agent = TestingAgent(model_client=client, retriever=retriever)

    with patch.object(agent, "_ensure_index"):
        response = agent.handle(
            AgentRequest(
                task_id="test-abs",
                agent_type=AgentType.TESTING,
                instruction="Generate tests.",
                context={"repo_path": str(tmp_path)},
            )
        )

    assert response.success is False
    assert response.output is not None
    assert response.output.abstention is not None
    assert "no supported python files" in response.output.abstention.reason.lower()
    assert client.generate.call_count == 0


def test_documentation_abstains_on_unverifiable_model_output(tmp_path: Path) -> None:
    """Bad documentation JSON becomes an explicit abstention."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    client = _mock_client(content="not-json")
    retriever = _mock_retriever([])
    agent = DocumentationAgent(model_client=client, retriever=retriever)

    with patch.object(agent, "_ensure_index"):
        result = agent.generate_readme(str(tmp_path))

    assert result.abstention is not None
    assert result.abstention.reason == "LLM response could not be verified."
    assert result.summary == ""


def test_testing_abstains_on_unverifiable_model_output(tmp_path: Path) -> None:
    """Bad testing JSON becomes an explicit abstention."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    client = _mock_client(content="not-json")
    retriever = _mock_retriever([])
    agent = TestingAgent(model_client=client, retriever=retriever)

    with patch.object(agent, "_ensure_index"):
        result = agent.generate_unit_tests(str(tmp_path))

    assert result.abstention is not None
    assert result.abstention.reason == "LLM response could not be verified."
    assert result.generated_tests == {}
