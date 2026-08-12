"""Unit tests for Analysis-specific retrieval selection helpers."""

from __future__ import annotations

from codebase_assistant.agents.code_analysis_agent import (
    DEFAULT_QUESTION,
    CodeAnalysisAgent,
    _ANALYSIS_RETRIEVAL_QUERY_EXPANSION,
    _SMALL_REPO_ANALYSIS_MAX_CHUNKS,
)
from codebase_assistant.schemas.schemas import RetrievedChunk


def _chunk(
    source: str,
    content: str = "def f():\n    return 1\n",
    *,
    language: str = "python",
    function_name: str | None = "f",
    chunk_id: str | None = None,
    score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        source=source,
        content=content,
        score=score,
        metadata={
            "chunk_id": chunk_id or f"{source}:{function_name}",
            "file_path": source,
            "language": language,
            "function_name": function_name,
            "line_start": 1,
            "line_end": 2,
        },
    )


def test_analysis_retrieval_query_keeps_question_and_expands() -> None:
    query = CodeAnalysisAgent._analysis_retrieval_query(
        "Find bugs and potential issues"
    )
    assert query.startswith("Find bugs and potential issues")
    assert _ANALYSIS_RETRIEVAL_QUERY_EXPANSION in query


def test_analysis_retrieval_query_defaults_when_blank() -> None:
    query = CodeAnalysisAgent._analysis_retrieval_query("   ")
    assert query.startswith(DEFAULT_QUESTION)


def test_select_prefers_code_over_readme_and_init() -> None:
    candidates = [
        _chunk("README.md", "# bugs", language="markdown", function_name=None, score=0.9),
        _chunk("shop/__init__.py", "from .x import y", function_name=None, score=0.8),
        _chunk("shop/auth.py", function_name="is_admin", score=0.2),
        _chunk("shop/billing.py", function_name="apply_discount", score=0.1),
    ]
    selected = CodeAnalysisAgent._select_analysis_chunks(candidates, limit=2)
    assert [c.source for c in selected] == ["shop/auth.py", "shop/billing.py"]


def test_select_fills_with_overview_only_when_needed() -> None:
    candidates = [
        _chunk("README.md", "# docs", language="markdown", function_name=None, score=0.9),
        _chunk("shop/auth.py", function_name="authorize", score=0.1),
    ]
    selected = CodeAnalysisAgent._select_analysis_chunks(candidates, limit=2)
    assert [c.source for c in selected] == ["shop/auth.py", "README.md"]


def test_select_dedupes_by_chunk_id() -> None:
    a = _chunk("shop/a.py", chunk_id="same", score=0.5)
    b = _chunk("shop/a.py", content="other", chunk_id="same", score=0.4)
    selected = CodeAnalysisAgent._select_analysis_chunks([a, b], limit=2)
    assert len(selected) == 1


def test_small_repo_max_constant_is_reasonable() -> None:
    # Guardrail: coverage improvement must stay modest vs dumping a repo.
    assert 8 < _SMALL_REPO_ANALYSIS_MAX_CHUNKS <= 24
