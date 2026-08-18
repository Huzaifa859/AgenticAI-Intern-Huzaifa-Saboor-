"""Unit tests for Analysis-specific retrieval selection helpers."""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

from codebase_assistant.agents.code_analysis_agent import (
    DEFAULT_QUESTION,
    CodeAnalysisAgent,
    _ANALYSIS_CONTEXT_LIMITS,
    _ANALYSIS_FETCH_POOL_CAP,
    _ANALYSIS_LARGE_MAX_CHUNKS,
    _ANALYSIS_MEDIUM_MAX_CHUNKS,
    _ANALYSIS_MEDIUM_PREFERRED_MAX,
    _ANALYSIS_RETRIEVAL_QUERY_EXPANSION,
    _ANALYSIS_SMALL_MAX_CHUNKS,
    _ANALYSIS_SMALL_PREFERRED_MAX,
)
from codebase_assistant.config import Config
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


def _agent_with_preferred(preferred: Optional[int]) -> CodeAnalysisAgent:
    agent = CodeAnalysisAgent(config=Config(retrieval_top_k=8, rerank_candidates=8))
    agent._preferred_code_chunk_count = MagicMock(return_value=preferred)  # type: ignore[method-assign]
    return agent


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


def test_size_bucket_thresholds() -> None:
    assert CodeAnalysisAgent._analysis_repo_size_bucket(1) == "small"
    assert (
        CodeAnalysisAgent._analysis_repo_size_bucket(_ANALYSIS_SMALL_PREFERRED_MAX)
        == "small"
    )
    assert (
        CodeAnalysisAgent._analysis_repo_size_bucket(_ANALYSIS_SMALL_PREFERRED_MAX + 1)
        == "medium"
    )
    assert (
        CodeAnalysisAgent._analysis_repo_size_bucket(_ANALYSIS_MEDIUM_PREFERRED_MAX)
        == "medium"
    )
    assert (
        CodeAnalysisAgent._analysis_repo_size_bucket(_ANALYSIS_MEDIUM_PREFERRED_MAX + 1)
        == "large"
    )
    assert CodeAnalysisAgent._analysis_repo_size_bucket(None) == "medium"


def test_small_repo_max_chunks_is_12() -> None:
    agent = _agent_with_preferred(10)
    retriever = MagicMock()
    assert agent._analysis_retrieval_limit(retriever, top_k=None) == _ANALYSIS_SMALL_MAX_CHUNKS
    assert _ANALYSIS_CONTEXT_LIMITS["small"] == 12


def test_medium_repo_max_chunks_is_20() -> None:
    agent = _agent_with_preferred(_ANALYSIS_SMALL_PREFERRED_MAX + 5)
    retriever = MagicMock()
    assert agent._analysis_retrieval_limit(retriever, top_k=None) == _ANALYSIS_MEDIUM_MAX_CHUNKS
    assert _ANALYSIS_CONTEXT_LIMITS["medium"] == 20


def test_large_repo_max_chunks_is_32() -> None:
    agent = _agent_with_preferred(_ANALYSIS_MEDIUM_PREFERRED_MAX + 5)
    retriever = MagicMock()
    assert agent._analysis_retrieval_limit(retriever, top_k=None) == _ANALYSIS_LARGE_MAX_CHUNKS
    assert _ANALYSIS_CONTEXT_LIMITS["large"] == 32


def test_analysis_ignores_global_retrieval_top_k_of_eight() -> None:
    """Config.retrieval_top_k=8 must not silently cap Analysis at 8."""
    agent = _agent_with_preferred(_ANALYSIS_MEDIUM_PREFERRED_MAX + 1)
    assert agent.config.retrieval_top_k == 8
    assert agent._analysis_retrieval_limit(MagicMock(), top_k=None) == 32


def test_fetch_pool_supports_each_size_maximum() -> None:
    agent = CodeAnalysisAgent(config=Config(rerank_candidates=8))
    for limit in (
        _ANALYSIS_SMALL_MAX_CHUNKS,
        _ANALYSIS_MEDIUM_MAX_CHUNKS,
        _ANALYSIS_LARGE_MAX_CHUNKS,
    ):
        fetch_k = agent._analysis_fetch_k(limit, preferred_n=limit * 3)
        assert fetch_k >= limit
        assert fetch_k <= _ANALYSIS_FETCH_POOL_CAP


def test_fewer_available_chunks_sends_only_available() -> None:
    available = [
        _chunk(f"shop/mod{i}.py", function_name=f"f{i}", chunk_id=f"id{i}")
        for i in range(5)
    ]
    selected = CodeAnalysisAgent._select_analysis_chunks(available, limit=12)
    assert len(selected) == 5


def test_retrieve_for_analysis_uses_size_limit_and_wider_fetch() -> None:
    agent = _agent_with_preferred(25)  # small → max 12
    captured: dict[str, int] = {}

    def fake_retrieve(query: str, top_k: int = 8, where=None):
        captured["top_k"] = top_k
        return [
            _chunk(f"shop/a{i}.py", function_name=f"f{i}", chunk_id=f"c{i}")
            for i in range(top_k)
        ]

    retriever = MagicMock()
    retriever.retrieve.side_effect = fake_retrieve
    Retriever = MagicMock()
    # build_filter is a static/classmethod on Retriever used inside method
    from codebase_assistant.rag.retriever import Retriever as RealRetriever

    original = RealRetriever.build_filter
    RealRetriever.build_filter = staticmethod(lambda **kwargs: {"language": "python"})  # type: ignore[assignment]
    try:
        selected = agent._retrieve_for_analysis(retriever, "Find bugs", top_k=None)
    finally:
        RealRetriever.build_filter = original  # type: ignore[assignment]

    assert len(selected) == _ANALYSIS_SMALL_MAX_CHUNKS
    assert captured["top_k"] >= _ANALYSIS_SMALL_MAX_CHUNKS
    # Global rerank_candidates=8 must not win over Analysis fetch sizing.
    assert captured["top_k"] > 8


def test_retrieve_for_analysis_medium_and_large_keep_maxima() -> None:
    cases = [
        (_ANALYSIS_SMALL_PREFERRED_MAX + 10, _ANALYSIS_MEDIUM_MAX_CHUNKS),
        (_ANALYSIS_MEDIUM_PREFERRED_MAX + 10, _ANALYSIS_LARGE_MAX_CHUNKS),
    ]
    from codebase_assistant.rag.retriever import Retriever as RealRetriever

    original = RealRetriever.build_filter
    RealRetriever.build_filter = staticmethod(lambda **kwargs: {"language": "python"})  # type: ignore[assignment]
    try:
        for preferred, expected_keep in cases:
            agent = _agent_with_preferred(preferred)
            agent.config = Config(retrieval_top_k=8, rerank_candidates=8)

            def fake_retrieve(
                query: str,
                top_k: int = 8,
                where=None,
                _expected: int = expected_keep,
            ) -> List[RetrievedChunk]:
                return [
                    _chunk(
                        f"shop/x{i}.py",
                        function_name=f"f{i}",
                        chunk_id=f"{_expected}-{i}",
                    )
                    for i in range(top_k)
                ]

            retriever = MagicMock()
            retriever.retrieve.side_effect = fake_retrieve
            selected = agent._retrieve_for_analysis(
                retriever, "Find bugs", top_k=None
            )
            assert len(selected) == expected_keep
            assert retriever.retrieve.call_args.kwargs["top_k"] >= expected_keep
    finally:
        RealRetriever.build_filter = original  # type: ignore[assignment]
