"""
test_retriever_rerank.py
=========================

Unit tests for Retriever.rerank() and the optional cross-encoder pass
inside retrieve(). Vector search and the cross-encoder are mocked so
tests stay offline.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

from codebase_assistant.config import Config
from codebase_assistant.rag.retriever import Retriever
from codebase_assistant.rag.vectordb import SearchResult
from codebase_assistant.schemas.schemas import CodeChunk, RetrievedChunk


def _chunk(file_path: str, content: str, chunk_id: str = "") -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id or file_path,
        file_path=file_path,
        content=content,
        language="python",
        line_start=1,
        line_end=2,
    )


def _hit(file_path: str, content: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=_chunk(file_path, content),
        distance=1.0 - score,
        score=score,
    )


def _retrieved(source: str, content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(source=source, content=content, score=score, metadata={})


def _retriever(
    hits: List[SearchResult],
    *,
    rerank_enabled: bool = False,
    reranker: object = None,
    rerank_candidates: int = 24,
    top_k: int = 3,
) -> Retriever:
    config = Config(
        retrieval_top_k=top_k,
        rerank_enabled=rerank_enabled,
        rerank_candidates=rerank_candidates,
    )
    retriever = Retriever(
        config=config,
        rerank_enabled=rerank_enabled,
        reranker=reranker,
    )
    retriever._vector_db = MagicMock()
    retriever._vector_db.count.return_value = max(len(hits), 1)
    retriever._vector_db.search.return_value = list(hits)
    retriever._embedder = MagicMock()
    retriever._embedder.embed_text.return_value = [0.1, 0.2, 0.3]
    return retriever


def test_retrieval_unchanged_without_reranker() -> None:
    """With reranking off, retrieve must keep the vector-search order."""
    hits = [
        _hit("a.py", "def alpha(): pass", 0.9),
        _hit("b.py", "def beta(): pass", 0.8),
        _hit("c.py", "def gamma(): pass", 0.7),
    ]
    retriever = _retriever(hits, rerank_enabled=False, top_k=3)

    results = retriever.retrieve("find alpha")

    assert [chunk.source for chunk in results] == ["a.py", "b.py", "c.py"]
    assert [chunk.score for chunk in results] == [0.9, 0.8, 0.7]
    retriever.vector_db.search.assert_called_once()
    assert retriever.vector_db.search.call_args.kwargs["top_k"] == 3


def test_reranking_changes_ordering_when_enabled() -> None:
    """A cross-encoder that prefers later chunks should reorder results."""
    hits = [
        _hit("a.py", "alpha helper", 0.9),
        _hit("b.py", "beta helper", 0.8),
        _hit("c.py", "gamma helper", 0.7),
    ]
    reranker = MagicMock()
    # Prefer c, then a, then b.
    reranker.predict.return_value = [0.2, 0.1, 0.95]
    retriever = _retriever(
        hits, rerank_enabled=True, reranker=reranker, top_k=3, rerank_candidates=3
    )

    results = retriever.retrieve("gamma")

    assert [chunk.source for chunk in results] == ["c.py", "a.py", "b.py"]
    assert results[0].score == 0.95
    assert results[0].metadata["vector_score"] == 0.7
    assert results[0].metadata["rerank_score"] == 0.95
    reranker.predict.assert_called_once()


def test_rerank_method_reorders_explicit_candidates() -> None:
    """Calling rerank() directly should reorder the supplied list."""
    chunks = [
        _retrieved("a.py", "alpha", 0.9),
        _retrieved("b.py", "beta", 0.5),
    ]
    reranker = MagicMock()
    reranker.predict.return_value = [0.1, 0.9]
    retriever = _retriever([], rerank_enabled=True, reranker=reranker)

    ranked = retriever.rerank("beta", chunks)

    assert [chunk.source for chunk in ranked] == ["b.py", "a.py"]


def test_graceful_fallback_when_reranker_missing() -> None:
    """Enabled reranking with no model must return the original hits."""
    hits = [
        _hit("a.py", "alpha", 0.9),
        _hit("b.py", "beta", 0.5),
    ]
    retriever = _retriever(
        hits, rerank_enabled=True, reranker=None, top_k=2, rerank_candidates=2
    )
    # Force the lazy loader to report unavailable without downloading.
    retriever._get_reranker = lambda: None  # type: ignore[method-assign]

    results = retriever.retrieve("alpha")

    assert [chunk.source for chunk in results] == ["a.py", "b.py"]
    assert [chunk.score for chunk in results] == [0.9, 0.5]


def test_graceful_fallback_when_predict_raises() -> None:
    """A scoring failure must not discard the vector-search results."""
    chunks = [
        _retrieved("a.py", "alpha", 0.9),
        _retrieved("b.py", "beta", 0.4),
    ]
    reranker = MagicMock()
    reranker.predict.side_effect = RuntimeError("cuda unavailable")
    retriever = _retriever([], rerank_enabled=True, reranker=reranker)

    ranked = retriever.rerank("alpha", chunks)

    assert ranked == chunks


def test_retrieve_fetches_wider_candidate_pool_when_reranking() -> None:
    """Reranking should search more candidates than the final top_k."""
    hits = [
        _hit(f"f{i}.py", f"content {i}", 1.0 - (i * 0.01)) for i in range(6)
    ]
    reranker = MagicMock()
    reranker.predict.return_value = [float(i) for i in range(6)]
    retriever = _retriever(
        hits,
        rerank_enabled=True,
        reranker=reranker,
        top_k=2,
        rerank_candidates=6,
    )

    results = retriever.retrieve("query")

    assert retriever.vector_db.search.call_args.kwargs["top_k"] == 6
    assert len(results) == 2
    # Highest cross-encoder scores were the last candidates.
    assert [chunk.source for chunk in results] == ["f5.py", "f4.py"]


def test_disabled_rerank_sorts_by_existing_score_only() -> None:
    """Disabled rerank keeps the score-sort fallback for merged lists."""
    chunks = [
        _retrieved("low.py", "x", 0.1),
        _retrieved("high.py", "y", 0.9),
    ]
    retriever = _retriever([], rerank_enabled=False)

    ranked = retriever.rerank("anything", chunks)

    assert [chunk.source for chunk in ranked] == ["high.py", "low.py"]
