"""Focused tests for Documentation size-based retrieval (8 / 16 / 24)."""

from __future__ import annotations

import re
from typing import Optional
from unittest.mock import MagicMock

from codebase_assistant.agents.code_analysis_agent import (
    _ANALYSIS_CONTEXT_LIMITS,
    _ANALYSIS_MEDIUM_PREFERRED_MAX,
    _ANALYSIS_SMALL_PREFERRED_MAX,
    _ANALYSIS_SMALL_MAX_CHUNKS,
    _LOW_VALUE_ANALYSIS_BASENAMES,
)
from codebase_assistant.agents.documentation_agent import (
    DocumentationAgent,
    _DOC_CONTEXT_LIMITS,
    _DOC_FETCH_POOL_CAP,
    _DOC_LARGE_MAX_CHUNKS,
    _DOC_MEDIUM_MAX_CHUNKS,
    _DOC_MEDIUM_SIZE_MAX,
    _DOC_SMALL_MAX_CHUNKS,
    _DOC_SMALL_SIZE_MAX,
    _MAX_CHUNK_CHARS,
    _MAX_CONTEXT_CHUNKS,
)
from codebase_assistant.agents.testing_agent import _MAX_CONTEXT_CHUNKS as TESTING_MAX_CHUNKS
from codebase_assistant.config import Config
from codebase_assistant.schemas.schemas import RetrievedChunk


def _chunk(
    source: str,
    idx: int = 0,
    *,
    language: str = "python",
    content: str | None = None,
    score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        source=source,
        content=content or f"def f{idx}():\n    return {idx}\n",
        score=score,
        metadata={
            "chunk_id": f"{source}:{idx}",
            "file_path": source,
            "language": language,
            "function_name": f"f{idx}",
            "line_start": 1,
            "line_end": 2,
        },
    )


def _indexed_chunk(path: str, language: str = "python"):
    chunk = MagicMock()
    chunk.file_path = path
    chunk.metadata = {"file_path": path, "language": language}
    return chunk


def _agent_with_size_count(relevant: Optional[int]) -> DocumentationAgent:
    retriever = MagicMock()
    retriever.config = Config(retrieval_top_k=8, rerank_candidates=8)
    retriever.vector_db = MagicMock()
    agent = DocumentationAgent(retriever=retriever)
    agent._documentation_relevant_chunk_count = MagicMock(return_value=relevant)  # type: ignore[method-assign]
    return agent


def test_docs_bucket_thresholds_match_analysis_cutoffs() -> None:
    assert _DOC_SMALL_SIZE_MAX == _ANALYSIS_SMALL_PREFERRED_MAX == 40
    assert _DOC_MEDIUM_SIZE_MAX == _ANALYSIS_MEDIUM_PREFERRED_MAX == 150
    assert DocumentationAgent._docs_repo_size_bucket(10) == "small"
    assert DocumentationAgent._docs_repo_size_bucket(_DOC_SMALL_SIZE_MAX + 1) == "medium"
    assert DocumentationAgent._docs_repo_size_bucket(_DOC_MEDIUM_SIZE_MAX + 1) == "large"
    assert _DOC_CONTEXT_LIMITS == {"small": 8, "medium": 16, "large": 24}
    assert _ANALYSIS_CONTEXT_LIMITS["small"] == _ANALYSIS_SMALL_MAX_CHUNKS == 12


def test_readme_counts_for_docs_size_unlike_analysis() -> None:
    assert "readme.md" in _LOW_VALUE_ANALYSIS_BASENAMES
    assert DocumentationAgent._is_documentation_size_relevant_path("README.md")
    assert DocumentationAgent._is_documentation_size_relevant_path("docs/guide.md")
    assert DocumentationAgent._is_documentation_size_relevant_path("pkg/__init__.py")


def test_vendor_lock_cache_paths_do_not_count() -> None:
    assert not DocumentationAgent._is_documentation_size_relevant_path(
        "node_modules/left-pad/index.py"
    )
    assert not DocumentationAgent._is_documentation_size_relevant_path(
        "vendor/lib/util.py"
    )
    assert not DocumentationAgent._is_documentation_size_relevant_path("poetry.lock")
    assert not DocumentationAgent._is_documentation_size_relevant_path(
        ".pytest_cache/v/cache.txt"
    )
    assert not DocumentationAgent._is_documentation_size_relevant_path(
        "dist/pkg-1.0/__init__.py"
    )


def test_documentation_relevant_chunk_count_includes_readme() -> None:
    agent = DocumentationAgent(retriever=MagicMock())
    agent.retriever.vector_db.list_chunks.return_value = [
        _indexed_chunk("shop/auth.py"),
        _indexed_chunk("README.md", language="markdown"),
        _indexed_chunk("poetry.lock", language="text"),
        _indexed_chunk("node_modules/x/index.py"),
    ]
    assert agent._documentation_relevant_chunk_count() == 2


def test_small_repo_docs_max_is_8() -> None:
    agent = _agent_with_size_count(10)
    assert agent._docs_retrieval_limit() == _DOC_SMALL_MAX_CHUNKS
    assert _MAX_CONTEXT_CHUNKS == 8


def test_medium_repo_docs_max_is_16() -> None:
    agent = _agent_with_size_count(_DOC_SMALL_SIZE_MAX + 5)
    assert agent._docs_retrieval_limit() == _DOC_MEDIUM_MAX_CHUNKS


def test_large_repo_docs_max_is_24() -> None:
    agent = _agent_with_size_count(_DOC_MEDIUM_SIZE_MAX + 5)
    assert agent._docs_retrieval_limit() == _DOC_LARGE_MAX_CHUNKS


def test_global_retrieval_top_k_eight_does_not_cap_docs() -> None:
    agent = _agent_with_size_count(_DOC_MEDIUM_SIZE_MAX + 1)
    assert agent.retriever.config.retrieval_top_k == 8
    assert agent._docs_retrieval_limit() == 24


def test_fetch_pool_supports_medium_and_large() -> None:
    agent = _agent_with_size_count(200)
    for limit in (_DOC_MEDIUM_MAX_CHUNKS, _DOC_LARGE_MAX_CHUNKS):
        fetch_k = agent._docs_fetch_k(limit, relevant_n=limit * 3)
        assert fetch_k >= limit
        assert fetch_k <= _DOC_FETCH_POOL_CAP
        assert fetch_k > 8


def test_fewer_available_chunks_keeps_only_available() -> None:
    agent = _agent_with_size_count(10)  # small → max 8
    available = [_chunk("mod.py", i) for i in range(5)]
    agent.retriever.retrieve.return_value = available
    selected = agent._retrieve_context("document", "mod.py")
    assert len(selected) == 5
    assert agent.retriever.retrieve.call_args.kwargs["top_k"] >= 8


def test_retrieve_requests_fetch_k_not_global_eight() -> None:
    agent = _agent_with_size_count(_DOC_SMALL_SIZE_MAX + 10)  # medium → 16
    agent.retriever.retrieve.return_value = [
        _chunk(f"a{i}.py", i) for i in range(40)
    ]
    selected = agent._retrieve_context("readme overview", ".")
    assert len(selected) == 16
    assert agent.retriever.retrieve.call_args.kwargs["top_k"] >= 16
    assert agent.retriever.retrieve.call_args.kwargs["top_k"] > 8


def test_build_prompt_keeps_inventory_and_exact_rag_count() -> None:
    """RAG keep count is independent of dedicated README/inventory sections."""
    agent = DocumentationAgent(retriever=MagicMock())
    chunks = [_chunk(f"m{i}.py", i) for i in range(16)]
    prompt = agent._build_prompt(
        mode="readme",
        instruction="Write a README",
        target_path=".",
        function_name="README",
        chunks=chunks,
        source_excerpts=["### a.py\npass\n"],
        inventory=["a.py", "README.md"],
        project_files=["### README.md\n# Demo\n"],
    )
    markers = re.findall(r"^\[\d+\] source=", prompt, flags=re.M)
    assert len(markers) == 16
    assert "RETRIEVED CONTEXT" in prompt
    assert "REPOSITORY INVENTORY" in prompt
    assert "PROJECT FILES" in prompt
    assert "README.md" in prompt
    assert _MAX_CHUNK_CHARS == 1200


def test_analysis_and_testing_limits_unchanged() -> None:
    assert _ANALYSIS_CONTEXT_LIMITS == {"small": 12, "medium": 20, "large": 32}
    assert TESTING_MAX_CHUNKS == 5


def test_large_retrieve_caps_at_24_in_prompt_path() -> None:
    agent = _agent_with_size_count(_DOC_MEDIUM_SIZE_MAX + 20)
    agent.retriever.retrieve.return_value = [
        _chunk(f"x{i}.py", i) for i in range(60)
    ]
    selected = agent._retrieve_context("documentation", ".")
    assert len(selected) == 24
    prompt = agent._build_prompt(
        mode="readme",
        instruction="Write a README",
        target_path=".",
        function_name="README",
        chunks=selected,
        source_excerpts=[],
        inventory=["x0.py", "README.md"],
        project_files=["### README.md\nOverview\n"],
    )
    assert len(re.findall(r"^\[\d+\] source=", prompt, flags=re.M)) == 24
    assert "PROJECT FILES" in prompt
