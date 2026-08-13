"""Focused tests for Documentation size-based retrieval (8 / 16 / 24)."""

from __future__ import annotations

import re
from typing import Optional
from unittest.mock import MagicMock

from codebase_assistant.agents.code_analysis_agent import (
    CodeAnalysisAgent,
    _ANALYSIS_CONTEXT_LIMITS,
    _ANALYSIS_MEDIUM_PREFERRED_MAX,
    _ANALYSIS_SMALL_PREFERRED_MAX,
    _ANALYSIS_SMALL_MAX_CHUNKS,
)
from codebase_assistant.agents.documentation_agent import (
    DocumentationAgent,
    _DOC_CONTEXT_LIMITS,
    _DOC_FETCH_POOL_CAP,
    _DOC_LARGE_MAX_CHUNKS,
    _DOC_MEDIUM_MAX_CHUNKS,
    _DOC_SMALL_MAX_CHUNKS,
    _MAX_CHUNK_CHARS,
    _MAX_CONTEXT_CHUNKS,
)
from codebase_assistant.agents.testing_agent import _MAX_CONTEXT_CHUNKS as TESTING_MAX_CHUNKS
from codebase_assistant.config import Config
from codebase_assistant.schemas.schemas import RetrievedChunk


def _chunk(source: str, idx: int, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        source=source,
        content=f"def f{idx}():\n    return {idx}\n",
        score=score,
        metadata={
            "chunk_id": f"{source}:{idx}",
            "file_path": source,
            "language": "python",
            "function_name": f"f{idx}",
            "line_start": 1,
            "line_end": 2,
        },
    )


def _agent_with_preferred(preferred: Optional[int]) -> DocumentationAgent:
    retriever = MagicMock()
    retriever.config = Config(retrieval_top_k=8, rerank_candidates=8)
    retriever.vector_db = MagicMock()
    agent = DocumentationAgent(retriever=retriever)
    agent._preferred_code_chunk_count = MagicMock(return_value=preferred)  # type: ignore[method-assign]
    return agent


def test_docs_shares_analysis_size_buckets() -> None:
    assert CodeAnalysisAgent._analysis_repo_size_bucket(10) == "small"
    assert (
        CodeAnalysisAgent._analysis_repo_size_bucket(_ANALYSIS_SMALL_PREFERRED_MAX + 1)
        == "medium"
    )
    assert (
        CodeAnalysisAgent._analysis_repo_size_bucket(_ANALYSIS_MEDIUM_PREFERRED_MAX + 1)
        == "large"
    )
    # Docs uses those buckets with its own keep ceilings.
    assert _DOC_CONTEXT_LIMITS["small"] == 8
    assert _DOC_CONTEXT_LIMITS["medium"] == 16
    assert _DOC_CONTEXT_LIMITS["large"] == 24
    assert _ANALYSIS_CONTEXT_LIMITS["small"] == _ANALYSIS_SMALL_MAX_CHUNKS == 12


def test_small_repo_docs_max_is_8() -> None:
    agent = _agent_with_preferred(10)
    assert agent._docs_retrieval_limit() == _DOC_SMALL_MAX_CHUNKS
    assert _MAX_CONTEXT_CHUNKS == 8


def test_medium_repo_docs_max_is_16() -> None:
    agent = _agent_with_preferred(_ANALYSIS_SMALL_PREFERRED_MAX + 5)
    assert agent._docs_retrieval_limit() == _DOC_MEDIUM_MAX_CHUNKS


def test_large_repo_docs_max_is_24() -> None:
    agent = _agent_with_preferred(_ANALYSIS_MEDIUM_PREFERRED_MAX + 5)
    assert agent._docs_retrieval_limit() == _DOC_LARGE_MAX_CHUNKS


def test_global_retrieval_top_k_eight_does_not_cap_docs() -> None:
    agent = _agent_with_preferred(_ANALYSIS_MEDIUM_PREFERRED_MAX + 1)
    assert agent.retriever.config.retrieval_top_k == 8
    assert agent._docs_retrieval_limit() == 24


def test_fetch_pool_supports_medium_and_large() -> None:
    agent = _agent_with_preferred(200)
    for limit in (_DOC_MEDIUM_MAX_CHUNKS, _DOC_LARGE_MAX_CHUNKS):
        fetch_k = agent._docs_fetch_k(limit, preferred_n=limit * 3)
        assert fetch_k >= limit
        assert fetch_k <= _DOC_FETCH_POOL_CAP
        assert fetch_k > 8


def test_fewer_available_chunks_keeps_only_available() -> None:
    agent = _agent_with_preferred(10)  # small → max 8
    available = [_chunk("mod.py", i) for i in range(5)]
    agent.retriever.retrieve.return_value = available
    selected = agent._retrieve_context("document", "mod.py")
    assert len(selected) == 5
    assert agent.retriever.retrieve.call_args.kwargs["top_k"] >= 8


def test_retrieve_requests_fetch_k_not_global_eight() -> None:
    agent = _agent_with_preferred(_ANALYSIS_SMALL_PREFERRED_MAX + 10)  # medium → 16
    agent.retriever.retrieve.return_value = [
        _chunk(f"a{i}.py", i) for i in range(40)
    ]
    selected = agent._retrieve_context("readme overview", ".")
    assert len(selected) == 16
    assert agent.retriever.retrieve.call_args.kwargs["top_k"] >= 16
    assert agent.retriever.retrieve.call_args.kwargs["top_k"] > 8


def test_build_prompt_includes_exactly_kept_chunks() -> None:
    """Final LLM prompt must contain one RETRIEVED CONTEXT entry per kept chunk."""
    agent = DocumentationAgent(retriever=MagicMock())
    chunks = [_chunk(f"m{i}.py", i) for i in range(16)]
    prompt = agent._build_prompt(
        mode="readme",
        instruction="Write a README",
        target_path=".",
        function_name="README",
        chunks=chunks,
        source_excerpts=["### a.py\npass\n"],
        inventory=["a.py", "b.py"],
        project_files=[],
    )
    markers = re.findall(r"^\[\d+\] source=", prompt, flags=re.M)
    assert len(markers) == 16
    assert "RETRIEVED CONTEXT" in prompt
    assert "REPOSITORY INVENTORY" in prompt
    assert "REPOSITORY CONTENTS" in prompt
    # Character safeguard still applies.
    assert _MAX_CHUNK_CHARS == 1200


def test_analysis_and_testing_limits_unchanged() -> None:
    assert _ANALYSIS_CONTEXT_LIMITS == {"small": 12, "medium": 20, "large": 32}
    assert TESTING_MAX_CHUNKS == 5


def test_large_retrieve_caps_at_24_in_prompt_path() -> None:
    agent = _agent_with_preferred(_ANALYSIS_MEDIUM_PREFERRED_MAX + 20)
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
        inventory=["x0.py"],
        project_files=[],
    )
    assert len(re.findall(r"^\[\d+\] source=", prompt, flags=re.M)) == 24
