"""
test_rag.py
============

Placeholder tests for the RAG pipeline — ingestion, AST-aware chunking,
embedding, vector storage, and retrieval.

This is the other half of the Week 6 coverage target, and where the
proposal's retrieval-accuracy metric is measured.

TODO: Replace every skip below with real assertions as each pipeline
stage is implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: assert ingestion walks a repo and reports files indexed")
def test_ingestor_indexes_repository() -> None:
    """Ingesting a repo should index its Python sources and report counts."""


@pytest.mark.skip(reason="TODO: assert notebooks are skipped with an explicit message")
def test_ingestor_skips_notebooks_with_message() -> None:
    """Notebooks are out of scope but must not be dropped silently."""


@pytest.mark.skip(reason="TODO: assert binaries and ignored directories never reach the chunker")
def test_ingestor_skips_binaries_and_ignored_directories() -> None:
    """Binary files and .git/__pycache__/venv/node_modules are excluded."""


@pytest.mark.skip(reason="TODO: assert one chunk is produced per function and per class")
def test_chunker_produces_one_chunk_per_function() -> None:
    """AST-aware chunking should split by definition, not fixed size."""


@pytest.mark.skip(reason="TODO: assert chunks carry file path, class, function, line numbers, imports")
def test_chunker_attaches_required_metadata() -> None:
    """Chunk metadata must include the line numbers grounding depends on."""


@pytest.mark.skip(reason="TODO: assert a syntactically invalid file fails without aborting the run")
def test_chunker_handles_unparseable_file() -> None:
    """One bad file should not abort ingestion of the whole repo."""


@pytest.mark.skip(reason="TODO: assert embeddings have the expected dimensionality")
def test_embedding_generator_produces_vectors() -> None:
    """Embedding a chunk should return a vector of the model's width."""


@pytest.mark.skip(reason="TODO: assert EmbeddingError is raised and logged on failure")
def test_embedding_failure_is_handled() -> None:
    """Embedding failures must be caught rather than crash ingestion."""


@pytest.mark.skip(reason="TODO: assert retrieve returns at most top_k chunks, default 8")
def test_retriever_respects_top_k() -> None:
    """Retrieval width should default to the proposal's k=8."""


@pytest.mark.skip(reason="TODO: assert the correct function is surfaced for a known question")
def test_retriever_surfaces_relevant_function() -> None:
    """This is the proposal's retrieval-accuracy metric."""
