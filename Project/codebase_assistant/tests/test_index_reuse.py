"""
test_index_reuse.py
===================

Verify Analysis / Documentation / Testing share one per-repo Chroma store
and that a second update_index after a full index embeds nothing new.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from codebase_assistant.agents.code_analysis_agent import CodeAnalysisAgent
from codebase_assistant.agents.documentation_agent import DocumentationAgent
from codebase_assistant.agents.testing_agent import TestingAgent
from codebase_assistant.config import Config
from codebase_assistant.rag.indexer import Indexer
from codebase_assistant.rag.ingest import IngestionResult
from codebase_assistant.rag.retriever import Retriever
from codebase_assistant.rag.store_paths import vector_store_for_repository


def _tiny_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "demo"
    repo.mkdir()
    (repo / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    return repo


def test_vector_store_for_repository_is_stable_per_root(tmp_path: Path) -> None:
    root = str((tmp_path / "repo").resolve())
    os.makedirs(root, exist_ok=True)
    a = vector_store_for_repository(str(tmp_path / "chroma"), root)
    b = vector_store_for_repository(str(tmp_path / "chroma"), root + os.sep)
    assert a == b
    assert a.startswith(str((tmp_path / "chroma").resolve()) or str(tmp_path / "chroma"))
    other = vector_store_for_repository(str(tmp_path / "chroma"), str(tmp_path / "other"))
    assert a != other


def test_agents_share_analysis_store_path(tmp_path: Path) -> None:
    """Docs/Testing must target the same per-repo path Analysis uses."""
    repo = _tiny_repo(tmp_path)
    chroma = tmp_path / "chroma"
    config = Config(chroma_persist_directory=str(chroma))

    analysis = CodeAnalysisAgent(config=config)
    expected = analysis._store_for(str(repo.resolve()))

    retriever = Retriever(vector_store_path=str(chroma), config=config)
    docs = DocumentationAgent(retriever=retriever)
    testing = TestingAgent(retriever=retriever)

    with patch.object(Indexer, "update_index") as mock_update:
        mock_update.return_value = MagicMock(
            summary=lambda: "0 added, 0 modified, 0 removed, 1 unchanged, 0 chunk(s) written in 0.0s",
            added=[],
            modified=[],
            removed=[],
            unchanged=["math_utils.py"],
        )
        docs._ensure_index(str(repo.resolve()))
        testing._ensure_index(str(repo.resolve()))

    assert docs.retriever is not None
    assert testing.retriever is not None
    assert docs.retriever.vector_store_path == expected
    assert testing.retriever.vector_store_path == expected
    assert mock_update.call_count == 2


def test_second_update_skips_unchanged_files(tmp_path: Path) -> None:
    """After one index, a second update_index must report files unchanged."""
    repo = _tiny_repo(tmp_path)
    chroma = tmp_path / "chroma"
    config = Config(chroma_persist_directory=str(chroma))
    workspace = str(repo.resolve())
    store = vector_store_for_repository(str(chroma), workspace)

    indexer = Indexer(
        vector_store_path=store,
        config=config,
        workspace_root=workspace,
    )

    outcome = IngestionResult(files_indexed=1, chunks_indexed=1, lines_indexed=2)

    with patch.object(indexer.ingestor, "ingest_file", return_value=outcome) as ingest:
        first = indexer.update_index(".")
        assert first.added == ["math_utils.py"]
        assert first.unchanged == []
        assert ingest.call_count == 1

        second = indexer.update_index(".")
        assert second.added == []
        assert second.modified == []
        assert second.unchanged == ["math_utils.py"]
        assert ingest.call_count == 1  # no re-embed
