"""
test_ingest_batching.py
========================

Unit tests for the cross-file embedding batching in
`Ingestor.ingest_repository`.

Uses a real Chunker/FilesystemTools (fast, no ML) with fake embedder
and vector_db collaborators, so these tests run without loading a real
sentence-transformers model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from codebase_assistant.config import Config
from codebase_assistant.rag.chunker import Chunker
from codebase_assistant.rag.ingest import (
    _MAX_FILES_PER_EMBEDDING_BATCH,
    Ingestor,
)
from codebase_assistant.schemas.schemas import CodeChunk
from codebase_assistant.tools.filesystem_tools import FilesystemTools


class _FakeEmbedder:
    """Records how many chunks each embed_chunks() call received."""

    def __init__(self, batch_size: int = 32, fail_first_n: int = 0) -> None:
        self.batch_size = batch_size
        self.calls: List[int] = []
        self._remaining_failures = fail_first_n

    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> List[List[float]]:
        self.calls.append(len(chunks))
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("simulated embedding failure")
        return [[0.0, 0.0] for _ in chunks]


class _FakeVectorDB:
    """Records how many chunks each add_chunks() call stored."""

    def __init__(self) -> None:
        self.added: List[int] = []
        self.deleted: List[Dict[str, Any]] = []

    def add_chunks(
        self,
        chunks: Sequence[CodeChunk],
        embeddings: Sequence[Sequence[float]],
        batch_size: Any = None,
    ) -> int:
        self.added.append(len(chunks))
        return len(chunks)

    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        self.deleted.append(where)
        return 0

    def delete_all(self) -> int:
        return 0


def _make_repo(tmp_path: Path, n_files: int) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(n_files):
        (repo / f"mod_{i}.py").write_text(
            f"def fn_{i}(a, b):\n    return a + b\n", encoding="utf-8"
        )
    return repo


def _make_ingestor(repo: Path, embedder: _FakeEmbedder, vector_db: _FakeVectorDB) -> Ingestor:
    config = Config(workspace_root=str(repo))
    fs = FilesystemTools(workspace_root=str(repo), config=config)
    chunker = Chunker(config=config, filesystem=fs)
    return Ingestor(
        config=config,
        workspace_root=str(repo),
        filesystem=fs,
        chunker=chunker,
        embedder=embedder,
        vector_db=vector_db,
    )


def test_ingest_repository_batches_embedding_across_files(tmp_path: Path) -> None:
    """Several small files should be embedded in one call, not one per file."""
    repo = _make_repo(tmp_path, n_files=5)
    embedder = _FakeEmbedder(batch_size=32)
    vector_db = _FakeVectorDB()
    ingestor = _make_ingestor(repo, embedder, vector_db)

    result = ingestor.ingest_repository(".", clear=False, prune=False)

    assert result.files_indexed == 5
    assert result.failed == []
    # 5 one-chunk files fit comfortably under batch_size=32, so they
    # should collapse into a single embed_chunks() call.
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == result.chunks_indexed
    assert len(vector_db.added) == 1


def test_ingest_repository_flushes_after_max_files_per_batch(tmp_path: Path) -> None:
    """A file-count cap should force a flush even under the chunk-size cap."""
    n_files = _MAX_FILES_PER_EMBEDDING_BATCH + 5
    repo = _make_repo(tmp_path, n_files=n_files)
    embedder = _FakeEmbedder(batch_size=1000)  # chunk cap never trips
    vector_db = _FakeVectorDB()
    ingestor = _make_ingestor(repo, embedder, vector_db)

    result = ingestor.ingest_repository(".", clear=False, prune=False)

    assert result.files_indexed == n_files
    assert result.failed == []
    # Must flush at least twice since n_files > _MAX_FILES_PER_EMBEDDING_BATCH.
    assert len(embedder.calls) >= 2
    assert sum(embedder.calls) == result.chunks_indexed


def test_ingest_repository_falls_back_per_file_when_batch_embedding_fails(
    tmp_path: Path,
) -> None:
    """A failed batch call should retry per-file instead of losing the batch."""
    repo = _make_repo(tmp_path, n_files=3)
    embedder = _FakeEmbedder(batch_size=32, fail_first_n=1)
    vector_db = _FakeVectorDB()
    ingestor = _make_ingestor(repo, embedder, vector_db)

    result = ingestor.ingest_repository(".", clear=False, prune=False)

    # 1 failed whole-batch call, then 3 individual per-file retries.
    assert len(embedder.calls) == 4
    assert result.files_indexed == 3
    assert result.failed == []
