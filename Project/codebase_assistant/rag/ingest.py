"""
ingest.py
=========

Orchestrates the indexing pipeline: walk a repository, chunk what is
supported, embed it, and store it.

Every step is delegated. FilesystemTools decides what the workspace
contains and enforces the size ceiling, Chunker decides what a chunk is
and which extensions are in scope, EmbeddingGenerator turns chunks into
vectors, VectorDB persists them. This module contributes the sequencing,
the Scope & Limits accounting, and the error handling that keeps one bad
file from taking down a whole repository index.

The one distinction worth knowing: a failure that is *about a file* is
recorded and skipped, while a failure that means the pipeline itself is
unusable -- the embedding model will not load, the vector store cannot
be opened -- aborts immediately. Grinding through ninety more files to
collect ninety identical errors helps nobody.

Re-indexing deletes a file's existing chunks before writing the new
ones. Upserting alone would leave orphans behind, because a chunk ID
encodes a function name or a line range, and both change when the file
does.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ..config import Config
from ..exceptions.base import CodebaseAssistantError
from ..exceptions.model_exceptions import ProviderUnavailableError
from ..schemas.schemas import CodeChunk
from ..tools.filesystem_tools import FilesystemTools
from .chunker import Chunker
from .embeddings import EmbeddingGenerator
from .vectordb import VectorDB

logger = logging.getLogger(__name__)

#: Signature of a progress callback.
ProgressCallback = Callable[[str], None]


@dataclass
class FileOutcome:
    """
    Why a single file was skipped or failed.

    Attributes:
        file_path: The file in question, workspace-relative.
        reason: Human-readable explanation, surfaced in the notebook so
            a skipped file is visibly skipped rather than silently
            missing.
    """

    file_path: str
    reason: str


@dataclass
class IngestionResult:
    """
    Summary of one ingestion run.

    Attributes:
        files_indexed: Files successfully chunked, embedded, and stored.
        chunks_indexed: Chunks written across those files.
        lines_indexed: Source lines read from those files, counted
            against `Config.max_total_lines_of_code`.
        chunks_removed: Chunks deleted during the run, whether by
            replacing a re-indexed file or by pruning a deleted one.
        files_pruned: Files dropped from the index because they no
            longer exist on disk.
        skipped: Files deliberately not indexed, with reasons.
        failed: Files that raised, with reasons.
        limit_reached: The Scope & Limits ceiling that stopped the run
            early, or None if it ran to completion.
        duration_seconds: Wall-clock duration of the run.
    """

    files_indexed: int = 0
    chunks_indexed: int = 0
    lines_indexed: int = 0
    chunks_removed: int = 0
    files_pruned: int = 0
    skipped: List[FileOutcome] = field(default_factory=list)
    failed: List[FileOutcome] = field(default_factory=list)
    limit_reached: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def files_skipped(self) -> int:
        """Number of files deliberately not indexed."""
        return len(self.skipped)

    @property
    def files_failed(self) -> int:
        """Number of files that raised during indexing."""
        return len(self.failed)

    def summary(self) -> str:
        """
        Render a one-line summary of the run.

        Returns:
            A readable summary suitable for logs or a notebook cell.
        """
        parts = [
            f"{self.files_indexed} file(s) indexed",
            f"{self.chunks_indexed} chunk(s)",
            f"{self.lines_indexed} line(s)",
        ]
        if self.chunks_removed:
            parts.append(f"{self.chunks_removed} chunk(s) removed")
        if self.files_pruned:
            parts.append(f"{self.files_pruned} file(s) pruned")
        if self.skipped:
            parts.append(f"{self.files_skipped} skipped")
        if self.failed:
            parts.append(f"{self.files_failed} failed")
        if self.limit_reached:
            parts.append(f"stopped at limit: {self.limit_reached}")
        return ", ".join(parts) + f" in {self.duration_seconds:.1f}s"


class Ingestor:
    """
    Builds and maintains the vector index for a repository.

    Collaborators are injectable so a caller can share an already-loaded
    embedding model or point at a different vector store, and so tests
    can substitute fakes without touching disk.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        workspace_root: Optional[str] = None,
        filesystem: Optional[FilesystemTools] = None,
        chunker: Optional[Chunker] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        vector_db: Optional[VectorDB] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        """
        Initialize the Ingestor.

        Args:
            config: Optional Config instance. A default is loaded when
                not supplied.
            workspace_root: Repository root to index. Falls back to
                `Config.workspace_root`.
            filesystem: Optional FilesystemTools. Built from the config
                and workspace root when omitted.
            chunker: Optional Chunker. Built when omitted, sharing this
                Ingestor's FilesystemTools so both agree on what the
                workspace is.
            embedder: Optional EmbeddingGenerator. Built when omitted.
            vector_db: Optional VectorDB. Built when omitted.
            progress: Optional callback receiving progress messages.
                Defaults to logging at INFO.

        Raises:
            ToolExecutionError: If the workspace root does not exist.
        """
        self.config = config or Config.load()
        self.filesystem = filesystem or FilesystemTools(
            workspace_root=workspace_root, config=self.config
        )
        self.chunker = chunker or Chunker(
            config=self.config, filesystem=self.filesystem
        )
        self.embedder = embedder or EmbeddingGenerator(config=self.config)
        self.vector_db = vector_db or VectorDB(config=self.config)
        self._progress = progress

    # ------------------------------------------------------------------
    # Repository-level ingestion
    # ------------------------------------------------------------------

    def ingest(self, source_path: str = ".") -> IngestionResult:
        """
        Ingest a file or a directory.

        Convenience entry point that dispatches on what the path is.

        Args:
            source_path: File or directory to ingest, relative to the
                workspace root.

        Returns:
            The run summary.

        Raises:
            ValueError: If `source_path` is empty.
            PathOutsideWorkspaceError: If the path escapes the workspace.
            ToolExecutionError: If the path does not exist.
        """
        if self.filesystem.file_exists(source_path):
            return self.ingest_file(source_path)
        return self.ingest_repository(source_path)

    def ingest_repository(
        self,
        directory: str = ".",
        clear: bool = False,
        prune: bool = True,
    ) -> IngestionResult:
        """
        Index every supported file under a directory.

        Directories in `Config.ignore_directories` are never visited;
        that pruning happens inside the FilesystemTools walk rather than
        being re-implemented here.

        Args:
            directory: Directory to index, relative to the workspace
                root.
            clear: When True, empty the collection first, for a clean
                rebuild.
            prune: When True, drop chunks belonging to files that no
                longer exist on disk. Ignored when `clear` is set, since
                clearing already removes them.

        Returns:
            The run summary, including per-file skip and failure
            reasons.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the
                workspace.
            ToolExecutionError: If the directory does not exist.
            ProviderUnavailableError: If the embedding model or vector
                store is unusable. Raised rather than recorded, because
                it would fail identically for every remaining file.
        """
        started = time.time()
        result = IngestionResult()

        if clear:
            result.chunks_removed += self.vector_db.delete_all()
            self._report(
                f"Cleared the index ({result.chunks_removed} chunk(s) removed)."
            )

        candidates = self.filesystem.list_files(directory, recursive=True)
        self._report(f"Scanning {directory!r}: {len(candidates)} file(s) found.")

        supported = self._partition_supported(candidates, result)

        for position, file_path in enumerate(supported, start=1):
            limit = self._limit_reached(result)
            if limit is not None:
                result.limit_reached = limit
                remaining = supported[position - 1 :]
                for pending in remaining:
                    result.skipped.append(FileOutcome(pending, f"limit: {limit}"))
                self._report(f"Stopping early -- {limit}.")
                break

            self._report(f"[{position}/{len(supported)}] {file_path}")
            self._ingest_one(file_path, result, replace=not clear)

        if prune and not clear:
            self._prune_missing(result)

        result.duration_seconds = time.time() - started
        self._report(result.summary())
        return result

    def ingest_file(self, file_path: str, replace: bool = True) -> IngestionResult:
        """
        Index a single file.

        Args:
            file_path: File to index, relative to the workspace root.
            replace: When True, delete the file's existing chunks first.

        Returns:
            The run summary for that one file.

        Raises:
            ValueError: If `file_path` is empty.
            ProviderUnavailableError: If the embedding model or vector
                store is unusable.
        """
        started = time.time()
        result = IngestionResult()

        if not self.chunker.supports(file_path):
            result.skipped.append(
                FileOutcome(file_path, "unsupported file type")
            )
        else:
            self._ingest_one(file_path, result, replace=replace)

        result.duration_seconds = time.time() - started
        return result

    def reindex_file(self, file_path: str) -> IngestionResult:
        """
        Re-index a file that has changed on disk.

        The file's existing chunks are deleted before the new ones are
        written. An upsert on its own is not enough: a chunk ID encodes
        a function name or a line range, so renaming a function or
        adding a line above it produces new IDs and leaves the old
        records behind to surface in search results as code that no
        longer exists.

        Args:
            file_path: File to re-index, relative to the workspace root.

        Returns:
            The run summary for that one file.

        Raises:
            ValueError: If `file_path` is empty.
            ProviderUnavailableError: If the embedding model or vector
                store is unusable.
        """
        return self.ingest_file(file_path, replace=True)

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove_file(self, file_path: str) -> int:
        """
        Delete every chunk belonging to a file.

        Args:
            file_path: File whose chunks should be removed, as recorded
                in the index.

        Returns:
            The number of chunks removed.

        Raises:
            ValueError: If `file_path` is empty.
            ProviderUnavailableError: If the delete fails.
        """
        if not file_path or not str(file_path).strip():
            raise ValueError("file_path must be a non-empty string.")

        removed = self.vector_db.delete_by_metadata({"file_path": file_path})
        if removed:
            self._report(f"Removed {removed} chunk(s) for {file_path}.")
        return removed

    def remove_missing_files(self) -> List[str]:
        """
        Drop indexed files that no longer exist on disk.

        Keeps the index honest after a branch switch or a deletion: a
        retrieved chunk from a file that is gone reads as real code and
        would be cited as evidence for a bug that cannot exist.

        Args:
            None.

        Returns:
            The paths that were removed from the index.

        Raises:
            ProviderUnavailableError: If the index cannot be read or
                written.
        """
        missing = self._missing_indexed_files()
        for file_path in missing:
            self.remove_file(file_path)
        return missing

    def indexed_files(self) -> List[str]:
        """
        List the distinct files currently represented in the index.

        Args:
            None.

        Returns:
            Workspace-relative file paths.

        Raises:
            ProviderUnavailableError: If the index cannot be read.

        TODO: This reads every chunk's content to collect its path.
        Acceptable at the proposal's 100-file ceiling, but VectorDB
        should grow a metadata-only listing to make it cheap.
        """
        paths = {chunk.file_path for chunk in self.vector_db.list_chunks()}
        return sorted(paths)

    def stats(self) -> Dict[str, object]:
        """
        Report the current state of the index.

        Args:
            None.

        Returns:
            The VectorDB statistics for the configured collection.

        Raises:
            ProviderUnavailableError: If the index cannot be read.
        """
        return self.vector_db.stats()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ingest_one(
        self, file_path: str, result: IngestionResult, replace: bool
    ) -> None:
        """
        Run one file through read -> chunk -> embed -> store.

        Failures are caught and recorded on `result` rather than
        raised, so a single unreadable or malformed file cannot abort a
        repository-wide run. The exception is
        ProviderUnavailableError, which is re-raised: it means the model
        or the store is down, not that this file is bad.

        Args:
            file_path: File to ingest, relative to the workspace root.
            result: Run summary to update in place.
            replace: When True, delete the file's existing chunks first.

        Raises:
            ProviderUnavailableError: If the pipeline itself is
                unusable.
        """
        try:
            content = self.filesystem.read_file(file_path)

            if not content.strip():
                result.skipped.append(FileOutcome(file_path, "file is empty"))
                return

            chunks = self.chunker.chunk(content, file_path)
            if not chunks:
                result.skipped.append(
                    FileOutcome(file_path, "produced no chunks")
                )
                return

            if replace:
                result.chunks_removed += self.vector_db.delete_by_metadata(
                    {"file_path": file_path}
                )

            self._store(chunks)

        except ProviderUnavailableError:
            # Not this file's fault, and every remaining file would hit
            # the same wall.
            raise
        except CodebaseAssistantError as exc:
            # Oversized files, binaries, and unreadable paths land here;
            # FilesystemTools has already classified them.
            result.skipped.append(FileOutcome(file_path, str(exc)))
            logger.info("Skipped %s: %s", file_path, exc)
            return
        except Exception as exc:
            result.failed.append(
                FileOutcome(file_path, f"{type(exc).__name__}: {exc}")
            )
            logger.warning("Failed to index %s: %s", file_path, exc)
            return

        result.files_indexed += 1
        result.chunks_indexed += len(chunks)
        result.lines_indexed += content.count("\n") + 1

    def _store(self, chunks: Sequence[CodeChunk]) -> None:
        """
        Embed a file's chunks and write them to the vector store.

        Embedding is done per file so EmbeddingGenerator gets a whole
        batch at once, which is where its throughput comes from, while
        keeping a failure attributable to one file.

        Args:
            chunks: The file's chunks.

        Raises:
            ProviderUnavailableError: If the model or store is
                unusable.
            EmbeddingError: If encoding fails.
        """
        embeddings = self.embedder.embed_chunks(chunks)
        self.vector_db.add_chunks(chunks, embeddings)

    def _partition_supported(
        self, candidates: Sequence[str], result: IngestionResult
    ) -> List[str]:
        """
        Split a file list into what can be chunked and what cannot.

        Unsupported types are recorded as skips rather than dropped, so
        the notebook can show that a notebook or an image was seen and
        deliberately passed over.

        Args:
            candidates: Paths found by the filesystem walk.
            result: Run summary to update in place.

        Returns:
            The supported paths, in the order given.
        """
        supported: List[str] = []
        for file_path in candidates:
            if self.chunker.supports(file_path):
                supported.append(file_path)
            else:
                result.skipped.append(
                    FileOutcome(file_path, "unsupported file type")
                )
        return supported

    def _limit_reached(self, result: IngestionResult) -> Optional[str]:
        """
        Check the run against the proposal's Scope & Limits ceilings.

        Args:
            result: The run so far.

        Returns:
            A description of the ceiling that has been hit, or None if
            there is room to continue.
        """
        if result.files_indexed >= self.config.max_repository_files:
            return (
                f"reached max_repository_files "
                f"({self.config.max_repository_files})"
            )
        if result.lines_indexed >= self.config.max_total_lines_of_code:
            return (
                f"reached max_total_lines_of_code "
                f"({self.config.max_total_lines_of_code})"
            )
        return None

    def _missing_indexed_files(self) -> List[str]:
        """
        Find indexed files that are no longer readable on disk.

        Args:
            None.

        Returns:
            Workspace-relative paths present in the index but not on
            disk.

        Raises:
            ProviderUnavailableError: If the index cannot be read.
        """
        missing: List[str] = []
        for file_path in self.indexed_files():
            try:
                present = self.filesystem.file_exists(file_path)
            except CodebaseAssistantError:
                # A path that no longer resolves inside the workspace is
                # as gone as one that was deleted.
                present = False

            if not present:
                missing.append(file_path)
        return missing

    def _prune_missing(self, result: IngestionResult) -> None:
        """
        Remove indexed files that are no longer on disk.

        Shares `_missing_indexed_files` with the public
        `remove_missing_files` so the two cannot disagree about what
        counts as missing; this variant additionally records how many
        chunks went with each file.

        Args:
            result: Run summary to update in place.
        """
        missing = self._missing_indexed_files()
        if not missing:
            return

        for file_path in missing:
            result.chunks_removed += self.remove_file(file_path)
        result.files_pruned += len(missing)
        self._report(
            f"Pruned {len(missing)} file(s) no longer on disk: "
            f"{', '.join(missing)}"
        )

    def _report(self, message: str) -> None:
        """
        Emit a progress message.

        Args:
            message: The message to surface.
        """
        logger.info(message)
        if self._progress is not None:
            self._progress(message)
