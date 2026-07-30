"""
indexer.py
==========

High-level façade over the indexing pipeline.

The Ingestor already knows how to walk a repository and push files
through chunk -> embed -> store. What it does not know is which files
have *changed*, so calling it twice re-embeds a repository that may be
byte-for-byte identical to the last run. Embedding is the expensive step
by a wide margin, so this class exists to avoid paying for it twice.

It does that with a manifest: a small JSON file recording a content hash
per indexed file, written next to the vector store. An update compares
current hashes against the manifest and hands the Ingestor only what is
new, changed, or gone. Everything else -- reading, chunking, embedding,
storing, skipping -- stays where it already lives.

This is the entry point agents and the notebook should use. The
Supervisor already constructs one.

TODO: The manifest is keyed by content hash, so a file that is moved
without being edited is treated as a delete plus an add. Detecting
renames would avoid re-embedding identical content.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config
from ..exceptions.base import CodebaseAssistantError
from ..tools.filesystem_tools import FilesystemTools
from .ingest import FileOutcome, IngestionResult, Ingestor, ProgressCallback
from .vectordb import VectorDB

logger = logging.getLogger(__name__)

#: Filename of the manifest, stored inside the vector store directory so
#: the index and its bookkeeping travel together.
MANIFEST_FILENAME = "index_manifest.json"

#: Manifest schema version. Bumped when the format changes, so an old
#: manifest is discarded rather than misread.
MANIFEST_VERSION = 1


@dataclass
class IndexUpdate:
    """
    Outcome of an incremental index update.

    Attributes:
        added: Files indexed for the first time.
        modified: Files whose content changed and were re-indexed.
        removed: Files dropped from the index because they are gone.
        unchanged: Files skipped because their content hash matched.
        ingestion: Aggregated Ingestor result across every file that was
            actually processed, carrying the chunk counts and per-file
            skip and failure reasons.
        limit_reached: The Config ceiling that stopped the update early,
            or None if it ran to completion.
        duration_seconds: Wall-clock duration of the update.
    """

    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    ingestion: IngestionResult = field(default_factory=IngestionResult)
    limit_reached: Optional[str] = None
    duration_seconds: float = 0.0

    @property
    def changed(self) -> bool:
        """True if the update altered the index in any way."""
        return bool(self.added or self.modified or self.removed)

    def summary(self) -> str:
        """
        Render a one-line summary of the update.

        Returns:
            A readable summary suitable for logs or a notebook cell.
        """
        parts = [
            f"{len(self.added)} added",
            f"{len(self.modified)} modified",
            f"{len(self.removed)} removed",
            f"{len(self.unchanged)} unchanged",
            f"{self.ingestion.chunks_indexed} chunk(s) written",
        ]
        if self.ingestion.skipped:
            parts.append(f"{self.ingestion.files_skipped} skipped")
        if self.ingestion.failed:
            parts.append(f"{self.ingestion.files_failed} failed")
        if self.limit_reached:
            parts.append(f"stopped at limit: {self.limit_reached}")
        return ", ".join(parts) + f" in {self.duration_seconds:.1f}s"


class Indexer:
    """
    Builds and maintains the vector index used for retrieval.

    Owns no pipeline logic of its own. Chunking, embedding, storage, and
    per-file error handling all belong to the Ingestor and its
    collaborators; what lives here is change detection and the public
    surface agents call.
    """

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        config: Optional[Config] = None,
        workspace_root: Optional[str] = None,
        ingestor: Optional[Ingestor] = None,
        vector_db: Optional[VectorDB] = None,
        manifest_path: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        """
        Initialize the Indexer.

        Nothing is loaded here. The Supervisor constructs an Indexer
        during its own startup, so this must stay free of model loads
        and vector store connections; both are deferred until an
        indexing call actually needs them.

        Args:
            vector_store_path: Directory the vector store persists to.
                Kept as the leading parameter because the Supervisor
                already passes it by name. Falls back to
                `Config.chroma_persist_directory`.
            config: Optional Config instance. A default is loaded when
                not supplied.
            workspace_root: Repository root to index. Falls back to
                `Config.workspace_root`.
            ingestor: Optional Ingestor. Built on first use when
                omitted.
            vector_db: Optional VectorDB. Shared with the Ingestor so
                both read and write the same collection.
            manifest_path: Override for the manifest location. Defaults
                to `index_manifest.json` inside the vector store
                directory.
            progress: Optional callback receiving progress messages,
                forwarded to the Ingestor.
        """
        self.config = config or Config.load()
        self.vector_store_path = (
            vector_store_path or self.config.chroma_persist_directory
        )
        self.workspace_root = workspace_root or self.config.workspace_root

        self._vector_db = vector_db
        self._ingestor = ingestor
        self._progress = progress
        self._manifest_path = Path(
            manifest_path or Path(self.vector_store_path) / MANIFEST_FILENAME
        ).expanduser()

    # ------------------------------------------------------------------
    # Lazily built collaborators
    # ------------------------------------------------------------------

    @property
    def vector_db(self) -> VectorDB:
        """
        The vector store, built on first use.

        Returns:
            The VectorDB pointed at the configured collection.
        """
        if self._vector_db is None:
            self._vector_db = VectorDB(
                config=self.config, persist_directory=self.vector_store_path
            )
        return self._vector_db

    @property
    def ingestor(self) -> Ingestor:
        """
        The ingestion pipeline, built on first use.

        Returns:
            An Ingestor sharing this Indexer's config, workspace root,
            and vector store.
        """
        if self._ingestor is None:
            self._ingestor = Ingestor(
                config=self.config,
                workspace_root=self.workspace_root,
                vector_db=self.vector_db,
                progress=self._progress,
            )
        return self._ingestor

    @property
    def filesystem(self) -> FilesystemTools:
        """
        The workspace filesystem, shared with the Ingestor.

        Returns:
            The FilesystemTools scoped to the workspace root.
        """
        return self.ingestor.filesystem

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build_index(
        self, directory: str = ".", rebuild: bool = False
    ) -> IngestionResult:
        """
        Index a repository in full.

        Delegates the whole walk to `Ingestor.ingest_repository`, which
        is where the `Config` ceilings on file count and total lines are
        enforced across the run. The manifest is refreshed afterwards so
        the next `update_index` can work incrementally.

        Args:
            directory: Directory to index, relative to the workspace
                root.
            rebuild: When True, clear the index first for a clean
                rebuild.

        Returns:
            The Ingestor's run summary.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the
                workspace.
            ToolExecutionError: If the directory does not exist.
            ProviderUnavailableError: If the embedding model or vector
                store is unusable.
        """
        result = self.ingestor.ingest_repository(directory, clear=rebuild)
        self._refresh_manifest(directory)
        return result

    def rebuild_index(self, directory: str = ".") -> IngestionResult:
        """
        Discard the index and build it again from scratch.

        The escape hatch for when the index is suspect -- a changed
        embedding model, a corrupted store, a manifest that has drifted
        out of step with reality.

        Args:
            directory: Directory to index, relative to the workspace
                root.

        Returns:
            The Ingestor's run summary.

        Raises:
            ProviderUnavailableError: If the embedding model or vector
                store is unusable.
        """
        return self.build_index(directory, rebuild=True)

    def index_directory(self, directory: str = ".") -> int:
        """
        Index a directory and report how many files were indexed.

        Retained from the scaffold's original interface. The `patterns`
        argument it used to take is gone: which files are in scope is
        decided by `Chunker.supports` and `Config.ignore_directories`,
        and a second, overlapping filter here would be a way for the
        index to disagree with the rest of the pipeline.

        Args:
            directory: Directory to index, relative to the workspace
                root.

        Returns:
            The number of files indexed.

        Raises:
            ProviderUnavailableError: If the embedding model or vector
                store is unusable.
        """
        return self.build_index(directory).files_indexed

    def index_file(self, path: str) -> bool:
        """
        Index a single file, replacing anything already stored for it.

        Retained from the scaffold's original interface, minus the
        `content` argument. Content is read from disk by
        FilesystemTools, which is also what enforces the sandbox and the
        size ceiling; accepting caller-supplied text would bypass both
        and could store chunks whose line numbers do not match the file
        on disk, breaking grounding.

        Args:
            path: File to index, relative to the workspace root.

        Returns:
            True if the file was indexed, False if it was skipped or
            failed. Reasons are available via `index_file_detailed`.

        Raises:
            ValueError: If `path` is empty.
            ProviderUnavailableError: If the embedding model or vector
                store is unusable.
        """
        return self.index_file_detailed(path).files_indexed == 1

    def index_file_detailed(self, path: str) -> IngestionResult:
        """
        Index a single file and return the full outcome.

        The reporting counterpart to `index_file`, for callers that need
        to know *why* a file was skipped rather than just that it was.

        Args:
            path: File to index, relative to the workspace root.

        Returns:
            The Ingestor's run summary for that file.

        Raises:
            ValueError: If `path` is empty.
            ProviderUnavailableError: If the embedding model or vector
                store is unusable.
        """
        if not path or not str(path).strip():
            raise ValueError("path must be a non-empty string.")

        result = self.ingestor.ingest_file(path, replace=True)

        manifest = self._load_manifest()
        if result.files_indexed:
            fingerprint = self._fingerprint(path)
            if fingerprint is not None:
                manifest["files"][path] = fingerprint
        else:
            manifest["files"].pop(path, None)
        self._save_manifest(manifest)

        return result

    def reindex_file(self, path: str) -> bool:
        """
        Re-index a file that has changed on disk.

        Args:
            path: File to re-index, relative to the workspace root.

        Returns:
            True if the file was indexed.

        Raises:
            ValueError: If `path` is empty.
            ProviderUnavailableError: If the embedding model or vector
                store is unusable.
        """
        return self.index_file(path)

    # ------------------------------------------------------------------
    # Incremental updating
    # ------------------------------------------------------------------

    def update_index(self, directory: str = ".") -> IndexUpdate:
        """
        Bring the index up to date, touching only what changed.

        Compares a content hash of every in-scope file against the
        manifest, then re-indexes the new and modified ones and drops
        the deleted ones. Files whose hash is unchanged are left alone,
        which is the point: embedding dominates the cost of indexing,
        and re-embedding identical content buys nothing.

        Hashing does re-read each file, but reading is cheap next to a
        forward pass through the embedding model, so the trade is
        heavily in favor of hashing.

        Args:
            directory: Directory to update, relative to the workspace
                root.

        Returns:
            What changed, and the aggregated ingestion outcome.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the
                workspace.
            ToolExecutionError: If the directory does not exist.
            ProviderUnavailableError: If the embedding model or vector
                store is unusable.
        """
        started = time.time()
        update = IndexUpdate()

        manifest = self._load_manifest()
        previous: Dict[str, str] = dict(manifest.get("files") or {})
        current, unreadable = self._scan(directory)

        for file_path, fingerprint in current.items():
            if file_path not in previous:
                update.added.append(file_path)
            elif previous[file_path] != fingerprint:
                update.modified.append(file_path)
            else:
                update.unchanged.append(file_path)

        update.removed = sorted(set(previous) - set(current))

        self._report(
            f"Update scan: {len(update.added)} new, "
            f"{len(update.modified)} modified, {len(update.removed)} removed, "
            f"{len(update.unchanged)} unchanged."
        )

        for file_path in update.removed:
            update.ingestion.chunks_removed += self.ingestor.remove_file(file_path)
            previous.pop(file_path, None)
        update.ingestion.files_pruned = len(update.removed)

        pending = update.added + update.modified
        for position, file_path in enumerate(pending, start=1):
            limit = self._limit_reached(update, previous)
            if limit is not None:
                update.limit_reached = limit
                self._defer(update, pending[position - 1 :], limit)
                break

            self._report(f"[{position}/{len(pending)}] {file_path}")
            outcome = self.ingestor.ingest_file(file_path, replace=True)
            self._merge(update.ingestion, outcome)

            if outcome.files_indexed:
                previous[file_path] = current[file_path]
            else:
                previous.pop(file_path, None)

        # Unreadable files stay out of the manifest so a later run
        # retries them instead of treating them as up to date.
        for file_path, reason in unreadable:
            previous.pop(file_path, None)
            update.ingestion.skipped.append(FileOutcome(file_path, reason))

        manifest["files"] = previous
        self._save_manifest(manifest)

        update.duration_seconds = time.time() - started
        self._report(update.summary())
        return update

    # ------------------------------------------------------------------
    # Removing and clearing
    # ------------------------------------------------------------------

    def remove_file(self, path: str) -> int:
        """
        Drop a single file from the index.

        Args:
            path: File whose chunks should be removed, as recorded in
                the index.

        Returns:
            The number of chunks removed.

        Raises:
            ValueError: If `path` is empty.
            ProviderUnavailableError: If the delete fails.
        """
        removed = self.ingestor.remove_file(path)

        manifest = self._load_manifest()
        if manifest["files"].pop(path, None) is not None:
            self._save_manifest(manifest)

        return removed

    def remove_missing_files(self) -> List[str]:
        """
        Drop indexed files that no longer exist on disk.

        Keeps the index honest after a deletion or a branch switch. A
        chunk from a file that is gone still reads as real code, and
        would be cited as evidence for a bug that cannot exist.

        Args:
            None.

        Returns:
            The paths removed from the index.

        Raises:
            ProviderUnavailableError: If the index cannot be read or
                written.
        """
        removed = self.ingestor.remove_missing_files()
        if not removed:
            return []

        manifest = self._load_manifest()
        for file_path in removed:
            manifest["files"].pop(file_path, None)
        self._save_manifest(manifest)

        return removed

    def clear_index(self) -> bool:
        """
        Delete every chunk and forget all bookkeeping.

        Args:
            None.

        Returns:
            True once the index is empty.

        Raises:
            ProviderUnavailableError: If the collection cannot be
                cleared.
        """
        removed = self.vector_db.delete_all()
        self._save_manifest(self._empty_manifest())
        self._report(f"Index cleared ({removed} chunk(s) removed).")
        return True

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def is_indexed(self) -> bool:
        """
        Report whether the index holds anything.

        Lets the Retriever fail with "the repository has not been
        indexed" instead of returning an empty result set that looks
        like "nothing matched".

        Args:
            None.

        Returns:
            True if at least one chunk is stored.

        Raises:
            ProviderUnavailableError: If the index cannot be read.
        """
        return self.vector_db.count() > 0

    def indexed_files(self) -> List[str]:
        """
        List the files currently represented in the index.

        Args:
            None.

        Returns:
            Workspace-relative file paths.

        Raises:
            ProviderUnavailableError: If the index cannot be read.
        """
        return self.ingestor.indexed_files()

    def stats(self) -> Dict[str, Any]:
        """
        Summarize the index and its bookkeeping.

        Args:
            None.

        Returns:
            The VectorDB statistics, plus the workspace root, manifest
            location, tracked-file count, and last-update timestamp.

        Raises:
            ProviderUnavailableError: If the index cannot be read.
        """
        manifest = self._load_manifest()
        summary = dict(self.vector_db.stats())
        summary.update(
            {
                "workspace_root": self.workspace_root,
                "manifest_path": str(self._manifest_path),
                "tracked_files": len(manifest.get("files") or {}),
                "last_updated": manifest.get("updated_at"),
            }
        )
        return summary

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def _scan(
        self, directory: str
    ) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
        """
        Fingerprint every in-scope file under a directory.

        Scope is decided entirely by existing components: the walk skips
        `Config.ignore_directories`, and `Chunker.supports` decides
        which extensions count.

        Args:
            directory: Directory to scan, relative to the workspace
                root.

        Returns:
            A mapping of file path to content hash, and a list of
            (path, reason) pairs for files that could not be read.

        Raises:
            PathOutsideWorkspaceError: If the path escapes the
                workspace.
            ToolExecutionError: If the directory does not exist.
        """
        fingerprints: Dict[str, str] = {}
        unreadable: List[Tuple[str, str]] = []

        for file_path in self.filesystem.list_files(directory, recursive=True):
            if not self.ingestor.chunker.supports(file_path):
                continue

            fingerprint = self._fingerprint(file_path)
            if fingerprint is None:
                unreadable.append((file_path, "unreadable or over the size limit"))
            else:
                fingerprints[file_path] = fingerprint

        return fingerprints, unreadable

    def _fingerprint(self, file_path: str) -> Optional[str]:
        """
        Hash a file's contents.

        Args:
            file_path: File to hash, relative to the workspace root.

        Returns:
            A hex SHA-256 digest, or None if the file could not be read.
            Oversized and binary files land here, and returning None
            rather than raising lets the caller record them as skips
            without special-casing each reason.
        """
        try:
            content = self.filesystem.read_file(file_path)
        except CodebaseAssistantError as exc:
            logger.info("Cannot fingerprint %s: %s", file_path, exc)
            return None

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _limit_reached(
        self, update: IndexUpdate, tracked: Dict[str, str]
    ) -> Optional[str]:
        """
        Check an incremental run against the Config ceilings.

        The Ingestor applies these across a full repository walk, but an
        incremental update calls it file by file, so the totals have to
        be tracked here. They count the whole index, not just this run:
        the ceiling is on how large the index may grow, not on how much
        one update may add.

        Args:
            update: The update so far.
            tracked: Files currently recorded in the manifest.

        Returns:
            A description of the ceiling that has been hit, or None if
            there is room to continue.
        """
        if len(tracked) >= self.config.max_repository_files:
            return (
                f"reached max_repository_files "
                f"({self.config.max_repository_files})"
            )
        if update.ingestion.lines_indexed >= self.config.max_total_lines_of_code:
            return (
                f"reached max_total_lines_of_code "
                f"({self.config.max_total_lines_of_code})"
            )
        return None

    def _defer(
        self, update: IndexUpdate, pending: List[str], reason: str
    ) -> None:
        """
        Record files left unprocessed because a ceiling was hit.

        Recorded rather than dropped so a truncated index is visibly
        truncated instead of quietly incomplete.

        Args:
            update: Update to modify in place.
            pending: Files that will not be indexed.
            reason: The ceiling that stopped the run.
        """
        for file_path in pending:
            update.ingestion.skipped.append(
                FileOutcome(file_path, f"limit: {reason}")
            )
            if file_path in update.added:
                update.added.remove(file_path)
            if file_path in update.modified:
                update.modified.remove(file_path)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _empty_manifest(self) -> Dict[str, Any]:
        """
        Build a fresh, empty manifest.

        Returns:
            A manifest with no tracked files.
        """
        return {
            "version": MANIFEST_VERSION,
            "workspace_root": self.workspace_root,
            "collection_name": self.config.chroma_collection_name,
            "embedding_model": self.config.embedding_model_name,
            "updated_at": None,
            "files": {},
        }

    def _load_manifest(self) -> Dict[str, Any]:
        """
        Read the manifest, tolerating absence and corruption.

        A manifest that cannot be read is replaced with an empty one
        rather than raising. The worst case is a full re-index, which is
        slow but correct; refusing to index at all because a cache file
        is malformed would not be.

        A manifest written by a different embedding model is also
        discarded, since vectors from one model are not comparable with
        another's and the stored ones have to be regenerated.

        Returns:
            The manifest, guaranteed to have a `files` mapping.
        """
        if not self._manifest_path.is_file():
            return self._empty_manifest()

        try:
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not read the index manifest at %s (%s); treating the "
                "index as unknown.",
                self._manifest_path,
                exc,
            )
            return self._empty_manifest()

        if not isinstance(manifest, dict):
            return self._empty_manifest()
        if manifest.get("version") != MANIFEST_VERSION:
            logger.info("Manifest version mismatch; starting a new manifest.")
            return self._empty_manifest()
        if manifest.get("embedding_model") != self.config.embedding_model_name:
            logger.info(
                "Manifest was written with a different embedding model; "
                "starting a new manifest."
            )
            return self._empty_manifest()

        files = manifest.get("files")
        manifest["files"] = files if isinstance(files, dict) else {}
        return manifest

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """
        Write the manifest to disk.

        A write failure is logged, not raised: the chunks are already
        stored, so the index is correct. Only incrementality is lost,
        and the next run will rebuild rather than fail.

        Args:
            manifest: The manifest to persist.
        """
        manifest["version"] = MANIFEST_VERSION
        manifest["workspace_root"] = self.workspace_root
        manifest["collection_name"] = self.config.chroma_collection_name
        manifest["embedding_model"] = self.config.embedding_model_name
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        try:
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self._manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(
                "Could not write the index manifest to %s (%s); the next "
                "update will re-index everything.",
                self._manifest_path,
                exc,
            )

    def _refresh_manifest(self, directory: str) -> None:
        """
        Rebuild the manifest from what is currently on disk.

        Called after a full build, where the Ingestor did the walk and
        this class never saw the individual files. Only files that
        actually made it into the index are recorded, so a skipped file
        is retried on the next update rather than being mistaken for
        up-to-date.

        Args:
            directory: Directory that was indexed.
        """
        try:
            indexed = set(self.indexed_files())
        except CodebaseAssistantError as exc:
            logger.warning("Could not refresh the manifest: %s", exc)
            return

        fingerprints, _ = self._scan(directory)
        manifest = self._empty_manifest()
        manifest["files"] = {
            path: digest
            for path, digest in fingerprints.items()
            if path in indexed
        }
        self._save_manifest(manifest)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(into: IngestionResult, other: IngestionResult) -> None:
        """
        Fold one ingestion result into a running total.

        Args:
            into: The accumulator, modified in place.
            other: The result to absorb.
        """
        into.files_indexed += other.files_indexed
        into.chunks_indexed += other.chunks_indexed
        into.lines_indexed += other.lines_indexed
        into.chunks_removed += other.chunks_removed
        into.skipped.extend(other.skipped)
        into.failed.extend(other.failed)

    def _report(self, message: str) -> None:
        """
        Emit a progress message.

        Args:
            message: The message to surface.
        """
        logger.info(message)
        if self._progress is not None:
            self._progress(message)
