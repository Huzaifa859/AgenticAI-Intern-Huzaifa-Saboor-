"""
memory_store.py
================

Defines MemoryStore, a persistent long-term memory store for facts,
learned preferences, conversation snapshots, and past analysis results
that should survive across sessions.

Records are stored as one JSON file per key under
``.codebase_assistant/memory_store/`` (or a configured path). Missing
directories are created automatically. Corrupted files are logged,
removed, and treated as a miss so a fresh record can be written later.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas.schemas import MemoryRecord

logger = logging.getLogger(__name__)

#: Characters allowed in on-disk filenames derived from record keys.
_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")


class MemoryStore:
    """
    Persistent, long-term memory store backed by JSON files.

    Intended to hold durable knowledge (e.g. conversation snapshots,
    "this repo uses Poetry") that agents and ConversationMemory can
    consult across multiple sessions.
    """

    def __init__(self, storage_path: str = "./.codebase_assistant/memory_store") -> None:
        """
        Initialize the MemoryStore and ensure its directory exists.

        Args:
            storage_path: Filesystem path where memory records are
                persisted.
        """
        self.storage_path = storage_path
        self._ensure_storage()

    def save(self, record: MemoryRecord) -> bool:
        """
        Persist a memory record as JSON.

        Args:
            record: The MemoryRecord to save.

        Returns:
            True if the save succeeded, False on I/O or serialization
            failure (errors are logged, never raised).
        """
        if not record or not str(record.key or "").strip():
            logger.warning("MemoryStore.save: refusing to persist a record without a key.")
            return False

        try:
            self._ensure_storage()
            path = self._path_for(record.key)
            payload = {
                "key": record.key,
                "value": record.value,
                "metadata": dict(record.metadata or {}),
            }
            self._atomic_write(path, payload)
            return True
        except Exception as exc:
            logger.warning(
                "MemoryStore.save failed for key %r: %s", record.key, exc
            )
            return False

    def load(self, key: str) -> Optional[MemoryRecord]:
        """
        Load a single memory record by key.

        Corrupted JSON is removed and treated as a miss so the next
        save can create a clean file.

        Args:
            key: Key identifying the memory record.

        Returns:
            The MemoryRecord if found and valid, else None.
        """
        if not key or not str(key).strip():
            return None

        path = self._path_for(key)
        if not path.is_file():
            return None

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "MemoryStore: corrupted record at %s (%s); recreating store entry.",
                path,
                exc,
            )
            self._recreate_corrupt(path)
            return None

        if not isinstance(data, dict):
            logger.warning(
                "MemoryStore: invalid record shape at %s; recreating store entry.",
                path,
            )
            self._recreate_corrupt(path)
            return None

        try:
            return MemoryRecord(
                key=str(data.get("key") or key),
                value=data.get("value"),
                metadata=dict(data.get("metadata") or {}),
            )
        except Exception as exc:
            logger.warning(
                "MemoryStore: could not build MemoryRecord from %s (%s); "
                "recreating store entry.",
                path,
                exc,
            )
            self._recreate_corrupt(path)
            return None

    def search(self, query: str, top_k: int = 5) -> List[MemoryRecord]:
        """
        Search memory records relevant to a query.

        Performs a simple case-insensitive substring match over the
        serialized record. Good enough for small JSON stores; not a
        semantic index.

        Args:
            query: Natural language query string.
            top_k: Maximum number of records to return.

        Returns:
            Matching MemoryRecord objects, up to ``top_k``.
        """
        needle = (query or "").strip().lower()
        if not needle or top_k <= 0:
            return []

        matches: List[MemoryRecord] = []
        for path in self._iter_record_files():
            record = self.load(path.stem)
            if record is None:
                continue
            haystack = json.dumps(
                {"key": record.key, "value": record.value, "metadata": record.metadata},
                default=str,
            ).lower()
            if needle in haystack:
                matches.append(record)
            if len(matches) >= top_k:
                break
        return matches

    def delete(self, key: str) -> bool:
        """
        Delete a memory record by key.

        Args:
            key: Key identifying the memory record.

        Returns:
            True if the file was removed or already absent, False on
            unexpected I/O failure.
        """
        if not key or not str(key).strip():
            return False
        path = self._path_for(key)
        try:
            if path.is_file():
                path.unlink()
            return True
        except OSError as exc:
            logger.warning("MemoryStore.delete failed for key %r: %s", key, exc)
            return False

    def clear(self) -> bool:
        """
        Clear all stored memory records.

        Returns:
            True if every record file was removed (or none existed),
            False if any deletion failed.
        """
        ok = True
        for path in self._iter_record_files():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("MemoryStore.clear could not remove %s: %s", path, exc)
                ok = False
        return ok

    def list_keys(self) -> List[str]:
        """
        List the keys of every persisted record.

        Returns:
            Sorted list of record keys (filename stems).
        """
        return sorted(path.stem for path in self._iter_record_files())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_storage(self) -> None:
        """Create the storage directory when it is missing."""
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        """
        Map a record key to a JSON file path under the store.

        Args:
            key: Logical record key.

        Returns:
            Absolute path to the JSON file for that key.
        """
        safe = _SAFE_KEY.sub("_", str(key).strip()).strip("._") or "record"
        return Path(self.storage_path).joinpath(f"{safe}.json").resolve()

    def _iter_record_files(self) -> List[Path]:
        """Return every ``*.json`` file currently in the store."""
        root = Path(self.storage_path)
        if not root.is_dir():
            return []
        return sorted(path for path in root.glob("*.json") if path.is_file())

    def _recreate_corrupt(self, path: Path) -> None:
        """
        Remove a corrupted record and ensure the store directory exists.

        Args:
            path: Path of the corrupted JSON file.
        """
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            logger.warning("MemoryStore: could not remove corrupt file %s: %s", path, exc)
        self._ensure_storage()

    @staticmethod
    def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
        """
        Write JSON atomically via a temporary file in the same directory.

        Args:
            path: Destination JSON path.
            payload: JSON-serializable document.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
                handle.write("\n")
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
