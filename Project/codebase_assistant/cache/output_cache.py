"""
output_cache.py
================

Persistent, cross-run cache of LLM output for one repository.

Documentation, Testing, and Code Analysis each generate output for the
same public symbols / retrieval context every time they run against an
unchanged repository. Without a cache, an unchanged codebase pays for a
full round of model calls on every single invocation, even though the
result would be byte-for-byte the same as last time.

This mirrors the design of the RAG index manifest (`rag/indexer.py`): a
small JSON file next to the vector store, mapping a cache key to the
last generated result. The key is a hash of whatever the caller
considers "the input" (symbol source, instruction, model, prompt
version for docs/tests; retrieved-chunk + static-finding fingerprints
for analysis) -- so any change to that input invalidates the entry
automatically, without needing an explicit "is this stale?" check.

A cache hit skips the LLM call entirely. It does not skip anything
downstream: grounding, validation, and repair logic all still run
against the reused result exactly as they would against a fresh one.

Never raises. A cache that cannot be read or written degrades to
"always miss" / "write silently failed", which only costs the LLM call
this change exists to avoid -- never correctness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

#: Bumped when the on-disk shape of a cache file changes incompatibly.
CACHE_VERSION = 1

#: Separator used to join cache-key parts before hashing. Chosen to be
#: extremely unlikely to appear inside any part (source code, prompts,
#: model names), so two different part-tuples cannot collide onto the
#: same joined string.
_KEY_SEPARATOR = "\x1f"


class OutputCache:
    """
    A persistent key -> JSON-serializable-value cache for one
    repository and one call site (e.g. "documentation", "testing",
    "analysis"), backed by a single JSON file.

    Thread-safe: Documentation and Testing generate per-symbol output
    concurrently via ThreadPoolExecutor, and every `get`/`set` call may
    come from a different worker thread.
    """

    def __init__(self, cache_path: str) -> None:
        """
        Initialize the cache. Nothing is read from disk until first use.

        Args:
            cache_path: Path to the JSON file backing this cache.
        """
        self._path = Path(cache_path)
        self._lock = threading.Lock()
        self._entries: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def for_repository(
        cls, config: "Config", workspace_root: str, namespace: str
    ) -> "OutputCache":
        """
        Build the cache for one repository and one call site.

        Args:
            config: Config carrying `output_cache_directory`.
            workspace_root: Repository root the cache is scoped to.
            namespace: Call-site name (e.g. "documentation", "testing",
                "analysis"), so the three agents never share one file
                despite indexing the same repository.

        Returns:
            An OutputCache rooted at
            `<output_cache_directory>/<repo-hash>/<namespace>.json`.
        """
        root = os.path.abspath(os.path.expanduser(str(workspace_root or ".")))
        digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:12]
        base = os.path.abspath(
            os.path.expanduser(str(config.output_cache_directory or "."))
        )
        safe_namespace = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(namespace)
        ) or "default"
        path = os.path.join(base, digest, f"{safe_namespace}.json")
        return cls(path)

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(*parts: Any) -> str:
        """
        Build a stable cache key from arbitrary string-able parts.

        Callers pass whatever should invalidate the entry when it
        changes: symbol source, instruction text, model name, and a
        prompt-template version constant for docs/tests; a hash of the
        retrieved chunk set plus static-finding fingerprints for
        analysis.

        Args:
            *parts: Values joined (in order) before hashing. Each is
                converted with `str()`.

        Returns:
            A hex digest stable across processes and runs for the same
            parts, in the same order.
        """
        joined = _KEY_SEPARATOR.join(str(part) for part in parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Reading and writing
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """
        Look up a cached value.

        Args:
            key: Key produced by `make_key`.

        Returns:
            The cached value (whatever JSON-serializable object was
            passed to `set`), or None on a miss or any read failure.
        """
        with self._lock:
            entries = self._load_locked()
            entry = entries.get(key)
            if not isinstance(entry, dict):
                return None
            return entry.get("value")

    def set(self, key: str, value: Any) -> None:
        """
        Store a value and persist the cache immediately.

        Persisting on every write (rather than batching) keeps this
        simple and crash-safe: entries computed before a mid-run
        failure are not lost. Cache files here stay small -- one entry
        per symbol/target in a single agent run -- so the extra writes
        are not a practical cost.

        Args:
            key: Key produced by `make_key`.
            value: A JSON-serializable value to store.
        """
        with self._lock:
            entries = self._load_locked()
            entries[key] = {
                "value": value,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._save_locked(entries)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_locked(self) -> Dict[str, Any]:
        """Load and cache the on-disk entries. Caller must hold `_lock`."""
        if self._entries is not None:
            return self._entries

        entries: Dict[str, Any] = {}
        if self._path.is_file():
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                if (
                    isinstance(payload, dict)
                    and payload.get("version") == CACHE_VERSION
                    and isinstance(payload.get("entries"), dict)
                ):
                    entries = payload["entries"]
                else:
                    logger.info(
                        "Output cache at %s is missing or has an unexpected "
                        "shape; starting empty.",
                        self._path,
                    )
            except (OSError, ValueError) as exc:
                logger.info(
                    "Could not read output cache at %s (%s); starting empty.",
                    self._path,
                    exc,
                )

        self._entries = entries
        return entries

    def _save_locked(self, entries: Dict[str, Any]) -> None:
        """Persist `entries` to disk. Caller must hold `_lock`."""
        payload = {"version": CACHE_VERSION, "entries": entries}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Could not write output cache to %s (%s); this entry will "
                "not persist across runs.",
                self._path,
                exc,
            )
