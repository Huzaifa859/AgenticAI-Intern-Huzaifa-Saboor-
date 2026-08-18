"""
test_output_cache.py
=====================

Unit tests for the persistent cross-run LLM output cache.
"""

from __future__ import annotations

import threading
from pathlib import Path

from codebase_assistant.cache.output_cache import OutputCache
from codebase_assistant.config import Config


def test_get_missing_key_returns_none(tmp_path: Path) -> None:
    cache = OutputCache(str(tmp_path / "cache.json"))
    assert cache.get("missing") is None


def test_set_then_get_round_trips_value(tmp_path: Path) -> None:
    cache = OutputCache(str(tmp_path / "cache.json"))
    key = OutputCache.make_key("source", "instruction", "model-x", "v1")
    cache.set(key, {"summary": "does a thing", "count": 3})

    assert cache.get(key) == {"summary": "does a thing", "count": 3}


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = str(tmp_path / "cache.json")
    key = OutputCache.make_key("a", "b")

    first = OutputCache(path)
    first.set(key, {"value": 1})

    second = OutputCache(path)
    assert second.get(key) == {"value": 1}


def test_make_key_is_stable_and_order_sensitive() -> None:
    a = OutputCache.make_key("foo", "bar")
    b = OutputCache.make_key("foo", "bar")
    c = OutputCache.make_key("bar", "foo")

    assert a == b
    assert a != c


def test_make_key_distinguishes_different_inputs() -> None:
    a = OutputCache.make_key("source-v1", "doc", "model", "v1")
    b = OutputCache.make_key("source-v2", "doc", "model", "v1")
    assert a != b


def test_corrupted_cache_file_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("{not valid json", encoding="utf-8")

    cache = OutputCache(str(path))
    assert cache.get("anything") is None

    # Writing after a corrupt read should still work and overwrite cleanly.
    cache.set("k", {"ok": True})
    assert cache.get("k") == {"ok": True}


def test_version_mismatch_starts_fresh(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        '{"version": 999, "entries": {"k": {"value": "stale"}}}',
        encoding="utf-8",
    )

    cache = OutputCache(str(path))
    assert cache.get("k") is None


def test_for_repository_scopes_by_workspace_and_namespace(tmp_path: Path) -> None:
    config = Config(output_cache_directory=str(tmp_path / "output_cache"))

    docs_cache = OutputCache.for_repository(config, str(tmp_path / "repo-a"), "documentation")
    testing_cache = OutputCache.for_repository(config, str(tmp_path / "repo-a"), "testing")
    other_repo_cache = OutputCache.for_repository(config, str(tmp_path / "repo-b"), "documentation")

    assert docs_cache._path != testing_cache._path
    assert docs_cache._path != other_repo_cache._path

    docs_cache.set("k", {"v": 1})
    assert testing_cache.get("k") is None  # different namespace, no bleed-through


def test_concurrent_writes_do_not_corrupt_cache(tmp_path: Path) -> None:
    cache = OutputCache(str(tmp_path / "cache.json"))

    def _writer(i: int) -> None:
        cache.set(f"key-{i}", {"index": i})

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(20):
        assert cache.get(f"key-{i}") == {"index": i}
