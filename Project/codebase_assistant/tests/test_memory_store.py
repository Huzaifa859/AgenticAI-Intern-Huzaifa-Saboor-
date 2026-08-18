"""
test_memory_store.py
=====================

Unit tests for JSON-backed MemoryStore persistence and for
ConversationMemory auto load/save across restarts.
"""

from __future__ import annotations

import json
from pathlib import Path

from codebase_assistant.memory.conversation_memory import ConversationMemory
from codebase_assistant.memory.memory_store import MemoryStore
from codebase_assistant.schemas.schemas import MemoryRecord, ModelMessage


def test_first_run_creates_memory_directory_and_file(tmp_path: Path) -> None:
    """Saving a record should create the store directory and a JSON file."""
    store_path = tmp_path / "memory_store"
    store = MemoryStore(storage_path=str(store_path))

    assert store_path.is_dir()
    assert store.save(
        MemoryRecord(key="default", value={"hello": "world"}, metadata={"k": 1})
    )
    assert (store_path / "default.json").is_file()


def test_restart_reloads_memory(tmp_path: Path) -> None:
    """A new MemoryStore instance should read what the previous one wrote."""
    store_path = tmp_path / "memory_store"
    first = MemoryStore(storage_path=str(store_path))
    first.save(
        MemoryRecord(
            key="session-1",
            value={"messages": [{"role": "user", "content": "hi"}]},
            metadata={"repo": "demo"},
        )
    )

    second = MemoryStore(storage_path=str(store_path))
    record = second.load("session-1")

    assert record is not None
    assert record.key == "session-1"
    assert record.value["messages"][0]["content"] == "hi"
    assert record.metadata["repo"] == "demo"


def test_corrupted_file_handled_gracefully(tmp_path: Path) -> None:
    """Corrupt JSON should be removed and treated as a miss."""
    store_path = tmp_path / "memory_store"
    store = MemoryStore(storage_path=str(store_path))
    bad = store_path / "broken.json"
    bad.write_text("{not-json", encoding="utf-8")

    assert store.load("broken") is None
    assert not bad.exists()
    assert store_path.is_dir()

    # A later save should recreate a clean file.
    assert store.save(MemoryRecord(key="broken", value={"ok": True}, metadata={}))
    assert json.loads(bad.read_text(encoding="utf-8"))["value"]["ok"] is True


def test_conversation_memory_persists_and_reloads(tmp_path: Path) -> None:
    """ConversationMemory should auto-save and restore across instances."""
    store = MemoryStore(storage_path=str(tmp_path / "memory_store"))
    first = ConversationMemory(
        memory_store=store,
        conversation_id="demo",
        metadata={"repository": "/tmp/demo"},
    )
    first.add_message(ModelMessage(role="user", content="analyze this repo"))
    first.add_message(ModelMessage(role="assistant", content="found one bug"))
    first._summary = "- unresolved: coverage"
    first._save_to_store()

    second = ConversationMemory(memory_store=store, conversation_id="demo")
    history = second.get_history()

    assert [message.content for message in history] == [
        "analyze this repo",
        "found one bug",
    ]
    assert second._summary == "- unresolved: coverage"
    assert second.metadata["repository"] == "/tmp/demo"

    on_disk = json.loads((tmp_path / "memory_store" / "demo.json").read_text(encoding="utf-8"))
    snapshot = on_disk["value"]
    assert snapshot["conversation_id"] == "demo"
    assert "timestamp" in snapshot
    assert snapshot["messages"][0]["role"] == "user"
    assert snapshot["summary"] == "- unresolved: coverage"
    assert snapshot["metadata"]["repository"] == "/tmp/demo"


def test_conversation_without_store_stays_ephemeral(tmp_path: Path) -> None:
    """Omitting MemoryStore must preserve the previous in-memory-only behavior."""
    memory = ConversationMemory()
    memory.add_message(ModelMessage(role="user", content="hello"))

    assert len(memory.get_history()) == 1
    assert not any(tmp_path.rglob("*.json"))


def test_conversation_survives_corrupt_store_file(tmp_path: Path) -> None:
    """A corrupt persisted conversation should start fresh, not crash."""
    store_path = tmp_path / "memory_store"
    store_path.mkdir()
    (store_path / "default.json").write_text("[[[", encoding="utf-8")

    memory = ConversationMemory(
        memory_store=MemoryStore(storage_path=str(store_path)),
        conversation_id="default",
    )
    assert memory.get_history() == []
    memory.add_message(ModelMessage(role="user", content="new session"))
    assert memory.get_history()[0].content == "new session"
    assert (store_path / "default.json").is_file()


def test_delete_and_clear(tmp_path: Path) -> None:
    """delete and clear should remove persisted files."""
    store = MemoryStore(storage_path=str(tmp_path / "memory_store"))
    store.save(MemoryRecord(key="a", value=1, metadata={}))
    store.save(MemoryRecord(key="b", value=2, metadata={}))

    assert store.delete("a") is True
    assert store.load("a") is None
    assert store.clear() is True
    assert store.list_keys() == []


def test_search_finds_matching_records(tmp_path: Path) -> None:
    """Substring search should surface records that mention the query."""
    store = MemoryStore(storage_path=str(tmp_path / "memory_store"))
    store.save(
        MemoryRecord(
            key="one",
            value={"summary": "Flask routing bugs"},
            metadata={},
        )
    )
    store.save(
        MemoryRecord(
            key="two",
            value={"summary": "Django templates"},
            metadata={},
        )
    )

    hits = store.search("flask", top_k=5)
    assert len(hits) == 1
    assert hits[0].key == "one"
