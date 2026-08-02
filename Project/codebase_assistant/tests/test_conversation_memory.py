"""
test_conversation_memory.py
============================

Unit tests for ConversationMemory summarization.

The LLMClient is mocked so tests never call a live provider.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from codebase_assistant.memory.conversation_memory import ConversationMemory
from codebase_assistant.schemas.schemas import ModelMessage, ModelResponse


def _messages(count: int, prefix: str = "msg") -> List[ModelMessage]:
    """Build ``count`` alternating user/assistant messages."""
    roles = ("user", "assistant")
    return [
        ModelMessage(role=roles[index % 2], content=f"{prefix}-{index}")
        for index in range(count)
    ]


def _client(available: bool = True, content: str = "- repo: demo\n- goal: analyze") -> MagicMock:
    """Build a mock LLMClient."""
    client = MagicMock()
    client.is_available.return_value = available
    client.generate.return_value = ModelResponse(content=content, usage={}, raw={})
    return client


def test_short_conversation_is_unchanged() -> None:
    """Histories at or below the threshold must not be summarized."""
    client = _client()
    memory = ConversationMemory(max_messages=10, keep_recent=4, model_client=client)
    for message in _messages(10):
        memory.add_message(message)

    assert len(memory.get_history()) == 10
    assert memory.summarize() == ""
    client.generate.assert_not_called()
    assert [message.content for message in memory.get_history()] == [
        f"msg-{index}" for index in range(10)
    ]


def test_long_conversation_is_summarized() -> None:
    """Crossing the threshold should replace older turns with a summary."""
    client = _client(content="- repo: /tmp/demo\n- goal: find bugs")
    memory = ConversationMemory(max_messages=6, keep_recent=3, model_client=client)

    for message in _messages(7):
        memory.add_message(message)

    history = memory.get_history()
    assert history[0].role == "system"
    assert "Conversation summary:" in history[0].content
    assert "/tmp/demo" in history[0].content
    # Recent tail preserved in order.
    assert [message.content for message in history[1:]] == ["msg-4", "msg-5", "msg-6"]
    client.generate.assert_called_once()


def test_recent_messages_are_preserved() -> None:
    """Only older messages are summarized; the recent window stays intact."""
    client = _client()
    memory = ConversationMemory(max_messages=5, keep_recent=2, model_client=client)
    for message in _messages(5, prefix="early"):
        memory.add_message(message)

    # Still under/at threshold after 5; force over with one more.
    memory.add_message(ModelMessage(role="user", content="late-goal"))

    history = memory.get_history()
    assert [message.content for message in history[1:]] == ["early-4", "late-goal"]
    assert all(message.content != "early-0" for message in history[1:])


def test_provider_unavailable_leaves_history_intact() -> None:
    """An unavailable provider must not drop or rewrite history."""
    client = _client(available=False)
    memory = ConversationMemory(max_messages=4, keep_recent=2, model_client=client)
    seeded = _messages(5)
    for message in seeded:
        memory.add_message(message)

    assert len(memory.get_history()) == 5
    assert [message.content for message in memory.get_history()] == [
        message.content for message in seeded
    ]
    client.generate.assert_not_called()
    assert memory.summarize() == ""


def test_missing_client_leaves_history_intact() -> None:
    """No injected client means summarization is a silent no-op."""
    memory = ConversationMemory(max_messages=3, keep_recent=1)
    for message in _messages(4):
        memory.add_message(message)

    assert len(memory.get_history()) == 4
    assert memory.summarize() == ""


def test_failed_generation_leaves_history_intact() -> None:
    """A raising generate() must not erase conversation history."""
    client = _client()
    client.generate.side_effect = RuntimeError("boom")
    memory = ConversationMemory(max_messages=3, keep_recent=1, model_client=client)
    seeded = _messages(4)
    # Bypass auto-summarize so we can assert the explicit path.
    memory._history = list(seeded)

    assert memory.summarize() == ""
    assert memory.get_history() == seeded


def test_empty_model_summary_leaves_history_intact() -> None:
    """Whitespace-only model output must not replace real history."""
    client = _client(content="   \n")
    memory = ConversationMemory(max_messages=3, keep_recent=1, model_client=client)
    seeded = _messages(4)
    memory._history = list(seeded)

    assert memory.summarize() == ""
    assert memory.get_history() == seeded


def test_summary_prompt_asks_for_required_fields() -> None:
    """The summarizer prompt must request the fields the step requires."""
    client = _client()
    memory = ConversationMemory(max_messages=3, keep_recent=1, model_client=client)
    memory._history = _messages(4)
    memory.summarize()

    sent = client.generate.call_args.args[0]
    system = sent[0].content.lower()
    for phrase in (
        "repository",
        "findings",
        "goals",
        "decisions",
        "unresolved",
    ):
        assert phrase in system


def test_manual_summarize_returns_summary_text() -> None:
    """summarize() should return the model text when it rewrites history."""
    client = _client(content="- unresolved: coverage gaps")
    memory = ConversationMemory(max_messages=3, keep_recent=1, model_client=client)
    memory._history = _messages(4)

    summary = memory.summarize()

    assert summary == "- unresolved: coverage gaps"
    assert len(memory.get_history()) == 2


def test_supervisor_wires_model_client_into_conversation_memory() -> None:
    """Supervisor should inject its OpenRouter-backed client for summaries."""
    from unittest.mock import patch

    from codebase_assistant.supervisor import Supervisor

    with patch.object(Supervisor, "_init_openrouter_provider", return_value=None), \
        patch.object(Supervisor, "_init_ollama_provider", return_value=None), \
        patch("codebase_assistant.supervisor.Indexer"), \
        patch("codebase_assistant.supervisor.Retriever"), \
        patch("codebase_assistant.supervisor.MemoryStore"), \
        patch("codebase_assistant.supervisor.GitHubTools"), \
        patch("codebase_assistant.supervisor.FilesystemTools"), \
        patch("codebase_assistant.supervisor.CodeAnalysisAgent"), \
        patch("codebase_assistant.supervisor.DocumentationAgent"), \
        patch("codebase_assistant.supervisor.TestingAgent"):
        supervisor = Supervisor()

    assert supervisor.conversation_memory.model_client is supervisor.model_client
    assert supervisor.conversation_memory.memory_store is supervisor.memory_store
