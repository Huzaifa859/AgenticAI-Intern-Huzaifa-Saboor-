"""
conversation_memory.py
=======================

Defines ConversationMemory, a short-term, in-session store of
conversational turns / agent interactions used to maintain context
during a single run of the Supervisor.

History is kept purely in-memory (a plain Python list) and is NOT
persisted anywhere — it disappears once the process/session ends.

TODO: Add truncation/summarization strategies for long conversations
and token-budget-aware retrieval. Persistence (if ever needed) would
be a separate concern, likely layered on top via MemoryStore.
"""

from __future__ import annotations

from typing import List, Optional

from ..schemas.schemas import ModelMessage


class ConversationMemory:
    """
    Short-term, in-memory store for the current session's
    conversational history between the user, Supervisor, and Agents.

    History is stored in a plain Python list and is never persisted
    to disk or any external store.
    """

    def __init__(self, max_messages: int = 100) -> None:
        """
        Initialize ConversationMemory.

        Args:
            max_messages: Maximum number of messages to retain before
                truncation/summarization kicks in.

        TODO: Enforce max_messages once truncation/summarization is
        implemented; currently history can grow unbounded.
        """
        self.max_messages = max_messages
        self._history: List[ModelMessage] = []

    def add_message(self, message: ModelMessage) -> None:
        """
        Append a message to the conversation history.

        Args:
            message: The ModelMessage to add.
        """
        self._history.append(message)

    def get_history(self, limit: Optional[int] = None) -> List[ModelMessage]:
        """
        Retrieve the current conversation history.

        Args:
            limit: Optional cap on the number of most recent messages
                to return. If None, the full history is returned.

        Returns:
            A list of ModelMessage objects, oldest first.
        """
        if limit is None:
            return list(self._history)
        return list(self._history[-limit:])

    def clear_history(self) -> None:
        """
        Clear the entire conversation history.
        """
        self._history.clear()

    def summarize(self) -> str:
        """
        Produce a condensed summary of the conversation so far.

        Returns:
            A summary string (placeholder empty string).

        TODO: Implement real summarization, likely via the LLMClient.
        """
        # TODO: implement real summarization
        return ""
