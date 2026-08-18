"""
conversation_memory.py
=======================

Defines ConversationMemory, a short-term, in-session store of
conversational turns / agent interactions used to maintain context
during a single run of the Supervisor.

When a MemoryStore is injected, the current conversation snapshot
(messages, summary, metadata) is loaded on startup and saved after
each update so history survives across application runs. Summarization
behavior is unchanged: long histories still collapse through the
LLMClient into a system summary plus the recent tail.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..schemas.schemas import MemoryRecord, ModelMessage

if TYPE_CHECKING:
    from ..models.model_client import LLMClient
    from .memory_store import MemoryStore
    from ..tracing.tracer import Tracer

logger = logging.getLogger(__name__)

#: Default number of newest messages kept after a summarization pass.
_DEFAULT_KEEP_RECENT = 20

#: Soft ceiling on summary length so the replacement message stays compact.
_SUMMARY_MAX_TOKENS = 512

#: Default persisted conversation key when the caller does not supply one.
_DEFAULT_CONVERSATION_ID = "default"

_SUMMARY_SYSTEM_PROMPT = """\
You summarize prior conversation turns for a multi-agent codebase assistant.

Preserve only what later turns will need:
- repository path or URL under discussion
- important findings (bugs, risks, grounded conclusions)
- user goals and requests
- previous decisions or choices already made
- unresolved issues or open questions

Rules:
1. Stay grounded in the supplied turns. Never invent findings or paths.
2. Prefer concise bullet points over prose.
3. If a detail is missing from the turns, omit it rather than guessing.
4. Return markdown bullets only — no preamble, no JSON.
"""


class ConversationMemory:
    """
    Short-term conversation history for the current session.

    History lives in a plain Python list for the duration of the process.
    When a MemoryStore is provided, the same snapshot is also persisted
    under ``.codebase_assistant/memory_store/`` so a later run can reload
    it. When the list grows past ``max_messages``, older turns are
    summarized through the injected LLMClient.
    """

    def __init__(
        self,
        max_messages: int = 100,
        model_client: Optional["LLMClient"] = None,
        keep_recent: int = _DEFAULT_KEEP_RECENT,
        memory_store: Optional["MemoryStore"] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tracer: Optional["Tracer"] = None,
    ) -> None:
        """
        Initialize ConversationMemory.

        Args:
            max_messages: Message count that triggers summarization.
                History at or below this length is left unchanged.
            model_client: Optional LLMClient used to produce summaries.
                When omitted or unavailable, summarization is a no-op
                and history is left intact.
            keep_recent: Number of newest messages retained after a
                successful summarization. Clamped so at least one older
                message is available to summarize when the threshold is
                crossed.
            memory_store: Optional persistent MemoryStore. When set,
                existing memory for ``conversation_id`` is loaded and
                later updates are saved automatically.
            conversation_id: Stable id for the persisted conversation.
                Defaults to ``\"default\"``.
            metadata: Optional metadata merged into the persisted
                snapshot (e.g. repository path).
            tracer: Optional shared Tracer for summarize/persist events.
        """
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2.")
        self.max_messages = max_messages
        self.model_client = model_client
        # Always leave room for at least one older message to summarize.
        self.keep_recent = max(1, min(int(keep_recent), max_messages - 1))
        self.memory_store = memory_store
        self.conversation_id = (
            (conversation_id or "").strip() or _DEFAULT_CONVERSATION_ID
        )
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.tracer = tracer
        self._history: List[ModelMessage] = []
        self._summary: str = ""

        if self.memory_store is not None:
            self._load_from_store()

    def _trace(self, name: str, *, success: Optional[bool] = True, **metadata: Any) -> None:
        """Emit a ConversationMemory lifecycle event when tracing is on."""
        if self.tracer is None:
            return
        from ..tracing.events import TraceEventType

        self.tracer.record(
            TraceEventType.MEMORY,
            name,
            component="ConversationMemory",
            success=success,
            **metadata,
        )

    def add_message(self, message: ModelMessage) -> None:
        """
        Append a message to the conversation history.

        When the history exceeds ``max_messages``, summarization is
        attempted automatically. A failed or unavailable summarization
        leaves the full history in place. After the update (and any
        summarization), the snapshot is persisted when a MemoryStore is
        configured.

        Args:
            message: The ModelMessage to add.
        """
        self._history.append(message)
        if len(self._history) > self.max_messages:
            self.summarize()
        # Always persist after an update. Successful summarize() also
        # saves; a second write is harmless and covers the case where
        # summarization was skipped or failed.
        self._save_to_store()

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
        Clear the entire conversation history and persisted summary.
        """
        self._history.clear()
        self._summary = ""
        self._save_to_store()

    def summarize(self) -> str:
        """
        Condense older turns into a system summary when over threshold.

        Behavior:
            - Short histories (``len <= max_messages``) are unchanged and
              yield an empty string.
            - When over the threshold, older messages are sent to the
              LLMClient. On success they are replaced by one system
              summary message followed by the recent tail.
            - If the provider is unavailable, the client is missing, or
              generation fails, history is left unchanged and an empty
              string is returned so nothing is lost.

        Returns:
            The summary text produced by the model, or an empty string
            when summarization did not run or did not succeed.
        """
        if len(self._history) <= self.max_messages:
            return ""

        self._trace("summarize_started", messages=len(self._history))

        if not self._model_available():
            logger.info(
                "ConversationMemory: provider unavailable; leaving %d "
                "message(s) unchanged.",
                len(self._history),
            )
            self._trace(
                "summarize_finished",
                success=False,
                error="provider unavailable",
                messages=len(self._history),
            )
            return ""

        recent_count = min(self.keep_recent, len(self._history) - 1)
        old_messages = self._history[:-recent_count]
        recent_messages = self._history[-recent_count:]

        try:
            summary = self._generate_summary(old_messages)
        except Exception as exc:
            logger.warning(
                "ConversationMemory: summarization failed; leaving "
                "history unchanged: %s",
                exc,
            )
            self._trace(
                "summarize_finished",
                success=False,
                error=str(exc),
            )
            return ""

        if not summary or not summary.strip():
            logger.warning(
                "ConversationMemory: empty summary returned; leaving "
                "history unchanged."
            )
            self._trace(
                "summarize_finished",
                success=False,
                error="empty summary",
            )
            return ""

        summary = summary.strip()
        self._summary = summary
        self._history = [
            ModelMessage(
                role="system",
                content=f"Conversation summary:\n{summary}",
            ),
            *recent_messages,
        ]
        logger.info(
            "ConversationMemory: summarized %d older message(s); "
            "kept %d recent message(s).",
            len(old_messages),
            len(recent_messages),
        )
        self._trace(
            "summarize_finished",
            success=True,
            old_messages=len(old_messages),
            kept_recent=len(recent_messages),
            summary_chars=len(summary),
        )
        self._save_to_store()
        return summary

    def _model_available(self) -> bool:
        """Report whether the injected client can serve a summary request."""
        if self.model_client is None:
            return False
        try:
            return bool(self.model_client.is_available())
        except Exception as exc:
            logger.warning(
                "ConversationMemory: availability check failed: %s", exc
            )
            return False

    def _generate_summary(self, messages: List[ModelMessage]) -> str:
        """
        Ask the LLMClient to summarize a slice of the history.

        Args:
            messages: Older turns to condense.

        Returns:
            Summary text from the model.

        Raises:
            Exception: Propagated from the client so ``summarize`` can
                leave history unchanged.
        """
        assert self.model_client is not None
        transcript = self._format_transcript(messages)
        response = self.model_client.generate(
            [
                ModelMessage(role="system", content=_SUMMARY_SYSTEM_PROMPT),
                ModelMessage(
                    role="user",
                    content=(
                        "Summarize the following conversation turns.\n\n"
                        f"{transcript}"
                    ),
                ),
            ],
            max_tokens=_SUMMARY_MAX_TOKENS,
            temperature=0.0,
        )
        return (response.content or "").strip()

    def _load_from_store(self) -> None:
        """
        Restore history and summary from the MemoryStore when present.

        Missing or corrupted records leave an empty in-memory history;
        the next save will write a clean snapshot.
        """
        if self.memory_store is None:
            return

        try:
            record = self.memory_store.load(self.conversation_id)
        except Exception as exc:
            logger.warning(
                "ConversationMemory: load failed for %r; starting fresh: %s",
                self.conversation_id,
                exc,
            )
            return

        if record is None:
            logger.info(
                "ConversationMemory: no persisted memory for %r; starting fresh.",
                self.conversation_id,
            )
            return

        payload = record.value
        if not isinstance(payload, dict):
            # Older / alternate shape: treat the whole value as opaque.
            logger.warning(
                "ConversationMemory: unexpected record value for %r; "
                "starting fresh.",
                self.conversation_id,
            )
            return

        messages = payload.get("messages") or []
        restored: List[ModelMessage] = []
        if isinstance(messages, list):
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "").strip()
                content = str(item.get("content") or "")
                if role and content.strip():
                    restored.append(ModelMessage(role=role, content=content))

        self._history = restored
        self._summary = str(payload.get("summary") or "")
        stored_meta = payload.get("metadata")
        if isinstance(stored_meta, dict):
            merged = dict(stored_meta)
            merged.update(self.metadata)
            self.metadata = merged
        elif isinstance(record.metadata, dict) and record.metadata:
            merged = dict(record.metadata)
            merged.update(self.metadata)
            self.metadata = merged

        logger.info(
            "ConversationMemory: loaded %d message(s) for %r.",
            len(self._history),
            self.conversation_id,
        )

    def _save_to_store(self) -> None:
        """
        Persist the current conversation snapshot to the MemoryStore.

        Failures are logged and swallowed so persistence never aborts
        the session.
        """
        if self.memory_store is None:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        snapshot = {
            "conversation_id": self.conversation_id,
            "timestamp": timestamp,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in self._history
            ],
            "summary": self._summary,
            "metadata": dict(self.metadata),
        }
        record = MemoryRecord(
            key=self.conversation_id,
            value=snapshot,
            metadata={
                "conversation_id": self.conversation_id,
                "timestamp": timestamp,
                **dict(self.metadata),
            },
        )
        try:
            saved = self.memory_store.save(record)
        except Exception as exc:
            logger.warning(
                "ConversationMemory: save failed for %r: %s",
                self.conversation_id,
                exc,
            )
            self._trace(
                "persisted",
                success=False,
                conversation_id=self.conversation_id,
                error=str(exc),
            )
            return

        if not saved:
            logger.warning(
                "ConversationMemory: MemoryStore refused save for %r.",
                self.conversation_id,
            )
            self._trace(
                "persisted",
                success=False,
                conversation_id=self.conversation_id,
                error="store refused save",
            )
            return

        self._trace(
            "persisted",
            success=True,
            conversation_id=self.conversation_id,
            messages=len(self._history),
        )

    @staticmethod
    def _format_transcript(messages: List[ModelMessage]) -> str:
        """
        Render turns as a plain transcript for the summarizer prompt.

        Args:
            messages: Turns to format.

        Returns:
            A role-tagged transcript string.
        """
        lines: List[str] = []
        for index, message in enumerate(messages, start=1):
            role = (message.role or "unknown").strip() or "unknown"
            content = (message.content or "").strip()
            if not content:
                continue
            lines.append(f"[{index}] {role}:\n{content}")
        return "\n\n".join(lines) if lines else "(no content)"


def new_conversation_id() -> str:
    """
    Generate a fresh conversation identifier.

    Returns:
        A UUID4 string suitable for use as ``conversation_id``.
    """
    return str(uuid.uuid4())
