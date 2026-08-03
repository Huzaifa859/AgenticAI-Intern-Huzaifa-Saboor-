"""
events.py
=========

Record types emitted while tracing a run.

Not to be confused with `hooks/events.py`. HookEvent enumerates the
lifecycle *points* where instrumentation may attach; the types here are
the *measurements* recorded when one of those points fires.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TraceEventType(str, Enum):
    """
    Categories of measurement recorded during a run.

    Attributes:
        SPAN_START: A timed stage began.
        SPAN_END: A timed stage completed.
        INGESTION: A repository ingestion step.
        RETRIEVAL: A RAG retrieval query.
        MODEL_CALL: A request to an LLM provider.
        TOOL_CALL: A tool invocation.
        AGENT_RUN: A full agent dispatch or lifecycle marker.
        ERROR: A failure encountered at any stage.
        LIFECYCLE: Application / orchestration lifecycle marker.
        MEMORY: Conversation or persistent memory activity.
    """

    SPAN_START = "span_start"
    SPAN_END = "span_end"
    INGESTION = "ingestion"
    RETRIEVAL = "retrieval"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    AGENT_RUN = "agent_run"
    ERROR = "error"
    LIFECYCLE = "lifecycle"
    MEMORY = "memory"


@dataclass
class TraceEvent:
    """
    A single recorded measurement.

    Defined as a dataclass rather than a Pydantic model because trace
    records stay internal to this package and never cross the
    subsystem boundaries that ``schemas/`` exists to type.

    Attributes:
        event_type: Which category of measurement this is.
        name: Identifier of the stage (e.g. ``analysis_started``).
        timestamp: When the event occurred, as a Unix timestamp.
        duration_ms: Elapsed time in milliseconds, when applicable.
        metadata: Free-form extra detail (paths, counts, scores).
        error: Error message, when the stage failed.
        component: Subsystem that emitted the event (CLI, Supervisor, …).
        success: Whether the stage succeeded (None when not applicable).
        sequence: Monotonic order index within the run (deterministic).
    """

    event_type: TraceEventType
    name: str = ""
    timestamp: float = 0.0
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    component: str = ""
    success: Optional[bool] = None
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this event for export.

        Returns:
            A JSON-friendly dict with enum values as strings and
            metadata keys sorted for deterministic output.
        """
        payload = asdict(self)
        payload["event_type"] = self.event_type.value
        payload["metadata"] = {
            key: payload["metadata"][key]
            for key in sorted(payload["metadata"].keys())
        }
        return payload
