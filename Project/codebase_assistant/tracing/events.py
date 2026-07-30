"""
events.py
=========

Record types emitted while tracing a run.

Not to be confused with `hooks/events.py`. HookEvent enumerates the
lifecycle *points* where instrumentation may attach; the types here are
the *measurements* recorded when one of those points fires.

TODO: Emit these from the Supervisor, agents, tools, RAG pipeline, and
model providers once the Tracer is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        AGENT_RUN: A full agent dispatch.
        ERROR: A failure encountered at any stage.
    """

    SPAN_START = "span_start"
    SPAN_END = "span_end"
    INGESTION = "ingestion"
    RETRIEVAL = "retrieval"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    AGENT_RUN = "agent_run"
    ERROR = "error"


@dataclass
class TraceEvent:
    """
    A single recorded measurement.

    Defined as a dataclass rather than a Pydantic model because trace
    records stay internal to this package and never cross the
    subsystem boundaries that `schemas/` exists to type.

    Attributes:
        event_type: Which category of measurement this is.
        name: Identifier of the stage being measured (e.g. an agent
            type, tool name, or model slug).
        timestamp: When the event occurred, as a Unix timestamp.
        duration_ms: Elapsed time in milliseconds, for completed spans.
        metadata: Free-form extra detail (token counts, chunk counts,
            retrieval scores).
        error: Error message, when event_type is ERROR.
    """

    event_type: TraceEventType
    name: str = ""
    timestamp: float = 0.0
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
