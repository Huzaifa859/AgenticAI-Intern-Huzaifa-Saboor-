"""
tracing_hook.py
===============

Bridges HookManager events into the shared Tracer without replacing it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..tracing.events import TraceEventType
from ..tracing.tracer import Tracer
from .base import BaseHook
from .events import HookEvent

_EVENT_TRACE_TYPE = {
    HookEvent.BEFORE_INGEST: TraceEventType.INGESTION,
    HookEvent.AFTER_INGEST: TraceEventType.INGESTION,
    HookEvent.BEFORE_AGENT_RUN: TraceEventType.AGENT_RUN,
    HookEvent.AFTER_AGENT_RUN: TraceEventType.AGENT_RUN,
    HookEvent.BEFORE_TOOL_CALL: TraceEventType.TOOL_CALL,
    HookEvent.AFTER_TOOL_CALL: TraceEventType.TOOL_CALL,
    HookEvent.BEFORE_MODEL_CALL: TraceEventType.MODEL_CALL,
    HookEvent.AFTER_MODEL_CALL: TraceEventType.MODEL_CALL,
    HookEvent.ON_ERROR: TraceEventType.ERROR,
}


class TracingHook(BaseHook):
    """
    Records one Tracer event whenever its HookEvent fires.
    """

    def __init__(
        self,
        event: HookEvent,
        tracer: Optional[Tracer],
        name: str = "",
    ) -> None:
        """
        Args:
            event: Lifecycle event this instance listens for.
            tracer: Shared Tracer; when None, ``run`` is a no-op.
            name: Optional hook name.
        """
        self.event = event
        self.tracer = tracer
        self.name = name or f"tracing_{event.value}"

    def run(self, context: Dict[str, Any]) -> None:
        """Forward the hook payload into ``tracer.record``."""
        if self.tracer is None:
            return
        meta = dict(context or {})
        meta.pop("event", None)
        success = meta.pop("success", None)
        error = meta.pop("error", None)
        duration_ms = meta.pop("duration_ms", None)
        component = str(meta.pop("component", "") or "HookManager")
        self.tracer.record(
            _EVENT_TRACE_TYPE.get(self.event, TraceEventType.LIFECYCLE),
            self.event.value,
            component=component,
            success=success,
            duration_ms=duration_ms,
            error=error,
            **meta,
        )
