"""
tracing
=======

Observability layer — timing, token usage, and per-stage metrics for a
run.

The proposal's tech stack refers to the agent layer's "tracing layer";
this is it. Tracing is the consumer, `hooks/` is the transport: hooks
define *where* instrumentation can attach, tracing defines *what gets
recorded* when it fires.

Contains:
- TraceEventType / TraceEvent: the record types emitted during a run.
- Tracer: collects those records and exports them.
"""

from .events import TraceEvent, TraceEventType
from .tracer import Tracer

__all__ = ["TraceEventType", "TraceEvent", "Tracer"]
