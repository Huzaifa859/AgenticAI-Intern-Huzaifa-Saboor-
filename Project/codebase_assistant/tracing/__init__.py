"""
tracing
=======

Observability layer — timing, token usage, and per-stage metrics for a
run.

The proposal's tech stack refers to the agent layer's "tracing layer";
this is it. Tracing is the consumer, `hooks/` is the transport: hooks
define *where* instrumentation can attach, tracing defines *what gets
recorded* when it fires. Keeping them apart means the Supervisor and
agents never carry timing code themselves.

The metrics this feeds are the ones the proposal's Evaluation Criteria
asks for, in particular end-to-end response time per agent.

Contains:
- TraceEventType / TraceEvent: the record types emitted during a run.
- Tracer: collects those records and exports them.

NOTE: Placeholder only. Nothing is recorded or exported yet, and no
existing code emits trace events.
"""

from .events import TraceEvent, TraceEventType
from .tracer import Tracer

__all__ = ["TraceEventType", "TraceEvent", "Tracer"]
