"""
hooks
=====

Lifecycle hooks — callbacks fired at defined points in a run
(ingestion, agent dispatch, tool calls, errors).

This is the seam the proposal's tracing layer is meant to attach to:
timing, logging, and per-agent metrics can subscribe to hook events
without any of that instrumentation being threaded through the
Supervisor and agents by hand. The Week 6 error-handling requirements
(invalid repo URL, empty files, embedding failures) surface through
the ON_ERROR event.

Contains:
- HookEvent: the enumerated points where hooks fire.
- BaseHook: abstract interface every hook implements.
- HookManager: registration and dispatch of hooks.

NOTE: Placeholder only. No hook is implemented and no event is fired
by the existing code yet.
"""

from .base import BaseHook
from .events import HookEvent
from .manager import HookManager

__all__ = ["HookEvent", "BaseHook", "HookManager"]
