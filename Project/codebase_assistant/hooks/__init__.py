"""
hooks
=====

Lifecycle hooks — callbacks fired at defined points in a run
(ingestion, agent dispatch, tool calls, model calls, errors).

This is the seam the proposal's tracing layer attaches to: timing,
logging, and per-agent metrics can subscribe to hook events without
instrumentation being threaded through every agent by hand.

Contains:
- HookEvent: the enumerated points where hooks fire.
- BaseHook: abstract interface every hook implements.
- HookManager: registration and dispatch of hooks.
- LoggingHook / TracingHook: built-in subscribers.
- install_default_hooks: register logging + tracing for all events.
"""

from .base import BaseHook
from .defaults import install_default_hooks
from .events import HookEvent
from .logging_hook import LoggingHook
from .manager import HookManager
from .tracing_hook import TracingHook

__all__ = [
    "HookEvent",
    "BaseHook",
    "HookManager",
    "LoggingHook",
    "TracingHook",
    "install_default_hooks",
]
