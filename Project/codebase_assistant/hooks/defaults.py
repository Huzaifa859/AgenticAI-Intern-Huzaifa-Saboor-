"""
defaults.py
===========

Register the built-in logging and tracing hooks on a HookManager.
"""

from __future__ import annotations

from typing import Optional

from ..tracing.tracer import Tracer
from .events import HookEvent
from .logging_hook import LoggingHook
from .manager import HookManager
from .tracing_hook import TracingHook


def install_default_hooks(
    manager: HookManager,
    tracer: Optional[Tracer] = None,
) -> None:
    """
    Attach LoggingHook + TracingHook for every HookEvent.

    Args:
        manager: Target HookManager.
        tracer: Shared Tracer for TracingHook instances.
    """
    for event in HookEvent:
        manager.register_hook(LoggingHook(event))
        manager.register_hook(TracingHook(event, tracer=tracer))
