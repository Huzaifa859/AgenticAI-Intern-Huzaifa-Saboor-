"""
manager.py
==========

Defines HookManager, which holds registered hooks and fires them at
the appropriate lifecycle events.

TODO: Implement real registration and dispatch, and call `trigger`
from the Supervisor, agents, tools, and the ingestion pipeline. A
failing hook must never break the run it is observing.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseHook
from .events import HookEvent


class HookManager:
    """
    Registers lifecycle hooks and dispatches events to them.
    """

    def __init__(self) -> None:
        """Initialize an empty hook manager."""
        # Intended backing state: event -> hooks listening for it.
        self._hooks: Dict[HookEvent, List[BaseHook]] = {}

    def register_hook(self, hook: BaseHook) -> None:
        """
        Register a hook against the event it declares.

        Args:
            hook: The BaseHook instance to register.

        TODO: Implement real registration, including ordering/priority
        when several hooks share one event.
        """
        # TODO: implement real hook registration
        pass

    def unregister_hook(self, hook: BaseHook) -> None:
        """
        Remove a previously registered hook.

        Args:
            hook: The hook to remove.

        TODO: Implement real deregistration.
        """
        # TODO: implement real hook deregistration
        pass

    def trigger(self, event: HookEvent, context: Dict[str, Any]) -> None:
        """
        Fire all hooks registered for an event.

        Args:
            event: The lifecycle event that occurred.
            context: Event-specific payload passed to each hook.

        TODO: Implement real dispatch. Exceptions raised by a hook must
        be caught and logged rather than propagated, so instrumentation
        can never break the run it observes.
        """
        # TODO: implement real hook dispatch
        pass

    def list_hooks(self, event: HookEvent) -> List[BaseHook]:
        """
        List the hooks registered for a given event.

        Args:
            event: The event to inspect.

        Returns:
            The hooks listening for that event (placeholder empty
            list).

        TODO: Implement real lookup.
        """
        # TODO: implement real hook listing
        return []
