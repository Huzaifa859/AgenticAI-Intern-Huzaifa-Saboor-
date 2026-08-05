"""
manager.py
==========

Defines HookManager, which holds registered hooks and fires them at
the appropriate lifecycle events.

Failing hooks are logged and swallowed so instrumentation can never
break the run it observes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseHook
from .events import HookEvent

logger = logging.getLogger(__name__)


class HookManager:
    """
    Registers lifecycle hooks and dispatches events to them.
    """

    def __init__(self) -> None:
        """Initialize an empty hook manager."""
        self._hooks: Dict[HookEvent, List[BaseHook]] = {}

    def register_hook(self, hook: BaseHook) -> None:
        """
        Register a hook against the event it declares.

        Args:
            hook: The BaseHook instance to register.

        Raises:
            TypeError: If ``hook`` is not a BaseHook.
            ValueError: If ``hook.event`` is not a HookEvent.
        """
        if not isinstance(hook, BaseHook):
            raise TypeError(
                f"hook must be a BaseHook, got {type(hook).__name__}"
            )
        event = getattr(hook, "event", None)
        if not isinstance(event, HookEvent):
            raise ValueError(
                f"hook.event must be a HookEvent, got {type(event).__name__}"
            )
        bucket = self._hooks.setdefault(event, [])
        if hook in bucket:
            return
        bucket.append(hook)
        logger.debug(
            "Registered hook %s for %s",
            getattr(hook, "name", "") or type(hook).__name__,
            event.value,
        )

    def unregister_hook(self, hook: BaseHook) -> None:
        """
        Remove a previously registered hook.

        Args:
            hook: The hook to remove.
        """
        if not isinstance(hook, BaseHook):
            return
        event = getattr(hook, "event", None)
        if not isinstance(event, HookEvent):
            return
        bucket = self._hooks.get(event)
        if not bucket:
            return
        try:
            bucket.remove(hook)
        except ValueError:
            return
        if not bucket:
            self._hooks.pop(event, None)

    def trigger(
        self,
        event: HookEvent,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fire all hooks registered for an event.

        Args:
            event: The lifecycle event that occurred.
            context: Event-specific payload passed to each hook.
        """
        if not isinstance(event, HookEvent):
            logger.warning("HookManager.trigger ignored non-HookEvent: %r", event)
            return
        payload = dict(context or {})
        payload.setdefault("event", event.value)
        for hook in list(self._hooks.get(event, ())):
            name = getattr(hook, "name", "") or type(hook).__name__
            try:
                hook.run(payload)
            except Exception as exc:
                logger.warning(
                    "Hook %s failed for %s: %s",
                    name,
                    event.value,
                    exc,
                )

    def list_hooks(self, event: HookEvent) -> List[BaseHook]:
        """
        List the hooks registered for a given event.

        Args:
            event: The event to inspect.

        Returns:
            The hooks listening for that event, in registration order.
        """
        return list(self._hooks.get(event, ()))
