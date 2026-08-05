"""
logging_hook.py
===============

Logs lifecycle hook events at INFO for visibility during a run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .base import BaseHook
from .events import HookEvent

logger = logging.getLogger(__name__)


class LoggingHook(BaseHook):
    """
    Side-effect-only hook that logs the event name and selected context.
    """

    def __init__(self, event: HookEvent, name: str = "") -> None:
        """
        Args:
            event: Lifecycle event this instance listens for.
            name: Optional hook name for registration logs.
        """
        self.event = event
        self.name = name or f"logging_{event.value}"

    def run(self, context: Dict[str, Any]) -> None:
        """Log a compact summary of the event payload."""
        keys = (
            "agent_type",
            "task_id",
            "success",
            "error",
            "component",
            "model",
            "workspace",
        )
        parts = [f"{key}={context[key]!r}" for key in keys if key in context]
        detail = ", ".join(parts)
        if detail:
            logger.info("hook.%s: %s", self.event.value, detail)
        else:
            logger.info("hook.%s", self.event.value)
