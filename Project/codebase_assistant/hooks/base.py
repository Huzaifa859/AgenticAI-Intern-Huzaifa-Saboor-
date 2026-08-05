"""
base.py
=======

Defines BaseHook, the abstract interface every lifecycle hook
implements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .events import HookEvent


class BaseHook(ABC):
    """
    Abstract base class for all lifecycle hooks.

    A hook declares which `event` it listens for and implements `run`,
    which receives a context dict describing what just happened.
    """

    event: HookEvent
    name: str = ""

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> None:
        """
        React to the hook's event.

        Args:
            context: Event-specific payload (e.g. agent type, tool
                name, elapsed time, raised exception).

        Hooks are side-effect-only and must not alter control flow.
        Exceptions raised here are caught by HookManager.trigger.
        """
        raise NotImplementedError
