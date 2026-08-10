"""
base.py
=======

Defines BasePlugin, the abstract interface every plugin implements.

TODO: Implement concrete plugins for the proposal's stretch goals
(Refactoring Agent, architecture diagram generation from the import
graph) once the guaranteed MVP is complete.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePlugin(ABC):
    """
    Abstract base class for all plugins.

    A plugin receives the running system's context on `setup` (giving
    it access to the supervisor, registries, and config), registers
    whatever it contributes, and releases those resources on
    `teardown`.
    """

    name: str = ""
    version: str = "0.0.0"
    description: str = ""

    @abstractmethod
    def setup(self, context: Dict[str, Any]) -> None:
        """
        Initialize the plugin and register what it contributes.

        Args:
            context: Shared system context (supervisor, registries,
                config) the plugin may hook into.

        TODO: Implement in each concrete subclass. Define the exact
        contents of `context` once a real plugin exists.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """
        Release any resources held by the plugin.

        TODO: Override in subclasses that acquire resources.
        """
        # TODO: implement real teardown in subclasses
        pass
