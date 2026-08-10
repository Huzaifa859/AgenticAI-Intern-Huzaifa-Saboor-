"""
manager.py
==========

Defines PluginManager, responsible for discovering, loading, and
unloading plugins at runtime.

TODO: Implement real discovery (entry points or a plugins directory
scan), dependency-aware load ordering, and failure isolation so one
broken plugin cannot take down the system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BasePlugin


class PluginManager:
    """
    Discovers and manages the lifecycle of plugins.
    """

    def __init__(self, plugin_dir: str = "./plugins") -> None:
        """
        Initialize the PluginManager.

        Args:
            plugin_dir: Directory scanned for available plugins.
        """
        self.plugin_dir = plugin_dir

        # Intended backing state: plugin name -> loaded plugin instance.
        self._plugins: Dict[str, BasePlugin] = {}

    def discover(self) -> List[str]:
        """
        Find plugins available to be loaded.

        Returns:
            A list of discoverable plugin names (placeholder empty
            list).

        TODO: Implement real discovery via entry points or a directory
        scan of `plugin_dir`.
        """
        # TODO: implement real plugin discovery
        return []

    def load(self, name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load a plugin by name and run its `setup`.

        Args:
            name: Name of the plugin to load.
            context: Shared system context passed to the plugin.

        Returns:
            True if the plugin loaded successfully (placeholder always
            returns False).

        TODO: Implement real import, instantiation, and setup, with
        errors isolated to the failing plugin.
        """
        # TODO: implement real plugin loading
        return False

    def unload(self, name: str) -> bool:
        """
        Unload a plugin by name, running its `teardown`.

        Args:
            name: Name of the plugin to unload.

        Returns:
            True if the plugin unloaded successfully (placeholder
            always returns False).

        TODO: Implement real teardown and deregistration.
        """
        # TODO: implement real plugin unloading
        return False

    def list_plugins(self) -> List[str]:
        """
        List the names of all currently loaded plugins.

        Returns:
            A list of loaded plugin names (placeholder empty list).

        TODO: Return real loaded-plugin metadata.
        """
        # TODO: implement real plugin listing
        return []
