"""
plugins
=======

Extension point for optional, self-contained features that can be added
without modifying the core system.

This is where the proposal's stretch goals are meant to land — the
Refactoring Agent and import-graph architecture diagram generation —
so that shipping or dropping them never touches the guaranteed MVP.

Contains:
- BasePlugin: abstract interface every plugin implements.
- PluginManager: discovery, loading, and lifecycle management.

NOTE: Placeholder only. No plugin is implemented and no discovery or
loading happens yet.
"""

from .base import BasePlugin
from .manager import PluginManager

__all__ = ["BasePlugin", "PluginManager"]
