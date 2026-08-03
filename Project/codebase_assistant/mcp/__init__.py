"""
mcp
===

Model Context Protocol (MCP) integration layer.

Exposes Supervisor ToolRegistry tools through a local MCP server so
the same capabilities are reachable by MCP clients without changing
agent or routing logic.

Contains:
- MCPServer: publishes Codebase Assistant tools over a local MCP surface.
- MCPClient: consumes tools exposed by that server.
"""

from .client import MCPClient
from .server import MCPServer, get_running_server

__all__ = ["MCPServer", "MCPClient", "get_running_server"]
