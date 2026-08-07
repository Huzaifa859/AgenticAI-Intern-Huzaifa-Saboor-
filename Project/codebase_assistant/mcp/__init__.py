"""
mcp
===

Model Context Protocol (MCP) integration layer.

Exposes Supervisor ToolRegistry tools through:

- MCPServer / MCPClient: local in-process transport (tests, notebook)
- stdio FastMCP bridge: official MCP protocol for external hosts

Agent and routing logic stay in the Supervisor.
"""

from .client import MCPClient
from .server import MCPServer, get_running_server
from .stdio_server import build_fastmcp, main, run_stdio

__all__ = [
    "MCPServer",
    "MCPClient",
    "get_running_server",
    "build_fastmcp",
    "run_stdio",
    "main",
]
