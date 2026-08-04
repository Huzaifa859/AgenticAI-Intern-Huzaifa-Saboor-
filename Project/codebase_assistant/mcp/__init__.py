"""
mcp
===

Model Context Protocol (MCP) integration layer.

The proposal exposes the filesystem tools (`read_file`, `list_files`,
`search_codebase`) to agents as MCP tools/resources rather than only as
in-process callables, so the same capabilities are reachable by any
MCP-speaking client.

Contains:
- MCPServer: publishes Codebase Assistant tools/resources over MCP.
- MCPClient: consumes tools/resources exposed by an MCP server.

NOTE: Placeholder only. No transport, handshake, protocol handling, or
tool advertisement is implemented yet.
"""

from .client import MCPClient
from .server import MCPServer

__all__ = ["MCPServer", "MCPClient"]
