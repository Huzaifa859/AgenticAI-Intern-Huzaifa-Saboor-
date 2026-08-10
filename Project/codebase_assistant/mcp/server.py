"""
server.py
=========

Placeholder for the MCP server that publishes Codebase Assistant's
tools and resources over the Model Context Protocol.

Per the proposal, the filesystem tools (`read_file`, `list_files`,
`search_codebase`) are the first capabilities to be exposed here.

TODO: Implement real MCP support — transport (stdio/HTTP), handshake,
tool and resource advertisement, request handling, and error responses.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


class MCPServer:
    """
    Publishes Codebase Assistant tools and resources over MCP.

    Intended to sit alongside the in-process ToolRegistry rather than
    replace it: the registry stays the canonical list of capabilities,
    while this server exposes a chosen subset to external MCP clients.
    """

    def __init__(
        self,
        name: str = "codebase-assistant",
        host: str = "localhost",
        port: int = 8000,
    ) -> None:
        """
        Initialize the MCP server.

        Args:
            name: Server name advertised to connecting clients.
            host: Host interface to bind to.
            port: Port to listen on.
        """
        self.name = name
        self.host = host
        self.port = port

        # Intended backing state for advertised capabilities.
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._resources: Dict[str, str] = {}

    def register_tool(self, name: str, handler: Callable[..., Any], description: str = "") -> None:
        """
        Expose a callable as an MCP tool.

        Args:
            name: Tool name advertised to clients.
            handler: Callable implementing the tool.
            description: Human-readable description sent to clients.

        TODO: Implement real MCP tool registration, including the
        JSON-Schema parameter definition each tool must advertise.
        """
        # TODO: implement real MCP tool registration
        pass

    def register_resource(self, uri: str, description: str = "") -> None:
        """
        Expose a readable resource (e.g. a source file) over MCP.

        Args:
            uri: URI identifying the resource.
            description: Human-readable description sent to clients.

        TODO: Implement real MCP resource registration and content
        resolution.
        """
        # TODO: implement real MCP resource registration
        pass

    def list_tools(self) -> List[str]:
        """
        List the names of all tools currently advertised by this server.

        Returns:
            A list of tool names (placeholder empty list).

        TODO: Return real advertised tool metadata once registration
        is implemented.
        """
        # TODO: implement real tool listing
        return []

    def start(self) -> None:
        """
        Start serving MCP requests.

        TODO: Implement real server startup and the request loop.
        """
        # TODO: implement real server startup
        pass

    def stop(self) -> None:
        """
        Stop the server and release its transport.

        TODO: Implement real shutdown and cleanup.
        """
        # TODO: implement real server shutdown
        pass
