"""
client.py
=========

Local MCP client for Codebase Assistant.

Talks only to an in-process MCPServer registered at a host:port
address. This foundation matches the server's local transport so tests
and the notebook can exercise tool discovery/invocation without a
network stack.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .server import _err, _normalize_address, _ok, get_running_server

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Connects to a local MCP server and invokes its tools.

    Intended to consume capabilities published by ``MCPServer``, which
    itself mirrors the Supervisor ToolRegistry.
    """

    def __init__(self, server_url: str = "localhost:8000", timeout: float = 30.0) -> None:
        """
        Initialize the MCP client.

        Args:
            server_url: Address of the MCP server (``host:port`` or URL).
            timeout: Per-request timeout in seconds (reserved for future
                remote transports; local calls are in-process).
        """
        self.server_url = server_url or "localhost:8000"
        self.timeout = float(timeout)
        self.connected = False
        self._server = None

    def connect(self) -> Dict[str, Any]:
        """
        Establish a connection to the configured local MCP server.

        Returns:
            Structured success/error payload. Sets ``connected`` only on
            success.
        """
        try:
            address = _normalize_address(self.server_url)
            server = get_running_server(address)
            if server is None or not server.running:
                self.connected = False
                self._server = None
                return _err(
                    "connection_failure",
                    f"No running MCP server at {address}.",
                    address=address,
                )
            self._server = server
            self.connected = True
            return _ok({"address": address, "connected": True})
        except Exception as exc:
            logger.warning("MCPClient.connect failed: %s", exc)
            self.connected = False
            self._server = None
            return _err("connection_failure", str(exc))

    def disconnect(self) -> Dict[str, Any]:
        """
        Close the connection to the MCP server.

        Returns:
            Structured success payload. Safe to call when not connected.
        """
        try:
            self._server = None
            self.connected = False
            return _ok({"connected": False})
        except Exception as exc:
            logger.warning("MCPClient.disconnect failed: %s", exc)
            self.connected = False
            self._server = None
            return _err("disconnect_failed", str(exc))

    def health(self) -> Dict[str, Any]:
        """
        Query the connected server's health endpoint.

        Returns:
            Structured health payload, or a connection error.
        """
        try:
            server = self._require_server()
            if isinstance(server, dict):
                return server
            return server.health()
        except Exception as exc:
            logger.warning("MCPClient.health failed: %s", exc)
            return _err("health_failed", str(exc))

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Discover the tools advertised by the connected server.

        Returns:
            A list of tool descriptors. Connection failures yield ``[]``.
        """
        try:
            server = self._require_server()
            if isinstance(server, dict):
                return []
            tools = server.list_tools()
            return list(tools or [])
        except Exception as exc:
            logger.warning("MCPClient.list_tools failed: %s", exc)
            return []

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke a tool on the connected MCP server.

        Args:
            name: Name of the remote tool to invoke.
            arguments: Keyword arguments to pass to the tool.

        Returns:
            Structured success/error payload from the server.
        """
        try:
            server = self._require_server()
            if isinstance(server, dict):
                return server
            return server.invoke_tool(name, arguments)
        except Exception as exc:
            logger.warning("MCPClient.call_tool failed: %s", exc)
            return _err("call_failed", str(exc), tool_name=name)

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource exposed by the connected MCP server.

        Args:
            uri: URI of the resource to read.

        Returns:
            Structured payload. Foundation resources are descriptors
            only; missing URIs return an error.
        """
        try:
            server = self._require_server()
            if isinstance(server, dict):
                return server
            resources = getattr(server, "_resources", {}) or {}
            if uri not in resources:
                return _err("not_found", f"Resource not found: {uri}", uri=uri)
            return _ok({"uri": uri, "description": resources.get(uri, "")})
        except Exception as exc:
            logger.warning("MCPClient.read_resource failed: %s", exc)
            return _err("read_failed", str(exc), uri=uri)

    def _require_server(self):
        """
        Return the connected server or a structured connection error.
        """
        if not self.connected or self._server is None:
            return _err(
                "not_connected",
                "MCP client is not connected. Call connect() first.",
            )
        if not getattr(self._server, "running", False):
            self.connected = False
            self._server = None
            return _err(
                "connection_failure",
                "Connected MCP server is no longer running.",
            )
        return self._server
