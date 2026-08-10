"""
registry.py
===========

Defines the ToolRegistry, a central place where all tools (GitHub,
Filesystem, and future additions) are registered and looked up by
name.

NOTE: This only implements plain in-process tool registration/lookup.
MCP (Model Context Protocol) server-based tools are NOT implemented
yet — that will be added as a separate integration later.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..schemas.schemas import ToolCallRequest, ToolCallResult


class ToolRegistry:
    """
    Central registry for all tools available to agents.

    Tools are plain Python callables registered under a unique string
    name, stored in an internal dictionary, and can be looked up or
    invoked by that name. This decouples agents from the concrete
    implementation of each tool (GitHub API, filesystem access, etc).
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register_tool(self, name: str, handler: Callable[..., Any]) -> None:
        """
        Register a new tool under the given name.

        Args:
            name: Unique name identifying the tool. Used later to
                look up or invoke the tool.
            handler: Callable implementing the tool's behavior.

        Raises:
            ValueError: If a tool is already registered under `name`.
            TypeError: If `handler` is not callable.
        """
        if not callable(handler):
            raise TypeError(f"Tool handler for '{name}' must be callable.")
        if name in self._tools:
            raise ValueError(f"A tool named '{name}' is already registered.")
        self._tools[name] = handler

    def get_tool(self, name: str) -> Optional[Callable[..., Any]]:
        """
        Retrieve a registered tool handler by name.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            The tool handler callable if found, otherwise None.
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """
        List the names of all currently registered tools.

        Returns:
            A list of registered tool names, in registration order.

        TODO: Return richer metadata (descriptions, parameter schemas)
        alongside names once tools carry that information.
        """
        return list(self._tools.keys())

    def unregister_tool(self, name: str) -> None:
        """
        Remove a tool from the registry, if present.

        Args:
            name: Name of the tool to remove.
        """
        self._tools.pop(name, None)

    def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        """
        Invoke a registered tool with the given request.

        Args:
            request: A ToolCallRequest specifying the tool name and
                arguments.

        Returns:
            A ToolCallResult with the (placeholder) outcome of the call.

        TODO: Implement real dispatch (invoking the handler with
        request.arguments), argument validation, error handling, and
        result normalization. Not required yet — MCP-based tool
        invocation will likely replace/extend this.
        """
        # TODO: implement real tool dispatch logic
        return ToolCallResult(
            tool_name=request.tool_name,
            success=False,
            result=None,
            error="Not implemented",
        )
