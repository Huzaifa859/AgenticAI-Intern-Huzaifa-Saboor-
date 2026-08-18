"""
registry.py
===========

Central registry where all tools (GitHub, Filesystem, and future
additions) are registered, looked up, and invoked by name.

Tools are plain Python callables. Keeping invocation behind
`call_tool` means agents never hold a reference to a concrete tool
implementation, so a tool can be replaced -- eventually by an
MCP-hosted equivalent -- without touching calling code.

`call_tool` reports failure by returning an unsuccessful ToolCallResult
rather than raising. A tool call is an expected part of an agent's
reasoning loop, and a missing tool or a bad argument should be something
the agent can read and recover from, not an exception that unwinds the
run.

NOTE: This is plain in-process registration. MCP server-based tools are
not implemented yet.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..exceptions.tool_exceptions import ToolExecutionError, ToolNotFoundError
from ..schemas.schemas import ToolCallRequest, ToolCallResult


class ToolRegistry:
    """
    Central registry for all tools available to agents.

    Tools are registered under a unique string name, stored in an
    internal dictionary, and can be looked up or invoked by that name.
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
            ValueError: If `name` is empty, or a tool is already
                registered under it.
            TypeError: If `handler` is not callable.
        """
        if not name or not str(name).strip():
            raise ValueError("Tool name must be a non-empty string.")
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
            A ToolCallResult. On success, `result` carries the tool's
            return value. On failure, `error` explains what went wrong --
            an unknown tool lists the names that are available, and a
            signature mismatch reports the argument error verbatim.
        """
        name = request.tool_name

        try:
            handler = self._require_tool(name)
        except ToolNotFoundError as exc:
            return ToolCallResult(
                tool_name=name, success=False, result=None, error=str(exc)
            )

        arguments = request.arguments or {}

        try:
            result = handler(**arguments)
        except TypeError as exc:
            # Raised when the supplied arguments do not match the
            # handler's signature, which is the most common caller error.
            return ToolCallResult(
                tool_name=name,
                success=False,
                result=None,
                error=f"Invalid arguments for tool '{name}': {exc}",
            )
        except Exception as exc:
            # Deliberately broad: any tool failure becomes a readable
            # result rather than propagating out of the agent's loop.
            wrapped = ToolExecutionError(
                f"Tool '{name}' raised {type(exc).__name__}: {exc}"
            )
            return ToolCallResult(
                tool_name=name, success=False, result=None, error=str(wrapped)
            )

        return ToolCallResult(
            tool_name=name, success=True, result=result, error=None
        )

    def _require_tool(self, name: str) -> Callable[..., Any]:
        """
        Look up a tool, failing loudly when it is absent.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            The registered handler.

        Raises:
            ToolNotFoundError: If no tool is registered under `name`.
                The message lists the registered names, so a typo is
                obvious from the error alone.
        """
        handler = self._tools.get(name)
        if handler is None:
            available = ", ".join(sorted(self._tools)) or "none"
            raise ToolNotFoundError(
                f"No tool registered under '{name}'. Available tools: {available}."
            )
        return handler
