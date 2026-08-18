"""
server.py
=========

Local MCP server foundation for Codebase Assistant.

Owns a Supervisor (and therefore its ToolRegistry), publishes every
registered tool automatically, and dispatches invocations through the
registry. Transport is in-process for this foundation layer so clients
can connect without a network dependency; the same surface can later
sit behind stdio/HTTP MCP.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..exceptions.tool_exceptions import (
    InvalidRepositoryURLError,
    RepositoryCloneError,
    ToolExecutionError,
)
from ..schemas.schemas import ToolCallRequest
from ..tracing.events import TraceEventType

if TYPE_CHECKING:
    from ..config import Config
    from ..supervisor import Supervisor
    from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

#: Agent pipelines exposed through MCP (registered onto ToolRegistry).
_AGENT_MCP_TOOLS = (
    "analysis.run",
    "documentation.run",
    "testing.run",
    "goal.run",
)

#: In-process directory of running servers keyed by ``host:port``.
_RUNNING_SERVERS: Dict[str, "MCPServer"] = {}
_RUNNING_LOCK = threading.RLock()


def _server_address(host: str, port: int) -> str:
    """Build the canonical local address key for a server."""
    return f"{(host or 'localhost').strip().lower()}:{int(port)}"


def get_running_server(address: str) -> Optional["MCPServer"]:
    """
    Look up a running MCPServer by address.

    Args:
        address: ``host:port`` or ``http(s)://host:port``.

    Returns:
        The running server, or None when nothing is bound there.
    """
    key = _normalize_address(address)
    with _RUNNING_LOCK:
        return _RUNNING_SERVERS.get(key)


def _normalize_address(address: str) -> str:
    """Normalize a client URL/address into ``host:port``."""
    text = (address or "").strip()
    if not text:
        return _server_address("localhost", 8000)
    lowered = text.lower()
    for prefix in ("http://", "https://", "mcp://"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.split("/", 1)[0].strip()
    if ":" not in text:
        return _server_address(text or "localhost", 8000)
    host, _, port_text = text.rpartition(":")
    try:
        port = int(port_text)
    except ValueError:
        port = 8000
    return _server_address(host or "localhost", port)


def _ok(result: Any = None, **extra: Any) -> Dict[str, Any]:
    """Build a successful structured MCP response."""
    payload: Dict[str, Any] = {
        "ok": True,
        "error": None,
        "code": None,
        "result": _json_safe(result),
    }
    payload.update(extra)
    return payload


def _err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    """Build a failed structured MCP response."""
    payload: Dict[str, Any] = {
        "ok": False,
        "error": str(message),
        "code": str(code),
        "result": None,
    }
    payload.update(extra)
    return payload


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of tool results into JSON-friendly data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return _json_safe(asdict(value))
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


class MCPServer:
    """
    Publishes Codebase Assistant tools over a local MCP interface.

    On startup the server constructs (or accepts) a Supervisor so the
    ToolRegistry is populated exactly as the CLI path populates it.
    Clients then list and invoke those tools without touching agents.
    """

    def __init__(
        self,
        name: str = "codebase-assistant",
        host: str = "localhost",
        port: int = 8000,
        *,
        config: Optional["Config"] = None,
        supervisor: Optional["Supervisor"] = None,
    ) -> None:
        """
        Initialize the MCP server.

        Args:
            name: Server name advertised to connecting clients.
            host: Host interface identity for local addressing.
            port: Port identity for local addressing.
            config: Optional Config used when constructing a Supervisor.
            supervisor: Optional pre-built Supervisor (tests / embedding).
        """
        self.name = name
        self.host = host
        self.port = int(port)
        self.config = config
        self._injected_supervisor = supervisor

        self.supervisor: Optional["Supervisor"] = None
        self.tool_registry: Optional["ToolRegistry"] = None
        self._running = False
        self._resources: Dict[str, str] = {}
        self._extra_tools: Dict[str, Callable[..., Any]] = {}
        self._registered_agent_tools: List[str] = []
        self._temp_clones: List[str] = []
        self._lock = threading.RLock()

    @property
    def address(self) -> str:
        """Canonical ``host:port`` address for this server."""
        return _server_address(self.host, self.port)

    @property
    def running(self) -> bool:
        """Whether the server has completed startup and is serving."""
        return self._running

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Expose an additional callable alongside ToolRegistry tools.

        Prefer registry tools. This exists for ad-hoc resources during
        tests or demos and never replaces Supervisor registration.

        Args:
            name: Tool name advertised to clients.
            handler: Callable implementing the tool.
            description: Optional human-readable description.

        Returns:
            Structured success/error payload.
        """
        try:
            if not name or not str(name).strip():
                return _err("invalid_arguments", "Tool name must be non-empty.")
            if not callable(handler):
                return _err("invalid_arguments", "Tool handler must be callable.")
            with self._lock:
                self._extra_tools[str(name).strip()] = handler
                if description:
                    self._resources[f"tool://{name}"] = description
            return _ok({"name": str(name).strip()})
        except Exception as exc:
            logger.warning("MCPServer.register_tool failed: %s", exc)
            return _err("internal_error", str(exc))

    def register_resource(self, uri: str, description: str = "") -> Dict[str, Any]:
        """
        Expose a readable resource descriptor.

        Args:
            uri: URI identifying the resource.
            description: Human-readable description.

        Returns:
            Structured success/error payload.
        """
        try:
            if not uri or not str(uri).strip():
                return _err("invalid_arguments", "Resource URI must be non-empty.")
            with self._lock:
                self._resources[str(uri).strip()] = description or ""
            return _ok({"uri": str(uri).strip()})
        except Exception as exc:
            logger.warning("MCPServer.register_resource failed: %s", exc)
            return _err("internal_error", str(exc))

    def start(self) -> Dict[str, Any]:
        """
        Start the server: create Supervisor, publish ToolRegistry tools.

        Returns:
            Structured success/error payload. Duplicate startup returns
            an error without crashing.
        """
        try:
            with self._lock:
                if self._running:
                    return _err(
                        "already_running",
                        f"MCP server already running at {self.address}.",
                        address=self.address,
                    )

                if self._injected_supervisor is not None:
                    self.supervisor = self._injected_supervisor
                else:
                    from ..config import Config
                    from ..supervisor import Supervisor

                    cfg = self.config or Config.load()
                    self.supervisor = Supervisor(config=cfg)

                self.tool_registry = self.supervisor.tool_registry
                self._register_agent_tools()
                self._running = True

            with _RUNNING_LOCK:
                existing = _RUNNING_SERVERS.get(self.address)
                if existing is not None and existing is not self and existing.running:
                    with self._lock:
                        self._unregister_agent_tools()
                        self._running = False
                        self.supervisor = None
                        self.tool_registry = None
                    return _err(
                        "address_in_use",
                        f"Another MCP server is already bound to {self.address}.",
                        address=self.address,
                    )
                _RUNNING_SERVERS[self.address] = self

            tool_count = len(self.list_tools())
            logger.info(
                "MCP server %r started at %s with %d tool(s).",
                self.name,
                self.address,
                tool_count,
            )
            return _ok(
                {
                    "name": self.name,
                    "address": self.address,
                    "tool_count": tool_count,
                }
            )
        except Exception as exc:
            logger.warning("MCPServer.start failed: %s", exc)
            with self._lock:
                self._running = False
                self.supervisor = None
                self.tool_registry = None
            return _err("startup_failed", str(exc))

    def stop(self) -> Dict[str, Any]:
        """Alias for ``shutdown`` kept for scaffolding compatibility."""
        return self.shutdown()

    def shutdown(self) -> Dict[str, Any]:
        """
        Stop the server and release its local address.

        Returns:
            Structured success/error payload. Shutdown without startup
            returns an error instead of raising.
        """
        try:
            with self._lock:
                if not self._running:
                    return _err(
                        "not_running",
                        "MCP server is not running.",
                        address=self.address,
                    )
                self._unregister_agent_tools()
                self._cleanup_temp_clones()
                self._running = False
                self.supervisor = None
                self.tool_registry = None

            with _RUNNING_LOCK:
                current = _RUNNING_SERVERS.get(self.address)
                if current is self:
                    _RUNNING_SERVERS.pop(self.address, None)

            return _ok({"address": self.address, "running": False})
        except Exception as exc:
            logger.warning("MCPServer.shutdown failed: %s", exc)
            return _err("shutdown_failed", str(exc))

    def health(self) -> Dict[str, Any]:
        """
        Report server health and basic capacity.

        Returns:
            Structured payload with running state and tool counts.
        """
        try:
            with self._lock:
                running = self._running
                registry = self.tool_registry
            names = []
            if running and registry is not None:
                try:
                    names = list(registry.list_tools())
                except Exception as exc:
                    return _err("health_failed", str(exc), running=running)
            return _ok(
                {
                    "name": self.name,
                    "address": self.address,
                    "running": running,
                    "tool_count": len(names) + (len(self._extra_tools) if running else 0),
                    "tools": names,
                }
            )
        except Exception as exc:
            logger.warning("MCPServer.health failed: %s", exc)
            return _err("health_failed", str(exc))

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List tools currently advertised by this server.

        Names come directly from ToolRegistry (plus any extras). Never
        raises — failures yield an empty list after logging.
        """
        try:
            with self._lock:
                if not self._running or self.tool_registry is None:
                    return []
                names = list(self.tool_registry.list_tools())
                extras = list(self._extra_tools.keys())
            descriptors = [
                {"name": name, "source": "ToolRegistry"} for name in names
            ]
            for name in extras:
                if name not in names:
                    descriptors.append({"name": name, "source": "MCPServer"})
            return descriptors
        except Exception as exc:
            logger.warning("MCPServer.list_tools failed: %s", exc)
            return []

    def invoke_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch a tool call through the ToolRegistry.

        Args:
            name: Registered tool name (e.g. ``filesystem.read_file``).
            arguments: Keyword arguments for the tool.

        Returns:
            Structured success/error payload. Unknown tools, bad
            arguments, and tool exceptions become ``ok=False`` results.
        """
        try:
            with self._lock:
                if not self._running:
                    return _err(
                        "not_running",
                        "MCP server is not running. Call start() first.",
                    )
                registry = self.tool_registry
                extra = self._extra_tools.get(name)

            if not name or not str(name).strip():
                return _err("invalid_arguments", "Tool name must be non-empty.")

            if arguments is not None and not isinstance(arguments, dict):
                return _err(
                    "invalid_arguments",
                    "arguments must be a mapping of keyword arguments.",
                )

            args = dict(arguments or {})

            if extra is not None and (registry is None or registry.get_tool(name) is None):
                try:
                    result = extra(**args)
                    return _ok(result, tool_name=name)
                except TypeError as exc:
                    return _err(
                        "invalid_arguments",
                        f"Invalid arguments for tool '{name}': {exc}",
                        tool_name=name,
                    )
                except Exception as exc:
                    return _err(
                        "tool_error",
                        f"Tool '{name}' raised {type(exc).__name__}: {exc}",
                        tool_name=name,
                    )

            if registry is None:
                return _err("not_running", "ToolRegistry is not available.")

            request = ToolCallRequest(tool_name=str(name).strip(), arguments=args)
            outcome = registry.call_tool(request)
            if outcome.success:
                return _ok(outcome.result, tool_name=outcome.tool_name)

            message = outcome.error or f"Tool '{name}' failed."
            code = "tool_error"
            lowered = message.lower()
            if "no tool registered" in lowered:
                code = "unknown_tool"
            elif "invalid arguments" in lowered:
                code = "invalid_arguments"
            elif "provider_unavailable" in lowered or "unavailable" in lowered:
                code = "provider_unavailable"
            elif "repository_not_found" in lowered or "does not exist" in lowered:
                code = "repository_not_found"
            elif "unknown task" in lowered or "unknown agent" in lowered:
                code = "unknown_agent"
            elif "supervisor" in lowered:
                code = "supervisor_error"
            return _err(code, message, tool_name=name)
        except Exception as exc:
            logger.warning("MCPServer.invoke_tool failed: %s", exc)
            return _err("internal_error", str(exc), tool_name=name)

    # ------------------------------------------------------------------
    # Agent MCP tools (Supervisor pipelines)
    # ------------------------------------------------------------------

    def _register_agent_tools(self) -> None:
        """Register analysis/documentation/testing/goal tools on ToolRegistry."""
        if self.tool_registry is None:
            return
        handlers = {
            "analysis.run": self._tool_analysis_run,
            "documentation.run": self._tool_documentation_run,
            "testing.run": self._tool_testing_run,
            "goal.run": self._tool_goal_run,
        }
        for name, handler in handlers.items():
            if self.tool_registry.get_tool(name) is not None:
                continue
            try:
                self.tool_registry.register_tool(name, handler)
                self._registered_agent_tools.append(name)
            except Exception as exc:
                logger.warning("Could not register MCP agent tool %s: %s", name, exc)

    def _unregister_agent_tools(self) -> None:
        """Remove agent tools registered by this server instance."""
        if self.tool_registry is None:
            self._registered_agent_tools.clear()
            return
        for name in list(self._registered_agent_tools):
            try:
                self.tool_registry.unregister_tool(name)
            except Exception as exc:
                logger.warning("Could not unregister MCP agent tool %s: %s", name, exc)
        self._registered_agent_tools.clear()

    def _cleanup_temp_clones(self) -> None:
        """Delete temporary GitHub clones created for MCP agent calls."""
        while self._temp_clones:
            path = self._temp_clones.pop()
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception as exc:
                logger.warning("Failed cleaning MCP clone %s: %s", path, exc)

    def _trace_mcp(
        self,
        name: str,
        *,
        success: Optional[bool] = True,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Record an MCP lifecycle event on the Supervisor Tracer."""
        if self.supervisor is None:
            return
        tracer = getattr(self.supervisor, "tracer", None)
        if tracer is None:
            return
        try:
            tracer.record(
                TraceEventType.LIFECYCLE,
                name,
                component="MCPServer",
                success=success,
                duration_ms=duration_ms,
                error=error,
                **metadata,
            )
        except Exception as exc:
            logger.warning("MCP tracing failed for %r: %s", name, exc)

    def _prepare_repository(self, repository: str) -> str:
        """
        Resolve a local path or clone a GitHub URL via Supervisor tools.

        Args:
            repository: Local path or GitHub HTTPS URL.

        Returns:
            Absolute local repository path.

        Raises:
            ToolExecutionError: Invalid/missing repository arguments.
            InvalidRepositoryURLError / RepositoryCloneError: Clone/path
                failures from GitHubTools.
        """
        if self.supervisor is None:
            raise ToolExecutionError("supervisor: MCP server has no Supervisor.")
        reference = str(repository or "").strip()
        if not reference:
            raise ToolExecutionError(
                "invalid arguments: 'repository' is required."
            )

        github = self.supervisor.github_tools
        if github.is_remote_reference(reference):
            github.validate_repository(reference)
            temporary_root = tempfile.mkdtemp(prefix="mcp_codebase_clone_")
            self._temp_clones.append(temporary_root)
            destination = os.path.join(temporary_root, "repo")
            github.clone_repository(reference, destination)
            return destination

        path = os.path.abspath(os.path.expanduser(reference))
        if not os.path.exists(path):
            raise ToolExecutionError(
                f"repository_not_found: path does not exist: {path}"
            )
        if not os.path.isdir(path):
            raise ToolExecutionError(
                f"repository_not_found: path is not a directory: {path}"
            )
        return path

    def _raise_from_response(self, response: Any, action: str) -> None:
        """Convert a failed AgentResponse into a ToolExecutionError."""
        errors = list(getattr(response, "errors", None) or [])
        message = "; ".join(str(item) for item in errors) or f"{action} failed."
        lowered = message.lower()
        if "unknown task" in lowered:
            raise ToolExecutionError(f"unknown agent: {message}")
        if "unavailable" in lowered:
            raise ToolExecutionError(f"provider_unavailable: {message}")
        raise ToolExecutionError(f"supervisor: {message}")

    def _run_supervisor_task(
        self,
        *,
        tool_name: str,
        repository: str,
        task_name: str,
    ) -> Any:
        """
        Prepare a repository and run Supervisor.handle_task.

        Returns the AgentResponse.output on success.
        """
        started = time.perf_counter()
        self._trace_mcp(
            "mcp_request",
            tool=tool_name,
            repository=repository,
            task_name=task_name,
        )
        self._trace_mcp("tool_invoked", tool=tool_name, task_name=task_name)
        try:
            if self.supervisor is None:
                raise ToolExecutionError("supervisor: MCP server has no Supervisor.")
            repo_path = self._prepare_repository(repository)
            response = self.supervisor.handle_task(task_name, repo_path)
            if not getattr(response, "success", False):
                self._raise_from_response(response, tool_name)
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._trace_mcp(
                "mcp_response",
                success=True,
                duration_ms=duration_ms,
                tool=tool_name,
            )
            return response.output
        except (InvalidRepositoryURLError, RepositoryCloneError) as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._trace_mcp(
                "mcp_response",
                success=False,
                duration_ms=duration_ms,
                tool=tool_name,
                error=str(exc),
            )
            raise ToolExecutionError(f"repository_not_found: {exc}") from exc
        except ToolExecutionError as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._trace_mcp(
                "mcp_response",
                success=False,
                duration_ms=duration_ms,
                tool=tool_name,
                error=str(exc),
            )
            raise
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._trace_mcp(
                "mcp_response",
                success=False,
                duration_ms=duration_ms,
                tool=tool_name,
                error=str(exc),
            )
            raise ToolExecutionError(f"supervisor: {exc}") from exc

    def _tool_analysis_run(
        self,
        repository: str = "",
        question: str = "Find likely bugs and correctness problems in this code.",
    ) -> Any:
        """MCP tool: run Code Analysis through Supervisor.handle_task."""
        question_text = str(question or "").strip() or (
            "Find likely bugs and correctness problems in this code."
        )
        # Keep routing keyword while forwarding the user question as instruction.
        task_name = f"analysis: {question_text}"
        return self._run_supervisor_task(
            tool_name="analysis.run",
            repository=repository,
            task_name=task_name,
        )

    def _tool_documentation_run(
        self,
        repository: str = "",
        target: str = "README",
    ) -> Any:
        """MCP tool: run Documentation through Supervisor.handle_task."""
        target_text = str(target or "README").strip() or "README"
        task_name = f"documentation {target_text}"
        return self._run_supervisor_task(
            tool_name="documentation.run",
            repository=repository,
            task_name=task_name,
        )

    def _tool_testing_run(
        self,
        repository: str = "",
        target: str = "",
    ) -> Any:
        """MCP tool: run Testing through Supervisor.handle_task."""
        target_text = str(target or "").strip()
        task_name = (
            f"testing {target_text}" if target_text else "testing"
        )
        return self._run_supervisor_task(
            tool_name="testing.run",
            repository=repository,
            task_name=task_name,
        )

    def _tool_goal_run(
        self,
        repository: str = "",
        goal: str = "",
    ) -> Any:
        """MCP tool: run Supervisor.handle_goal for a multi-agent goal."""
        started = time.perf_counter()
        goal_text = str(goal or "").strip()
        self._trace_mcp(
            "mcp_request",
            tool="goal.run",
            repository=repository,
            goal=goal_text,
        )
        self._trace_mcp("tool_invoked", tool="goal.run")
        try:
            if not goal_text:
                raise ToolExecutionError(
                    "invalid arguments: 'goal' is required."
                )
            if self.supervisor is None:
                raise ToolExecutionError("supervisor: MCP server has no Supervisor.")
            repo_path = self._prepare_repository(repository)
            responses = self.supervisor.handle_goal(goal_text, repo_path=repo_path)
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._trace_mcp(
                "mcp_response",
                success=True,
                duration_ms=duration_ms,
                tool="goal.run",
                response_count=len(responses or []),
            )
            return list(responses or [])
        except (InvalidRepositoryURLError, RepositoryCloneError) as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._trace_mcp(
                "mcp_response",
                success=False,
                duration_ms=duration_ms,
                tool="goal.run",
                error=str(exc),
            )
            raise ToolExecutionError(f"repository_not_found: {exc}") from exc
        except ToolExecutionError as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._trace_mcp(
                "mcp_response",
                success=False,
                duration_ms=duration_ms,
                tool="goal.run",
                error=str(exc),
            )
            raise
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._trace_mcp(
                "mcp_response",
                success=False,
                duration_ms=duration_ms,
                tool="goal.run",
                error=str(exc),
            )
            raise ToolExecutionError(f"supervisor: {exc}") from exc
