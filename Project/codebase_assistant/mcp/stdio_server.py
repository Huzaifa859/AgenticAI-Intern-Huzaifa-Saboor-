"""
stdio_server.py
===============

Official MCP stdio transport bridged to the in-process MCPServer.

External hosts (Cursor, Claude Desktop, etc.) speak JSON-RPC on stdio.
This module starts a local ``MCPServer`` (Supervisor + ToolRegistry) and
exposes its tools through the Python MCP SDK FastMCP server. Agent logic
stays in the Supervisor — this file is transport only.

Stdout is reserved for the MCP wire protocol. All logging goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .server import MCPServer, _json_safe

logger = logging.getLogger(__name__)

#: Local address identity for the in-process server (not a network bind).
_STDIO_HOST = "localhost"
_STDIO_PORT = 18765

#: Stable public agent tools: protocol name -> local ToolRegistry name.
_AGENT_TOOL_MAP: Tuple[Tuple[str, str, str], ...] = (
    (
        "analysis_run",
        "analysis.run",
        "Run code analysis on a local path or GitHub HTTPS URL.",
    ),
    (
        "documentation_run",
        "documentation.run",
        "Generate documentation for a repository target (README, file, etc.).",
    ),
    (
        "testing_run",
        "testing.run",
        "Generate and optionally execute pytest tests for a repository target.",
    ),
    (
        "goal_run",
        "goal.run",
        "Run a multi-agent goal through the Supervisor.",
    ),
)

_AGENT_LOCAL_NAMES = {local for _, local, _ in _AGENT_TOOL_MAP}


def protocol_tool_name(local_name: str) -> str:
    """Map a dotted ToolRegistry name to an MCP-safe protocol name."""
    text = (local_name or "").strip()
    return text.replace(".", "_").replace("-", "_")


def configure_stdio_logging(level: int = logging.INFO) -> None:
    """Send logging to stderr so stdout stays free for MCP JSON-RPC."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy third-party loggers on the MCP wire process.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def invoke_as_text(
    server: MCPServer,
    local_name: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Invoke a local MCPServer tool and return a JSON string for hosts.

    Never raises into the MCP SDK — failures are encoded in the payload.
    """
    payload = server.invoke_tool(local_name, dict(arguments or {}))
    try:
        return json.dumps(_json_safe(payload), ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps(
            {"ok": False, "error": "Failed to serialize tool result.", "code": "internal_error"},
            ensure_ascii=False,
        )


def _parse_arguments_json(arguments_json: str) -> Dict[str, Any]:
    """Parse a JSON object string into kwargs; empty becomes {}."""
    text = (arguments_json or "").strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments_json must be a JSON object: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("arguments_json must decode to a JSON object.")
    return dict(loaded)


def build_fastmcp(
    server: MCPServer,
    *,
    mirror_registry_tools: bool = False,
) -> Any:
    """
    Build a FastMCP app that forwards tools to ``server.invoke_tool``.

    Args:
        server: A started in-process MCPServer.
        mirror_registry_tools: When True, also expose filesystem/GitHub
            registry tools via ``arguments_json``. Default False keeps the
            Claude/Cursor surface small (agent pipelines only).
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "The 'mcp' package is required for stdio transport. "
            "Install with: pip install 'mcp>=1.28,<2'"
        ) from exc

    if not server.running:
        raise RuntimeError("MCPServer must be started before building FastMCP.")

    app = FastMCP(
        name=server.name or "codebase-assistant",
        instructions=(
            "Codebase Assistant MCP tools. Pass a local repository path or "
            "GitHub HTTPS URL as 'repository'. Long agent runs may take minutes."
        ),
    )

    # --- Stable agent surface -------------------------------------------------
    @app.tool(
        name="analysis_run",
        description=_AGENT_TOOL_MAP[0][2],
    )
    def analysis_run(
        repository: str,
        question: str = "Find likely bugs and correctness problems in this code.",
    ) -> str:
        return invoke_as_text(
            server,
            "analysis.run",
            {"repository": repository, "question": question},
        )

    @app.tool(
        name="documentation_run",
        description=_AGENT_TOOL_MAP[1][2],
    )
    def documentation_run(repository: str, target: str = "README") -> str:
        return invoke_as_text(
            server,
            "documentation.run",
            {"repository": repository, "target": target},
        )

    @app.tool(
        name="testing_run",
        description=_AGENT_TOOL_MAP[2][2],
    )
    def testing_run(repository: str, target: str = "") -> str:
        return invoke_as_text(
            server,
            "testing.run",
            {"repository": repository, "target": target},
        )

    @app.tool(
        name="goal_run",
        description=_AGENT_TOOL_MAP[3][2],
    )
    def goal_run(repository: str, goal: str) -> str:
        return invoke_as_text(
            server,
            "goal.run",
            {"repository": repository, "goal": goal},
        )

    # --- Optional mirror of remaining registry tools -------------------------
    if mirror_registry_tools:
        registered_protocol = {proto for proto, _, _ in _AGENT_TOOL_MAP}
        for descriptor in server.list_tools():
            local_name = str(descriptor.get("name") or "").strip()
            if not local_name or local_name in _AGENT_LOCAL_NAMES:
                continue
            proto_name = protocol_tool_name(local_name)
            if not proto_name or proto_name in registered_protocol:
                logger.warning(
                    "Skipping MCP mirror for %r (name collision or empty).",
                    local_name,
                )
                continue
            _register_generic_tool(app, server, proto_name, local_name)
            registered_protocol.add(proto_name)

    return app


def _register_generic_tool(
    app: Any,
    server: MCPServer,
    proto_name: str,
    local_name: str,
) -> None:
    """
    Register a mirrored registry tool that accepts a JSON kwargs object.

    Awkward callables stay usable without inventing per-tool schemas.
    """

    def _handler(arguments_json: str = "{}") -> str:
        try:
            args = _parse_arguments_json(arguments_json)
        except ValueError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "code": "invalid_arguments",
                    "tool_name": local_name,
                },
                ensure_ascii=False,
                indent=2,
            )
        return invoke_as_text(server, local_name, args)

    _handler.__name__ = proto_name
    _handler.__doc__ = (
        f"Invoke ToolRegistry tool '{local_name}'. "
        "Pass keyword arguments as a JSON object string in arguments_json."
    )
    app.add_tool(
        _handler,
        name=proto_name,
        description=_handler.__doc__,
    )


def listed_protocol_tools(
    server: MCPServer,
    *,
    mirror_registry_tools: bool = False,
) -> List[str]:
    """
    Return protocol tool names that ``build_fastmcp`` would expose.

    Useful for tests and ``--help`` without starting the stdio loop.
    """
    names = [proto for proto, _, _ in _AGENT_TOOL_MAP]
    seen = set(names)
    if not mirror_registry_tools or not server.running:
        return names
    for descriptor in server.list_tools():
        local_name = str(descriptor.get("name") or "").strip()
        if not local_name or local_name in _AGENT_LOCAL_NAMES:
            continue
        proto_name = protocol_tool_name(local_name)
        if proto_name and proto_name not in seen:
            names.append(proto_name)
            seen.add(proto_name)
    return names


def create_local_server(
    *,
    host: str = _STDIO_HOST,
    port: int = _STDIO_PORT,
    config: Any = None,
    supervisor: Any = None,
) -> MCPServer:
    """Construct an in-process MCPServer for stdio bridging."""
    return MCPServer(
        name="codebase-assistant",
        host=host,
        port=port,
        config=config,
        supervisor=supervisor,
    )


def run_stdio(
    *,
    config: Any = None,
    supervisor: Any = None,
    host: str = _STDIO_HOST,
    port: int = _STDIO_PORT,
) -> None:
    """
    Start the local MCPServer and serve tools over MCP stdio.

    Blocks until the host disconnects or the process is interrupted.
    """
    configure_stdio_logging()
    server = create_local_server(
        host=host, port=port, config=config, supervisor=supervisor
    )
    started = server.start()
    if not started.get("ok"):
        message = started.get("error") or started.get("message") or "MCPServer.start failed"
        logger.error("Failed to start local MCP server: %s", message)
        raise RuntimeError(str(message))

    mirror = os.environ.get("MCP_MIRROR_REGISTRY_TOOLS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    tools = listed_protocol_tools(server, mirror_registry_tools=mirror)
    logger.info(
        "Stdio MCP ready (%d tool(s)): %s",
        len(tools),
        ", ".join(tools[:12]) + ("..." if len(tools) > 12 else ""),
    )

    app = build_fastmcp(server, mirror_registry_tools=mirror)
    try:
        app.run(transport="stdio")
    finally:
        shutdown = server.shutdown()
        if not shutdown.get("ok"):
            logger.warning("MCPServer shutdown: %s", shutdown.get("error"))


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m codebase_assistant.mcp``."""
    parser = argparse.ArgumentParser(
        prog="python -m codebase_assistant.mcp",
        description=(
            "Run Codebase Assistant as an MCP stdio server for Cursor / "
            "Claude Desktop and other MCP hosts. Logging goes to stderr; "
            "stdout is the MCP wire protocol."
        ),
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Start the local server, print protocol tool names to stderr, then exit.",
    )
    parser.add_argument(
        "--host",
        default=_STDIO_HOST,
        help="In-process server host identity (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_STDIO_PORT,
        help="In-process server port identity (default: %(default)s).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint used by ``python -m codebase_assistant.mcp``."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_tools:
        configure_stdio_logging()
        server = create_local_server(host=args.host, port=args.port)
        started = server.start()
        try:
            if not started.get("ok"):
                print(started.get("error") or "start failed", file=sys.stderr)
                return 1
            for name in listed_protocol_tools(server):
                print(name, file=sys.stderr)
            return 0
        finally:
            server.shutdown()

    try:
        run_stdio(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down.")
        return 0
    except Exception as exc:
        logger.error("Stdio MCP failed: %s", exc)
        return 1
    return 0
