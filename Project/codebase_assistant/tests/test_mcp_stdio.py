"""
test_mcp_stdio.py
=================

Unit tests for the stdio MCP bridge (FastMCP ↔ in-process MCPServer).

Does not open a real host stdio session — focuses on name mapping,
serialization, and agent tool forwarding with an offline Supervisor.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from codebase_assistant.config import Config
from codebase_assistant.mcp.server import MCPServer, _RUNNING_SERVERS
from codebase_assistant.mcp.stdio_server import (
    build_arg_parser,
    build_fastmcp,
    create_local_server,
    invoke_as_text,
    listed_protocol_tools,
    main,
    protocol_tool_name,
)
from codebase_assistant.supervisor import Supervisor


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Isolated config rooted at a temporary workspace."""
    return Config(
        workspace_root=str(tmp_path),
        memory_store_path=str(tmp_path / "memory"),
        chroma_persist_directory=str(tmp_path / "chroma"),
        github_token=None,
    )


@pytest.fixture
def supervisor(config: Config) -> Supervisor:
    """Supervisor with offline providers."""
    with patch.object(Supervisor, "_init_openrouter_provider", return_value=None), \
        patch.object(Supervisor, "_init_ollama_provider", return_value=None):
        return Supervisor(config=config)


@pytest.fixture
def server(supervisor: Supervisor) -> MCPServer:
    """Started in-process MCP server on a dedicated stdio test port."""
    _RUNNING_SERVERS.clear()
    instance = create_local_server(port=18766, supervisor=supervisor)
    started = instance.start()
    assert started["ok"] is True
    yield instance
    instance.shutdown()
    _RUNNING_SERVERS.clear()


def test_protocol_tool_name_maps_dots() -> None:
    """Dotted registry names become MCP-safe identifiers."""
    assert protocol_tool_name("analysis.run") == "analysis_run"
    assert protocol_tool_name("filesystem.read_file") == "filesystem_read_file"


def test_listed_protocol_tools_include_agents(server: MCPServer) -> None:
    """Agent pipelines are always advertised under protocol names."""
    names = listed_protocol_tools(server)
    for expected in (
        "analysis_run",
        "documentation_run",
        "testing_run",
        "goal_run",
    ):
        assert expected in names


def test_build_fastmcp_registers_agent_tools(server: MCPServer) -> None:
    """FastMCP app exposes the four agent tools by protocol name."""
    app = build_fastmcp(server)
    tools = asyncio.run(app.list_tools())
    names = {getattr(tool, "name", "") for tool in tools}
    assert "analysis_run" in names
    assert "documentation_run" in names
    assert "testing_run" in names
    assert "goal_run" in names


def test_invoke_as_text_unknown_tool(server: MCPServer) -> None:
    """Unknown tools serialize as ok=False JSON instead of raising."""
    text = invoke_as_text(server, "does.not.exist", {})
    payload = json.loads(text)
    assert payload["ok"] is False
    assert payload.get("code") in {"unknown_tool", "tool_error"}


def test_analysis_run_forwards_dotted_name(server: MCPServer) -> None:
    """analysis_run wrapper calls invoke_tool('analysis.run', ...)."""
    app = build_fastmcp(server)
    seen: dict = {}

    original = server.invoke_tool

    def _spy(name: str, arguments=None):  # type: ignore[no-untyped-def]
        seen["name"] = name
        seen["arguments"] = dict(arguments or {})
        return {
            "ok": True,
            "error": None,
            "code": None,
            "result": {"stub": True},
            "tool_name": name,
        }

    server.invoke_tool = _spy  # type: ignore[method-assign]
    try:
        result = asyncio.run(
            app.call_tool(
                "analysis_run",
                {
                    "repository": "C:/tmp/repo",
                    "question": "Find bugs",
                },
            )
        )
    finally:
        server.invoke_tool = original  # type: ignore[method-assign]

    assert seen["name"] == "analysis.run"
    assert seen["arguments"]["repository"] == "C:/tmp/repo"
    assert seen["arguments"]["question"] == "Find bugs"
    # FastMCP may wrap content; ensure our JSON payload is present.
    blob = json.dumps(result, default=str)
    assert "stub" in blob or '"ok": true' in blob.lower() or "True" in blob


def test_generic_mirror_uses_arguments_json(server: MCPServer) -> None:
    """Non-agent registry tools are mirrored with a JSON kwargs string."""
    # Register a simple extra tool on the local server.
    server.register_tool(
        "demo.echo",
        lambda message="": {"echo": message},
        description="Echo helper",
    )
    app = build_fastmcp(server, mirror_registry_tools=True)
    tools = asyncio.run(app.list_tools())
    names = {getattr(tool, "name", "") for tool in tools}
    assert "demo_echo" in names

    result = asyncio.run(
        app.call_tool(
            "demo_echo",
            {"arguments_json": json.dumps({"message": "hi"})},
        )
    )
    blob = json.dumps(result, default=str)
    assert "hi" in blob
    assert "echo" in blob


def test_cli_help_exits_zero() -> None:
    """``--help`` is available on the module CLI."""
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_cli_list_tools_prints_agents(
    supervisor: Supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--list-tools`` prints protocol names to stderr and exits 0."""
    _RUNNING_SERVERS.clear()
    with patch(
        "codebase_assistant.mcp.stdio_server.create_local_server",
        side_effect=lambda **kwargs: create_local_server(
            port=18767, supervisor=supervisor
        ),
    ):
        code = main(["--list-tools", "--port", "18767"])
    _RUNNING_SERVERS.clear()
    assert code == 0
    err = capsys.readouterr().err
    assert "analysis_run" in err
    assert "goal_run" in err
