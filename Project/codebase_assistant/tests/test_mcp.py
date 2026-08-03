"""
test_mcp.py
============

Unit tests for the local MCP server/client foundation.

Supervisor construction is stubbed lightly (providers only) so tests
stay offline. ToolRegistry remains real and is populated by Supervisor.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from codebase_assistant.config import Config
from codebase_assistant.mcp.client import MCPClient
from codebase_assistant.mcp.server import MCPServer, _RUNNING_SERVERS
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
    """Fresh MCP server bound to a unique local port."""
    # Clear any leftover registry entries from prior failures.
    _RUNNING_SERVERS.clear()
    instance = MCPServer(
        host="localhost",
        port=18000,
        supervisor=supervisor,
    )
    yield instance
    instance.shutdown()
    _RUNNING_SERVERS.clear()


def test_server_starts(server: MCPServer) -> None:
    """Startup creates a running server with registry tools."""
    result = server.start()
    assert result["ok"] is True
    assert server.running is True
    assert result["result"]["tool_count"] > 0


def test_client_connects(server: MCPServer) -> None:
    """Client connects to a running local server."""
    assert server.start()["ok"] is True
    client = MCPClient(server_url="localhost:18000")
    result = client.connect()
    assert result["ok"] is True
    assert client.connected is True


def test_health_succeeds(server: MCPServer) -> None:
    """Health reports running state and tool inventory."""
    server.start()
    client = MCPClient("localhost:18000")
    client.connect()
    health = client.health()
    assert health["ok"] is True
    assert health["result"]["running"] is True
    assert health["result"]["tool_count"] >= 1


def test_tool_list_returned(server: MCPServer) -> None:
    """list_tools reflects ToolRegistry names automatically."""
    server.start()
    client = MCPClient("localhost:18000")
    client.connect()
    tools = client.list_tools()
    names = {item["name"] for item in tools}
    assert "filesystem.read_file" in names
    assert "filesystem.write_file" in names
    assert "filesystem.list_files" in names
    assert "github.clone_repository" in names
    assert "github.validate_repository" in names
    assert "github.list_issues" in names
    assert "github.create_pull_request" in names


def test_invoke_filesystem_tool(server: MCPServer, tmp_path: Path) -> None:
    """Filesystem tools execute through MCP invoke/call."""
    server.start()
    client = MCPClient("localhost:18000")
    client.connect()

    target = tmp_path / "note.txt"
    write = client.call_tool(
        "filesystem.write_file",
        {"path": str(target), "content": "hello mcp"},
    )
    assert write["ok"] is True

    read = client.call_tool(
        "filesystem.read_file",
        {"path": str(target)},
    )
    assert read["ok"] is True
    assert read["result"] == "hello mcp"


def test_invoke_github_tool(server: MCPServer, tmp_path: Path) -> None:
    """GitHub validate_repository works for a local path via MCP."""
    server.start()
    client = MCPClient("localhost:18000")
    client.connect()

    result = client.call_tool(
        "github.validate_repository",
        {"repo_url": str(tmp_path)},
    )
    assert result["ok"] is True
    assert result["result"] is True


def test_unknown_tool(server: MCPServer) -> None:
    """Unknown tools return structured unknown_tool errors."""
    server.start()
    client = MCPClient("localhost:18000")
    client.connect()
    result = client.call_tool("does.not.exist", {})
    assert result["ok"] is False
    assert result["code"] == "unknown_tool"


def test_invalid_arguments(server: MCPServer) -> None:
    """Bad tool arguments return structured invalid_arguments errors."""
    server.start()
    client = MCPClient("localhost:18000")
    client.connect()
    result = client.call_tool("filesystem.read_file", {"nope": 1})
    assert result["ok"] is False
    assert result["code"] == "invalid_arguments"


def test_shutdown(server: MCPServer) -> None:
    """Shutdown stops the server; double shutdown is structured."""
    assert server.start()["ok"] is True
    stopped = server.shutdown()
    assert stopped["ok"] is True
    assert server.running is False
    again = server.shutdown()
    assert again["ok"] is False
    assert again["code"] == "not_running"


def test_reconnect(server: MCPServer) -> None:
    """Client can disconnect and connect again to a running server."""
    server.start()
    client = MCPClient("localhost:18000")
    first = client.connect()
    assert first["ok"] is True
    client.disconnect()
    assert client.connected is False
    second = client.connect()
    assert second["ok"] is True
    assert client.connected is True
    health = client.health()
    assert health["ok"] is True


def test_duplicate_startup(server: MCPServer) -> None:
    """Starting twice returns already_running without crashing."""
    assert server.start()["ok"] is True
    again = server.start()
    assert again["ok"] is False
    assert again["code"] == "already_running"


def test_shutdown_without_startup() -> None:
    """Shutdown before start is a structured error."""
    _RUNNING_SERVERS.clear()
    idle = MCPServer(host="localhost", port=18001)
    result = idle.shutdown()
    assert result["ok"] is False
    assert result["code"] == "not_running"


def test_connection_failure_without_server() -> None:
    """Connecting when nothing is listening fails cleanly."""
    _RUNNING_SERVERS.clear()
    client = MCPClient("localhost:19999")
    result = client.connect()
    assert result["ok"] is False
    assert result["code"] == "connection_failure"
