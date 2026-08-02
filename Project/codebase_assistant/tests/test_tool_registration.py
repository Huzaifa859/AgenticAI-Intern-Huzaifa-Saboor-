"""
test_tool_registration.py
==========================

Unit tests for tool registration during Supervisor initialization.

Model providers and RAG subsystems are stubbed, but FilesystemTools and
GitHubTools are real, so these tests check that the registry genuinely
wraps the shared tool instances. No network access is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from codebase_assistant.config import Config
from codebase_assistant.schemas.schemas import ToolCallRequest
from codebase_assistant.supervisor import Supervisor

REQUIRED_TOOLS = [
    "filesystem.read_file",
    "filesystem.write_file",
    "filesystem.list_files",
    "filesystem.file_exists",
    "github.validate_repository",
    "github.clone_repository",
]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Temporary workspace with one file for the filesystem tools."""
    (tmp_path / "math_utils.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def supervisor(workspace: Path) -> Iterator[Supervisor]:
    """Supervisor with real tools but stubbed providers and RAG."""
    with patch.object(Supervisor, "_init_openrouter_provider", return_value=None), \
        patch.object(Supervisor, "_init_ollama_provider", return_value=None), \
        patch("codebase_assistant.supervisor.Indexer"), \
        patch("codebase_assistant.supervisor.Retriever"), \
        patch("codebase_assistant.supervisor.MemoryStore"), \
        patch("codebase_assistant.supervisor.CodeAnalysisAgent"), \
        patch("codebase_assistant.supervisor.DocumentationAgent"), \
        patch("codebase_assistant.supervisor.TestingAgent"):
        yield Supervisor(config=Config(workspace_root=str(workspace)))


@pytest.mark.parametrize("tool_name", REQUIRED_TOOLS)
def test_required_tools_are_registered(supervisor: Supervisor, tool_name: str) -> None:
    """The tools this step must expose should all be present."""
    assert tool_name in supervisor.tool_registry.list_tools()
    assert supervisor.tool_registry.get_tool(tool_name) is not None


def test_registered_handlers_are_the_shared_instances(
    supervisor: Supervisor,
) -> None:
    """Registration must wrap the existing tools, not new objects."""
    read_file = supervisor.tool_registry.get_tool("filesystem.read_file")
    clone = supervisor.tool_registry.get_tool("github.clone_repository")

    assert read_file.__self__ is supervisor.filesystem_tools
    assert clone.__self__ is supervisor.github_tools


def test_every_public_tool_method_is_registered(supervisor: Supervisor) -> None:
    """No public method of either tool class should be missing."""
    registered = set(supervisor.tool_registry.list_tools())

    for namespace, instance in (
        ("filesystem", supervisor.filesystem_tools),
        ("github", supervisor.github_tools),
    ):
        for name, _ in Supervisor._public_methods(instance):
            assert f"{namespace}.{name}" in registered


def test_private_helpers_are_not_registered(supervisor: Supervisor) -> None:
    """Internal helpers must stay out of the agent-facing registry."""
    assert not any(
        "._" in name for name in supervisor.tool_registry.list_tools()
    )


def test_registry_invokes_a_filesystem_tool(
    supervisor: Supervisor, workspace: Path
) -> None:
    """call_tool should reach the real implementation and return its result."""
    result = supervisor.tool_registry.call_tool(
        ToolCallRequest(tool_name="filesystem.read_file", arguments={"path": "math_utils.py"})
    )

    assert result.success is True
    assert "def add(a, b)" in result.result


def test_registry_invokes_a_write_then_read_round_trip(
    supervisor: Supervisor,
) -> None:
    """Writing through the registry should be visible to a registry read."""
    write = supervisor.tool_registry.call_tool(
        ToolCallRequest(
            tool_name="filesystem.write_file",
            arguments={"path": "notes.txt", "content": "hello"},
        )
    )
    read = supervisor.tool_registry.call_tool(
        ToolCallRequest(tool_name="filesystem.read_file", arguments={"path": "notes.txt"})
    )

    assert write.success is True
    assert read.success is True
    assert read.result == "hello"


def test_registry_invokes_a_github_tool(supervisor: Supervisor) -> None:
    """A GitHub tool should be callable by name without touching the network."""
    result = supervisor.tool_registry.call_tool(
        ToolCallRequest(
            tool_name="github.is_remote_reference",
            arguments={"repo_url": "https://github.com/psf/requests"},
        )
    )

    assert result.success is True
    assert result.result is True


def test_unknown_tool_reports_available_names(supervisor: Supervisor) -> None:
    """A typo should produce a readable failure, not an exception."""
    result = supervisor.tool_registry.call_tool(
        ToolCallRequest(tool_name="filesystem.reed_file", arguments={})
    )

    assert result.success is False
    assert "No tool registered" in result.error
    assert "filesystem.read_file" in result.error


def test_registration_is_not_duplicated(supervisor: Supervisor) -> None:
    """Tool names must be unique, so the registry stays an unambiguous lookup."""
    names = supervisor.tool_registry.list_tools()
    assert len(names) == len(set(names))
