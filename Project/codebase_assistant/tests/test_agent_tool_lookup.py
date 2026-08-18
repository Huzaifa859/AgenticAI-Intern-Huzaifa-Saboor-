"""
test_agent_tool_lookup.py
==========================

Unit tests verifying agents resolve FilesystemTools through ToolRegistry
rather than constructing a parallel instance when the registry already
holds one for the same workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import pytest

from codebase_assistant.agents.code_analysis_agent import CodeAnalysisAgent
from codebase_assistant.agents.documentation_agent import DocumentationAgent
from codebase_assistant.agents.testing_agent import TestingAgent
from codebase_assistant.config import Config
from codebase_assistant.tools.filesystem_tools import FilesystemTools
from codebase_assistant.tools.github_tools import GitHubTools
from codebase_assistant.tools.registry import ToolRegistry


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Temporary repository with one Python module."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def registered_tools(
    workspace: Path,
) -> Tuple[ToolRegistry, FilesystemTools, GitHubTools]:
    """Registry holding shared filesystem and github tool instances."""
    registry = ToolRegistry()
    filesystem = FilesystemTools(workspace_root=str(workspace))
    github = GitHubTools()
    registry.register_tool("filesystem.read_file", filesystem.read_file)
    registry.register_tool("filesystem.list_files", filesystem.list_files)
    registry.register_tool("filesystem.file_exists", filesystem.file_exists)
    registry.register_tool("github.validate_repository", github.validate_repository)
    registry.register_tool("github.clone_repository", github.clone_repository)
    return registry, filesystem, github


def test_documentation_agent_uses_registered_filesystem(
    workspace: Path,
    registered_tools: Tuple[ToolRegistry, FilesystemTools, GitHubTools],
) -> None:
    """DocumentationAgent should reuse the registry's FilesystemTools."""
    registry, filesystem, _ = registered_tools
    agent = DocumentationAgent(tool_registry=registry)
    assert agent._filesystem_tools(str(workspace)) is filesystem


def test_testing_agent_uses_registered_filesystem(
    workspace: Path,
    registered_tools: Tuple[ToolRegistry, FilesystemTools, GitHubTools],
) -> None:
    """TestingAgent should reuse the registry's FilesystemTools."""
    registry, filesystem, _ = registered_tools
    agent = TestingAgent(tool_registry=registry)
    assert agent._filesystem_tools(str(workspace)) is filesystem


def test_code_analysis_agent_uses_registered_filesystem(
    workspace: Path,
    registered_tools: Tuple[ToolRegistry, FilesystemTools, GitHubTools],
) -> None:
    """CodeAnalysisAgent should reuse the registry's FilesystemTools."""
    registry, filesystem, _ = registered_tools
    agent = CodeAnalysisAgent(
        tool_registry=registry,
        config=Config(workspace_root=str(workspace)),
    )
    assert agent._filesystem_tools(str(workspace), config=agent.config) is filesystem


def test_mismatched_workspace_builds_request_scoped_tools(
    workspace: Path,
    registered_tools: Tuple[ToolRegistry, FilesystemTools, GitHubTools],
    tmp_path: Path,
) -> None:
    """A different repository root must not reuse the wrong sandbox."""
    registry, filesystem, _ = registered_tools
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.py").write_text("x = 1\n", encoding="utf-8")

    agent = DocumentationAgent(tool_registry=registry)
    resolved = agent._filesystem_tools(str(other))

    assert resolved is not filesystem
    assert Path(resolved.workspace_root).resolve() == other.resolve()


def test_without_registry_agents_still_function(workspace: Path) -> None:
    """Unit-test style construction without a registry must keep working."""
    agent = DocumentationAgent()
    resolved = agent._filesystem_tools(str(workspace))
    assert isinstance(resolved, FilesystemTools)
    assert "def add" in resolved.read_file("math_utils.py")


def test_github_tools_resolved_from_registry(
    registered_tools: Tuple[ToolRegistry, FilesystemTools, GitHubTools],
) -> None:
    """GitHubTools should be reachable through the same lookup path."""
    registry, _, github = registered_tools
    agent = DocumentationAgent(tool_registry=registry)
    assert agent._github_tools() is github
    assert agent.get_tool("github.clone_repository") is not None


@patch.object(DocumentationAgent, "_ensure_index", autospec=True)
def test_documentation_pipeline_reads_via_registry(
    _mock_index: object,
    workspace: Path,
    registered_tools: Tuple[ToolRegistry, FilesystemTools, GitHubTools],
) -> None:
    """generate_readme should resolve tools through the registry."""
    registry, filesystem, _ = registered_tools
    client = MagicMock()
    client.is_available.return_value = False
    agent = DocumentationAgent(model_client=client, tool_registry=registry)

    assert agent._filesystem_tools(str(workspace)) is filesystem
    agent.generate_readme(str(workspace))
    assert agent.get_tool("filesystem.read_file") is not None
