"""
test_tools.py
==============

Placeholder tests for the tool layer — ToolRegistry, GitHubTools, and
FilesystemTools.

Also covers the Week 6 error-handling requirements: invalid repository
URLs and empty files must fail cleanly rather than crash.

TODO: Replace every skip below with real assertions as each tool is
implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: assert register_tool then get_tool round-trips")
def test_registry_registers_and_retrieves_tool() -> None:
    """A registered tool should be retrievable by name."""


@pytest.mark.skip(reason="TODO: assert duplicate names raise ValueError")
def test_registry_rejects_duplicate_tool_names() -> None:
    """Registering the same name twice should be refused."""


@pytest.mark.skip(reason="TODO: assert call_tool dispatches to the handler and returns its result")
def test_registry_call_tool_invokes_handler() -> None:
    """Calling a registered tool should run it and wrap the result."""


@pytest.mark.skip(reason="TODO: assert InvalidRepositoryURLError is raised for a malformed URL")
def test_github_tools_rejects_invalid_repository_url() -> None:
    """A malformed repository URL should raise, not silently succeed."""


@pytest.mark.skip(reason="TODO: assert a public repo clones into the destination")
def test_github_tools_clones_public_repository() -> None:
    """Cloning a public repo should place it at the destination path."""


@pytest.mark.skip(reason="TODO: assert read_file returns real file content")
def test_filesystem_tools_reads_file() -> None:
    """Reading a file should return its actual contents."""


@pytest.mark.skip(reason="TODO: assert PathOutsideWorkspaceError is raised for an escaping path")
def test_filesystem_tools_blocks_paths_outside_workspace() -> None:
    """Paths escaping the workspace root should be refused."""


@pytest.mark.skip(reason="TODO: assert empty files raise EmptyFileError")
def test_filesystem_tools_handles_empty_file() -> None:
    """An empty file should fail cleanly rather than crash."""


@pytest.mark.skip(reason="TODO: assert ignored directories and oversized files are excluded")
def test_path_utils_applies_scope_limits() -> None:
    """Ignore list and the size ceiling should exclude the right files."""
