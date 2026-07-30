"""
tool_exceptions.py
===================

Errors raised by the tool layer — the registry, GitHub cloning, and
filesystem access.

These cover the failure modes the milestone plan requires be caught and
logged rather than surfaced as tracebacks: invalid repository URLs and
empty or unreadable files.

TODO: Raise these from `tools/` and the ingestion pipeline once real
I/O is implemented.
"""

from __future__ import annotations

from .base import CodebaseAssistantError


class ToolError(CodebaseAssistantError):
    """Base class for every tool-layer failure."""


class ToolNotFoundError(ToolError):
    """No tool is registered under the requested name."""


class ToolExecutionError(ToolError):
    """A registered tool raised while running."""


class InvalidRepositoryURLError(ToolError):
    """The supplied repository URL is malformed or unreachable."""


class RepositoryCloneError(ToolError):
    """Cloning the repository failed."""


class EmptyFileError(ToolError):
    """A file that was expected to have content is empty."""


class FileTooLargeError(ToolError):
    """A file exceeds the per-file size ceiling in Scope & Limits."""


class UnsupportedFileTypeError(ToolError):
    """
    A file's type is out of scope for the MVP.

    Covers binaries and Jupyter notebooks, which must be skipped with
    an explicit message rather than dropped silently.
    """


class PathOutsideWorkspaceError(ToolError):
    """A requested path escapes the configured workspace root."""
