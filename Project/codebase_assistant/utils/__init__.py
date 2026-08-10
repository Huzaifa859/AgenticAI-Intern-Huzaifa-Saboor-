"""
utils
=====

Shared, dependency-light helpers used across the package.

Nothing here may import from agents, tools, rag, or memory — utils is
the bottom of the dependency graph so any subsystem can use it without
creating a cycle.

Contains:
- logging_utils: logger construction, replacing the bare `print()`
  calls currently scattered through the tools.
- path_utils: path filtering and file classification for ingestion —
  the ignore list, Python/notebook/binary detection, and the
  proposal's per-file size limit.

NOTE: Placeholder only. Every helper returns a placeholder value.
"""

from .logging_utils import get_logger
from .path_utils import (
    is_binary_file,
    is_ignored_path,
    is_notebook,
    is_python_file,
    is_within_size_limit,
)

__all__ = [
    "get_logger",
    "is_ignored_path",
    "is_python_file",
    "is_notebook",
    "is_binary_file",
    "is_within_size_limit",
]
