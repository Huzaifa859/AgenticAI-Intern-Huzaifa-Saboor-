"""
path_utils.py
==============

Placeholder helpers for deciding which files the ingestion pipeline
should index, skip, or reject.

These encode the proposal's Scope & Limits: Python only, binaries
ignored, notebooks detected and skipped with a clear message rather
than silently dropped, and a per-file size ceiling.

TODO: Implement the real checks, and source the ignore list and size
limit from Config rather than the module-level defaults below.
"""

from __future__ import annotations

# Directories excluded from ingestion, per the proposal's indexing
# pipeline. TODO: move onto Config so this is configurable.
DEFAULT_IGNORED_DIRECTORIES = (".git", "__pycache__", "venv", "node_modules")

# Per-file ceiling from the proposal's Scope & Limits (500 KB).
# TODO: move onto Config alongside the file-count and LOC caps.
DEFAULT_MAX_FILE_SIZE_BYTES = 500 * 1024


def is_ignored_path(path: str) -> bool:
    """
    Report whether a path falls inside an ignored directory.

    Args:
        path: Path to check.

    Returns:
        True if the path should be skipped (placeholder always returns
        False).

    TODO: Implement the real check against the configured ignore list.
    """
    # TODO: implement real ignore-list matching
    return False


def is_python_file(path: str) -> bool:
    """
    Report whether a path points at a Python source file.

    Args:
        path: Path to check.

    Returns:
        True for Python sources (placeholder always returns False).

    TODO: Implement the real extension check.
    """
    # TODO: implement real Python source detection
    return False


def is_notebook(path: str) -> bool:
    """
    Report whether a path points at a Jupyter notebook.

    Notebooks are out of scope for the MVP but must be detected so the
    pipeline can skip them with an explicit message instead of
    dropping them silently.

    Args:
        path: Path to check.

    Returns:
        True for notebooks (placeholder always returns False).

    TODO: Implement the real check.
    """
    # TODO: implement real notebook detection
    return False


def is_binary_file(path: str) -> bool:
    """
    Report whether a path points at a binary file.

    Args:
        path: Path to check.

    Returns:
        True for binaries (placeholder always returns False).

    TODO: Implement real binary detection (null-byte sniff or an
    extension allowlist).
    """
    # TODO: implement real binary detection
    return False


def is_within_size_limit(path: str, max_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> bool:
    """
    Report whether a file is small enough to ingest.

    Args:
        path: Path to check.
        max_bytes: Maximum permitted size in bytes.

    Returns:
        True if the file is within the limit (placeholder always
        returns False).

    TODO: Implement the real size check.
    """
    # TODO: implement real size check
    return False
