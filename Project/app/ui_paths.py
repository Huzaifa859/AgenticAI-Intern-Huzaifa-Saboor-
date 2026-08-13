"""
app/ui_paths.py
===============

Shared Streamlit/runtime data directories.

Honors Docker-friendly env vars when set; otherwise defaults under the
OS temp directory so local ``run_ui.bat`` keeps working unchanged.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional


def streamlit_data_dir() -> str:
    """
    Root directory for Streamlit Chroma / memory / history.

    Env: ``CODEBASE_ASSISTANT_DATA_DIR``
    Default: ``{temp}/codebase_assistant_streamlit``
    """
    override = (os.environ.get("CODEBASE_ASSISTANT_DATA_DIR") or "").strip()
    if override:
        return override
    return os.path.join(tempfile.gettempdir(), "codebase_assistant_streamlit")


def chroma_persist_dir() -> str:
    """Env ``CHROMA_PERSIST_DIR`` or ``{data_dir}/chroma``."""
    override = (os.environ.get("CHROMA_PERSIST_DIR") or "").strip()
    if override:
        return override
    return os.path.join(streamlit_data_dir(), "chroma")


def github_clones_dir() -> str:
    """
    Durable on-disk cache for GitHub repository clones.

    Env: ``GITHUB_CLONES_DIR``
    Default: ``{data_dir}/github_clones``

    Unlike ``tempfile.mkdtemp`` workspaces, this directory survives process
    exit and is never removed by ``cleanup_temporary_clones()``.
    """
    override = (os.environ.get("GITHUB_CLONES_DIR") or "").strip()
    if override:
        return override
    return os.path.join(streamlit_data_dir(), "github_clones")


def durable_github_clone_path(
    repo_url: str, *, clones_root: Optional[str] = None
) -> str:
    """
    Deterministic clone directory for a canonical GitHub HTTPS URL.

    Layout: ``<clones_root>/<host>/<owner>/<repo>``

    Only the first owner/repo path segments are used so URLs with extra
    suffixes (``.git``, ``/tree/...``) still map to the same directory.
    """
    from urllib.parse import urlparse

    parsed = urlparse(str(repo_url or "").strip())
    host = (parsed.netloc or "").lower()
    segments = [segment for segment in parsed.path.split("/") if segment]
    if segments and segments[-1].lower().endswith(".git"):
        segments[-1] = segments[-1][: -len(".git")]
    if not host or len(segments) < 2:
        raise ValueError(
            f"Cannot derive durable clone path from repository URL: {repo_url!r}"
        )
    owner = segments[0].lower()
    name = segments[1].lower()
    root = os.path.abspath(
        os.path.expanduser(clones_root or github_clones_dir())
    )
    return os.path.abspath(os.path.join(root, host, owner, name))


def memory_store_path() -> str:
    """Env ``MEMORY_STORE_PATH`` or ``{data_dir}/memory_store``."""
    override = (os.environ.get("MEMORY_STORE_PATH") or "").strip()
    if override:
        return override
    return os.path.join(streamlit_data_dir(), "memory_store")


def run_history_path() -> str:
    """
    Env ``RUN_HISTORY_PATH`` or ``{data_dir}/ui_run_history.jsonl``.

    When Compose sets ``/data/history/ui_run_history.jsonl``, the parent
    directory is created by callers that write the file.
    """
    override = (os.environ.get("RUN_HISTORY_PATH") or "").strip()
    if override:
        return override
    return os.path.join(streamlit_data_dir(), "ui_run_history.jsonl")


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` when needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_dir(path: Optional[str] = None) -> str:
    """Ensure ``path`` (or the Streamlit data root) exists; return it."""
    target = (path or streamlit_data_dir()).strip() or streamlit_data_dir()
    os.makedirs(target, exist_ok=True)
    return target
