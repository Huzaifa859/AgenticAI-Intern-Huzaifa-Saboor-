"""
store_paths.py
==============

Shared helpers for per-repository Chroma persistence directories.

Analysis, Documentation, and Testing must index into the same
subdirectory for a given workspace so incremental manifests (and the
warm worker) can skip re-embedding unchanged files across agents.
"""

from __future__ import annotations

import hashlib
import os


def vector_store_for_repository(persist_directory: str, workspace_root: str) -> str:
    """
    Return the Chroma directory for one repository workspace.

    Each repository gets ``persist_directory/<sha256(abs_root)[:12]>`` so
    chunks from different repos never share a collection.
    """
    root = os.path.abspath(os.path.expanduser(str(workspace_root or ".")))
    base = os.path.abspath(os.path.expanduser(str(persist_directory or ".")))
    digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:12]
    return os.path.join(base, digest)
