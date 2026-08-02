"""
main.py
=======

Deprecated entry point for Codebase Assistant.

Prefer the primary CLI:

    python app/main.py

This module exists only so older invocations of
``python -m codebase_assistant.main`` (or a direct import of
``codebase_assistant.main``) keep working. It forwards every call to
``app.main.main()`` — there is no second execution path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from typing import List, Optional


def _load_app_main():
    """
    Load ``app/main.py`` as a module without requiring ``app`` to be a package.

    Returns:
        The loaded ``app.main`` module.

    Raises:
        ImportError: If ``app/main.py`` cannot be located or loaded.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_main_path = os.path.join(project_root, "app", "main.py")
    if not os.path.isfile(app_main_path):
        raise ImportError(
            f"Primary CLI not found at {app_main_path}. "
            "Run `python app/main.py` from the project root."
        )

    # Ensure the same import path setup `app/main.py` expects when
    # executed directly.
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    app_dir = os.path.join(project_root, "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    spec = importlib.util.spec_from_file_location(
        "codebase_assistant_app_main", app_main_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load primary CLI from {app_main_path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Optional[List[str]] = None) -> None:
    """
    Deprecated forwarder to the primary CLI in ``app/main.py``.

    Args:
        argv: Optional argument list forwarded to ``app.main.main``.
    """
    warnings.warn(
        "codebase_assistant.main is deprecated; use `python app/main.py` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    app_main = _load_app_main()
    app_main.main(argv)


if __name__ == "__main__":
    main()
