"""
base.py
=======

Defines the root exception every Codebase Assistant error inherits
from.

Kept in its own module rather than in `__init__.py` so the three
category modules can import it without a circular import back through
the package.

TODO: Add shared context carrying (file path, task id, agent type) once
errors are actually raised, so the notebook can render a readable
message instead of a raw traceback.
"""

from __future__ import annotations


class CodebaseAssistantError(Exception):
    """
    Root of the exception hierarchy.

    Catching this catches every error the system raises deliberately,
    which lets the notebook distinguish expected failures from genuine
    crashes.
    """
