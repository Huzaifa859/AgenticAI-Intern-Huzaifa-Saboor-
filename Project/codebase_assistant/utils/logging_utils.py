"""
logging_utils.py
=================

Placeholder for centralized logger construction.

The tools currently announce themselves with bare `print()` calls.
Routing everything through a single logger factory is what makes the
proposal's tracing layer possible, and what turns raw tracebacks into
the human-readable errors the milestone plan calls for.

TODO: Implement real logger configuration — level from Config,
structured/JSON formatting, and a handler that can also feed the
hooks subsystem.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Return the logger for a given module.

    Args:
        name: Logger name, conventionally the caller's `__name__`.

    Returns:
        A standard library Logger. No handlers or formatting are
        configured yet, so it currently inherits root defaults.

    TODO: Apply the project's shared handler/formatter and honor the
    log level from Config.
    """
    # TODO: configure handlers, formatting, and level from Config
    return logging.getLogger(name)
