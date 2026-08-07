"""
Allow ``python -m codebase_assistant.mcp`` to serve MCP over stdio.
"""

from __future__ import annotations

import sys

from .stdio_server import main


if __name__ == "__main__":
    sys.exit(main())
