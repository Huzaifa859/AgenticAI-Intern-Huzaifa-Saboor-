#!/usr/bin/env python3
"""
run_demo.py
===========

Run the Week 6 CLI demo against the bundled example repository.

Usage (from the Project/ directory):

    python examples/run_demo.py
    python examples/run_demo.py --no-color
    python examples/run_demo.py --question "Find likely bugs"
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_REPO = PROJECT_ROOT / "examples" / "demo_repo"
MAIN = PROJECT_ROOT / "app" / "main.py"


def main() -> None:
    """Invoke app/main.py on the demo repository."""
    if not DEMO_REPO.is_dir():
        print(f"Demo repository not found: {DEMO_REPO}", file=sys.stderr)
        raise SystemExit(1)

    cmd = [
        sys.executable,
        str(MAIN),
        str(DEMO_REPO.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        *sys.argv[1:],
    ]

    raise SystemExit(
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode
    )


if __name__ == "__main__":
    main()
