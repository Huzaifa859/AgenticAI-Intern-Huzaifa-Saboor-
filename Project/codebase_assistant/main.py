"""
main.py
=======

Placeholder entry point for running Codebase Assistant from the
command line.

TODO: Implement real CLI (e.g. via argparse/click/typer) for invoking
the Supervisor with a user-provided goal and workspace path.
"""

from __future__ import annotations

from .config import Config
from .supervisor import Supervisor


def main() -> None:
    """
    Entry point for running Codebase Assistant.

    TODO: Parse CLI arguments (goal, workspace path, config path),
    instantiate the Supervisor, and print/format the results.
    """
    config = Config.load()
    supervisor = Supervisor(config=config)

    # TODO: replace with real CLI-driven goal input
    goal = "Placeholder goal: implement real CLI argument parsing."
    responses = supervisor.handle_goal(goal)
    print(f"Received {len(responses)} agent response(s) (placeholder run).")


if __name__ == "__main__":
    main()
