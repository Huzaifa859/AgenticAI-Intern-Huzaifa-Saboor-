"""
app/main.py
===========

Demonstrates the Codebase Assistant scaffold running end-to-end using
ONLY placeholder/stub logic.

Flow:
    Create Supervisor
        -> Agents auto-registered (inside Supervisor.__init__)
        -> Build a fake task ("request")
        -> Supervisor routes the task to the right agent
        -> Agent returns a placeholder response
        -> Response is printed

No AI calls. No RAG. No MCP. This file only proves the wiring between
Supervisor, Agents, and schemas actually works end-to-end.

Run from the project root (the directory containing both `app/` and
`codebase_assistant/`):

    python app/main.py
"""

from __future__ import annotations

import os
import sys

# This file lives in `app/`, one level below the project root, but it's
# executed directly (`python app/main.py`) rather than as a package
# module. That means Python only puts `app/` on sys.path by default, so
# without this, `import codebase_assistant` would fail. Insert the
# project root explicitly so the package can be found.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from codebase_assistant.config import Config
from codebase_assistant.schemas.schemas import AgentType
from codebase_assistant.supervisor import Supervisor

# Human-readable names for print output only -- purely cosmetic.
_AGENT_DISPLAY_NAMES = {
    AgentType.CODE_ANALYSIS: "Code Analysis Agent",
    AgentType.DOCUMENTATION: "Documentation Agent",
    AgentType.TESTING: "Testing Agent",
}


def main() -> None:
    """
    Wire up the Supervisor (and its agents), dispatch one fake task
    through the full routing path, and print the placeholder result.

    This is a pure architecture demo: Supervisor -> routing -> Agent
    -> response. Every value involved is fake/placeholder data.
    """
    print("Supervisor started.")
    supervisor = Supervisor(config=Config.load())

    # A fake incoming request -- in the real system this would come
    # from a user prompt, CLI argument, or notebook cell.
    task_name = "analyze code for bugs"
    repo_path = "/path/to/fake/repo"

    print("Routing task...")
    agent_type = supervisor.route_task(task_name)
    print(f"Running {_AGENT_DISPLAY_NAMES[agent_type]}...")

    response = supervisor.handle_task(task_name, repo_path=repo_path)

    # response is a plain dict, e.g. {"status": "success", "message": "..."}
    print(response["message"])
    print("Finished.")


if __name__ == "__main__":
    main()
