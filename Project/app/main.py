"""
app/main.py
===========

Demonstration entry point for the Codebase Assistant analysis pipeline.

Run from the project root (the directory containing both `app/` and
`codebase_assistant/`):

    python app/main.py /path/to/repository
    python app/main.py . --question "Find security bugs"

If no repository path is given on the command line, you will be prompted
for one interactively.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# This file lives in `app/`, one level below the project root, but it is
# executed directly (`python app/main.py`) rather than as a package
# module. Insert the project root so `import codebase_assistant` works.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from codebase_assistant.agents.code_analysis_agent import CodeAnalysisAgent  # noqa: E402
from codebase_assistant.config import Config  # noqa: E402
from codebase_assistant.schemas.schemas import AgentType  # noqa: E402
from codebase_assistant.supervisor import Supervisor  # noqa: E402
from report_formatter import print_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the demo entry point.

    Returns:
        Parsed arguments with an optional repository path and question.
    """
    parser = argparse.ArgumentParser(
        description="Run the Codebase Assistant analysis pipeline on a repository.",
    )
    parser.add_argument(
        "repository",
        nargs="?",
        help="Path to the repository to analyze. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--question",
        "-q",
        default="Find likely bugs and correctness problems in this code.",
        help="Natural language question to drive retrieval and analysis.",
    )
    color = parser.add_mutually_exclusive_group()
    color.add_argument(
        "--color",
        action="store_true",
        help="Force ANSI color output.",
    )
    color.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    return parser.parse_args()


def resolve_repository_path(raw_path: Optional[str]) -> str:
    """
    Resolve and validate the repository path.

    Accepts a command-line value or prompts when none was supplied.
    The path must exist and be a directory.

    Args:
        raw_path: Path from the command line, or None to prompt.

    Returns:
        The absolute, normalized repository path.

    Raises:
        SystemExit: If the path is missing, not a directory, or empty
            after prompting.
    """
    candidate = (raw_path or "").strip()

    while not candidate:
        try:
            candidate = input("Repository path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            raise SystemExit(1) from None

    resolved = os.path.abspath(os.path.expanduser(candidate))

    if not os.path.exists(resolved):
        print(f"Error: path does not exist: {resolved}", file=sys.stderr)
        raise SystemExit(1)

    if not os.path.isdir(resolved):
        print(f"Error: path is not a directory: {resolved}", file=sys.stderr)
        raise SystemExit(1)

    return resolved


def main() -> None:
    """
    Run the analysis pipeline and print a readable report.

    Creates a Supervisor, obtains the CodeAnalysisAgent, and runs the
    full repository analysis without modifying the pipeline itself.
    """
    args = parse_args()
    repository_path = resolve_repository_path(args.repository)

    color: Optional[bool] = None
    if args.color:
        color = True
    elif args.no_color:
        color = False

    print("Starting Codebase Assistant...")
    supervisor = Supervisor(config=Config.load())

    agent = supervisor.agents[AgentType.CODE_ANALYSIS]
    if not isinstance(agent, CodeAnalysisAgent):
        print("Error: Code Analysis Agent is not available.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Analyzing {repository_path} ...")
    report = agent.analyze_repository(
        repository_path=repository_path,
        question=args.question,
    )
    print_report(report, color=color)


if __name__ == "__main__":
    main()
