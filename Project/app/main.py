"""
app/main.py
===========

Demonstration entry point for the Codebase Assistant analysis pipeline.

Run from the project root (the directory containing both `app/` and
`codebase_assistant/`):

    python app/main.py /path/to/repository
    python app/main.py . --question "Find security bugs"
    python app/main.py https://github.com/pallets/flask

A GitHub HTTPS URL is cloned into a temporary directory, analyzed, and
then deleted. Local paths are analyzed in place.

If no repository is given on the command line, you will be prompted for
one interactively.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import stat
import sys
import tempfile
from typing import Dict, List, Optional
from urllib.parse import urlparse

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
from codebase_assistant.exceptions.tool_exceptions import (  # noqa: E402
    InvalidRepositoryURLError,
    RepositoryCloneError,
)
from codebase_assistant.schemas.schemas import AgentType  # noqa: E402
from codebase_assistant.supervisor import Supervisor  # noqa: E402
from codebase_assistant.tools.github_tools import GitHubTools  # noqa: E402
from report_formatter import print_report  # noqa: E402

#: Clones made during this execution, keyed by canonical repository URL.
_CLONE_CACHE: Dict[str, str] = {}

#: Temporary directories holding those clones, removed on exit.
_TEMPORARY_ROOTS: List[str] = []


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
        help=(
            "Local path or GitHub HTTPS URL of the repository to analyze. "
            "A URL is cloned to a temporary directory and removed afterwards. "
            "If omitted, you will be prompted."
        ),
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


def read_repository_reference(raw_reference: Optional[str]) -> str:
    """
    Read the repository reference, prompting when none was supplied.

    The reference may be a local path or a GitHub HTTPS URL; classifying
    it is left to the caller.

    Args:
        raw_reference: Value from the command line, or None to prompt.

    Returns:
        A non-empty repository reference.

    Raises:
        SystemExit: If the prompt is cancelled.
    """
    reference = (raw_reference or "").strip()

    while not reference:
        try:
            reference = input("Repository path or GitHub URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            raise SystemExit(1) from None

    return reference


def _remove_readonly(func, path, _excinfo) -> None:
    """
    Retry a failed removal after clearing the read-only bit.

    Git writes object files read-only, which blocks deletion on Windows.

    Args:
        func: The removal function that failed.
        path: Path it failed on.
        _excinfo: Exception (or exc_info) reported by shutil.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_temporary_tree(path: str) -> None:
    """
    Delete a temporary directory tree, tolerating read-only git objects.

    Args:
        path: Directory to remove. Missing paths are ignored.
    """
    if not os.path.isdir(path):
        return

    try:
        try:
            shutil.rmtree(path, onexc=_remove_readonly)
        except TypeError:
            # Python < 3.12 does not accept `onexc`.
            shutil.rmtree(path, onerror=_remove_readonly)
    except OSError as exc:
        print(
            f"Warning: could not remove temporary clone at {path}: {exc}",
            file=sys.stderr,
        )


def normalize_repository_url(repo_url: str) -> str:
    """
    Build a canonical key identifying a remote repository.

    A trailing `.git`, a trailing slash, and letter case are all
    cosmetic on GitHub, so URLs differing only in those ways describe
    the same repository and should share one clone.

    Args:
        repo_url: Remote repository URL.

    Returns:
        A canonical `host/owner/name` key.
    """
    parsed = urlparse(str(repo_url).strip())
    segments = [segment for segment in parsed.path.split("/") if segment]

    if segments and segments[-1].lower().endswith(".git"):
        segments[-1] = segments[-1][: -len(".git")]

    return "/".join([parsed.netloc.lower(), *(s.lower() for s in segments)])


def clone_or_reuse_repository(github_tools: GitHubTools, repo_url: str) -> str:
    """
    Make a remote repository available locally, reusing an earlier clone.

    Clones made during this execution are cached by canonical URL, so
    asking for the same repository twice does not clone it twice. Every
    clone lives under a temporary directory removed on exit.

    Args:
        github_tools: The existing GitHubTools component to clone with.
        repo_url: GitHub HTTPS URL to clone.

    Returns:
        Path to the cloned repository on disk.

    Raises:
        InvalidRepositoryURLError: If the URL fails validation.
        RepositoryCloneError: If the repository cannot be cloned.
    """
    key = normalize_repository_url(repo_url)
    cached = _CLONE_CACHE.get(key)
    if cached and os.path.isdir(cached):
        print(f"Reusing existing clone of {key}.")
        return cached

    print("Validating repository...")
    github_tools.validate_repository(repo_url)

    temporary_root = tempfile.mkdtemp(prefix="codebase_assistant_clone_")
    _TEMPORARY_ROOTS.append(temporary_root)
    destination = os.path.join(temporary_root, "repo")

    print("Cloning repository...")
    github_tools.clone_repository(repo_url, destination)
    print("Repository cloned.")

    _CLONE_CACHE[key] = destination
    return destination


def cleanup_temporary_clones() -> None:
    """
    Remove every temporary clone created during this execution.

    Safe to call more than once; registered with `atexit` so an
    unexpected exit still leaves no clone behind.
    """
    while _TEMPORARY_ROOTS:
        remove_temporary_tree(_TEMPORARY_ROOTS.pop())
    _CLONE_CACHE.clear()


atexit.register(cleanup_temporary_clones)


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


def run_analysis(
    agent: CodeAnalysisAgent,
    repository_path: str,
    question: str,
    color: Optional[bool],
) -> None:
    """
    Analyze a repository already present on disk and print the report.

    Args:
        agent: The Code Analysis Agent to run.
        repository_path: Local directory to analyze.
        question: Natural language question driving the analysis.
        color: Force ANSI color on/off, or None to auto-detect.
    """
    print(f"Analyzing {repository_path} ...")
    report = agent.analyze_repository(
        repository_path=repository_path,
        question=question,
    )
    print_report(report, color=color)


def main() -> None:
    """
    Run the analysis pipeline and print a readable report.

    Creates a Supervisor, obtains the CodeAnalysisAgent, and runs the
    full repository analysis without modifying the pipeline itself.
    A GitHub URL is cloned to a temporary directory first and removed
    once the run finishes.
    """
    args = parse_args()
    reference = read_repository_reference(args.repository)

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

    if not GitHubTools.is_remote_reference(reference):
        run_analysis(
            agent, resolve_repository_path(reference), args.question, color
        )
        return

    try:
        repository_path = clone_or_reuse_repository(
            supervisor.github_tools, reference
        )
        print("Starting analysis...")
        run_analysis(agent, repository_path, args.question, color)
    except InvalidRepositoryURLError as exc:
        print(f"Error: invalid repository URL. {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except RepositoryCloneError as exc:
        print(f"Error: could not clone repository. {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        cleanup_temporary_clones()


if __name__ == "__main__":
    main()
