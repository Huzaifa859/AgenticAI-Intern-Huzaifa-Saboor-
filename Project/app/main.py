"""
app/main.py
===========

Primary CLI for the Codebase Assistant.

Run from the project root (the directory containing both `app/` and
`codebase_assistant/`):

    python app/main.py
    python app/main.py /path/to/repository
    python app/main.py . --question "Find security bugs"
    python app/main.py https://github.com/pallets/flask
    python app/main.py . --agent analysis
    python app/main.py . --agent documentation
    python app/main.py . --agent testing
    python app/main.py . --agent all
    python app/main.py . --agent documentation --file app/auth.py
    python app/main.py . --agent documentation --file app/auth.py --function authenticate
    python app/main.py . --agent documentation --class UserService
    python app/main.py . --agent documentation --write-to-disk
    python app/main.py . --agent testing --file math_utils.py
    python app/main.py . --agent testing --function add

Without ``--agent``, the repository is prepared once and an interactive
menu dispatches to Code Analysis, Documentation, or Testing. With
``--agent``, the chosen agent (or all three) runs once and the process
exits. Documentation and Testing support file/function/class targeting
and optional write-back through existing agent request context fields.
"""

from __future__ import annotations

import argparse
import atexit
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple
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
from codebase_assistant.agents.documentation_agent import DocumentationAgent  # noqa: E402
from codebase_assistant.agents.testing_agent import TestingAgent  # noqa: E402
from codebase_assistant.config import Config  # noqa: E402
from codebase_assistant.exceptions.tool_exceptions import (  # noqa: E402
    InvalidRepositoryURLError,
    RepositoryCloneError,
)
from codebase_assistant.memory.conversation_memory import (  # noqa: E402
    ConversationMemory,
)
from codebase_assistant.schemas.schemas import (  # noqa: E402
    AgentRequest,
    AgentType,
    DocumentationResult,
    ModelMessage,
    TestingResult,
)
from codebase_assistant.supervisor import Supervisor  # noqa: E402
from codebase_assistant.tools.github_tools import GitHubTools  # noqa: E402
from codebase_assistant.tracing.events import TraceEventType  # noqa: E402
from codebase_assistant.tracing.tracer import Tracer  # noqa: E402
from report_formatter import (  # noqa: E402
    print_documentation_result,
    print_report,
    print_testing_result,
)

#: Soft cap so CLI memory never persists full reports or generated source.
_MAX_MEMORY_CONTENT_CHARS = 400

#: Clones made during this execution, keyed by canonical repository URL.
_CLONE_CACHE: Dict[str, str] = {}

#: Temporary directories holding those clones, removed on exit.
_TEMPORARY_ROOTS: List[str] = []

#: Non-interactive ``--agent`` values and their meanings.
_AGENT_CHOICES = ("analysis", "documentation", "testing", "all")

MENU_TEXT = """\
=================================
Select an agent
=================================

1. Code Analysis Agent

2. Documentation Agent

3. Testing Agent

4. Exit

Choice: """

DOCUMENTATION_MENU_TEXT = """\
=================================
Documentation Options
=================================

1. Repository README

2. Document a file

3. Document a function

4. Document a class

5. Back

Choice: """

TESTING_MENU_TEXT = """\
=================================
Testing Options
=================================

1. Test repository

2. Test file

3. Test function

4. Back

Choice: """

#: Tracer event name -> user-facing progress line (order preserved).
_PROGRESS_LABELS: Tuple[Tuple[str, str], ...] = (
    ("indexing", "Indexing repository..."),
    ("retrieval", "Retrieving context..."),
    ("documentation_started", "Generating documentation..."),
    ("documentation_symbol_started", "Generating documentation..."),
    ("documentation_grounding_started", "Grounding documentation..."),
    ("documentation_write_started", "Writing documentation..."),
    ("documentation_finished", "Done."),
    ("testing_started", "Generating tests..."),
    ("testing_symbol_generation_started", "Generating tests..."),
    ("testing_import_validation_started", "Validating imports..."),
    ("pytest_execution_started", "Running pytest..."),
    ("testing_coverage_started", "Measuring coverage..."),
    ("testing_repair_started", "Repairing tests..."),
    ("testing_finished", "Done."),
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the CLI.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments with an optional repository path, question,
        agent selection, and targeting / write-back flags.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Codebase Assistant CLI. Prepare a repository, then run agents "
            "interactively or once via --agent."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python app/main.py\n"
            "  python app/main.py . --agent analysis\n"
            "  python app/main.py . --agent documentation\n"
            "  python app/main.py . --agent testing\n"
            "  python app/main.py . --agent all\n"
            "  python app/main.py . --agent documentation --file app/auth.py\n"
            "  python app/main.py . --agent documentation "
            "--file app/auth.py --function authenticate\n"
            "  python app/main.py . --agent documentation "
            "--class UserService\n"
            "  python app/main.py . --agent documentation --write-to-disk\n"
            "  python app/main.py . --agent testing --file math_utils.py\n"
            "  python app/main.py . --agent analysis --question "
            '"Find security bugs"\n'
        ),
    )
    parser.add_argument(
        "repository",
        nargs="?",
        help=(
            "Local path or GitHub HTTPS URL of the repository. "
            "A URL is cloned to a temporary directory and removed on exit. "
            "If omitted, you will be prompted."
        ),
    )
    parser.add_argument(
        "--agent",
        "-a",
        choices=_AGENT_CHOICES,
        default=None,
        help=(
            "Run one agent (or all) non-interactively and exit. "
            "Omit this flag to keep the interactive menu."
        ),
    )
    parser.add_argument(
        "--question",
        "-q",
        default="Find likely bugs and correctness problems in this code.",
        help="Natural language question for the Code Analysis Agent.",
    )
    parser.add_argument(
        "--file",
        default="",
        help="Target file path for documentation or testing agents.",
    )
    parser.add_argument(
        "--function",
        default="",
        help="Target function name for documentation or testing agents.",
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        default="",
        help="Target class name for the documentation agent.",
    )
    parser.add_argument(
        "--write-to-disk",
        action="store_true",
        help="Ask DocumentationAgent to write README/docstrings to disk.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="When writing docs to disk, replace existing documentation.",
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
    return parser.parse_args(argv)


def read_repository_reference(raw_reference: Optional[str]) -> str:
    """
    Read the repository reference, prompting when none was supplied.

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
    """Retry a failed removal after clearing the read-only bit."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_temporary_tree(path: str) -> None:
    """Delete a temporary directory tree, tolerating read-only git objects."""
    if not os.path.isdir(path):
        return

    try:
        try:
            shutil.rmtree(path, onexc=_remove_readonly)
        except TypeError:
            shutil.rmtree(path, onerror=_remove_readonly)
    except OSError as exc:
        print(
            f"Warning: could not remove temporary clone at {path}: {exc}",
            file=sys.stderr,
        )


def normalize_repository_url(repo_url: str) -> str:
    """
    Build a canonical key identifying a remote repository.

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


def _cli_trace(
    tracer: Optional[Tracer],
    name: str,
    *,
    success: Optional[bool] = True,
    **metadata: object,
) -> None:
    """Record a CLI lifecycle event; never raises."""
    if tracer is None:
        return
    try:
        tracer.record(
            TraceEventType.LIFECYCLE,
            name,
            component="CLI",
            success=success,
            **metadata,
        )
    except Exception:
        pass


def clone_or_reuse_repository(
    github_tools: GitHubTools,
    repo_url: str,
    tracer: Optional[Tracer] = None,
) -> str:
    """
    Make a remote repository available locally, reusing an earlier clone.

    Args:
        github_tools: The existing GitHubTools component to clone with.
        repo_url: GitHub HTTPS URL to clone.
        tracer: Optional shared Tracer for clone events.

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
        _cli_trace(
            tracer,
            "repository_cloned",
            repository_url=repo_url,
            repository_path=cached,
            reused=True,
        )
        return cached

    print("Validating repository...")
    github_tools.validate_repository(repo_url)

    temporary_root = tempfile.mkdtemp(prefix="codebase_assistant_clone_")
    _TEMPORARY_ROOTS.append(temporary_root)
    destination = os.path.join(temporary_root, "repo")

    print("Cloning repository...")
    github_tools.clone_repository(repo_url, destination)
    print("Repository cloned.")
    _cli_trace(
        tracer,
        "repository_cloned",
        repository_url=repo_url,
        repository_path=destination,
        reused=False,
    )

    _CLONE_CACHE[key] = destination
    return destination


def cleanup_temporary_clones() -> None:
    """Remove every temporary clone created during this execution."""
    while _TEMPORARY_ROOTS:
        remove_temporary_tree(_TEMPORARY_ROOTS.pop())
    _CLONE_CACHE.clear()


atexit.register(cleanup_temporary_clones)


def resolve_repository_path(raw_path: Optional[str]) -> str:
    """
    Resolve and validate a local repository path.

    Args:
        raw_path: Local path from the command line or prompt.

    Returns:
        The absolute, normalized repository path.

    Raises:
        SystemExit: If the path is missing or not a directory.
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


def prepare_repository(supervisor: Supervisor, reference: str) -> str:
    """
    Resolve a local path or clone a GitHub URL once for the session.

    Args:
        supervisor: Running Supervisor (provides GitHubTools).
        reference: Local path or GitHub HTTPS URL.

    Returns:
        Absolute path to the repository on disk.

    Raises:
        SystemExit: If the reference cannot be prepared.
    """
    tracer = getattr(supervisor, "tracer", None)
    if not GitHubTools.is_remote_reference(reference):
        path = resolve_repository_path(reference)
        _cli_trace(
            tracer,
            "repository_selected",
            repository_reference=reference,
            repository_path=path,
            remote=False,
        )
        return path

    try:
        path = clone_or_reuse_repository(
            supervisor.github_tools, reference, tracer=tracer
        )
        _cli_trace(
            tracer,
            "repository_selected",
            repository_reference=reference,
            repository_path=path,
            remote=True,
        )
        return path
    except InvalidRepositoryURLError as exc:
        print(f"Error: invalid repository URL. {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except RepositoryCloneError as exc:
        print(f"Error: could not clone repository. {exc}", file=sys.stderr)
        raise SystemExit(1) from None


def prompt_choice(menu_text: str = MENU_TEXT, *, default: str = "") -> str:
    """
    Show a menu and return the user's raw choice.

    Args:
        menu_text: Menu text to display.
        default: Value returned on EOF/interrupt when non-empty.

    Returns:
        The stripped choice string.
    """
    try:
        return input(menu_text).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def ask_yes_no(prompt: str) -> bool:
    """
    Ask a yes/no question; default to no on empty/cancel.

    Args:
        prompt: Question to display.

    Returns:
        True only for an affirmative answer.
    """
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}


def prompt_text(label: str, default: str = "") -> str:
    """
    Prompt for a line of text, using ``default`` when the user submits empty.

    Args:
        label: Field label (without trailing colon).
        default: Default value shown in brackets when present.

    Returns:
        Entered text, or ``default`` when the input is empty.
    """
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return value or default


def resolve_repo_path(repository_path: str, raw_path: str) -> str:
    """
    Resolve a user-supplied path against the repository root when relative.

    Args:
        repository_path: Prepared repository root.
        raw_path: Absolute or repository-relative path.

    Returns:
        Absolute path when possible; otherwise the original string.
    """
    text = (raw_path or "").strip()
    if not text:
        return ""
    expanded = os.path.expanduser(text)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(repository_path, expanded))


def memory_target(memory: Optional[ConversationMemory]) -> Dict[str, str]:
    """Read last documentation/testing target fields from memory metadata."""
    if memory is None:
        return {"file_path": "", "function_name": "", "class_name": ""}
    meta = getattr(memory, "metadata", {}) or {}
    return {
        "file_path": str(meta.get("last_file_path") or ""),
        "function_name": str(meta.get("last_function_name") or ""),
        "class_name": str(meta.get("last_class_name") or ""),
    }


def store_memory_target(
    memory: Optional[ConversationMemory],
    *,
    file_path: str = "",
    function_name: str = "",
    class_name: str = "",
) -> None:
    """Persist the latest target selection for follow-up turns."""
    if memory is None:
        return
    if file_path:
        memory.metadata["last_file_path"] = file_path
    if function_name:
        memory.metadata["last_function_name"] = function_name
    if class_name:
        memory.metadata["last_class_name"] = class_name


def print_progress(message: str) -> None:
    """Print one progress line."""
    text = (message or "").strip()
    if text:
        print(text)


def emit_progress_from_tracer(
    tracer: Optional[Tracer],
    *,
    start_count: int = 0,
) -> None:
    """
    Print progress lines for tracer events recorded since ``start_count``.

    Uses existing agent/CLI trace names; does not invent timing logic.
    """
    if tracer is None:
        return
    try:
        events = list(tracer.get_events())
    except Exception:
        return
    seen_labels: set[str] = set()
    for event in events[start_count:]:
        name = getattr(event, "name", "") or ""
        for event_name, label in _PROGRESS_LABELS:
            if name == event_name and label not in seen_labels:
                print_progress(label)
                seen_labels.add(label)
                break


def tracer_event_count(tracer: Optional[Tracer]) -> int:
    """Return how many events the tracer currently holds."""
    if tracer is None:
        return 0
    try:
        return len(tracer.get_events())
    except Exception:
        return 0


def detect_writeback_note(summary: str) -> str:
    """Extract a short write-back note from a documentation summary."""
    text = summary or ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("write-back"):
            return stripped
    if "Write-back" in text:
        match = re.search(r"Write-back[^\n]*", text)
        if match:
            return match.group(0).strip()
    return ""


def parse_execution_counts(summary: str) -> Dict[str, int]:
    """Parse pytest pass/fail/skip/error counts from a TestingResult summary."""
    text = summary or ""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    match = re.search(
        r"(\d+)\s+passed,\s+(\d+)\s+failed,\s+(\d+)\s+skipped,\s+(\d+)\s+errors",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        counts["passed"] = int(match.group(1))
        counts["failed"] = int(match.group(2))
        counts["skipped"] = int(match.group(3))
        counts["errors"] = int(match.group(4))
    return counts


def format_cli_documentation_result(
    result: DocumentationResult,
    *,
    requested_target: str = "",
) -> str:
    """Format documentation output for the enhanced CLI display."""
    write_note = detect_writeback_note(result.summary or "")
    grounded = "No" if result.abstention is not None else "Yes"
    target = requested_target or result.file_path or result.function_name or "(repository)"
    body = (result.summary or "").strip()
    if write_note and write_note in body:
        body = body.replace(write_note, "").rstrip()
    lines = [
        "Documentation Summary",
        "---------------------",
        "",
        "Target:",
        str(target),
        "",
        "Summary:",
        body or "(empty)",
        "",
        "Written:",
        write_note or "(not written)",
        "",
        "Grounded:",
        grounded,
    ]
    if result.abstention is not None:
        lines.extend(
            [
                "",
                "Abstention:",
                (result.abstention.reason or "(none)").strip() or "(none)",
            ]
        )
    return "\n".join(lines)


def format_cli_testing_result(
    result: TestingResult,
    *,
    tracer: Optional[Tracer] = None,
) -> str:
    """Format testing output for the enhanced CLI display."""
    summary = result.summary or ""
    counts = parse_execution_counts(summary)
    names = sorted((result.generated_tests or {}).keys())
    event_names = tracer.event_names() if tracer is not None else []
    repair_attempted = any(
        name.startswith("testing_repair") for name in event_names
    )
    repair_success = "testing_repair_finished" in event_names and not any(
        name == "testing_repair_failed" for name in event_names
    )
    if "Repair: attempted one fix iteration." in summary:
        repair_attempted = True
    removed_imports = 0
    match = re.search(
        r"removed\s+(\d+)\s+unused invalid import", summary, flags=re.IGNORECASE
    )
    if match:
        removed_imports = int(match.group(1))
    elif "testing_import_validation_failed" in event_names:
        removed_imports = event_names.count("testing_import_validation_failed")

    coverage_pct = float(result.coverage_estimate or 0.0) * 100.0
    lines = [
        "Testing Summary",
        "---------------",
        "",
        "Generated:",
        f"{len(names)} test file" + ("" if len(names) == 1 else "s"),
        "",
        "Execution:",
        f"{counts['passed']} passed",
        f"{counts['failed']} failed",
        "",
        "Coverage:",
        f"{coverage_pct:.0f}%",
        "",
        "Repair:",
        (
            ("Attempted once\n" + ("Succeeded" if repair_success else "Did not fully succeed"))
            if repair_attempted
            else "Not needed"
        ),
        "",
        "Import Validation:",
        (
            f"Removed {removed_imports} invalid import"
            + ("" if removed_imports == 1 else "s")
            if removed_imports
            else "No invalid imports removed"
        ),
    ]
    if names:
        lines.extend(["", "Files:"])
        for name in names:
            lines.append(f"  - {name}")
    if result.abstention is not None:
        lines.extend(
            [
                "",
                "Abstention:",
                (result.abstention.reason or "(none)").strip() or "(none)",
            ]
        )
    return "\n".join(lines)


def failing_test_sources(result: TestingResult) -> Dict[str, str]:
    """
    Heuristically select generated modules that likely still fail.

    Prefers modules named in failure detail; otherwise returns all
    generated modules when execution reported failures.
    """
    summary = (result.summary or "").lower()
    generated = dict(result.generated_tests or {})
    if not generated:
        return {}
    counts = parse_execution_counts(result.summary or "")
    if counts["failed"] <= 0 and counts["errors"] <= 0:
        if "failed" not in summary and "error" not in summary:
            return {}
    matched = {
        name: source
        for name, source in generated.items()
        if name.lower() in summary or os.path.basename(name).lower() in summary
    }
    return matched or generated


def collect_documentation_options(
    repository_path: str,
    memory: Optional[ConversationMemory],
    *,
    interactive: bool,
    file_path: str = "",
    function_name: str = "",
    class_name: str = "",
    write_to_disk: bool = False,
    replace_existing: bool = False,
    mode: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Build documentation request options from flags or interactive prompts.

    Returns:
        Option dict, or ``None`` when the user chooses Back.
    """
    remembered = memory_target(memory)
    selected_mode = (mode or "").strip().lower()

    if interactive and not selected_mode:
        while True:
            choice = prompt_choice(DOCUMENTATION_MENU_TEXT, default="5")
            if choice == "1":
                selected_mode = "readme"
                break
            if choice == "2":
                selected_mode = "file"
                break
            if choice == "3":
                selected_mode = "function"
                break
            if choice == "4":
                selected_mode = "class"
                break
            if choice == "5":
                return None
            print("Invalid selection. Please choose 1-5.")

    if not selected_mode:
        if class_name:
            selected_mode = "class"
        elif function_name:
            selected_mode = "function"
        elif file_path:
            selected_mode = "file"
        else:
            selected_mode = "readme"

    out_file = file_path
    out_function = function_name
    out_class = class_name
    out_write = write_to_disk
    out_replace = replace_existing

    if interactive:
        if selected_mode in {"file", "function", "class"}:
            out_file = prompt_text(
                "File path",
                out_file or remembered.get("file_path", ""),
            )
        if selected_mode == "function":
            out_function = prompt_text(
                "Function name",
                out_function or remembered.get("function_name", ""),
            )
        if selected_mode == "class":
            out_class = prompt_text(
                "Class name",
                out_class or remembered.get("class_name", ""),
            )
        out_write = ask_yes_no("Write documentation to disk? (y/n): ")
        out_replace = False
        if out_write:
            out_replace = ask_yes_no("Replace existing documentation? (y/n): ")

    if out_file:
        out_file = resolve_repo_path(repository_path, out_file)

    if selected_mode == "readme":
        doc_type = "readme"
        instruction = "Generate a README summary for this repository."
    elif selected_mode == "class":
        doc_type = "module"
        instruction = (
            f"Document class {out_class}"
            + (f" in {out_file}" if out_file else "")
            + "."
        )
    elif selected_mode == "function":
        doc_type = "docstring"
        instruction = (
            f"Document function {out_function}"
            + (f" in {out_file}" if out_file else "")
            + "."
        )
    else:
        doc_type = "module"
        instruction = f"Document {out_file or 'the selected module'}."

    return {
        "mode": selected_mode,
        "doc_type": doc_type,
        "instruction": instruction,
        "file_path": out_file,
        "function_name": out_function,
        "class_name": out_class,
        "write_to_disk": bool(out_write),
        "replace_existing": bool(out_replace),
    }


def collect_testing_options(
    repository_path: str,
    memory: Optional[ConversationMemory],
    *,
    interactive: bool,
    file_path: str = "",
    function_name: str = "",
    mode: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Build testing request options from flags or interactive prompts.

    Returns:
        Option dict, or ``None`` when the user chooses Back.
    """
    remembered = memory_target(memory)
    selected_mode = (mode or "").strip().lower()

    if interactive and not selected_mode:
        while True:
            choice = prompt_choice(TESTING_MENU_TEXT, default="4")
            if choice == "1":
                selected_mode = "repository"
                break
            if choice == "2":
                selected_mode = "file"
                break
            if choice == "3":
                selected_mode = "function"
                break
            if choice == "4":
                return None
            print("Invalid selection. Please choose 1-4.")

    if not selected_mode:
        if function_name:
            selected_mode = "function"
        elif file_path:
            selected_mode = "file"
        else:
            selected_mode = "repository"

    out_file = file_path
    out_function = function_name

    if interactive:
        if selected_mode in {"file", "function"}:
            out_file = prompt_text(
                "File path",
                out_file or remembered.get("file_path", ""),
            )
        if selected_mode == "function":
            out_function = prompt_text(
                "Function name",
                out_function or remembered.get("function_name", ""),
            )

    if out_file:
        out_file = resolve_repo_path(repository_path, out_file)

    if selected_mode == "function":
        instruction = (
            f"Generate pytest unit tests for function {out_function}"
            + (f" in {out_file}" if out_file else "")
            + "."
        )
    elif selected_mode == "file":
        instruction = (
            f"Generate pytest unit tests for {out_file}, covering "
            "functions, methods, edge cases, invalid inputs, and "
            "common failure scenarios."
        )
    else:
        instruction = (
            "Generate pytest unit tests for this repository, covering "
            "functions, methods, edge cases, invalid inputs, and "
            "common failure scenarios."
        )

    return {
        "mode": selected_mode,
        "instruction": instruction,
        "file_path": out_file,
        "function_name": out_function,
    }


def record_memory_message(
    memory: Optional[ConversationMemory],
    role: str,
    content: str,
) -> None:
    """
    Append one short turn to ConversationMemory when available.

    Uses the existing ``add_message`` API so summarization and
    MemoryStore persistence run automatically. Content is truncated so
    full agent reports never enter memory.

    Args:
        memory: Session ConversationMemory, or None to skip.
        role: Message role (``user`` or ``assistant``).
        content: Short summary text to store.
    """
    if memory is None:
        return
    text = (content or "").strip()
    if not text:
        return
    if len(text) > _MAX_MEMORY_CONTENT_CHARS:
        text = text[: _MAX_MEMORY_CONTENT_CHARS - 3].rstrip() + "..."
    memory.add_message(ModelMessage(role=role, content=text))


def record_repository_loaded(
    memory: Optional[ConversationMemory],
    reference: str,
    repository_path: str,
) -> None:
    """
    Record the selected repository for this CLI session.

    Args:
        memory: Session ConversationMemory, or None to skip.
        reference: Original path or URL the user supplied.
        repository_path: Resolved local path used by agents.
    """
    if memory is None:
        return
    memory.metadata["repository_reference"] = reference
    memory.metadata["repository_path"] = repository_path
    record_memory_message(memory, "user", f"Repository: {reference}")
    record_memory_message(memory, "assistant", "Repository loaded.")


def summarize_analysis_for_memory(report: object) -> str:
    """
    Build a short analysis summary for ConversationMemory.

    Args:
        report: AnalysisReport-like object with finding lists.

    Returns:
        Compact counts only — never the full report body.
    """
    static_count = len(getattr(report, "static_findings", []) or [])
    llm_count = len(getattr(report, "llm_findings", []) or [])
    static_label = "finding" if static_count == 1 else "findings"
    llm_label = "finding" if llm_count == 1 else "findings"
    return (
        f"{static_count} static {static_label}. "
        f"{llm_count} grounded LLM {llm_label}."
    )


def summarize_documentation_for_memory(result: DocumentationResult) -> str:
    """
    Build a short documentation summary for ConversationMemory.

    Args:
        result: Documentation agent structured result.

    Returns:
        One-line outcome — never the README or docstring body.
    """
    name = (result.function_name or "").strip()
    if not name or name.upper() == "README":
        return "README generated."
    return f"Documentation generated for {name}."


def summarize_testing_for_memory(result: TestingResult) -> str:
    """
    Build a short testing summary for ConversationMemory.

    Args:
        result: Testing agent structured result.

    Returns:
        One-line outcome — never generated test source.
    """
    module_count = len(result.generated_tests or {})
    module_label = "module" if module_count == 1 else "modules"
    return f"Generated tests for {module_count} {module_label}."


def run_code_analysis(
    agent: CodeAnalysisAgent,
    repository_path: str,
    question: str,
    color: Optional[bool],
    memory: Optional[ConversationMemory] = None,
) -> None:
    """Run the existing CodeAnalysisAgent and print its report."""
    record_memory_message(
        memory,
        "user",
        f"Run Code Analysis\nQuestion: {question}",
    )
    print(f"\nRunning Code Analysis Agent on {repository_path} ...")
    report = agent.analyze_repository(
        repository_path=repository_path,
        question=question,
    )
    record_memory_message(
        memory, "assistant", summarize_analysis_for_memory(report)
    )
    print_report(report, color=color)


def run_documentation_agent(
    agent: DocumentationAgent,
    repository_path: str,
    memory: Optional[ConversationMemory] = None,
    *,
    interactive: bool = True,
    file_path: str = "",
    function_name: str = "",
    class_name: str = "",
    write_to_disk: bool = False,
    replace_existing: bool = False,
    mode: str = "",
    tracer: Optional[Tracer] = None,
) -> None:
    """Run the existing DocumentationAgent and print its result."""
    options = collect_documentation_options(
        repository_path,
        memory,
        interactive=interactive,
        file_path=file_path,
        function_name=function_name,
        class_name=class_name,
        write_to_disk=write_to_disk,
        replace_existing=replace_existing,
        mode=mode,
    )
    if options is None:
        return

    store_memory_target(
        memory,
        file_path=str(options.get("file_path") or ""),
        function_name=str(options.get("function_name") or ""),
        class_name=str(options.get("class_name") or ""),
    )
    record_memory_message(
        memory,
        "user",
        f"Generate documentation\nMode: {options['mode']}\n"
        f"Target: {options.get('file_path') or options.get('class_name') or 'repository'}",
    )

    print_progress("Preparing documentation request...")
    context = {
        "repo_path": repository_path,
        "repository_path": repository_path,
        "doc_type": options["doc_type"],
        "write_to_disk": bool(options["write_to_disk"]),
        "replace_existing": bool(options["replace_existing"]),
    }
    if options.get("file_path"):
        context["file_path"] = options["file_path"]
    if options.get("function_name"):
        context["function_name"] = options["function_name"]
    if options.get("class_name"):
        context["class_name"] = options["class_name"]

    start_count = tracer_event_count(tracer)
    print_progress("Generating documentation...")
    response = agent.handle(
        AgentRequest(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.DOCUMENTATION,
            instruction=str(options["instruction"]),
            context=context,
        )
    )
    emit_progress_from_tracer(tracer, start_count=start_count)

    if not response.success and not isinstance(response.output, DocumentationResult):
        errors = response.errors or ["Documentation generation failed."]
        record_memory_message(
            memory, "assistant", "Documentation generation failed."
        )
        print("Documentation Agent error:")
        for error in errors:
            print(f"  - {error}")
        return

    if isinstance(response.output, DocumentationResult):
        record_memory_message(
            memory,
            "assistant",
            summarize_documentation_for_memory(response.output),
        )
        target_label = (
            options.get("file_path")
            or options.get("class_name")
            or options.get("function_name")
            or "repository"
        )
        print()
        print(
            format_cli_documentation_result(
                response.output, requested_target=str(target_label)
            )
        )
        if not response.success and response.errors:
            print("\nNotes:")
            for error in response.errors:
                print(f"  - {error}")
    else:
        record_memory_message(memory, "assistant", "Documentation generated.")
        print(response.output)
    print_progress("Done.")


def run_testing_agent(
    agent: TestingAgent,
    repository_path: str,
    *,
    interactive: bool = True,
    memory: Optional[ConversationMemory] = None,
    file_path: str = "",
    function_name: str = "",
    mode: str = "",
    tracer: Optional[Tracer] = None,
) -> None:
    """
    Run the existing TestingAgent and print its result.

    Args:
        agent: Testing agent instance.
        repository_path: Repository to generate tests for.
        interactive: When True, show menus / optional source prompts.
        memory: Optional ConversationMemory for short session turns.
        file_path: Optional non-interactive file target.
        function_name: Optional non-interactive function target.
        mode: Optional forced mode (repository/file/function).
        tracer: Optional tracer used for progress event mapping.
    """
    options = collect_testing_options(
        repository_path,
        memory,
        interactive=interactive,
        file_path=file_path,
        function_name=function_name,
        mode=mode,
    )
    if options is None:
        return

    store_memory_target(
        memory,
        file_path=str(options.get("file_path") or ""),
        function_name=str(options.get("function_name") or ""),
    )
    record_memory_message(
        memory,
        "user",
        f"Generate tests\nMode: {options['mode']}\n"
        f"Target: {options.get('file_path') or 'repository'}",
    )

    print_progress("Preparing testing request...")
    context = {
        "repo_path": repository_path,
        "repository_path": repository_path,
    }
    if options.get("file_path"):
        context["file_path"] = options["file_path"]
    if options.get("function_name"):
        context["function_name"] = options["function_name"]

    start_count = tracer_event_count(tracer)
    print_progress("Generating tests...")
    response = agent.handle(
        AgentRequest(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.TESTING,
            instruction=str(options["instruction"]),
            context=context,
        )
    )
    emit_progress_from_tracer(tracer, start_count=start_count)

    if not response.success and not isinstance(response.output, TestingResult):
        errors = response.errors or ["Test generation failed."]
        record_memory_message(memory, "assistant", "Test generation failed.")
        print("Testing Agent error:")
        for error in errors:
            print(f"  - {error}")
        return

    if not isinstance(response.output, TestingResult):
        record_memory_message(memory, "assistant", "Tests generated.")
        print(response.output)
        print_progress("Done.")
        return

    record_memory_message(
        memory, "assistant", summarize_testing_for_memory(response.output)
    )
    print()
    print(format_cli_testing_result(response.output, tracer=tracer))
    if response.errors:
        print("\nNotes:")
        for error in response.errors:
            print(f"  - {error}")

    failing = failing_test_sources(response.output)
    if interactive and failing and ask_yes_no("View failing test source? (y/n): "):
        print("\nFailing test source")
        print("-------------------")
        for name, source in sorted(failing.items()):
            print(f"\n--- {name} ---")
            print((source or "").rstrip() or "(empty file)")
            print()
    print_progress("Done.")


def resolve_agents(
    supervisor: Supervisor,
) -> Tuple[CodeAnalysisAgent, DocumentationAgent, TestingAgent]:
    """
    Pull the three concrete agents out of the Supervisor.

    Args:
        supervisor: Wired Supervisor holding the agents.

    Returns:
        The Code Analysis, Documentation, and Testing agents.

    Raises:
        SystemExit: If any expected agent is missing or mistyped.
    """
    analysis_agent = supervisor.agents[AgentType.CODE_ANALYSIS]
    documentation_agent = supervisor.agents[AgentType.DOCUMENTATION]
    testing_agent = supervisor.agents[AgentType.TESTING]

    if not isinstance(analysis_agent, CodeAnalysisAgent):
        print("Error: Code Analysis Agent is not available.", file=sys.stderr)
        raise SystemExit(1)
    if not isinstance(documentation_agent, DocumentationAgent):
        print("Error: Documentation Agent is not available.", file=sys.stderr)
        raise SystemExit(1)
    if not isinstance(testing_agent, TestingAgent):
        print("Error: Testing Agent is not available.", file=sys.stderr)
        raise SystemExit(1)

    return analysis_agent, documentation_agent, testing_agent


def interactive_loop(
    supervisor: Supervisor,
    repository_path: str,
    question: str,
    color: Optional[bool],
) -> None:
    """
    Repeatedly offer the agent menu until the user exits.

    Args:
        supervisor: Wired Supervisor holding the three agents.
        repository_path: Prepared local repository path for the session.
        question: Default analysis question.
        color: Color override for the analysis report.
    """
    analysis_agent, documentation_agent, testing_agent = resolve_agents(
        supervisor
    )
    memory = supervisor.conversation_memory

    print(f"\nRepository ready: {repository_path}")
    _cli_trace(
        getattr(supervisor, "tracer", None),
        "selected_agents",
        agents=["interactive_menu"],
        mode="interactive",
    )

    while True:
        choice = prompt_choice()

        if choice == "1":
            _cli_trace(
                getattr(supervisor, "tracer", None),
                "selected_agents",
                agents=["analysis"],
                mode="interactive",
            )
            try:
                run_code_analysis(
                    analysis_agent,
                    repository_path,
                    question,
                    color,
                    memory=memory,
                )
            except Exception as exc:
                record_memory_message(
                    memory, "assistant", f"Code Analysis failed: {exc}"
                )
                print(f"Code Analysis Agent error:\n  - {exc}")
            continue

        if choice == "2":
            _cli_trace(
                getattr(supervisor, "tracer", None),
                "selected_agents",
                agents=["documentation"],
                mode="interactive",
            )
            try:
                run_documentation_agent(
                    documentation_agent,
                    repository_path,
                    memory=memory,
                    interactive=True,
                    tracer=getattr(supervisor, "tracer", None),
                )
            except Exception as exc:
                record_memory_message(
                    memory, "assistant", f"Documentation failed: {exc}"
                )
                print(f"Documentation Agent error:\n  - {exc}")
            continue

        if choice == "3":
            _cli_trace(
                getattr(supervisor, "tracer", None),
                "selected_agents",
                agents=["testing"],
                mode="interactive",
            )
            try:
                run_testing_agent(
                    testing_agent,
                    repository_path,
                    memory=memory,
                    interactive=True,
                    tracer=getattr(supervisor, "tracer", None),
                )
            except Exception as exc:
                record_memory_message(
                    memory, "assistant", f"Testing failed: {exc}"
                )
                print(f"Testing Agent error:\n  - {exc}")
            continue

        if choice == "4":
            print("Goodbye!")
            return

        print("Invalid selection. Please choose 1-4.")


def run_noninteractive(
    supervisor: Supervisor,
    repository_path: str,
    agent: str,
    question: str,
    color: Optional[bool],
    *,
    file_path: str = "",
    function_name: str = "",
    class_name: str = "",
    write_to_disk: bool = False,
    replace_existing: bool = False,
) -> None:
    """
    Run one agent (or all three) once and return.

    Args:
        supervisor: Wired Supervisor holding the three agents.
        repository_path: Prepared local repository path.
        agent: One of ``analysis``, ``documentation``, ``testing``,
            or ``all``.
        question: Analysis question used when Code Analysis runs.
        color: Color override for the analysis report.
        file_path: Optional target file for docs/testing.
        function_name: Optional target function for docs/testing.
        class_name: Optional target class for documentation.
        write_to_disk: Optional documentation write-back flag.
        replace_existing: Optional documentation replace flag.
    """
    analysis_agent, documentation_agent, testing_agent = resolve_agents(
        supervisor
    )
    memory = supervisor.conversation_memory
    tracer = getattr(supervisor, "tracer", None)
    selected = (
        ("analysis", "documentation", "testing")
        if agent == "all"
        else (agent,)
    )

    print(f"\nRepository ready: {repository_path}")
    print(f"Running agent(s): {', '.join(selected)}")
    _cli_trace(
        tracer,
        "selected_agents",
        agents=list(selected),
        mode="noninteractive",
    )

    for name in selected:
        try:
            if name == "analysis":
                run_code_analysis(
                    analysis_agent,
                    repository_path,
                    question,
                    color,
                    memory=memory,
                )
            elif name == "documentation":
                run_documentation_agent(
                    documentation_agent,
                    repository_path,
                    memory=memory,
                    interactive=False,
                    file_path=file_path,
                    function_name=function_name,
                    class_name=class_name,
                    write_to_disk=write_to_disk,
                    replace_existing=replace_existing,
                    tracer=tracer,
                )
            else:
                run_testing_agent(
                    testing_agent,
                    repository_path,
                    interactive=False,
                    memory=memory,
                    file_path=file_path,
                    function_name=function_name,
                    tracer=tracer,
                )
        except Exception as exc:
            record_memory_message(
                memory, "assistant", f"{name} agent failed: {exc}"
            )
            print(f"{name} agent error:\n  - {exc}")


def main(argv: Optional[List[str]] = None) -> None:
    """
    Prepare one repository, then run agents interactively or once.

    Without ``--agent``, the interactive menu is used. With ``--agent``,
    the selected agent runs once and the process exits. Temporary GitHub
    clones are cleaned up when the process exits. Short conversation
    turns are recorded on the Supervisor's ConversationMemory so history
    persists across runs via MemoryStore.

    Args:
        argv: Optional argument list for tests. Defaults to
            ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    reference = read_repository_reference(args.repository)

    color: Optional[bool] = None
    if args.color:
        color = True
    elif args.no_color:
        color = False

    print("Starting Codebase Assistant...")
    print_progress("Preparing repository...")
    supervisor = Supervisor(config=Config.load())
    print(supervisor.provider_status_message())
    _cli_trace(supervisor.tracer, "application_started")
    repository_path = prepare_repository(supervisor, reference)
    record_repository_loaded(
        supervisor.conversation_memory, reference, repository_path
    )

    try:
        if args.agent:
            run_noninteractive(
                supervisor,
                repository_path,
                args.agent,
                args.question,
                color,
                file_path=str(getattr(args, "file", "") or ""),
                function_name=str(getattr(args, "function", "") or ""),
                class_name=str(getattr(args, "class_name", "") or ""),
                write_to_disk=bool(getattr(args, "write_to_disk", False)),
                replace_existing=bool(
                    getattr(args, "replace_existing", False)
                ),
            )
        else:
            interactive_loop(
                supervisor, repository_path, args.question, color
            )
    finally:
        _cli_trace(supervisor.tracer, "application_exit")
        cleanup_temporary_clones()


if __name__ == "__main__":
    main()
