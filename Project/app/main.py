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

Without ``--agent``, the repository is prepared once and an interactive
menu dispatches to Code Analysis, Documentation, or Testing. With
``--agent``, the chosen agent (or all three) runs once and the process
exits.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import stat
import sys
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the CLI.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments with an optional repository path, question,
        and agent selection.
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


def prompt_choice() -> str:
    """
    Show the agent menu and return the user's raw choice.

    Returns:
        The stripped choice string, or empty on EOF/interrupt.
    """
    try:
        return input(MENU_TEXT).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "4"


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
) -> None:
    """Run the existing DocumentationAgent and print its result."""
    record_memory_message(memory, "user", "Generate documentation")
    print(f"\nRunning Documentation Agent on {repository_path} ...")
    response = agent.handle(
        AgentRequest(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.DOCUMENTATION,
            instruction="Generate a README summary for this repository.",
            context={
                "repo_path": repository_path,
                "repository_path": repository_path,
                "doc_type": "readme",
            },
        )
    )

    if not response.success:
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
        print_documentation_result(response.output)
    else:
        record_memory_message(memory, "assistant", "Documentation generated.")
        print(response.output)


def run_testing_agent(
    agent: TestingAgent,
    repository_path: str,
    *,
    interactive: bool = True,
    memory: Optional[ConversationMemory] = None,
) -> None:
    """
    Run the existing TestingAgent and print its result.

    Args:
        agent: Testing agent instance.
        repository_path: Repository to generate tests for.
        interactive: When True, optionally prompt to view generated
            source. When False (``--agent`` mode), print the summary
            only so the CLI never blocks on stdin.
        memory: Optional ConversationMemory for short session turns.
    """
    record_memory_message(memory, "user", "Generate tests")
    print(f"\nRunning Testing Agent on {repository_path} ...")
    response = agent.handle(
        AgentRequest(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.TESTING,
            instruction=(
                "Generate pytest unit tests for this repository, covering "
                "functions, methods, edge cases, invalid inputs, and "
                "common failure scenarios."
            ),
            context={
                "repo_path": repository_path,
                "repository_path": repository_path,
            },
        )
    )

    if not response.success:
        errors = response.errors or ["Test generation failed."]
        record_memory_message(memory, "assistant", "Test generation failed.")
        print("Testing Agent error:")
        for error in errors:
            print(f"  - {error}")
        return

    if not isinstance(response.output, TestingResult):
        record_memory_message(memory, "assistant", "Tests generated.")
        print(response.output)
        return

    record_memory_message(
        memory, "assistant", summarize_testing_for_memory(response.output)
    )
    print_testing_result(response.output, include_source=False)
    if (
        interactive
        and response.output.generated_tests
        and ask_yes_no("View generated test source? (y/n): ")
    ):
        print_testing_result(response.output, include_source=True)


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
                    documentation_agent, repository_path, memory=memory
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
                    testing_agent, repository_path, memory=memory
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
    """
    analysis_agent, documentation_agent, testing_agent = resolve_agents(
        supervisor
    )
    memory = supervisor.conversation_memory
    selected = (
        ("analysis", "documentation", "testing")
        if agent == "all"
        else (agent,)
    )

    print(f"\nRepository ready: {repository_path}")
    print(f"Running agent(s): {', '.join(selected)}")
    _cli_trace(
        getattr(supervisor, "tracer", None),
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
                    documentation_agent, repository_path, memory=memory
                )
            else:
                run_testing_agent(
                    testing_agent,
                    repository_path,
                    interactive=False,
                    memory=memory,
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
    supervisor = Supervisor(config=Config.load())
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
