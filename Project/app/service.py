"""
app/service.py
==============

Shared non-interactive orchestration for CLI-adjacent frontends
(Streamlit). Wraps Supervisor / agents and returns result objects
instead of printing. Does not own agent logic.
"""

from __future__ import annotations

import atexit
import os
import shutil
import stat
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from codebase_assistant.agents.code_analysis_agent import (  # noqa: E402
    CodeAnalysisAgent,
    CodeAnalysisReport,
)
from codebase_assistant.agents.documentation_agent import (  # noqa: E402
    DocumentationAgent,
)
from codebase_assistant.agents.testing_agent import TestingAgent  # noqa: E402
from codebase_assistant.config import Config  # noqa: E402
from codebase_assistant.exceptions.tool_exceptions import (  # noqa: E402
    InvalidRepositoryURLError,
    RepositoryCloneError,
)
from codebase_assistant.schemas.schemas import (  # noqa: E402
    AgentRequest,
    AgentType,
    DocumentationResult,
    TestingResult,
)
from codebase_assistant.supervisor import Supervisor  # noqa: E402
from codebase_assistant.tools.github_tools import GitHubTools  # noqa: E402
from codebase_assistant.tracing.events import TraceEventType  # noqa: E402
from codebase_assistant.tracing.tracer import Tracer  # noqa: E402

#: Clones made during this process, keyed by canonical repository URL.
_CLONE_CACHE: Dict[str, str] = {}

#: Temporary directories holding those clones, removed on exit.
_TEMPORARY_ROOTS: List[str] = []


class RepositoryPathError(ValueError):
    """Raised when a local repository reference is missing or invalid."""


class AgentRunError(RuntimeError):
    """Raised when an agent run fails without a usable result object."""

    def __init__(self, message: str, *, errors: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.errors = list(errors or [])


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
    except OSError:
        pass


def normalize_repository_url(repo_url: str) -> str:
    """Build a canonical `host/owner/name` key for a remote repository."""
    parsed = urlparse(str(repo_url).strip())
    segments = [segment for segment in parsed.path.split("/") if segment]
    if segments and segments[-1].lower().endswith(".git"):
        segments[-1] = segments[-1][: -len(".git")]
    return "/".join([parsed.netloc.lower(), *(s.lower() for s in segments)])


def _service_trace(
    tracer: Optional[Tracer],
    name: str,
    *,
    success: Optional[bool] = True,
    **metadata: object,
) -> None:
    """Record a lifecycle event; never raises."""
    if tracer is None:
        return
    try:
        tracer.record(
            TraceEventType.LIFECYCLE,
            name,
            component="Service",
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

    Raises:
        InvalidRepositoryURLError: If the URL fails validation.
        RepositoryCloneError: If the repository cannot be cloned.
    """
    key = normalize_repository_url(repo_url)
    cached = _CLONE_CACHE.get(key)
    if cached and os.path.isdir(cached):
        _service_trace(
            tracer,
            "repository_cloned",
            repository_url=repo_url,
            repository_path=cached,
            reused=True,
        )
        return cached

    github_tools.validate_repository(repo_url)

    temporary_root = tempfile.mkdtemp(prefix="codebase_assistant_clone_")
    _TEMPORARY_ROOTS.append(temporary_root)
    destination = os.path.join(temporary_root, "repo")

    github_tools.clone_repository(repo_url, destination)
    _service_trace(
        tracer,
        "repository_cloned",
        repository_url=repo_url,
        repository_path=destination,
        reused=False,
    )

    _CLONE_CACHE[key] = destination
    return destination


def cleanup_temporary_clones() -> None:
    """Remove every temporary clone created during this process."""
    while _TEMPORARY_ROOTS:
        remove_temporary_tree(_TEMPORARY_ROOTS.pop())
    _CLONE_CACHE.clear()


atexit.register(cleanup_temporary_clones)


def resolve_repository_path(raw_path: str) -> str:
    """
    Resolve and validate a local repository path.

    Relative paths are tried against the process cwd first, then against
    the Project root (parent of ``app/``), so defaults like
    ``examples/demo_repo`` work when Streamlit's cwd differs.

    Raises:
        RepositoryPathError: If the path is missing or not a directory.
    """
    candidate = (raw_path or "").strip()
    if not candidate:
        raise RepositoryPathError("Repository path is empty.")

    expanded = os.path.expanduser(candidate)
    candidates = []
    if os.path.isabs(expanded):
        candidates.append(os.path.abspath(expanded))
    else:
        candidates.append(os.path.abspath(expanded))
        project_candidate = os.path.abspath(os.path.join(_PROJECT_ROOT, expanded))
        if project_candidate not in candidates:
            candidates.append(project_candidate)

    resolved = None
    for path in candidates:
        if os.path.isdir(path):
            resolved = path
            break

    if resolved is None:
        raise RepositoryPathError(
            f"Path does not exist or is not a directory: {candidates[0]}"
        )
    return resolved


def resolve_repo_path(repository_path: str, raw_path: str) -> str:
    """Resolve a user path against the repository root when relative."""
    text = (raw_path or "").strip()
    if not text:
        return ""
    expanded = os.path.expanduser(text)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(repository_path, expanded))


def build_supervisor(config: Optional[Config] = None) -> Supervisor:
    """Construct a Supervisor with default Config (loads .env)."""
    return Supervisor(config=config or Config.load())


def provider_status_message(config: Optional[Config] = None) -> str:
    """
    Lightweight provider summary without constructing a full Supervisor.

    Prefer OpenRouter when an API key is present; otherwise mention Ollama.
    """
    cfg = config or Config.load()
    key = (getattr(cfg, "openrouter_api_key", None) or "").strip()
    model = getattr(cfg, "openrouter_model", None) or getattr(
        cfg, "model_name", "model"
    )
    if key:
        label = str(model)
        if "gemma-3-27b" in label:
            label = "Gemma 3 27B"
        elif "/" in label:
            label = label.split("/", 1)[-1]
        return f"Using OpenRouter ({label})"
    return "Using Ollama (no OPENROUTER_API_KEY)"


def prepare_repository(
    reference: str,
    *,
    supervisor: Optional[Supervisor] = None,
    github_token: Optional[str] = None,
) -> str:
    """
    Resolve a local path or clone a GitHub URL once for the session.

    A full Supervisor is optional — local paths need none, remotes only
    need GitHubTools. Prefer calling without a Supervisor from Streamlit
    so embedding models are never loaded in the UI process.

    Raises:
        RepositoryPathError: Local path invalid.
        InvalidRepositoryURLError: Remote URL invalid.
        RepositoryCloneError: Clone failed.
    """
    ref = (reference or "").strip()
    if not ref:
        raise RepositoryPathError("Repository reference is empty.")

    tracer = getattr(supervisor, "tracer", None) if supervisor is not None else None
    if not GitHubTools.is_remote_reference(ref):
        path = resolve_repository_path(ref)
        _service_trace(
            tracer,
            "repository_selected",
            repository_reference=ref,
            repository_path=path,
            remote=False,
        )
        return path

    if supervisor is not None:
        github_tools = supervisor.github_tools
    else:
        token = github_token
        if token is None:
            token = Config.load().github_token
        github_tools = GitHubTools(token=token)

    path = clone_or_reuse_repository(github_tools, ref, tracer=tracer)
    _service_trace(
        tracer,
        "repository_selected",
        repository_reference=ref,
        repository_path=path,
        remote=True,
    )
    return path


def resolve_agents(
    supervisor: Supervisor,
) -> Tuple[CodeAnalysisAgent, DocumentationAgent, TestingAgent]:
    """Pull the three concrete agents out of the Supervisor."""
    analysis_agent = supervisor.agents[AgentType.CODE_ANALYSIS]
    documentation_agent = supervisor.agents[AgentType.DOCUMENTATION]
    testing_agent = supervisor.agents[AgentType.TESTING]

    if not isinstance(analysis_agent, CodeAnalysisAgent):
        raise AgentRunError("Code Analysis Agent is not available.")
    if not isinstance(documentation_agent, DocumentationAgent):
        raise AgentRunError("Documentation Agent is not available.")
    if not isinstance(testing_agent, TestingAgent):
        raise AgentRunError("Testing Agent is not available.")

    return analysis_agent, documentation_agent, testing_agent


def _documentation_options(
    repository_path: str,
    *,
    mode: str = "",
    file_path: str = "",
    function_name: str = "",
    class_name: str = "",
    write_to_disk: bool = False,
    replace_existing: bool = False,
) -> Dict[str, Any]:
    """Build non-interactive documentation request options."""
    selected_mode = (mode or "").strip().lower()
    if not selected_mode:
        if class_name:
            selected_mode = "class"
        elif function_name:
            selected_mode = "function"
        elif file_path:
            selected_mode = "file"
        else:
            selected_mode = "readme"

    out_file = resolve_repo_path(repository_path, file_path) if file_path else ""
    out_function = (function_name or "").strip()
    out_class = (class_name or "").strip()

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
        "write_to_disk": bool(write_to_disk),
        "replace_existing": bool(replace_existing),
    }


def _testing_options(
    repository_path: str,
    *,
    mode: str = "",
    file_path: str = "",
    function_name: str = "",
) -> Dict[str, Any]:
    """Build non-interactive testing request options."""
    selected_mode = (mode or "").strip().lower()
    if not selected_mode:
        if function_name:
            selected_mode = "function"
        elif file_path:
            selected_mode = "file"
        else:
            selected_mode = "repository"

    out_file = resolve_repo_path(repository_path, file_path) if file_path else ""
    out_function = (function_name or "").strip()

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


def run_analysis(
    supervisor: Supervisor,
    repository_path: str,
    question: str = "Find bugs and potential issues",
) -> CodeAnalysisReport:
    """Run Code Analysis and return the report."""
    analysis_agent, _, _ = resolve_agents(supervisor)
    return analysis_agent.analyze_repository(
        repository_path=repository_path,
        question=question,
    )


def run_documentation(
    supervisor: Supervisor,
    repository_path: str,
    *,
    mode: str = "",
    file_path: str = "",
    function_name: str = "",
    class_name: str = "",
    write_to_disk: bool = False,
    replace_existing: bool = False,
) -> DocumentationResult:
    """
    Run Documentation and return the result.

    Raises:
        AgentRunError: When the agent fails without a DocumentationResult.
    """
    _, documentation_agent, _ = resolve_agents(supervisor)
    options = _documentation_options(
        repository_path,
        mode=mode,
        file_path=file_path,
        function_name=function_name,
        class_name=class_name,
        write_to_disk=write_to_disk,
        replace_existing=replace_existing,
    )

    context: Dict[str, Any] = {
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

    response = documentation_agent.handle(
        AgentRequest(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.DOCUMENTATION,
            instruction=str(options["instruction"]),
            context=context,
        )
    )

    if isinstance(response.output, DocumentationResult):
        return response.output

    errors = response.errors or ["Documentation generation failed."]
    raise AgentRunError(errors[0], errors=errors)


def run_testing(
    supervisor: Supervisor,
    repository_path: str,
    *,
    mode: str = "",
    file_path: str = "",
    function_name: str = "",
) -> TestingResult:
    """
    Run Testing and return the result.

    Raises:
        AgentRunError: When the agent fails without a TestingResult.
    """
    _, _, testing_agent = resolve_agents(supervisor)
    options = _testing_options(
        repository_path,
        mode=mode,
        file_path=file_path,
        function_name=function_name,
    )

    context: Dict[str, Any] = {
        "repo_path": repository_path,
        "repository_path": repository_path,
    }
    if options.get("file_path"):
        context["file_path"] = options["file_path"]
    if options.get("function_name"):
        context["function_name"] = options["function_name"]

    response = testing_agent.handle(
        AgentRequest(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.TESTING,
            instruction=str(options["instruction"]),
            context=context,
        )
    )

    if isinstance(response.output, TestingResult):
        return response.output

    errors = response.errors or ["Test generation failed."]
    raise AgentRunError(errors[0], errors=errors)
