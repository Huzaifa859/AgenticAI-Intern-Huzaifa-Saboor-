"""
test_cli_menus_and_flags.py
============================

Coverage for interactive documentation/testing menus, targeting flags,
progress helpers, and memory-based follow-up defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _PROJECT_ROOT / "app"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import main as app_main  # noqa: E402
from codebase_assistant.agents.documentation_agent import (  # noqa: E402
    DocumentationAgent,
)
from codebase_assistant.agents.testing_agent import TestingAgent  # noqa: E402
from codebase_assistant.memory.conversation_memory import (  # noqa: E402
    ConversationMemory,
)
from codebase_assistant.schemas.schemas import (  # noqa: E402
    AbstentionResult,
    AgentResponse,
    AgentType,
    DocumentationResult,
    TestingResult,
)
from codebase_assistant.tracing.tracer import Tracer  # noqa: E402


TestingResult.__test__ = False


def test_parse_args_accepts_targeting_and_write_flags() -> None:
    """New CLI flags should parse without changing agent schemas."""
    args = app_main.parse_args(
        [
            "repo",
            "--agent",
            "documentation",
            "--file",
            "app/auth.py",
            "--function",
            "authenticate",
            "--class",
            "UserService",
            "--write-to-disk",
            "--replace-existing",
        ]
    )
    assert args.file == "app/auth.py"
    assert args.function == "authenticate"
    assert args.class_name == "UserService"
    assert args.write_to_disk is True
    assert args.replace_existing is True


def test_documentation_submenu_repository_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Documentation menu option 1 builds a repository README request."""
    inputs = iter(["1", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    options = app_main.collect_documentation_options(
        str(tmp_path), None, interactive=True
    )
    assert options is not None
    assert options["mode"] == "readme"
    assert options["doc_type"] == "readme"
    assert options["write_to_disk"] is False


def test_documentation_submenu_file_function_class_and_writeback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """File/function/class modes collect paths and write-back answers."""
    # File mode with write + replace.
    inputs = iter(["2", "app/auth.py", "y", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    file_opts = app_main.collect_documentation_options(
        str(tmp_path), None, interactive=True
    )
    assert file_opts is not None
    assert file_opts["mode"] == "file"
    assert file_opts["doc_type"] == "module"
    assert file_opts["file_path"].replace("\\", "/").endswith("app/auth.py")
    assert file_opts["write_to_disk"] is True
    assert file_opts["replace_existing"] is True

    # Function mode.
    inputs = iter(["3", "app/auth.py", "authenticate", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    fn_opts = app_main.collect_documentation_options(
        str(tmp_path), None, interactive=True
    )
    assert fn_opts is not None
    assert fn_opts["mode"] == "function"
    assert fn_opts["doc_type"] == "docstring"
    assert fn_opts["function_name"] == "authenticate"

    # Class mode.
    inputs = iter(["4", "app/auth.py", "UserService", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    class_opts = app_main.collect_documentation_options(
        str(tmp_path), None, interactive=True
    )
    assert class_opts is not None
    assert class_opts["mode"] == "class"
    assert class_opts["class_name"] == "UserService"


def test_documentation_submenu_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Back should cancel documentation without building options."""
    monkeypatch.setattr("builtins.input", lambda _: "5")
    assert (
        app_main.collect_documentation_options(
            str(tmp_path), None, interactive=True
        )
        is None
    )


def test_testing_submenu_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Testing menu supports repository/file/function and Back."""
    monkeypatch.setattr("builtins.input", lambda _: "1")
    repo_opts = app_main.collect_testing_options(
        str(tmp_path), None, interactive=True
    )
    assert repo_opts is not None
    assert repo_opts["mode"] == "repository"

    inputs = iter(["2", "math_utils.py"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    file_opts = app_main.collect_testing_options(
        str(tmp_path), None, interactive=True
    )
    assert file_opts is not None
    assert file_opts["mode"] == "file"
    assert file_opts["file_path"].replace("\\", "/").endswith("math_utils.py")

    inputs = iter(["3", "math_utils.py", "add"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    fn_opts = app_main.collect_testing_options(
        str(tmp_path), None, interactive=True
    )
    assert fn_opts is not None
    assert fn_opts["mode"] == "function"
    assert fn_opts["function_name"] == "add"

    monkeypatch.setattr("builtins.input", lambda _: "4")
    assert (
        app_main.collect_testing_options(str(tmp_path), None, interactive=True)
        is None
    )


def test_memory_follow_up_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Empty prompts reuse last file/function from ConversationMemory."""
    memory = ConversationMemory(max_messages=20, conversation_id="cli-follow")
    app_main.store_memory_target(
        memory,
        file_path=str(tmp_path / "app" / "auth.py"),
        function_name="authenticate",
        class_name="UserService",
    )
    inputs = iter(["3", "", "", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    options = app_main.collect_documentation_options(
        str(tmp_path), memory, interactive=True
    )
    assert options is not None
    assert options["file_path"].replace("\\", "/").endswith("auth.py")
    assert options["function_name"] == "authenticate"


def test_progress_messages_from_tracer(capsys: pytest.CaptureFixture[str]) -> None:
    """Progress helper maps tracer events to user-facing lines."""
    tracer = Tracer(run_id="cli-progress")
    tracer.record("lifecycle", "indexing")
    tracer.record("lifecycle", "documentation_grounding_started")
    tracer.record("lifecycle", "documentation_finished")
    app_main.emit_progress_from_tracer(tracer, start_count=0)
    out = capsys.readouterr().out
    assert "Indexing repository..." in out
    assert "Grounding documentation..." in out
    assert "Done." in out


def test_format_cli_documentation_result_shows_abstention_and_writeback() -> None:
    """Documentation display includes target, write-back, and abstention."""
    result = DocumentationResult(
        file_path="app/auth.py",
        function_name="authenticate",
        summary="Auth helper.\n\nWrite-back wrote docstring in auth.py.",
        parameters=[],
        returns="",
        example_usage="",
        abstention=AbstentionResult(
            reason="target not found",
            confidence=1.0,
            evidence_available=[],
            recommended_next_steps=[],
        ),
    )
    text = app_main.format_cli_documentation_result(
        result, requested_target="app/auth.py"
    )
    assert "Documentation Summary" in text
    assert "app/auth.py" in text
    assert "Write-back wrote docstring" in text
    assert "Grounded:" in text
    assert "Abstention:" in text
    assert "target not found" in text


def test_format_cli_testing_result_parses_summary_fields() -> None:
    """Testing display extracts execution, coverage, repair, imports."""
    result = TestingResult(
        summary=(
            "Generated tests.\n"
            "Import validation: removed 2 unused invalid import(s).\n"
            "Execution: 5 passed, 1 failed, 0 skipped, 0 errors in 0.20s.\n"
            "Repair: attempted one fix iteration.\n"
            "Coverage: 84.0% lines (measured)."
        ),
        generated_tests={
            "test_a.py": "def test_a():\n    assert True\n",
            "test_b.py": "def test_b():\n    assert False\n",
        },
        coverage_estimate=0.84,
    )
    tracer = Tracer(run_id="cli-test-fmt")
    tracer.record("lifecycle", "testing_repair_started")
    tracer.record("lifecycle", "testing_repair_finished", success=True)
    text = app_main.format_cli_testing_result(result, tracer=tracer)
    assert "Generated:" in text
    assert "2 test files" in text
    assert "5 passed" in text
    assert "1 failed" in text
    assert "84%" in text
    assert "Attempted once" in text
    assert "Removed 2 invalid imports" in text


def test_run_documentation_agent_passes_context_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-interactive docs should pass existing context keys to handle()."""
    agent = MagicMock(spec=DocumentationAgent)
    agent.handle.return_value = AgentResponse(
        task_id="d1",
        agent_type=AgentType.DOCUMENTATION,
        success=True,
        output=DocumentationResult(
            file_path="app/auth.py",
            function_name="authenticate",
            summary="Auth docs.",
            parameters=[],
            returns="",
            example_usage="",
        ),
    )
    app_main.run_documentation_agent(
        agent,
        str(tmp_path),
        interactive=False,
        file_path="app/auth.py",
        function_name="authenticate",
        write_to_disk=True,
        replace_existing=False,
    )
    request = agent.handle.call_args.args[0]
    assert request.context["file_path"].replace("\\", "/").endswith("auth.py")
    assert request.context["function_name"] == "authenticate"
    assert request.context["write_to_disk"] is True
    assert request.context["doc_type"] == "docstring"
    out = capsys.readouterr().out
    assert "Documentation Summary" in out
    assert "Generating documentation..." in out


def test_run_testing_agent_passes_file_and_function(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-interactive testing should pass file_path/function_name context."""
    agent = MagicMock(spec=TestingAgent)
    agent.handle.return_value = AgentResponse(
        task_id="t1",
        agent_type=AgentType.TESTING,
        success=True,
        output=TestingResult(
            summary="Execution: 1 passed, 0 failed, 0 skipped, 0 errors in 0.01s.",
            generated_tests={"test_math_utils.py": "def test_add():\n    assert True\n"},
            coverage_estimate=0.9,
        ),
    )
    app_main.run_testing_agent(
        agent,
        str(tmp_path),
        interactive=False,
        file_path="math_utils.py",
        function_name="add",
    )
    request = agent.handle.call_args.args[0]
    assert request.context["file_path"].replace("\\", "/").endswith(
        "math_utils.py"
    )
    assert request.context["function_name"] == "add"
    out = capsys.readouterr().out
    assert "Testing Summary" in out
    assert "Generating tests..." in out


def test_run_noninteractive_forwards_new_flags(tmp_path: Path) -> None:
    """--agent path should forward targeting flags into runners."""
    from codebase_assistant.agents.code_analysis_agent import CodeAnalysisAgent
    from codebase_assistant.schemas.schemas import AgentType

    supervisor = MagicMock()
    supervisor.conversation_memory = None
    supervisor.tracer = None
    supervisor.agents = {
        AgentType.CODE_ANALYSIS: MagicMock(spec=CodeAnalysisAgent),
        AgentType.DOCUMENTATION: MagicMock(spec=DocumentationAgent),
        AgentType.TESTING: MagicMock(spec=TestingAgent),
    }
    with patch.object(app_main, "run_documentation_agent") as run_d:
        app_main.run_noninteractive(
            supervisor,
            str(tmp_path),
            "documentation",
            "Find bugs",
            color=False,
            file_path="app/auth.py",
            function_name="authenticate",
            write_to_disk=True,
            replace_existing=True,
        )
    assert run_d.call_args.kwargs["file_path"] == "app/auth.py"
    assert run_d.call_args.kwargs["function_name"] == "authenticate"
    assert run_d.call_args.kwargs["write_to_disk"] is True
    assert run_d.call_args.kwargs["replace_existing"] is True
    assert run_d.call_args.kwargs["interactive"] is False


def test_failing_test_source_selection() -> None:
    """Failing-source helper returns generated modules when failures exist."""
    result = TestingResult(
        summary="Execution: 0 passed, 1 failed, 0 skipped, 0 errors in 0.01s.",
        generated_tests={
            "test_ok.py": "def test_ok():\n    assert True\n",
            "test_bad.py": "def test_bad():\n    assert False\n",
        },
        coverage_estimate=0.5,
    )
    failing = app_main.failing_test_sources(result)
    assert set(failing) == {"test_ok.py", "test_bad.py"}
