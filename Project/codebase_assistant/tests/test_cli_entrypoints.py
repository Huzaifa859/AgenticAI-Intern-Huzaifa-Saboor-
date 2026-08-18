"""
test_cli_entrypoints.py
========================

Unit tests for the primary CLI argument parsing and the deprecated
codebase_assistant.main forwarder. Agent pipelines are stubbed so these
tests stay offline and fast.
"""

from __future__ import annotations

import importlib
import sys
import warnings
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


@pytest.mark.parametrize(
    ("argv", "agent"),
    [
        (["repo", "--agent", "analysis"], "analysis"),
        (["repo", "--agent", "documentation"], "documentation"),
        (["repo", "--agent", "testing"], "testing"),
        (["repo", "--agent", "all"], "all"),
        (["repo", "-a", "analysis", "--question", "Find bugs"], "analysis"),
    ],
)
def test_parse_args_accepts_agent_flags(argv: List[str], agent: str) -> None:
    """Non-interactive --agent values should parse cleanly."""
    args = app_main.parse_args(argv)
    assert args.agent == agent
    assert args.repository == "repo"


def test_parse_args_defaults_to_interactive_menu() -> None:
    """Omitting --agent must keep the interactive path."""
    args = app_main.parse_args(["repo"])
    assert args.agent is None
    assert args.repository == "repo"


def test_main_routes_to_noninteractive_when_agent_set(tmp_path: Path) -> None:
    """--agent should call run_noninteractive instead of the menu loop."""
    repo = tmp_path / "demo"
    repo.mkdir()

    with patch.object(app_main, "Supervisor") as mock_supervisor_cls, \
        patch.object(app_main, "prepare_repository", return_value=str(repo)), \
        patch.object(app_main, "run_noninteractive") as mock_noninteractive, \
        patch.object(app_main, "interactive_loop") as mock_interactive, \
        patch.object(app_main, "cleanup_temporary_clones"):
        mock_supervisor_cls.return_value = MagicMock()
        app_main.main([str(repo), "--agent", "analysis", "--no-color"])

    mock_noninteractive.assert_called_once()
    mock_interactive.assert_not_called()
    assert mock_noninteractive.call_args.args[2] == "analysis"


def test_main_routes_to_interactive_when_agent_omitted(tmp_path: Path) -> None:
    """Without --agent the interactive menu must still run."""
    repo = tmp_path / "demo"
    repo.mkdir()

    with patch.object(app_main, "Supervisor") as mock_supervisor_cls, \
        patch.object(app_main, "prepare_repository", return_value=str(repo)), \
        patch.object(app_main, "run_noninteractive") as mock_noninteractive, \
        patch.object(app_main, "interactive_loop") as mock_interactive, \
        patch.object(app_main, "cleanup_temporary_clones"):
        mock_supervisor_cls.return_value = MagicMock()
        app_main.main([str(repo), "--no-color"])

    mock_interactive.assert_called_once()
    mock_noninteractive.assert_not_called()


def test_run_noninteractive_all_runs_three_agents(tmp_path: Path) -> None:
    """--agent all should invoke analysis, documentation, and testing."""
    from codebase_assistant.agents.code_analysis_agent import CodeAnalysisAgent
    from codebase_assistant.agents.documentation_agent import DocumentationAgent
    from codebase_assistant.agents.testing_agent import TestingAgent
    from codebase_assistant.schemas.schemas import AgentType

    repo = tmp_path / "demo"
    repo.mkdir()

    supervisor = MagicMock()
    analysis = MagicMock(spec=CodeAnalysisAgent)
    documentation = MagicMock(spec=DocumentationAgent)
    testing = MagicMock(spec=TestingAgent)
    supervisor.agents = {
        AgentType.CODE_ANALYSIS: analysis,
        AgentType.DOCUMENTATION: documentation,
        AgentType.TESTING: testing,
    }

    with patch.object(app_main, "run_code_analysis") as run_a, \
        patch.object(app_main, "run_documentation_agent") as run_d, \
        patch.object(app_main, "run_testing_agent") as run_t:
        app_main.run_noninteractive(
            supervisor, str(repo), "all", "Find bugs", color=False
        )

    run_a.assert_called_once()
    run_d.assert_called_once()
    run_t.assert_called_once()
    assert run_t.call_args.kwargs["interactive"] is False


def test_deprecated_package_main_forwards_to_app_main(tmp_path: Path) -> None:
    """codebase_assistant.main should warn and forward to app.main.main."""
    package_main = importlib.import_module("codebase_assistant.main")
    repo = tmp_path / "demo"
    repo.mkdir()

    with patch.object(app_main, "main") as mock_app_main, \
        warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Reload forwarder path by calling package main with argv that
        # reaches the loaded app module's main.
        with patch.object(package_main, "_load_app_main", return_value=app_main):
            package_main.main([str(repo), "--agent", "testing"])

    mock_app_main.assert_called_once_with([str(repo), "--agent", "testing"])
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
