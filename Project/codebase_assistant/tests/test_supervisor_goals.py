"""
test_supervisor_goals.py
=========================

Unit tests for Supervisor.handle_goal() keyword routing.

Providers, RAG subsystems, and the agents themselves are replaced with
mocks, so these tests exercise routing, ordering, and aggregation only.
No network access and no model provider are required.
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from codebase_assistant.schemas.schemas import AgentResponse, AgentType
from codebase_assistant.supervisor import Supervisor

ORDER = [AgentType.CODE_ANALYSIS, AgentType.DOCUMENTATION, AgentType.TESTING]


def _mock_agent(agent_type: AgentType) -> MagicMock:
    """Agent whose handle() echoes back a successful AgentResponse."""
    agent = MagicMock()
    agent.handle.side_effect = lambda request: AgentResponse(
        task_id=request.task_id,
        agent_type=agent_type,
        success=True,
        output=None,
        errors=[],
    )
    return agent


@pytest.fixture
def supervisor() -> Supervisor:
    """Supervisor with stubbed providers, subsystems, and agents."""
    with patch.object(Supervisor, "_init_openrouter_provider", return_value=None), \
        patch.object(Supervisor, "_init_ollama_provider", return_value=None), \
        patch("codebase_assistant.supervisor.Indexer"), \
        patch("codebase_assistant.supervisor.Retriever"), \
        patch("codebase_assistant.supervisor.MemoryStore"), \
        patch("codebase_assistant.supervisor.GitHubTools"), \
        patch("codebase_assistant.supervisor.FilesystemTools"), \
        patch("codebase_assistant.supervisor.CodeAnalysisAgent"), \
        patch("codebase_assistant.supervisor.DocumentationAgent"), \
        patch("codebase_assistant.supervisor.TestingAgent"):
        instance = Supervisor()

    instance.agents = {agent_type: _mock_agent(agent_type) for agent_type in ORDER}
    return instance


def _routed(supervisor: Supervisor, goal: str) -> List[AgentType]:
    """Agent types returned for a goal, in response order."""
    return [response.agent_type for response in supervisor.handle_goal(goal, "repo")]


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ("analyze this repository", [AgentType.CODE_ANALYSIS]),
        ("review the code for bugs", [AgentType.CODE_ANALYSIS]),
        ("generate documentation", [AgentType.DOCUMENTATION]),
        ("write a readme", [AgentType.DOCUMENTATION]),
        ("write tests", [AgentType.TESTING]),
        ("improve pytest coverage", [AgentType.TESTING]),
        (
            "analyze and document",
            [AgentType.CODE_ANALYSIS, AgentType.DOCUMENTATION],
        ),
        (
            "document and generate tests",
            [AgentType.DOCUMENTATION, AgentType.TESTING],
        ),
        ("analyze, document and generate tests", ORDER),
    ],
)
def test_goal_routing_table(
    supervisor: Supervisor, goal: str, expected: List[AgentType]
) -> None:
    """Each documented goal wording should select the documented agents."""
    assert _routed(supervisor, goal) == expected


def test_execution_order_is_always_analysis_documentation_testing(
    supervisor: Supervisor,
) -> None:
    """Wording order must not change pipeline order."""
    assert _routed(supervisor, "write tests, document, then analyze") == ORDER


def test_unmatched_goal_defaults_to_code_analysis(supervisor: Supervisor) -> None:
    """A goal with no keywords should fall back to code analysis."""
    assert _routed(supervisor, "what does this project do?") == [
        AgentType.CODE_ANALYSIS
    ]


def test_word_boundaries_prevent_false_matches(supervisor: Supervisor) -> None:
    """Substrings like 'latest' must not select the testing agent."""
    assert _routed(supervisor, "review the latest changes") == [
        AgentType.CODE_ANALYSIS
    ]


def test_each_selected_agent_is_called_once_via_handle(
    supervisor: Supervisor,
) -> None:
    """Agents are invoked through their existing handle() method."""
    supervisor.handle_goal("analyze, document and test", "repo")

    for agent_type in ORDER:
        supervisor.agents[agent_type].handle.assert_called_once()


def test_requests_carry_the_repository_path(supervisor: Supervisor) -> None:
    """Every dispatched request should name the target repository."""
    supervisor.handle_goal("analyze and document", "/tmp/demo-repo")

    for agent_type in (AgentType.CODE_ANALYSIS, AgentType.DOCUMENTATION):
        request = supervisor.agents[agent_type].handle.call_args.args[0]
        assert request.instruction == "analyze and document"
        assert request.context["repo_path"] == "/tmp/demo-repo"
        assert request.context["repository_path"] == "/tmp/demo-repo"


@pytest.mark.parametrize(
    ("goal", "doc_type"),
    [
        ("generate documentation", "readme"),
        ("write docstrings for the helpers", "docstring"),
        ("produce an api reference", "api_reference"),
    ],
)
def test_documentation_mode_is_derived_from_the_goal(
    supervisor: Supervisor, goal: str, doc_type: str
) -> None:
    """Documentation requests should carry the mode the goal implies."""
    supervisor.handle_goal(goal, "repo")

    request = supervisor.agents[AgentType.DOCUMENTATION].handle.call_args.args[0]
    assert request.context["doc_type"] == doc_type


def test_task_ids_are_unique_per_agent(supervisor: Supervisor) -> None:
    """Each dispatched task should be individually identifiable."""
    responses = supervisor.handle_goal("analyze, document and test", "repo")

    task_ids = [response.task_id for response in responses]
    assert len(set(task_ids)) == len(task_ids)


def test_failing_agent_does_not_discard_other_results(
    supervisor: Supervisor,
) -> None:
    """One raising agent should become a failed response, not an abort."""
    supervisor.agents[AgentType.DOCUMENTATION].handle.side_effect = RuntimeError(
        "documentation exploded"
    )

    responses = supervisor.handle_goal("analyze, document and test", "repo")
    by_type: Dict[AgentType, AgentResponse] = {
        response.agent_type: response for response in responses
    }

    assert [response.agent_type for response in responses] == ORDER
    assert by_type[AgentType.CODE_ANALYSIS].success is True
    assert by_type[AgentType.TESTING].success is True
    assert by_type[AgentType.DOCUMENTATION].success is False
    assert "documentation exploded" in by_type[AgentType.DOCUMENTATION].errors[0]


def test_repo_path_defaults_to_workspace_root(supervisor: Supervisor) -> None:
    """Omitting the repository should fall back to the configured root."""
    supervisor.handle_goal("analyze this repository")

    request = supervisor.agents[AgentType.CODE_ANALYSIS].handle.call_args.args[0]
    assert request.context["repo_path"] == str(supervisor.config.workspace_root)
