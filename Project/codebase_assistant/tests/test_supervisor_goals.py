"""
test_supervisor_goals.py
=========================

Unit tests for Supervisor goal and task routing: handle_goal() keyword
routing and handle_task() dispatch into the real agent pipelines.

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


def _failing_agent(agent_type: AgentType, error: str) -> MagicMock:
    """Agent that reports failure as data rather than raising."""
    agent = MagicMock()
    agent.handle.side_effect = lambda request: AgentResponse(
        task_id=request.task_id,
        agent_type=agent_type,
        success=False,
        output=None,
        errors=[error],
    )
    return agent


def test_aggregates_two_responses_for_analyze_and_document(
    supervisor: Supervisor,
) -> None:
    """analyze + documentation should aggregate exactly two responses."""
    responses = supervisor.handle_goal("analyze and document", "repo")

    assert len(responses) == 2
    assert [response.agent_type for response in responses] == [
        AgentType.CODE_ANALYSIS,
        AgentType.DOCUMENTATION,
    ]
    assert all(response.success for response in responses)


def test_aggregates_three_responses_for_the_full_pipeline(
    supervisor: Supervisor,
) -> None:
    """All three agents should produce three ordered responses."""
    responses = supervisor.handle_goal("analyze, document and generate tests", "repo")

    assert len(responses) == 3
    assert [response.agent_type for response in responses] == ORDER


def test_documentation_failure_still_returns_testing_success(
    supervisor: Supervisor,
) -> None:
    """A failed agent must not stop or discard the agents after it."""
    supervisor.agents[AgentType.DOCUMENTATION] = _failing_agent(
        AgentType.DOCUMENTATION, "documentation produced an empty summary"
    )

    responses = supervisor.handle_goal("document and generate tests", "repo")

    assert [response.agent_type for response in responses] == [
        AgentType.DOCUMENTATION,
        AgentType.TESTING,
    ]
    documentation, testing = responses
    assert documentation.success is False
    assert documentation.errors == ["documentation produced an empty summary"]
    assert testing.success is True
    supervisor.agents[AgentType.TESTING].handle.assert_called_once()


def test_aggregation_preserves_each_response_verbatim(
    supervisor: Supervisor,
) -> None:
    """success, output, and errors must survive aggregation untouched."""
    payload = {"findings": ["one"]}
    supervisor.agents[AgentType.CODE_ANALYSIS].handle.side_effect = (
        lambda request: AgentResponse(
            task_id=request.task_id,
            agent_type=AgentType.CODE_ANALYSIS,
            success=True,
            output=payload,
            errors=["degraded: model unavailable"],
        )
    )

    analysis = supervisor.handle_goal("analyze and document", "repo")[0]

    assert analysis.success is True
    assert analysis.output is payload
    assert analysis.errors == ["degraded: model unavailable"]


def test_every_selected_agent_runs_even_when_the_first_fails(
    supervisor: Supervisor,
) -> None:
    """A failure in the first agent must not skip the rest of the pipeline."""
    supervisor.agents[AgentType.CODE_ANALYSIS] = _failing_agent(
        AgentType.CODE_ANALYSIS, "analysis failed"
    )

    responses = supervisor.handle_goal("analyze, document and test", "repo")

    assert [response.success for response in responses] == [False, True, True]
    for agent_type in ORDER:
        supervisor.agents[agent_type].handle.assert_called_once()


@pytest.mark.parametrize(
    ("task_name", "expected"),
    [
        ("analysis", AgentType.CODE_ANALYSIS),
        ("analyze the repository", AgentType.CODE_ANALYSIS),
        ("find bugs", AgentType.CODE_ANALYSIS),
        ("documentation", AgentType.DOCUMENTATION),
        ("generate a readme", AgentType.DOCUMENTATION),
        ("testing", AgentType.TESTING),
        ("write unit tests", AgentType.TESTING),
    ],
)
def test_handle_task_routes_to_the_real_agent_pipeline(
    supervisor: Supervisor, task_name: str, expected: AgentType
) -> None:
    """Each task name should reach the matching agent's handle()."""
    response = supervisor.handle_task(task_name, "repo")

    assert response.agent_type == expected
    assert response.success is True
    supervisor.agents[expected].handle.assert_called_once()
    # The placeholder entry point must no longer be used.
    supervisor.agents[expected].run.assert_not_called()


def test_handle_task_passes_the_repository_and_instruction(
    supervisor: Supervisor,
) -> None:
    """The dispatched request should carry the task name and repository."""
    supervisor.handle_task("generate documentation", "/tmp/demo-repo")

    request = supervisor.agents[AgentType.DOCUMENTATION].handle.call_args.args[0]
    assert request.instruction == "generate documentation"
    assert request.context["repo_path"] == "/tmp/demo-repo"
    assert request.context["doc_type"] == "readme"


def test_handle_task_returns_failed_response_for_unknown_task(
    supervisor: Supervisor,
) -> None:
    """An unrecognized task should fail as data, never as an exception."""
    response = supervisor.handle_task("deploy to production", "repo")

    assert response.success is False
    assert response.output is None
    assert "Unknown task" in response.errors[0]
    assert "documentation" in response.errors[0]
    for agent in supervisor.agents.values():
        agent.handle.assert_not_called()


def test_handle_task_reports_agent_exceptions_as_failed_response(
    supervisor: Supervisor,
) -> None:
    """A raising agent should be reported, not propagated."""
    supervisor.agents[AgentType.TESTING].handle.side_effect = RuntimeError("boom")

    response = supervisor.handle_task("write tests", "repo")

    assert response.success is False
    assert "boom" in response.errors[0]


def test_handle_task_defaults_to_workspace_root(supervisor: Supervisor) -> None:
    """An empty repository path should fall back to the configured root."""
    supervisor.handle_task("analyze this repository", "")

    request = supervisor.agents[AgentType.CODE_ANALYSIS].handle.call_args.args[0]
    assert request.context["repo_path"] == str(supervisor.config.workspace_root)


def test_repo_path_defaults_to_workspace_root(supervisor: Supervisor) -> None:
    """Omitting the repository should fall back to the configured root."""
    supervisor.handle_goal("analyze this repository")

    request = supervisor.agents[AgentType.CODE_ANALYSIS].handle.call_args.args[0]
    assert request.context["repo_path"] == str(supervisor.config.workspace_root)
