"""
test_agents.py
===============

Placeholder tests for the Supervisor and the three specialized agents.

The Code Analysis Agent is the priority here — it is the proposal's
primary feature and half of the Week 6 coverage target.

TODO: Replace every skip below with real assertions as each agent is
implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: assert route_task maps task names to the right AgentType")
def test_supervisor_routes_tasks_to_correct_agent() -> None:
    """Routing should send test/doc/analysis tasks to their agents."""


@pytest.mark.skip(reason="TODO: assert dispatch returns an error response for an unknown agent type")
def test_supervisor_handles_unknown_agent_type() -> None:
    """Dispatching to an unregistered agent type should fail cleanly."""


@pytest.mark.skip(reason="TODO: assert the agent returns grounded BugReport objects")
def test_code_analysis_agent_returns_bug_reports() -> None:
    """Analysis should produce reports that passed the grounding check."""


@pytest.mark.skip(reason="TODO: assert the agent abstains when retrieved context is insufficient")
def test_code_analysis_agent_abstains_on_low_confidence() -> None:
    """Insufficient context should abstain rather than guess."""


@pytest.mark.skip(reason="TODO: assert zero findings on the clean benchmark repo")
def test_code_analysis_agent_reports_no_false_positives_on_clean_repo() -> None:
    """A repo with no seeded bugs should produce no reports."""


@pytest.mark.skip(reason="TODO: assert generated documentation validates as DocumentationResult")
def test_documentation_agent_returns_valid_result() -> None:
    """Generated docs should satisfy the DocumentationResult schema."""


@pytest.mark.skip(reason="TODO: assert generated tests are executed and real pass/fail is reported")
def test_testing_agent_executes_generated_tests() -> None:
    """Generated tests should actually run, not just be produced."""


@pytest.mark.skip(reason="TODO: assert conversation memory carries context across turns")
def test_agents_retain_conversation_context() -> None:
    """A follow-up question should see the previous turn's context."""
