"""
test_agents.py
===============

Placeholder tests for the Supervisor and the three specialized agents.

The Code Analysis Agent is the priority here — it is the proposal's
primary feature and half of the Week 6 coverage target.

TODO: Replace remaining skips below with real assertions as each agent
capability is finalized.
"""

from __future__ import annotations

from pathlib import Path

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


def test_testing_agent_executes_generated_tests(tmp_path: Path) -> None:
    """Generated tests should actually run, not just be produced."""
    from codebase_assistant.agents.testing_agent import TestingAgent

    module = tmp_path / "math_utils.py"
    module.write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    agent = TestingAgent(model_client=None, retriever=None)
    generated = {
        "test_math_utils.py": (
            "from math_utils import add\n\n"
            "def test_add():\n"
            "    assert add(1, 1) == 2\n"
        )
    }
    original = dict(generated)

    summary = agent._execute_generated_tests(str(tmp_path), generated)

    assert "Execution:" in summary
    assert "1 passed" in summary
    assert generated == original


@pytest.mark.skip(reason="TODO: assert conversation memory carries context across turns")
def test_agents_retain_conversation_context() -> None:
    """A follow-up question should see the previous turn's context."""
