"""
test_tracing.py
================

Verification for Phase 4 tracing: events are recorded, ordered,
survive failures, separate by agent, and export deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from codebase_assistant.agents.code_analysis_agent import CodeAnalysisAgent
from codebase_assistant.agents.documentation_agent import DocumentationAgent
from codebase_assistant.agents.testing_agent import TestingAgent
from codebase_assistant.config import Config
from codebase_assistant.memory.conversation_memory import ConversationMemory
from codebase_assistant.memory.memory_store import MemoryStore
from codebase_assistant.schemas.schemas import (
    AgentRequest,
    AgentResponse,
    AgentType,
    ModelMessage,
    ModelResponse,
)
from codebase_assistant.supervisor import Supervisor
from codebase_assistant.tracing.events import TraceEventType
from codebase_assistant.tracing.tracer import Tracer


class _Clock:
    """Deterministic increasing clock for stable timestamps."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


def test_tracer_records_events_in_order(tmp_path: Path) -> None:
    """Events keep insertion order and carry required fields."""
    tracer = Tracer(run_id="run-1", time_fn=_Clock())
    tracer.record(
        TraceEventType.LIFECYCLE,
        "application_started",
        component="CLI",
        success=True,
    )
    tracer.record(
        TraceEventType.LIFECYCLE,
        "repository_selected",
        component="CLI",
        success=True,
        repository_path=str(tmp_path),
    )
    tracer.record(
        TraceEventType.AGENT_RUN,
        "analysis_started",
        component="CodeAnalysisAgent",
        success=True,
    )

    names = tracer.event_names()
    assert names == [
        "application_started",
        "repository_selected",
        "analysis_started",
    ]
    events = tracer.get_events()
    assert events[0].component == "CLI"
    assert events[0].success is True
    assert events[0].sequence == 1
    assert events[1].metadata["repository_path"] == str(tmp_path)
    assert events[2].component == "CodeAnalysisAgent"


def test_tracer_never_raises_on_bad_input() -> None:
    """Tracing failures must only warn, never crash callers."""
    tracer = Tracer(run_id="run-safe")
    tracer.record("not-a-real-type", "boom", component="CLI")  # type: ignore[arg-type]
    # Still usable afterward.
    tracer.record(
        TraceEventType.LIFECYCLE, "still_works", component="CLI", success=True
    )
    assert "still_works" in tracer.event_names()


def test_export_is_deterministic(tmp_path: Path) -> None:
    """Two identical runs export the same JSON payload."""
    def _build() -> Tracer:
        clock = _Clock(50.0)
        tracer = Tracer(run_id="fixed-run", time_fn=clock)
        tracer.record(
            TraceEventType.LIFECYCLE,
            "application_started",
            component="CLI",
            success=True,
            mode="test",
        )
        tracer.record(
            TraceEventType.MEMORY,
            "save",
            component="MemoryStore",
            success=True,
            key="default",
        )
        return tracer

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    assert _build().export(str(first)) is True
    assert _build().export(str(second)) is True
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["run_id"] == "fixed-run"
    assert [event["name"] for event in payload["events"]] == [
        "application_started",
        "save",
    ]


def test_failed_execution_is_traced() -> None:
    """Failed stages set success=False and may carry an error."""
    tracer = Tracer(run_id="fail-run", time_fn=_Clock())
    tracer.record(
        TraceEventType.MODEL_CALL,
        "model_response",
        component="TestingAgent",
        success=False,
        error="connection reset",
    )
    event = tracer.get_events()[0]
    assert event.success is False
    assert event.error == "connection reset"


def test_supervisor_goal_trace_order(tmp_path: Path) -> None:
    """Supervisor goal handling emits routing and dispatch events in order."""
    config = Config(
        workspace_root=str(tmp_path),
        memory_store_path=str(tmp_path / "memory"),
        chroma_persist_directory=str(tmp_path / "chroma"),
    )
    with patch.object(Supervisor, "_init_openrouter_provider", return_value=None), \
        patch.object(Supervisor, "_init_ollama_provider", return_value=None):
        supervisor = Supervisor(config=config)

    clock = _Clock()
    supervisor.tracer = Tracer(run_id="goal-run", time_fn=clock)
    for agent in supervisor.agents.values():
        agent.tracer = supervisor.tracer

    def _fake_handle(request: AgentRequest) -> AgentResponse:
        agent = supervisor.agents[request.agent_type]
        agent._trace(
            f"{request.agent_type.value}_started",
            agent=request.agent_type.value,
        )
        agent._trace(
            f"{request.agent_type.value}_finished",
            agent=request.agent_type.value,
            success=True,
        )
        return AgentResponse(
            task_id=request.task_id,
            agent_type=request.agent_type,
            success=True,
            output={"ok": True},
        )

    with patch.object(Supervisor, "dispatch", side_effect=_fake_handle):
        responses = supervisor.handle_goal(
            "analyze and document this repository",
            repo_path=str(tmp_path),
        )

    assert len(responses) == 2
    names = supervisor.tracer.event_names()
    assert names.index("goal_received") < names.index("routing_decision")
    assert names.index("routing_decision") < names.index("dispatch_start")
    assert names.index("dispatch_start") < names.index("dispatch_finish")
    assert names.index("dispatch_finish") < names.index("aggregation_complete")
    # Two agents produce separate start/finish markers.
    assert names.count("dispatch_start") == 2
    assert "code_analysis_started" in names
    assert "documentation_started" in names


def test_multiple_agents_generate_separate_component_traces(
    tmp_path: Path,
) -> None:
    """Each agent records events under its own component name."""
    tracer = Tracer(run_id="multi", time_fn=_Clock())
    analysis = CodeAnalysisAgent(tracer=tracer)
    docs = DocumentationAgent(tracer=tracer)
    testing = TestingAgent(tracer=tracer)

    analysis._trace("analysis_started")
    docs._trace("documentation_started")
    testing._trace("testing_started")

    by_component = {
        event.component: event.name for event in tracer.get_events()
    }
    assert by_component["CodeAnalysisAgent"] == "analysis_started"
    assert by_component["DocumentationAgent"] == "documentation_started"
    assert by_component["TestingAgent"] == "testing_started"


def test_memory_store_and_conversation_memory_trace(tmp_path: Path) -> None:
    """MemoryStore load/save and ConversationMemory summarize/persist."""
    tracer = Tracer(run_id="mem", time_fn=_Clock())
    store = MemoryStore(storage_path=str(tmp_path / "store"), tracer=tracer)
    client = MagicMock()
    client.is_available.return_value = True
    client.generate.return_value = ModelResponse(
        content="- prior turns summarized", usage={}, raw={}
    )
    memory = ConversationMemory(
        max_messages=3,
        keep_recent=1,
        model_client=client,
        memory_store=store,
        conversation_id="default",
        tracer=tracer,
    )
    memory.add_message(ModelMessage(role="user", content="one"))
    memory.add_message(ModelMessage(role="assistant", content="two"))
    memory.add_message(ModelMessage(role="user", content="three"))
    memory.add_message(ModelMessage(role="assistant", content="four"))

    names = tracer.event_names()
    assert "load" in names
    assert "save" in names
    assert "persisted" in names
    assert "summarize_started" in names
    assert "summarize_finished" in names


def test_testing_agent_execution_events(tmp_path: Path) -> None:
    """Generated-test write/execute stages emit dedicated events."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    tracer = Tracer(run_id="exec", time_fn=_Clock())
    agent = TestingAgent(tracer=tracer)
    generated = {
        "test_math_utils.py": (
            "from math_utils import add\n\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
        )
    }
    summary = agent._execute_generated_tests(str(tmp_path), generated)
    names = tracer.event_names()
    assert "generated_tests_written" in names
    assert "pytest_execution_started" in names
    assert "pytest_execution_finished" in names
    assert "1 passed" in summary


def test_cli_lifecycle_helpers_record(tmp_path: Path) -> None:
    """CLI helper emits application/repo lifecycle markers."""
    import sys

    project_root = Path(__file__).resolve().parents[2]
    app_dir = project_root / "app"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    import main as app_main

    tracer = Tracer(run_id="cli", time_fn=_Clock())
    app_main._cli_trace(tracer, "application_started")
    app_main._cli_trace(
        tracer,
        "repository_selected",
        repository_path=str(tmp_path),
        remote=False,
    )
    app_main._cli_trace(tracer, "selected_agents", agents=["analysis"])
    app_main._cli_trace(tracer, "application_exit")
    assert tracer.event_names() == [
        "application_started",
        "repository_selected",
        "selected_agents",
        "application_exit",
    ]
