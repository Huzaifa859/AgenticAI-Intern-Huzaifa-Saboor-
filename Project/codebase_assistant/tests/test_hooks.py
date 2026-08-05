"""
test_hooks.py
=============

Lifecycle HookManager dispatch, isolation, and Supervisor wiring.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from codebase_assistant.config import Config
from codebase_assistant.hooks import (
    BaseHook,
    HookEvent,
    HookManager,
    LoggingHook,
    TracingHook,
    install_default_hooks,
)
from codebase_assistant.schemas.schemas import (
    AgentRequest,
    AgentResponse,
    AgentType,
)
from codebase_assistant.supervisor import Supervisor
from codebase_assistant.tracing.tracer import Tracer


class _RecordingHook(BaseHook):
    def __init__(self, event: HookEvent, sink: List[Dict[str, Any]]) -> None:
        self.event = event
        self.name = f"record_{event.value}"
        self._sink = sink

    def run(self, context: Dict[str, Any]) -> None:
        self._sink.append({"event": self.event, **dict(context)})


class _BrokenHook(BaseHook):
    event = HookEvent.BEFORE_AGENT_RUN
    name = "broken"

    def run(self, context: Dict[str, Any]) -> None:
        raise RuntimeError("hook boom")


def test_register_and_trigger_delivers_context() -> None:
    manager = HookManager()
    seen: List[Dict[str, Any]] = []
    manager.register_hook(_RecordingHook(HookEvent.BEFORE_AGENT_RUN, seen))

    manager.trigger(
        HookEvent.BEFORE_AGENT_RUN,
        {"agent_type": "testing", "task_id": "t1"},
    )

    assert len(seen) == 1
    assert seen[0]["event"] == HookEvent.BEFORE_AGENT_RUN
    assert seen[0]["agent_type"] == "testing"
    assert seen[0]["task_id"] == "t1"


def test_failing_hook_does_not_raise_to_caller() -> None:
    manager = HookManager()
    seen: List[Dict[str, Any]] = []
    manager.register_hook(_BrokenHook())
    manager.register_hook(_RecordingHook(HookEvent.BEFORE_AGENT_RUN, seen))

    manager.trigger(HookEvent.BEFORE_AGENT_RUN, {"ok": True})

    assert len(seen) == 1
    assert seen[0]["ok"] is True


def test_unregister_and_list_hooks() -> None:
    manager = HookManager()
    hook = LoggingHook(HookEvent.AFTER_AGENT_RUN)
    manager.register_hook(hook)
    assert manager.list_hooks(HookEvent.AFTER_AGENT_RUN) == [hook]
    manager.unregister_hook(hook)
    assert manager.list_hooks(HookEvent.AFTER_AGENT_RUN) == []


def test_tracing_hook_records_tracer_events() -> None:
    tracer = Tracer(run_id="hooks-trace")
    hook = TracingHook(HookEvent.AFTER_MODEL_CALL, tracer=tracer)
    hook.run(
        {
            "success": True,
            "model": "google/gemma-3-27b-it",
            "duration_ms": 12.5,
        }
    )
    names = [event.name for event in tracer.get_events()]
    assert "after_model_call" in names


def test_install_default_hooks_registers_all_events() -> None:
    manager = HookManager()
    tracer = Tracer(run_id="defaults")
    install_default_hooks(manager, tracer=tracer)
    for event in HookEvent:
        hooks = manager.list_hooks(event)
        assert len(hooks) == 2
        assert any(isinstance(h, LoggingHook) for h in hooks)
        assert any(isinstance(h, TracingHook) for h in hooks)


def test_supervisor_emits_before_and_after_agent_run() -> None:
    seen: List[Dict[str, Any]] = []

    with patch.object(Supervisor, "_init_openrouter_provider", return_value=None), \
        patch.object(Supervisor, "_init_ollama_provider", return_value=None):
        supervisor = Supervisor(config=Config(workspace_root="."))

    supervisor.hook_manager.register_hook(
        _RecordingHook(HookEvent.BEFORE_AGENT_RUN, seen)
    )
    supervisor.hook_manager.register_hook(
        _RecordingHook(HookEvent.AFTER_AGENT_RUN, seen)
    )

    agent = MagicMock()
    agent.handle.return_value = AgentResponse(
        task_id="task-1",
        agent_type=AgentType.CODE_ANALYSIS,
        success=True,
        output={"ok": True},
        errors=[],
    )
    supervisor.agents[AgentType.CODE_ANALYSIS] = agent

    request = AgentRequest(
        task_id="task-1",
        agent_type=AgentType.CODE_ANALYSIS,
        instruction="analyze",
        context={"repo_path": "."},
    )
    response = supervisor._dispatch_safely(request)

    assert response.success is True
    events = [item["event"] for item in seen]
    assert HookEvent.BEFORE_AGENT_RUN in events
    assert HookEvent.AFTER_AGENT_RUN in events


def test_supervisor_on_error_when_agent_raises() -> None:
    seen: List[Dict[str, Any]] = []

    with patch.object(Supervisor, "_init_openrouter_provider", return_value=None), \
        patch.object(Supervisor, "_init_ollama_provider", return_value=None):
        supervisor = Supervisor(config=Config(workspace_root="."))

    supervisor.hook_manager.register_hook(_RecordingHook(HookEvent.ON_ERROR, seen))

    agent = MagicMock()
    agent.handle.side_effect = RuntimeError("agent crashed")
    supervisor.agents[AgentType.TESTING] = agent

    request = AgentRequest(
        task_id="task-err",
        agent_type=AgentType.TESTING,
        instruction="test",
        context={"repo_path": "."},
    )
    response = supervisor._dispatch_safely(request)

    assert response.success is False
    assert any(item["event"] == HookEvent.ON_ERROR for item in seen)
    assert "agent crashed" in (seen[0].get("error") or "")
