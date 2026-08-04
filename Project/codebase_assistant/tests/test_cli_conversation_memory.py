"""
test_cli_conversation_memory.py
================================

Verify that the CLI records short ConversationMemory turns during
normal use, persists them through MemoryStore, and lets summarize()
fire automatically when max_messages is exceeded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _PROJECT_ROOT / "app"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import main as app_main  # noqa: E402
from codebase_assistant.agents.code_analysis_agent import (  # noqa: E402
    CodeAnalysisAgent,
)
from codebase_assistant.agents.documentation_agent import (  # noqa: E402
    DocumentationAgent,
)
from codebase_assistant.agents.testing_agent import TestingAgent  # noqa: E402
from codebase_assistant.memory.conversation_memory import (  # noqa: E402
    ConversationMemory,
)
from codebase_assistant.memory.memory_store import MemoryStore  # noqa: E402
from codebase_assistant.schemas.schemas import (  # noqa: E402
    AgentResponse,
    AgentType,
    DocumentationResult,
    ModelResponse,
    TestingResult,
)


class _FakeClient:
    """Minimal LLMClient stand-in for summarization tests."""

    def __init__(self, summary: str = "- prior session summarized") -> None:
        self.summary = summary
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def generate(self, messages, **kwargs) -> ModelResponse:
        self.calls += 1
        return ModelResponse(content=self.summary, raw={}, usage={})


def _history_texts(memory: ConversationMemory) -> List[str]:
    return [message.content for message in memory.get_history()]


def test_record_repository_and_agent_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI helpers should store short turns, never report bodies."""
    store = MemoryStore(storage_path=str(tmp_path / "memory_store"))
    memory = ConversationMemory(
        max_messages=50,
        memory_store=store,
        conversation_id="cli-session",
    )

    app_main.record_repository_loaded(memory, "flask", str(tmp_path / "flask"))

    analysis_agent = MagicMock(spec=CodeAnalysisAgent)
    report = MagicMock()
    report.static_findings = [object()] * 7
    report.llm_findings = [object()]
    analysis_agent.analyze_repository.return_value = report

    documentation_agent = MagicMock(spec=DocumentationAgent)
    documentation_agent.handle.return_value = AgentResponse(
        task_id="doc-1",
        agent_type=AgentType.DOCUMENTATION,
        success=True,
        output=DocumentationResult(
            file_path="README.md",
            function_name="README",
            summary="# Huge README\n\n" + ("x" * 5000),
            parameters=[],
            returns="",
            example_usage="",
        ),
    )

    testing_agent = MagicMock(spec=TestingAgent)
    testing_agent.handle.return_value = AgentResponse(
        task_id="test-1",
        agent_type=AgentType.TESTING,
        success=True,
        output=TestingResult(
            summary="long summary " + ("y" * 2000),
            generated_tests={
                "tests/test_a.py": "def test_a():\n    assert True\n" * 200,
                "tests/test_b.py": "def test_b():\n    assert True\n",
            },
            coverage_estimate=0.4,
        ),
    )

    monkeypatch.setattr(app_main, "print_report", lambda *a, **k: None)
    monkeypatch.setattr(
        app_main, "print_documentation_result", lambda *a, **k: None
    )
    monkeypatch.setattr(
        app_main, "print_testing_result", lambda *a, **k: None
    )
    app_main.run_code_analysis(
        analysis_agent,
        str(tmp_path / "flask"),
        "Find security bugs",
        color=False,
        memory=memory,
    )
    app_main.run_documentation_agent(
        documentation_agent,
        str(tmp_path / "flask"),
        memory=memory,
        interactive=False,
        mode="readme",
    )
    app_main.run_testing_agent(
        testing_agent,
        str(tmp_path / "flask"),
        interactive=False,
        memory=memory,
        mode="repository",
    )

    texts = _history_texts(memory)
    assert "Repository: flask" in texts
    assert "Repository loaded." in texts
    assert any("Run Code Analysis" in text for text in texts)
    assert any("Find security bugs" in text for text in texts)
    assert "7 static findings. 1 grounded LLM finding." in texts
    assert any("Generate documentation" in text for text in texts)
    assert "README generated." in texts
    assert any("Generate tests" in text for text in texts)
    assert "Generated tests for 2 modules." in texts

    joined = "\n".join(texts)
    assert "# Huge README" not in joined
    assert "def test_a()" not in joined
    assert "xxxx" not in joined
    assert all(len(text) <= app_main._MAX_MEMORY_CONTENT_CHARS for text in texts)


def test_persisted_memory_restores_on_restart(tmp_path: Path) -> None:
    """A new ConversationMemory should reload prior CLI turns."""
    store_path = str(tmp_path / "memory_store")
    store = MemoryStore(storage_path=store_path)
    first = ConversationMemory(
        max_messages=50,
        memory_store=store,
        conversation_id="default",
    )
    app_main.record_repository_loaded(first, "flask", str(tmp_path / "flask"))
    app_main.record_memory_message(first, "user", "Generate documentation")
    app_main.record_memory_message(first, "assistant", "README generated.")

    second = ConversationMemory(
        max_messages=50,
        memory_store=MemoryStore(storage_path=store_path),
        conversation_id="default",
    )
    texts = _history_texts(second)
    assert "Repository: flask" in texts
    assert "Repository loaded." in texts
    assert "Generate documentation" in texts
    assert "README generated." in texts
    assert second.metadata.get("repository_reference") == "flask"


def test_summarize_triggers_automatically_via_cli_recording(
    tmp_path: Path,
) -> None:
    """Crossing max_messages through CLI recording should summarize."""
    client = _FakeClient(summary="- flask repo\n- analysis already run")
    store = MemoryStore(storage_path=str(tmp_path / "memory_store"))
    memory = ConversationMemory(
        max_messages=4,
        keep_recent=2,
        model_client=client,
        memory_store=store,
        conversation_id="summarize-cli",
    )

    app_main.record_repository_loaded(memory, "flask", str(tmp_path / "flask"))
    # Two messages so far. Add three more pairs-worth of short turns
    # until we cross max_messages=4.
    app_main.record_memory_message(memory, "user", "Run Code Analysis")
    app_main.record_memory_message(
        memory, "assistant", "7 static findings. 1 grounded LLM finding."
    )
    assert client.calls == 0
    assert len(memory.get_history()) == 4

    app_main.record_memory_message(memory, "user", "Generate documentation")
    assert client.calls == 1

    history = memory.get_history()
    assert history[0].role == "system"
    assert "Conversation summary:" in history[0].content
    assert history[0].content.startswith("Conversation summary:")
    # Recent tail kept.
    recent = [message.content for message in history[1:]]
    assert "Generate documentation" in recent
    # Older raw turns replaced by summary.
    older_contents = [message.content for message in history]
    assert "Repository: flask" not in older_contents


def test_interactive_loop_records_chosen_agent(tmp_path: Path, monkeypatch) -> None:
    """Menu choice 2 should record documentation turns on Supervisor memory."""
    from codebase_assistant.schemas.schemas import AgentType

    store = MemoryStore(storage_path=str(tmp_path / "memory_store"))
    memory = ConversationMemory(
        max_messages=50,
        memory_store=store,
        conversation_id="interactive",
    )

    documentation = MagicMock(spec=DocumentationAgent)
    documentation.handle.return_value = AgentResponse(
        task_id="doc-2",
        agent_type=AgentType.DOCUMENTATION,
        success=True,
        output=DocumentationResult(
            file_path="README.md",
            function_name="README",
            summary="body should not be stored",
            parameters=[],
            returns="",
            example_usage="",
        ),
    )

    supervisor = MagicMock()
    supervisor.conversation_memory = memory
    supervisor.agents = {
        AgentType.CODE_ANALYSIS: MagicMock(spec=CodeAnalysisAgent),
        AgentType.DOCUMENTATION: documentation,
        AgentType.TESTING: MagicMock(spec=TestingAgent),
    }

    answers = iter(["2", "1", "4"])
    monkeypatch.setattr(
        app_main, "prompt_choice", lambda *a, **k: next(answers)
    )
    monkeypatch.setattr(app_main, "ask_yes_no", lambda *a, **k: False)
    monkeypatch.setattr(
        app_main, "print_documentation_result", lambda *a, **k: None
    )

    app_main.interactive_loop(
        supervisor, str(tmp_path / "repo"), "Find bugs", color=False
    )

    texts = _history_texts(memory)
    assert any("Generate documentation" in text for text in texts)
    assert "README generated." in texts
    assert "body should not be stored" not in "\n".join(texts)


def test_example_persisted_memory_json_shape(tmp_path: Path) -> None:
    """Persisted snapshot should match the MemoryStore JSON shape."""
    store_path = tmp_path / "memory_store"
    store = MemoryStore(storage_path=str(store_path))
    memory = ConversationMemory(
        max_messages=50,
        memory_store=store,
        conversation_id="default",
    )
    app_main.record_repository_loaded(memory, "flask", "/tmp/flask")
    app_main.record_memory_message(memory, "user", "Run Code Analysis")
    app_main.record_memory_message(
        memory, "assistant", "7 static findings. 1 grounded LLM finding."
    )
    app_main.record_memory_message(memory, "user", "Generate documentation")
    app_main.record_memory_message(memory, "assistant", "README generated.")

    files = list(store_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["key"] == "default"
    assert payload["value"]["conversation_id"] == "default"
    messages = payload["value"]["messages"]
    assert messages == [
        {"role": "user", "content": "Repository: flask"},
        {"role": "assistant", "content": "Repository loaded."},
        {"role": "user", "content": "Run Code Analysis"},
        {
            "role": "assistant",
            "content": "7 static findings. 1 grounded LLM finding.",
        },
        {"role": "user", "content": "Generate documentation"},
        {"role": "assistant", "content": "README generated."},
    ]
    assert payload["value"]["metadata"]["repository_reference"] == "flask"
