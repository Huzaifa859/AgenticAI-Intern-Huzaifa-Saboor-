"""Focused tests for TestingAgent live streaming."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

from codebase_assistant.agents.testing_agent import TestingAgent
from codebase_assistant.schemas.schemas import (
    AgentRequest,
    AgentType,
    ModelResponse,
    RetrievedChunk,
)
from codebase_assistant.tracing.tracer import Tracer


VALID_TEST = {
    "summary": "Generated unit tests for add.",
    "generated_tests": {
        "test_math_utils.py": (
            "from math_utils import add\n\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
        )
    },
    "coverage_estimate": 0.8,
}


def _mock_retriever() -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    retriever.vector_store_path = "./.codebase_assistant/chroma"
    retriever.config = MagicMock()
    retriever.vector_db = MagicMock()
    return retriever


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_testing_streams_deltas_via_generate_stream(
    _mock_index: Any, tmp_path: Path
) -> None:
    """Symbol generation should prefer generate_stream and emit deltas."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    chunks: List[str] = []
    content = json.dumps(VALID_TEST)
    client = MagicMock()
    client.is_available.return_value = True

    def _stream(_messages, on_chunk=None, **_kwargs):
        for piece in ("{\"summary\":", " \"ok\"}"):
            if on_chunk is not None:
                on_chunk(piece)
                chunks.append(piece)
        return ModelResponse(content=content, usage={}, raw={})

    client.generate_stream.side_effect = _stream
    tracer = Tracer(enabled=True)
    agent = TestingAgent(
        model_client=client,
        retriever=_mock_retriever(),
        tracer=tracer,
    )

    # Capture stream deltas that would normally go to the worker progress file.
    recorded: List[str] = []
    original_record = tracer.record

    def _record(event_type, name, **metadata):
        if name == "testing_stream_delta":
            recorded.append(str(metadata.get("text") or ""))
            return
        return original_record(event_type, name, **metadata)

    tracer.record = _record  # type: ignore[method-assign]

    response = agent.handle(
        AgentRequest(
            task_id="stream-1",
            agent_type=AgentType.TESTING,
            instruction="Generate pytest unit tests.",
            context={"repo_path": str(tmp_path)},
        )
    )

    assert response.success is True
    assert client.generate_stream.call_count == 1
    assert client.generate.call_count == 0
    assert any("Generating tests for" in text for text in recorded)
    assert "".join(chunks) in "".join(recorded)
