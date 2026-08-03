"""
test_testing_agent.py
=======================

Unit tests for TestingAgent.

The LLM client and Retriever are mocked so tests do not require
OpenRouter or a populated Chroma index. Filesystem reads use a
temporary repository on disk.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from codebase_assistant.agents.testing_agent import (
    TestingAgent,
    _MAX_CHUNK_CHARS,
    _MAX_PROMPT_CHARS,
    _TEST_MAX_TOKENS,
)
from codebase_assistant.schemas.schemas import (
    AgentRequest,
    AgentType,
    ModelResponse,
    RetrievedChunk,
    TestingResult,
)

# Avoid pytest collecting the schema model imported for assertions.
TestingResult.__test__ = False


VALID_TEST_PAYLOAD = {
    "summary": "Pytest coverage for add covering happy path and type errors.",
    "generated_tests": {
        "test_math_utils.py": (
            "import pytest\n"
            "from math_utils import add\n\n"
            "def test_add_happy_path():\n"
            "    assert add(1, 2) == 3\n\n"
            "def test_add_invalid_type():\n"
            "    with pytest.raises(TypeError):\n"
            "        add('a', 1)\n"
        )
    },
    "coverage_estimate": 0.72,
}


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a tiny repository with one Python module."""
    module = tmp_path / "math_utils.py"
    module.write_text(
        '"""Tiny math helpers."""\n\n'
        "def add(a, b):\n"
        '    """Return a + b."""\n'
        "    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


def _mock_client(available: bool = True, content: str = "") -> MagicMock:
    """Build a mock LLMClient."""
    client = MagicMock()
    client.is_available.return_value = available
    client.generate.return_value = ModelResponse(content=content, usage={}, raw={})
    return client


def _mock_retriever(chunks: Optional[List[RetrievedChunk]] = None) -> MagicMock:
    """Build a mock Retriever."""
    retriever = MagicMock()
    retriever.retrieve.return_value = list(chunks or [])
    retriever.vector_store_path = "./.codebase_assistant/chroma"
    retriever.config = MagicMock()
    retriever.vector_db = MagicMock()
    return retriever


def _agent(client: MagicMock, retriever: MagicMock) -> TestingAgent:
    """Construct a TestingAgent with mocked collaborators."""
    return TestingAgent(model_client=client, retriever=retriever)


def _request(repo: Path, file_path: str = "") -> AgentRequest:
    return AgentRequest(
        task_id="test-1",
        agent_type=AgentType.TESTING,
        instruction="Generate pytest unit tests.",
        context={
            "repo_path": str(repo),
            "file_path": file_path or str(repo / "math_utils.py"),
        },
    )


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_successful_test_generation(_mock_index: Any, sample_repo: Path) -> None:
    """A valid LLM response should yield a successful AgentResponse."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert response.errors == []
    assert isinstance(response.output, TestingResult)
    assert VALID_TEST_PAYLOAD["summary"] in response.output.summary
    assert "Execution:" in response.output.summary
    assert "passed" in response.output.summary
    assert "test_math_utils.py" in response.output.generated_tests
    assert "def test_add_happy_path" in response.output.generated_tests["test_math_utils.py"]
    assert response.output.coverage_estimate == pytest.approx(0.72)
    client.generate.assert_called_once()


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_retriever_returns_context(_mock_index: Any, sample_repo: Path) -> None:
    """Retrieved chunks should appear in the LLM prompt."""
    chunk = RetrievedChunk(
        source="math_utils.py",
        content="def add(a, b):\n    return a + b",
        score=0.88,
        metadata={"file_path": "math_utils.py"},
    )
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    retriever = _mock_retriever([chunk])
    agent = _agent(client, retriever)

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    retriever.retrieve.assert_called()
    prompt = client.generate.call_args.args[0][1].content
    assert "RETRIEVED CONTEXT" in prompt
    assert "def add(a, b)" in prompt


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_empty_retrieval_uses_source_files(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Empty retrieval should still succeed using repository files."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    retriever = _mock_retriever([])
    agent = _agent(client, retriever)

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    retriever.retrieve.assert_called()
    prompt = client.generate.call_args.args[0][1].content
    assert "RETRIEVED CONTEXT" in prompt
    assert "(none)" in prompt
    assert "math_utils.py" in prompt


def test_provider_unavailable(sample_repo: Path) -> None:
    """Unavailable OpenRouter should fail without calling generate."""
    client = _mock_client(available=False)
    retriever = _mock_retriever()
    agent = _agent(client, retriever)

    response = agent.handle(_request(sample_repo))

    assert response.success is False
    assert response.errors
    assert "unavailable" in response.errors[0].lower()
    assert isinstance(response.output, TestingResult)
    assert response.output.generated_tests == {}
    client.generate.assert_not_called()
    retriever.retrieve.assert_not_called()


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_malformed_model_response(_mock_index: Any, sample_repo: Path) -> None:
    """Non-JSON model output should not raise and should report failure."""
    client = _mock_client(content="not json at all")
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_request(sample_repo))

    assert response.success is False
    assert isinstance(response.output, TestingResult)
    assert response.output.generated_tests == {}
    assert response.errors


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_generation_failure(_mock_index: Any, sample_repo: Path) -> None:
    """Transport/model errors during generate should not escape uncaught."""
    client = _mock_client(available=True)
    client.generate.side_effect = RuntimeError("connection reset")
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_request(sample_repo))

    assert response.success is False
    assert isinstance(response.output, TestingResult)
    assert response.output.generated_tests == {}
    assert response.errors


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_generate_unit_tests_populated(
    _mock_index: Any, sample_repo: Path
) -> None:
    """generate_unit_tests should return a populated TestingResult."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    result = agent.generate_unit_tests(str(sample_repo / "math_utils.py"))

    assert result.summary
    assert result.generated_tests
    assert 0.0 < result.coverage_estimate <= 1.0


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_suggest_test_cases_from_generated_names(
    _mock_index: Any, sample_repo: Path
) -> None:
    """suggest_test_cases should surface generated pytest function names."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    suggestions = agent.suggest_test_cases(str(sample_repo / "math_utils.py"))

    assert any("test_add_happy_path" in item for item in suggestions)
    assert any("test_add_invalid_type" in item for item in suggestions)


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_run_returns_success_dict(_mock_index: Any, sample_repo: Path) -> None:
    """run() should surface success when tests are generated."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    outcome = agent.run(str(sample_repo))

    assert outcome["status"] == "success"
    assert "Pytest coverage" in outcome["message"]


# ---------------------------------------------------------------------------
# Prompt construction / context packing (structural properties only)
# ---------------------------------------------------------------------------


def _user_prompt(client: MagicMock) -> str:
    """Return the user-message content passed to LLMClient.generate()."""
    messages = client.generate.call_args.args[0]
    user_messages = [m.content for m in messages if m.role == "user"]
    assert user_messages, "expected a user prompt in generate()"
    return user_messages[0]


def _system_prompt(client: MagicMock) -> str:
    """Return the system-message content passed to LLMClient.generate()."""
    messages = client.generate.call_args.args[0]
    system_messages = [m.content for m in messages if m.role == "system"]
    assert system_messages, "expected a system prompt in generate()"
    return system_messages[0]


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_retrieved_chunks_appear_before_repository_excerpts(
    _mock_index: Any, sample_repo: Path
) -> None:
    """RAG context section must precede repository source excerpts."""
    marker_chunk = "UNIQUE_RETRIEVED_SYMBOL_xyz"
    marker_file = "def add(a, b)"
    chunk = RetrievedChunk(
        source="math_utils.py",
        content=f"def helper():\n    return '{marker_chunk}'",
        score=0.95,
        metadata={"file_path": "math_utils.py"},
    )
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever([chunk]))

    agent.handle(_request(sample_repo))
    prompt = _user_prompt(client)

    retrieved_at = prompt.find("RETRIEVED CONTEXT")
    repo_at = prompt.find("REPOSITORY CONTENTS")
    chunk_at = prompt.find(marker_chunk)
    file_at = prompt.find(marker_file)

    assert retrieved_at != -1
    assert repo_at != -1
    assert retrieved_at < repo_at
    assert chunk_at != -1
    assert file_at != -1
    assert chunk_at < file_at


def test_duplicate_retrieved_chunks_removed_before_prompt_construction() -> None:
    """Whitespace-normalized duplicate chunks must collapse to one entry."""
    duplicate_body = "def add(a, b):\n    return a + b"
    chunks = [
        RetrievedChunk(
            source="math_utils.py",
            content=duplicate_body,
            score=0.9,
            metadata={"file_path": "math_utils.py"},
        ),
        RetrievedChunk(
            source="math_utils.py",
            content=duplicate_body,
            score=0.8,
            metadata={"file_path": "math_utils.py"},
        ),
        RetrievedChunk(
            source="math_utils.py",
            content="   def add(a, b):\n    return a + b   ",
            score=0.7,
            metadata={"file_path": "math_utils.py"},
        ),
    ]
    agent = TestingAgent(model_client=None, retriever=None)
    prompt = agent._build_prompt(
        instruction="Generate pytest unit tests for add.",
        target_path="math_utils.py",
        chunks=chunks,
        source_excerpts=["### math_utils.py\npass\n"],
    )

    assert prompt.count("source=math_utils.py") == 1
    assert prompt.count("def add(a, b):") == 1


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_repository_excerpts_included_when_retrieval_empty(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Empty retrieval must still pack repository source into the prompt."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever([]))

    agent.handle(_request(sample_repo))
    prompt = _user_prompt(client)

    assert "RETRIEVED CONTEXT" in prompt
    assert "(none)" in prompt
    assert "REPOSITORY CONTENTS" in prompt
    assert "math_utils.py" in prompt
    assert "def add" in prompt


def test_oversized_retrieved_context_is_truncated() -> None:
    """Retrieved chunk bodies longer than _MAX_CHUNK_CHARS must be cut."""
    oversized = "TOKEN_" + ("X" * (_MAX_CHUNK_CHARS + 250))
    assert len(oversized) > _MAX_CHUNK_CHARS

    chunks = [
        RetrievedChunk(
            source="big.py",
            content=oversized,
            score=0.99,
            metadata={"file_path": "big.py"},
        )
    ]
    agent = TestingAgent(model_client=None, retriever=None)
    prompt = agent._build_prompt(
        instruction="Generate tests.",
        target_path="big.py",
        chunks=chunks,
        source_excerpts=[],
    )

    assert oversized not in prompt
    assert "TOKEN_" in prompt
    assert "..." in prompt
    # Still within the overall assembled-prompt budget.
    assert len(prompt) <= _MAX_PROMPT_CHARS


def test_assembled_prompt_respects_max_prompt_chars() -> None:
    """Assembled prompts over _MAX_PROMPT_CHARS must be truncated."""
    agent = TestingAgent(model_client=None, retriever=None)
    huge_excerpt = "### huge.py\n" + ("Y" * (_MAX_PROMPT_CHARS + 500))
    prompt = agent._build_prompt(
        instruction="Generate tests.",
        target_path="huge.py",
        chunks=[],
        source_excerpts=[huge_excerpt],
    )

    assert len(prompt) <= _MAX_PROMPT_CHARS
    assert "truncated" in prompt.lower()


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_system_prompt_contains_required_testing_instructions(
    _mock_index: Any, sample_repo: Path
) -> None:
    """System prompt must include key testing-instruction phrases."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    agent.handle(_request(sample_repo))
    system = _system_prompt(client).lower()

    assert "pytest" in system
    assert "retrieved context" in system
    assert "repository contents" in system
    assert "json" in system
    assert "generated_tests" in system
    assert "coverage_estimate" in system
    # Grounding / anti-hallucination signals (phrase-level, not full text).
    assert "never invent" in system or "do not invent" in system or "only symbols" in system


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_prompt_includes_requested_target_file_and_function(
    _mock_index: Any, sample_repo: Path
) -> None:
    """User prompt must name the target file and requested function."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())
    target = str(sample_repo / "math_utils.py")
    request = AgentRequest(
        task_id="test-target",
        agent_type=AgentType.TESTING,
        instruction="Generate pytest unit tests for function add in math_utils.py.",
        context={"repo_path": str(sample_repo), "file_path": target},
    )

    agent.handle(request)
    prompt = _user_prompt(client)

    assert "TARGET" in prompt
    assert "math_utils.py" in prompt
    assert "add" in prompt


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_prompt_contains_testing_result_output_contract(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Prompt must request a TestingResult JSON object as the output contract."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    agent.handle(_request(sample_repo))
    prompt = _user_prompt(client)
    system = _system_prompt(client)

    assert "OUTPUT CONTRACT" in prompt
    assert "TestingResult" in prompt
    assert "JSON" in prompt or "JSON" in system
    assert "generated_tests" in system
    assert "coverage_estimate" in system


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_generate_receives_configured_temperature_and_max_tokens(
    _mock_index: Any, sample_repo: Path
) -> None:
    """LLMClient.generate must receive the agent's configured sampling limits."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    agent.handle(_request(sample_repo))

    kwargs = client.generate.call_args.kwargs
    assert kwargs.get("temperature") == 0.0
    assert kwargs.get("max_tokens") == _TEST_MAX_TOKENS


# ---------------------------------------------------------------------------
# Generated-test execution
# ---------------------------------------------------------------------------


def test_execute_passing_tests_reports_passes(sample_repo: Path) -> None:
    """Passing generated tests should report passed > 0 and failed == 0."""
    agent = TestingAgent(model_client=None, retriever=None)
    generated = {
        "test_math_utils.py": (
            "from math_utils import add\n\n"
            "def test_add_happy_path():\n"
            "    assert add(1, 2) == 3\n"
        )
    }

    summary = agent._execute_generated_tests(str(sample_repo), generated)

    assert "Execution:" in summary
    assert "1 passed" in summary
    assert "0 failed" in summary
    assert generated["test_math_utils.py"].startswith("from math_utils")


def test_execute_failing_tests_reports_failures(sample_repo: Path) -> None:
    """Intentionally failing asserts should surface as failed counts."""
    agent = TestingAgent(model_client=None, retriever=None)
    generated = {
        "test_math_utils.py": (
            "from math_utils import add\n\n"
            "def test_add_wrong_expectation():\n"
            "    assert add(1, 2) == 999\n"
        )
    }

    summary = agent._execute_generated_tests(str(sample_repo), generated)

    assert "Execution:" in summary
    assert "1 failed" in summary
    assert "0 passed" in summary


def test_execute_syntax_errors_reported_without_crash(
    sample_repo: Path,
) -> None:
    """Syntax errors must not raise; errors are reported in the summary."""
    agent = TestingAgent(model_client=None, retriever=None)
    generated = {
        "test_broken.py": "def test_oops(\n    assert True\n"
    }

    summary = agent._execute_generated_tests(str(sample_repo), generated)

    assert "Execution:" in summary
    assert "errors" in summary
    # Source mapping left untouched.
    assert generated["test_broken.py"] == "def test_oops(\n    assert True\n"


def test_execute_cleans_up_temp_directory(
    sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temporary test directories must be removed after execution."""
    agent = TestingAgent(model_client=None, retriever=None)
    generated = {
        "test_math_utils.py": (
            "from math_utils import add\n\n"
            "def test_add():\n"
            "    assert add(2, 2) == 4\n"
        )
    }
    created: List[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)

    agent._execute_generated_tests(str(sample_repo), generated)

    assert created, "expected a temp directory to be created"
    assert all(not os.path.isdir(path) for path in created)

def test_execute_missing_pytest_still_returns_generated_tests(
    sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing pytest skips execution but leaves generated_tests intact."""
    agent = TestingAgent(model_client=None, retriever=None)
    generated = {"test_math_utils.py": "def test_ok():\n    assert True\n"}
    original = generated["test_math_utils.py"]

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "pytest":
            raise ImportError("No module named pytest")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    summary = agent._execute_generated_tests(str(sample_repo), generated)

    assert "skipped" in summary.lower()
    assert "pytest" in summary.lower()
    assert generated["test_math_utils.py"] == original


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_pipeline_appends_execution_and_keeps_source(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Full handle() path executes tests and preserves generated source."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())
    original_source = VALID_TEST_PAYLOAD["generated_tests"]["test_math_utils.py"]

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert response.output.generated_tests["test_math_utils.py"] == original_source
    assert "Execution:" in response.output.summary
    assert "2 passed" in response.output.summary or "passed" in response.output.summary
