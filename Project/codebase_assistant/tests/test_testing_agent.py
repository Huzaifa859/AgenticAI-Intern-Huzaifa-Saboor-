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
    assert "Coverage:" in response.output.summary
    assert 0.0 < response.output.coverage_estimate <= 1.0
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


# ---------------------------------------------------------------------------
# Repair loop (exactly one iteration)
# ---------------------------------------------------------------------------

_FAILING_TEST_SOURCE = (
    "from math_utils import add\n\n"
    "def test_add_wrong_expectation():\n"
    "    assert add(1, 2) == 999\n"
)

_FIXED_TEST_SOURCE = (
    "from math_utils import add\n\n"
    "def test_add_wrong_expectation():\n"
    "    assert add(1, 2) == 3\n"
)

_FAILING_PAYLOAD = {
    "summary": "Initial tests with a wrong assertion.",
    "generated_tests": {"test_math_utils.py": _FAILING_TEST_SOURCE},
    "coverage_estimate": 0.4,
}

_REPAIRED_PAYLOAD = {
    "summary": "Fixed the failing assertion to match add().",
    "generated_tests": {"test_math_utils.py": _FIXED_TEST_SOURCE},
    "coverage_estimate": 0.5,
}

_STILL_FAILING_REPAIR_PAYLOAD = {
    "summary": "Attempted repair but assertion is still wrong.",
    "generated_tests": {
        "test_math_utils.py": (
            "from math_utils import add\n\n"
            "def test_add_wrong_expectation():\n"
            "    assert add(1, 2) == 42\n"
        )
    },
    "coverage_estimate": 0.45,
}


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_repair_succeeds_after_initial_failure(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Failing first run should trigger one repair that can make tests pass."""
    from codebase_assistant.tracing.tracer import Tracer

    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(_FAILING_PAYLOAD), usage={}, raw={}),
        ModelResponse(content=json.dumps(_REPAIRED_PAYLOAD), usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="repair-ok")

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert client.generate.call_count == 2
    assert (
        response.output.generated_tests["test_math_utils.py"] == _FIXED_TEST_SOURCE
    )
    assert "Repair: attempted one fix iteration." in response.output.summary
    assert response.output.summary.count("Execution:") == 2
    assert "1 passed" in response.output.summary
    # Repair call uses the repair system prompt.
    repair_messages = client.generate.call_args_list[1].args[0]
    assert "failing" in repair_messages[0].content.lower()
    assert "PYTEST FAILURE OUTPUT" in repair_messages[1].content
    names = agent.tracer.event_names()
    assert "testing_repair_started" in names
    assert "testing_repair_generated" in names
    assert "testing_repair_finished" in names


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_repair_still_fails_returns_repaired_tests(
    _mock_index: Any, sample_repo: Path
) -> None:
    """If the repaired suite still fails, return repaired sources + 2nd run."""
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(_FAILING_PAYLOAD), usage={}, raw={}),
        ModelResponse(
            content=json.dumps(_STILL_FAILING_REPAIR_PAYLOAD), usage={}, raw={}
        ),
    ]
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert client.generate.call_count == 2
    assert (
        response.output.generated_tests["test_math_utils.py"]
        == _STILL_FAILING_REPAIR_PAYLOAD["generated_tests"]["test_math_utils.py"]
    )
    assert "Repair: attempted one fix iteration." in response.output.summary
    assert response.output.summary.count("Execution:") == 2
    assert "failed" in response.output.summary


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_repair_generation_failure_keeps_original_tests(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Repair LLM failure must preserve the original generated tests."""
    from codebase_assistant.tracing.tracer import Tracer

    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(_FAILING_PAYLOAD), usage={}, raw={}),
        RuntimeError("OpenRouter unavailable"),
    ]
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="repair-fail")

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert client.generate.call_count == 2
    assert (
        response.output.generated_tests["test_math_utils.py"] == _FAILING_TEST_SOURCE
    )
    assert "Repair: attempted one fix iteration." not in response.output.summary
    assert response.output.summary.count("Execution:") == 1
    assert "1 failed" in response.output.summary
    assert "testing_repair_failed" in agent.tracer.event_names()


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_repair_invalid_output_keeps_original_tests(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Invalid repair JSON must not replace the original generated tests."""
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(_FAILING_PAYLOAD), usage={}, raw={}),
        ModelResponse(content="not-json {{{", usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert (
        response.output.generated_tests["test_math_utils.py"] == _FAILING_TEST_SOURCE
    )
    assert response.output.summary.count("Execution:") == 1


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_repair_skipped_when_tests_pass_initially(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Passing first pytest run must not call the repair model."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    client.generate.assert_called_once()
    assert "Repair: attempted one fix iteration." not in response.output.summary
    assert response.output.summary.count("Execution:") == 1


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_only_one_repair_attempt_occurs(
    _mock_index: Any, sample_repo: Path
) -> None:
    """A still-failing repair must not trigger a second repair loop."""
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(_FAILING_PAYLOAD), usage={}, raw={}),
        ModelResponse(
            content=json.dumps(_STILL_FAILING_REPAIR_PAYLOAD), usage={}, raw={}
        ),
        ModelResponse(content=json.dumps(_REPAIRED_PAYLOAD), usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())

    with patch.object(
        agent,
        "_repair_failing_tests",
        wraps=agent._repair_failing_tests,
    ) as repair_spy:
        response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert client.generate.call_count == 2
    assert repair_spy.call_count == 1


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_repair_has_no_infinite_loop(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Even if every execution fails, generate is called at most twice."""
    always_fail = {
        "summary": "Always wrong.",
        "generated_tests": {"test_math_utils.py": _FAILING_TEST_SOURCE},
        "coverage_estimate": 0.1,
    }
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(always_fail), usage={}, raw={}),
        ModelResponse(content=json.dumps(always_fail), usage={}, raw={}),
    ] + [
        ModelResponse(content=json.dumps(always_fail), usage={}, raw={})
        for _ in range(5)
    ]
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert client.generate.call_count == 2
    assert response.output.summary.count("Execution:") == 2


# ---------------------------------------------------------------------------
# AST-based per-symbol generation
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_module_repo(tmp_path: Path) -> Path:
    """Repository with two modules exposing public symbols."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def sub(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    (tmp_path / "auth.py").write_text(
        "class UserService:\n"
        "    def login(self, user):\n"
        "        return bool(user)\n\n"
        "    def _hidden(self):\n"
        "        return None\n\n"
        "    def __str__(self):\n"
        "        return 'UserService'\n\n"
        "def authenticate(user):\n"
        "    return bool(user)\n\n"
        "def _private_helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ignore_me.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


def test_ast_scan_multiple_modules_and_functions(multi_module_repo: Path) -> None:
    """AST scan should find public functions across modules."""
    agent = TestingAgent(model_client=None, retriever=None)
    filesystem = agent._filesystem_tools(str(multi_module_repo))
    inventory = agent._list_python_inventory(filesystem, str(multi_module_repo))
    symbols, skipped = agent._collect_testable_symbols(filesystem, inventory)

    names = {(s.module_path, s.qualname) for s in symbols}
    assert ("math_utils.py", "add") in names
    assert ("math_utils.py", "sub") in names
    assert ("auth.py", "authenticate") in names
    assert ("auth.py", "UserService") in names
    assert not any(path.startswith("tests/") for path, _ in names)
    assert any("private" in item for item in skipped)


def test_ast_scan_classes_with_public_methods(multi_module_repo: Path) -> None:
    """Class symbols should include only public methods."""
    agent = TestingAgent(model_client=None, retriever=None)
    filesystem = agent._filesystem_tools(str(multi_module_repo))
    symbols, skipped = agent._collect_testable_symbols(
        filesystem, ["auth.py"]
    )
    service = next(s for s in symbols if s.kind == "class" and s.name == "UserService")
    method_names = {m.name for m in service.methods}
    assert method_names == {"login"}
    assert any("UserService._hidden:private" in item for item in skipped)
    assert any("UserService.__str__:private" in item or "__str__" in item for item in skipped)


def test_ast_scan_skips_private_methods_and_functions(
    multi_module_repo: Path,
) -> None:
    """Private helpers and dunders must not become generation symbols."""
    agent = TestingAgent(model_client=None, retriever=None)
    filesystem = agent._filesystem_tools(str(multi_module_repo))
    symbols, _skipped = agent._collect_testable_symbols(filesystem, ["auth.py"])
    qualnames = {s.qualname for s in symbols}
    assert "_private_helper" not in qualnames
    assert "_hidden" not in qualnames
    assert "__str__" not in qualnames


def test_ast_scan_skips_duplicate_symbols(tmp_path: Path) -> None:
    """Duplicate definitions in one module should be counted once."""
    (tmp_path / "dupes.py").write_text(
        "def once():\n    return 1\n",
        encoding="utf-8",
    )
    agent = TestingAgent(model_client=None, retriever=None)
    filesystem = agent._filesystem_tools(str(tmp_path))
    symbols, _skipped = agent._collect_testable_symbols(filesystem, ["dupes.py"])
    # Simulate a second discovery of the same key via direct call on same inventory.
    symbols2, skipped2 = agent._collect_testable_symbols(filesystem, ["dupes.py", "dupes.py"])
    assert len([s for s in symbols if s.name == "once"]) == 1
    assert len([s for s in symbols2 if s.name == "once"]) == 1
    assert any("duplicate" in item for item in skipped2)


def test_ast_scan_no_public_symbols(tmp_path: Path) -> None:
    """Modules with only private symbols yield an empty public set."""
    (tmp_path / "secret.py").write_text(
        "def _hidden():\n    return 1\n\n"
        "class _Private:\n"
        "    def visible(self):\n"
        "        return 2\n",
        encoding="utf-8",
    )
    agent = TestingAgent(model_client=None, retriever=None)
    filesystem = agent._filesystem_tools(str(tmp_path))
    symbols, skipped = agent._collect_testable_symbols(filesystem, ["secret.py"])
    assert symbols == []
    assert skipped


def test_merge_preserves_all_generated_tests() -> None:
    """Merged suites keep imports, fixtures, and unique tests."""
    left = (
        "import pytest\n"
        "from math_utils import add\n\n"
        "@pytest.fixture\n"
        "def value():\n"
        "    return 1\n\n"
        "def test_add_a():\n"
        "    assert add(1, 1) == 2\n"
    )
    right = (
        "import pytest\n"
        "from math_utils import add\n\n"
        "def test_add_b():\n"
        "    assert add(2, 2) == 4\n"
    )
    merged = TestingAgent._merge_module_sources([left, right])
    assert merged.count("import pytest") == 1
    assert "def test_add_a" in merged
    assert "def test_add_b" in merged
    assert "def value" in merged


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_pipeline_generates_per_symbol_and_merges(
    _mock_index: Any, multi_module_repo: Path
) -> None:
    """Focused prompts run per public symbol and merge into module files."""
    from codebase_assistant.tracing.tracer import Tracer

    merged_suite = {
        "summary": "merged suite",
        "generated_tests": {
            "test_math_utils.py": (
                "from math_utils import add, sub\n\n"
                "def test_add():\n"
                "    assert add(1, 2) == 3\n\n"
                "def test_sub():\n"
                "    assert sub(3, 1) == 2\n"
            ),
            "test_auth.py": (
                "from auth import UserService, authenticate\n\n"
                "def test_authenticate():\n"
                "    assert authenticate('u') is True\n\n"
                "def test_login():\n"
                "    assert UserService().login('u') is True\n"
            ),
        },
        "coverage_estimate": 0.6,
    }

    def _payload_for_call(messages):
        system = messages[0].content
        user = messages[1].content
        if "repairing failing" in system.lower() or "PYTEST FAILURE OUTPUT" in user:
            return merged_suite
        if "FOCUS" in user and "function add()" in user:
            return {
                "summary": "tests for add",
                "generated_tests": {
                    "test_math_utils.py": (
                        "from math_utils import add\n\n"
                        "def test_add():\n"
                        "    assert add(1, 2) == 3\n"
                    )
                },
                "coverage_estimate": 0.5,
            }
        if "FOCUS" in user and "function sub()" in user:
            return {
                "summary": "tests for sub",
                "generated_tests": {
                    "test_math_utils.py": (
                        "from math_utils import sub\n\n"
                        "def test_sub():\n"
                        "    assert sub(3, 1) == 2\n"
                    )
                },
                "coverage_estimate": 0.5,
            }
        if "FOCUS" in user and "function authenticate()" in user:
            return {
                "summary": "tests for authenticate",
                "generated_tests": {
                    "test_auth.py": (
                        "from auth import authenticate\n\n"
                        "def test_authenticate():\n"
                        "    assert authenticate('u') is True\n"
                    )
                },
                "coverage_estimate": 0.4,
            }
        if "FOCUS" in user and "public methods of UserService" in user:
            return {
                "summary": "tests for UserService",
                "generated_tests": {
                    "test_auth.py": (
                        "from auth import UserService\n\n"
                        "def test_login():\n"
                        "    assert UserService().login('u') is True\n"
                    )
                },
                "coverage_estimate": 0.4,
            }
        raise AssertionError(f"unexpected prompt:\n{user[:500]}")

    client = _mock_client()

    def _generate(messages, **_kwargs):
        return ModelResponse(
            content=json.dumps(_payload_for_call(messages)),
            usage={},
            raw={},
        )

    client.generate.side_effect = _generate
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="ast-gen")

    response = agent.handle(
        AgentRequest(
            task_id="ast-1",
            agent_type=AgentType.TESTING,
            instruction="Generate pytest unit tests.",
            context={"repo_path": str(multi_module_repo)},
        )
    )

    assert response.success is True
    assert client.generate.call_count >= 4
    assert "test_math_utils.py" in response.output.generated_tests
    assert "test_auth.py" in response.output.generated_tests
    math_src = response.output.generated_tests["test_math_utils.py"]
    assert "def test_add" in math_src
    assert "def test_sub" in math_src
    auth_src = response.output.generated_tests["test_auth.py"]
    assert "def test_authenticate" in auth_src
    assert "def test_login" in auth_src
    names = agent.tracer.event_names()
    assert "testing_ast_scan_started" in names
    assert "testing_ast_scan_finished" in names
    assert "testing_symbol_generation_started" in names
    assert "testing_symbol_generation_finished" in names
    assert "testing_merge_completed" in names
    first_prompt = client.generate.call_args_list[0].args[0][1].content
    assert "FOCUS" in first_prompt
    assert "Generate pytest tests for the function" in first_prompt or (
        "public methods of" in first_prompt
    )


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_pipeline_abstains_when_no_public_symbols(
    _mock_index: Any, tmp_path: Path
) -> None:
    """No public symbols should abstain without calling the model."""
    (tmp_path / "secret.py").write_text(
        "def _only_private():\n    return 1\n",
        encoding="utf-8",
    )
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    response = agent.handle(
        AgentRequest(
            task_id="ast-2",
            agent_type=AgentType.TESTING,
            instruction="Generate pytest unit tests.",
            context={"repo_path": str(tmp_path)},
        )
    )

    assert response.success is False
    assert response.output.abstention is not None
    assert "no public symbols" in response.output.abstention.reason.lower()
    client.generate.assert_not_called()


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_existing_repair_loop_still_works_with_ast_generation(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Symbol-scoped generation still feeds the one-shot repair loop."""
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(_FAILING_PAYLOAD), usage={}, raw={}),
        ModelResponse(content=json.dumps(_REPAIRED_PAYLOAD), usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert client.generate.call_count == 2
    assert "Repair: attempted one fix iteration." in response.output.summary
    assert response.output.generated_tests["test_math_utils.py"] == _FIXED_TEST_SOURCE


# ---------------------------------------------------------------------------
# Real coverage measurement (pytest-cov)
# ---------------------------------------------------------------------------


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_successful_coverage_collection(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Measured pytest-cov coverage should populate coverage_estimate."""
    from codebase_assistant.tracing.tracer import Tracer

    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="cov-ok")

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert "Coverage:" in response.output.summary
    assert "line coverage" in response.output.summary.lower()
    assert "files measured" in response.output.summary.lower()
    assert 0.0 < response.output.coverage_estimate <= 1.0
    # Real coverage replaces the LLM placeholder (0.72).
    assert response.output.coverage_estimate != pytest.approx(0.72)
    names = agent.tracer.event_names()
    assert "testing_coverage_started" in names
    assert "testing_coverage_finished" in names


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_coverage_unavailable_without_pytest_cov(
    _mock_index: Any, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing pytest-cov should fall back without crashing."""
    import sys

    from codebase_assistant.tracing.tracer import Tracer

    for key in list(sys.modules):
        if key == "pytest_cov" or key.startswith("pytest_cov."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "pytest_cov" or name.startswith("pytest_cov."):
            raise ImportError("No module named pytest_cov")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())
    agent.tracer = Tracer(run_id="cov-missing")

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert "Coverage: unavailable (pytest-cov not installed)." in (
        response.output.summary
    )
    # Fallback keeps the model estimate.
    assert response.output.coverage_estimate == pytest.approx(0.72)
    assert "testing_coverage_failed" in agent.tracer.event_names()


def test_malformed_coverage_output_is_handled() -> None:
    """Malformed coverage JSON should not raise and should mark unavailable."""
    from codebase_assistant.agents.testing_agent import TestingAgent
    import tempfile

    agent = TestingAgent(model_client=None, retriever=None)
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, "coverage.json")
        with open(report, "w", encoding="utf-8") as handle:
            handle.write("{not-json")
        measurement = agent._parse_coverage_report(report, fallback_text="")

    assert measurement.available is True
    assert measurement.measured is False
    assert "malformed" in measurement.summary.lower()


def test_coverage_execution_failure_is_handled(
    sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverage pytest failures should degrade gracefully."""
    import pytest as pytest_api

    agent = TestingAgent(model_client=None, retriever=None)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("cov plugin crashed")

    monkeypatch.setattr(pytest_api, "main", _boom)
    measurement = agent._measure_coverage(
        pytest_api,
        workspace=str(sample_repo),
        temp_dir=str(sample_repo),
    )

    assert measurement.measured is False
    assert "unavailable" in measurement.summary.lower()
    assert "crashed" in measurement.error.lower()


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_coverage_with_multiple_files(
    _mock_index: Any, multi_module_repo: Path
) -> None:
    """Coverage over a multi-module repo should report files measured."""
    payload = {
        "summary": "tests for add and authenticate",
        "generated_tests": {
            "test_math_utils.py": (
                "from math_utils import add\n\n"
                "def test_add():\n"
                "    assert add(1, 2) == 3\n"
            ),
            "test_auth.py": (
                "from auth import authenticate\n\n"
                "def test_authenticate():\n"
                "    assert authenticate('u') is True\n"
            ),
        },
        "coverage_estimate": 0.11,
    }
    client = _mock_client(content=json.dumps(payload))
    agent = _agent(client, _mock_retriever())

    # Force a single generation path by targeting math_utils only would
    # under-test auth; call execution helper directly on multi-file suite.
    outcome = agent._run_generated_tests(
        str(multi_module_repo), payload["generated_tests"]
    )

    assert outcome.coverage is not None
    assert outcome.coverage.measured is True
    assert outcome.coverage.files_measured >= 2
    assert "files measured" in outcome.summary.lower()
    assert outcome.coverage.ratio == pytest.approx(
        outcome.coverage.percent / 100.0
    )


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_coverage_stored_in_testing_result(
    _mock_index: Any, sample_repo: Path
) -> None:
    """TestingResult.coverage_estimate must hold the measured ratio."""
    client = _mock_client(content=json.dumps(VALID_TEST_PAYLOAD))
    agent = _agent(client, _mock_retriever())

    result = agent.generate_unit_tests(str(sample_repo / "math_utils.py"))

    assert "Coverage:" in result.summary
    assert 0.0 < result.coverage_estimate <= 1.0
    assert result.coverage_estimate != pytest.approx(0.72)


@patch.object(TestingAgent, "_ensure_index", autospec=True)
def test_repair_loop_still_works_with_coverage(
    _mock_index: Any, sample_repo: Path
) -> None:
    """Coverage measurement must not break the one-shot repair loop."""
    client = _mock_client()
    client.generate.side_effect = [
        ModelResponse(content=json.dumps(_FAILING_PAYLOAD), usage={}, raw={}),
        ModelResponse(content=json.dumps(_REPAIRED_PAYLOAD), usage={}, raw={}),
    ]
    agent = _agent(client, _mock_retriever())

    response = agent.handle(_request(sample_repo))

    assert response.success is True
    assert client.generate.call_count == 2
    assert "Repair: attempted one fix iteration." in response.output.summary
    assert "Coverage:" in response.output.summary
    assert response.output.generated_tests["test_math_utils.py"] == _FIXED_TEST_SOURCE


def test_parse_coverage_term_fallback() -> None:
    """Terminal TOTAL lines should parse when JSON is absent."""
    term = (
        "Name                     Stmts   Miss  Cover\n"
        "--------------------------------------------\n"
        "math_utils.py                4      0   100%\n"
        "auth.py                      8      2    75%\n"
        "--------------------------------------------\n"
        "TOTAL                       12      2    83%\n"
    )
    measurement = TestingAgent._parse_coverage_term(term)
    assert measurement.measured is True
    assert measurement.percent == pytest.approx(83.0)
    assert measurement.statements == 12
    assert measurement.missing == 2
    assert measurement.files_measured == 2
