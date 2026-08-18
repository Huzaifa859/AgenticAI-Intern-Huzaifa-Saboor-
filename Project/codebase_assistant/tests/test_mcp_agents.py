"""
test_mcp_agents.py
===================

MCP agent endpoints: analysis/documentation/testing/goal.run.

Providers are mocked offline. Supervisor and ToolRegistry stay real.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from codebase_assistant.config import Config
from codebase_assistant.mcp.client import MCPClient
from codebase_assistant.mcp.server import MCPServer, _RUNNING_SERVERS
from codebase_assistant.schemas.schemas import (
    AgentType,
    ModelResponse,
)
from codebase_assistant.supervisor import Supervisor


VALID_DOC = {
    "file_path": "README.md",
    "function_name": "README",
    "summary": "# Demo\n\nA tiny repository.",
    "parameters": [],
    "returns": "",
    "example_usage": "",
}

VALID_TESTS = {
    "summary": "Generated tests for add.",
    "generated_tests": {
        "test_math_utils.py": (
            "from math_utils import add\n\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
        )
    },
    "coverage_estimate": 0.5,
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Tiny local repository for agent pipelines."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def config(repo: Path) -> Config:
    return Config(
        workspace_root=str(repo),
        memory_store_path=str(repo / ".memory"),
        chroma_persist_directory=str(repo / ".chroma"),
        github_token=None,
    )


@pytest.fixture
def supervisor(config: Config) -> Supervisor:
    with patch.object(Supervisor, "_init_openrouter_provider", return_value=None), \
        patch.object(Supervisor, "_init_ollama_provider", return_value=None):
        return Supervisor(config=config)


def _enable_llm(supervisor: Supervisor, content: str) -> MagicMock:
    """Attach a fake available LLM client to documentation and testing agents."""
    client = MagicMock()
    client.is_available.return_value = True
    client.generate.return_value = ModelResponse(
        content=content, usage={}, raw={}
    )
    supervisor.agents[AgentType.DOCUMENTATION].model_client = client
    supervisor.agents[AgentType.TESTING].model_client = client
    supervisor.agents[AgentType.CODE_ANALYSIS].model_client = client
    return client


@pytest.fixture
def server(supervisor: Supervisor) -> MCPServer:
    _RUNNING_SERVERS.clear()
    instance = MCPServer(host="localhost", port=18100, supervisor=supervisor)
    yield instance
    instance.shutdown()
    _RUNNING_SERVERS.clear()


def test_agent_tools_registered_on_tool_registry(server: MCPServer) -> None:
    """The four MCP agent tools appear on ToolRegistry after start."""
    assert server.start()["ok"] is True
    names = set(server.tool_registry.list_tools())
    assert "analysis.run" in names
    assert "documentation.run" in names
    assert "testing.run" in names
    assert "goal.run" in names


def test_analysis_run_local_repository(server: MCPServer, repo: Path) -> None:
    """analysis.run executes Supervisor.handle_task for a local repo."""
    server.start()
    client = MCPClient("localhost:18100")
    client.connect()

    result = client.call_tool(
        "analysis.run",
        {
            "repository": str(repo),
            "question": "Find security bugs",
        },
    )
    assert result["ok"] is True
    assert isinstance(result["result"], dict)
    assert "findings" in result["result"] or "repository_path" in result["result"]
    assert "mcp_request" in server.supervisor.tracer.event_names()
    assert "mcp_response" in server.supervisor.tracer.event_names()
    assert "tool_invoked" in server.supervisor.tracer.event_names()


def test_documentation_run(server: MCPServer, repo: Path, supervisor: Supervisor) -> None:
    """documentation.run returns DocumentationResult via Supervisor."""
    _enable_llm(supervisor, json.dumps(VALID_DOC))
    with patch(
        "codebase_assistant.agents.documentation_agent.DocumentationAgent._ensure_index"
    ):
        server.start()
        client = MCPClient("localhost:18100")
        client.connect()
        result = client.call_tool(
            "documentation.run",
            {"repository": str(repo), "target": "README"},
        )

    assert result["ok"] is True
    assert result["result"]["function_name"] == "README"
    assert "Demo" in result["result"]["summary"]


def test_testing_run(server: MCPServer, repo: Path, supervisor: Supervisor) -> None:
    """testing.run returns TestingResult via Supervisor."""
    _enable_llm(supervisor, json.dumps(VALID_TESTS))
    with patch(
        "codebase_assistant.agents.testing_agent.TestingAgent._ensure_index"
    ):
        server.start()
        client = MCPClient("localhost:18100")
        client.connect()
        result = client.call_tool(
            "testing.run",
            {"repository": str(repo), "target": "math_utils.py"},
        )

    assert result["ok"] is True
    assert "generated_tests" in result["result"]
    assert "test_math_utils.py" in result["result"]["generated_tests"]


def test_goal_run(server: MCPServer, repo: Path, supervisor: Supervisor) -> None:
    """goal.run returns the ordered AgentResponse list from handle_goal."""
    _enable_llm(supervisor, json.dumps(VALID_DOC))
    with patch(
        "codebase_assistant.agents.documentation_agent.DocumentationAgent._ensure_index"
    ), patch(
        "codebase_assistant.agents.testing_agent.TestingAgent._ensure_index"
    ):
        # testing agent needs valid test JSON when selected; switch content per call
        def _generate(messages, **kwargs):
            text = json.dumps(VALID_DOC)
            joined = " ".join(getattr(m, "content", "") for m in messages)
            if "pytest" in joined.lower() or "unit test" in joined.lower():
                text = json.dumps(VALID_TESTS)
            return ModelResponse(content=text, usage={}, raw={})

        supervisor.agents[AgentType.DOCUMENTATION].model_client.generate.side_effect = (
            _generate
        )
        supervisor.agents[AgentType.TESTING].model_client.generate.side_effect = (
            _generate
        )

        server.start()
        client = MCPClient("localhost:18100")
        client.connect()
        result = client.call_tool(
            "goal.run",
            {
                "repository": str(repo),
                "goal": "Analyze, document and generate tests.",
            },
        )

    assert result["ok"] is True
    assert isinstance(result["result"], list)
    assert len(result["result"]) == 3
    agent_types = [item["agent_type"] for item in result["result"]]
    assert agent_types == [
        AgentType.CODE_ANALYSIS.value,
        AgentType.DOCUMENTATION.value,
        AgentType.TESTING.value,
    ]


def test_github_url_accepted(server: MCPServer, repo: Path) -> None:
    """GitHub URLs are prepared through existing GitHubTools clone logic."""
    server.start()
    client = MCPClient("localhost:18100")
    client.connect()

    def _clone(url: str, destination: str = ".") -> bool:
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "math_utils.py").write_text(
            "def add(a, b):\n    return a + b\n",
            encoding="utf-8",
        )
        return True

    with patch.object(
        server.supervisor.github_tools,
        "is_remote_reference",
        return_value=True,
    ), patch.object(
        server.supervisor.github_tools,
        "validate_repository",
        return_value=True,
    ), patch.object(
        server.supervisor.github_tools,
        "clone_repository",
        side_effect=_clone,
    ):
        result = client.call_tool(
            "analysis.run",
            {
                "repository": "https://github.com/example/demo",
                "question": "Find bugs",
            },
        )

    assert result["ok"] is True


def test_local_repository_accepted(server: MCPServer, repo: Path) -> None:
    """Local repository paths are accepted without cloning."""
    server.start()
    client = MCPClient("localhost:18100")
    client.connect()
    result = client.call_tool(
        "analysis.run",
        {"repository": str(repo), "question": "Review code quality"},
    )
    assert result["ok"] is True


def test_tracing_events_emitted(server: MCPServer, repo: Path) -> None:
    """MCP agent calls emit request/response/tool_invoked trace events."""
    server.start()
    client = MCPClient("localhost:18100")
    client.connect()
    client.call_tool(
        "analysis.run",
        {"repository": str(repo), "question": "Find bugs"},
    )
    names = server.supervisor.tracer.event_names()
    assert "mcp_request" in names
    assert "tool_invoked" in names
    assert "mcp_response" in names
    events = [
        event
        for event in server.supervisor.tracer.get_events()
        if event.name == "mcp_response"
    ]
    assert events
    assert events[-1].success is True
    assert events[-1].duration_ms is not None


def test_failures_returned_correctly(server: MCPServer) -> None:
    """Missing repository returns a structured MCP error."""
    server.start()
    client = MCPClient("localhost:18100")
    client.connect()
    result = client.call_tool(
        "analysis.run",
        {"repository": "", "question": "Find bugs"},
    )
    assert result["ok"] is False
    assert result["code"] in {"invalid_arguments", "tool_error", "repository_not_found"}


def test_repository_not_found(server: MCPServer, tmp_path: Path) -> None:
    """Non-existent local path fails with repository_not_found."""
    server.start()
    client = MCPClient("localhost:18100")
    client.connect()
    missing = tmp_path / "does-not-exist"
    result = client.call_tool(
        "analysis.run",
        {"repository": str(missing), "question": "Find bugs"},
    )
    assert result["ok"] is False
    assert result["code"] == "repository_not_found"


def test_provider_unavailable_for_documentation(
    server: MCPServer, repo: Path
) -> None:
    """Documentation with no provider returns provider_unavailable."""
    server.start()
    client = MCPClient("localhost:18100")
    client.connect()
    result = client.call_tool(
        "documentation.run",
        {"repository": str(repo), "target": "README"},
    )
    assert result["ok"] is False
    assert result["code"] == "provider_unavailable"
