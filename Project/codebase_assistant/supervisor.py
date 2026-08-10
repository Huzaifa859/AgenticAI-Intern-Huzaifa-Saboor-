"""
supervisor.py
==============

Defines the Supervisor, the top-level orchestrator for Codebase
Assistant. The Supervisor receives high-level user goals, breaks them
down into tasks, routes those tasks to the appropriate specialized
agent (Code Analysis, Documentation, Testing), and aggregates results.

TODO: Implement real task planning/decomposition (likely LLM-driven),
routing logic, and result aggregation.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from .agents.base import BaseAgent
from .agents.code_analysis_agent import CodeAnalysisAgent
from .agents.documentation_agent import DocumentationAgent
from .agents.testing_agent import TestingAgent
from .config import Config
from .memory.conversation_memory import ConversationMemory
from .memory.memory_store import MemoryStore
from .models.model_client import LLMClient
from .rag.indexer import Indexer
from .rag.retriever import Retriever
from .schemas.schemas import AgentRequest, AgentResponse, AgentType
from .tools.filesystem_tools import FilesystemTools
from .tools.github_tools import GitHubTools
from .tools.registry import ToolRegistry


class Supervisor:
    """
    Top-level orchestrator that coordinates the specialized agents,
    shared tool registry, RAG subsystem, memory subsystem, and model
    client to fulfill user goals.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize the Supervisor and wire up all shared subsystems.

        Args:
            config: Optional Config instance. If not provided, a
                default Config is loaded.
        """
        self.config = config or Config.load()

        # Shared subsystems.
        self.model_client = LLMClient(
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens,
        )
        self.tool_registry = ToolRegistry()
        self.indexer = Indexer(vector_store_path=self.config.vector_store_path)
        self.retriever = Retriever(vector_store_path=self.config.vector_store_path)
        self.memory_store = MemoryStore(storage_path=self.config.memory_store_path)
        self.conversation_memory = ConversationMemory()

        # Tools.
        self.github_tools = GitHubTools(token=self.config.github_token)
        self.filesystem_tools = FilesystemTools(workspace_root=self.config.workspace_root)

        # Agents.
        self.agents: Dict[AgentType, BaseAgent] = self._init_agents()

        # TODO: register GitHub/Filesystem tool methods into self.tool_registry
        # TODO: register MCP-based tools once MCP integration is implemented

    def _init_agents(self) -> Dict[AgentType, BaseAgent]:
        """
        Construct and wire up the specialized agents.

        Returns:
            A mapping from AgentType to the corresponding agent instance.
        """
        shared_kwargs = dict(
            model_client=self.model_client,
            tool_registry=self.tool_registry,
            retriever=self.retriever,
            memory_store=self.memory_store,
        )
        return {
            AgentType.CODE_ANALYSIS: CodeAnalysisAgent(**shared_kwargs),
            AgentType.DOCUMENTATION: DocumentationAgent(**shared_kwargs),
            AgentType.TESTING: TestingAgent(**shared_kwargs),
        }

    def handle_goal(self, goal: str) -> List[AgentResponse]:
        """
        Handle a high-level user goal by decomposing it into tasks,
        routing them to the appropriate agents, and collecting results.

        Args:
            goal: Natural language description of what the user wants
                accomplished.

        Returns:
            A list of AgentResponse objects from all dispatched tasks
            (placeholder empty list).

        TODO: Implement real goal decomposition (likely via the model
        client), task routing, and aggregation of agent responses.
        """
        # TODO: implement real goal decomposition and orchestration
        return []

    def route_task(self, task_name: str) -> AgentType:
        """
        Determine which agent type should handle a given task name.

        Keyword-based routing: "test" -> Testing, "doc" ->
        Documentation, everything else -> Code Analysis (treated as
        the default/general-purpose agent).

        Args:
            task_name: Name/description of the task.

        Returns:
            The AgentType best matching the task name.

        TODO: Replace with LLM-driven routing once real functionality
        is implemented.
        """
        normalized = task_name.lower()
        if "test" in normalized:
            return AgentType.TESTING
        if "doc" in normalized:
            return AgentType.DOCUMENTATION
        return AgentType.CODE_ANALYSIS

    def handle_task(self, task_name: str, repo_path: str) -> Dict[str, str]:
        """
        Route a task to the appropriate agent's simple `run` method and
        return its (currently fake) result.

        Args:
            task_name: Name/description of the task, used for routing.
            repo_path: Path to the repository the agent should operate on.

        Returns:
            A dict with "status" and "message" keys produced by the
            selected agent.
        """
        agent_type = self.route_task(task_name)
        agent = self.agents[agent_type]
        return agent.run(repo_path)

    def dispatch(self, request: AgentRequest) -> AgentResponse:
        """
        Dispatch a single AgentRequest to the appropriate agent.

        Args:
            request: The request to dispatch.

        Returns:
            The AgentResponse produced by the target agent.

        TODO: Add error handling, retries, and logging around the
        agent invocation.
        """
        agent = self.agents.get(request.agent_type)
        if agent is None:
            return AgentResponse(
                task_id=request.task_id,
                agent_type=request.agent_type,
                success=False,
                output=None,
                errors=[f"No agent registered for type: {request.agent_type}"],
            )
        # TODO: wrap in try/except with proper error propagation
        return agent.handle(request)

    def new_task_id(self) -> str:
        """
        Generate a new unique task identifier.

        Returns:
            A UUID4 string.
        """
        return str(uuid.uuid4())
