"""
supervisor.py
==============

Defines the Supervisor, the top-level orchestrator for Codebase
Assistant. The Supervisor receives high-level user goals, breaks them
down into tasks, routes those tasks to the appropriate specialized
agent (Code Analysis, Documentation, Testing), and aggregates results.

Goal routing is deterministic keyword matching: a goal selects one or
more agents, which always run in pipeline order (code analysis →
documentation → testing).

TODO: Replace keyword routing with LLM-driven task decomposition.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Dict, List, Optional, Pattern, Tuple

from .agents.base import BaseAgent
from .agents.code_analysis_agent import CodeAnalysisAgent
from .agents.documentation_agent import DocumentationAgent
from .agents.testing_agent import TestingAgent
from .config import Config
from .memory.conversation_memory import ConversationMemory
from .memory.memory_store import MemoryStore
from .models.model_client import LLMClient
from .models.providers.base import BaseProvider
from .models.providers.ollama_provider import OllamaProvider
from .models.providers.openrouter_provider import OpenRouterProvider
from .rag.indexer import Indexer
from .rag.retriever import Retriever
from .schemas.schemas import AgentRequest, AgentResponse, AgentType
from .tools.filesystem_tools import FilesystemTools
from .tools.github_tools import GitHubTools
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

#: Goal keywords that select each agent, listed in execution order:
#: code analysis, then documentation, then testing. Matching is on whole
#: words so "latest" does not select testing and "documented" does.
_GOAL_KEYWORDS: List[Tuple[AgentType, Pattern[str]]] = [
    (
        AgentType.CODE_ANALYSIS,
        re.compile(
            r"\b(analy[sz]e[sd]?|analy[sz]ing|analysis|review|inspect|"
            r"examine|audit|bugs?|smells?|complexity|quality|security|lint)\b"
        ),
    ),
    (
        AgentType.DOCUMENTATION,
        re.compile(
            r"\b(document|documents|documented|documenting|documentation|"
            r"docs?|docstrings?|readme|api reference)\b"
        ),
    ),
    (
        AgentType.TESTING,
        re.compile(
            r"\b(test|tests|tested|testing|pytest|unittest|unit tests?|coverage)\b"
        ),
    ),
]

#: Narrower documentation wordings, checked before defaulting to README.
_DOCSTRING_PATTERN = re.compile(r"\bdocstrings?\b")
_API_REFERENCE_PATTERN = re.compile(r"\b(api reference|api docs?)\b")


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

        # Shared OpenRouter provider for code analysis. Construction
        # failures never abort startup; static-only mode still works.
        self.provider: Optional[BaseProvider] = self._init_openrouter_provider()
        # Shared Ollama provider, kept for local-model experiments. Not
        # injected into any agent.
        self.ollama_provider: Optional[BaseProvider] = self._init_ollama_provider()
        self.model_client = LLMClient(
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens,
            provider=self.provider,
            config=self.config,
        )
        # Local-model client kept separate from the OpenRouter client. No
        # agent is wired to it by default. May have provider=None when
        # Ollama construction failed; that must not prevent Supervisor
        # construction.
        self.ollama_model_client = LLMClient(
            model_name=self.config.ollama_model,
            max_tokens=self.config.max_tokens,
            provider=self.ollama_provider,
            config=self.config,
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

    def _init_openrouter_provider(self) -> Optional[BaseProvider]:
        """
        Construct the shared OpenRouterProvider from Config.

        A missing API key or probe failure does not prevent construction;
        the provider simply reports itself unavailable. Only unexpected
        construction errors are swallowed so Supervisor startup never
        crashes because of the model layer.

        Returns:
            The OpenRouterProvider instance, or None if construction
            itself failed.
        """
        try:
            provider = OpenRouterProvider(
                model=self.config.openrouter_model or self.config.claude_model,
                api_key=self.config.openrouter_api_key,
                max_tokens=self.config.max_tokens,
                base_url=self.config.openrouter_base_url,
                config=self.config,
            )
        except Exception as exc:
            logger.warning(
                "OpenRouterProvider initialization failed; continuing without "
                "a model provider: %s",
                exc,
            )
            return None

        if not provider.is_available():
            logger.info(
                "OpenRouterProvider is configured but unavailable; "
                "analysis will run in static-only mode until a valid "
                "OPENROUTER_API_KEY and network path are present."
            )
        else:
            logger.info(
                "OpenRouterProvider is available (model=%s).",
                provider.model,
            )
        return provider

    def _init_ollama_provider(self) -> Optional[BaseProvider]:
        """
        Construct the shared OllamaProvider from Config.

        Held on the Supervisor for local-model experiments. Not injected
        into any agent. Construction or availability failures never abort
        startup and never affect OpenRouter wiring.

        Returns:
            The OllamaProvider instance, or None if construction itself
            failed.
        """
        try:
            provider = OllamaProvider(
                model=self.config.ollama_model,
                max_tokens=self.config.max_tokens,
                base_url=self.config.ollama_base_url,
                config=self.config,
            )
        except Exception as exc:
            logger.warning(
                "OllamaProvider initialization failed; continuing without "
                "an Ollama provider: %s",
                exc,
            )
            return None

        logger.info(
            "OllamaProvider initialized (model=%s, base_url=%s).",
            provider.model,
            provider.base_url,
        )
        if not provider.is_available():
            logger.info(
                "Ollama unavailable; local-model calls will fail until "
                "Ollama is running."
            )
        else:
            logger.info(
                "Ollama available (model=%s).",
                provider.model,
            )
        return provider

    def _init_agents(self) -> Dict[AgentType, BaseAgent]:
        """
        Construct and wire up the specialized agents.

        Returns:
            A mapping from AgentType to the corresponding agent instance.
        """
        # CodeAnalysisAgent builds a per-repository Indexer under
        # chroma/<repo-hash>/. Injecting the Supervisor's default
        # Retriever would point retrieval at a different store than the
        # agent just indexed, so leave retriever unset and let _bind()
        # attach a Retriever to that Indexer.
        #
        # DocumentationAgent uses the OpenRouter-backed client: repository
        # documentation needs longer, better-structured prose than the
        # local model produced. `ollama_model_client` stays available on
        # the Supervisor for local-model experiments.
        documentation_agent = DocumentationAgent(
            model_client=self.model_client,
            tool_registry=self.tool_registry,
            retriever=self.retriever,
            memory_store=self.memory_store,
        )
        logger.info(
            "DocumentationAgent received an OpenRouter-backed LLMClient "
            "(provider=%s).",
            type(self.provider).__name__ if self.provider is not None else None,
        )

        return {
            AgentType.CODE_ANALYSIS: CodeAnalysisAgent(
                model_client=self.model_client,
                tool_registry=self.tool_registry,
                memory_store=self.memory_store,
            ),
            AgentType.DOCUMENTATION: documentation_agent,
            AgentType.TESTING: TestingAgent(
                model_client=self.model_client,
                tool_registry=self.tool_registry,
                retriever=self.retriever,
                memory_store=self.memory_store,
            ),
        }

    def handle_goal(
        self, goal: str, repo_path: Optional[str] = None
    ) -> List[AgentResponse]:
        """
        Handle a high-level user goal by routing it to the agents whose
        keywords it mentions and collecting their responses.

        Routing is deterministic keyword matching, not model-driven
        planning. A goal can select several agents ("analyze and
        document"), and selected agents always run in pipeline order:
        code analysis, then documentation, then testing. A goal that
        matches nothing falls back to code analysis, matching
        `route_task`.

        Args:
            goal: Natural language description of what the user wants
                accomplished.
            repo_path: Repository the agents should operate on. Defaults
                to the configured workspace root.

        Returns:
            One AgentResponse per selected agent, in execution order. An
            agent that raises is reported as a failed AgentResponse so a
            single failure never discards the other results.
        """
        target = str(repo_path or self.config.workspace_root or ".")
        agent_types = self._select_agent_types(goal)

        logger.info(
            "Goal routed to %d agent(s): %s",
            len(agent_types),
            ", ".join(agent_type.value for agent_type in agent_types),
        )

        responses: List[AgentResponse] = []
        for agent_type in agent_types:
            request = self._build_request(agent_type, goal, target)
            logger.info("Dispatching %s for goal.", agent_type.value)
            try:
                responses.append(self.dispatch(request))
            except Exception as exc:
                logger.warning("%s failed while handling the goal: %s", agent_type.value, exc)
                responses.append(
                    AgentResponse(
                        task_id=request.task_id,
                        agent_type=agent_type,
                        success=False,
                        output=None,
                        errors=[str(exc)],
                    )
                )
        return responses

    def _select_agent_types(self, goal: str) -> List[AgentType]:
        """
        Choose which agents a goal asks for, in execution order.

        Args:
            goal: The user's natural language goal.

        Returns:
            The matching AgentTypes ordered code analysis → documentation
            → testing. Defaults to code analysis when nothing matches.
        """
        normalized = (goal or "").lower()
        selected = [
            agent_type
            for agent_type, pattern in _GOAL_KEYWORDS
            if pattern.search(normalized)
        ]
        if not selected:
            logger.info(
                "Goal matched no routing keywords; defaulting to code analysis."
            )
            return [AgentType.CODE_ANALYSIS]
        return selected

    def _build_request(
        self, agent_type: AgentType, goal: str, repo_path: str
    ) -> AgentRequest:
        """
        Build the AgentRequest for one agent selected by a goal.

        Args:
            agent_type: Agent the request is for.
            goal: The original user goal, passed through as the
                instruction so agents keep their own interpretation.
            repo_path: Repository to operate on.

        Returns:
            An AgentRequest carrying the repository under both key names
            the agents accept.
        """
        context: Dict[str, object] = {
            "repo_path": repo_path,
            "repository_path": repo_path,
        }
        if agent_type is AgentType.DOCUMENTATION:
            context["doc_type"] = self._documentation_type(goal)
        return AgentRequest(
            task_id=self.new_task_id(),
            agent_type=agent_type,
            instruction=goal,
            context=context,
        )

    @staticmethod
    def _documentation_type(goal: str) -> str:
        """
        Pick the documentation mode a goal is asking for.

        A repository-level goal wants a README overview, so that is the
        default; narrower wordings select the narrower modes the
        DocumentationAgent already supports.

        Args:
            goal: The user's natural language goal.

        Returns:
            One of "docstring", "api_reference", or "readme".
        """
        normalized = (goal or "").lower()
        if _DOCSTRING_PATTERN.search(normalized):
            return "docstring"
        if _API_REFERENCE_PATTERN.search(normalized):
            return "api_reference"
        return "readme"

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
