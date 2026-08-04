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

import inspect
import logging
import re
import uuid
from typing import Callable, Dict, List, Optional, Pattern, Tuple

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
from .models.providers.provider_manager import ProviderManager
from .rag.indexer import Indexer
from .rag.retriever import Retriever
from .schemas.schemas import AgentRequest, AgentResponse, AgentType
from .tools.filesystem_tools import FilesystemTools
from .tools.github_tools import GitHubTools
from .tools.registry import ToolRegistry
from .tracing.events import TraceEventType
from .tracing.tracer import Tracer

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

        # Shared run tracer — never aborts startup if construction fails.
        try:
            self.tracer = Tracer(run_id=str(uuid.uuid4()))
        except Exception as exc:
            logger.warning("Tracer construction failed; continuing without tracing: %s", exc)
            self.tracer = Tracer(enabled=False)

        # Shared OpenRouter + Ollama providers. Construction failures
        # never abort startup; static-only mode still works.
        self.provider: Optional[BaseProvider] = self._init_openrouter_provider()
        self.ollama_provider: Optional[BaseProvider] = self._init_ollama_provider()
        # Transparent OpenRouter → Ollama failover behind one client.
        self.provider_manager = ProviderManager(
            preferred=self.provider,
            fallback=self.ollama_provider,
            preferred_name=self.config.preferred_provider,
            fallback_name=self.config.fallback_provider,
            cache_seconds=self.config.provider_cache_seconds,
            tracer=self.tracer,
        )
        self.model_client = LLMClient(
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens,
            provider=self.provider_manager,
            config=self.config,
        )
        # Direct Ollama client for local-model experiments. Agents use
        # model_client (with failover), not this handle.
        self.ollama_model_client = LLMClient(
            model_name=self.config.ollama_model,
            max_tokens=self.config.max_tokens,
            provider=self.ollama_provider,
            config=self.config,
        )
        self.tool_registry = ToolRegistry()
        self.indexer = Indexer(vector_store_path=self.config.vector_store_path)
        self.retriever = Retriever(vector_store_path=self.config.vector_store_path)
        self.memory_store = MemoryStore(
            storage_path=self.config.memory_store_path,
            tracer=self.tracer,
        )
        # Short-term session history. Uses the OpenRouter-backed client
        # for summarization and the shared MemoryStore so conversations
        # reload across runs.
        self.conversation_memory = ConversationMemory(
            model_client=self.model_client,
            memory_store=self.memory_store,
            tracer=self.tracer,
        )

        # Tools.
        self.github_tools = GitHubTools(token=self.config.github_token)
        self.filesystem_tools = FilesystemTools(workspace_root=self.config.workspace_root)
        self._register_tools()

        # Agents.
        self.agents: Dict[AgentType, BaseAgent] = self._init_agents()

        # TODO: register MCP-based tools once MCP integration is implemented

    def _trace(self, name: str, *, success: Optional[bool] = True, **metadata: object) -> None:
        """Record a Supervisor lifecycle event; never raises."""
        try:
            self.tracer.record(
                TraceEventType.LIFECYCLE,
                name,
                component="Supervisor",
                success=success,
                **metadata,
            )
        except Exception as exc:
            logger.warning("Supervisor tracing failed for %r: %s", name, exc)

    def provider_status_message(self) -> str:
        """
        One-line CLI summary of the active LLM provider.

        Returns:
            Status such as ``Using OpenRouter (Claude Sonnet 4)`` or an
            Ollama fallback / static-only message.
        """
        try:
            return self.provider_manager.status_message()
        except Exception as exc:
            logger.warning("Provider status message failed: %s", exc)
            return "LLM provider status unavailable"

    def _register_tools(self) -> None:
        """
        Expose the shared tool instances through the ToolRegistry.

        Bound methods of the already-constructed FilesystemTools and
        GitHubTools are registered, so the registry is a lookup table
        over the existing instances rather than a second implementation.
        Names are namespaced (`filesystem.read_file`, `github.clone_repository`)
        because both classes expose similarly named operations.

        A tool that is already registered is skipped with a warning:
        double registration is a wiring mistake, not a reason to abort
        Supervisor startup.
        """
        for namespace, instance in (
            ("filesystem", self.filesystem_tools),
            ("github", self.github_tools),
        ):
            for name, handler in self._public_methods(instance):
                qualified = f"{namespace}.{name}"
                try:
                    self.tool_registry.register_tool(qualified, handler)
                except (ValueError, TypeError) as exc:
                    logger.warning("Could not register tool %s: %s", qualified, exc)

        registered = self.tool_registry.list_tools()
        logger.info(
            "Registered %d tool(s) in the ToolRegistry: %s",
            len(registered),
            ", ".join(registered),
        )

    @staticmethod
    def _public_methods(instance: object) -> List[Tuple[str, Callable[..., object]]]:
        """
        Collect the public callables a tool instance exposes.

        Args:
            instance: Tool instance to introspect.

        Returns:
            (name, bound callable) pairs for every public method, sorted
            by name. Underscore-prefixed helpers are internal to the tool
            and stay out of the registry.
        """
        return [
            (name, member)
            for name, member in inspect.getmembers(instance)
            if not name.startswith("_")
            and (inspect.ismethod(member) or inspect.isfunction(member))
        ]

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

        Held on the Supervisor and used as the failover target behind
        ProviderManager. Construction or availability failures never
        abort startup.

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
            tracer=self.tracer,
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
                tracer=self.tracer,
            ),
            AgentType.DOCUMENTATION: documentation_agent,
            AgentType.TESTING: TestingAgent(
                model_client=self.model_client,
                tool_registry=self.tool_registry,
                retriever=self.retriever,
                memory_store=self.memory_store,
                tracer=self.tracer,
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
            One AgentResponse per selected agent, in execution order.
            Responses are aggregated verbatim: each agent's own success,
            output, and errors are passed through untouched. A failing
            agent is recorded as a failed response and the remaining
            agents still run, so a partial pipeline still returns every
            result it managed to produce.
        """
        target = str(repo_path or self.config.workspace_root or ".")
        self._trace("goal_received", goal=goal, repo_path=target)
        agent_types = self._select_agent_types(goal)
        self._trace(
            "routing_decision",
            goal=goal,
            agents=[agent_type.value for agent_type in agent_types],
        )

        logger.info(
            "Goal routed to %d agent(s): %s",
            len(agent_types),
            ", ".join(agent_type.value for agent_type in agent_types),
        )

        responses: List[AgentResponse] = []
        for agent_type in agent_types:
            request = self._build_request(agent_type, goal, target)
            logger.info("Dispatching %s for goal.", agent_type.value)
            self._trace(
                "dispatch_start",
                agent=agent_type.value,
                task_id=request.task_id,
            )
            response = self._dispatch_safely(request)
            self._trace(
                "dispatch_finish",
                agent=agent_type.value,
                task_id=request.task_id,
                success=response.success,
                errors=list(response.errors or []),
            )
            if not response.success:
                logger.warning(
                    "%s reported failure; continuing with the remaining agents.",
                    agent_type.value,
                )
            responses.append(response)

        self._log_aggregate(responses)
        self._trace(
            "aggregation_complete",
            agents=[r.agent_type.value for r in responses],
            succeeded=sum(1 for r in responses if r.success),
            failed=sum(1 for r in responses if not r.success),
        )
        return responses

    @staticmethod
    def _log_aggregate(responses: List[AgentResponse]) -> None:
        """
        Log the outcome of an aggregated multi-agent run.

        Args:
            responses: The collected responses, in execution order.
        """
        succeeded = [r.agent_type.value for r in responses if r.success]
        failed = [r.agent_type.value for r in responses if not r.success]
        logger.info(
            "Goal complete: %d/%d agent(s) succeeded (ok: %s; failed: %s).",
            len(succeeded),
            len(responses),
            ", ".join(succeeded) or "none",
            ", ".join(failed) or "none",
        )

    def _dispatch_safely(self, request: AgentRequest) -> AgentResponse:
        """
        Dispatch one request, converting a raised error into a response.

        Orchestration reports failures as data so one broken agent never
        discards the results of the others or aborts the caller.

        Args:
            request: The request to dispatch.

        Returns:
            The agent's AgentResponse, or a failed AgentResponse
            describing the exception it raised.
        """
        try:
            return self.dispatch(request)
        except Exception as exc:
            logger.warning(
                "%s failed while handling a request: %s",
                request.agent_type.value,
                exc,
            )
            return AgentResponse(
                task_id=request.task_id,
                agent_type=request.agent_type,
                success=False,
                output=None,
                errors=[str(exc)],
            )

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
        Build the AgentRequest for one agent selected by a goal or task.

        Args:
            agent_type: Agent the request is for.
            goal: The original goal or task name, passed through as the
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

    def handle_task(self, task_name: str, repo_path: str) -> AgentResponse:
        """
        Route one task to the real pipeline of the matching agent.

        The task name selects an agent by keyword and is passed through
        as the instruction, so the agent runs its full `handle()`
        pipeline (retrieval, prompting, grounding) rather than the
        placeholder `run()` entry point.

        Args:
            task_name: Name/description of the task, used for routing.
            repo_path: Path to the repository the agent should operate on.

        Returns:
            The AgentResponse produced by the selected agent. A task name
            matching no agent returns a failed AgentResponse naming the
            supported task types; nothing is raised.
        """
        agent_type = self._route_task_strict(task_name)
        self._trace(
            "task_received",
            task_name=task_name,
            repo_path=str(repo_path or ""),
        )
        if agent_type is None:
            self._trace(
                "routing_decision",
                task_name=task_name,
                agents=[],
                success=False,
            )
            response = self._unknown_task_response(task_name)
            self._trace(
                "aggregation_complete",
                agents=[],
                succeeded=0,
                failed=1,
            )
            return response

        target = str(repo_path or self.config.workspace_root or ".")
        request = self._build_request(agent_type, task_name, target)
        self._trace(
            "routing_decision",
            task_name=task_name,
            agents=[agent_type.value],
        )
        logger.info("Dispatching %s for task %r.", agent_type.value, task_name)
        self._trace(
            "dispatch_start",
            agent=agent_type.value,
            task_id=request.task_id,
        )
        response = self._dispatch_safely(request)
        self._trace(
            "dispatch_finish",
            agent=agent_type.value,
            task_id=request.task_id,
            success=response.success,
            errors=list(response.errors or []),
        )
        self._trace(
            "aggregation_complete",
            agents=[response.agent_type.value],
            succeeded=1 if response.success else 0,
            failed=0 if response.success else 1,
        )
        return response

    @staticmethod
    def _route_task_strict(task_name: str) -> Optional[AgentType]:
        """
        Route a task name to one agent, or to nothing when unrecognized.

        Unlike `route_task`, which treats code analysis as a catch-all,
        this reports an unmatched task so the caller can explain what it
        supports instead of silently running the wrong pipeline. A name
        mentioning several agents takes the earliest in pipeline order.

        Args:
            task_name: Name/description of the task.

        Returns:
            The matching AgentType, or None when no keyword matches.
        """
        normalized = (task_name or "").lower()
        for agent_type, pattern in _GOAL_KEYWORDS:
            if pattern.search(normalized):
                return agent_type
        return None

    def _unknown_task_response(self, task_name: str) -> AgentResponse:
        """
        Build the failed response for a task that matched no agent.

        Args:
            task_name: The unrecognized task name.

        Returns:
            A failed AgentResponse naming the supported task types.
        """
        logger.warning("Task %r matched no agent; returning a failed response.", task_name)
        # AgentType has no "none" member and schemas are fixed, so the
        # response is attributed to the codebase-wide default agent. The
        # failure itself is carried by success=False and errors.
        return AgentResponse(
            task_id=self.new_task_id(),
            agent_type=AgentType.CODE_ANALYSIS,
            success=False,
            output=None,
            errors=[
                f"Unknown task {task_name!r}. Supported tasks mention "
                f"analysis (analyze, review, bugs), documentation "
                f"(document, readme, docstrings), or testing (tests, "
                f"pytest, coverage)."
            ],
        )

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
