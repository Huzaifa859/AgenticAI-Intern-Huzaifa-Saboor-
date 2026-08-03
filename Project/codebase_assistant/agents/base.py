"""
base.py
=======

Defines BaseAgent, the abstract base class that all specialized agents
(Code Analysis, Documentation, Testing) inherit from.

Shared helpers resolve FilesystemTools and GitHubTools through the
injected ToolRegistry so agents do not construct those tools themselves
when the Supervisor has already registered them.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from ..memory.memory_store import MemoryStore
from ..models.model_client import LLMClient
from ..rag.retriever import Retriever
from ..schemas.schemas import AgentRequest, AgentResponse, AgentType
from ..tools.registry import ToolRegistry
from ..tracing.events import TraceEventType
from ..tracing.tracer import Tracer

if TYPE_CHECKING:
    from ..config import Config
    from ..tools.filesystem_tools import FilesystemTools
    from ..tools.github_tools import GitHubTools

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.

    Provides shared references to the LLMClient, ToolRegistry,
    Retriever, and MemoryStore so that concrete agents can focus on
    their domain-specific logic.
    """

    agent_type: AgentType

    def __init__(
        self,
        model_client: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        retriever: Optional[Retriever] = None,
        memory_store: Optional[MemoryStore] = None,
        tracer: Optional[Tracer] = None,
    ) -> None:
        """
        Initialize the BaseAgent with its shared dependencies.

        Args:
            model_client: LLMClient used to make model calls.
            tool_registry: Registry used to invoke tools.
            retriever: RAG retriever used to fetch relevant context.
            memory_store: Long-term memory store.
            tracer: Optional shared Tracer for lifecycle events.
        """
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.retriever = retriever
        self.memory_store = memory_store
        self.tracer = tracer

    def _trace(
        self,
        name: str,
        *,
        component: Optional[str] = None,
        event_type: TraceEventType = TraceEventType.AGENT_RUN,
        success: Optional[bool] = True,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """
        Record one trace event without affecting agent business logic.

        Tracing failures are swallowed by the Tracer itself.
        """
        if self.tracer is None:
            return
        self.tracer.record(
            event_type,
            name,
            component=component or self.__class__.__name__,
            success=success,
            duration_ms=duration_ms,
            error=error,
            **metadata,
        )

    def _trace_span_start(
        self, name: str, *, component: Optional[str] = None, **metadata: Any
    ) -> str:
        """Begin a timed span; returns "" when tracing is unavailable."""
        if self.tracer is None:
            return ""
        meta = dict(metadata)
        meta["component"] = component or self.__class__.__name__
        return self.tracer.start_span(name, meta)

    def _trace_span_end(
        self,
        span_id: str,
        *,
        success: bool = True,
        error: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Close a timed span started by ``_trace_span_start``."""
        if self.tracer is None or not span_id:
            return
        meta = dict(metadata)
        meta["success"] = success
        if error:
            meta["error"] = error
        self.tracer.end_span(span_id, meta)

    def _timed_trace(
        self,
        name: str,
        *,
        component: Optional[str] = None,
        event_type: TraceEventType = TraceEventType.AGENT_RUN,
        **metadata: Any,
    ):
        """
        Context manager that records started/finished style durations.

        Yields a mutable dict the caller may update with result metadata
        before exit. Never raises into the agent.
        """
        agent = self

        class _Timer:
            def __init__(self) -> None:
                self.meta: Dict[str, Any] = dict(metadata)
                self.success: bool = True
                self.error: Optional[str] = None
                self._started = time.perf_counter()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, _tb) -> bool:
                duration_ms = (time.perf_counter() - self._started) * 1000.0
                if exc_type is not None:
                    self.success = False
                    self.error = str(exc)
                agent._trace(
                    name,
                    component=component,
                    event_type=event_type,
                    success=self.success,
                    duration_ms=duration_ms,
                    error=self.error,
                    **self.meta,
                )
                return False

        return _Timer()

    def get_tool(self, name: str) -> Optional[Callable[..., Any]]:
        """
        Look up a registered tool handler by name.

        Args:
            name: Qualified tool name (e.g. ``filesystem.read_file``).

        Returns:
            The registered callable, or None when the registry is absent
            or the name is not registered.
        """
        if self.tool_registry is None:
            return None
        return self.tool_registry.get_tool(name)

    def _bound_tool_owner(self, name: str) -> Optional[Any]:
        """
        Return the instance a bound registry tool belongs to.

        Args:
            name: Qualified tool name whose bound method should be
                inspected.

        Returns:
            The tool instance (``handler.__self__``), or None when the
            tool is missing or not a bound method.
        """
        handler = self.get_tool(name)
        if handler is None:
            return None
        return getattr(handler, "__self__", None)

    def _filesystem_tools(
        self,
        workspace_root: str,
        config: Optional["Config"] = None,
    ) -> "FilesystemTools":
        """
        Resolve FilesystemTools through the ToolRegistry.

        When the registry holds a FilesystemTools whose workspace matches
        ``workspace_root``, that shared instance is returned. When the
        agent is operating on a different repository (for example a
        temporary clone), or when no registry is injected, a
        request-scoped FilesystemTools is built for that root so
        sandboxing stays correct.

        Args:
            workspace_root: Repository the tools should be scoped to.
            config: Optional Config forwarded to a request-scoped
                fallback instance.

        Returns:
            A FilesystemTools instance rooted at ``workspace_root``.
        """
        from ..tools.filesystem_tools import FilesystemTools

        root = os.path.abspath(os.path.expanduser(workspace_root or "."))
        owner = self._bound_tool_owner("filesystem.read_file")
        if isinstance(owner, FilesystemTools):
            try:
                registered_root = Path(owner.workspace_root).expanduser().resolve()
                if registered_root == Path(root).resolve():
                    logger.debug(
                        "%s using registered FilesystemTools for %s",
                        getattr(self, "agent_type", type(self).__name__),
                        root,
                    )
                    return owner
            except (OSError, RuntimeError):
                pass

        logger.debug(
            "%s building request-scoped FilesystemTools for %s",
            getattr(self, "agent_type", type(self).__name__),
            root,
        )
        return FilesystemTools(workspace_root=root, config=config)

    def _github_tools(self) -> Optional["GitHubTools"]:
        """
        Resolve GitHubTools through the ToolRegistry.

        Returns:
            The shared GitHubTools instance when registered, otherwise
            None. Agents that need GitHub access without a registry
            should obtain one from the Supervisor rather than
            constructing it themselves.
        """
        from ..tools.github_tools import GitHubTools

        owner = self._bound_tool_owner("github.validate_repository")
        if isinstance(owner, GitHubTools):
            return owner
        return None

    @abstractmethod
    def handle(self, request: AgentRequest) -> AgentResponse:
        """
        Handle an incoming AgentRequest and produce an AgentResponse.

        Args:
            request: The request to process.

        Returns:
            An AgentResponse describing the outcome.

        TODO: Implement in each concrete subclass with real logic.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, repo_path: str) -> Dict[str, str]:
        """
        Simple entry point used by the Supervisor for direct routing.

        Args:
            repo_path: Path to the repository the agent operates on.

        Returns:
            A dict with "status" and "message" keys. Currently every
            concrete agent returns fake/placeholder data here.

        TODO: Replace fake data with real agent logic (analysis,
        doc generation, test generation) once implemented.
        """
        raise NotImplementedError

    def gather_context(self, query: str) -> list:
        """
        Convenience method to gather relevant context via the retriever.

        Args:
            query: Query string to search for relevant context.

        Returns:
            A list of retrieved context chunks (placeholder empty list).

        TODO: Implement real context gathering using self.retriever.
        """
        # TODO: implement real context gathering
        return []
