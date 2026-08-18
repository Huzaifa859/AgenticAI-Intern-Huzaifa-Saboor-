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

import ast
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

#: How long one agent's index update is trusted by another agent
#: sharing the same `index_reuse_cache`, before a fresh
#: `Indexer.update_index` walk-and-hash pass is required again. Sized
#: to comfortably cover one `--agent all` pipeline run (code analysis
#: -> documentation -> testing against the same workspace), not to
#: mask genuine staleness across separate runs.
_INDEX_REUSE_WINDOW_SECONDS = 30.0

from ..hooks.events import HookEvent
from ..memory.memory_store import MemoryStore
from ..models.model_client import LLMClient
from ..rag.retriever import Retriever
from ..schemas.schemas import AgentRequest, AgentResponse, AgentType
from ..tools.registry import ToolRegistry
from ..tracing.events import TraceEventType
from ..tracing.tracer import Tracer

if TYPE_CHECKING:
    from ..cache.output_cache import OutputCache
    from ..config import Config
    from ..hooks.manager import HookManager
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
        hook_manager: Optional["HookManager"] = None,
        index_reuse_cache: Optional[Dict[str, float]] = None,
        ast_cache: Optional[Dict[Tuple[str, float], ast.Module]] = None,
    ) -> None:
        """
        Initialize the BaseAgent with its shared dependencies.

        Args:
            model_client: LLMClient used to make model calls.
            tool_registry: Registry used to invoke tools.
            retriever: RAG retriever used to fetch relevant context.
            memory_store: Long-term memory store.
            tracer: Optional shared Tracer for lifecycle events.
            hook_manager: Optional HookManager for lifecycle hooks.
            index_reuse_cache: Optional dict shared across the agents
                built by one Supervisor, mapping an absolute workspace
                path to the monotonic time it was last indexed. When
                given, `_recently_indexed`/`_mark_indexed` let one
                agent's `Indexer.update_index` call cover the others
                for `_INDEX_REUSE_WINDOW_SECONDS`, instead of code
                analysis, documentation, and testing each re-walking
                and re-hashing the same repository moments apart in a
                `--agent all` run. Omitted (None) by default, which
                disables the skip entirely and preserves the previous
                per-agent behavior.
            ast_cache: Optional dict shared across the agents built by
                one Supervisor, mapping `(absolute_file_path, mtime)` to
                its parsed `ast.Module`. When given, `_parse_module_ast`
                lets Documentation and Testing reuse each other's parse
                of the same file version instead of each running
                `ast.parse` on it independently during their AST
                inventory scans. Omitted (None) by default, which
                disables sharing and parses directly, as before.
        """
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.retriever = retriever
        self.memory_store = memory_store
        self.tracer = tracer
        self.hook_manager = hook_manager
        self.index_reuse_cache = index_reuse_cache
        self.ast_cache = ast_cache

    def _recently_indexed(self, workspace: str) -> bool:
        """
        True if some agent sharing `index_reuse_cache` indexed this
        workspace within `_INDEX_REUSE_WINDOW_SECONDS`.

        Args:
            workspace: Repository root the caller is about to index.

        Returns:
            True when a fresh `Indexer.update_index` call can safely be
            skipped because another agent already did the equivalent
            work moments ago in this same process.
        """
        if self.index_reuse_cache is None:
            return False
        checked_at = self.index_reuse_cache.get(self._index_cache_key(workspace))
        if checked_at is None:
            return False
        return (time.monotonic() - checked_at) < _INDEX_REUSE_WINDOW_SECONDS

    def _mark_indexed(self, workspace: str) -> None:
        """
        Record that `workspace` was just indexed.

        No-op when no `index_reuse_cache` was injected, so agents built
        without one behave exactly as before.

        Args:
            workspace: Repository root that was just indexed.
        """
        if self.index_reuse_cache is None:
            return
        self.index_reuse_cache[self._index_cache_key(workspace)] = time.monotonic()

    @staticmethod
    def _index_cache_key(workspace: str) -> str:
        """Normalize a workspace path so different callers agree on its key."""
        return os.path.abspath(os.path.expanduser(workspace or "."))

    def _parse_module_ast(
        self,
        filesystem: "FilesystemTools",
        module_path: str,
        source: str,
    ) -> ast.Module:
        """
        Parse a module's source into an AST, reusing a shared parse when
        possible.

        When `ast_cache` was injected (e.g. by the Supervisor, shared
        between DocumentationAgent and TestingAgent), the parsed tree
        for a given `(absolute_path, mtime)` is cached so the second
        agent to scan a file in one pipeline run reuses the first
        agent's parse instead of running `ast.parse` on it again. Safe
        because both agents only read from the tree during their AST
        inventory scans (iterating `tree.body`, slicing by line
        numbers, reading docstrings) and never mutate it.

        Args:
            filesystem: FilesystemTools used to resolve `module_path` to
                an absolute path for the cache key.
            module_path: Workspace-relative path; used as `ast.parse`'s
                `filename` for readable error messages.
            source: The file's already-read source text.

        Returns:
            The parsed module.

        Raises:
            SyntaxError: If `source` is not valid Python -- same as
                calling `ast.parse` directly; existing callers already
                handle this.
        """
        if self.ast_cache is None:
            return ast.parse(source or "", filename=module_path)

        try:
            abs_path = str(filesystem.resolve_path(module_path))
            mtime = os.path.getmtime(abs_path)
        except (OSError, ValueError):
            return ast.parse(source or "", filename=module_path)

        key = (abs_path, mtime)
        cached = self.ast_cache.get(key)
        if cached is not None:
            return cached

        tree = ast.parse(source or "", filename=module_path)
        self.ast_cache[key] = tree
        return tree

    def _build_output_cache(
        self, workspace: str, namespace: str
    ) -> Optional["OutputCache"]:
        """
        Build the persistent LLM-output cache for one repository, if
        enabled.

        Args:
            workspace: Repository root the cache is scoped to.
            namespace: Call-site name (e.g. "documentation", "testing",
                "analysis").

        Returns:
            An OutputCache, or None when caching is disabled
            (`Config.output_cache_enabled` is False) or no Config is
            reachable from `self.model_client` (e.g. a mock client in
            tests), in which case callers should behave exactly as if
            caching did not exist.
        """
        from ..cache.output_cache import OutputCache
        from ..config import Config

        cfg = getattr(self.model_client, "config", None)
        if not isinstance(cfg, Config) or not cfg.output_cache_enabled:
            return None
        try:
            return OutputCache.for_repository(cfg, workspace, namespace)
        except Exception as exc:
            logger.warning("Could not open output cache for %s: %s", namespace, exc)
            return None

    def _hook(self, event: HookEvent, **context: Any) -> None:
        """Fire a lifecycle hook; never raises into agent logic."""
        if self.hook_manager is None:
            return
        payload = dict(context)
        payload.setdefault("component", self.__class__.__name__)
        payload.setdefault(
            "agent_type",
            getattr(self.agent_type, "value", str(self.agent_type)),
        )
        try:
            self.hook_manager.trigger(event, payload)
        except Exception as exc:
            logger.warning(
                "%s hook %s failed: %s",
                self.__class__.__name__,
                getattr(event, "value", event),
                exc,
            )

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
