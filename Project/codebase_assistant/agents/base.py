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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from ..memory.memory_store import MemoryStore
from ..models.model_client import LLMClient
from ..rag.retriever import Retriever
from ..schemas.schemas import AgentRequest, AgentResponse, AgentType
from ..tools.registry import ToolRegistry

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
    ) -> None:
        """
        Initialize the BaseAgent with its shared dependencies.

        Args:
            model_client: LLMClient used to make model calls.
            tool_registry: Registry used to invoke tools.
            retriever: RAG retriever used to fetch relevant context.
            memory_store: Long-term memory store.
        """
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.retriever = retriever
        self.memory_store = memory_store

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
