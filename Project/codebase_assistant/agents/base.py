"""
base.py
=======

Defines BaseAgent, the abstract base class that all specialized agents
(Code Analysis, Documentation, Testing) inherit from.

TODO: Flesh out shared agent behavior (e.g. common prompt construction,
tool-use loop, retry/error handling) once real functionality is added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from ..memory.memory_store import MemoryStore
from ..models.model_client import LLMClient
from ..rag.retriever import Retriever
from ..schemas.schemas import AgentRequest, AgentResponse, AgentType
from ..tools.registry import ToolRegistry


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
