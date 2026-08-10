"""
base.py
=======

Defines BaseProvider, the abstract interface every LLM provider
backend implements.

TODO: Once concrete providers are implemented, have LLMClient select
among them (and fall back between them) rather than holding a single
model name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ...schemas.schemas import ModelMessage, ModelResponse


class BaseProvider(ABC):
    """
    Abstract base class for LLM provider backends.

    Concrete providers wrap one vendor/transport (OpenRouter, Ollama,
    ...) and expose a uniform generation interface so LLMClient can
    route between them.
    """

    name: str = ""

    def __init__(self, model: str = "", max_tokens: int = 4096) -> None:
        """
        Initialize the provider.

        Args:
            model: Model identifier this provider should call.
            max_tokens: Default maximum tokens per generation.
        """
        self.model = model
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, messages: List[ModelMessage], **kwargs) -> ModelResponse:
        """
        Generate a completion for a conversation.

        Args:
            messages: Conversation history to send to the model.
            **kwargs: Provider-specific generation options.

        Returns:
            A ModelResponse carrying the generated content.

        TODO: Implement in each concrete subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """
        Report whether this provider can currently serve requests.

        Returns:
            True if the provider is reachable and configured.

        TODO: Implement in each concrete subclass. Used to drive
        fallback when a provider is unreachable.
        """
        raise NotImplementedError
