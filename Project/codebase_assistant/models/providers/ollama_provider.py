"""
ollama_provider.py
===================

Placeholder provider for models served by a local Ollama instance —
llama3 in particular, which the proposal assigns to documentation
generation.

TODO: Implement real Ollama API calls, model-pulled/running checks,
and clear errors when the Ollama service is not up.
"""

from __future__ import annotations

from typing import List

from ...schemas.schemas import ModelMessage, ModelResponse
from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """
    Calls models served by a local Ollama instance.

    Used for the higher-volume, lower-stakes path (documentation
    generation), where a local model is good enough and costs nothing
    per call.
    """

    name: str = "ollama"

    def __init__(
        self,
        model: str = "llama3",
        max_tokens: int = 4096,
        base_url: str = "http://localhost:11434",
    ) -> None:
        """
        Initialize the Ollama provider.

        Args:
            model: Name of the local Ollama model to call.
            max_tokens: Default maximum tokens per generation.
            base_url: Base URL of the local Ollama service.
        """
        super().__init__(model=model, max_tokens=max_tokens)
        self.base_url = base_url

    def generate(self, messages: List[ModelMessage], **kwargs) -> ModelResponse:
        """
        Generate a completion via local Ollama.

        Args:
            messages: Conversation history to send.
            **kwargs: Generation options (temperature, etc).

        Returns:
            A placeholder ModelResponse. No API call is made yet.

        TODO: Implement the real request and error handling.
        """
        # TODO: implement real Ollama API call
        return ModelResponse()

    def is_available(self) -> bool:
        """
        Report whether the local Ollama service is running.

        Returns:
            False (placeholder — no reachability check is performed).

        TODO: Implement a real check that Ollama is up and the model
        has been pulled.
        """
        # TODO: implement real availability check
        return False
