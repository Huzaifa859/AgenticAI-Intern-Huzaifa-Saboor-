"""
model_client.py
================

Defines LLMClient, the abstraction all agents/tools go through to
make model calls, decoupling the rest of the system from any specific
provider.

Planned provider fallback chain (not implemented yet):

    Claude -> OpenRouter -> Llama -> Ollama

For now, LLMClient makes NO real API calls of any kind. generate()
always returns a placeholder string, and switch_model() just updates
which model name future calls would (eventually) use.

TODO: Implement real provider integration, request routing across the
Claude -> OpenRouter -> Llama -> Ollama chain, retries/fallback when a
provider is unavailable, and streaming support.
"""

from __future__ import annotations


class LLMClient:
    """
    Abstraction over an LLM provider's completion API.

    Intended to be the single choke point through which all agents
    and tools request model generations, so providers can be swapped
    (or chained/fallback-ordered) without touching calling code.
    """

    def __init__(self, model_name: str = "placeholder-model", max_tokens: int = 4096) -> None:
        """
        Initialize the LLMClient.

        Args:
            model_name: Identifier of the model currently in use.
            max_tokens: Default maximum tokens per generation.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The input prompt/text to send to the model.

        Returns:
            A placeholder response string. No model call is made yet.

        TODO: Route this call to the active provider (Claude,
        OpenRouter, Llama, or Ollama depending on self.model_name),
        handle errors/retries, and return the real generated text.
        """
        # TODO: implement real model call
        return "Placeholder response"

    def switch_model(self, model_name: str) -> None:
        """
        Switch which underlying model future generate() calls will use.

        Args:
            model_name: Identifier of the model to switch to (e.g. a
                Claude model, an OpenRouter model slug, a local Llama/
                Ollama model name).

        TODO: Validate model_name against supported providers/models,
        and actually reconfigure the underlying client/credentials
        used by generate().
        """
        # TODO: implement real model switching logic
        self.model_name = model_name
