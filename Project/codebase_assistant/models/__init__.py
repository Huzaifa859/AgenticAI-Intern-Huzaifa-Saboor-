"""
models
======

Contains the LLMClient abstraction used to interact with an
underlying LLM provider. Decoupled from any specific provider SDK so
it can be swapped out later.

Planned provider chain: Claude -> OpenRouter -> Llama -> Ollama.
"""

from .model_client import LLMClient

__all__ = ["LLMClient"]
