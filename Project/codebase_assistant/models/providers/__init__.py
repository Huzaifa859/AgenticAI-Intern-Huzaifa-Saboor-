"""
providers
=========

Concrete LLM provider backends sitting behind LLMClient.

LLMClient stays the single choke point every agent calls; the classes
here are what it eventually routes to, so a provider can be swapped
without touching calling code.

The proposal's deliberate cost/quality split maps onto two providers:

- OpenRouterProvider — Claude, for code analysis and bug-finding
  (correctness-critical, where false positives are costly).
- OllamaProvider — local llama3, for documentation generation
  (higher-volume, lower-stakes, no marginal cost).

NOTE: Placeholder only. No provider makes a real API call, and
LLMClient does not route to them yet.
"""

from .base import BaseProvider
from .ollama_provider import OllamaProvider
from .openrouter_provider import OpenRouterProvider

__all__ = ["BaseProvider", "OpenRouterProvider", "OllamaProvider"]
