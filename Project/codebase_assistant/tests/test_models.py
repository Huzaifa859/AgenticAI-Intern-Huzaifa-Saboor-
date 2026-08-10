"""
test_models.py
===============

Placeholder tests for the model layer — LLMClient and its providers.

Covers the proposal's two-model split (Claude via OpenRouter for
analysis, local llama3 via Ollama for documentation), provider
fallback, and the Week 7 retry-on-malformed-output path.

TODO: Replace every skip below with real assertions as each provider is
implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO: assert analysis routes to Claude and documentation to llama3")
def test_client_routes_task_to_correct_provider() -> None:
    """The cost/quality split should be enforced by routing, not by hand."""


@pytest.mark.skip(reason="TODO: assert the client falls back when a provider is unavailable")
def test_client_falls_back_when_provider_unavailable() -> None:
    """An unreachable provider should not abort the run."""


@pytest.mark.skip(reason="TODO: assert switch_model rejects unknown model identifiers")
def test_client_rejects_unknown_model() -> None:
    """Switching to an unsupported model should be refused."""


@pytest.mark.skip(reason="TODO: assert the provider returns a populated ModelResponse")
def test_openrouter_provider_returns_model_response() -> None:
    """A generation should come back as a valid ModelResponse."""


@pytest.mark.skip(reason="TODO: assert RateLimitError triggers retry with backoff")
def test_openrouter_provider_retries_on_rate_limit() -> None:
    """Rate limits should back off and retry rather than fail outright."""


@pytest.mark.skip(reason="TODO: assert is_available is False when Ollama is not running")
def test_ollama_provider_detects_unavailable_service() -> None:
    """A stopped Ollama service should be reported, not raise obscurely."""


@pytest.mark.skip(reason="TODO: assert malformed JSON raises MalformedOutputError and retries")
def test_malformed_output_is_retried() -> None:
    """Invalid structured output should be retried before surfacing."""


@pytest.mark.skip(reason="TODO: assert both models answer the same documentation prompt")
def test_multi_model_comparison_runs_both_providers() -> None:
    """Backs the Week 7 Claude vs. llama3 side-by-side comparison."""
