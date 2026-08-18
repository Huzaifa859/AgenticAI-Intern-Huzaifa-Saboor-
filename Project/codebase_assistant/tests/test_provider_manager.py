"""
test_provider_manager.py
========================

Unit tests for OpenRouter → Ollama failover behind ProviderManager.

All providers are fakes; no network calls.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from codebase_assistant.exceptions.model_exceptions import (
    ModelResponseError,
    ProviderUnavailableError,
    RateLimitError,
)
from codebase_assistant.models.model_client import LLMClient
from codebase_assistant.models.providers.base import BaseProvider
from codebase_assistant.models.providers.provider_manager import ProviderManager
from codebase_assistant.schemas.schemas import ModelMessage, ModelResponse
from codebase_assistant.tracing.tracer import Tracer

MESSAGES = [ModelMessage(role="user", content="hello")]


class FakeProvider(BaseProvider):
    """Configurable BaseProvider for failover tests."""

    name = "fake"

    def __init__(
        self,
        *,
        label: str,
        available: bool = True,
        content: str = "ok",
        error: Optional[BaseException] = None,
        model: str = "fake-model",
    ) -> None:
        super().__init__(model=model, max_tokens=128)
        self.label = label
        self._available = available
        self._content = content
        self._error = error
        self.generate_calls = 0
        self.availability_probes = 0

    def is_available(self) -> bool:
        self.availability_probes += 1
        return self._available

    def generate(self, messages: List[ModelMessage], **kwargs: Any) -> ModelResponse:
        self.generate_calls += 1
        if self._error is not None:
            raise self._error
        return ModelResponse(
            content=self._content,
            raw={"provider": self.label, "model": self.model},
        )


def _manager(
    preferred: Optional[FakeProvider] = None,
    fallback: Optional[FakeProvider] = None,
    **kwargs: Any,
) -> ProviderManager:
    return ProviderManager(
        preferred=preferred,
        fallback=fallback,
        cache_seconds=kwargs.pop("cache_seconds", 60),
        tracer=kwargs.pop("tracer", None),
        **kwargs,
    )


def test_uses_openrouter_when_available() -> None:
    preferred = FakeProvider(label="openrouter", content="from-openrouter")
    fallback = FakeProvider(label="ollama", content="from-ollama")
    manager = _manager(preferred, fallback)

    response = manager.generate(MESSAGES)

    assert response.content == "from-openrouter"
    assert preferred.generate_calls == 1
    assert fallback.generate_calls == 0
    assert "OpenRouter" in manager.status_message()


def test_falls_back_when_openrouter_unavailable_at_selection() -> None:
    preferred = FakeProvider(label="openrouter", available=False)
    fallback = FakeProvider(label="ollama", content="from-ollama")
    manager = _manager(preferred, fallback)

    response = manager.generate(MESSAGES)

    assert response.content == "from-ollama"
    assert preferred.generate_calls == 0
    assert fallback.generate_calls == 1
    assert "Ollama fallback" in manager.status_message()


def test_falls_back_on_openrouter_timeout_during_generate() -> None:
    preferred = FakeProvider(
        label="openrouter",
        available=True,
        error=ProviderUnavailableError("timeout talking to OpenRouter"),
    )
    fallback = FakeProvider(label="ollama", content="from-ollama")
    manager = _manager(preferred, fallback)

    response = manager.generate(MESSAGES)

    assert response.content == "from-ollama"
    assert preferred.generate_calls == 1
    assert fallback.generate_calls == 1
    assert manager.preferred_is_available() is False


def test_falls_back_on_auth_failure() -> None:
    preferred = FakeProvider(
        label="openrouter",
        available=True,
        error=ProviderUnavailableError("invalid API key"),
    )
    fallback = FakeProvider(label="ollama", content="from-ollama")
    manager = _manager(preferred, fallback)

    assert manager.generate(MESSAGES).content == "from-ollama"


def test_falls_back_on_rate_limit() -> None:
    preferred = FakeProvider(
        label="openrouter",
        available=True,
        error=RateLimitError("rate limited"),
    )
    fallback = FakeProvider(label="ollama", content="from-ollama")
    manager = _manager(preferred, fallback)

    assert manager.generate(MESSAGES).content == "from-ollama"


def test_does_not_failover_on_model_response_error() -> None:
    preferred = FakeProvider(
        label="openrouter",
        available=True,
        error=ModelResponseError("bad JSON"),
    )
    fallback = FakeProvider(label="ollama", content="from-ollama")
    manager = _manager(preferred, fallback)

    with pytest.raises(ModelResponseError, match="bad JSON"):
        manager.generate(MESSAGES)

    assert preferred.generate_calls == 1
    assert fallback.generate_calls == 0


def test_both_providers_fail_raises() -> None:
    preferred = FakeProvider(label="openrouter", available=False)
    fallback = FakeProvider(label="ollama", available=False)
    manager = _manager(preferred, fallback)

    with pytest.raises(ProviderUnavailableError, match="No LLM provider"):
        manager.generate(MESSAGES)

    assert "static-only" in manager.status_message()


def test_availability_cache_avoids_repeated_probes() -> None:
    preferred = FakeProvider(label="openrouter", available=True)
    fallback = FakeProvider(label="ollama")
    manager = _manager(preferred, fallback, cache_seconds=60)

    assert manager.preferred_is_available() is True
    assert manager.preferred_is_available() is True
    assert preferred.availability_probes == 1


def test_cache_expiry_reprobes_preferred() -> None:
    preferred = FakeProvider(label="openrouter", available=True)
    manager = _manager(preferred, FakeProvider(label="ollama"), cache_seconds=1)

    assert manager.preferred_is_available() is True
    manager._preferred_checked_at = time.monotonic() - 2.0
    preferred._available = False

    assert manager.preferred_is_available() is False
    assert preferred.availability_probes == 2


def test_after_failover_uses_cache_then_retries_preferred() -> None:
    preferred = FakeProvider(
        label="openrouter",
        available=True,
        error=ProviderUnavailableError("network down"),
    )
    fallback = FakeProvider(label="ollama", content="from-ollama")
    manager = _manager(preferred, fallback, cache_seconds=60)

    assert manager.generate(MESSAGES).content == "from-ollama"
    # Cached unavailable: skip preferred generate entirely.
    preferred._error = None
    preferred._content = "recovered"
    assert manager.generate(MESSAGES).content == "from-ollama"
    assert preferred.generate_calls == 1

    # Expire cache so preferred is probed and used again.
    manager._preferred_checked_at = time.monotonic() - 120.0
    assert manager.generate(MESSAGES).content == "recovered"
    assert preferred.generate_calls == 2


def test_llm_client_uses_provider_manager_transparently() -> None:
    preferred = FakeProvider(label="openrouter", available=False)
    fallback = FakeProvider(label="ollama", content="via-manager")
    manager = _manager(preferred, fallback)
    client = LLMClient(provider=manager)

    response = client.generate(MESSAGES)

    assert response.content == "via-manager"
    assert client.is_available() is True


def test_traces_provider_selected_and_fallback() -> None:
    tracer = Tracer(run_id="provider-test")
    preferred = FakeProvider(
        label="openrouter",
        error=ProviderUnavailableError("boom"),
    )
    fallback = FakeProvider(label="ollama", content="ok")
    manager = _manager(preferred, fallback, tracer=tracer)

    manager.generate(MESSAGES)

    names = [event.name for event in tracer.get_events()]
    assert "provider_failed" in names
    assert "provider_fallback" in names
    assert "provider_selected" in names
