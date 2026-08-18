"""
test_models.py
===============

Lightweight ModelClient / LLMClient checks. Provider failover coverage
lives in ``test_provider_manager.py``.
"""

from __future__ import annotations

import pytest

from codebase_assistant.exceptions.model_exceptions import ProviderUnavailableError
from codebase_assistant.models.model_client import LLMClient, ModelClient
from codebase_assistant.models.providers.base import BaseProvider
from codebase_assistant.schemas.schemas import ModelMessage, ModelResponse


class _StubProvider(BaseProvider):
    name = "stub"

    def __init__(self) -> None:
        super().__init__(model="stub-model", max_tokens=64)

    def is_available(self) -> bool:
        return True

    def generate(self, messages, **kwargs) -> ModelResponse:
        return ModelResponse(content="stub-ok", raw={"model": self.model})


def test_model_client_requires_provider() -> None:
    client = ModelClient(provider=None)
    with pytest.raises(ProviderUnavailableError, match="No provider configured"):
        client.generate([ModelMessage(role="user", content="hi")])


def test_model_client_delegates_to_provider() -> None:
    client = ModelClient(provider=_StubProvider())
    response = client.generate([ModelMessage(role="user", content="hi")])
    assert response.content == "stub-ok"
    assert client.is_available() is True


def test_llm_client_keeps_legacy_fields() -> None:
    client = LLMClient(model_name="x", max_tokens=10, provider=_StubProvider())
    assert client.model_name == "x"
    assert client.max_tokens == 10
