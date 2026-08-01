"""
test_openrouter_provider.py
=============================

Unit tests for OpenRouterProvider.

All HTTP traffic is mocked. Tests never call the real OpenRouter API
and never require OPENROUTER_API_KEY.
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

from codebase_assistant.config import Config
from codebase_assistant.exceptions.model_exceptions import (
    ModelResponseError,
    ProviderUnavailableError,
    RateLimitError,
)
from codebase_assistant.models.providers.openrouter_provider import OpenRouterProvider
from codebase_assistant.schemas.schemas import ModelMessage, ModelResponse

MESSAGES = [ModelMessage(role="user", content="hello")]


def _config() -> Config:
    """Config without credentials so tests stay isolated from local .env."""
    return Config(openrouter_api_key=None)


def _provider(**kwargs: Any) -> OpenRouterProvider:
    """Build a provider with an explicit test key unless overridden."""
    defaults = dict(
        api_key="sk-test-key",
        model="anthropic/claude-sonnet-4",
        max_tokens=256,
        timeout=5.0,
        config=_config(),
    )
    defaults.update(kwargs)
    return OpenRouterProvider(**defaults)


def _http_response(
    status_code: int,
    payload: Optional[dict] = None,
    text: str = "",
) -> MagicMock:
    """Build a mock requests.Response."""
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.text = text

    if payload is None:
        response.json.side_effect = ValueError("no json")
    else:
        response.json.return_value = payload
    return response


def _success_payload(
    content: str = "Hello from the model.",
    usage: Optional[dict] = None,
) -> dict:
    """OpenRouter-shaped success body."""
    body: dict = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def test_is_available_false_without_api_key() -> None:
    """Missing API key should make is_available() return False immediately."""
    provider = _provider(api_key="")
    assert provider.is_available() is False


@patch("codebase_assistant.models.providers.openrouter_provider.requests.get")
def test_is_available_true_when_probe_succeeds(mock_get: MagicMock) -> None:
    """A 200 from /models with a key should report available."""
    mock_get.return_value = _http_response(200, {"data": []})
    provider = _provider()
    assert provider.is_available() is True
    mock_get.assert_called_once()


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_successful_chat_completion(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """A normal chat completion should return a populated ModelResponse."""
    mock_post.return_value = _http_response(
        200,
        _success_payload(
            content="Grounded answer.",
            usage={"prompt_tokens": 10, "completion_tokens": 4},
        ),
    )
    provider = _provider()

    result = provider.generate(MESSAGES)

    assert isinstance(result, ModelResponse)
    assert result.content == "Grounded answer."
    assert result.usage["prompt_tokens"] == 10
    assert result.raw["choices"][0]["message"]["content"] == "Grounded answer."
    mock_post.assert_called_once()
    mock_sleep.assert_not_called()


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_malformed_response_raises_model_response_error(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """Missing choices/message.content should raise ModelResponseError."""
    mock_post.return_value = _http_response(200, {"choices": []})
    provider = _provider()

    with pytest.raises(ModelResponseError):
        provider.generate(MESSAGES)

    mock_post.assert_called_once()
    mock_sleep.assert_not_called()


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_retries_on_http_429(mock_post: MagicMock, mock_sleep: MagicMock) -> None:
    """HTTP 429 should retry and eventually raise RateLimitError."""
    mock_post.return_value = _http_response(429, {"error": {"message": "rate limited"}})
    provider = _provider()

    with pytest.raises(RateLimitError):
        provider.generate(MESSAGES)

    assert mock_post.call_count == 4
    assert mock_sleep.call_count == 3


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_retries_on_http_5xx(
    mock_post: MagicMock,
    mock_sleep: MagicMock,
    status_code: int,
) -> None:
    """Retryable 5xx responses should back off then raise ProviderUnavailableError."""
    mock_post.return_value = _http_response(status_code, {"error": {"message": "server"}})
    provider = _provider()

    with pytest.raises(ProviderUnavailableError):
        provider.generate(MESSAGES)

    assert mock_post.call_count == 4
    assert mock_sleep.call_count == 3


@pytest.mark.parametrize(
    ("status_code", "exc_type"),
    [
        (400, ModelResponseError),
        (401, ProviderUnavailableError),
        (403, ProviderUnavailableError),
        (404, ModelResponseError),
    ],
)
@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_no_retries_on_client_errors(
    mock_post: MagicMock,
    mock_sleep: MagicMock,
    status_code: int,
    exc_type: type,
) -> None:
    """400/401/403/404 must fail immediately without retrying."""
    mock_post.return_value = _http_response(
        status_code, {"error": {"message": f"http {status_code}"}}
    )
    provider = _provider()

    with pytest.raises(exc_type):
        provider.generate(MESSAGES)

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_timeout_handling(mock_post: MagicMock, mock_sleep: MagicMock) -> None:
    """Timeouts should retry and then raise ProviderUnavailableError."""
    mock_post.side_effect = requests.Timeout("timed out")
    provider = _provider()

    with pytest.raises(ProviderUnavailableError):
        provider.generate(MESSAGES)

    assert mock_post.call_count == 4
    assert mock_sleep.call_count == 3


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_connection_error_handling(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """Connection errors should retry and then raise ProviderUnavailableError."""
    mock_post.side_effect = requests.ConnectionError("refused")
    provider = _provider()

    with pytest.raises(ProviderUnavailableError):
        provider.generate(MESSAGES)

    assert mock_post.call_count == 4
    assert mock_sleep.call_count == 3


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_usage_parsing_when_usage_missing(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """Missing usage metadata should become an empty usage dict."""
    mock_post.return_value = _http_response(200, _success_payload(content="ok"))
    provider = _provider()

    result = provider.generate(MESSAGES)

    assert result.content == "ok"
    assert result.usage == {}
    mock_sleep.assert_not_called()


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_whitespace_only_model_output_raises(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """Whitespace-only assistant content should raise ModelResponseError."""
    mock_post.return_value = _http_response(200, _success_payload(content="   \n\t  "))
    provider = _provider()

    with pytest.raises(ModelResponseError, match="empty assistant content"):
        provider.generate(MESSAGES)

    mock_post.assert_called_once()
    mock_sleep.assert_not_called()


def test_generate_without_api_key_raises() -> None:
    """generate() without a key should raise ProviderUnavailableError."""
    provider = _provider(api_key="")
    with pytest.raises(ProviderUnavailableError):
        provider.generate(MESSAGES)
