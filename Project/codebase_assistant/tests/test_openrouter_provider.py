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
from codebase_assistant.models.providers.openrouter_provider import (
    _FALLBACK_MODELS,
    OpenRouterProvider,
)
from codebase_assistant.schemas.schemas import ModelMessage, ModelResponse

MESSAGES = [ModelMessage(role="user", content="hello")]

FALLBACK_CHAIN = _FALLBACK_MODELS

CLAUDE, LLAMA, GEMMA, NEMOTRON = FALLBACK_CHAIN


def _models_called(mock_post: MagicMock) -> List[str]:
    """Model slug sent on each POST, in call order."""
    return [call.kwargs["json"]["model"] for call in mock_post.call_args_list]


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
    """HTTP 429 should back off per model, then exhaust the fallback chain."""
    mock_post.return_value = _http_response(429, {"error": {"message": "rate limited"}})
    provider = _provider()

    with pytest.raises(RateLimitError):
        provider.generate(MESSAGES)

    # 4 backoff attempts on each of the 4 models in the chain.
    assert mock_post.call_count == 4 * len(FALLBACK_CHAIN)
    assert mock_sleep.call_count == 3 * len(FALLBACK_CHAIN)


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

    assert mock_post.call_count == 4 * len(FALLBACK_CHAIN)
    assert mock_sleep.call_count == 3 * len(FALLBACK_CHAIN)


@pytest.mark.parametrize(
    ("status_code", "exc_type"),
    [
        (400, ModelResponseError),
        (401, ProviderUnavailableError),
        (403, ProviderUnavailableError),
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
    """Malformed requests and auth failures must fail on the first model."""
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


@pytest.mark.parametrize("status_code", [402, 404])
@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_falls_back_to_llama_on_402_and_404(
    mock_post: MagicMock, mock_sleep: MagicMock, status_code: int
) -> None:
    """Insufficient credits or a missing model should switch to Llama."""
    mock_post.side_effect = [
        _http_response(status_code, {"error": {"message": f"http {status_code}"}}),
        _http_response(200, _success_payload(content="Answer from Llama.")),
    ]
    provider = _provider()

    result = provider.generate(MESSAGES)

    assert result.content == "Answer from Llama."
    assert result.raw["model_used"] == LLAMA
    assert _models_called(mock_post) == [CLAUDE, LLAMA]
    # Credits and unknown models never recover, so no backoff is spent.
    mock_sleep.assert_not_called()


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_falls_back_to_gemma_when_claude_rate_limited(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """A rate-limited Claude and an unusable Llama should reach Gemma."""
    mock_post.side_effect = (
        [_http_response(429, {"error": {"message": "rate limited"}})] * 4
        + [_http_response(402, {"error": {"message": "no credits"}})]
        + [_http_response(200, _success_payload(content="Answer from Gemma."))]
    )
    provider = _provider()

    result = provider.generate(MESSAGES)

    assert result.content == "Answer from Gemma."
    assert result.raw["model_used"] == GEMMA
    assert _models_called(mock_post) == [CLAUDE] * 4 + [LLAMA, GEMMA]
    assert mock_sleep.call_count == 3


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_falls_back_through_whole_chain_to_nemotron(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """The last model in the chain should still be tried."""
    mock_post.side_effect = [
        _http_response(402, {"error": {"message": "no credits"}}),
        _http_response(404, {"error": {"message": "unknown model"}}),
        _http_response(402, {"error": {"message": "no credits"}}),
        _http_response(200, _success_payload(content="Answer from Nemotron.")),
    ]
    provider = _provider()

    result = provider.generate(MESSAGES)

    assert result.content == "Answer from Nemotron."
    assert result.raw["model_used"] == NEMOTRON
    assert _models_called(mock_post) == list(FALLBACK_CHAIN)


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_all_models_fail_raises_for_graceful_degradation(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """Exhausting the chain should raise so callers degrade to static-only."""
    mock_post.return_value = _http_response(402, {"error": {"message": "no credits"}})
    provider = _provider()

    with pytest.raises(ProviderUnavailableError, match="insufficient credits"):
        provider.generate(MESSAGES)

    assert _models_called(mock_post) == list(FALLBACK_CHAIN)


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_no_fallback_on_authentication_or_malformed_request(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """Auth and 400 errors must not burn through the fallback chain."""
    provider = _provider()

    mock_post.return_value = _http_response(401, {"error": {"message": "bad key"}})
    with pytest.raises(ProviderUnavailableError):
        provider.generate(MESSAGES)
    assert mock_post.call_count == 1

    mock_post.reset_mock()
    mock_post.return_value = _http_response(400, {"error": {"message": "malformed"}})
    with pytest.raises(ModelResponseError):
        provider.generate(MESSAGES)
    assert mock_post.call_count == 1


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_fallback_preserves_prompt_and_generation_options(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """The fallback request must be identical except for the model slug."""
    mock_post.side_effect = [
        _http_response(402, {"error": {"message": "no credits"}}),
        _http_response(200, _success_payload(content="Answer from Llama.")),
    ]
    messages = [
        ModelMessage(role="system", content="Stay grounded in the context."),
        ModelMessage(role="user", content="Document the add function."),
    ]
    provider = _provider()

    provider.generate(messages, max_tokens=1234, temperature=0.1)

    first, second = (call.kwargs["json"] for call in mock_post.call_args_list)
    assert first["messages"] == second["messages"]
    assert second["max_tokens"] == 1234
    assert second["temperature"] == 0.1
    assert first["model"] != second["model"]


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_model_used_reported_without_any_fallback(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """A first-attempt success should still report the model in raw."""
    mock_post.return_value = _http_response(200, _success_payload(content="ok"))
    provider = _provider()

    result = provider.generate(MESSAGES)

    assert result.raw["model_used"] == CLAUDE
    mock_post.assert_called_once()


@patch("codebase_assistant.models.providers.openrouter_provider.time.sleep")
@patch("codebase_assistant.models.providers.openrouter_provider.requests.post")
def test_custom_primary_model_is_tried_before_the_chain(
    mock_post: MagicMock, mock_sleep: MagicMock
) -> None:
    """A configured non-default model leads the chain without duplication."""
    mock_post.side_effect = [
        _http_response(404, {"error": {"message": "unknown model"}}),
        _http_response(200, _success_payload(content="ok")),
    ]
    provider = _provider(model="openai/gpt-4o-mini")

    result = provider.generate(MESSAGES)

    assert result.raw["model_used"] == CLAUDE
    assert _models_called(mock_post) == ["openai/gpt-4o-mini", CLAUDE]


def test_generate_without_api_key_raises() -> None:
    """generate() without a key should raise ProviderUnavailableError."""
    provider = _provider(api_key="")
    with pytest.raises(ProviderUnavailableError):
        provider.generate(MESSAGES)
