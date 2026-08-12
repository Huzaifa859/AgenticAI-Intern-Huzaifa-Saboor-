"""
openrouter_provider.py
=======================

OpenRouter provider for models served via the OpenRouter Chat Completions
API — Claude in particular, which the proposal assigns to code analysis
and bug-finding.

Makes real HTTPS requests with `requests`, retries transient failures
with exponential backoff, and maps transport/API failures onto the
project's model exception hierarchy.

When the configured model cannot serve a request for a model-level
reason - no credits, unknown model, rate limit, or a server-side error -
the provider walks a fallback chain of alternative models with the same
prompt, temperature, and token budget. Credential and request errors
never trigger a fallback: another model would fail identically.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from ...config import Config
from ...exceptions.model_exceptions import (
    ModelResponseError,
    ProviderUnavailableError,
    RateLimitError,
)
from ...schemas.schemas import ModelMessage, ModelResponse
from .base import BaseProvider

logger = logging.getLogger(__name__)

#: HTTP status codes that are worth retrying with backoff.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

#: Default request timeout in seconds when Config/env do not specify one.
#: Trimmed from 60s so a stalled model fails fast enough that the
#: fallback chain below has a bounded worst case.
_DEFAULT_TIMEOUT_SECONDS = 30.0

#: Maximum attempts for a single generate() call (1 initial + retries).
#: Trimmed from 4 to 2: with a 5-model chain, 4 attempts per model made
#: the worst case (all models failing) take ~20 minutes of pure waiting;
#: 2 attempts still absorbs one transient blip per model without that
#: multiplying out so badly.
_MAX_ATTEMPTS = 2

#: Initial backoff delay in seconds; doubles after each retryable failure.
_INITIAL_BACKOFF_SECONDS = 1.0

#: Models tried in order when the configured model is not usable.
#: Trimmed from 4 fallbacks to 2: each additional fallback multiplies
#: the worst-case latency by another `_MAX_ATTEMPTS * timeout`, and the
#: two kept here are the most reliable free-tier options observed.
_FALLBACK_MODELS = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-3-27b-it",
)

#: Status codes that permanently rule this model out for the request, so
#: the next model is tried immediately without backing off. Rate limits
#: and 5xx also fall back, but only after their backoff retries.
_FALLBACK_STATUS_CODES = frozenset({402, 404})

#: Model slug prefixes/exact slugs known to honour ``response_format``.
#: For all others the parameter is silently dropped before the request
#: so the model does not misinterpret it and wrap JSON in prose.
_JSON_MODE_SUPPORTED_PREFIXES = (
    "openai/",
    "anthropic/",
    "meta-llama/llama-3",
    "google/gemma-3",
    "google/gemma-4",
    "mistralai/",
    "cohere/",
)


def _supports_json_mode(model: str) -> bool:
    """Return True if *model* is known to honour ``response_format``."""
    return any(model.startswith(prefix) for prefix in _JSON_MODE_SUPPORTED_PREFIXES)


class _ModelUnavailable(Exception):
    """
    Internal signal that one model failed for a model-level reason.

    Never escapes ``generate()``: it carries the real exception, which is
    re-raised once every model in the chain has been tried.
    """

    def __init__(self, error: Exception, reason: str) -> None:
        super().__init__(reason)
        self.error = error
        self.reason = reason


class OpenRouterProvider(BaseProvider):
    """
    Calls models hosted behind OpenRouter.

    Used for the correctness-critical path (code analysis and grounded
    bug reports), where reasoning quality matters most.
    """

    name: str = "openrouter"

    def __init__(
        self,
        model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: Optional[float] = None,
        config: Optional[Config] = None,
        fallback_models: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize the OpenRouter provider.

        Args:
            model: OpenRouter model slug to call.
            api_key: OpenRouter API key. When omitted, loaded from
                Config / ``OPENROUTER_API_KEY``. Never hardcoded.
            max_tokens: Default maximum tokens per generation.
            base_url: OpenRouter API base URL.
            timeout: Per-request timeout in seconds. When omitted,
                read from ``OPENROUTER_TIMEOUT`` or a safe default.
            config: Optional Config instance. Loaded when not supplied.
            fallback_models: Ordered OpenRouter fallbacks for this
                provider instance. When ``None``, the module default
                ``_FALLBACK_MODELS`` chain is used. Pass an explicit
                sequence (including empty) to override that default —
                agent clients supply their Config-defined chains here.
        """
        cfg = config or Config.load()

        resolved_key = api_key if api_key is not None else cfg.openrouter_api_key
        resolved_model = model or cfg.openrouter_model or cfg.claude_model
        resolved_base = (base_url or cfg.openrouter_base_url).rstrip("/")
        resolved_max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        resolved_timeout = (
            timeout
            if timeout is not None
            else self._timeout_from_environment(_DEFAULT_TIMEOUT_SECONDS)
        )

        super().__init__(model=resolved_model, max_tokens=resolved_max_tokens)
        self.api_key = resolved_key
        self.base_url = resolved_base
        self.timeout = float(resolved_timeout)
        self._config = cfg
        if fallback_models is None:
            self._fallback_models: tuple[str, ...] = tuple(_FALLBACK_MODELS)
        else:
            self._fallback_models = tuple(
                str(item).strip()
                for item in fallback_models
                if str(item or "").strip()
            )

    def generate(self, messages: List[ModelMessage], **kwargs) -> ModelResponse:
        """
        Generate a completion via OpenRouter, falling back across models.

        The configured model is tried first. If it fails for a
        model-level reason (HTTP 402, 404, 429, or 5xx), the same prompt,
        temperature, and token budget are sent to the next model in the
        fallback chain. Authentication and malformed-request failures are
        raised immediately, since retrying them on another model cannot
        help.

        Args:
            messages: Conversation history to send (system/user/assistant).
            **kwargs: Generation options. Recognized keys:
                ``temperature``, ``max_tokens``, ``model``,
                ``response_format`` (OpenAI-compatible structured JSON /
                JSON-schema object when the routed provider supports it).

        Returns:
            A ModelResponse with content, raw payload, and usage metadata.
            ``raw["model_used"]`` names the model that answered.

        Raises:
            ProviderUnavailableError: Missing credentials, unreachable
                endpoint, or exhausted retries on connection failures.
            RateLimitError: Rate limited after retries are exhausted.
            ModelResponseError: Non-retryable API errors or unusable
                response bodies.
        """
        if not self.api_key:
            logger.error("OpenRouter generate() called with no API key configured.")
            raise ProviderUnavailableError(
                "OpenRouter API key is not configured. "
                "Set OPENROUTER_API_KEY in the environment."
            )

        if not messages:
            raise ModelResponseError("OpenRouter generate() requires a non-empty messages list.")

        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = kwargs.get("temperature", 0.0)
        response_format = kwargs.get("response_format")
        chain = self._model_chain(kwargs.get("model", self.model))

        last_error: Optional[Exception] = None
        for position, model in enumerate(chain):
            logger.info("Attempting model: %s", model)
            try:
                return self._generate_with_model(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
            except _ModelUnavailable as signal:
                last_error = signal.error
                logger.warning(
                    "Model unavailable: %s (%s).", model, signal.reason
                )
                remaining = chain[position + 1 :]
                if remaining:
                    logger.info("Switching to: %s", remaining[0])

        logger.error(
            "All OpenRouter models failed (%s); falling back to static-only mode.",
            ", ".join(chain),
        )
        raise last_error  # type: ignore[misc]

    def generate_stream(
        self,
        messages: List[ModelMessage],
        *,
        on_chunk: Optional[Any] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Stream a completion via OpenRouter, falling back across models.

        Calls ``on_chunk(text)`` for each content delta when provided.
        """
        if not self.api_key:
            raise ProviderUnavailableError(
                "OpenRouter API key is not configured. "
                "Set OPENROUTER_API_KEY in the environment."
            )
        if not messages:
            raise ModelResponseError(
                "OpenRouter generate_stream() requires a non-empty messages list."
            )

        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = kwargs.get("temperature", 0.0)
        # Freeform docs path: never force JSON mode while streaming.
        kwargs.pop("response_format", None)
        chain = self._model_chain(kwargs.get("model", self.model))

        last_error: Optional[Exception] = None
        for position, model in enumerate(chain):
            logger.info("Attempting streamed model: %s", model)
            try:
                return self._generate_with_model_stream(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    on_chunk=on_chunk,
                )
            except _ModelUnavailable as signal:
                last_error = signal.error
                logger.warning(
                    "Streamed model unavailable: %s (%s).", model, signal.reason
                )
                remaining = chain[position + 1 :]
                if remaining:
                    logger.info("Switching stream to: %s", remaining[0])

        logger.error(
            "All OpenRouter streamed models failed (%s).",
            ", ".join(chain),
        )
        raise last_error  # type: ignore[misc]

    def _model_chain(self, primary: str) -> List[str]:
        """
        Build the ordered list of models to try for one request.

        Args:
            primary: The requested or configured model, tried first.

        Returns:
            The primary model followed by the fallback models, without
            duplicates.
        """
        chain: List[str] = []
        for candidate in (str(primary or ""),) + tuple(self._fallback_models):
            if candidate and candidate not in chain:
                chain.append(candidate)
        return chain

    def _generate_with_model(
        self,
        model: str,
        messages: List[ModelMessage],
        max_tokens: int,
        temperature: Any,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """
        Run one chat completion against a single model.

        The prompt, temperature, and token budget are identical for every
        model in the chain, so a fallback answer is grounded in exactly
        the same context as the primary attempt would have been.

        Args:
            model: OpenRouter model slug to call.
            messages: Conversation history to send.
            max_tokens: Token ceiling for the completion.
            temperature: Sampling temperature.
            response_format: Optional structured-output constraint
                (``json_object`` or ``json_schema``). When the provider
                rejects it with HTTP 400, the request is retried once
                without ``response_format`` (API compatibility only —
                not an LLM content repair).

        Returns:
            A ModelResponse from this model.

        Raises:
            _ModelUnavailable: The model itself could not serve the
                request; the caller should try the next model.
            ProviderUnavailableError: Credentials or transport failed.
            ModelResponseError: Malformed request or unusable body.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        active_format = response_format
        if active_format is not None:
            if _supports_json_mode(model):
                payload["response_format"] = active_format
            else:
                logger.debug(
                    "Skipping response_format for %s (not in supported list).",
                    model,
                )
                active_format = None

        url = f"{self.base_url}/chat/completions"
        headers = self._headers()

        logger.info(
            "OpenRouter request start: model=%s messages=%d max_tokens=%d "
            "response_format=%s",
            model,
            len(messages),
            max_tokens,
            bool(active_format),
        )

        last_error: Optional[Exception] = None
        format_stripped = False
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                # Copy so later retries (e.g. stripping response_format) do
                # not mutate the body already observed by callers/tests.
                response = requests.post(
                    url,
                    headers=headers,
                    json=dict(payload),
                    timeout=self.timeout,
                )
            except requests.Timeout as exc:
                last_error = exc
                if attempt >= _MAX_ATTEMPTS:
                    break
                delay = self._backoff_seconds(attempt)
                logger.warning(
                    "OpenRouter timeout on attempt %d/%d; retrying in %.1fs",
                    attempt,
                    _MAX_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
                continue
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= _MAX_ATTEMPTS:
                    break
                delay = self._backoff_seconds(attempt)
                logger.warning(
                    "OpenRouter connection error on attempt %d/%d; retrying in %.1fs",
                    attempt,
                    _MAX_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
                continue

            status = response.status_code

            if self._is_retryable(status):
                if attempt >= _MAX_ATTEMPTS:
                    logger.error(
                        "OpenRouter request failed after %d attempts: HTTP %s",
                        _MAX_ATTEMPTS,
                        status,
                    )
                    if status == 429:
                        error: Exception = RateLimitError(
                            f"OpenRouter rate limit exceeded (HTTP 429) on "
                            f"{model} after {_MAX_ATTEMPTS} attempts."
                        )
                    else:
                        error = ProviderUnavailableError(
                            f"OpenRouter unavailable (HTTP {status}) on "
                            f"{model} after {_MAX_ATTEMPTS} attempts."
                        )
                    raise _ModelUnavailable(error, f"HTTP {status}")

                delay = self._backoff_seconds(attempt)
                logger.warning(
                    "OpenRouter HTTP %s on attempt %d/%d; retrying in %.1fs",
                    status,
                    attempt,
                    _MAX_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
                continue

            if status in _FALLBACK_STATUS_CODES:
                # Credits and unknown models do not recover on retry, so
                # move straight to the next model instead of backing off.
                raise _ModelUnavailable(
                    self._model_level_error(status, model, response),
                    f"HTTP {status}",
                )

            if status >= 400:
                if (
                    status == 400
                    and active_format is not None
                    and not format_stripped
                ):
                    logger.warning(
                        "OpenRouter rejected response_format on %s "
                        "(HTTP 400); retrying without structured output.",
                        model,
                    )
                    payload.pop("response_format", None)
                    active_format = None
                    format_stripped = True
                    continue
                self._raise_for_client_error(response)

            model_response = self._parse_response(response, model)
            logger.info(
                "OpenRouter request succeeded: model=%s content_chars=%d",
                model,
                len(model_response.content or ""),
            )
            logger.info("Model used: %s", model)
            return model_response

        logger.error(
            "OpenRouter request failed after %d attempts: connection/timeout",
            _MAX_ATTEMPTS,
        )
        raise ProviderUnavailableError(
            f"OpenRouter request failed after {_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        ) from last_error

    def _generate_with_model_stream(
        self,
        model: str,
        messages: List[ModelMessage],
        max_tokens: int,
        temperature: Any,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> ModelResponse:
        """
        Stream one chat completion against a single model.

        Raises ``_ModelUnavailable`` for model-level failures so the
        caller can walk the fallback chain.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        url = f"{self.base_url}/chat/completions"
        headers = self._headers()
        logger.info(
            "OpenRouter stream start: model=%s messages=%d max_tokens=%d",
            model,
            len(messages),
            max_tokens,
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=dict(payload),
                    timeout=self.timeout,
                    stream=True,
                )
            except requests.Timeout as exc:
                last_error = exc
                if attempt >= _MAX_ATTEMPTS:
                    break
                delay = self._backoff_seconds(attempt)
                time.sleep(delay)
                continue
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= _MAX_ATTEMPTS:
                    break
                delay = self._backoff_seconds(attempt)
                time.sleep(delay)
                continue

            status = response.status_code
            if self._is_retryable(status):
                if attempt >= _MAX_ATTEMPTS:
                    if status == 429:
                        error: Exception = RateLimitError(
                            f"OpenRouter rate limit exceeded (HTTP 429) on "
                            f"{model} after {_MAX_ATTEMPTS} attempts."
                        )
                    else:
                        error = ProviderUnavailableError(
                            f"OpenRouter unavailable (HTTP {status}) on "
                            f"{model} after {_MAX_ATTEMPTS} attempts."
                        )
                    raise _ModelUnavailable(error, f"HTTP {status}")
                delay = self._backoff_seconds(attempt)
                time.sleep(delay)
                continue

            if status in _FALLBACK_STATUS_CODES:
                raise _ModelUnavailable(
                    self._model_level_error(status, model, response),
                    f"HTTP {status}",
                )

            if status >= 400:
                self._raise_for_client_error(response)

            pieces: List[str] = []
            try:
                # Small chunk_size so SSE deltas flush promptly instead of
                # buffering until the response completes.
                for raw_line in response.iter_lines(
                    decode_unicode=True,
                    chunk_size=64,
                ):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if not piece:
                        continue
                    text = str(piece)
                    pieces.append(text)
                    if on_chunk is not None:
                        try:
                            on_chunk(text)
                        except Exception:
                            logger.debug(
                                "OpenRouter stream on_chunk failed",
                                exc_info=True,
                            )
            finally:
                response.close()

            content = "".join(pieces)
            if not content.strip():
                # Treat empty HTTP-200 streams as a model-level failure so
                # generate_stream() can try the next model in the chain.
                # Do not retry the same model: empty content is not a
                # transient transport blip.
                raise _ModelUnavailable(
                    ModelResponseError(
                        "OpenRouter stream completed with empty assistant "
                        "content."
                    ),
                    "empty content",
                )
            logger.info(
                "OpenRouter stream succeeded: model=%s content_chars=%d",
                model,
                len(content),
            )
            logger.info("Model used: %s", model)
            return ModelResponse(
                content=content,
                raw={"model_used": model, "streamed": True},
                usage={},
            )

        raise ProviderUnavailableError(
            f"OpenRouter stream failed after {_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        ) from last_error

    def is_available(self) -> bool:
        """
        Report whether OpenRouter can currently serve requests.

        Returns True only when an API key is present and a lightweight
        authenticated probe against the models endpoint succeeds.

        Returns:
            True if credentials exist and the provider responds usable.
        """
        if not self.api_key:
            logger.info("OpenRouter unavailable: no API key configured.")
            return False

        url = f"{self.base_url}/models"
        try:
            response = requests.get(
                url,
                headers=self._headers(),
                timeout=min(self.timeout, 15.0),
            )
        except requests.RequestException as exc:
            logger.warning("OpenRouter availability probe failed: %s", type(exc).__name__)
            return False

        if response.status_code == 200:
            return True

        logger.info(
            "OpenRouter unavailable: availability probe returned HTTP %s",
            response.status_code,
        )
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        """
        Build request headers for OpenRouter.

        Returns:
            Authorization and content-type headers. Never logs the key.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Huzaifa859/AgenticAI-Intern-Huzaifa-Saboor-",
            "X-Title": "Codebase Assistant",
        }

    @staticmethod
    def _timeout_from_environment(default: float) -> float:
        """
        Read the request timeout from the environment.

        Args:
            default: Fallback when unset or unparseable.

        Returns:
            Timeout in seconds.
        """
        raw = os.environ.get("OPENROUTER_TIMEOUT")
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return value if value > 0 else default

    @staticmethod
    def _is_retryable(status: int) -> bool:
        """
        Report whether a status code is worth retrying on the same model.

        Args:
            status: HTTP status code from OpenRouter.

        Returns:
            True for rate limits and server-side errors.
        """
        return status in _RETRYABLE_STATUS_CODES or status >= 500

    def _model_level_error(
        self, status: int, model: str, response: requests.Response
    ) -> Exception:
        """
        Build the exception for a model-level failure.

        Args:
            status: HTTP status code (402 or 404).
            model: Model that produced the failure.
            response: Failed HTTP response.

        Returns:
            The exception to raise if no fallback model succeeds.
        """
        detail = self._safe_error_detail(response)
        if status == 402:
            return ProviderUnavailableError(
                f"OpenRouter has insufficient credits for {model} "
                f"(HTTP 402). {detail}"
            )
        return ModelResponseError(
            f"OpenRouter model {model} is unavailable (HTTP 404). {detail}"
        )

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        """
        Exponential backoff delay for the given 1-based attempt number.

        Args:
            attempt: Attempt that just failed (1, 2, ...).

        Returns:
            Seconds to sleep before the next attempt.
        """
        return _INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))

    def _raise_for_client_error(self, response: requests.Response) -> None:
        """
        Map a non-retryable HTTP error onto the model exception hierarchy.

        Args:
            response: Failed HTTP response.

        Raises:
            ProviderUnavailableError: Missing/invalid credentials.
            ModelResponseError: Invalid model, malformed request, or
                other client/server errors that must not be retried.
        """
        status = response.status_code
        detail = self._safe_error_detail(response)
        logger.error("OpenRouter request failed: HTTP %s", status)

        if status in (401, 403):
            raise ProviderUnavailableError(
                f"OpenRouter authentication failed (HTTP {status}). "
                f"Check OPENROUTER_API_KEY. {detail}"
            )
        if status == 404:
            raise ModelResponseError(
                f"OpenRouter model not found (HTTP 404). "
                f"Check the configured model name. {detail}"
            )
        if status == 400:
            raise ModelResponseError(
                f"OpenRouter rejected the request as malformed (HTTP 400). {detail}"
            )
        raise ModelResponseError(
            f"OpenRouter API error (HTTP {status}). {detail}"
        )

    @staticmethod
    def _safe_error_detail(response: requests.Response) -> str:
        """
        Extract a short error message from a failed response body.

        Args:
            response: Failed HTTP response.

        Returns:
            A brief detail string safe for logs and exceptions (no secrets).
        """
        try:
            payload = response.json()
        except ValueError:
            text = (response.text or "").strip()
            return text[:200] if text else ""

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error.get("code")
                if message:
                    return str(message)[:200]
            message = payload.get("message")
            if message:
                return str(message)[:200]
        return ""

    @staticmethod
    def _parse_response(
        response: requests.Response, model: str = ""
    ) -> ModelResponse:
        """
        Convert an OpenRouter JSON body into the project's ModelResponse.

        Args:
            response: Successful HTTP response.
            model: Model that produced the response, recorded in
                ``raw["model_used"]`` so callers can report which model
                actually answered after a fallback.

        Returns:
            Populated ModelResponse.

        Raises:
            ModelResponseError: If the body is not usable JSON or lacks
                assistant content.
        """
        try:
            data = response.json()
        except ValueError as exc:
            raise ModelResponseError(
                "OpenRouter returned a non-JSON response body."
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError(
                "OpenRouter response did not contain choices[0].message.content."
            ) from exc

        if content is None:
            raise ModelResponseError(
                "OpenRouter response contained empty assistant content."
            )
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            raise ModelResponseError(
                "OpenRouter response contained empty assistant content."
            )

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        raw = dict(data)
        raw["model_used"] = model or data.get("model", "")
        return ModelResponse(content=content, raw=raw, usage=usage)
