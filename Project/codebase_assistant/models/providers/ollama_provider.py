"""
ollama_provider.py
===================

Ollama provider for models served by a local Ollama instance —
llama3 in particular, which the proposal assigns to documentation
generation.

Makes real HTTP requests to the Ollama chat API, retries transient
failures with exponential backoff, and maps transport/API failures onto
the project's model exception hierarchy.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from ...config import Config
from ...exceptions.model_exceptions import (
    ModelResponseError,
    ProviderUnavailableError,
)
from ...schemas.schemas import ModelMessage, ModelResponse
from .base import BaseProvider

logger = logging.getLogger(__name__)

#: HTTP status codes that are worth retrying with backoff.
_RETRYABLE_STATUS_CODES = frozenset({500, 502, 503})

#: Default request timeout in seconds when unset.
_DEFAULT_TIMEOUT_SECONDS = 60.0

#: Maximum attempts for a single generate() call (1 initial + retries).
_MAX_ATTEMPTS = 4

#: Initial backoff delay in seconds; doubles after each retryable failure.
_INITIAL_BACKOFF_SECONDS = 1.0


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
        timeout: Optional[float] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize the Ollama provider.

        Args:
            model: Name of the local Ollama model to call. When the
                default is left in place, ``OLLAMA_MODEL`` / Config may
                override it.
            max_tokens: Default maximum tokens per generation
                (``num_predict`` in the Ollama options).
            base_url: Base URL of the local Ollama service. When the
                default is left in place, ``OLLAMA_HOST`` /
                ``OLLAMA_BASE_URL`` / Config may override it.
            timeout: Per-request timeout in seconds. When omitted,
                read from ``OLLAMA_TIMEOUT`` or a safe default.
            config: Optional Config instance. Loaded when not supplied.
        """
        cfg = config or Config.load()

        resolved_model = self._resolve_model(model, cfg)
        resolved_base = self._resolve_base_url(base_url, cfg)
        resolved_max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        resolved_timeout = (
            timeout
            if timeout is not None
            else self._timeout_from_environment(_DEFAULT_TIMEOUT_SECONDS)
        )

        super().__init__(model=resolved_model, max_tokens=resolved_max_tokens)
        self.base_url = resolved_base.rstrip("/")
        self.timeout = float(resolved_timeout)
        self._config = cfg

    def generate(self, messages: List[ModelMessage], **kwargs) -> ModelResponse:
        """
        Generate a completion via local Ollama.

        Args:
            messages: Conversation history to send (system/user/assistant).
            **kwargs: Generation options. Recognized keys:
                ``temperature``, ``max_tokens`` / ``num_predict``,
                ``model``.

        Returns:
            A ModelResponse with content, raw payload, and usage metadata.

        Raises:
            ProviderUnavailableError: Unreachable endpoint or exhausted
                retries on connection/timeout/5xx failures.
            ModelResponseError: Non-retryable API errors or unusable
                response bodies.
        """
        if not messages:
            raise ModelResponseError(
                "Ollama generate() requires a non-empty messages list."
            )

        model = kwargs.get("model", self.model)
        max_tokens = int(
            kwargs.get("num_predict", kwargs.get("max_tokens", self.max_tokens))
        )
        temperature = kwargs.get("temperature", 0.0)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        url = f"{self.base_url}/api/chat"

        logger.info(
            "Ollama request start: model=%s messages=%d num_predict=%d",
            model,
            len(messages),
            max_tokens,
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= _MAX_ATTEMPTS:
                    break
                delay = self._backoff_seconds(attempt)
                logger.warning(
                    "Ollama %s on attempt %d/%d; retrying in %.1fs",
                    type(exc).__name__,
                    attempt,
                    _MAX_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
                continue
            except requests.RequestException as exc:
                # Other transport failures are not retried.
                logger.error("Ollama request failed: %s", type(exc).__name__)
                raise ProviderUnavailableError(
                    f"Ollama request failed: {type(exc).__name__}"
                ) from exc

            if response.status_code in _RETRYABLE_STATUS_CODES:
                if attempt >= _MAX_ATTEMPTS:
                    logger.error(
                        "Ollama request failed after %d attempts: HTTP %s",
                        _MAX_ATTEMPTS,
                        response.status_code,
                    )
                    raise ProviderUnavailableError(
                        f"Ollama unavailable (HTTP {response.status_code}) "
                        f"after {_MAX_ATTEMPTS} attempts."
                    )

                delay = self._backoff_seconds(attempt)
                logger.warning(
                    "Ollama HTTP %s on attempt %d/%d; retrying in %.1fs",
                    response.status_code,
                    attempt,
                    _MAX_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
                continue

            if response.status_code >= 400:
                self._raise_for_client_error(response)

            model_response = self._parse_response(response)
            logger.info(
                "Ollama request succeeded: model=%s content_chars=%d",
                model,
                len(model_response.content or ""),
            )
            return model_response

        logger.error(
            "Ollama request failed after %d attempts: connection/timeout",
            _MAX_ATTEMPTS,
        )
        raise ProviderUnavailableError(
            f"Ollama request failed after {_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        ) from last_error

    def is_available(self) -> bool:
        """
        Report whether the local Ollama service is running.

        Returns True only when ``GET {base_url}/api/tags`` returns HTTP
        200. Connection errors, timeouts, and missing servers return
        False without raising.

        Returns:
            True if the Ollama service responds successfully.
        """
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(
                url,
                timeout=min(self.timeout, 5.0),
            )
        except requests.RequestException as exc:
            logger.info(
                "Ollama unavailable: availability probe failed (%s).",
                type(exc).__name__,
            )
            return False

        if response.status_code == 200:
            return True

        logger.info(
            "Ollama unavailable: availability probe returned HTTP %s",
            response.status_code,
        )
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model(model: str, cfg: Config) -> str:
        """
        Resolve the model name from constructor args, env, and Config.

        Args:
            model: Constructor argument.
            cfg: Loaded Config.

        Returns:
            The model identifier to use.
        """
        env_model = os.environ.get("OLLAMA_MODEL")
        if env_model:
            return env_model
        if model and model != "llama3":
            return model
        return cfg.ollama_model or model or "llama3"

    @staticmethod
    def _resolve_base_url(base_url: str, cfg: Config) -> str:
        """
        Resolve the Ollama host from constructor args, env, and Config.

        Prefers ``OLLAMA_HOST``, then ``OLLAMA_BASE_URL`` / Config.

        Args:
            base_url: Constructor argument.
            cfg: Loaded Config.

        Returns:
            The base URL without a trailing slash.
        """
        env_host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE_URL")
        if env_host:
            return env_host.rstrip("/")
        if base_url and base_url != "http://localhost:11434":
            return base_url.rstrip("/")
        return (cfg.ollama_base_url or base_url).rstrip("/")

    @staticmethod
    def _timeout_from_environment(default: float) -> float:
        """
        Read the request timeout from the environment.

        Args:
            default: Fallback when unset or unparseable.

        Returns:
            Timeout in seconds.
        """
        raw = os.environ.get("OLLAMA_TIMEOUT")
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return value if value > 0 else default

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
            ProviderUnavailableError: Missing service or auth-style failures.
            ModelResponseError: Invalid model, malformed request, or
                other client errors that must not be retried.
        """
        status = response.status_code
        detail = self._safe_error_detail(response)
        logger.error("Ollama request failed: HTTP %s", status)

        if status == 404:
            raise ModelResponseError(
                f"Ollama model or endpoint not found (HTTP 404). "
                f"Check that the model is pulled. {detail}"
            )
        if status == 400:
            raise ModelResponseError(
                f"Ollama rejected the request as malformed (HTTP 400). {detail}"
            )
        raise ModelResponseError(
            f"Ollama API error (HTTP {status}). {detail}"
        )

    @staticmethod
    def _safe_error_detail(response: requests.Response) -> str:
        """
        Extract a short error message from a failed response body.

        Args:
            response: Failed HTTP response.

        Returns:
            A brief detail string safe for logs and exceptions.
        """
        try:
            payload = response.json()
        except ValueError:
            text = (response.text or "").strip()
            return text[:200] if text else ""

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, str) and error:
                return error[:200]
            if isinstance(error, dict):
                message = error.get("message") or error.get("code")
                if message:
                    return str(message)[:200]
            message = payload.get("message")
            if message:
                return str(message)[:200]
        return ""

    @staticmethod
    def _parse_response(response: requests.Response) -> ModelResponse:
        """
        Convert an Ollama JSON body into the project's ModelResponse.

        Args:
            response: Successful HTTP response.

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
                "Ollama returned a non-JSON response body."
            ) from exc

        if not isinstance(data, dict):
            raise ModelResponseError("Ollama response was not a JSON object.")

        try:
            message = data["message"]
            content = message["content"]
        except (KeyError, TypeError) as exc:
            raise ModelResponseError(
                "Ollama response did not contain message.content."
            ) from exc

        if content is None:
            raise ModelResponseError(
                "Ollama response contained empty assistant content."
            )
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            raise ModelResponseError(
                "Ollama response contained empty assistant content."
            )

        usage: Dict[str, Any] = {}
        if "prompt_eval_count" in data:
            usage["prompt_tokens"] = data.get("prompt_eval_count")
        if "eval_count" in data:
            usage["completion_tokens"] = data.get("eval_count")
        if usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
            try:
                usage["total_tokens"] = int(usage["prompt_tokens"]) + int(
                    usage["completion_tokens"]
                )
            except (TypeError, ValueError):
                pass

        return ModelResponse(content=content, raw=data, usage=usage)
