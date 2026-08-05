"""
provider_manager.py
===================

Selects between a preferred LLM provider and a fallback, with a short
availability cache and one-shot failover on hard transport failures.

Injected into LLMClient as a BaseProvider so agents keep calling a
single client with no knowledge of OpenRouter vs Ollama.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Tuple

from ...exceptions.model_exceptions import (
    ModelResponseError,
    ProviderUnavailableError,
    RateLimitError,
)
from ...schemas.schemas import ModelMessage, ModelResponse
from ...tracing.events import TraceEventType
from ...tracing.tracer import Tracer
from .base import BaseProvider

logger = logging.getLogger(__name__)

# Failures that mean the preferred provider cannot serve this request.
# Content/JSON problems stay on the caller — never failover those.
_FAILOVER_EXCEPTIONS = (ProviderUnavailableError, RateLimitError)


def _friendly_model_label(model: str) -> str:
    """Turn a provider model slug into a short CLI label."""
    slug = (model or "").strip()
    known = {
        "google/gemma-3-27b-it": "Gemma 3 27B",
        "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
        "nvidia/nemotron-nano-9b-v2": "Nemotron Nano 9B",
        "llama3": "llama3",
        "llama3.1": "llama3.1",
        "llama3.2": "llama3.2",
    }
    if slug in known:
        return known[slug]
    if "/" in slug:
        slug = slug.split("/", 1)[1]
    return slug.replace("-", " ").strip() or "unknown model"


class ProviderManager(BaseProvider):
    """
    Routes generate() to the preferred provider when healthy, otherwise
    the fallback. Caches preferred availability for ``cache_seconds``.
    """

    name = "provider_manager"

    def __init__(
        self,
        preferred: Optional[BaseProvider] = None,
        fallback: Optional[BaseProvider] = None,
        *,
        preferred_name: str = "openrouter",
        fallback_name: str = "ollama",
        cache_seconds: int = 60,
        tracer: Optional[Tracer] = None,
    ) -> None:
        """
        Initialize the manager.

        Args:
            preferred: Primary provider (typically OpenRouter).
            fallback: Secondary provider (typically Ollama).
            preferred_name: Label used in traces and status lines.
            fallback_name: Label used in traces and status lines.
            cache_seconds: How long a preferred availability result is
                reused before re-probing.
            tracer: Optional shared Tracer for provider events.
        """
        model = ""
        max_tokens = 0
        if preferred is not None:
            model = getattr(preferred, "model", "") or model
            max_tokens = int(getattr(preferred, "max_tokens", 0) or 0)
        elif fallback is not None:
            model = getattr(fallback, "model", "") or model
            max_tokens = int(getattr(fallback, "max_tokens", 0) or 0)
        super().__init__(model=model, max_tokens=max_tokens)

        self.preferred = preferred
        self.fallback = fallback
        self.preferred_name = (preferred_name or "openrouter").strip().lower()
        self.fallback_name = (fallback_name or "ollama").strip().lower()
        self.cache_seconds = max(0, int(cache_seconds))
        self.tracer = tracer

        self._preferred_available: Optional[bool] = None
        self._preferred_checked_at: float = 0.0
        self._status_message: Optional[str] = None

    # ------------------------------------------------------------------
    # Public status helpers
    # ------------------------------------------------------------------

    def status_message(self) -> str:
        """
        One-line description of which provider will be used.

        Safe to call at CLI startup. Probes preferred availability when
        the cache is cold.
        """
        if self.preferred_is_available() and self.preferred is not None:
            label = _friendly_model_label(getattr(self.preferred, "model", "") or "")
            return f"Using OpenRouter ({label})"
        if self._fallback_is_available():
            label = _friendly_model_label(getattr(self.fallback, "model", "") or "")
            return (
                f"Using Ollama fallback ({label}) — OpenRouter unavailable"
            )
        return "No LLM provider available — static-only mode"

    def preferred_is_available(self) -> bool:
        """
        Return whether the preferred provider is considered healthy.

        Uses a short TTL cache so startup and repeated generate() calls
        do not probe the network every time.
        """
        if self._cache_valid() and self._preferred_available is not None:
            return self._preferred_available

        available = self._probe_preferred()
        self._preferred_available = available
        self._preferred_checked_at = time.monotonic()
        return available

    def mark_preferred_unavailable(self, reason: str = "") -> None:
        """
        Force the preferred provider unavailable for the cache window.

        Args:
            reason: Optional failure reason for logging/tracing.
        """
        self._preferred_available = False
        self._preferred_checked_at = time.monotonic()
        if reason:
            logger.warning(
                "Preferred provider %s marked unavailable for %ss: %s",
                self.preferred_name,
                self.cache_seconds,
                reason,
            )

    def is_available(self) -> bool:
        """True if preferred or fallback can serve requests."""
        return self.preferred_is_available() or self._fallback_is_available()

    def generate(self, messages: List[ModelMessage], **kwargs: Any) -> ModelResponse:
        """
        Generate via preferred provider, with one failover to fallback.

        OpenRouter (preferred) is tried when the availability cache says
        it is healthy. On ProviderUnavailableError / RateLimitError the
        preferred provider is marked unavailable for the cache window and
        the request is retried once on the fallback. ModelResponseError
        and other content failures are never retried on another provider.
        """
        if self.preferred is not None and self.preferred_is_available():
            return self._generate_with_failover(messages, **kwargs)

        if self.fallback is not None and self._fallback_is_available():
            return self._call_provider(
                self.fallback,
                self.fallback_name,
                messages,
                event="provider_selected",
                reason="preferred_unavailable",
                **kwargs,
            )

        raise ProviderUnavailableError(
            "No LLM provider is available. Configure OPENROUTER_API_KEY "
            "or start Ollama."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_with_failover(
        self,
        messages: List[ModelMessage],
        **kwargs: Any,
    ) -> ModelResponse:
        """Call preferred; on hard failure, mark down and try fallback once."""
        assert self.preferred is not None
        started = time.perf_counter()
        try:
            response = self.preferred.generate(messages, **kwargs)
        except _FAILOVER_EXCEPTIONS as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._trace(
                "provider_failed",
                provider=self.preferred_name,
                model=getattr(self.preferred, "model", ""),
                reason=str(exc),
                latency_ms=latency_ms,
                success=False,
            )
            self.mark_preferred_unavailable(str(exc))

            if self.fallback is None or not self._fallback_is_available():
                raise

            self._trace(
                "provider_fallback",
                provider=self.fallback_name,
                model=getattr(self.fallback, "model", ""),
                reason=str(exc),
                from_provider=self.preferred_name,
                success=True,
            )
            return self._call_provider(
                self.fallback,
                self.fallback_name,
                messages,
                event="provider_selected",
                reason=f"fallback_after_{self.preferred_name}_failure",
                **kwargs,
            )
        except ModelResponseError:
            # Bad JSON / unusable content — do not hop providers.
            raise

        latency_ms = int((time.perf_counter() - started) * 1000)
        self._trace(
            "provider_selected",
            provider=self.preferred_name,
            model=getattr(self.preferred, "model", ""),
            reason="preferred_healthy",
            latency_ms=latency_ms,
            success=True,
        )
        self.model = getattr(self.preferred, "model", self.model)
        return response

    def _call_provider(
        self,
        provider: BaseProvider,
        provider_name: str,
        messages: List[ModelMessage],
        *,
        event: str,
        reason: str,
        **kwargs: Any,
    ) -> ModelResponse:
        started = time.perf_counter()
        try:
            response = provider.generate(messages, **kwargs)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._trace(
                "provider_failed",
                provider=provider_name,
                model=getattr(provider, "model", ""),
                reason=str(exc),
                latency_ms=latency_ms,
                success=False,
            )
            raise

        latency_ms = int((time.perf_counter() - started) * 1000)
        self._trace(
            event,
            provider=provider_name,
            model=getattr(provider, "model", ""),
            reason=reason,
            latency_ms=latency_ms,
            success=True,
        )
        self.model = getattr(provider, "model", self.model)
        return response

    def _probe_preferred(self) -> bool:
        if self.preferred is None:
            return False
        try:
            return bool(self.preferred.is_available())
        except Exception as exc:
            logger.warning(
                "Preferred provider %s availability probe failed: %s",
                self.preferred_name,
                type(exc).__name__,
            )
            return False

    def _fallback_is_available(self) -> bool:
        if self.fallback is None:
            return False
        try:
            return bool(self.fallback.is_available())
        except Exception as exc:
            logger.warning(
                "Fallback provider %s availability probe failed: %s",
                self.fallback_name,
                type(exc).__name__,
            )
            return False

    def _cache_valid(self) -> bool:
        if self.cache_seconds <= 0:
            return False
        if self._preferred_available is None:
            return False
        return (time.monotonic() - self._preferred_checked_at) < self.cache_seconds

    def _trace(self, name: str, **metadata: Any) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.record(
                TraceEventType.MODEL_CALL,
                name,
                component="ProviderManager",
                **metadata,
            )
        except Exception as exc:
            logger.warning("ProviderManager tracing failed for %r: %s", name, exc)

    def active_providers(self) -> Tuple[Optional[BaseProvider], Optional[BaseProvider]]:
        """Return (preferred, fallback) for introspection/tests."""
        return self.preferred, self.fallback
