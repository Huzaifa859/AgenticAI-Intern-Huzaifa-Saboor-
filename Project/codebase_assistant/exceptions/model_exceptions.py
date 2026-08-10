"""
model_exceptions.py
====================

Errors raised by the model layer — LLMClient, its providers, and
embedding generation.

Covers the embedding failures the milestone plan requires be handled,
and the malformed-JSON case that drives retry-on-invalid-output.

TODO: Raise these from `models/` and `rag/embeddings.py` once real
provider calls are implemented.
"""

from __future__ import annotations

from .base import CodebaseAssistantError


class ModelError(CodebaseAssistantError):
    """Base class for every model-layer failure."""


class ProviderUnavailableError(ModelError):
    """
    A provider cannot currently serve requests.

    Missing credentials, an unreachable endpoint, or an Ollama service
    that is not running. Intended to drive fallback to another
    provider rather than aborting the run.
    """


class ModelResponseError(ModelError):
    """A provider returned an error or an unusable response."""


class MalformedOutputError(ModelError):
    """
    A response failed Pydantic validation.

    Drives the retry-on-malformed-JSON path rather than letting an
    invalid structure reach the user.
    """


class RateLimitError(ModelError):
    """A provider rejected the request for exceeding its rate limit."""


class TokenLimitExceededError(ModelError):
    """A request exceeded the model's context window."""


class EmbeddingError(ModelError):
    """Generating an embedding failed."""
