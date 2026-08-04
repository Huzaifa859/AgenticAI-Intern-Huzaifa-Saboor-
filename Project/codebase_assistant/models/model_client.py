"""
model_client.py
================

Defines ModelClient, the single choke point every agent and tool goes
through to make model calls.

The client is deliberately provider-agnostic: it validates the
conversation going out, delegates to whichever BaseProvider it was given,
validates the response coming back, and does nothing else. All
vendor-specific concerns -- endpoints, authentication, request shape,
retries, token accounting -- belong in the provider implementations
under `models/providers/`, not here.

Because providers are injected rather than constructed internally,
swapping Claude for a local llama3, or stubbing a fake provider in
tests, requires no change to calling code.

Multi-provider routing and OpenRouter → Ollama failover live in
ProviderManager, which is injected here as the BaseProvider.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..config import Config
from ..exceptions.model_exceptions import ModelResponseError, ProviderUnavailableError
from ..schemas.schemas import ModelMessage, ModelResponse
from .providers.base import BaseProvider

# Roles a ModelMessage may carry. Kept here rather than on the provider
# because it is a property of the conversation format, not of any vendor.
VALID_ROLES = frozenset({"system", "user", "assistant"})


class ModelClient:
    """
    Provider-agnostic wrapper around an LLM provider.

    Holds a BaseProvider and forwards validated conversations to it.
    Contains no vendor-specific logic of any kind.
    """

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize the ModelClient.

        Args:
            provider: The provider backend to delegate to. May be None
                at construction time and supplied later via
                `set_provider`, which lets the Supervisor wire itself up
                before any provider is configured.
            config: Optional Config instance. A default is loaded when
                not supplied.
        """
        self.config = config or Config.load()
        self._provider = provider

    @property
    def provider(self) -> Optional[BaseProvider]:
        """
        The provider this client currently delegates to.

        Returns:
            The configured BaseProvider, or None if none is set.
        """
        return self._provider

    def set_provider(self, provider: BaseProvider) -> None:
        """
        Replace the provider this client delegates to.

        Args:
            provider: The provider to use for subsequent calls.

        Raises:
            TypeError: If `provider` is not a BaseProvider.
        """
        if not isinstance(provider, BaseProvider):
            raise TypeError(
                f"provider must be a BaseProvider, got {type(provider).__name__}"
            )
        self._provider = provider

    def is_available(self) -> bool:
        """
        Report whether a provider is configured and reachable.

        Returns:
            True if a provider is set and reports itself available.
        """
        if self._provider is None:
            return False
        return self._provider.is_available()

    def generate(self, messages: Sequence[ModelMessage], **options: Any) -> ModelResponse:
        """
        Generate a completion for a conversation.

        Args:
            messages: Conversation history to send, oldest first.
            **options: Provider-specific generation options, passed
                through untouched.

        Returns:
            The provider's ModelResponse.

        Raises:
            ValueError: If `messages` is empty or contains an invalid
                entry. This signals a caller bug rather than a runtime
                condition, so it is not part of the model exception
                hierarchy.
            ProviderUnavailableError: If no provider is configured.
            ModelResponseError: If the provider returns something that
                is not a valid ModelResponse.
        """
        self._validate_messages(messages)

        if self._provider is None:
            raise ProviderUnavailableError(
                "No provider configured on this ModelClient. "
                "Pass one to the constructor or call set_provider()."
            )

        response = self._provider.generate(list(messages), **options)
        self._validate_response(response)
        return response

    def generate_from_prompt(
        self,
        prompt: str,
        system: Optional[str] = None,
        **options: Any,
    ) -> ModelResponse:
        """
        Generate a completion from a single prompt string.

        Convenience wrapper for the common single-turn case; builds the
        message list and delegates to `generate`.

        Args:
            prompt: The user prompt.
            system: Optional system instruction placed before the
                prompt.
            **options: Provider-specific generation options.

        Returns:
            The provider's ModelResponse.

        Raises:
            ValueError: If `prompt` is empty or whitespace only.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        messages = []
        if system:
            messages.append(ModelMessage(role="system", content=system))
        messages.append(ModelMessage(role="user", content=prompt))
        return self.generate(messages, **options)

    @staticmethod
    def _validate_messages(messages: Sequence[ModelMessage]) -> None:
        """
        Check that a conversation is well formed before sending it.

        Args:
            messages: The conversation to validate.

        Raises:
            ValueError: If the conversation is empty, contains a
                non-ModelMessage entry, uses an unrecognized role, or
                carries empty content.
        """
        if not messages:
            raise ValueError("messages must contain at least one ModelMessage.")

        for index, message in enumerate(messages):
            if not isinstance(message, ModelMessage):
                raise ValueError(
                    f"messages[{index}] must be a ModelMessage, "
                    f"got {type(message).__name__}."
                )
            if message.role not in VALID_ROLES:
                raise ValueError(
                    f"messages[{index}] has unsupported role {message.role!r}. "
                    f"Expected one of {sorted(VALID_ROLES)}."
                )
            if not message.content or not message.content.strip():
                raise ValueError(f"messages[{index}] has empty content.")

    @staticmethod
    def _validate_response(response: Any) -> None:
        """
        Check that a provider returned a usable response.

        Args:
            response: Whatever the provider returned.

        Raises:
            ModelResponseError: If the value is not a ModelResponse or
                carries no content.
        """
        if not isinstance(response, ModelResponse):
            raise ModelResponseError(
                f"Provider returned {type(response).__name__}, expected ModelResponse."
            )
        if not response.content:
            raise ModelResponseError("Provider returned an empty response.")


class LLMClient(ModelClient):
    """
    Backward-compatible alias for ModelClient.

    Deprecated. `Supervisor` and `BaseAgent` still refer to this name and
    still pass `model_name`/`max_tokens`; this subclass keeps them
    working unchanged while ModelClient takes over as the real
    interface.

    TODO: Update Supervisor and BaseAgent to construct ModelClient with
    an injected provider, then delete this class.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        provider: Optional[BaseProvider] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initialize the deprecated client.

        Args:
            model_name: Legacy model identifier. Recorded only; the
                active model is a property of the injected provider.
            max_tokens: Legacy token ceiling. Recorded only, for the
                same reason.
            provider: Provider backend to delegate to.
            config: Optional Config instance.
        """
        super().__init__(provider=provider, config=config)
        self.model_name = model_name or self.config.model_name
        self.max_tokens = max_tokens or self.config.max_tokens

    def switch_model(self, model_name: str) -> None:
        """
        Record a change of model identifier.

        Args:
            model_name: Identifier to switch to.

        TODO: Remove alongside this class. Selecting a model is now done
        by injecting the appropriate provider.
        """
        self.model_name = model_name
