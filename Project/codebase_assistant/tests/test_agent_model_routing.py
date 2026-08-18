"""
test_agent_model_routing.py
=============================

Focused checks for per-agent OpenRouter primary/fallback routing.
"""

from __future__ import annotations

from unittest.mock import patch

from codebase_assistant.config import Config
from codebase_assistant.models.providers.openrouter_provider import OpenRouterProvider
from codebase_assistant.schemas.schemas import AgentType
from codebase_assistant.supervisor import Supervisor

#: Models that must not appear in the three agents' active routing chains.
_STALE_AGENT_MODELS = frozenset(
    {
        "cohere/north-mini-code:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "openai/gpt-oss-20b:free",
    }
)


def _clear_routing_env(monkeypatch) -> None:
    for name in (
        "ANALYSIS_MODEL",
        "ANALYSIS_FALLBACK_MODELS",
        "DOCUMENTATION_MODEL",
        "DOCUMENTATION_FALLBACK_MODELS",
        "TESTING_MODEL",
        "TESTING_FALLBACK_MODELS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_analysis_routing_defaults() -> None:
    cfg = Config()
    assert cfg.analysis_model == "anthropic/claude-sonnet-4.5"
    assert cfg.analysis_fallback_models == (
        "qwen/qwen3-coder-plus",
        "qwen/qwen3-coder",
    )
    assert cfg.analysis_model_chain() == (
        "anthropic/claude-sonnet-4.5",
        "qwen/qwen3-coder-plus",
        "qwen/qwen3-coder",
    )


def test_documentation_routing_defaults() -> None:
    cfg = Config()
    assert cfg.documentation_model == "google/gemini-2.5-flash"
    assert cfg.documentation_fallback_models == (
        "qwen/qwen3-coder-flash",
        "google/gemini-2.5-flash-lite",
    )
    assert cfg.documentation_model_chain() == (
        "google/gemini-2.5-flash",
        "qwen/qwen3-coder-flash",
        "google/gemini-2.5-flash-lite",
    )


def test_testing_routing_defaults() -> None:
    cfg = Config()
    assert cfg.testing_model == "google/gemini-2.5-flash"
    assert cfg.testing_fallback_models == (
        "anthropic/claude-sonnet-4.5",
        "qwen/qwen3-coder-plus",
    )
    assert cfg.testing_model_chain() == (
        "google/gemini-2.5-flash",
        "anthropic/claude-sonnet-4.5",
        "qwen/qwen3-coder-plus",
    )


def test_env_overrides_for_agent_routing(monkeypatch) -> None:
    _clear_routing_env(monkeypatch)
    monkeypatch.setenv("ANALYSIS_MODEL", "anthropic/claude-sonnet-4.5")
    monkeypatch.setenv(
        "ANALYSIS_FALLBACK_MODELS",
        "qwen/qwen3-coder-plus,qwen/qwen3-coder",
    )
    monkeypatch.setenv("DOCUMENTATION_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setenv(
        "DOCUMENTATION_FALLBACK_MODELS",
        "qwen/qwen3-coder-flash,google/gemini-2.5-flash-lite",
    )
    monkeypatch.setenv("TESTING_MODEL", "custom/testing-primary")
    monkeypatch.setenv(
        "TESTING_FALLBACK_MODELS",
        "custom/testing-fb1, custom/testing-fb2",
    )

    cfg = Config.load()

    assert cfg.analysis_model_chain() == (
        "anthropic/claude-sonnet-4.5",
        "qwen/qwen3-coder-plus",
        "qwen/qwen3-coder",
    )
    assert cfg.documentation_model_chain() == (
        "google/gemini-2.5-flash",
        "qwen/qwen3-coder-flash",
        "google/gemini-2.5-flash-lite",
    )
    assert cfg.testing_model == "custom/testing-primary"
    assert cfg.testing_fallback_models == (
        "custom/testing-fb1",
        "custom/testing-fb2",
    )
    assert cfg.testing_model_chain() == (
        "custom/testing-primary",
        "custom/testing-fb1",
        "custom/testing-fb2",
    )


def test_stale_models_absent_from_default_agent_chains() -> None:
    cfg = Config()
    active = set(
        cfg.analysis_model_chain()
        + cfg.documentation_model_chain()
        + cfg.testing_model_chain()
    )
    assert active.isdisjoint(_STALE_AGENT_MODELS)


def test_openrouter_provider_uses_agent_fallback_chain() -> None:
    cfg = Config(openrouter_api_key="test-key")
    provider = OpenRouterProvider(
        model=cfg.analysis_model,
        api_key="test-key",
        config=cfg,
        fallback_models=cfg.analysis_fallback_models,
    )
    assert provider._model_chain(cfg.analysis_model) == list(
        cfg.analysis_model_chain()
    )


def test_openrouter_provider_empty_fallback_override() -> None:
    provider = OpenRouterProvider(
        model="google/gemini-2.5-flash",
        api_key="test-key",
        config=Config(openrouter_api_key="test-key"),
        fallback_models=(),
    )
    assert provider._model_chain("google/gemini-2.5-flash") == [
        "google/gemini-2.5-flash"
    ]


def _provider_chain(agent) -> list:
    client = agent.model_client
    manager = client.provider
    preferred = manager.preferred
    assert manager.fallback is None
    assert isinstance(preferred, OpenRouterProvider)
    return preferred._model_chain(preferred.model)


def test_supervisor_agent_clients_use_config_chains() -> None:
    cfg = Config(
        openrouter_api_key="test-key",
        analysis_model="anthropic/claude-sonnet-4.5",
        analysis_fallback_models=(
            "qwen/qwen3-coder-plus",
            "qwen/qwen3-coder",
        ),
        documentation_model="google/gemini-2.5-flash",
        documentation_fallback_models=(
            "qwen/qwen3-coder-flash",
            "google/gemini-2.5-flash-lite",
        ),
        testing_model="google/gemini-2.5-flash",
        testing_fallback_models=(
            "anthropic/claude-sonnet-4.5",
            "qwen/qwen3-coder-plus",
        ),
    )
    with patch(
        "codebase_assistant.supervisor.OpenRouterProvider.is_available",
        return_value=True,
    ), patch(
        "codebase_assistant.supervisor.OllamaProvider.is_available",
        return_value=False,
    ):
        supervisor = Supervisor(config=cfg)

    analysis = supervisor.agents[AgentType.CODE_ANALYSIS]
    docs = supervisor.agents[AgentType.DOCUMENTATION]
    testing = supervisor.agents[AgentType.TESTING]

    assert _provider_chain(analysis) == list(cfg.analysis_model_chain())
    assert _provider_chain(docs) == list(cfg.documentation_model_chain())
    assert _provider_chain(testing) == list(cfg.testing_model_chain())
    for chain in (
        _provider_chain(analysis),
        _provider_chain(docs),
        _provider_chain(testing),
    ):
        assert set(chain).isdisjoint(_STALE_AGENT_MODELS)
