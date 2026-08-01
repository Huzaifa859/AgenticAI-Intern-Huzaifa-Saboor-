"""
config.py
=========

Centralized configuration for Codebase Assistant.

Every project-wide constant lives here: the Scope & Limits ceilings from
the proposal, the RAG/ChromaDB settings, model identifiers, and logging
setup. No other module should hardcode these values -- they take a
Config instance (or read one via `Config.load()`) instead, so a single
edit changes behavior everywhere.

`Config.load()` applies environment overrides on top of the defaults,
which keeps API keys and machine-specific paths out of source control.

TODO: Add file-based loading (YAML/TOML/JSON) alongside the environment
overrides implemented below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at install time
    load_dotenv = None  # type: ignore[assignment]


def _env_str(name: str, default: str) -> str:
    """
    Read a string setting from the environment.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is unset or empty.

    Returns:
        The environment value, or `default`.
    """
    return os.environ.get(name) or default


def _env_optional_str(name: str) -> Optional[str]:
    """
    Read a setting that is legitimately absent when unset.

    Used for credentials, where "not configured" is a meaningful state
    rather than something needing a default.

    Args:
        name: Environment variable name.

    Returns:
        The environment value, or None.
    """
    return os.environ.get(name) or None


def _env_int(name: str, default: int) -> int:
    """
    Read an integer setting from the environment.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is unset or unparseable.

    Returns:
        The parsed environment value, or `default` if it is not a valid
        integer. A malformed override falls back rather than raising, so
        a typo in the environment cannot prevent startup.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Config:
    """
    Application-wide configuration container.

    Grouped by concern: ingestion limits, RAG settings, model
    identifiers, filesystem paths, and logging.

    Attributes:
        max_repository_files: Ceiling on source files ingested from one
            repository.
        max_total_lines_of_code: Ceiling on total lines ingested across
            the repository.
        max_file_size_bytes: Per-file size ceiling; larger files are
            skipped.
        ignore_directories: Directory names excluded from ingestion.

        chroma_persist_directory: Directory ChromaDB persists to.
        chroma_collection_name: Name of the ChromaDB collection holding
            code chunks.
        embedding_model_name: sentence-transformers model used to embed
            chunks.
        retrieval_top_k: Number of chunks retrieved per query.

        openrouter_base_url: Base URL of the OpenRouter API.
        openrouter_api_key: OpenRouter API key. Sourced from the
            environment; never commit a value here.
        openrouter_model: Default model slug requested through
            OpenRouter.
        claude_model: Claude slug used for code analysis and
            bug-finding. Reached via OpenRouter, hence the same slug
            namespace.
        ollama_base_url: Base URL of the local Ollama service.
        ollama_model: Local model used for documentation generation.
        max_tokens: Default maximum tokens per model call.
        model_name: Legacy single-model identifier, superseded by
            `claude_model` and `ollama_model`. Retained because existing
            callers still read it.

        github_token: Optional GitHub token. Not required for the MVP,
            which clones public repositories only.
        workspace_root: Root directory of the codebase being analyzed.
        memory_store_path: Path to the persistent memory store on disk.

        log_level: Logging threshold name (DEBUG, INFO, WARNING, ...).
        log_format: Format string applied to log records.
        log_date_format: Timestamp format applied to log records.

        extra: Escape hatch for ad-hoc settings that do not yet warrant
            a dedicated field.
    """

    # --- Ingestion limits (proposal: Scope & Limits) ------------------
    max_repository_files: int = 100
    max_total_lines_of_code: int = 20_000
    max_file_size_bytes: int = 500 * 1024
    ignore_directories: Tuple[str, ...] = (
        ".git",
        "__pycache__",
        "venv",
        "node_modules",
    )

    # --- RAG / vector store (proposal: Indexing Design) ---------------
    chroma_persist_directory: str = "./.codebase_assistant/chroma"
    chroma_collection_name: str = "codebase_chunks"
    embedding_model_name: str = "all-mpnet-base-v2"
    retrieval_top_k: int = 8

    # --- Models (proposal: Tech Stack) --------------------------------
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-sonnet-4"
    claude_model: str = "anthropic/claude-sonnet-4"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    max_tokens: int = 2048
    model_name: str = "anthropic/claude-sonnet-4"

    # --- Filesystem ---------------------------------------------------
    github_token: Optional[str] = None
    workspace_root: str = "."
    memory_store_path: str = "./.codebase_assistant/memory_store"

    # --- Logging ------------------------------------------------------
    log_level: str = "INFO"
    log_format: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"

    extra: dict = field(default_factory=dict)

    @property
    def vector_store_path(self) -> str:
        """
        Path to the RAG vector store on disk.

        Backward-compatible alias for `chroma_persist_directory`, kept so
        existing callers keep working while there remains a single
        source of truth for the location.

        Returns:
            The ChromaDB persistence directory.
        """
        return self.chroma_persist_directory

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """
        Build a Config from defaults plus environment overrides.

        Only the settings that genuinely vary per machine or must stay
        out of source control are overridable: credentials, service
        URLs, storage locations, and the log level.

        Args:
            path: Optional path to a config file. Not yet honored.

        Returns:
            A Config instance.

        TODO: Parse `path` (YAML/TOML/JSON) and layer it between the
        defaults and the environment overrides.
        """
        # TODO: load settings from `path` before applying env overrides
        # Load local `.env` into os.environ (does not override existing
        # vars). Credentials stay out of source; never logged here.
        if load_dotenv is not None:
            project_env = Path(__file__).resolve().parent.parent / ".env"
            load_dotenv(project_env)
            load_dotenv()

        defaults = cls()
        return cls(
            openrouter_api_key=_env_optional_str("OPENROUTER_API_KEY"),
            openrouter_base_url=_env_str(
                "OPENROUTER_BASE_URL", defaults.openrouter_base_url
            ),
            ollama_base_url=_env_str("OLLAMA_BASE_URL", defaults.ollama_base_url),
            github_token=_env_optional_str("GITHUB_TOKEN"),
            workspace_root=_env_str("WORKSPACE_ROOT", defaults.workspace_root),
            chroma_persist_directory=_env_str(
                "CHROMA_PERSIST_DIR", defaults.chroma_persist_directory
            ),
            memory_store_path=_env_str(
                "MEMORY_STORE_PATH", defaults.memory_store_path
            ),
            retrieval_top_k=_env_int("RETRIEVAL_TOP_K", defaults.retrieval_top_k),
            log_level=_env_str("LOG_LEVEL", defaults.log_level),
        )
