"""
config.py
=========

Centralized configuration for Codebase Assistant.

TODO: Replace placeholder defaults with a real settings system
(e.g. pydantic-settings / environment variable loading / YAML config).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """
    Application-wide configuration container.

    Attributes:
        model_name: Identifier of the LLM model to use.
        max_tokens: Maximum tokens per model call.
        github_token: Optional GitHub API token for GitHub tools.
        workspace_root: Root directory of the codebase being analyzed.
        vector_store_path: Path to the RAG vector store on disk.
        memory_store_path: Path to the persistent memory store on disk.
    """

    model_name: str = "placeholder-model"
    max_tokens: int = 4096
    github_token: Optional[str] = None
    workspace_root: str = "."
    vector_store_path: str = "./.codebase_assistant/vector_store"
    memory_store_path: str = "./.codebase_assistant/memory_store"
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """
        Load configuration from a file or environment.

        Args:
            path: Optional path to a config file (YAML/TOML/JSON).

        Returns:
            A Config instance.

        TODO: Implement actual file/env loading logic.
        """
        # TODO: implement real config loading (env vars, file parsing, etc.)
        return cls()
