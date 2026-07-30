"""
exceptions
==========

The system's exception hierarchy, rooted at CodebaseAssistantError and
split by the layer that raises it.

A single root exists so the notebook can catch everything the system
raises deliberately and render a readable message, while letting
genuine crashes propagate. Grouping by layer lets a caller catch a
whole category (`ToolError`) without naming each member.

Contains:
- base: CodebaseAssistantError, the root.
- analysis_exceptions: static pass, grounding check, report assembly.
- tool_exceptions: registry, GitHub cloning, filesystem access.
- model_exceptions: LLMClient, providers, embeddings.

NOTE: Placeholder only. None of these are raised by the existing code
yet.
"""

from .analysis_exceptions import (
    AnalysisError,
    GroundingVerificationError,
    InsufficientContextError,
    SourceParseError,
    StaticAnalysisError,
)
from .base import CodebaseAssistantError
from .model_exceptions import (
    EmbeddingError,
    MalformedOutputError,
    ModelError,
    ModelResponseError,
    ProviderUnavailableError,
    RateLimitError,
    TokenLimitExceededError,
)
from .tool_exceptions import (
    EmptyFileError,
    FileTooLargeError,
    InvalidRepositoryURLError,
    PathOutsideWorkspaceError,
    RepositoryCloneError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    UnsupportedFileTypeError,
)

__all__ = [
    "CodebaseAssistantError",
    # analysis
    "AnalysisError",
    "StaticAnalysisError",
    "SourceParseError",
    "GroundingVerificationError",
    "InsufficientContextError",
    # tools
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "InvalidRepositoryURLError",
    "RepositoryCloneError",
    "EmptyFileError",
    "FileTooLargeError",
    "UnsupportedFileTypeError",
    "PathOutsideWorkspaceError",
    # models
    "ModelError",
    "ProviderUnavailableError",
    "ModelResponseError",
    "MalformedOutputError",
    "RateLimitError",
    "TokenLimitExceededError",
    "EmbeddingError",
]
