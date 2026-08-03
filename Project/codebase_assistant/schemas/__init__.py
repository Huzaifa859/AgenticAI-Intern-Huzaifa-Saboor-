"""
schemas
=======

Pydantic data models shared across the Codebase Assistant system:
agents, tools, RAG, memory, and the model client all communicate using
these schemas to keep interfaces well-typed and consistent.

BugReport, DocumentationResult, and TestGenerationResult mirror the
proposal's structured output schemas exactly.
"""

from .schemas import (
    AgentType,
    AgentRequest,
    AgentResponse,
    ToolCallRequest,
    ToolCallResult,
    CodeAnalysisResult,
    BugReport,
    AbstentionResult,
    DocumentationResult,
    TestingResult,
    TestGenerationResult,
    RetrievedChunk,
    MemoryRecord,
    ModelMessage,
    ModelResponse,
)

__all__ = [
    "AgentType",
    "AgentRequest",
    "AgentResponse",
    "ToolCallRequest",
    "ToolCallResult",
    "CodeAnalysisResult",
    "BugReport",
    "AbstentionResult",
    "DocumentationResult",
    "TestingResult",
    "TestGenerationResult",
    "RetrievedChunk",
    "MemoryRecord",
    "ModelMessage",
    "ModelResponse",
]
