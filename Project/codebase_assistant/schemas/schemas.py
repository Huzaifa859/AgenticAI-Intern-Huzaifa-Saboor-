"""
schemas.py
==========

Pydantic models used throughout Codebase Assistant.

These define the "shape" of data passed between the Supervisor, Agents,
Tool Registry, RAG package, Memory package, and Model Client.

BugReport, DocumentationResult, and TestGenerationResult mirror the
"Structured output schemas" section of the Codebase Assistant proposal
exactly, field-for-field. No business logic lives here — these are
pure data containers.

TODO: Expand other models as real functionality is implemented
(e.g. add validation, more granular fields, nested sub-models).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Enumerates the specialized agents available in the system."""

    CODE_ANALYSIS = "code_analysis"
    DOCUMENTATION = "documentation"
    TESTING = "testing"


class AgentRequest(BaseModel):
    """
    Represents a unit of work dispatched from the Supervisor to an Agent.

    Attributes:
        task_id: Unique identifier for this task.
        agent_type: Which agent should handle this request.
        instruction: Natural language instruction / goal for the agent.
        context: Arbitrary additional context (file paths, repo info, etc).
    """

    task_id: str
    agent_type: AgentType
    instruction: str
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """
    Represents the result returned by an Agent to the Supervisor.

    Attributes:
        task_id: Identifier matching the originating AgentRequest.
        agent_type: Which agent produced this response.
        success: Whether the agent completed its task successfully.
        output: Free-form result payload.
        errors: List of error messages, if any.
    """

    task_id: str
    agent_type: AgentType
    success: bool = False
    output: Optional[Any] = None
    errors: List[str] = Field(default_factory=list)


class ToolCallRequest(BaseModel):
    """
    Represents a request to invoke a tool from the Tool Registry.

    Attributes:
        tool_name: Registered name of the tool to invoke.
        arguments: Keyword arguments to pass to the tool.
    """

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    """
    Represents the result of a tool invocation.

    Attributes:
        tool_name: Name of the tool that was invoked.
        success: Whether the tool call succeeded.
        result: Free-form result payload.
        error: Error message, if the call failed.
    """

    tool_name: str
    success: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None


class CodeAnalysisResult(BaseModel):
    """
    Placeholder structured result for the Code Analysis Agent.

    Attributes:
        summary: High-level summary of the analysis.
        issues: List of identified issues (lint, complexity, smells, etc).
        metrics: Arbitrary computed metrics (e.g. complexity scores).
    """

    summary: str = ""
    issues: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class BugReport(BaseModel):
    """
    Represents a single bug identified by the Code Analysis Agent.

    Mirrors the proposal's "Structured output schemas" section exactly.
    Per the proposal's grounding pipeline, `evidence` is expected to be
    an exact quoted snippet that is mechanically verified against the
    real source before a report reaches the user.

    Attributes:
        bug_type: Category of bug (e.g. "missing_argument", "undefined_variable").
        description: Human-readable description of the bug.
        severity: How severe the bug is.
        confidence: Confidence score for this finding, 0.0 to 1.0.
        file_path: Path to the file where the bug was found.
        function_name: Name of the function containing the bug.
        line_start: First line number the bug spans.
        line_end: Last line number the bug spans.
        evidence: Exact quoted code the claim is based on; must match
            the source at (file_path, line_start-line_end).
        suggested_fix: Optional suggested fix or remediation.
        detection_method: How the bug was found — "static" (pyflakes/
            ast), "llm" (model reasoning only), "hybrid" (heuristic +
            LLM), or "dynamic" (via actually running generated tests).
    """

    bug_type: str
    description: str
    severity: Literal["low", "medium", "high"]
    confidence: float  # 0.0 - 1.0
    file_path: str
    function_name: str
    line_start: int
    line_end: int
    evidence: str
    suggested_fix: Optional[str] = None
    detection_method: Literal["static", "llm", "hybrid", "dynamic"]


class DocumentationResult(BaseModel):
    """
    Represents generated documentation for a single function.

    Mirrors the proposal's "Structured output schemas" section exactly.

    Attributes:
        file_path: Path to the file containing the documented function.
        function_name: Name of the function being documented.
        summary: One-paragraph summary of what the function does.
        parameters: List of parameter descriptors, each expected to
            carry "name", "type", and "description" keys.
        returns: Description of the function's return value.
        example_usage: A short example showing how to call the function.
    """

    file_path: str
    function_name: str
    summary: str
    parameters: List[Dict[str, Any]]
    returns: str
    example_usage: str


class TestingResult(BaseModel):
    """
    Placeholder structured result for the Testing Agent's `handle()`
    path (broader, multi-file summary form).

    Note: TestGenerationResult (below) is the per-function schema that
    mirrors the proposal exactly. This model remains for the coarser
    "summary across the whole run" use inside testing_agent.handle().

    Attributes:
        summary: High-level summary of testing activity.
        generated_tests: Mapping of file path -> generated test code.
        coverage_estimate: Placeholder coverage estimate (0.0 - 1.0).
    """

    summary: str = ""
    generated_tests: Dict[str, str] = Field(default_factory=dict)
    coverage_estimate: float = 0.0


class TestGenerationResult(BaseModel):
    """
    Represents the result of generating and running tests for a
    single function.

    Mirrors the proposal's "Structured output schemas" section exactly.
    Per the proposal, generated tests are actually executed, so
    execution_status/tests_passed/tests_failed reflect real pass/fail
    results rather than a guess.

    Attributes:
        function_name: Name of the function the tests target.
        generated_test_code: The generated pytest test source code.
        execution_status: Overall outcome of running the generated tests.
        tests_passed: Number of generated tests that passed.
        tests_failed: Number of generated tests that failed.
    """

    function_name: str
    generated_test_code: str
    execution_status: Literal["passed", "failed", "error"]
    tests_passed: int
    tests_failed: int


class RetrievedChunk(BaseModel):
    """
    Represents a single retrieved chunk from the RAG subsystem.

    Attributes:
        source: Origin identifier (e.g. file path) of the chunk.
        content: Text content of the chunk.
        score: Similarity/relevance score.
        metadata: Arbitrary additional metadata.
    """

    source: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    """
    Represents a single entry stored in the Memory subsystem.

    Attributes:
        key: Identifier/key for the memory entry.
        value: Stored value/content.
        metadata: Arbitrary additional metadata (timestamps, tags, etc).
    """

    key: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelMessage(BaseModel):
    """
    Represents a single message in a conversation sent to the model client.

    Attributes:
        role: Role of the message sender (e.g. "system", "user", "assistant").
        content: Text content of the message.
    """

    role: str
    content: str


class ModelResponse(BaseModel):
    """
    Represents a response returned by the Model Client.

    Attributes:
        content: Generated text content.
        raw: Raw underlying response payload, if applicable.
        usage: Token usage / cost metadata.
    """

    content: str = ""
    raw: Optional[Any] = None
    usage: Dict[str, Any] = Field(default_factory=dict)
