"""
documentation_agent.py
========================

Defines DocumentationAgent, responsible for generating and updating
documentation (docstrings, README sections, API docs) for a codebase.

Uses the injected Ollama-backed LLMClient, Retriever for RAG context,
and ToolRegistry-resolved FilesystemTools for reading repository source.

When the model returns malformed or unparseable JSON, the agent performs
exactly one JSON-repair retry before falling back to the existing
abstention path.

After a DocumentationResult is parsed, a mechanical grounding stage
checks referenced files, modules, packages, classes, and functions
against the repository inventory (and AST symbols) and removes or
rewrites unsupported claims before the result is returned.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..analysis.report_builder import ReportBuilder
from ..rag.indexer import Indexer
from ..schemas.schemas import (
    AbstentionResult,
    AgentRequest,
    AgentResponse,
    AgentType,
    DocumentationResult,
    ModelMessage,
    RetrievedChunk,
)
from ..tools.filesystem_tools import FilesystemTools
from ..tracing.events import TraceEventType
from .base import BaseAgent

logger = logging.getLogger(__name__)

#: Cap on source files read into the prompt when building repository context.
_MAX_SOURCE_FILES = 12

#: Cap on characters taken from each source file (fallback when no RAG).
_MAX_FILE_CHARS = 2500

#: Shorter file excerpts when retrieved chunks are already present.
_MAX_FILE_CHARS_WITH_RETRIEVAL = 1500

#: Cap on retrieved chunks rendered into the prompt.
_MAX_CONTEXT_CHUNKS = 8

#: Cap on characters per retrieved chunk in the prompt.
_MAX_CHUNK_CHARS = 1200

#: Generation ceiling for documentation calls.
_DOC_MAX_TOKENS = 2048

#: Soft ceiling for text embedded in a JSON-repair retry prompt.
_MAX_RETRY_PROMPT_CHARS = 8_000

#: Cap on raw model output embedded in a JSON-repair retry prompt.
_MAX_RETRY_RAW_CHARS = 4_000

#: Cap on file paths listed in the repository inventory section.
_MAX_INVENTORY_FILES = 150

#: Project metadata files that reveal dependencies and run instructions.
_PROJECT_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "environment.yml",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    ".env.example",
    "README.md",
    "README.rst",
)

#: Cap on characters read from each project metadata file.
_MAX_PROJECT_FILE_CHARS = 1200

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

#: Path-like tokens ending in ``.py`` (possibly nested).
_PATH_IN_TEXT = re.compile(
    r"(?<![\w/])((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py|[A-Za-z0-9_.-]+\.py)\b"
)

#: Backtick-quoted identifiers / paths in markdown.
_BACKTICK_REF = re.compile(r"`([^`\n]+)`")

#: Dotted import-style module paths (``pkg.sub.mod``).
_DOTTED_MODULE = re.compile(
    r"\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+)\b"
)

#: Explicit ``class Name`` / ``function name`` / ``module name`` claims.
_CLASS_CLAIM = re.compile(
    r"\b(?:class|Class)\s+`?([A-Z][A-Za-z0-9_]*)`?"
)
_FUNC_CLAIM = re.compile(
    r"\b(?:function|method|def)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?"
)
_MODULE_CLAIM = re.compile(
    r"\b(?:module|package)\s+`?([A-Za-z_][A-Za-z0-9_.]*)`?"
)

#: CamelCase tokens that look like invented service/type names.
_CAMEL_SYMBOL = re.compile(
    r"\b([A-Z][a-zA-Z0-9]+(?:Service|Manager|Client|Handler|Controller|"
    r"Factory|Builder|Provider|Error|Exception|Agent|Result|Utils|Helper))\b"
)

#: Function names that are documentation modes, not code symbols.
_SPECIAL_FUNCTION_NAMES = frozenset(
    {"README", "readme", "API", "api", "module", "MODULE"}
)

#: Neutral filler used when an unsupported claim is rewritten.
_UNGROUNDED_PLACEHOLDER = (
    "Not documented in the provided repository evidence."
)

_JSON_RETRY_SYSTEM_PROMPT = """\
You are repairing malformed DocumentationResult JSON.

This is a JSON repair step, not a new documentation generation.
Return ONLY valid DocumentationResult JSON.
Do not add explanations.
Do not wrap the JSON in markdown fences.
Do not add prose before or after the JSON.
Follow the DocumentationResult schema exactly.
Preserve all information that can be recovered from the raw model output.

Required shape:

{
  "file_path": "path/to/file.py",
  "function_name": "name_or_README_or_module",
  "summary": "Markdown documentation body.",
  "parameters": [
    {"name": "param_or_module", "type": "str_or_kind", "description": "What it is."}
  ],
  "returns": "Return value description, or empty string if N/A.",
  "example_usage": "Realistic usage or run commands, or empty string."
}
"""

_SYSTEM_PROMPT = """\
You are a senior technical writer producing developer documentation for
Python codebases. You write structured, markdown-formatted documentation
that is grounded strictly in the evidence you are given.

Grounding rules (follow exactly):
1. Use only the provided RETRIEVED CONTEXT, REPOSITORY INVENTORY,
   PROJECT FILES, and REPOSITORY CONTENTS as evidence.
2. Never invent functions, classes, parameters, files, commands,
   dependencies, or APIs. Every name you write must appear in the
   evidence.
3. Prefer RETRIEVED CONTEXT when it is present; use REPOSITORY CONTENTS
   only to fill gaps.
4. When evidence for a section is missing, write
   "Not documented in the provided repository evidence." for that
   section instead of guessing, or omit the section entirely.
5. Explain purpose before implementation details. Keep wording precise
   and technical - no marketing language.

Formatting rules:
- The "summary" field holds markdown. Use `##` headings, bullet lists,
  and fenced code blocks. Do not produce a single paragraph.
- Because the output is JSON, escape newlines inside strings as \\n and
  quotes as \\".
- Do not wrap the JSON itself in markdown fences.

Return ONE JSON object only (no prose outside JSON):

{
  "file_path": "path/to/file.py",
  "function_name": "name_or_README_or_module",
  "summary": "Markdown documentation body.",
  "parameters": [
    {"name": "param_or_module", "type": "str_or_kind", "description": "What it is."}
  ],
  "returns": "Return value description, or empty string if N/A.",
  "example_usage": "Realistic usage or run commands, or empty string."
}

Mode guidance:
- docstring: document one function; summary = purpose; fill parameters/returns.
- module: summarize the module's role; parameters may list public symbols.
- readme: full repository documentation; function_name=README.
- api_reference: public API summary; parameters list public callables/classes.
"""


@dataclass
class _SymbolCatalog:
    """Repository paths and symbols used to ground documentation claims."""

    files: Set[str] = field(default_factory=set)
    modules: Set[str] = field(default_factory=set)
    packages: Set[str] = field(default_factory=set)
    functions: Set[str] = field(default_factory=set)
    classes: Set[str] = field(default_factory=set)

    def has_file(self, path: str) -> bool:
        """Return True when ``path`` matches a known repository file."""
        normalized = DocumentationAgent._normalize_repo_path(path)
        if not normalized:
            return False
        normalized = normalized.replace("\\", "/")
        if normalized in self.files:
            return True
        # Claim ``app/auth.py`` while inventory is workspace-relative ``auth.py``.
        parts = normalized.split("/")
        if len(parts) >= 2:
            suffix = "/".join(parts[1:])
            if suffix in self.files:
                return True
        return False

    def has_module(self, name: str) -> bool:
        """Return True when ``name`` is a known module or package."""
        key = (name or "").strip()
        if not key:
            return False
        if key in self.modules or key in self.packages:
            return True
        # Accept ``math_utils.py`` style mentions as modules.
        if key.endswith(".py"):
            return self.has_file(key) or key[:-3] in self.modules
        # Claim ``app.auth`` vs inventory module ``auth``.
        for known in self.modules | self.packages:
            if key.endswith("." + known) or known.endswith("." + key):
                return True
        return False

    def has_function(self, name: str) -> bool:
        """Return True when ``name`` is a known function symbol."""
        return (name or "").strip() in self.functions

    def has_class(self, name: str) -> bool:
        """Return True when ``name`` is a known class symbol."""
        return (name or "").strip() in self.classes


@dataclass
class _GroundingStats:
    """Counters / samples recorded while grounding one DocumentationResult."""

    verified: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)

    def note_verified(self, kind: str, value: str) -> None:
        token = f"{kind}:{value}"
        if token not in self.verified:
            self.verified.append(token)

    def note_removed(self, kind: str, value: str) -> None:
        token = f"{kind}:{value}"
        if token not in self.removed:
            self.removed.append(token)
        if token not in self.unsupported:
            self.unsupported.append(token)


_README_GUIDANCE = """\
Write complete repository documentation suitable for a README overview.
Put the whole document in "summary" as markdown, using these sections in
order and only when the repository evidence supports them:

## Project Overview
## Purpose
## Main Functionality
## Architecture Overview
## Directory and Module Summary
## Important Classes, Functions, and Modules
## Technologies and Frameworks
## Installation Requirements
## How to Run
## Key Workflows
## Limitations
## Suggested Future Improvements

Section rules:
- Directory and Module Summary: describe only paths present in the
  REPOSITORY INVENTORY.
- Technologies and Frameworks: name only libraries that appear in
  imports or in PROJECT FILES.
- Installation Requirements and How to Run: derive strictly from
  PROJECT FILES (requirements.txt, pyproject.toml, Makefile, Dockerfile,
  entry-point modules). If nothing is discoverable, state that plainly
  instead of inventing commands.
- Key Workflows: describe end-to-end flows visible in the code, naming
  the real modules or functions that implement each step.
- Limitations: only gaps visible in the evidence, such as missing tests,
  TODO markers, or unimplemented branches.
- Suggested Future Improvements: only items justified by a limitation
  you just documented. Omit the section otherwise.

Field rules for this mode:
- function_name: "README".
- file_path: the repository path given in TARGET.
- parameters: the main modules or packages, one entry each, with
  name = path, type = "module" or "package", description = its role.
- returns: one sentence naming the primary entry point, or "".
- example_usage: real run commands taken from the evidence, or "".
"""

_MODE_GUIDANCE = {
    "docstring": (
        "Write a function docstring-style DocumentationResult. "
        "Start the summary with what the function is for, then note "
        "important behavior or caveats visible in the code. "
        "Use markdown with a short purpose paragraph followed by "
        "bullet points for behavior and caveats. "
        "List only parameters that appear in the signature."
    ),
    "module": (
        "Write a module summary in markdown with '## Purpose', "
        "'## Public Surface', and '## Notes' sections. Explain the "
        "module's role first, then the public symbols it exposes and "
        "how they fit into the wider system. "
        "Do not invent symbols that are not in the code."
    ),
    "readme": _README_GUIDANCE,
    "api_reference": (
        "Write a public API reference in markdown, grouping symbols "
        "under '## Classes' and '## Functions' headings with a short "
        "grounded description and signature for each. List the same "
        "real public functions/classes in parameters "
        "(name/type/description). "
        "Omit private helpers (names starting with underscore) unless "
        "they are the only content."
    ),
}


class DocumentationAgent(BaseAgent):
    """
    Agent specialized in generating and maintaining documentation for
    a codebase, including docstrings, README files, and API references.
    """

    agent_type: AgentType = AgentType.DOCUMENTATION

    def run(self, repo_path: str) -> Dict[str, str]:
        """
        Simple entry point used by the Supervisor for direct routing.

        Args:
            repo_path: Path to the repository to document.

        Returns:
            A dict with "status" and "message" keys summarizing the run.
        """
        result = self.generate_readme(repo_path)
        if result.summary.strip():
            return {
                "status": "success",
                "message": result.summary[:500],
            }
        return {
            "status": "error",
            "message": "Documentation generation failed or produced no summary.",
        }

    def handle(self, request: AgentRequest) -> AgentResponse:
        """
        Handle a documentation generation request.

        Args:
            request: The AgentRequest describing what to document.

        Returns:
            An AgentResponse wrapping a DocumentationResult. Success is
            reported on AgentResponse (DocumentationResult has no
            success field in the existing schema).
        """
        context = request.context or {}
        instruction = (request.instruction or "").strip()
        repo_path = str(context.get("repo_path") or context.get("repository_path") or ".")
        file_path = str(context.get("file_path") or "")
        function_name = str(context.get("function_name") or "")
        doc_type = str(context.get("doc_type") or "").lower()

        if not self._model_available():
            logger.info("Documentation model unavailable; returning failed response.")
            empty = self._empty_result(
                file_path=file_path or repo_path,
                function_name=function_name or "",
            )
            return AgentResponse(
                task_id=request.task_id,
                agent_type=self.agent_type,
                success=False,
                output=empty,
                errors=[
                    "Documentation model provider is unavailable; "
                    "documentation was not generated."
                ],
            )

        try:
            lowered = instruction.lower()
            if doc_type == "readme" or "readme" in lowered:
                result = self.generate_readme(repo_path)
            elif doc_type in {"api", "api_reference"} or "api reference" in lowered:
                result = self.generate_api_reference(file_path or repo_path)
            elif doc_type in {"module", "module_summary"} or "module summary" in lowered:
                target = file_path or repo_path
                workspace = self._workspace_for(target)
                result = self._run_pipeline(
                    mode="module",
                    workspace=workspace,
                    target_path=target,
                    instruction=(
                        instruction
                        or f"Summarize the purpose and public surface of module {target}."
                    ),
                    function_name=os.path.basename(target) or "module",
                )
            else:
                target = file_path or repo_path
                workspace = self._workspace_for(target)
                doc_instruction = instruction or (
                    f"Document function {function_name} in {target}."
                    if function_name
                    else f"Document the primary public function in {target}."
                )
                result = self._run_pipeline(
                    mode="docstring",
                    workspace=workspace,
                    target_path=target,
                    instruction=doc_instruction,
                    function_name=function_name,
                )
        except Exception as exc:
            logger.warning("DocumentationAgent.handle failed: %s", exc)
            empty = self._empty_result(
                file_path=file_path or repo_path,
                function_name=function_name or "",
            )
            return AgentResponse(
                task_id=request.task_id,
                agent_type=self.agent_type,
                success=False,
                output=empty,
                errors=[str(exc)],
            )

        if result.abstention is not None:
            return AgentResponse(
                task_id=request.task_id,
                agent_type=self.agent_type,
                success=False,
                output=result,
                errors=[result.abstention.reason],
            )
        success = bool(result.summary and result.summary.strip())
        return AgentResponse(
            task_id=request.task_id,
            agent_type=self.agent_type,
            success=success,
            output=result,
            errors=[] if success else ["Documentation generation produced an empty summary."],
        )

    def generate_docstring(self, file_path: str) -> DocumentationResult:
        """
        Generate a docstring for a function in a given file.

        Args:
            file_path: Path to the file containing the function.

        Returns:
            A DocumentationResult populated from the model, or an empty
            result when the model is unavailable or generation fails.
        """
        workspace = self._workspace_for(file_path)
        return self._run_pipeline(
            mode="docstring",
            workspace=workspace,
            target_path=file_path or workspace,
            instruction=(
                "Write a concise technical docstring for the primary public "
                f"function in {file_path}. Purpose first, then behavior. "
                "Do not invent parameters."
            ),
            function_name="",
        )

    def generate_readme(self, repo_path: str) -> DocumentationResult:
        """
        Generate a README file summarizing the repository.

        Args:
            repo_path: Path to the repository root.

        Returns:
            A DocumentationResult whose summary holds the README body.
        """
        workspace = self._workspace_for(repo_path)
        return self._run_pipeline(
            mode="readme",
            workspace=workspace,
            target_path=workspace,
            instruction=(
                f"Write complete repository documentation for {repo_path} "
                "suitable for a README overview: project overview, purpose, "
                "main functionality, architecture, directory and module "
                "summary, important classes and functions, technologies "
                "used, installation requirements, how to run, key "
                "workflows, limitations, and justified improvements. "
                "Stay grounded in the repository evidence and state "
                "clearly when information is unavailable."
            ),
            function_name="README",
        )

    def generate_api_reference(self, module_path: str) -> DocumentationResult:
        """
        Generate API reference documentation for a module.

        Args:
            module_path: Path to the module to document.

        Returns:
            A DocumentationResult summarizing the module's public API.
        """
        workspace = self._workspace_for(module_path)
        return self._run_pipeline(
            mode="api_reference",
            workspace=workspace,
            target_path=module_path or workspace,
            instruction=(
                f"Summarize the public API of {module_path}. "
                "List only symbols visible in the code. Purpose before details."
            ),
            function_name=os.path.basename(module_path) or "module",
        )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        mode: str,
        workspace: str,
        target_path: str,
        instruction: str,
        function_name: str,
    ) -> DocumentationResult:
        """
        Run the documentation pipeline for one request.

        Stages: read repository → retrieve context → build prompt →
        call the documentation model → parse DocumentationResult.
        """
        empty = self._empty_result(
            file_path=target_path,
            function_name=function_name,
        )

        if not self._model_available():
            logger.info("Documentation model unavailable; skipping generation.")
            return empty

        self._trace(
            "documentation_started",
            mode=mode,
            target_path=target_path,
            workspace=workspace,
        )

        # Index first so retrieval can prioritize grounded chunks.
        logger.info("Indexing repository for documentation retrieval...")
        self._trace("indexing", event_type=TraceEventType.INGESTION, workspace=workspace)
        index_started = time.perf_counter()
        self._ensure_index(workspace)
        self._trace(
            "indexing",
            event_type=TraceEventType.INGESTION,
            success=True,
            duration_ms=(time.perf_counter() - index_started) * 1000.0,
            phase="finished",
        )

        logger.info("Retrieving documentation context...")
        self._trace(
            "retrieval",
            event_type=TraceEventType.RETRIEVAL,
            phase="started",
        )
        retrieval_started = time.perf_counter()
        query = " ".join(
            part for part in (instruction, function_name, target_path, mode) if part
        )
        chunks = self._retrieve_context(query, target_path)
        self._trace(
            "retrieval",
            event_type=TraceEventType.RETRIEVAL,
            success=True,
            duration_ms=(time.perf_counter() - retrieval_started) * 1000.0,
            chunks=len(chunks),
            phase="finished",
        )

        logger.info("Reading repository...")
        filesystem = self._filesystem_tools(workspace)
        # Repository-wide modes need the file listing to describe layout,
        # so they keep it even when retrieval already returned chunks.
        repository_wide = mode == "readme"
        source_excerpts = self._read_repository_sources(
            filesystem,
            target_path,
            max_file_chars=(
                _MAX_FILE_CHARS_WITH_RETRIEVAL if chunks else _MAX_FILE_CHARS
            ),
            prefer_target_only=bool(chunks) and not repository_wide,
        )
        inventory = self._repository_inventory(filesystem) if repository_wide else []
        project_files = self._read_project_files(filesystem) if repository_wide else []

        if not chunks and not source_excerpts and not project_files:
            abstained = self._abstain_result(
                empty,
                reason=(
                    "Repository contains no supported Python files."
                    if not inventory
                    else "No grounded evidence was found."
                ),
                evidence_available=(
                    [f"inventory listed {len(inventory)} path(s)"]
                    if inventory
                    else []
                ),
                recommended_next_steps=[
                    "Point documentation at a Python module or package.",
                    "Confirm the target path contains readable source files.",
                ],
            )
            self._trace(
                "documentation_finished",
                success=False,
                abstained=True,
                reason=abstained.abstention.reason if abstained.abstention else "",
            )
            return abstained

        logger.info("Building prompt...")
        prompt = self._build_prompt(
            mode=mode,
            instruction=instruction,
            target_path=target_path,
            function_name=function_name,
            chunks=chunks,
            source_excerpts=source_excerpts,
            inventory=inventory,
            project_files=project_files,
        )

        logger.info("Calling documentation model...")
        self._trace(
            "model_request",
            event_type=TraceEventType.MODEL_CALL,
            chunks=len(chunks),
        )
        model_started = time.perf_counter()
        try:
            response = self.model_client.generate(
                [
                    ModelMessage(role="system", content=_SYSTEM_PROMPT),
                    ModelMessage(role="user", content=prompt),
                ],
                max_tokens=_DOC_MAX_TOKENS,
                temperature=0.1,
            )
        except Exception as exc:
            logger.warning("Documentation model call failed: %s", exc)
            self._trace(
                "model_response",
                event_type=TraceEventType.MODEL_CALL,
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - model_started) * 1000.0,
            )
            self._trace(
                "documentation_finished",
                success=False,
                error=str(exc),
            )
            return empty

        self._trace(
            "model_response",
            event_type=TraceEventType.MODEL_CALL,
            success=True,
            duration_ms=(time.perf_counter() - model_started) * 1000.0,
            content_chars=len(response.content or ""),
        )

        result, parse_error = self._parse_response_with_status(
            response.content,
            default_file_path=target_path,
            default_function_name=function_name,
        )
        if parse_error is not None:
            result = self._retry_json_repair(
                original_prompt=prompt,
                raw_output=response.content or "",
                parse_error=parse_error,
                default_file_path=target_path,
                default_function_name=function_name,
            )

        if not (result.summary and result.summary.strip()):
            evidence = []
            if chunks:
                evidence.append(f"{len(chunks)} retrieved chunk(s)")
            if source_excerpts:
                evidence.append(f"{len(source_excerpts)} source excerpt(s)")
            result = self._abstain_result(
                empty,
                reason="LLM response could not be verified.",
                evidence_available=evidence,
                recommended_next_steps=[
                    "Retry with a narrower documentation target.",
                    "Confirm the model returned valid DocumentationResult JSON.",
                ],
            )
            self._trace(
                "documentation_finished",
                success=False,
                abstained=True,
                reason=result.abstention.reason if result.abstention else "",
            )
            return result

        # Ground references against the repository before returning.
        if not inventory:
            inventory = self._repository_inventory(filesystem)
        result, grounding_stats, grounding_abstained = self._ground_documentation(
            result,
            filesystem=filesystem,
            inventory=inventory,
            mode=mode,
            target_path=target_path,
        )
        if grounding_abstained or not (result.summary and result.summary.strip()):
            evidence = []
            if inventory:
                evidence.append(f"{len(inventory)} inventoried path(s)")
            if grounding_stats.removed:
                evidence.append(
                    f"{len(grounding_stats.removed)} unsupported claim(s) removed"
                )
            result = self._abstain_result(
                empty,
                reason="Documentation claims could not be grounded in the repository.",
                evidence_available=evidence,
                recommended_next_steps=[
                    "Retry with a narrower documentation target.",
                    "Ensure generated documentation only cites inventory paths and symbols.",
                ],
            )
            self._trace(
                "documentation_grounding_abstained",
                removed=len(grounding_stats.removed),
                unsupported=len(grounding_stats.unsupported),
                verified=len(grounding_stats.verified),
            )
            self._trace(
                "documentation_finished",
                success=False,
                abstained=True,
                reason=result.abstention.reason if result.abstention else "",
            )
            return result

        logger.info("Documentation generated.")
        self._trace(
            "documentation_finished",
            success=True,
            function_name=result.function_name,
            summary_chars=len(result.summary or ""),
            grounded_verified=len(grounding_stats.verified),
            grounded_removed=len(grounding_stats.removed),
        )
        return result

    def _retry_json_repair(
        self,
        *,
        original_prompt: str,
        raw_output: str,
        parse_error: str,
        default_file_path: str,
        default_function_name: str,
    ) -> DocumentationResult:
        """
        Perform exactly one JSON-repair model call.

        Never regenerates documentation from scratch: the retry prompt
        asks the model only to repair formatting / JSON validity while
        preserving recoverable content from the raw output.
        """
        self._trace(
            "documentation_retry_started",
            parse_error=parse_error,
            raw_chars=len(raw_output or ""),
        )

        retry_prompt = self._build_json_retry_prompt(
            original_prompt=original_prompt,
            raw_output=raw_output,
            parse_error=parse_error,
        )
        model_started = time.perf_counter()
        try:
            response = self.model_client.generate(
                [
                    ModelMessage(role="system", content=_JSON_RETRY_SYSTEM_PROMPT),
                    ModelMessage(role="user", content=retry_prompt),
                ],
                max_tokens=_DOC_MAX_TOKENS,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Documentation JSON retry call failed: %s", exc)
            self._trace(
                "documentation_retry_failed",
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - model_started) * 1000.0,
            )
            return self._empty_result(default_file_path, default_function_name)

        repaired, retry_error = self._parse_response_with_status(
            response.content,
            default_file_path=default_file_path,
            default_function_name=default_function_name,
        )
        if retry_error is not None or not (
            repaired.summary and repaired.summary.strip()
        ):
            self._trace(
                "documentation_retry_failed",
                success=False,
                error=retry_error or "repaired JSON produced an empty summary",
                duration_ms=(time.perf_counter() - model_started) * 1000.0,
            )
            return self._empty_result(default_file_path, default_function_name)

        self._trace(
            "documentation_retry_success",
            success=True,
            duration_ms=(time.perf_counter() - model_started) * 1000.0,
            summary_chars=len(repaired.summary or ""),
        )
        return repaired

    def _build_json_retry_prompt(
        self,
        *,
        original_prompt: str,
        raw_output: str,
        parse_error: str,
    ) -> str:
        """Build the user prompt for a single DocumentationResult JSON repair."""
        prompt_text = (original_prompt or "").strip() or "(none)"
        if len(prompt_text) > _MAX_RETRY_PROMPT_CHARS:
            prompt_text = (
                prompt_text[:_MAX_RETRY_PROMPT_CHARS]
                + "\n...[truncated original prompt]"
            )

        raw_text = (raw_output or "").strip() or "(empty)"
        if len(raw_text) > _MAX_RETRY_RAW_CHARS:
            raw_text = raw_text[:_MAX_RETRY_RAW_CHARS] + "\n...[truncated raw output]"

        return (
            "DOCUMENTATION JSON REPAIR MODE\n"
            "Return ONLY valid DocumentationResult JSON.\n"
            "Do not add explanations or markdown.\n"
            "Do not wrap the JSON in code fences.\n"
            "Preserve all information that can be recovered.\n"
            "Follow the DocumentationResult schema exactly.\n\n"
            f"PARSER / VALIDATION ERROR\n{parse_error}\n\n"
            f"RAW MODEL OUTPUT\n{raw_text}\n\n"
            f"ORIGINAL PROMPT\n{prompt_text}\n\n"
            "OUTPUT CONTRACT\n"
            "Return only the repaired DocumentationResult JSON object."
        )

    # ------------------------------------------------------------------
    # Documentation grounding (mechanical; no model call)
    # ------------------------------------------------------------------

    def _ground_documentation(
        self,
        result: DocumentationResult,
        *,
        filesystem: FilesystemTools,
        inventory: Sequence[str],
        mode: str,
        target_path: str,
    ) -> Tuple[DocumentationResult, _GroundingStats, bool]:
        """
        Verify documentation references against the repository inventory.

        Removes or rewrites unsupported file/module/class/function claims.
        Returns ``(grounded_result, stats, abstain)``.
        """
        stats = _GroundingStats()
        self._trace(
            "documentation_grounding_started",
            mode=mode,
            inventory=len(inventory),
        )
        started = time.perf_counter()

        catalog = self._build_symbol_catalog(filesystem, inventory)
        file_path = self._ground_file_path(
            result.file_path,
            catalog,
            stats,
            target_path=target_path,
            filesystem=filesystem,
        )
        function_name = self._ground_function_name(
            result.function_name, catalog, stats, mode=mode
        )
        parameters = self._ground_parameters(
            result.parameters, catalog, stats, mode=mode
        )
        summary = self._ground_text_field(
            result.summary, catalog, stats, field_name="summary"
        )
        returns = self._ground_text_field(
            result.returns, catalog, stats, field_name="returns"
        )
        example_usage = self._ground_text_field(
            result.example_usage, catalog, stats, field_name="example_usage"
        )

        # If unsupported claims wiped the body but grounded symbols remain,
        # keep a minimal neutral summary instead of inventing new symbols.
        # A repaired file_path alone is not enough to keep the result.
        if self._text_lacks_substance(summary):
            if (
                parameters
                or (
                    function_name
                    and function_name not in _SPECIAL_FUNCTION_NAMES
                )
            ):
                summary = self._minimal_grounded_summary(
                    file_path=file_path,
                    function_name=function_name,
                    parameters=parameters,
                )
            else:
                summary = ""

        grounded = DocumentationResult(
            file_path=file_path,
            function_name=function_name,
            summary=summary,
            parameters=parameters,
            returns=returns,
            example_usage=example_usage,
            abstention=None,
        )

        # Abstain only when nothing verifiable remains after scrubbing.
        remaining = (summary or "").strip()
        has_anchor = bool(
            parameters
            or (
                function_name
                and function_name not in _SPECIAL_FUNCTION_NAMES
            )
            or (
                function_name in _SPECIAL_FUNCTION_NAMES
                and file_path
                and not self._text_lacks_substance(remaining)
                and remaining != _UNGROUNDED_PLACEHOLDER
            )
        )
        abstain = (not remaining) or (
            bool(stats.removed)
            and not has_anchor
            and self._text_lacks_substance(remaining)
        )

        self._trace(
            "documentation_grounding_finished",
            success=not abstain,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            verified=len(stats.verified),
            removed=len(stats.removed),
            unsupported=len(stats.unsupported),
            verified_references=list(stats.verified[:20]),
            removed_references=list(stats.removed[:20]),
            unsupported_references=list(stats.unsupported[:20]),
        )
        return grounded, stats, abstain

    def _build_symbol_catalog(
        self,
        filesystem: FilesystemTools,
        inventory: Sequence[str],
    ) -> _SymbolCatalog:
        """Build path/module/symbol sets from inventory + AST of Python files."""
        catalog = _SymbolCatalog()
        for raw in inventory:
            path = self._normalize_repo_path(raw)
            if not path:
                continue
            catalog.files.add(path)
            if path.endswith(".py"):
                module = path[:-3].replace("/", ".")
                if module.endswith(".__init__"):
                    package = module[: -len(".__init__")]
                    if package:
                        catalog.packages.add(package)
                        catalog.modules.add(package)
                else:
                    catalog.modules.add(module)
                    if "." in module:
                        catalog.packages.add(module.rsplit(".", 1)[0])
                # Also index the basename module (math_utils).
                catalog.modules.add(os.path.basename(path)[:-3])

            # Directory segments act as packages when they contain code.
            parts = path.split("/")
            if len(parts) > 1:
                for index in range(1, len(parts)):
                    catalog.packages.add(".".join(parts[:index]).replace("/", "."))
                    catalog.packages.add("/".join(parts[:index]))

        py_files = sorted(path for path in catalog.files if path.endswith(".py"))
        for path in py_files[:_MAX_SOURCE_FILES * 3]:
            try:
                if not filesystem.file_exists(path):
                    continue
                source = filesystem.read_file(path)
            except Exception as exc:
                logger.warning("Grounding skipped unreadable %s: %s", path, exc)
                continue
            try:
                tree = ast.parse(source or "", filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    catalog.functions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    catalog.classes.add(node.name)
        return catalog

    def _ground_file_path(
        self,
        file_path: str,
        catalog: _SymbolCatalog,
        stats: _GroundingStats,
        *,
        target_path: str,
        filesystem: FilesystemTools,
    ) -> str:
        """Keep ``file_path`` only when it exists in the inventory."""
        candidate = self._normalize_repo_path(file_path)
        if candidate and catalog.has_file(candidate):
            stats.note_verified("file", candidate)
            return candidate

        # Fall back to a workspace-relative target when the model invented a path.
        rel_target = self._normalize_repo_path(
            self._relative_to_workspace(filesystem, target_path)
        )
        if rel_target and rel_target not in {".", ""} and catalog.has_file(rel_target):
            if candidate:
                stats.note_removed("file", candidate)
                self._trace(
                    "documentation_grounding_removed_claim",
                    kind="file",
                    value=candidate,
                    replacement=rel_target,
                )
            stats.note_verified("file", rel_target)
            return rel_target

        if candidate:
            stats.note_removed("file", candidate)
            self._trace(
                "documentation_grounding_removed_claim",
                kind="file",
                value=candidate,
            )
        return ""

    def _ground_function_name(
        self,
        function_name: str,
        catalog: _SymbolCatalog,
        stats: _GroundingStats,
        *,
        mode: str,
    ) -> str:
        """Keep special mode names or symbols that exist in the catalog."""
        name = (function_name or "").strip()
        if not name:
            return ""
        if name in _SPECIAL_FUNCTION_NAMES or (
            mode == "readme" and name.lower() == "readme"
        ):
            stats.note_verified("function_name", name)
            return name
        if catalog.has_function(name) or catalog.has_class(name):
            stats.note_verified("function", name)
            return name
        if catalog.has_module(name):
            stats.note_verified("module", name)
            return name
        # Allow documenting a module by its ``.py`` basename.
        if name.endswith(".py") and catalog.has_file(name):
            stats.note_verified("file", name)
            return name

        stats.note_removed("function", name)
        self._trace(
            "documentation_grounding_removed_claim",
            kind="function",
            value=name,
        )
        return ""

    def _ground_parameters(
        self,
        parameters: Sequence[Dict[str, Any]],
        catalog: _SymbolCatalog,
        stats: _GroundingStats,
        *,
        mode: str,
    ) -> List[Dict[str, Any]]:
        """Filter parameter / symbol entries that cannot be grounded."""
        kept: List[Dict[str, Any]] = []
        for item in parameters or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            type_hint = str(item.get("type", "") or "").strip().lower()
            description = str(item.get("description", "") or "")
            if not name:
                continue

            grounded_name = True
            kind = "parameter"
            if mode in {"readme", "api_reference", "module"} or type_hint in {
                "module",
                "package",
                "class",
                "function",
            }:
                if type_hint in {"module", "package"} or catalog.has_module(name):
                    kind = "module"
                    grounded_name = catalog.has_module(name) or catalog.has_file(name)
                elif type_hint == "class" or (
                    name[:1].isupper() and catalog.has_class(name)
                ):
                    kind = "class"
                    grounded_name = catalog.has_class(name)
                elif type_hint == "function" or catalog.has_function(name):
                    kind = "function"
                    grounded_name = catalog.has_function(name)
                elif name.endswith(".py") or "/" in name:
                    kind = "file"
                    grounded_name = catalog.has_file(name)
                elif name[:1].isupper():
                    kind = "class"
                    grounded_name = catalog.has_class(name)
                else:
                    # README/module entries that name unknown symbols are dropped.
                    grounded_name = (
                        catalog.has_function(name)
                        or catalog.has_module(name)
                        or catalog.has_class(name)
                    )
            else:
                # Docstring parameters are signature names; keep them when the
                # surrounding function is real, or when they match a symbol.
                grounded_name = True

            cleaned_description = self._ground_text_field(
                description, catalog, stats, field_name=f"param:{name}"
            )
            if not grounded_name:
                stats.note_removed(kind, name)
                self._trace(
                    "documentation_grounding_removed_claim",
                    kind=kind,
                    value=name,
                )
                continue

            stats.note_verified(kind, name)
            kept.append(
                {
                    "name": name,
                    "type": str(item.get("type", "")),
                    "description": cleaned_description,
                }
            )
        return kept

    def _ground_text_field(
        self,
        text: str,
        catalog: _SymbolCatalog,
        stats: _GroundingStats,
        *,
        field_name: str,
    ) -> str:
        """Remove or rewrite unsupported references inside a free-text field."""
        if not text:
            return ""

        cleaned = text
        # File paths first (most specific).
        for match in list(_PATH_IN_TEXT.finditer(cleaned)):
            path = match.group(1)
            if catalog.has_file(path):
                stats.note_verified("file", self._normalize_repo_path(path))
                continue
            stats.note_removed("file", path)
            self._trace(
                "documentation_grounding_removed_claim",
                kind="file",
                value=path,
                field=field_name,
            )
            cleaned = self._scrub_claim_from_text(cleaned, path)

        # Explicit class / function / module claims.
        for regex, kind, checker in (
            (_CLASS_CLAIM, "class", catalog.has_class),
            (_FUNC_CLAIM, "function", catalog.has_function),
            (_MODULE_CLAIM, "module", catalog.has_module),
        ):
            for match in list(regex.finditer(cleaned)):
                name = match.group(1)
                if checker(name):
                    stats.note_verified(kind, name)
                    continue
                stats.note_removed(kind, name)
                self._trace(
                    "documentation_grounding_removed_claim",
                    kind=kind,
                    value=name,
                    field=field_name,
                )
                cleaned = self._scrub_claim_from_text(cleaned, match.group(0))

        # Dotted modules (pkg.sub).
        for match in list(_DOTTED_MODULE.finditer(cleaned)):
            name = match.group(1)
            if catalog.has_module(name):
                stats.note_verified("module", name)
                continue
            # Only remove when it looks like a code reference under a known root.
            root = name.split(".", 1)[0]
            if root in catalog.packages or root in catalog.modules:
                stats.note_removed("module", name)
                self._trace(
                    "documentation_grounding_removed_claim",
                    kind="module",
                    value=name,
                    field=field_name,
                )
                cleaned = self._scrub_claim_from_text(cleaned, name)

        # Backtick references.
        for match in list(_BACKTICK_REF.finditer(cleaned)):
            inner = match.group(1).strip()
            if not inner or inner == _UNGROUNDED_PLACEHOLDER:
                continue
            if self._reference_is_grounded(inner, catalog, stats):
                continue
            # Leave non-code backticks (commands, flags) alone when they contain
            # spaces or look like shell snippets.
            if " " in inner or inner.startswith("-"):
                continue
            kind = self._classify_reference(inner, catalog)
            stats.note_removed(kind, inner)
            self._trace(
                "documentation_grounding_removed_claim",
                kind=kind,
                value=inner,
                field=field_name,
            )
            cleaned = self._scrub_claim_from_text(cleaned, match.group(0))

        # CamelCase service/type names that are not in the catalog.
        for match in list(_CAMEL_SYMBOL.finditer(cleaned)):
            name = match.group(1)
            if catalog.has_class(name) or catalog.has_function(name):
                stats.note_verified("class", name)
                continue
            stats.note_removed("class", name)
            self._trace(
                "documentation_grounding_removed_claim",
                kind="class",
                value=name,
                field=field_name,
            )
            cleaned = self._scrub_claim_from_text(cleaned, name)

        cleaned = cleaned.replace(_UNGROUNDED_PLACEHOLDER, "").strip()
        return cleaned

    def _reference_is_grounded(
        self,
        value: str,
        catalog: _SymbolCatalog,
        stats: _GroundingStats,
    ) -> bool:
        """Return True and record verification when ``value`` is known."""
        token = value.strip().strip("\"'")
        if not token:
            return True
        if catalog.has_file(token):
            stats.note_verified("file", self._normalize_repo_path(token))
            return True
        if catalog.has_module(token):
            stats.note_verified("module", token)
            return True
        if catalog.has_class(token):
            stats.note_verified("class", token)
            return True
        if catalog.has_function(token):
            stats.note_verified("function", token)
            return True
        return False

    @staticmethod
    def _classify_reference(value: str, catalog: _SymbolCatalog) -> str:
        """Best-effort kind label for an unsupported reference."""
        if "/" in value or value.endswith(".py"):
            return "file"
        if "." in value:
            return "module"
        if value[:1].isupper():
            return "class"
        if catalog.has_function(value) or value == value.lower():
            return "function"
        return "symbol"

    @staticmethod
    def _scrub_claim_from_text(text: str, claim: str) -> str:
        """
        Remove sentences/lines containing an unsupported claim.

        When removal would erase the whole field, replace the claim token
        with a neutral placeholder instead of inventing new symbols.
        """
        if not text or not claim:
            return text
        lines = text.splitlines()
        kept_lines: List[str] = []
        removed_any = False
        for line in lines:
            # Split lightly on sentence boundaries within the line.
            pieces = re.split(r"(?<=[.!?])\s+", line)
            kept_pieces = [piece for piece in pieces if claim not in piece]
            if len(kept_pieces) != len(pieces):
                removed_any = True
            rebuilt = " ".join(piece.strip() for piece in kept_pieces if piece.strip())
            if rebuilt:
                kept_lines.append(rebuilt)
        if kept_lines:
            return "\n".join(kept_lines).strip()
        if removed_any:
            return _UNGROUNDED_PLACEHOLDER
        return text.replace(claim, _UNGROUNDED_PLACEHOLDER)

    @staticmethod
    def _text_lacks_substance(text: str) -> bool:
        """Return True when text is empty or only neutral filler."""
        if not text or not text.strip():
            return True
        stripped = (
            text.replace(_UNGROUNDED_PLACEHOLDER, "")
            .replace("`", "")
            .strip(" \n\t.:;,-_")
        )
        return not stripped

    @staticmethod
    def _minimal_grounded_summary(
        *,
        file_path: str,
        function_name: str,
        parameters: Sequence[Dict[str, Any]],
    ) -> str:
        """Build a short summary from remaining grounded anchors only."""
        parts: List[str] = []
        if function_name and function_name not in _SPECIAL_FUNCTION_NAMES:
            parts.append(f"Documented symbol: `{function_name}`.")
        elif function_name in _SPECIAL_FUNCTION_NAMES:
            parts.append(f"Documentation for {function_name}.")
        if file_path:
            parts.append(f"Source: `{file_path}`.")
        if parameters:
            names = ", ".join(
                str(item.get("name", "")).strip()
                for item in parameters
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            )
            if names:
                parts.append(f"Grounded symbols: {names}.")
        return " ".join(parts).strip()

    @staticmethod
    def _normalize_repo_path(path: str) -> str:
        """Normalize a repository-relative path for inventory comparisons."""
        if not path:
            return ""
        normalized = str(path).replace("\\", "/").strip()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        # Drop absolute prefixes / drive letters when present.
        if re.match(r"^[A-Za-z]:/", normalized):
            normalized = normalized.split(":/", 1)[-1]
        if normalized.startswith("/"):
            normalized = normalized.lstrip("/")
        return normalized

    def _model_available(self) -> bool:
        """Report whether the injected model client can serve requests."""
        if self.model_client is None:
            return False
        try:
            return bool(self.model_client.is_available())
        except Exception as exc:
            logger.warning("Documentation model availability check failed: %s", exc)
            return False

    def _ensure_index(self, workspace: str) -> None:
        """
        Index the workspace into the store the injected Retriever reads.

        Uses the existing Indexer.update_index path. Failures are logged
        and swallowed so documentation can still continue from files.
        """
        if self.retriever is None:
            logger.info("No retriever configured; skipping documentation indexing.")
            return

        try:
            vector_db = None
            try:
                vector_db = self.retriever.vector_db
            except Exception:
                vector_db = None

            indexer = Indexer(
                vector_store_path=self.retriever.vector_store_path,
                config=self.retriever.config,
                workspace_root=workspace,
                vector_db=vector_db,
            )
            update = indexer.update_index(".")
            logger.info("Documentation index: %s", update.summary())
        except Exception as exc:
            logger.warning(
                "Documentation indexing failed; retrieval may be empty: %s",
                exc,
            )

    def _retrieve_context(
        self, query: str, target_path: str
    ) -> List[RetrievedChunk]:
        """
        Retrieve RAG chunks for the documentation request.

        Always attempts retrieval. Returns an empty list when the
        retriever is missing, the index is empty, or retrieval fails.
        """
        if self.retriever is None:
            logger.info("No retriever configured; continuing with repository files only.")
            return []

        try:
            chunks = self.retriever.retrieve(query=query or target_path or "documentation")
        except Exception as exc:
            logger.warning(
                "Retrieval failed; continuing with repository files only: %s",
                exc,
            )
            return []

        if not chunks:
            logger.info("Retriever returned no chunks; continuing with repository files only.")
            return []
        return self._dedupe_chunks(chunks)[:_MAX_CONTEXT_CHUNKS]

    @staticmethod
    def _dedupe_chunks(chunks: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Merge retrieved chunks into a de-duplicated list.

        Drops exact content duplicates and near-identical excerpts from
        the same source so the prompt stays compact.
        """
        unique: List[RetrievedChunk] = []
        seen_content: set[str] = set()
        for chunk in chunks:
            content = (chunk.content or "").strip()
            if not content:
                continue
            # Normalize whitespace for duplicate detection.
            key = " ".join(content.split())
            if key in seen_content:
                continue
            seen_content.add(key)
            unique.append(chunk)
        return unique

    def _read_repository_sources(
        self,
        filesystem: FilesystemTools,
        target_path: str,
        max_file_chars: int = _MAX_FILE_CHARS,
        prefer_target_only: bool = False,
    ) -> List[str]:
        """
        Read repository source files via FilesystemTools.

        Prefers the target file when it is a single file; otherwise lists
        Python sources under the workspace. When retrieved context is
        already available, prefer the target file only.
        """
        excerpts: List[str] = []
        relative_target = self._relative_to_workspace(filesystem, target_path)

        if relative_target and relative_target not in {".", ""}:
            try:
                if filesystem.file_exists(relative_target):
                    content = filesystem.read_file(relative_target)
                    excerpts.append(
                        self._format_file_excerpt(
                            relative_target, content, max_file_chars
                        )
                    )
                    return excerpts
            except Exception as exc:
                logger.warning(
                    "Could not read target file %s: %s", target_path, exc
                )

        if prefer_target_only:
            return excerpts

        try:
            files = filesystem.list_files(".", pattern="*.py", recursive=True)
        except Exception as exc:
            logger.warning("Could not list repository files: %s", exc)
            return excerpts

        for file_path in files[:_MAX_SOURCE_FILES]:
            try:
                content = filesystem.read_file(file_path)
            except Exception as exc:
                logger.warning("Skipping unreadable file %s: %s", file_path, exc)
                continue
            excerpts.append(
                self._format_file_excerpt(file_path, content, max_file_chars)
            )
        return excerpts

    @staticmethod
    def _repository_inventory(filesystem: FilesystemTools) -> List[str]:
        """
        List repository file paths so the model can describe layout.

        Only paths are collected, not contents, which keeps the prompt
        cheap while still grounding directory and module descriptions.
        """
        paths: List[str] = []
        seen: set[str] = set()
        for pattern in ("*.py", "*.md", "*.toml", "*.txt", "*.yml", "*.yaml", "*.cfg"):
            try:
                found = filesystem.list_files(".", pattern=pattern, recursive=True)
            except Exception as exc:
                logger.warning(
                    "Could not list %s files for the inventory: %s", pattern, exc
                )
                continue
            for path in found:
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
        paths.sort()
        return paths[:_MAX_INVENTORY_FILES]

    @staticmethod
    def _read_project_files(filesystem: FilesystemTools) -> List[str]:
        """
        Read project metadata files that reveal setup and run steps.

        Missing files are skipped silently: absence is normal and the
        prompt tells the model to say so rather than invent commands.
        """
        excerpts: List[str] = []
        for name in _PROJECT_FILES:
            try:
                if not filesystem.file_exists(name):
                    continue
                content = filesystem.read_file(name)
            except Exception as exc:
                logger.warning("Could not read project file %s: %s", name, exc)
                continue
            if not content.strip():
                continue
            excerpts.append(
                DocumentationAgent._format_file_excerpt(
                    name, content, _MAX_PROJECT_FILE_CHARS
                )
            )
        return excerpts

    @staticmethod
    def _relative_to_workspace(
        filesystem: FilesystemTools, target_path: str
    ) -> str:
        """Express target_path relative to the FilesystemTools workspace."""
        if not target_path or not str(target_path).strip():
            return "."
        absolute = os.path.abspath(os.path.expanduser(target_path))
        root = os.path.abspath(str(filesystem.workspace_root))
        try:
            common = os.path.commonpath([absolute, root])
        except ValueError:
            return os.path.basename(absolute) or "."
        if common != root:
            return os.path.basename(absolute) or "."
        relative = os.path.relpath(absolute, root).replace("\\", "/")
        return "." if relative in {".", ""} else relative

    @staticmethod
    def _format_file_excerpt(
        file_path: str, content: str, max_chars: int = _MAX_FILE_CHARS
    ) -> str:
        """Format one source file for inclusion in the prompt."""
        body = content if len(content) <= max_chars else content[:max_chars] + "\n..."
        return f"### {file_path}\n{body}"

    def _build_prompt(
        self,
        mode: str,
        instruction: str,
        target_path: str,
        function_name: str,
        chunks: Sequence[RetrievedChunk],
        source_excerpts: Sequence[str],
        inventory: Sequence[str] = (),
        project_files: Sequence[str] = (),
    ) -> str:
        """
        Build the user prompt for the documentation model call.

        Retrieved context is listed first and treated as primary evidence.
        Repository excerpts are secondary fillers. Inventory and project
        files ground layout, dependency, and run-instruction claims.
        """
        guidance = _MODE_GUIDANCE.get(mode, _MODE_GUIDANCE["docstring"])
        sections = [
            f"DOCUMENTATION MODE\n{mode}",
            f"WRITING INSTRUCTIONS\n{guidance}",
            f"REQUEST\n{instruction}",
            f"TARGET\n{target_path}",
        ]
        if function_name:
            sections.append(f"FUNCTION NAME\n{function_name}")

        if chunks:
            rendered = []
            for index, chunk in enumerate(chunks, start=1):
                source = getattr(chunk, "source", None) or (
                    chunk.metadata.get("file_path", "unknown")
                    if chunk.metadata
                    else "unknown"
                )
                body = chunk.content or ""
                if len(body) > _MAX_CHUNK_CHARS:
                    body = body[:_MAX_CHUNK_CHARS] + "\n..."
                rendered.append(
                    f"[{index}] source={source} score={float(chunk.score):.3f}\n{body}"
                )
            sections.append(
                "RETRIEVED CONTEXT (primary - prioritize this)\n"
                + "\n\n".join(rendered)
            )
        else:
            sections.append(
                "RETRIEVED CONTEXT (primary - prioritize this)\n(none)"
            )

        if inventory:
            sections.append(
                "REPOSITORY INVENTORY (every path that exists; describe "
                "only these)\n" + "\n".join(inventory)
            )

        if project_files:
            sections.append(
                "PROJECT FILES (dependencies and run instructions must come "
                "from here)\n" + "\n\n".join(project_files)
            )

        if source_excerpts:
            label = (
                "REPOSITORY CONTENTS (secondary - use only to fill gaps)"
                if chunks
                else "REPOSITORY CONTENTS (primary - no retrieved context)"
            )
            sections.append(f"{label}\n" + "\n\n".join(source_excerpts))
        else:
            sections.append("REPOSITORY CONTENTS\n(none)")

        sections.append(
            "Return only the DocumentationResult JSON object."
        )
        return "\n\n".join(sections)

    def _parse_response(
        self,
        content: str,
        default_file_path: str,
        default_function_name: str,
    ) -> DocumentationResult:
        """
        Parse model output into DocumentationResult.

        Returns an empty result when parsing fails rather than raising.
        """
        result, _error = self._parse_response_with_status(
            content,
            default_file_path=default_file_path,
            default_function_name=default_function_name,
        )
        return result

    def _parse_response_with_status(
        self,
        content: str,
        default_file_path: str,
        default_function_name: str,
    ) -> Tuple[DocumentationResult, Optional[str]]:
        """
        Parse model output and report whether a JSON object was recovered.

        Returns:
            ``(result, None)`` when a JSON object was parsed (summary may
            still be empty). ``(empty, error)`` when the response is
            malformed / unparseable.
        """
        empty = self._empty_result(default_file_path, default_function_name)
        if not content or not str(content).strip():
            return empty, "Model response was empty."

        payload, extract_error = self._extract_json_object_with_error(str(content))
        if payload is None:
            logger.warning("Documentation model response was not valid JSON.")
            return empty, extract_error or (
                "Documentation model response was not valid JSON."
            )

        parameters = payload.get("parameters") or []
        if not isinstance(parameters, list):
            parameters = []
        cleaned_params: List[Dict[str, Any]] = []
        for item in parameters:
            if isinstance(item, dict):
                cleaned_params.append(
                    {
                        "name": str(item.get("name", "")),
                        "type": str(item.get("type", "")),
                        "description": str(item.get("description", "")),
                    }
                )

        return (
            DocumentationResult(
                file_path=str(payload.get("file_path") or default_file_path or ""),
                function_name=str(
                    payload.get("function_name") or default_function_name or ""
                ),
                summary=str(payload.get("summary") or ""),
                parameters=cleaned_params,
                returns=str(payload.get("returns") or ""),
                example_usage=str(payload.get("example_usage") or ""),
            ),
            None,
        )

    @staticmethod
    def _extract_json_object(content: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object from a model response."""
        payload, _error = DocumentationAgent._extract_json_object_with_error(content)
        return payload

    @staticmethod
    def _extract_json_object_with_error(
        content: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Extract the first JSON object, returning a parse error on failure."""
        text = content.strip()
        fence = _FENCE.search(text)
        if fence:
            text = fence.group(1).strip()

        last_error = "No JSON object found in model response."
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data, ""
            return None, "JSON root value was not an object."
        except json.JSONDecodeError as exc:
            last_error = f"JSON decode error: {exc}"

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict):
                    return data, ""
                return None, "JSON root value was not an object."
            except json.JSONDecodeError as exc:
                return None, f"JSON decode error: {exc}"
        return None, last_error

    @staticmethod
    def _empty_result(
        file_path: str = "", function_name: str = ""
    ) -> DocumentationResult:
        """Build an empty DocumentationResult for failure paths."""
        return DocumentationResult(
            file_path=file_path or "",
            function_name=function_name or "",
            summary="",
            parameters=[],
            returns="",
            example_usage="",
            abstention=None,
        )

    @staticmethod
    def _abstain_result(
        empty: DocumentationResult,
        *,
        reason: str,
        evidence_available: Optional[List[str]] = None,
        recommended_next_steps: Optional[List[str]] = None,
    ) -> DocumentationResult:
        """Attach an AbstentionResult to an empty documentation payload."""
        abstention: AbstentionResult = ReportBuilder().abstain(
            reason,
            confidence=1.0,
            evidence_available=evidence_available,
            recommended_next_steps=recommended_next_steps,
        )
        return DocumentationResult(
            file_path=empty.file_path,
            function_name=empty.function_name,
            summary="",
            parameters=[],
            returns="",
            example_usage="",
            abstention=abstention,
        )

    @staticmethod
    def _workspace_for(path: str) -> str:
        """
        Resolve a workspace root for FilesystemTools.

        A directory path is used as-is. A file path uses its parent
        directory. Missing paths fall back to the current directory.
        """
        if not path or not str(path).strip():
            return os.path.abspath(".")
        absolute = os.path.abspath(os.path.expanduser(path))
        if os.path.isdir(absolute):
            return absolute
        if os.path.isfile(absolute):
            return os.path.dirname(absolute) or absolute
        parent = os.path.dirname(absolute)
        return parent if parent else os.path.abspath(".")
