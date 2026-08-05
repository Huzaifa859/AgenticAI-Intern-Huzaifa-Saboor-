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

Optional write-back (``write_to_disk=False`` by default) can persist a
generated README.md or insert missing docstrings through FilesystemTools
without changing the returned DocumentationResult schema.

Optional targeting via ``file_path`` / ``function_name`` / ``class_name``
scopes retrieval and prompting to a single file or symbol. When those
fields are absent, repository-wide README behaviour is unchanged.

Public symbols are documented incrementally (one LLM call per symbol)
and merged into a single DocumentationResult, reducing context size and
hallucinations versus a single repository-wide generation call.
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

#: Soft cap on public symbols documented in one pipeline run.
_MAX_SYMBOLS_TO_DOCUMENT = 20

#: Cap on Python files scanned during the AST inventory pass.
_MAX_AST_FILES = 40

#: Directory names excluded from documentation symbol inventory.
_SKIP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        "tests",
        "test",
        ".git",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "site-packages",
        "node_modules",
        "generated",
        "migrations",
        "alembic",
    }
)

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


@dataclass
class _DocumentationTarget:
    """Resolved documentation scope for one pipeline run."""

    scope: str  # repository | file | function | class
    relative_file: str = ""
    function_name: str = ""
    class_name: str = ""
    symbol_source: str = ""
    nearby_source: str = ""
    focus_instruction: str = ""
    not_found_reason: str = ""
    searched_location: str = ""

    @property
    def found(self) -> bool:
        return not bool(self.not_found_reason)


@dataclass
class _DocumentableSymbol:
    """A public function, class, or method selected for focused docs."""

    kind: str  # function | class | method
    name: str
    qualname: str
    module_path: str
    source: str
    nearby_source: str = ""
    parent_class: str = ""

    @property
    def focus_instruction(self) -> str:
        """Human-readable focus line used in the symbol prompt."""
        module = self.module_path or "module"
        if self.kind == "class":
            return f"Document class {self.name}."
        if self.kind == "method":
            owner = self.parent_class or "Class"
            return f"Document method {owner}.{self.name}()."
        return f"Document function {self.name}()."

    @property
    def result_name(self) -> str:
        """Name stored on DocumentationResult.function_name."""
        if self.kind == "method" and self.parent_class:
            return f"{self.parent_class}.{self.name}"
        return self.qualname or self.name


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
        class_name = str(context.get("class_name") or "")
        doc_type = str(context.get("doc_type") or "").lower()
        write_to_disk = self._context_flag(context, "write_to_disk", False)
        replace_existing = self._context_flag(context, "replace_existing", False)

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
                result = self.generate_readme(
                    repo_path,
                    write_to_disk=write_to_disk,
                    replace_existing=replace_existing,
                )
            elif doc_type in {"api", "api_reference"} or "api reference" in lowered:
                result = self.generate_api_reference(
                    file_path or repo_path,
                    write_to_disk=write_to_disk,
                    replace_existing=replace_existing,
                )
            elif doc_type in {"module", "module_summary"} or "module summary" in lowered:
                target = file_path or repo_path
                workspace = self._workspace_for(target)
                result = self._run_pipeline(
                    mode="module",
                    workspace=workspace,
                    target_path=target,
                    instruction=(
                        instruction
                        or (
                            f"Generate documentation only for class {class_name} "
                            f"inside {target}."
                            if class_name
                            else f"Generate documentation only for {target}."
                            if file_path
                            else f"Summarize the purpose and public surface of module {target}."
                        )
                    ),
                    function_name=(
                        class_name
                        or function_name
                        or os.path.basename(target)
                        or "module"
                    ),
                    class_name=class_name,
                    write_to_disk=write_to_disk,
                    replace_existing=replace_existing,
                )
            else:
                target = file_path or repo_path
                workspace = self._workspace_for(target)
                if class_name and not function_name:
                    doc_instruction = instruction or (
                        f"Generate documentation only for class {class_name} "
                        f"inside {target}."
                    )
                    result_name = class_name
                elif function_name:
                    doc_instruction = instruction or (
                        f"Generate documentation only for function "
                        f"{function_name} inside {target}."
                    )
                    result_name = function_name
                elif file_path:
                    doc_instruction = instruction or (
                        f"Generate documentation only for {target}."
                    )
                    result_name = os.path.basename(target) or "module"
                else:
                    doc_instruction = instruction or (
                        f"Document the primary public function in {target}."
                    )
                    result_name = function_name
                result = self._run_pipeline(
                    mode="docstring",
                    workspace=workspace,
                    target_path=target,
                    instruction=doc_instruction,
                    function_name=result_name,
                    class_name=class_name,
                    write_to_disk=write_to_disk,
                    replace_existing=replace_existing,
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

    def generate_docstring(
        self,
        file_path: str,
        *,
        write_to_disk: bool = False,
        replace_existing: bool = False,
    ) -> DocumentationResult:
        """
        Generate a docstring for a function in a given file.

        Args:
            file_path: Path to the file containing the function.
            write_to_disk: When True, insert missing docstrings via
                FilesystemTools after a successful generation.
            replace_existing: When True with write-back, replace existing
                docstrings; otherwise leave them untouched.

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
            write_to_disk=write_to_disk,
            replace_existing=replace_existing,
        )

    def generate_readme(
        self,
        repo_path: str,
        *,
        write_to_disk: bool = False,
        replace_existing: bool = False,
    ) -> DocumentationResult:
        """
        Generate a README file summarizing the repository.

        Args:
            repo_path: Path to the repository root.
            write_to_disk: When True, write README.md through
                FilesystemTools after a successful generation.
            replace_existing: When True with write-back, overwrite an
                existing README.md; otherwise leave it untouched.

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
            write_to_disk=write_to_disk,
            replace_existing=replace_existing,
        )

    def generate_api_reference(
        self,
        module_path: str,
        *,
        write_to_disk: bool = False,
        replace_existing: bool = False,
    ) -> DocumentationResult:
        """
        Generate API reference documentation for a module.

        Args:
            module_path: Path to the module to document.
            write_to_disk: When True, insert missing docstrings via
                FilesystemTools after a successful generation.
            replace_existing: When True with write-back, replace existing
                docstrings; otherwise leave them untouched.

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
            write_to_disk=write_to_disk,
            replace_existing=replace_existing,
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
        class_name: str = "",
        write_to_disk: bool = False,
        replace_existing: bool = False,
    ) -> DocumentationResult:
        """
        Run the documentation pipeline for one request.

        Stages: resolve target → AST inventory → per-symbol generation
        (retrieve → prompt → generate → JSON repair → ground) → merge →
        optional write-back. Falls back to a single repository-wide call
        when no public symbols exist but project metadata is available.
        """
        empty = self._empty_result(
            file_path=target_path,
            function_name=function_name or class_name,
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

        filesystem = self._filesystem_tools(workspace)
        doc_target = self._resolve_documentation_target(
            filesystem=filesystem,
            mode=mode,
            workspace=workspace,
            target_path=target_path,
            function_name=function_name,
            class_name=class_name,
        )
        if not doc_target.found:
            self._trace(
                "documentation_target_not_found",
                file_path=doc_target.relative_file or target_path,
                function_name=function_name,
                class_name=class_name,
                reason=doc_target.not_found_reason,
                searched_location=doc_target.searched_location,
            )
            abstained = self._abstain_result(
                empty,
                reason=doc_target.not_found_reason,
                evidence_available=[
                    item
                    for item in (
                        f"searched location: {doc_target.searched_location}"
                        if doc_target.searched_location
                        else "",
                        f"file_path={target_path}" if target_path else "",
                        f"function_name={function_name}" if function_name else "",
                        f"class_name={class_name}" if class_name else "",
                    )
                    if item
                ],
                recommended_next_steps=[
                    "Confirm the file path exists inside the selected repository.",
                    "Use an exact public function or class name from that file.",
                    "Omit function_name/class_name to document the whole file.",
                ],
            )
            self._trace(
                "documentation_finished",
                success=False,
                abstained=True,
                reason=abstained.abstention.reason if abstained.abstention else "",
            )
            return abstained

        self._trace(
            "documentation_target_selected",
            scope=doc_target.scope,
            file_path=doc_target.relative_file,
            function_name=doc_target.function_name,
            class_name=doc_target.class_name,
        )

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

        inventory = self._repository_inventory(filesystem)
        symbols = self._discover_documentable_symbols(filesystem, doc_target)
        selected = symbols[:_MAX_SYMBOLS_TO_DOCUMENT]
        self._trace(
            "documentation_ast_scan_finished",
            scope=doc_target.scope,
            symbols_discovered=len(symbols),
            symbols_selected=len(selected),
            inventory=len(inventory),
        )

        if not selected:
            # Preserve README-style generation when only project metadata exists.
            if doc_target.scope == "repository":
                return self._run_repository_fallback(
                    mode=mode,
                    workspace=workspace,
                    target_path=target_path,
                    instruction=instruction,
                    function_name=function_name,
                    class_name=class_name,
                    doc_target=doc_target,
                    filesystem=filesystem,
                    inventory=inventory,
                    empty=empty,
                    write_to_disk=write_to_disk,
                    replace_existing=replace_existing,
                )
            abstained = self._abstain_result(
                empty,
                reason="No public symbols were found to document.",
                evidence_available=[
                    f"{len(inventory)} inventoried path(s)",
                    f"scope={doc_target.scope}",
                ],
                recommended_next_steps=[
                    "Point documentation at a module with public functions or classes.",
                    "Confirm private-only modules are not the only targets.",
                ],
            )
            self._trace(
                "documentation_finished",
                success=False,
                abstained=True,
                reason=abstained.abstention.reason if abstained.abstention else "",
            )
            return abstained

        partial_results: List[DocumentationResult] = []
        warnings: List[str] = []
        verified_total = 0
        removed_total = 0

        for symbol in selected:
            self._trace(
                "documentation_symbol_started",
                symbol=symbol.qualname,
                module=symbol.module_path,
                kind=symbol.kind,
            )
            symbol_started = time.perf_counter()
            try:
                symbol_result, grounded_ok, v_count, r_count = self._document_one_symbol(
                    symbol=symbol,
                    mode=mode,
                    instruction=instruction,
                    filesystem=filesystem,
                    inventory=inventory,
                    target_path=target_path,
                )
            except Exception as exc:
                logger.warning(
                    "Documentation failed for symbol %s: %s", symbol.qualname, exc
                )
                self._trace(
                    "documentation_symbol_failed",
                    symbol=symbol.qualname,
                    module=symbol.module_path,
                    error=str(exc),
                    duration_ms=(time.perf_counter() - symbol_started) * 1000.0,
                )
                warnings.append(f"{symbol.qualname}: {exc}")
                continue

            duration_ms = (time.perf_counter() - symbol_started) * 1000.0
            if (
                grounded_ok
                and symbol_result.summary
                and symbol_result.summary.strip()
                and symbol_result.abstention is None
            ):
                partial_results.append(symbol_result)
                verified_total += v_count
                removed_total += r_count
                self._trace(
                    "documentation_symbol_finished",
                    symbol=symbol.qualname,
                    module=symbol.module_path,
                    duration_ms=duration_ms,
                    success=True,
                )
            else:
                reason = (
                    symbol_result.abstention.reason
                    if symbol_result.abstention is not None
                    else "empty or ungrounded documentation"
                )
                warnings.append(f"{symbol.qualname}: {reason}")
                self._trace(
                    "documentation_symbol_failed",
                    symbol=symbol.qualname,
                    module=symbol.module_path,
                    reason=reason,
                    duration_ms=duration_ms,
                )

        if not partial_results:
            # Prefer concrete failure reasons when they are uniform / singular
            # so existing abstention assertions remain stable.
            reason = "Documentation failed for every discovered public symbol."
            if warnings and all(
                "could not be grounded" in item.lower() for item in warnings
            ):
                reason = (
                    "Documentation claims could not be grounded in the repository."
                )
                self._trace(
                    "documentation_grounding_abstained",
                    symbols=len(selected),
                    warnings=len(warnings),
                )
            elif len(selected) == 1 and warnings:
                detail = warnings[0].split(": ", 1)
                if len(detail) == 2 and detail[1].strip():
                    reason = detail[1].strip()
                if "could not be grounded" in reason.lower():
                    self._trace(
                        "documentation_grounding_abstained",
                        symbols=1,
                        warnings=len(warnings),
                    )
            abstained = self._abstain_result(
                empty,
                reason=reason,
                evidence_available=[
                    f"{len(selected)} symbol(s) attempted",
                    *warnings[:5],
                ],
                recommended_next_steps=[
                    "Retry with a narrower documentation target.",
                    "Confirm the model returned valid DocumentationResult JSON.",
                ],
            )
            self._trace(
                "documentation_finished",
                success=False,
                abstained=True,
                reason=abstained.abstention.reason if abstained.abstention else "",
                warnings=len(warnings),
            )
            return abstained

        self._trace(
            "documentation_merge_started",
            symbols=len(partial_results),
            warnings=len(warnings),
        )
        merge_started = time.perf_counter()
        result = self._merge_documentation_results(
            partial_results,
            mode=mode,
            doc_target=doc_target,
            warnings=warnings,
        )
        self._trace(
            "documentation_merge_finished",
            success=True,
            duration_ms=(time.perf_counter() - merge_started) * 1000.0,
            symbols=len(partial_results),
            summary_chars=len(result.summary or ""),
            parameters=len(result.parameters or []),
        )

        self._trace(
            "documentation_target_grounded",
            scope=doc_target.scope,
            file_path=doc_target.relative_file or result.file_path,
            function_name=doc_target.function_name or result.function_name,
            class_name=doc_target.class_name,
            verified=verified_total,
            removed=removed_total,
        )

        if write_to_disk:
            result = self._maybe_write_to_disk(
                result,
                mode=mode,
                workspace=workspace,
                target_path=doc_target.relative_file or target_path,
                replace_existing=replace_existing,
            )

        logger.info(
            "Documentation generated for %s symbol(s) (%s warning(s)).",
            len(partial_results),
            len(warnings),
        )
        self._trace(
            "documentation_finished",
            success=True,
            function_name=result.function_name,
            summary_chars=len(result.summary or ""),
            grounded_verified=verified_total,
            grounded_removed=removed_total,
            write_to_disk=write_to_disk,
            scope=doc_target.scope,
            symbols=len(partial_results),
            warnings=len(warnings),
        )
        return result

    # ------------------------------------------------------------------
    # Per-symbol generation + merge
    # ------------------------------------------------------------------

    def _run_repository_fallback(
        self,
        *,
        mode: str,
        workspace: str,
        target_path: str,
        instruction: str,
        function_name: str,
        class_name: str,
        doc_target: _DocumentationTarget,
        filesystem: FilesystemTools,
        inventory: Sequence[str],
        empty: DocumentationResult,
        write_to_disk: bool,
        replace_existing: bool,
    ) -> DocumentationResult:
        """Single-call README path when no public symbols are available."""
        project_files = self._read_project_files(filesystem)
        source_excerpts = self._read_repository_sources(
            filesystem,
            target_path,
            max_file_chars=_MAX_FILE_CHARS,
            prefer_target_only=False,
        )
        if not source_excerpts and not project_files and not inventory:
            abstained = self._abstain_result(
                empty,
                reason="Repository contains no supported Python files.",
                evidence_available=[],
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

        chunks = self._retrieve_context(
            " ".join(
                part
                for part in (
                    instruction,
                    doc_target.focus_instruction,
                    "repository documentation",
                    mode,
                )
                if part
            ),
            target_path,
        )
        prompt = self._build_prompt(
            mode=mode,
            instruction=instruction,
            target_path=target_path,
            function_name=function_name or "README",
            class_name=class_name,
            focus_instruction=doc_target.focus_instruction
            or "Generate repository documentation.",
            chunks=chunks,
            source_excerpts=source_excerpts,
            inventory=list(inventory),
            project_files=project_files,
        )
        result = self._generate_from_prompt(
            prompt=prompt,
            default_file_path=target_path,
            default_function_name=function_name or "README",
        )
        if not (result.summary and result.summary.strip()):
            result = self._abstain_result(
                empty,
                reason="LLM response could not be verified.",
                evidence_available=[f"{len(inventory)} inventoried path(s)"],
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

        result, grounding_stats, grounding_abstained = self._ground_documentation(
            result,
            filesystem=filesystem,
            inventory=inventory,
            mode=mode,
            target_path=target_path,
        )
        if grounding_abstained or not (result.summary and result.summary.strip()):
            result = self._abstain_result(
                empty,
                reason="Documentation claims could not be grounded in the repository.",
                evidence_available=[f"{len(inventory)} inventoried path(s)"],
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

        self._trace(
            "documentation_target_grounded",
            scope=doc_target.scope,
            file_path=result.file_path,
            function_name=result.function_name,
            class_name=class_name,
            verified=len(grounding_stats.verified),
            removed=len(grounding_stats.removed),
        )
        if write_to_disk:
            result = self._maybe_write_to_disk(
                result,
                mode=mode,
                workspace=workspace,
                target_path=target_path,
                replace_existing=replace_existing,
            )
        self._trace(
            "documentation_finished",
            success=True,
            function_name=result.function_name,
            summary_chars=len(result.summary or ""),
            grounded_verified=len(grounding_stats.verified),
            grounded_removed=len(grounding_stats.removed),
            write_to_disk=write_to_disk,
            scope=doc_target.scope,
            fallback="repository_no_symbols",
        )
        return result

    def _document_one_symbol(
        self,
        *,
        symbol: _DocumentableSymbol,
        mode: str,
        instruction: str,
        filesystem: FilesystemTools,
        inventory: Sequence[str],
        target_path: str,
    ) -> Tuple[DocumentationResult, bool, int, int]:
        """
        Generate, repair, and ground documentation for one symbol.

        Returns ``(result, grounded_ok, verified_count, removed_count)``.
        """
        query = " ".join(
            part
            for part in (
                instruction,
                symbol.focus_instruction,
                symbol.qualname,
                symbol.module_path,
                mode,
            )
            if part
        )
        chunks = self._retrieve_context(
            query,
            symbol.module_path,
            source_file=symbol.module_path,
        )
        source_excerpts = self._build_symbol_excerpts_from_symbol(symbol)
        focus = symbol.focus_instruction
        if mode == "readme":
            focus = (
                f"{symbol.focus_instruction} "
                "This documentation will be merged into repository README docs."
            )
        prompt = self._build_prompt(
            mode=mode,
            instruction=instruction or symbol.focus_instruction,
            target_path=symbol.module_path,
            function_name=symbol.result_name,
            class_name=symbol.name if symbol.kind == "class" else symbol.parent_class,
            focus_instruction=focus,
            chunks=chunks,
            source_excerpts=source_excerpts,
            inventory=[],
            project_files=[],
        )

        result = self._generate_from_prompt(
            prompt=prompt,
            default_file_path=symbol.module_path,
            default_function_name=symbol.result_name,
        )
        if not (result.summary and result.summary.strip()):
            empty = self._empty_result(
                file_path=symbol.module_path,
                function_name=symbol.result_name,
            )
            return (
                self._abstain_result(
                    empty,
                    reason="LLM response could not be verified.",
                    evidence_available=[f"symbol={symbol.qualname}"],
                    recommended_next_steps=[
                        "Retry with a narrower documentation target.",
                        "Confirm the model returned valid DocumentationResult JSON.",
                    ],
                ),
                False,
                0,
                0,
            )

        grounded, stats, abstained = self._ground_documentation(
            result,
            filesystem=filesystem,
            inventory=inventory,
            mode=mode,
            target_path=symbol.module_path or target_path,
        )
        if abstained or not (grounded.summary and grounded.summary.strip()):
            self._trace(
                "documentation_grounding_abstained",
                removed=len(stats.removed),
                unsupported=len(stats.unsupported),
                verified=len(stats.verified),
                symbol=symbol.qualname,
            )
            empty = self._empty_result(
                file_path=symbol.module_path,
                function_name=symbol.result_name,
            )
            return (
                self._abstain_result(
                    empty,
                    reason="Documentation claims could not be grounded in the repository.",
                    evidence_available=[
                        f"symbol={symbol.qualname}",
                        f"{len(stats.removed)} unsupported claim(s) removed",
                    ],
                    recommended_next_steps=[
                        "Retry with a narrower documentation target.",
                        "Ensure generated documentation only cites inventory paths and symbols.",
                    ],
                ),
                False,
                len(stats.verified),
                len(stats.removed),
            )
        return grounded, True, len(stats.verified), len(stats.removed)

    def _generate_from_prompt(
        self,
        *,
        prompt: str,
        default_file_path: str,
        default_function_name: str,
    ) -> DocumentationResult:
        """Call the model once (plus optional JSON repair) for a prompt."""
        self._trace(
            "model_request",
            event_type=TraceEventType.MODEL_CALL,
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
            return self._empty_result(
                file_path=default_file_path,
                function_name=default_function_name,
            )

        self._trace(
            "model_response",
            event_type=TraceEventType.MODEL_CALL,
            success=True,
            duration_ms=(time.perf_counter() - model_started) * 1000.0,
            content_chars=len(response.content or ""),
        )
        result, parse_error = self._parse_response_with_status(
            response.content,
            default_file_path=default_file_path,
            default_function_name=default_function_name,
        )
        if parse_error is not None:
            result = self._retry_json_repair(
                original_prompt=prompt,
                raw_output=response.content or "",
                parse_error=parse_error,
                default_file_path=default_file_path,
                default_function_name=default_function_name,
            )
        return result

    def _discover_documentable_symbols(
        self,
        filesystem: FilesystemTools,
        doc_target: _DocumentationTarget,
    ) -> List[_DocumentableSymbol]:
        """Discover public symbols for the resolved documentation target."""
        if doc_target.scope in {"function", "class"} and doc_target.symbol_source:
            return [self._symbol_from_resolved_target(doc_target)]

        if doc_target.scope == "file" and doc_target.relative_file:
            return self._collect_symbols_from_module(
                filesystem, doc_target.relative_file
            )

        if doc_target.scope in {"function", "class"} and doc_target.relative_file:
            # Target resolved but source missing — scan file and filter.
            symbols = self._collect_symbols_from_module(
                filesystem, doc_target.relative_file
            )
            wanted = (
                doc_target.class_name
                if doc_target.scope == "class"
                else doc_target.function_name
            )
            if wanted:
                filtered = [
                    symbol
                    for symbol in symbols
                    if symbol.name == wanted or symbol.qualname == wanted
                ]
                if filtered:
                    return filtered
            return symbols

        inventory = self._list_python_inventory_for_docs(filesystem, "")
        symbols: List[_DocumentableSymbol] = []
        for module_path in inventory:
            symbols.extend(self._collect_symbols_from_module(filesystem, module_path))
            if len(symbols) >= _MAX_SYMBOLS_TO_DOCUMENT:
                break
        return symbols

    def _symbol_from_resolved_target(
        self, doc_target: _DocumentationTarget
    ) -> _DocumentableSymbol:
        """Build one documentable symbol from a resolved function/class target."""
        if doc_target.scope == "class":
            name = doc_target.class_name or "Class"
            return _DocumentableSymbol(
                kind="class",
                name=name,
                qualname=name,
                module_path=doc_target.relative_file,
                source=doc_target.symbol_source,
                nearby_source=doc_target.nearby_source,
            )
        name = doc_target.function_name or "function"
        kind = "method" if "." in name else "function"
        parent = name.split(".", 1)[0] if kind == "method" else ""
        short = name.split(".")[-1]
        return _DocumentableSymbol(
            kind=kind,
            name=short,
            qualname=name,
            module_path=doc_target.relative_file,
            source=doc_target.symbol_source,
            nearby_source=doc_target.nearby_source,
            parent_class=parent,
        )

    def _list_python_inventory_for_docs(
        self,
        filesystem: FilesystemTools,
        target_file: str,
    ) -> List[str]:
        """List inventory Python modules eligible for documentation."""
        if target_file and target_file.endswith(".py"):
            normalized = self._normalize_repo_path(target_file)
            if normalized and not self._should_skip_doc_path(normalized):
                try:
                    if filesystem.file_exists(normalized):
                        return [normalized]
                except Exception:
                    pass

        try:
            files = filesystem.list_files(".", pattern="*.py", recursive=True)
        except Exception as exc:
            logger.warning("Could not list Python inventory for docs: %s", exc)
            return []

        inventory: List[str] = []
        for path in files:
            normalized = self._normalize_repo_path(str(path))
            if not normalized or self._should_skip_doc_path(normalized):
                continue
            inventory.append(normalized)
            if len(inventory) >= _MAX_AST_FILES:
                break
        return inventory

    @staticmethod
    def _should_skip_doc_path(path: str) -> bool:
        """Return True for tests/, caches, generated modules, and migrations."""
        normalized = (path or "").replace("\\", "/").strip()
        if not normalized.endswith(".py"):
            return True
        parts = [part for part in normalized.split("/") if part]
        if any(part in _SKIP_DIR_NAMES for part in parts[:-1]):
            return True
        filename = parts[-1] if parts else normalized
        if filename.startswith("test_") or filename.endswith("_test.py"):
            return True
        if filename.endswith("_pb2.py") or filename.endswith("_pb2_grpc.py"):
            return True
        return False

    def _collect_symbols_from_module(
        self,
        filesystem: FilesystemTools,
        module_path: str,
    ) -> List[_DocumentableSymbol]:
        """Parse one module and collect public functions, classes, and methods."""
        try:
            source = filesystem.read_file(module_path)
        except Exception as exc:
            logger.warning("AST scan skipped %s: %s", module_path, exc)
            return []
        try:
            tree = ast.parse(source or "", filename=module_path)
        except SyntaxError:
            return []

        lines = (source or "").splitlines()
        nearby = self._module_import_source(tree, lines)
        symbols: List[_DocumentableSymbol] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_private_name(node.name) or self._is_dunder_name(node.name):
                    continue
                symbols.append(
                    _DocumentableSymbol(
                        kind="function",
                        name=node.name,
                        qualname=node.name,
                        module_path=module_path,
                        source=self._slice_source(lines, node),
                        nearby_source=nearby,
                    )
                )
                continue

            if isinstance(node, ast.ClassDef):
                if self._is_private_name(node.name) or self._is_dunder_name(node.name):
                    continue
                symbols.append(
                    _DocumentableSymbol(
                        kind="class",
                        name=node.name,
                        qualname=node.name,
                        module_path=module_path,
                        source=self._slice_source(lines, node),
                        nearby_source=nearby,
                    )
                )
                for member in node.body:
                    if not isinstance(
                        member, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        continue
                    if self._is_private_name(member.name) or self._is_dunder_name(
                        member.name
                    ):
                        continue
                    symbols.append(
                        _DocumentableSymbol(
                            kind="method",
                            name=member.name,
                            qualname=f"{node.name}.{member.name}",
                            module_path=module_path,
                            source=self._slice_source(lines, member),
                            nearby_source=nearby,
                            parent_class=node.name,
                        )
                    )
        return symbols

    @staticmethod
    def _module_import_source(tree: ast.AST, lines: Sequence[str]) -> str:
        """Collect top-level import lines as nearby context."""
        import_lines: List[str] = []
        for item in getattr(tree, "body", []):
            if isinstance(item, (ast.Import, ast.ImportFrom)):
                start = max(0, int(getattr(item, "lineno", 1) or 1) - 1)
                end = int(getattr(item, "end_lineno", None) or (start + 1))
                import_lines.extend(lines[start:end])
        return "\n".join(import_lines).strip()

    @staticmethod
    def _slice_source(lines: Sequence[str], node: ast.AST) -> str:
        """Return the source text spanning an AST node."""
        start = max(0, int(getattr(node, "lineno", 1) or 1) - 1)
        end = int(
            getattr(node, "end_lineno", None) or getattr(node, "lineno", start + 1)
        )
        end = max(start + 1, end)
        return "\n".join(lines[start:end])

    @staticmethod
    def _is_private_name(name: str) -> bool:
        """True for single-underscore private names (not dunders)."""
        return bool(name) and name.startswith("_") and not (
            name.startswith("__") and name.endswith("__")
        )

    @staticmethod
    def _is_dunder_name(name: str) -> bool:
        """True for ``__init__``-style dunder names."""
        return (
            bool(name)
            and name.startswith("__")
            and name.endswith("__")
            and len(name) > 4
        )

    @staticmethod
    def _build_symbol_excerpts_from_symbol(
        symbol: _DocumentableSymbol,
    ) -> List[str]:
        """Render focused symbol + nearby excerpts for the prompt."""
        excerpts: List[str] = []
        label = symbol.module_path or "target.py"
        if symbol.nearby_source.strip():
            excerpts.append(
                f"### {label} (nearby context)\n{symbol.nearby_source.strip()}"
            )
        if symbol.source.strip():
            excerpts.append(
                f"### {label} ({symbol.kind} {symbol.qualname})\n"
                f"{symbol.source.strip()}"
            )
        return excerpts

    def _merge_documentation_results(
        self,
        results: Sequence[DocumentationResult],
        *,
        mode: str,
        doc_target: _DocumentationTarget,
        warnings: Sequence[str],
    ) -> DocumentationResult:
        """Merge per-symbol DocumentationResult objects into one result."""
        if not results:
            return self._empty_result(
                file_path=doc_target.relative_file,
                function_name=doc_target.function_name or doc_target.class_name,
            )
        # Single targeted symbol docs can pass through unchanged.
        if (
            len(results) == 1
            and not warnings
            and mode != "readme"
            and doc_target.scope in {"function", "class"}
        ):
            return results[0]

        summary_blocks: List[str] = []
        seen_summaries: Set[str] = set()
        parameters: List[Dict[str, Any]] = []
        seen_params: Set[str] = set()
        returns_parts: List[str] = []
        seen_returns: Set[str] = set()
        example_parts: List[str] = []
        seen_examples: Set[str] = set()
        file_paths: List[str] = []

        for result in results:
            path = (result.file_path or "").strip()
            if path and path not in file_paths:
                file_paths.append(path)
            name = (result.function_name or "").strip() or "symbol"
            summary = (result.summary or "").strip()
            if summary:
                key = re.sub(r"\s+", " ", summary.lower())
                if key not in seen_summaries:
                    seen_summaries.add(key)
                    if len(results) == 1 and mode != "readme":
                        summary_blocks.append(summary)
                    else:
                        summary_blocks.append(f"## {name}\n\n{summary}")
            for item in result.parameters or []:
                if not isinstance(item, dict):
                    continue
                pname = str(item.get("name", "") or "").strip()
                if not pname:
                    continue
                pkey = f"{pname}|{item.get('type', '')}|{item.get('description', '')}"
                if pkey in seen_params:
                    continue
                seen_params.add(pkey)
                parameters.append(
                    {
                        "name": pname,
                        "type": str(item.get("type", "")),
                        "description": str(item.get("description", "")),
                    }
                )
            returns = (result.returns or "").strip()
            if returns:
                rkey = re.sub(r"\s+", " ", returns.lower())
                if rkey not in seen_returns:
                    seen_returns.add(rkey)
                    label = name if len(results) > 1 else ""
                    returns_parts.append(f"{label}: {returns}" if label else returns)
            example = (result.example_usage or "").strip()
            if example:
                ekey = re.sub(r"\s+", " ", example.lower())
                if ekey not in seen_examples:
                    seen_examples.add(ekey)
                    example_parts.append(example)

        if mode == "readme":
            function_name = "README"
            summary = "# Repository Documentation\n\n" + "\n\n".join(summary_blocks)
        elif doc_target.scope == "file":
            function_name = os.path.basename(doc_target.relative_file) or "module"
            summary = "\n\n".join(summary_blocks)
        elif len(results) == 1:
            function_name = results[0].function_name
            summary = summary_blocks[0] if summary_blocks else results[0].summary
            parameters = list(results[0].parameters or [])
            returns_parts = (
                [(results[0].returns or "").strip()]
                if (results[0].returns or "").strip()
                else []
            )
            example_parts = (
                [(results[0].example_usage or "").strip()]
                if (results[0].example_usage or "").strip()
                else []
            )
        else:
            function_name = (
                doc_target.function_name
                or doc_target.class_name
                or results[0].function_name
            )
            summary = "\n\n".join(summary_blocks)

        if warnings:
            warning_lines = "\n".join(f"- {item}" for item in warnings[:10])
            summary = (
                f"{summary.rstrip()}\n\n## Generation Warnings\n\n{warning_lines}"
            )

        file_path = doc_target.relative_file or (
            file_paths[0] if file_paths else ""
        )
        return DocumentationResult(
            file_path=file_path,
            function_name=function_name,
            summary=summary.strip(),
            parameters=parameters,
            returns="\n".join(returns_parts),
            example_usage="\n\n".join(example_parts),
            abstention=None,
        )

    # ------------------------------------------------------------------
    # Optional write-back (FilesystemTools)
    # ------------------------------------------------------------------

    @staticmethod
    def _context_flag(context: Dict[str, Any], key: str, default: bool) -> bool:
        """Parse a boolean flag from request context with a safe default."""
        if key not in context:
            return default
        value = context.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return default

    def _maybe_write_to_disk(
        self,
        result: DocumentationResult,
        *,
        mode: str,
        workspace: str,
        target_path: str,
        replace_existing: bool,
    ) -> DocumentationResult:
        """
        Persist generated documentation when write-back is enabled.

        Never raises. On failure, returns the original result with a short
        warning appended to ``summary``.
        """
        if not (result.summary and result.summary.strip()):
            return result

        self._trace(
            "documentation_write_started",
            mode=mode,
            workspace=workspace,
            target_path=target_path,
            replace_existing=replace_existing,
        )
        started = time.perf_counter()
        try:
            filesystem = self._filesystem_tools(workspace)
            if mode == "readme":
                note = self._write_readme_file(
                    filesystem,
                    result.summary,
                    replace_existing=replace_existing,
                )
            else:
                note = self._write_docstrings_to_file(
                    filesystem,
                    result=result,
                    target_path=target_path,
                    replace_existing=replace_existing,
                )
            self._trace(
                "documentation_write_finished",
                success=True,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                mode=mode,
                note=note,
            )
            if note:
                return DocumentationResult(
                    file_path=result.file_path,
                    function_name=result.function_name,
                    summary=self._append_summary_note(result.summary, note),
                    parameters=result.parameters,
                    returns=result.returns,
                    example_usage=result.example_usage,
                    abstention=result.abstention,
                )
            return result
        except Exception as exc:
            logger.warning("Documentation write-back failed: %s", exc)
            warning = f"Write-back warning: {exc}"
            self._trace(
                "documentation_write_failed",
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
            return DocumentationResult(
                file_path=result.file_path,
                function_name=result.function_name,
                summary=self._append_summary_note(result.summary, warning),
                parameters=result.parameters,
                returns=result.returns,
                example_usage=result.example_usage,
                abstention=result.abstention,
            )

    def _write_readme_file(
        self,
        filesystem: FilesystemTools,
        content: str,
        *,
        replace_existing: bool,
    ) -> str:
        """Write README.md at the workspace root; skip when preserved."""
        path = "README.md"
        exists = False
        try:
            exists = filesystem.file_exists(path)
        except Exception:
            exists = False

        if exists and not replace_existing:
            return "Write-back skipped: README.md already exists."

        body = content if content.endswith("\n") else content + "\n"
        filesystem.write_file(path, body)
        action = "replaced" if exists else "wrote"
        return f"Write-back {action} README.md."

    def _write_docstrings_to_file(
        self,
        filesystem: FilesystemTools,
        *,
        result: DocumentationResult,
        target_path: str,
        replace_existing: bool,
    ) -> str:
        """Insert or replace docstrings in the target Python file."""
        relative = self._relative_to_workspace(filesystem, target_path)
        if not relative or relative in {".", ""}:
            candidate = self._normalize_repo_path(result.file_path)
            relative = candidate if candidate.endswith(".py") else ""
        if not relative.endswith(".py"):
            return "Write-back skipped: no Python target file."

        try:
            if not filesystem.file_exists(relative):
                return f"Write-back skipped: target file not found ({relative})."
            source = filesystem.read_file(relative)
        except Exception as exc:
            raise RuntimeError(f"could not read {relative}: {exc}") from exc

        docstring_text = self._format_docstring_text(result)
        symbol_name = (result.function_name or "").strip()
        if symbol_name in _SPECIAL_FUNCTION_NAMES or symbol_name.lower() == "readme":
            # Module-level documentation: fill the module docstring when absent.
            updated, status = self._upsert_module_docstring(
                source,
                docstring_text,
                replace_existing=replace_existing,
            )
        else:
            updated, status = self._upsert_symbol_docstring(
                source,
                symbol_name=symbol_name,
                docstring_text=docstring_text,
                replace_existing=replace_existing,
            )

        if status.startswith("skipped"):
            return f"Write-back {status.replace('_', ' ')} for {relative}."
        if updated == source:
            return f"Write-back skipped: no docstring changes for {relative}."

        filesystem.write_file(relative, updated)
        return f"Write-back {status} docstring in {relative}."

    @staticmethod
    def _append_summary_note(summary: str, note: str) -> str:
        """Append a short write-back note to the documentation summary."""
        base = (summary or "").rstrip()
        text = (note or "").strip()
        if not text:
            return summary
        if not base:
            return text
        return f"{base}\n\n{text}"

    @staticmethod
    def _format_docstring_text(result: DocumentationResult) -> str:
        """Build a plain docstring body from a DocumentationResult."""
        parts: List[str] = []
        summary = (result.summary or "").strip()
        if summary:
            parts.append(summary)
        parameters = result.parameters or []
        if parameters:
            parts.append("")
            parts.append("Args:")
            for item in parameters:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "") or "").strip() or "arg"
                typ = str(item.get("type", "") or "").strip()
                desc = str(item.get("description", "") or "").strip()
                type_bit = f" ({typ})" if typ else ""
                parts.append(f"    {name}{type_bit}: {desc}".rstrip())
        returns = (result.returns or "").strip()
        if returns:
            parts.append("")
            parts.append("Returns:")
            parts.append(f"    {returns}")
        example = (result.example_usage or "").strip()
        if example:
            parts.append("")
            parts.append("Example:")
            parts.append(f"    {example}")
        return "\n".join(parts).strip() or "Documented by DocumentationAgent."

    def _upsert_module_docstring(
        self,
        source: str,
        docstring_text: str,
        *,
        replace_existing: bool,
    ) -> Tuple[str, str]:
        """Insert or replace a module-level docstring."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise RuntimeError(f"could not parse target file: {exc}") from exc

        existing = ast.get_docstring(tree) or ""
        if existing and not replace_existing:
            return source, "skipped_existing"
        lines = source.splitlines(keepends=True)
        rendered = self._render_indented_docstring(docstring_text, indent="")
        if tree.body and isinstance(tree.body[0], ast.Expr) and self._is_docstring_node(
            tree.body[0]
        ):
            start = tree.body[0].lineno - 1
            end = (tree.body[0].end_lineno or tree.body[0].lineno)
            new_lines = lines[:start] + [rendered] + lines[end:]
            return "".join(new_lines), "replaced"
        # Insert at top, after any shebang / encoding comments.
        insert_at = 0
        while insert_at < len(lines):
            stripped = lines[insert_at].lstrip()
            if stripped.startswith("#!") or (
                stripped.startswith("#") and "coding" in stripped
            ):
                insert_at += 1
                continue
            break
        new_lines = lines[:insert_at] + [rendered, "\n"] + lines[insert_at:]
        return "".join(new_lines), "wrote"

    def _upsert_symbol_docstring(
        self,
        source: str,
        *,
        symbol_name: str,
        docstring_text: str,
        replace_existing: bool,
    ) -> Tuple[str, str]:
        """Insert or replace a function/class docstring by symbol name."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise RuntimeError(f"could not parse target file: {exc}") from exc

        target = self._find_documentable_symbol(tree, symbol_name)
        if target is None:
            # Fall back to the first public function/class lacking a docstring.
            target = self._first_missing_doc_symbol(tree)
            if target is None:
                return source, "skipped_no_target"

        existing = ast.get_docstring(target) or ""
        if existing and not replace_existing:
            return source, "skipped_existing"

        lines = source.splitlines(keepends=True)
        indent = self._body_indent_for(lines, target)
        rendered = self._render_indented_docstring(docstring_text, indent=indent)

        if target.body and self._is_docstring_node(target.body[0]):
            first = target.body[0]
            start = first.lineno - 1
            end = first.end_lineno or first.lineno
            new_lines = lines[:start] + [rendered] + lines[end:]
            return "".join(new_lines), "replaced"

        # Insert before the first body statement, or right after the header.
        if target.body:
            insert_at = target.body[0].lineno - 1
        else:
            insert_at = target.lineno
        new_lines = lines[:insert_at] + [rendered] + lines[insert_at:]
        return "".join(new_lines), "wrote"

    @staticmethod
    def _find_documentable_symbol(
        tree: ast.AST, symbol_name: str
    ) -> Optional[ast.AST]:
        """Find a module-level function/class (or nested method) by name."""
        name = (symbol_name or "").strip()
        if not name:
            return None
        for node in getattr(tree, "body", []):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and node.name == name:
                return node
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(
                        member, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and member.name == name:
                        return member
        return None

    @staticmethod
    def _first_missing_doc_symbol(tree: ast.AST) -> Optional[ast.AST]:
        """Return the first public function/class missing a docstring."""
        for node in getattr(tree, "body", []):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if node.name.startswith("_"):
                continue
            if not ast.get_docstring(node):
                return node
        return None

    @staticmethod
    def _is_docstring_node(node: ast.AST) -> bool:
        """True when ``node`` is an expression docstring statement."""
        if not isinstance(node, ast.Expr):
            return False
        value = getattr(node, "value", None)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return True
        return isinstance(value, ast.Str)  # pragma: no cover - py<3.8 compat

    @staticmethod
    def _body_indent_for(lines: Sequence[str], node: ast.AST) -> str:
        """Infer the indentation used by a function/class body."""
        body = getattr(node, "body", None) or []
        if body:
            line = lines[body[0].lineno - 1]
            return line[: len(line) - len(line.lstrip(" \t"))]
        header = lines[(getattr(node, "lineno", 1) or 1) - 1]
        prefix = header[: len(header) - len(header.lstrip(" \t"))]
        if "\t" in prefix and " " not in prefix:
            return prefix + "\t"
        return prefix + "    "

    @staticmethod
    def _render_indented_docstring(text: str, *, indent: str) -> str:
        """Render a triple-quoted docstring block with the given indent."""
        body = (text or "").strip("\n")
        quote = '"""'
        if '"""' in body:
            quote = "'''"
        lines = body.splitlines() or [""]
        if len(lines) == 1 and len(lines[0]) < 72 and "\n" not in lines[0]:
            return f"{indent}{quote}{lines[0]}{quote}\n"
        rendered = [f"{indent}{quote}\n"]
        for line in lines:
            if line.strip():
                rendered.append(f"{indent}{line.rstrip()}\n")
            else:
                rendered.append("\n")
        rendered.append(f"{indent}{quote}\n")
        return "".join(rendered)

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

        from ..hooks.events import HookEvent

        self._hook(HookEvent.BEFORE_INGEST, workspace=workspace)
        started = time.perf_counter()
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
            self._hook(
                HookEvent.AFTER_INGEST,
                workspace=workspace,
                success=True,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                summary=update.summary(),
            )
        except Exception as exc:
            logger.warning(
                "Documentation indexing failed; retrieval may be empty: %s",
                exc,
            )
            self._hook(
                HookEvent.AFTER_INGEST,
                workspace=workspace,
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
            self._hook(
                HookEvent.ON_ERROR,
                workspace=workspace,
                error=str(exc),
                success=False,
                stage="indexing",
            )

    def _resolve_documentation_target(
        self,
        *,
        filesystem: FilesystemTools,
        mode: str,
        workspace: str,
        target_path: str,
        function_name: str,
        class_name: str,
    ) -> _DocumentationTarget:
        """
        Resolve repository / file / function / class documentation scope.

        Returns a target with ``not_found_reason`` set when a requested
        file or symbol cannot be located.
        """
        relative = self._relative_to_workspace(filesystem, target_path)
        searched = relative if relative not in {".", ""} else target_path or workspace
        requested_file = bool(
            target_path
            and str(target_path).strip()
            and (
                relative.endswith(".py")
                or os.path.abspath(os.path.expanduser(target_path))
                != os.path.abspath(workspace)
            )
        )
        wants_symbol = self._is_explicit_symbol_request(
            function_name=function_name,
            class_name=class_name,
            relative_file=relative,
        )
        symbol_name = (class_name or function_name or "").strip()

        # Explicit repository-wide README mode with no concrete file target.
        if mode == "readme" and not requested_file and not wants_symbol:
            return _DocumentationTarget(
                scope="repository",
                focus_instruction="Generate repository documentation.",
                function_name="README",
            )

        if requested_file or relative.endswith(".py"):
            exists = False
            try:
                exists = bool(
                    relative not in {".", ""} and filesystem.file_exists(relative)
                )
            except Exception:
                exists = False
            if not exists:
                return _DocumentationTarget(
                    scope="file",
                    relative_file=relative if relative.endswith(".py") else "",
                    function_name=function_name,
                    class_name=class_name,
                    not_found_reason=(
                        f"Documentation target not found: file {searched!r} "
                        "does not exist in the repository."
                    ),
                    searched_location=str(searched),
                )

        if wants_symbol:
            kind = "class" if class_name else "function"
            lookup_name = (class_name or function_name).strip()
            search_files: List[str] = []
            if relative.endswith(".py"):
                search_files = [relative]
            else:
                search_files = self._python_files_for_symbol_search(filesystem)

            if not search_files:
                return _DocumentationTarget(
                    scope=kind,
                    relative_file="",
                    function_name=lookup_name if kind == "function" else "",
                    class_name=lookup_name if kind == "class" else "",
                    not_found_reason=(
                        f"Documentation target not found: no Python files were "
                        f"available to locate {kind} {lookup_name!r}."
                    ),
                    searched_location=str(searched),
                )

            for candidate in search_files:
                try:
                    source = filesystem.read_file(candidate)
                except Exception as exc:
                    if len(search_files) == 1:
                        return _DocumentationTarget(
                            scope=kind,
                            relative_file=candidate,
                            function_name=lookup_name if kind == "function" else "",
                            class_name=lookup_name if kind == "class" else "",
                            not_found_reason=(
                                f"Documentation target not found: could not read "
                                f"{candidate!r} ({exc})."
                            ),
                            searched_location=candidate,
                        )
                    continue
                symbol_source, nearby = self._extract_symbol_with_nearby(
                    source,
                    symbol_name=lookup_name,
                    prefer_class=bool(class_name),
                )
                if not symbol_source:
                    continue
                if kind == "class":
                    focus = (
                        f"Generate documentation only for class {lookup_name} "
                        f"inside {candidate}."
                    )
                else:
                    focus = (
                        f"Generate documentation only for function {lookup_name} "
                        f"inside {candidate}."
                    )
                return _DocumentationTarget(
                    scope=kind,
                    relative_file=candidate,
                    function_name=lookup_name if kind == "function" else "",
                    class_name=lookup_name if kind == "class" else "",
                    symbol_source=symbol_source,
                    nearby_source=nearby,
                    focus_instruction=focus,
                )

            searched_hint = (
                relative
                if relative.endswith(".py")
                else f"{len(search_files)} Python file(s) under {searched}"
            )
            return _DocumentationTarget(
                scope=kind,
                relative_file=relative if relative.endswith(".py") else "",
                function_name=lookup_name if kind == "function" else "",
                class_name=lookup_name if kind == "class" else "",
                not_found_reason=(
                    f"Documentation target not found: {kind} {lookup_name!r} "
                    f"was not found in {searched_hint}."
                ),
                searched_location=str(searched_hint),
            )

        if relative.endswith(".py"):
            return _DocumentationTarget(
                scope="file",
                relative_file=relative,
                function_name="",
                class_name=class_name,
                focus_instruction=f"Generate documentation only for {relative}.",
            )

        # Directory / repo fallback.
        return _DocumentationTarget(
            scope="repository",
            focus_instruction="Generate repository documentation.",
            function_name=function_name or ("README" if mode == "readme" else ""),
        )

    @staticmethod
    def _is_explicit_symbol_request(
        *,
        function_name: str,
        class_name: str,
        relative_file: str,
    ) -> bool:
        """True when the caller asked for a real function/class symbol."""
        class_key = (class_name or "").strip()
        if class_key and class_key not in _SPECIAL_FUNCTION_NAMES:
            return True

        name = (function_name or "").strip()
        if not name:
            return False
        if name in _SPECIAL_FUNCTION_NAMES or name.lower() == "readme":
            return False
        if name.endswith(".py"):
            return False
        base = os.path.basename(relative_file or "")
        stem = os.path.splitext(base)[0]
        if name in {base, stem}:
            # Module-label placeholders from file-level docs, not symbols.
            return False
        return True

    def _python_files_for_symbol_search(
        self, filesystem: FilesystemTools
    ) -> List[str]:
        """List inventory Python files used to locate a named symbol."""
        paths: List[str] = []
        try:
            found = filesystem.list_files(".", pattern="*.py", recursive=True)
        except Exception as exc:
            logger.warning("Could not list Python files for symbol search: %s", exc)
            return []
        for path in found:
            normalized = self._normalize_repo_path(path)
            if normalized and normalized.endswith(".py"):
                paths.append(normalized)
            if len(paths) >= _MAX_SOURCE_FILES * 3:
                break
        return paths

    def _extract_symbol_with_nearby(
        self,
        source: str,
        *,
        symbol_name: str,
        prefer_class: bool,
    ) -> Tuple[str, str]:
        """
        Return ``(symbol_source, nearby_context)`` for a named symbol.

        Nearby context includes imports and a small window of lines before
        the symbol. Empty symbol_source means the symbol was not found.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return "", ""
        lines = source.splitlines()
        node: Optional[ast.AST] = None
        if prefer_class:
            for item in getattr(tree, "body", []):
                if isinstance(item, ast.ClassDef) and item.name == symbol_name:
                    node = item
                    break
        if node is None:
            node = self._find_documentable_symbol(tree, symbol_name)
        if node is None:
            return "", ""

        start = max(0, int(getattr(node, "lineno", 1) or 1) - 1)
        end = int(getattr(node, "end_lineno", None) or (start + 1))
        symbol_source = "\n".join(lines[start:end])

        # Nearby context is imports only so sibling top-level symbols stay out
        # of focused function/class documentation prompts.
        import_lines: List[str] = []
        for item in getattr(tree, "body", []):
            if isinstance(item, (ast.Import, ast.ImportFrom)):
                i_start = max(0, int(getattr(item, "lineno", 1) or 1) - 1)
                i_end = int(getattr(item, "end_lineno", None) or (i_start + 1))
                import_lines.extend(lines[i_start:i_end])
        nearby = "\n".join(import_lines).strip()
        return symbol_source, nearby

    @staticmethod
    def _build_symbol_excerpts(target: _DocumentationTarget) -> List[str]:
        """Render focused symbol + nearby excerpts for the prompt."""
        excerpts: List[str] = []
        label = target.relative_file or "target.py"
        if target.nearby_source.strip():
            excerpts.append(
                f"### {label} (nearby context)\n{target.nearby_source.strip()}"
            )
        if target.symbol_source.strip():
            kind = "class" if target.scope == "class" else "function"
            name = target.class_name or target.function_name or "symbol"
            excerpts.append(
                f"### {label} ({kind} {name})\n{target.symbol_source.strip()}"
            )
        return excerpts

    def _retrieve_context(
        self,
        query: str,
        target_path: str,
        *,
        source_file: str = "",
    ) -> List[RetrievedChunk]:
        """
        Retrieve RAG chunks for the documentation request.

        When ``source_file`` is set, keep only chunks from that file.
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
        if source_file:
            chunks = self._filter_chunks_by_file(chunks, source_file)
        return self._dedupe_chunks(chunks)[:_MAX_CONTEXT_CHUNKS]

    def _filter_chunks_by_file(
        self,
        chunks: Sequence[RetrievedChunk],
        file_path: str,
    ) -> List[RetrievedChunk]:
        """Keep retrieved chunks whose source matches ``file_path``."""
        wanted = self._normalize_repo_path(file_path)
        if not wanted:
            return list(chunks)
        wanted_base = os.path.basename(wanted)
        kept: List[RetrievedChunk] = []
        for chunk in chunks:
            source = getattr(chunk, "source", None) or (
                chunk.metadata.get("file_path", "") if chunk.metadata else ""
            )
            normalized = self._normalize_repo_path(str(source or ""))
            if not normalized:
                continue
            if (
                normalized == wanted
                or normalized.endswith("/" + wanted)
                or wanted.endswith("/" + normalized)
                or os.path.basename(normalized) == wanted_base
            ):
                kept.append(chunk)
        return kept

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
        class_name: str = "",
        focus_instruction: str = "",
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
        if focus_instruction:
            sections.append(
                "FOCUS\n"
                f"{focus_instruction}\n"
                "Do not document unrelated modules, files, or symbols."
            )
        if function_name:
            sections.append(f"FUNCTION NAME\n{function_name}")
        if class_name:
            sections.append(f"CLASS NAME\n{class_name}")

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
