"""
documentation_agent.py
========================

Defines DocumentationAgent, responsible for generating and updating
documentation (docstrings, README sections, API docs) for a codebase.

Uses the injected Ollama-backed LLMClient, Retriever for RAG context,
and FilesystemTools for reading repository source.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from ..rag.indexer import Indexer
from ..schemas.schemas import (
    AgentRequest,
    AgentResponse,
    AgentType,
    DocumentationResult,
    ModelMessage,
    RetrievedChunk,
)
from ..tools.filesystem_tools import FilesystemTools
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

        # Index first so retrieval can prioritize grounded chunks.
        logger.info("Indexing repository for documentation retrieval...")
        self._ensure_index(workspace)

        logger.info("Retrieving documentation context...")
        query = " ".join(
            part for part in (instruction, function_name, target_path, mode) if part
        )
        chunks = self._retrieve_context(query, target_path)

        logger.info("Reading repository...")
        filesystem = FilesystemTools(workspace_root=workspace)
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
            return empty

        result = self._parse_response(
            response.content,
            default_file_path=target_path,
            default_function_name=function_name,
        )
        logger.info("Documentation generated.")
        return result

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
        empty = self._empty_result(default_file_path, default_function_name)
        if not content or not str(content).strip():
            return empty

        payload = self._extract_json_object(str(content))
        if payload is None:
            logger.warning("Documentation model response was not valid JSON.")
            return empty

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

        return DocumentationResult(
            file_path=str(payload.get("file_path") or default_file_path or ""),
            function_name=str(
                payload.get("function_name") or default_function_name or ""
            ),
            summary=str(payload.get("summary") or ""),
            parameters=cleaned_params,
            returns=str(payload.get("returns") or ""),
            example_usage=str(payload.get("example_usage") or ""),
        )

    @staticmethod
    def _extract_json_object(content: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object from a model response."""
        text = content.strip()
        fence = _FENCE.search(text)
        if fence:
            text = fence.group(1).strip()

        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
        return None

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
