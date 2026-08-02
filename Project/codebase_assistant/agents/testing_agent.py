"""
testing_agent.py
=================

Defines TestingAgent, responsible for generating unit tests, estimating
coverage, and suggesting testing strategies for a codebase.

Uses the injected OpenRouter-backed LLMClient, Retriever for RAG
context, and FilesystemTools for reading repository source.
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
    ModelMessage,
    RetrievedChunk,
    TestingResult,
)
from ..tools.filesystem_tools import FilesystemTools
from .base import BaseAgent

logger = logging.getLogger(__name__)

#: Cap on source files read into the prompt when building repository context.
_MAX_SOURCE_FILES = 6

#: Cap on characters taken from each source file (fallback when no RAG).
_MAX_FILE_CHARS = 2000

#: Shorter file excerpts when retrieved chunks are already present.
_MAX_FILE_CHARS_WITH_RETRIEVAL = 800

#: Cap on retrieved chunks rendered into the prompt.
_MAX_CONTEXT_CHUNKS = 5

#: Cap on characters per retrieved chunk in the prompt.
_MAX_CHUNK_CHARS = 900

#: Soft ceiling on the assembled user-prompt size (chars).
_MAX_PROMPT_CHARS = 12_000

#: Generation ceiling for test-generation calls.
_TEST_MAX_TOKENS = 1536

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TEST_DEF = re.compile(r"^\s*def\s+(test_\w+)\s*\(", re.MULTILINE)

_SYSTEM_PROMPT = """\
You are a senior Python test engineer writing executable pytest modules.

Your job is to emit clean, runnable pytest source grounded ONLY in the \
code evidence supplied in the user message. Quality over quantity.

Hard rules (follow exactly):
1. Test ONLY symbols that appear in RETRIEVED CONTEXT or REPOSITORY \
CONTENTS. Never invent functions, classes, methods, modules, constants, \
exceptions, or return values.
2. Prefer RETRIEVED CONTEXT first. Use REPOSITORY CONTENTS only to fill \
gaps (imports, signatures, helpers) that the chunks omit.
3. If a symbol is not clearly present in the evidence, do not test it.
4. Write real pytest code: imports, `def test_...`, plain `assert`, and \
`pytest.raises` / `pytest.approx` where appropriate. No pseudo-code, \
no placeholders like `...` or `# TODO`.
5. Assertions must check meaningful observable behavior (return values, \
state changes, exceptions, types) - not tautologies (`assert True`) \
and not copies of the implementation under a different name.
6. For each public function/method you can see, prefer a small focused \
set: one happy path, one edge case, one invalid-input / failure case. \
Skip a category when the code gives no basis for it.
7. Do not emit duplicate or near-duplicate tests (same call pattern and \
same assertion intent under different names).
8. Prefer testing public APIs. Skip private helpers (`_name`) unless \
they are the only symbols available.
9. Keep modules minimal and idiomatic: one test file per source module \
when practical; use clear `test_<behavior>` names.
10. If evidence is thin, return fewer tests (or an empty \
`generated_tests` object) rather than guessing.

Return ONE JSON object only - no markdown fences, no commentary:

{
  "summary": "What was tested and which behaviors were covered.",
  "generated_tests": {
    "test_module_name.py": "complete pytest module source as a string"
  },
  "coverage_estimate": 0.0
}

`generated_tests` values must be complete pytest modules (imports \
included) that a developer could save and run. \
`coverage_estimate` is a float in [0.0, 1.0] for the visible public \
surface you actually wrote tests for - be conservative.
"""

_WRITING_INSTRUCTIONS = """\
Generate executable pytest unit tests for the TARGET.

Priority of evidence:
1) RETRIEVED CONTEXT - primary; write tests from these symbols first.
2) REPOSITORY CONTENTS - secondary; use only to complete signatures, \
imports, or adjacent helpers missing from retrieval.

Coverage goals (only when justified by evidence):
- happy-path behavior with meaningful asserts
- edge cases (empty inputs, boundaries, None if relevant)
- invalid inputs / failure paths (use pytest.raises when the code \
clearly raises or documents that behavior)

Style:
- clean, deterministic pytest (no network, no sleeps, no randomness)
- one behavior per test function; no duplicate scenarios
- import only modules/symbols that exist in the evidence
- never invent APIs"""


class TestingAgent(BaseAgent):
    """
    Agent specialized in generating tests, evaluating test coverage,
    and suggesting testing strategies for a codebase.
    """

    # Prevent pytest from treating this agent class as a test container.
    __test__ = False

    agent_type: AgentType = AgentType.TESTING

    def run(self, repo_path: str) -> Dict[str, str]:
        """
        Simple entry point used by the Supervisor for direct routing.

        Args:
            repo_path: Path to the repository to generate tests for.

        Returns:
            A dict with "status" and "message" keys summarizing the run.
        """
        result = self.generate_unit_tests(repo_path)
        if result.generated_tests or (result.summary and result.summary.strip()):
            files = ", ".join(sorted(result.generated_tests.keys())) or "none"
            return {
                "status": "success",
                "message": (
                    f"{result.summary[:400]} "
                    f"[files={files}; coverage≈{result.coverage_estimate:.2f}]"
                ).strip(),
            }
        return {
            "status": "error",
            "message": "Test generation failed or produced no tests.",
        }

    def handle(self, request: AgentRequest) -> AgentResponse:
        """
        Handle a testing-related request.

        Args:
            request: The AgentRequest describing what to test.

        Returns:
            An AgentResponse wrapping a TestingResult. Failures return
            success=False with an empty TestingResult and errors.
        """
        context = request.context or {}
        instruction = (request.instruction or "").strip()
        repo_path = str(
            context.get("repo_path") or context.get("repository_path") or "."
        )
        file_path = str(context.get("file_path") or "")
        target = file_path or repo_path

        if not self._model_available():
            logger.info("Calling OpenRouter... unavailable; returning failed response.")
            return AgentResponse(
                task_id=request.task_id,
                agent_type=self.agent_type,
                success=False,
                output=self._empty_result(),
                errors=[
                    "OpenRouter model provider is unavailable; "
                    "tests were not generated."
                ],
            )

        try:
            result = self._run_pipeline(
                workspace=self._workspace_for(target),
                target_path=target,
                instruction=instruction
                or (
                    f"Generate pytest unit tests for {target}, covering "
                    "functions, methods, edge cases, invalid inputs, and "
                    "common failure scenarios."
                ),
            )
        except Exception as exc:
            logger.warning("TestingAgent.handle failed: %s", exc)
            return AgentResponse(
                task_id=request.task_id,
                agent_type=self.agent_type,
                success=False,
                output=self._empty_result(),
                errors=[str(exc)],
            )

        success = bool(result.generated_tests) or bool(
            result.summary and result.summary.strip()
        )
        return AgentResponse(
            task_id=request.task_id,
            agent_type=self.agent_type,
            success=success,
            output=result,
            errors=[]
            if success
            else ["Test generation produced no usable tests or summary."],
        )

    def generate_unit_tests(self, file_path: str) -> TestingResult:
        """
        Generate unit tests for a given file.

        Args:
            file_path: Path to the file (or repository) to generate tests for.

        Returns:
            A populated TestingResult, or an empty result when the model
            is unavailable or generation fails.
        """
        workspace = self._workspace_for(file_path)
        return self._run_pipeline(
            workspace=workspace,
            target_path=file_path or workspace,
            instruction=(
                f"Generate pytest unit tests for {file_path}. "
                "Cover public functions and methods, edge cases, invalid "
                "inputs, and common failure scenarios. Stay grounded in "
                "the provided source."
            ),
        )

    def estimate_coverage(self, repo_path: str) -> float:
        """
        Estimate current test coverage for a repository.

        Args:
            repo_path: Path to the repository root.

        Returns:
            A coverage estimate between 0.0 and 1.0 from the model when
            available, otherwise 0.0.
        """
        result = self.generate_unit_tests(repo_path)
        try:
            value = float(result.coverage_estimate)
        except (TypeError, ValueError):
            return 0.0
        return min(max(value, 0.0), 1.0)

    def suggest_test_cases(self, file_path: str) -> list:
        """
        Suggest additional test cases that should be written for a file.

        Args:
            file_path: Path to the file to analyze.

        Returns:
            A list of suggested test case descriptions derived from
            generated pytest test function names when available.
        """
        result = self.generate_unit_tests(file_path)
        suggestions: List[str] = []
        for test_path, code in result.generated_tests.items():
            for match in _TEST_DEF.finditer(code or ""):
                suggestions.append(f"{match.group(1)} (in {test_path})")
        if not suggestions and result.summary:
            suggestions.append(result.summary.strip())
        return suggestions

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        workspace: str,
        target_path: str,
        instruction: str,
    ) -> TestingResult:
        """
        Run the testing pipeline for one request.

        Stages: index → retrieve → read files → build prompt →
        call LLM → parse TestingResult.
        """
        empty = self._empty_result()

        if not self._model_available():
            logger.info("Calling OpenRouter... unavailable.")
            return empty

        logger.info("Indexing repository for testing retrieval...")
        self._ensure_index(workspace)

        logger.info("Retrieving testing context...")
        query = " ".join(
            part
            for part in (
                instruction,
                target_path,
                "pytest unit tests edge cases invalid inputs",
            )
            if part
        )
        chunks = self._retrieve_context(query, target_path)

        logger.info("Reading repository...")
        filesystem = FilesystemTools(workspace_root=workspace)
        source_excerpts = self._read_repository_sources(
            filesystem,
            target_path,
            max_file_chars=(
                _MAX_FILE_CHARS_WITH_RETRIEVAL if chunks else _MAX_FILE_CHARS
            ),
            prefer_target_only=bool(chunks),
        )

        logger.info("Building prompt...")
        prompt = self._build_prompt(
            instruction=instruction,
            target_path=target_path,
            chunks=chunks,
            source_excerpts=source_excerpts,
        )

        logger.info("Calling OpenRouter...")
        try:
            response = self.model_client.generate(
                [
                    ModelMessage(role="system", content=_SYSTEM_PROMPT),
                    ModelMessage(role="user", content=prompt),
                ],
                max_tokens=_TEST_MAX_TOKENS,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("OpenRouter testing call failed: %s", exc)
            return empty

        result = self._parse_response(response.content)
        logger.info(
            "Test generation finished: %d file(s), coverage≈%.2f",
            len(result.generated_tests),
            result.coverage_estimate,
        )
        return result

    def _model_available(self) -> bool:
        """Report whether the injected model client can serve requests."""
        if self.model_client is None:
            return False
        try:
            return bool(self.model_client.is_available())
        except Exception as exc:
            logger.warning("Testing model availability check failed: %s", exc)
            return False

    def _ensure_index(self, workspace: str) -> None:
        """
        Index the workspace into the store the injected Retriever reads.

        Uses Indexer.update_index for incremental updates. Failures are
        logged and swallowed so generation can continue from files.
        """
        if self.retriever is None:
            logger.info("No retriever configured; skipping testing indexing.")
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
            logger.info("Testing index: %s", update.summary())
        except Exception as exc:
            logger.warning(
                "Testing indexing failed; retrieval may be empty: %s",
                exc,
            )

    def _retrieve_context(
        self, query: str, target_path: str
    ) -> List[RetrievedChunk]:
        """
        Retrieve RAG chunks for the testing request.

        Returns an empty list when the retriever is missing, the index
        is empty, or retrieval fails - callers continue with source files.
        """
        if self.retriever is None:
            logger.info("No retriever configured; continuing with repository files only.")
            return []

        try:
            chunks = self.retriever.retrieve(
                query=query or target_path or "unit tests"
            )
        except Exception as exc:
            logger.warning(
                "Retrieval failed; continuing with repository files only: %s",
                exc,
            )
            return []

        if not chunks:
            logger.info(
                "Retriever returned no chunks; continuing with repository files only."
            )
            return []
        return self._dedupe_chunks(chunks)[:_MAX_CONTEXT_CHUNKS]

    @staticmethod
    def _dedupe_chunks(chunks: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Collapse exact and near-duplicate retrieved chunks.

        Drops empty chunks, whitespace-normalized duplicates, and shorter
        excerpts already contained in a longer kept chunk. When a new
        chunk fully contains a shorter kept excerpt, the longer one wins.
        """
        unique: List[RetrievedChunk] = []
        keys: List[str] = []

        for chunk in chunks:
            content = (chunk.content or "").strip()
            if not content:
                continue
            key = " ".join(content.split())
            if not key:
                continue
            # Exact duplicate, or already covered by a longer excerpt.
            if any(key == kept or key in kept for kept in keys):
                continue
            # Replace a shorter kept excerpt that this chunk supersedes.
            replace_at = next(
                (i for i, kept in enumerate(keys) if kept in key and kept != key),
                None,
            )
            if replace_at is not None:
                unique[replace_at] = chunk
                keys[replace_at] = key
                continue
            unique.append(chunk)
            keys.append(key)
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
        instruction: str,
        target_path: str,
        chunks: Sequence[RetrievedChunk],
        source_excerpts: Sequence[str],
    ) -> str:
        """
        Build the user prompt for the test-generation model call.

        Packing order: instructions → retrieved chunks (deduped/truncated)
        → shorter repository excerpts → output contract. Oversized
        prompts are truncated from the secondary excerpts first.
        """
        packed_chunks = self._dedupe_chunks(chunks)[:_MAX_CONTEXT_CHUNKS]
        sections = [
            "TESTING MODE\npytest unit test generation",
            f"WRITING INSTRUCTIONS\n{_WRITING_INSTRUCTIONS}",
            f"REQUEST\n{instruction}",
            f"TARGET\n{target_path}",
        ]

        if packed_chunks:
            rendered = []
            for index, chunk in enumerate(packed_chunks, start=1):
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
                "RETRIEVED CONTEXT (primary - write tests from these symbols first)\n"
                + "\n\n".join(rendered)
            )
        else:
            sections.append(
                "RETRIEVED CONTEXT (primary - write tests from these symbols first)\n"
                "(none)\n"
                "Rely on REPOSITORY CONTENTS only; still do not invent APIs."
            )

        if source_excerpts:
            label = (
                "REPOSITORY CONTENTS (secondary - fill gaps only; shorter excerpts)"
                if packed_chunks
                else "REPOSITORY CONTENTS (primary - no retrieved context)"
            )
            sections.append(f"{label}\n" + "\n\n".join(source_excerpts))
        else:
            sections.append("REPOSITORY CONTENTS\n(none)")

        sections.append(
            "OUTPUT CONTRACT\n"
            "Return only the TestingResult JSON object. "
            "`generated_tests` values must be complete runnable pytest "
            "modules. Do not wrap the JSON in markdown fences."
        )
        return self._truncate_prompt("\n\n".join(sections))

    @staticmethod
    def _truncate_prompt(prompt: str, limit: int = _MAX_PROMPT_CHARS) -> str:
        """
        Truncate an oversized user prompt while keeping the head intact.

        Prefer keeping instructions + retrieved context; the tail is
        usually secondary repository excerpts and the output contract,
        so a short marker is appended when cutting.
        """
        if len(prompt) <= limit:
            return prompt
        marker = "\n\n[truncated: prompt exceeded size budget]\n"
        keep = max(0, limit - len(marker))
        return prompt[:keep] + marker

    def _parse_response(self, content: str) -> TestingResult:
        """
        Parse model output into TestingResult.

        Returns an empty result when parsing fails rather than raising.
        """
        empty = self._empty_result()
        if not content or not str(content).strip():
            return empty

        payload = self._extract_json_object(str(content))
        if payload is None:
            logger.warning("Testing model response was not valid JSON.")
            return empty

        generated_raw = payload.get("generated_tests") or {}
        generated_tests: Dict[str, str] = {}
        if isinstance(generated_raw, dict):
            for key, value in generated_raw.items():
                name = str(key or "").strip()
                code = value if isinstance(value, str) else str(value or "")
                if name and code.strip():
                    generated_tests[name] = code

        try:
            coverage = float(payload.get("coverage_estimate", 0.0) or 0.0)
        except (TypeError, ValueError):
            coverage = 0.0
        coverage = min(max(coverage, 0.0), 1.0)

        return TestingResult(
            summary=str(payload.get("summary") or ""),
            generated_tests=generated_tests,
            coverage_estimate=coverage,
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
    def _empty_result() -> TestingResult:
        """Build an empty TestingResult for failure paths."""
        return TestingResult(
            summary="",
            generated_tests={},
            coverage_estimate=0.0,
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
