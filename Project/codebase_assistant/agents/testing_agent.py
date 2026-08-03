"""
testing_agent.py
=================

Defines TestingAgent, responsible for generating unit tests, estimating
coverage, and suggesting testing strategies for a codebase.

Uses the injected OpenRouter-backed LLMClient, Retriever for RAG
context, and ToolRegistry-resolved FilesystemTools for reading
repository source. After generation succeeds, tests are written to a
temporary directory and executed via pytest's Python API; pass/fail
counts are appended to TestingResult.summary without changing schemas
or mutating generated_tests.

When the first pytest run reports failures or errors, the agent performs
exactly one repair iteration: it sends the original tests, pytest
output, and repository context back to the model, then reruns pytest
on the repaired sources.

Test generation is symbol-scoped: the agent scans the repository with
AST, then prompts once per public function or public class (with its
public methods) before merging modules and entering the execution /
repair pipeline.

After pytest execution, the agent measures real line coverage with
pytest-cov (JSON report preferred) and stores that value in
``coverage_estimate``, falling back to the model estimate only when
coverage tooling is unavailable.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
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
    ModelMessage,
    RetrievedChunk,
    TestingResult,
)
from ..tools.filesystem_tools import FilesystemTools
from ..tracing.events import TraceEventType
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

#: Cap on pytest failure text embedded in a repair prompt.
_MAX_PYTEST_OUTPUT_CHARS = 4_000

#: Cap on generated-test source embedded in a repair prompt.
_MAX_REPAIR_TEST_CHARS = 6_000

#: Generation ceiling for test-generation calls.
_TEST_MAX_TOKENS = 1536

#: Soft cap on public symbols prompted in one pipeline run.
_MAX_SYMBOLS_TO_TEST = 20

#: Cap on Python files scanned during the AST inventory pass.
_MAX_AST_FILES = 40

#: Directory names excluded from the testing inventory.
_SKIP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        "tests",
        ".git",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "site-packages",
        "node_modules",
        "generated",
    }
)

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

_REPAIR_SYSTEM_PROMPT = """\
You are a senior Python test engineer repairing failing pytest modules.

You receive:
1) the originally generated pytest sources,
2) the pytest failure / error output (primary debugging signal),
3) repository context for the code under test.

Hard rules (follow exactly):
1. Fix ONLY the failing tests. Preserve every passing test unchanged \
unless a minimal shared-import fix is required for the suite to load.
2. Do not invent modules, functions, classes, constants, or exceptions \
that are absent from the repository context.
3. Use the pytest error output as the primary signal: fix assertions, \
imports, fixtures, and call signatures that the traceback identifies.
4. Return complete, valid pytest module source (imports included). No \
pseudo-code, no placeholders, no markdown fences around the JSON.
5. Prefer the smallest change that makes the suite collect and pass.
6. If a failure cannot be fixed from the evidence, keep that test but \
adjust it to match observable behavior in the repository context — \
never invent APIs.

Return ONE JSON object only:

{
  "summary": "What was repaired and why.",
  "generated_tests": {
    "test_module_name.py": "complete repaired pytest module source"
  },
  "coverage_estimate": 0.0
}
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


@dataclass
class _PytestExecutionStats:
    """Counters collected from one pytest.main run."""

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    exit_code: int = 0
    detail: str = ""
    output: str = ""


@dataclass
class _CoverageMeasurement:
    """Result of one pytest-cov measurement attempt."""

    available: bool
    measured: bool = False
    percent: float = 0.0  # 0-100 line coverage
    files_measured: int = 0
    statements: int = 0
    missing: int = 0
    summary: str = ""
    error: str = ""

    @property
    def ratio(self) -> float:
        """Coverage as a 0.0-1.0 ratio for ``TestingResult.coverage_estimate``."""
        return min(max(self.percent / 100.0, 0.0), 1.0)


@dataclass
class _ExecutionOutcome:
    """Internal result of writing and running generated tests."""

    summary: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    output: str = ""
    skipped_execution: bool = False
    coverage: Optional["_CoverageMeasurement"] = None

    @property
    def needs_repair(self) -> bool:
        """True when pytest ran and reported failures or errors."""
        if self.skipped_execution:
            return False
        return self.failed > 0 or self.errors > 0


@dataclass
class _MethodInfo:
    """One public method discovered on a class."""

    name: str
    signature: str
    docstring: str
    source: str


@dataclass
class _TestableSymbol:
    """A public function or class selected for focused test generation."""

    kind: str  # "function" | "class"
    name: str
    qualname: str
    module_path: str
    signature: str
    docstring: str
    source: str
    methods: List[_MethodInfo] = field(default_factory=list)

    @property
    def focus_instruction(self) -> str:
        """Human-readable focus line used in the symbol prompt."""
        module = os.path.basename(self.module_path) or self.module_path
        if self.kind == "class":
            method_names = ", ".join(m.name for m in self.methods) or "(none)"
            return (
                f"Generate pytest tests for the public methods of "
                f"{self.name} defined in {module}. "
                f"Public methods: {method_names}."
            )
        return (
            f"Generate pytest tests for the function {self.name}() "
            f"defined in {module}."
        )


class _PytestStatsPlugin:
    """
    Minimal pytest plugin that records pass/fail/skip/error counts.

    Kept private to TestingAgent execution — not part of the public API.
    """

    def __init__(self) -> None:
        self.stats = _PytestExecutionStats()
        self._failure_messages: List[str] = []

    def pytest_runtest_logreport(self, report: Any) -> None:
        """Accumulate per-phase test reports into ``self.stats``."""
        if report.when == "call":
            if report.passed:
                self.stats.passed += 1
            elif report.failed:
                self.stats.failed += 1
                self._capture_failure(report)
            elif report.skipped:
                self.stats.skipped += 1
            return

        # Setup/teardown failures count as errors; skipped setup as skipped.
        if report.failed:
            self.stats.errors += 1
            self._capture_failure(report)
        elif report.skipped:
            self.stats.skipped += 1

    def pytest_collectreport(self, report: Any) -> None:
        """Count collection failures (e.g. syntax errors) as errors."""
        if report.failed:
            self.stats.errors += 1
            self._capture_failure(report)

    def _capture_failure(self, report: Any) -> None:
        """Keep a short human-readable failure snippet for the summary."""
        if self.stats.detail:
            return
        longrepr = getattr(report, "longrepr", None)
        text = str(longrepr or "").strip()
        if text:
            self.stats.detail = text


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

        if result.abstention is not None:
            return AgentResponse(
                task_id=request.task_id,
                agent_type=self.agent_type,
                success=False,
                output=result,
                errors=[result.abstention.reason],
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

        Stages: index → AST inventory/symbol scan → per-symbol generation →
        merge → execute pytest → optional one-shot repair.
        """
        empty = self._empty_result()

        if not self._model_available():
            logger.info("Calling OpenRouter... unavailable.")
            return empty

        self._trace(
            "testing_started",
            workspace=workspace,
            target_path=target_path,
        )

        logger.info("Indexing repository for testing retrieval...")
        self._trace("indexing", event_type=TraceEventType.INGESTION, phase="started")
        index_started = time.perf_counter()
        self._ensure_index(workspace)
        self._trace(
            "indexing",
            event_type=TraceEventType.INGESTION,
            success=True,
            duration_ms=(time.perf_counter() - index_started) * 1000.0,
            phase="finished",
        )

        logger.info("Scanning repository symbols with AST...")
        filesystem = self._filesystem_tools(workspace)
        inventory = self._list_python_inventory(filesystem, target_path)
        self._trace(
            "testing_ast_scan_started",
            files=len(inventory),
            target_path=target_path,
        )
        scan_started = time.perf_counter()
        symbols, skipped_symbols = self._collect_testable_symbols(
            filesystem, inventory
        )
        self._trace(
            "testing_ast_scan_finished",
            success=True,
            duration_ms=(time.perf_counter() - scan_started) * 1000.0,
            files_scanned=len(inventory),
            symbols_discovered=len(symbols),
            symbols_skipped=len(skipped_symbols),
        )

        if not inventory:
            abstained = self._abstain_result(
                reason="Repository contains no supported Python files.",
                evidence_available=[],
                recommended_next_steps=[
                    "Provide a repository that includes .py source files.",
                    "Confirm ignore rules are not excluding the entire tree.",
                ],
            )
            self._trace(
                "testing_finished",
                success=False,
                abstained=True,
                reason=abstained.abstention.reason if abstained.abstention else "",
            )
            return abstained

        if not symbols:
            abstained = self._abstain_result(
                reason="No public symbols were found to test.",
                evidence_available=[f"{len(inventory)} inventoried Python file(s)"],
                recommended_next_steps=[
                    "Point testing at a module with public functions or classes.",
                    "Confirm private-only modules are not the only targets.",
                ],
            )
            self._trace(
                "testing_finished",
                success=False,
                abstained=True,
                reason=abstained.abstention.reason if abstained.abstention else "",
            )
            return abstained

        selected = symbols[:_MAX_SYMBOLS_TO_TEST]
        partial_results: List[TestingResult] = []
        repair_chunks: List[RetrievedChunk] = []
        repair_excerpts: List[str] = []
        seen_excerpt_paths: Set[str] = set()

        for symbol in selected:
            self._trace(
                "testing_symbol_generation_started",
                kind=symbol.kind,
                symbol=symbol.qualname,
                module_path=symbol.module_path,
            )
            symbol_started = time.perf_counter()
            query = " ".join(
                part
                for part in (
                    instruction,
                    symbol.focus_instruction,
                    symbol.qualname,
                    symbol.module_path,
                    "pytest unit tests edge cases invalid inputs",
                )
                if part
            )
            chunks = self._retrieve_context(query, symbol.module_path)
            if chunks:
                repair_chunks.extend(chunks)

            excerpt = self._format_file_excerpt(
                symbol.module_path,
                symbol.source,
                max_chars=_MAX_FILE_CHARS,
            )
            source_excerpts = [excerpt]
            if symbol.module_path not in seen_excerpt_paths:
                # Prefer a broader module excerpt for repair context.
                try:
                    if filesystem.file_exists(symbol.module_path):
                        module_text = filesystem.read_file(symbol.module_path)
                        repair_excerpts.append(
                            self._format_file_excerpt(
                                symbol.module_path,
                                module_text,
                                _MAX_FILE_CHARS_WITH_RETRIEVAL,
                            )
                        )
                        seen_excerpt_paths.add(symbol.module_path)
                except Exception as exc:
                    logger.warning(
                        "Could not read module %s for repair context: %s",
                        symbol.module_path,
                        exc,
                    )

            prompt = self._build_symbol_prompt(
                instruction=instruction,
                symbol=symbol,
                chunks=chunks,
                source_excerpts=source_excerpts,
            )
            self._trace(
                "model_request",
                event_type=TraceEventType.MODEL_CALL,
                symbol=symbol.qualname,
                chunks=len(chunks),
            )
            model_started = time.perf_counter()
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
                logger.warning(
                    "OpenRouter testing call failed for %s: %s",
                    symbol.qualname,
                    exc,
                )
                self._trace(
                    "model_response",
                    event_type=TraceEventType.MODEL_CALL,
                    success=False,
                    error=str(exc),
                    symbol=symbol.qualname,
                    duration_ms=(time.perf_counter() - model_started) * 1000.0,
                )
                self._trace(
                    "testing_symbol_generation_finished",
                    success=False,
                    symbol=symbol.qualname,
                    error=str(exc),
                    duration_ms=(time.perf_counter() - symbol_started) * 1000.0,
                )
                continue

            self._trace(
                "model_response",
                event_type=TraceEventType.MODEL_CALL,
                success=True,
                duration_ms=(time.perf_counter() - model_started) * 1000.0,
                content_chars=len(response.content or ""),
                symbol=symbol.qualname,
            )
            parsed = self._parse_response(response.content)
            normalized = self._normalize_symbol_result(parsed, symbol)
            self._trace(
                "testing_symbol_generation_finished",
                success=bool(normalized.generated_tests),
                symbol=symbol.qualname,
                files=len(normalized.generated_tests),
                duration_ms=(time.perf_counter() - symbol_started) * 1000.0,
            )
            if normalized.generated_tests:
                partial_results.append(normalized)

        result = self._merge_testing_results(partial_results)
        self._trace(
            "testing_merge_completed",
            generated_test_files=len(result.generated_tests),
            symbols_generated=len(partial_results),
            symbols_selected=len(selected),
        )

        if not result.generated_tests:
            evidence = [
                f"{len(selected)} public symbol(s) selected",
                f"{len(inventory)} inventoried Python file(s)",
            ]
            abstained = self._abstain_result(
                reason="LLM response could not be verified.",
                evidence_available=evidence,
                recommended_next_steps=[
                    "Retry with a narrower target module.",
                    "Confirm the model returned valid TestingResult JSON.",
                ],
            )
            self._trace(
                "testing_finished",
                success=False,
                abstained=True,
                reason=abstained.abstention.reason if abstained.abstention else "",
            )
            return abstained

        logger.info(
            "Test generation finished: %d file(s), coverage≈%.2f",
            len(result.generated_tests),
            result.coverage_estimate,
        )

        chunks = self._dedupe_chunks(repair_chunks)[:_MAX_CONTEXT_CHUNKS]
        source_excerpts = repair_excerpts[:_MAX_SOURCE_FILES]

        if result.generated_tests:
            first_outcome = self._run_generated_tests(
                workspace, result.generated_tests
            )
            result = self._apply_execution_summary(result, first_outcome.summary)

            if first_outcome.needs_repair:
                result = self._repair_failing_tests(
                    workspace=workspace,
                    instruction=instruction,
                    target_path=target_path,
                    result=result,
                    first_outcome=first_outcome,
                    chunks=chunks,
                    source_excerpts=source_excerpts,
                )
            else:
                result = self._apply_coverage_measurement(result, first_outcome)

        self._trace(
            "testing_finished",
            success=True,
            generated_tests=len(result.generated_tests),
            coverage_estimate=result.coverage_estimate,
        )
        return result

    def _execute_generated_tests(
        self,
        workspace: str,
        generated_tests: Dict[str, str],
    ) -> str:
        """
        Write generated tests to a temp directory and run pytest on them.

        Never raises. Never mutates ``generated_tests``. Failures
        (missing pytest, I/O errors, collection/runtime errors) are
        reported as an execution summary string.

        Args:
            workspace: Repository root used for imports (added to
                ``sys.path`` for the duration of the run).
            generated_tests: Mapping of filename -> source produced by
                the LLM. Values are written unchanged.

        Returns:
            A short execution summary suitable for appending to
            ``TestingResult.summary``.
        """
        return self._run_generated_tests(workspace, generated_tests).summary

    def _run_generated_tests(
        self,
        workspace: str,
        generated_tests: Dict[str, str],
    ) -> _ExecutionOutcome:
        """
        Write and execute generated tests; return a structured outcome.

        Never raises. Never mutates ``generated_tests``.
        """
        if not generated_tests:
            return _ExecutionOutcome(summary="", skipped_execution=True)

        try:
            import pytest as pytest_api
        except ImportError:
            logger.info("pytest is not installed; skipping test execution.")
            return _ExecutionOutcome(
                summary="Execution: skipped (pytest is not installed).",
                skipped_execution=True,
            )

        temp_dir: Optional[str] = None
        try:
            temp_dir = self._create_temp_test_dir(workspace)
        except Exception as exc:
            logger.warning("Could not create temp test directory: %s", exc)
            return _ExecutionOutcome(
                summary=(
                    f"Execution: skipped (could not create temp directory: {exc})."
                ),
                skipped_execution=True,
            )

        try:
            try:
                written = self._write_generated_tests(temp_dir, generated_tests)
            except Exception as exc:
                logger.warning("Failed writing generated tests: %s", exc)
                return _ExecutionOutcome(
                    summary=f"Execution: skipped (could not write tests: {exc}).",
                    skipped_execution=True,
                )

            if not written:
                return _ExecutionOutcome(
                    summary="Execution: skipped (no test files were written).",
                    skipped_execution=True,
                )

            self._trace(
                "generated_tests_written",
                files=len(written),
                temp_dir=temp_dir,
            )

            self._trace("pytest_execution_started", files=len(written))
            pytest_started = time.perf_counter()
            stats = self._run_pytest(pytest_api, workspace, temp_dir)
            summary = self._format_execution_summary(stats)
            self._trace(
                "pytest_execution_finished",
                success=stats.failed == 0 and stats.errors == 0,
                duration_ms=(time.perf_counter() - pytest_started) * 1000.0,
                passed=stats.passed,
                failed=stats.failed,
                skipped=stats.skipped,
                errors=stats.errors,
                exit_code=stats.exit_code,
            )

            coverage = self._measure_coverage(
                pytest_api, workspace=workspace, temp_dir=temp_dir
            )
            if coverage.summary:
                summary = self._merge_summaries(summary, coverage.summary)

            return _ExecutionOutcome(
                summary=summary,
                passed=stats.passed,
                failed=stats.failed,
                skipped=stats.skipped,
                errors=stats.errors,
                output=stats.output or stats.detail or "",
                coverage=coverage,
            )
        finally:
            self._cleanup_temp_test_dir(temp_dir)

    def _repair_failing_tests(
        self,
        *,
        workspace: str,
        instruction: str,
        target_path: str,
        result: TestingResult,
        first_outcome: _ExecutionOutcome,
        chunks: Sequence[RetrievedChunk],
        source_excerpts: Sequence[str],
    ) -> TestingResult:
        """
        Perform exactly one repair iteration after a failing pytest run.

        On repair-generation failure, preserves the original generated
        tests and the first execution summary. On a successful repair
        parse, replaces ``generated_tests`` with the repaired sources and
        appends the second pytest summary (even if it still fails).
        """
        original_tests = dict(result.generated_tests)
        self._trace(
            "testing_repair_started",
            failed=first_outcome.failed,
            errors=first_outcome.errors,
            files=len(original_tests),
        )

        repair_prompt = self._build_repair_prompt(
            instruction=instruction,
            target_path=target_path,
            original_tests=original_tests,
            pytest_output=first_outcome.output,
            first_summary=first_outcome.summary,
            chunks=chunks,
            source_excerpts=source_excerpts,
        )

        model_started = time.perf_counter()
        try:
            response = self.model_client.generate(
                [
                    ModelMessage(role="system", content=_REPAIR_SYSTEM_PROMPT),
                    ModelMessage(role="user", content=repair_prompt),
                ],
                max_tokens=_TEST_MAX_TOKENS,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("OpenRouter testing repair call failed: %s", exc)
            self._trace(
                "testing_repair_failed",
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - model_started) * 1000.0,
            )
            return self._apply_coverage_measurement(result, first_outcome)

        repaired = self._parse_response(response.content)
        if not repaired.generated_tests:
            self._trace(
                "testing_repair_failed",
                success=False,
                error="repair response produced no generated_tests",
                duration_ms=(time.perf_counter() - model_started) * 1000.0,
            )
            return self._apply_coverage_measurement(result, first_outcome)

        self._trace(
            "testing_repair_generated",
            success=True,
            duration_ms=(time.perf_counter() - model_started) * 1000.0,
            files=len(repaired.generated_tests),
            content_chars=len(response.content or ""),
        )

        second_outcome = self._run_generated_tests(
            workspace, repaired.generated_tests
        )
        coverage = (
            repaired.coverage_estimate
            if repaired.coverage_estimate is not None
            else result.coverage_estimate
        )
        combined = self._merge_summaries(
            result.summary,
            "Repair: attempted one fix iteration.",
            second_outcome.summary,
        )
        finished = TestingResult(
            summary=combined,
            generated_tests=repaired.generated_tests,
            coverage_estimate=coverage,
            abstention=None,
        )
        finished = self._apply_coverage_measurement(finished, second_outcome)
        self._trace(
            "testing_repair_finished",
            success=not second_outcome.needs_repair,
            passed=second_outcome.passed,
            failed=second_outcome.failed,
            errors=second_outcome.errors,
            repaired_files=len(repaired.generated_tests),
        )
        return finished

    @staticmethod
    def _apply_execution_summary(
        result: TestingResult, execution_summary: str
    ) -> TestingResult:
        """Append an execution summary line to ``result.summary``."""
        if not execution_summary:
            return result
        return TestingResult(
            summary=TestingAgent._merge_summaries(
                result.summary, execution_summary
            ),
            generated_tests=result.generated_tests,
            coverage_estimate=result.coverage_estimate,
            abstention=result.abstention,
        )

    @staticmethod
    def _apply_coverage_measurement(
        result: TestingResult,
        outcome: "_ExecutionOutcome",
    ) -> TestingResult:
        """
        Prefer measured pytest-cov coverage when available.

        Keeps the model estimate when coverage could not be measured.
        Coverage prose is already appended via ``outcome.summary``.
        """
        coverage = outcome.coverage
        if coverage is None or not coverage.measured:
            return result
        return TestingResult(
            summary=result.summary,
            generated_tests=result.generated_tests,
            coverage_estimate=coverage.ratio,
            abstention=result.abstention,
        )

    @staticmethod
    def _merge_summaries(*parts: Optional[str]) -> str:
        """Join non-empty summary fragments with newlines."""
        chunks = [part.strip() for part in parts if part and str(part).strip()]
        return "\n".join(chunks)

    def _create_temp_test_dir(self, workspace: str) -> str:
        """
        Create a temporary directory for generated tests.

        Uses an isolated system temp directory so generated modules do
        not pollute the repository tree or confuse nested pytest runs.
        The repository ``workspace`` is used only for imports via
        ``sys.path`` during execution.
        """
        _ = workspace
        return tempfile.mkdtemp(prefix="codebase_assistant_temp_tests_")

    def _write_generated_tests(
        self,
        temp_dir: str,
        generated_tests: Dict[str, str],
    ) -> List[str]:
        """
        Write each generated test module into ``temp_dir``.

        Args:
            temp_dir: Destination directory.
            generated_tests: Filename -> source mapping (unchanged).

        Returns:
            Absolute paths of files successfully written.
        """
        written: List[str] = []
        used_names: set[str] = set()
        for raw_name, source in generated_tests.items():
            filename = self._safe_test_filename(raw_name, used_names)
            used_names.add(filename)
            path = os.path.join(temp_dir, filename)
            # Keep LLM source exactly as generated.
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(source if source is not None else "")
            written.append(path)
            logger.info("Wrote generated test file: %s", path)
        return written

    @staticmethod
    def _safe_test_filename(raw_name: str, used: set[str]) -> str:
        """
        Flatten a generated_tests key into a safe basename under temp_dir.

        Args:
            raw_name: Original key from the LLM payload.
            used: Filenames already claimed in this write pass.

        Returns:
            A unique ``*.py`` basename with no path separators.
        """
        name = (raw_name or "").replace("\\", "/").strip()
        base = os.path.basename(name) or "test_generated.py"
        base = base.replace("..", "_")
        if not base.endswith(".py"):
            base = f"{base}.py"
        if not base.startswith("test_") and not base.endswith("_test.py"):
            # Keep LLM names as-is when already pytest-discoverable;
            # otherwise leave unchanged — discovery still finds test_* defs.
            pass
        candidate = base
        counter = 2
        while candidate in used:
            stem, ext = os.path.splitext(base)
            candidate = f"{stem}_{counter}{ext}"
            counter += 1
        return candidate

    def _run_pytest(
        self,
        pytest_api: Any,
        workspace: str,
        temp_dir: str,
    ) -> "_PytestExecutionStats":
        """
        Execute pytest against ``temp_dir`` using the Python API.

        Args:
            pytest_api: The imported ``pytest`` module.
            workspace: Repository root to put on ``sys.path``.
            temp_dir: Directory containing the written test modules.

        Returns:
            Collected pass/fail/skip/error counts and duration.
        """
        plugin = _PytestStatsPlugin()
        path_inserted = False
        workspace_abs = os.path.abspath(workspace)
        previous_cwd = os.getcwd()
        started = time.perf_counter()
        loaded_modules = self._test_module_names(temp_dir)

        try:
            if workspace_abs not in sys.path:
                sys.path.insert(0, workspace_abs)
                path_inserted = True

            # Run from the workspace so relative imports and package
            # discovery behave like a developer running tests locally.
            try:
                os.chdir(workspace_abs)
            except OSError:
                pass

            # importlib mode avoids "import file mismatch" when the same
            # generated basename is executed more than once in-process
            # (e.g. nested pytest while the suite tests TestingAgent).
            args = [
                temp_dir,
                "-q",
                "--tb=line",
                "--import-mode=importlib",
                "-p",
                "no:cacheprovider",
                f"--rootdir={temp_dir}",
                "-o",
                "addopts=",
            ]
            sink = io.StringIO()
            try:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(
                    sink
                ):
                    exit_code = int(
                        pytest_api.main(args, plugins=[plugin])  # type: ignore[arg-type]
                    )
            except SystemExit as exc:
                code = exc.code
                exit_code = int(code) if isinstance(code, int) else 1
            except Exception as exc:
                logger.warning("pytest.main raised: %s", exc)
                plugin.stats.errors += 1
                plugin.stats.detail = str(exc)
                exit_code = 3

            plugin.stats.output = sink.getvalue()
            plugin.stats.exit_code = exit_code
            plugin.stats.duration_seconds = time.perf_counter() - started
            if (
                plugin.stats.passed == 0
                and plugin.stats.failed == 0
                and plugin.stats.skipped == 0
                and plugin.stats.errors == 0
                and exit_code not in (0, 5)
            ):
                # Collection crashed before any reports (e.g. syntax).
                plugin.stats.errors = max(plugin.stats.errors, 1)
                if not plugin.stats.detail:
                    plugin.stats.detail = (
                        f"pytest exited with code {exit_code}"
                    )
            return plugin.stats
        finally:
            for module_name in loaded_modules:
                sys.modules.pop(module_name, None)
            try:
                os.chdir(previous_cwd)
            except OSError:
                pass
            if path_inserted:
                try:
                    sys.path.remove(workspace_abs)
                except ValueError:
                    pass

    @staticmethod
    def _test_module_names(temp_dir: str) -> List[str]:
        """Return importable module names for ``*.py`` files under temp_dir."""
        names: List[str] = []
        try:
            entries = os.listdir(temp_dir)
        except OSError:
            return names
        for entry in entries:
            if entry.endswith(".py") and entry != "__init__.py":
                names.append(os.path.splitext(entry)[0])
        return names

    @staticmethod
    def _format_execution_summary(stats: "_PytestExecutionStats") -> str:
        """Render a one-line (plus optional detail) execution summary."""
        line = (
            "Execution: "
            f"{stats.passed} passed, {stats.failed} failed, "
            f"{stats.skipped} skipped, {stats.errors} errors "
            f"in {stats.duration_seconds:.2f}s."
        )
        detail = (stats.detail or "").strip()
        if detail:
            # Keep detail short so TestingResult.summary stays compact.
            if len(detail) > 240:
                detail = detail[:237].rstrip() + "..."
            return f"{line} Detail: {detail}"
        return line

    def _measure_coverage(
        self,
        pytest_api: Any,
        *,
        workspace: str,
        temp_dir: str,
    ) -> _CoverageMeasurement:
        """
        Run pytest-cov against the generated tests and parse line coverage.

        Prefers the JSON coverage report. Never raises; unavailable tooling
        or parse failures return a structured unavailable measurement.
        """
        self._trace("testing_coverage_started", temp_dir=temp_dir)
        started = time.perf_counter()

        try:
            import pytest_cov as _pytest_cov  # noqa: F401
        except ImportError:
            measurement = _CoverageMeasurement(
                available=False,
                summary="Coverage: unavailable (pytest-cov not installed).",
                error="pytest-cov not installed",
            )
            self._trace(
                "testing_coverage_failed",
                success=False,
                error=measurement.error,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
            return measurement

        report_path = os.path.join(temp_dir, "coverage.json")
        cov_data_file = os.path.join(temp_dir, ".coverage")
        targets = self._coverage_targets(workspace)
        args = [
            temp_dir,
            "-q",
            "--tb=no",
            "--import-mode=importlib",
            "-p",
            "no:cacheprovider",
            f"--rootdir={temp_dir}",
            "-o",
            "addopts=",
            f"--cov-report=json:{report_path}",
            "--cov-report=term",
        ]
        for target in targets:
            args.extend(["--cov", target])

        previous_cov_file = os.environ.get("COVERAGE_FILE")
        path_inserted = False
        workspace_abs = os.path.abspath(workspace)
        previous_cwd = os.getcwd()
        loaded_modules = self._test_module_names(temp_dir)
        sink = io.StringIO()

        try:
            os.environ["COVERAGE_FILE"] = cov_data_file
            if workspace_abs not in sys.path:
                sys.path.insert(0, workspace_abs)
                path_inserted = True
            try:
                os.chdir(workspace_abs)
            except OSError:
                pass

            try:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(
                    sink
                ):
                    pytest_api.main(args)  # type: ignore[arg-type]
            except SystemExit:
                pass
            except Exception as exc:
                logger.warning("pytest-cov execution failed: %s", exc)
                measurement = _CoverageMeasurement(
                    available=True,
                    measured=False,
                    summary="Coverage: unavailable (coverage execution failed).",
                    error=str(exc),
                )
                self._trace(
                    "testing_coverage_failed",
                    success=False,
                    error=measurement.error,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                )
                return measurement

            measurement = self._parse_coverage_report(
                report_path, fallback_text=sink.getvalue()
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
            if measurement.measured:
                self._trace(
                    "testing_coverage_finished",
                    success=True,
                    duration_ms=duration_ms,
                    coverage_percent=measurement.percent,
                    files_measured=measurement.files_measured,
                    statements=measurement.statements,
                    missing=measurement.missing,
                )
            else:
                self._trace(
                    "testing_coverage_failed",
                    success=False,
                    error=measurement.error or "coverage report missing",
                    duration_ms=duration_ms,
                )
            return measurement
        finally:
            for module_name in loaded_modules:
                sys.modules.pop(module_name, None)
            try:
                os.chdir(previous_cwd)
            except OSError:
                pass
            if path_inserted:
                try:
                    sys.path.remove(workspace_abs)
                except ValueError:
                    pass
            if previous_cov_file is None:
                os.environ.pop("COVERAGE_FILE", None)
            else:
                os.environ["COVERAGE_FILE"] = previous_cov_file

    def _coverage_targets(self, workspace: str) -> List[str]:
        """
        Choose ``--cov`` targets for the workspace under test.

        Prefers top-level modules/packages; falls back to ``.`` when the
        workspace inventory is empty.
        """
        targets: List[str] = []
        try:
            entries = sorted(os.listdir(workspace))
        except OSError:
            return ["."]

        for entry in entries:
            if entry.startswith("."):
                continue
            path = os.path.join(workspace, entry)
            if entry.endswith(".py"):
                if self._should_skip_path(entry):
                    continue
                targets.append(entry[:-3])
            elif os.path.isdir(path) and entry not in _SKIP_DIR_NAMES:
                init_py = os.path.join(path, "__init__.py")
                if os.path.isfile(init_py):
                    targets.append(entry)
            if len(targets) >= 12:
                break
        return targets or ["."]

    @staticmethod
    def _parse_coverage_report(
        report_path: str,
        *,
        fallback_text: str = "",
    ) -> _CoverageMeasurement:
        """Parse coverage.py JSON output, with term-report fallback."""
        if report_path and os.path.isfile(report_path):
            try:
                with open(report_path, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                term = TestingAgent._parse_coverage_term(fallback_text)
                if term.measured:
                    return term
                return _CoverageMeasurement(
                    available=True,
                    measured=False,
                    summary="Coverage: unavailable (malformed coverage output).",
                    error=f"malformed coverage JSON: {exc}",
                )

            if not isinstance(payload, dict):
                return _CoverageMeasurement(
                    available=True,
                    measured=False,
                    summary="Coverage: unavailable (malformed coverage output).",
                    error="coverage JSON root was not an object",
                )

            totals = payload.get("totals") or {}
            try:
                percent = float(totals.get("percent_covered", 0.0) or 0.0)
            except (TypeError, ValueError):
                percent = 0.0
            try:
                statements = int(totals.get("num_statements", 0) or 0)
            except (TypeError, ValueError):
                statements = 0
            try:
                missing = int(totals.get("missing_lines", 0) or 0)
            except (TypeError, ValueError):
                missing = 0
            files = payload.get("files") or {}
            files_measured = len(files) if isinstance(files, dict) else 0
            summary = (
                f"Coverage: {percent:.0f}% line coverage\n"
                f"{files_measured} files measured"
            )
            if statements:
                summary = (
                    f"{summary}\n"
                    f"{statements} statements, {missing} missed"
                )
            return _CoverageMeasurement(
                available=True,
                measured=True,
                percent=percent,
                files_measured=files_measured,
                statements=statements,
                missing=missing,
                summary=summary,
            )

        term = TestingAgent._parse_coverage_term(fallback_text)
        if term.measured:
            return term
        return _CoverageMeasurement(
            available=True,
            measured=False,
            summary="Coverage: unavailable (coverage report missing).",
            error="coverage JSON report missing",
        )

    @staticmethod
    def _parse_coverage_term(text: str) -> _CoverageMeasurement:
        """Parse a pytest-cov terminal TOTAL line when JSON is unavailable."""
        if not text:
            return _CoverageMeasurement(available=True, measured=False)
        # Example: TOTAL                                      40     8    80%
        match = re.search(
            r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+(?:\.\d+)?)%",
            text,
        )
        if not match:
            return _CoverageMeasurement(
                available=True,
                measured=False,
                error="could not parse coverage terminal output",
            )
        try:
            statements = int(match.group(1))
            missing = int(match.group(2))
            percent = float(match.group(3))
        except (TypeError, ValueError):
            return _CoverageMeasurement(
                available=True,
                measured=False,
                error="could not parse coverage terminal totals",
            )
        file_lines = [
            line
            for line in text.splitlines()
            if line.strip()
            and not line.startswith("Name")
            and not line.startswith("-")
            and not line.startswith("TOTAL")
            and "%" in line
        ]
        files_measured = len(file_lines)
        summary = (
            f"Coverage: {percent:.0f}% line coverage\n"
            f"{files_measured} files measured"
        )
        if statements:
            summary = (
                f"{summary}\n"
                f"{statements} statements, {missing} missed"
            )
        return _CoverageMeasurement(
            available=True,
            measured=True,
            percent=percent,
            files_measured=files_measured,
            statements=statements,
            missing=missing,
            summary=summary,
        )

    @staticmethod
    def _cleanup_temp_test_dir(temp_dir: Optional[str]) -> None:
        """Remove the temporary test directory; never raise."""
        if not temp_dir:
            return
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as exc:
            logger.warning("Failed cleaning temp test dir %s: %s", temp_dir, exc)

    def _model_available(self) -> bool:
        """Report whether the injected model client can serve requests."""
        if self.model_client is None:
            return False
        try:
            return bool(self.model_client.is_available())
        except Exception as exc:
            logger.warning("Testing model availability check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # AST inventory + symbol-scoped generation
    # ------------------------------------------------------------------

    def _list_python_inventory(
        self,
        filesystem: FilesystemTools,
        target_path: str,
    ) -> List[str]:
        """
        List inventory Python modules eligible for test generation.

        Prefers a single target file when ``target_path`` points at one;
        otherwise lists repository ``*.py`` files with test/generated
        paths filtered out.
        """
        relative_target = self._relative_to_workspace(filesystem, target_path)
        if (
            relative_target
            and relative_target not in {".", ""}
            and relative_target.endswith(".py")
        ):
            try:
                if filesystem.file_exists(relative_target) and not self._should_skip_path(
                    relative_target
                ):
                    return [relative_target.replace("\\", "/")]
            except Exception as exc:
                logger.warning("Could not resolve target %s: %s", target_path, exc)

        try:
            files = filesystem.list_files(".", pattern="*.py", recursive=True)
        except Exception as exc:
            logger.warning("Could not list Python inventory: %s", exc)
            return []

        inventory: List[str] = []
        for path in files:
            normalized = str(path).replace("\\", "/")
            if self._should_skip_path(normalized):
                continue
            inventory.append(normalized)
            if len(inventory) >= _MAX_AST_FILES:
                break
        return inventory

    @staticmethod
    def _should_skip_path(path: str) -> bool:
        """Return True for tests/, caches, generated modules, and test files."""
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

    def _collect_testable_symbols(
        self,
        filesystem: FilesystemTools,
        inventory: Sequence[str],
    ) -> Tuple[List[_TestableSymbol], List[str]]:
        """
        Parse inventory modules with AST and collect public symbols.

        Returns:
            ``(symbols, skipped_labels)`` where skipped labels describe
            private/dunder/duplicate/empty-class omissions.
        """
        symbols: List[_TestableSymbol] = []
        skipped: List[str] = []
        seen: Set[Tuple[str, str, str]] = set()

        for module_path in inventory:
            try:
                source = filesystem.read_file(module_path)
            except Exception as exc:
                skipped.append(f"{module_path}:unreadable")
                logger.warning("AST scan skipped %s: %s", module_path, exc)
                continue
            try:
                tree = ast.parse(source or "", filename=module_path)
            except SyntaxError:
                skipped.append(f"{module_path}:syntax_error")
                continue

            lines = (source or "").splitlines()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if self._is_private_name(node.name) or self._is_dunder_name(
                        node.name
                    ):
                        skipped.append(f"{module_path}:{node.name}:private")
                        continue
                    key = (module_path, "function", node.name)
                    if key in seen:
                        skipped.append(f"{module_path}:{node.name}:duplicate")
                        continue
                    seen.add(key)
                    symbols.append(
                        _TestableSymbol(
                            kind="function",
                            name=node.name,
                            qualname=node.name,
                            module_path=module_path,
                            signature=self._format_callable_signature(node),
                            docstring=ast.get_docstring(node) or "",
                            source=self._slice_source(lines, node),
                        )
                    )
                    continue

                if isinstance(node, ast.ClassDef):
                    if self._is_private_name(node.name):
                        skipped.append(f"{module_path}:{node.name}:private_class")
                        continue
                    methods: List[_MethodInfo] = []
                    for member in node.body:
                        if not isinstance(
                            member, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ):
                            continue
                        if self._is_private_name(member.name) or self._is_dunder_name(
                            member.name
                        ):
                            skipped.append(
                                f"{module_path}:{node.name}.{member.name}:private"
                            )
                            continue
                        methods.append(
                            _MethodInfo(
                                name=member.name,
                                signature=self._format_callable_signature(member),
                                docstring=ast.get_docstring(member) or "",
                                source=self._slice_source(lines, member),
                            )
                        )
                    if not methods:
                        skipped.append(f"{module_path}:{node.name}:no_public_methods")
                        continue
                    key = (module_path, "class", node.name)
                    if key in seen:
                        skipped.append(f"{module_path}:{node.name}:duplicate")
                        continue
                    seen.add(key)
                    class_source = self._slice_source(lines, node)
                    symbols.append(
                        _TestableSymbol(
                            kind="class",
                            name=node.name,
                            qualname=node.name,
                            module_path=module_path,
                            signature=node.name,
                            docstring=ast.get_docstring(node) or "",
                            source=class_source,
                            methods=methods,
                        )
                    )
        return symbols, skipped

    @staticmethod
    def _is_private_name(name: str) -> bool:
        """True for single-underscore private names (not dunders)."""
        return bool(name) and name.startswith("_") and not (
            name.startswith("__") and name.endswith("__")
        )

    @staticmethod
    def _is_dunder_name(name: str) -> bool:
        """True for ``__init__``-style dunder names."""
        return bool(name) and name.startswith("__") and name.endswith("__") and len(name) > 4

    @staticmethod
    def _slice_source(lines: Sequence[str], node: ast.AST) -> str:
        """Return the source text spanning an AST node."""
        start = max(0, int(getattr(node, "lineno", 1) or 1) - 1)
        end = int(getattr(node, "end_lineno", None) or getattr(node, "lineno", start + 1))
        end = max(start + 1, end)
        return "\n".join(lines[start:end])

    @staticmethod
    def _format_callable_signature(node: ast.AST) -> str:
        """Render ``name(args)`` for a function/method node."""
        name = getattr(node, "name", "callable")
        args = getattr(node, "args", None)
        if args is None:
            return f"{name}()"
        try:
            rendered = TestingAgent._ast_unparse(args)
        except Exception:
            rendered = ""
        if not rendered:
            arg_names = [
                getattr(arg, "arg", "")
                for arg in getattr(args, "args", [])
                if getattr(arg, "arg", "")
            ]
            rendered = ", ".join(arg_names)
        return f"{name}({rendered})"

    @staticmethod
    def _ast_unparse(node: ast.AST) -> str:
        """Best-effort ``ast.unparse`` with a safe fallback."""
        unparse = getattr(ast, "unparse", None)
        if callable(unparse):
            return str(unparse(node))
        return ""

    def _build_symbol_prompt(
        self,
        *,
        instruction: str,
        symbol: _TestableSymbol,
        chunks: Sequence[RetrievedChunk],
        source_excerpts: Sequence[str],
    ) -> str:
        """Build a focused user prompt for one public symbol."""
        packed_chunks = self._dedupe_chunks(chunks)[:_MAX_CONTEXT_CHUNKS]
        sections = [
            "TESTING MODE\npytest unit test generation for one symbol",
            f"WRITING INSTRUCTIONS\n{_WRITING_INSTRUCTIONS}",
            f"REQUEST\n{instruction}",
            f"FOCUS\n{symbol.focus_instruction}",
            (
                "SYMBOL\n"
                f"kind={symbol.kind}\n"
                f"name={symbol.name}\n"
                f"module={symbol.module_path}\n"
                f"signature={symbol.signature}\n"
                f"docstring={symbol.docstring or '(none)'}"
            ),
        ]
        if symbol.kind == "class" and symbol.methods:
            # Metadata only here so retrieved context still precedes bodies.
            method_lines = [
                f"- {method.signature} :: {method.docstring.strip() or '(no docstring)'}"
                for method in symbol.methods
            ]
            sections.append("PUBLIC METHODS\n" + "\n".join(method_lines))

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
                "Rely on SYMBOL SOURCE / REPOSITORY CONTENTS only; "
                "still do not invent APIs."
            )

        # Symbol/module source comes after retrieval, matching the
        # repository-contents secondary evidence role.
        sections.append(f"SYMBOL SOURCE\n{symbol.source}")
        if source_excerpts:
            sections.append(
                "REPOSITORY CONTENTS (secondary - fill gaps only)\n"
                + "\n\n".join(source_excerpts)
            )
        else:
            sections.append("REPOSITORY CONTENTS\n(none)")

        sections.append(
            "OUTPUT CONTRACT\n"
            "Return only the TestingResult JSON object. "
            f"Prefer a single module named "
            f"`{self._test_filename_for_module(symbol.module_path)}`. "
            "Test ONLY the focused symbol. "
            "`generated_tests` values must be complete runnable pytest "
            "modules. Do not wrap the JSON in markdown fences."
        )
        return self._truncate_prompt("\n\n".join(sections))

    @staticmethod
    def _test_filename_for_module(module_path: str) -> str:
        """Map ``pkg/mod.py`` → ``test_mod.py``."""
        stem = os.path.splitext(os.path.basename(module_path or "module.py"))[0]
        stem = stem or "module"
        return f"test_{stem}.py"

    def _normalize_symbol_result(
        self,
        result: TestingResult,
        symbol: _TestableSymbol,
    ) -> TestingResult:
        """Force per-symbol outputs into one canonical test module filename."""
        if not result.generated_tests:
            return result
        canonical = self._test_filename_for_module(symbol.module_path)
        merged_source = self._merge_module_sources(list(result.generated_tests.values()))
        if not merged_source.strip():
            return self._empty_result()
        return TestingResult(
            summary=result.summary,
            generated_tests={canonical: merged_source},
            coverage_estimate=result.coverage_estimate,
            abstention=None,
        )

    def _merge_testing_results(
        self,
        results: Sequence[TestingResult],
    ) -> TestingResult:
        """Merge per-symbol TestingResult objects into one suite."""
        if not results:
            return self._empty_result()

        by_file: Dict[str, List[str]] = {}
        summaries: List[str] = []
        coverages: List[float] = []

        for result in results:
            if result.summary and result.summary.strip():
                summaries.append(result.summary.strip())
            try:
                coverages.append(float(result.coverage_estimate))
            except (TypeError, ValueError):
                pass
            for name, source in (result.generated_tests or {}).items():
                if not source or not str(source).strip():
                    continue
                by_file.setdefault(name, []).append(source)

        merged_tests = {
            name: self._merge_module_sources(sources)
            for name, sources in by_file.items()
            if sources
        }
        merged_tests = {
            name: source for name, source in merged_tests.items() if source.strip()
        }
        coverage = sum(coverages) / len(coverages) if coverages else 0.0
        summary = " ".join(summaries).strip() or (
            f"Generated tests for {len(results)} public symbol(s)."
        )
        return TestingResult(
            summary=summary,
            generated_tests=merged_tests,
            coverage_estimate=min(max(coverage, 0.0), 1.0),
            abstention=None,
        )

    @classmethod
    def _merge_module_sources(cls, sources: Sequence[str]) -> str:
        """
        Merge pytest module sources: unique imports, fixtures, and tests.

        Uses AST when possible so duplicate ``test_*`` names and import
        statements are dropped while preserving order.
        """
        usable = [str(source) for source in sources if source and str(source).strip()]
        if not usable:
            return ""
        if len(usable) == 1:
            return usable[0]

        import_nodes: List[ast.AST] = []
        body_nodes: List[ast.AST] = []
        seen_imports: Set[str] = set()
        seen_defs: Set[str] = set()
        module_doc: Optional[str] = None
        fell_back = False

        for source in usable:
            try:
                tree = ast.parse(source)
            except SyntaxError:
                fell_back = True
                break
            if module_doc is None:
                module_doc = ast.get_docstring(tree)
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(
                    getattr(node, "value", None), ast.Constant
                ):
                    # Skip module docstring expression; re-emit later.
                    if isinstance(node.value.value, str) and module_doc == node.value.value:
                        continue
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    key = cls._ast_unparse(node) or ast.dump(node)
                    if key in seen_imports:
                        continue
                    seen_imports.add(key)
                    import_nodes.append(node)
                    continue
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name in seen_defs:
                        continue
                    seen_defs.add(node.name)
                    body_nodes.append(node)
                    continue
                # Keep other top-level statements (assignments, etc.) once.
                key = cls._ast_unparse(node) or ast.dump(node)
                if key in seen_imports:
                    continue
                seen_imports.add(key)
                body_nodes.append(node)

        if fell_back:
            return cls._merge_module_sources_textual(usable)

        parts: List[str] = []
        if module_doc:
            parts.append(f'"""{module_doc}"""')
        for node in import_nodes:
            rendered = cls._ast_unparse(node)
            if rendered:
                parts.append(rendered)
        if import_nodes and body_nodes:
            parts.append("")
        for node in body_nodes:
            rendered = cls._ast_unparse(node)
            if rendered:
                parts.append(rendered)
                parts.append("")
        return "\n".join(parts).rstrip() + ("\n" if parts else "")

    @staticmethod
    def _merge_module_sources_textual(sources: Sequence[str]) -> str:
        """Regex-based merge fallback when AST parsing fails."""
        imports: List[str] = []
        seen_imports: Set[str] = set()
        tests: List[str] = []
        seen_tests: Set[str] = set()
        other: List[str] = []

        for source in sources:
            blocks = re.split(r"\n(?=def\s+|async\s+def\s+|class\s+|@)", source)
            for block in blocks:
                text = block.strip("\n")
                if not text.strip():
                    continue
                first = text.lstrip().splitlines()[0] if text.lstrip() else ""
                if first.startswith("import ") or first.startswith("from "):
                    for line in text.splitlines():
                        stripped = line.strip()
                        if (
                            stripped.startswith("import ")
                            or stripped.startswith("from ")
                        ) and stripped not in seen_imports:
                            seen_imports.add(stripped)
                            imports.append(stripped)
                    continue
                match = _TEST_DEF.search(text)
                if match:
                    name = match.group(1)
                    if name in seen_tests:
                        continue
                    seen_tests.add(name)
                    tests.append(text)
                    continue
                other.append(text)

        parts = imports + ([""] if imports and (other or tests) else []) + other + tests
        return "\n\n".join(parts).rstrip() + ("\n" if parts else "")

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

    def _build_repair_prompt(
        self,
        *,
        instruction: str,
        target_path: str,
        original_tests: Dict[str, str],
        pytest_output: str,
        first_summary: str,
        chunks: Sequence[RetrievedChunk],
        source_excerpts: Sequence[str],
    ) -> str:
        """
        Build the user prompt for a single test-repair model call.

        The pytest failure output is the primary debugging signal; the
        original sources and repository context ground the fix.
        """
        packed_chunks = self._dedupe_chunks(chunks)[:_MAX_CONTEXT_CHUNKS]
        sections = [
            "TESTING REPAIR MODE\nFix failing pytest modules only.",
            (
                "REPAIR RULES\n"
                "- Fix only failing tests; preserve passing tests.\n"
                "- Use PYTEST FAILURE OUTPUT as the primary debugging signal.\n"
                "- Do not invent nonexistent modules or APIs.\n"
                "- Return only valid pytest source inside TestingResult JSON."
            ),
            f"REQUEST\n{instruction}",
            f"TARGET\n{target_path}",
            f"FIRST EXECUTION SUMMARY\n{first_summary or '(none)'}",
        ]

        failure_text = (pytest_output or "").strip() or "(no pytest output captured)"
        if len(failure_text) > _MAX_PYTEST_OUTPUT_CHARS:
            failure_text = (
                failure_text[:_MAX_PYTEST_OUTPUT_CHARS]
                + "\n...[truncated pytest output]"
            )
        sections.append(f"PYTEST FAILURE OUTPUT\n{failure_text}")

        rendered_tests: List[str] = []
        budget = _MAX_REPAIR_TEST_CHARS
        for name, source in original_tests.items():
            body = source if source is not None else ""
            if len(body) > budget:
                body = body[:budget] + "\n...[truncated test source]"
                budget = 0
            else:
                budget = max(0, budget - len(body))
            rendered_tests.append(f"### {name}\n{body}")
            if budget == 0:
                break
        sections.append(
            "ORIGINAL GENERATED TESTS\n" + "\n\n".join(rendered_tests)
        )

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
                "RETRIEVED CONTEXT\n" + "\n\n".join(rendered)
            )
        else:
            sections.append("RETRIEVED CONTEXT\n(none)")

        if source_excerpts:
            sections.append(
                "REPOSITORY CONTENTS\n" + "\n\n".join(source_excerpts)
            )
        else:
            sections.append("REPOSITORY CONTENTS\n(none)")

        sections.append(
            "OUTPUT CONTRACT\n"
            "Return only the TestingResult JSON object with repaired "
            "`generated_tests` values. Do not wrap the JSON in markdown fences."
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
            abstention=None,
        )

    @staticmethod
    def _abstain_result(
        *,
        reason: str,
        evidence_available: Optional[List[str]] = None,
        recommended_next_steps: Optional[List[str]] = None,
    ) -> TestingResult:
        """Build a TestingResult that carries an explicit abstention."""
        abstention: AbstentionResult = ReportBuilder().abstain(
            reason,
            confidence=1.0,
            evidence_available=evidence_available,
            recommended_next_steps=recommended_next_steps,
        )
        return TestingResult(
            summary="",
            generated_tests={},
            coverage_estimate=0.0,
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
