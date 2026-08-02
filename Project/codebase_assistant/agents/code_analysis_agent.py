"""
code_analysis_agent.py
=======================

The first agent that runs the complete pipeline end to end.

Joins five components that until now only existed side by side:

    Indexer  ->  Retriever  ->  StaticAnalyzer  ->  GroundingChecker
                                                          ^
                                     ModelClient ---------+

The order matters. Deterministic analysis runs first and its findings go
into the prompt, so the model is told what is already known rather than
left to rediscover it. Everything the model proposes then goes back
through the same grounding gate the static findings passed, and anything
that fails is dropped.

The guarantee this module makes is narrow and absolute: **no finding
reaches the caller without its evidence having been verified against the
real source.** A model finding and a pyflakes finding are put through
the identical check. That is why `CodeAnalysisReport.findings` can be
trusted and `rejected` is kept separately rather than merged in with a
lower score.

The agent degrades rather than fails. No provider configured, the model
unreachable, an unparseable response, an empty index -- each of these
costs the LLM half of the analysis and leaves the static half intact,
because a partial answer built from verified findings is worth more than
an exception.

TODO: Emit each stage to the tracing layer once `tracing/` is
implemented; the pipeline is currently observable only through logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..analysis.grounding_checker import GroundingChecker, GroundingResult
from ..analysis.static_analyzer import AnalysisReport as StaticAnalysisReport
from ..analysis.static_analyzer import StaticAnalyzer
from ..config import Config
from ..exceptions.base import CodebaseAssistantError
from ..memory.memory_store import MemoryStore
from ..models.model_client import LLMClient
from ..rag.indexer import IndexUpdate, Indexer
from ..rag.retriever import Retriever
from ..schemas.schemas import (
    AgentRequest,
    AgentResponse,
    AgentType,
    BugReport,
    CodeAnalysisResult,
    ModelMessage,
    RetrievedChunk,
)
from ..tools.filesystem_tools import FilesystemTools
from ..tools.registry import ToolRegistry
from .base import BaseAgent

logger = logging.getLogger(__name__)

#: Ordering for severity, highest first.
SEVERITY_RANK: Dict[str, int] = {"high": 3, "medium": 2, "low": 1}

#: Ceiling applied to every confidence a model reports. A model asserting
#: 0.99 has expressed a mood, not a measurement, so its findings are held
#: below the least certain deterministic check. This is what "static
#: findings outrank LLM findings" means in practice.
MAX_LLM_CONFIDENCE = 0.80

#: Used when a caller asks for analysis without asking a question.
DEFAULT_QUESTION = "Find likely bugs and correctness problems in this code."

#: How many static findings to show the model. Enough to stop it
#: re-reporting what is already known, not so many that they crowd out
#: the code itself.
MAX_STATIC_IN_PROMPT = 25

#: Matches the line-number gutter this module renders into prompts, so a
#: model that copies it into its evidence can be forgiven.
_GUTTER = re.compile(r"^\s*\d+\s*\|\s?")

#: Fenced code block around a JSON payload.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

SYSTEM_PROMPT = """\
You are a precise code analysis assistant. You find real bugs in real \
code, and you never invent them.

You will be given code from a repository. Every line is shown with its \
real line number in a left gutter, like:

   42 | def withdraw(self, amount):

The `42 | ` gutter is NOT part of the code. Never include it in evidence.

Rules you must follow exactly:

1. Only report problems visible in the CODE CONTEXT you are given. If \
you cannot see it, it does not exist.
2. `evidence` must be copied character for character from the code \
shown. Do not re-indent it, do not reformat it, do not fix it, do not \
abbreviate it. It is a quotation, not a description.
3. `line_start` and `line_end` must be the real line numbers from the \
gutter, and `evidence` must be exactly the lines in that range.
4. Do not repeat anything listed under KNOWN STATIC FINDINGS. Those are \
already confirmed.
5. If the context is not enough to answer, say so in `answer` and return \
an empty `findings` list. Abstaining is a correct answer. Guessing is \
not.
6. Reply with ONE JSON object and nothing else -- no prose before it, no \
prose after it.

Reply in exactly this shape:

{
  "answer": "Your prose answer to the question.",
  "findings": [
    {
      "bug_type": "short_snake_case_category",
      "description": "What is wrong and why it matters.",
      "severity": "low | medium | high",
      "confidence": 0.0,
      "file_path": "path/as/shown/in/the/header.py",
      "function_name": "enclosing_function_or_<module>",
      "line_start": 1,
      "line_end": 1,
      "evidence": "the exact source lines, copied verbatim",
      "suggested_fix": "How to fix it, or null."
    }
  ]
}

Every finding you return is checked against the real file. Any finding \
whose evidence does not match the source exactly is discarded, and a \
discarded finding helps no one."""


@dataclass
class CodeAnalysisReport:
    """
    The result of one full analysis run.

    Named `CodeAnalysisReport` rather than `AnalysisReport` because
    `static_analyzer.AnalysisReport` already exists and means something
    narrower -- the deterministic pass alone. Two types with the same
    name in one package, one of them a field of the other, would be a
    standing trap.

    Attributes:
        repository_path: Repository that was analyzed.
        question: The question that drove the run.
        findings: Verified findings, deduplicated, most severe first.
            Every one has had its evidence checked against the source.
        answer: The model's prose answer, empty when no model ran.
        rejected: Verdicts on findings that failed grounding. Kept so
            the hallucination rate is measurable; never merged into
            `findings`.
        context: Chunks retrieved and shown to the model.
        static_report: The deterministic pass's own result, including
            files it skipped.
        index_update: What indexing did, or None if it was skipped.
        model_used: Whether a model actually contributed.
        duplicates_removed: Findings dropped as restatements.
        notes: Anything that degraded the run -- no provider, empty
            index, unparseable response. Read this before trusting an
            empty result.
        duration_seconds: Wall-clock duration.
    """

    repository_path: str = ""
    question: str = ""
    findings: List[BugReport] = field(default_factory=list)
    answer: str = ""
    rejected: List[GroundingResult] = field(default_factory=list)
    context: List[RetrievedChunk] = field(default_factory=list)
    static_report: Optional[StaticAnalysisReport] = None
    index_update: Optional[IndexUpdate] = None
    model_used: bool = False
    duplicates_removed: int = 0
    notes: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def static_findings(self) -> List[BugReport]:
        """Verified findings that came from the deterministic pass."""
        return [f for f in self.findings if f.detection_method == "static"]

    @property
    def llm_findings(self) -> List[BugReport]:
        """Verified findings that came from the model."""
        return [f for f in self.findings if f.detection_method == "llm"]

    def by_severity(self) -> Dict[str, int]:
        """
        Count findings by severity.

        Returns:
            A mapping of severity to count, most severe first.
        """
        counts: Dict[str, int] = {}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return dict(
            sorted(counts.items(), key=lambda item: -SEVERITY_RANK.get(item[0], 0))
        )

    def summary(self) -> str:
        """
        Render a one-line summary of the run.

        Returns:
            A readable summary suitable for logs, a notebook cell, or
            the Supervisor's `run` payload.
        """
        parts = [
            f"{len(self.findings)} verified finding(s)",
            f"{len(self.static_findings)} static",
            f"{len(self.llm_findings)} llm",
        ]
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected as ungrounded")
        if self.duplicates_removed:
            parts.append(f"{self.duplicates_removed} duplicate(s) merged")
        if not self.model_used:
            parts.append("static-only")
        return ", ".join(parts)


@dataclass
class _Pipeline:
    """
    The collaborators bound to one repository for one run.

    Attributes:
        root: Absolute path of the repository they are rooted at.
        filesystem: Sandboxed file access.
        analyzer: The deterministic pass.
        checker: The grounding gate.
        indexer: Index builder, or None when RAG is not in play.
        retriever: Semantic search, or None when RAG is not in play.
    """

    root: str
    filesystem: FilesystemTools
    analyzer: StaticAnalyzer
    checker: GroundingChecker
    indexer: Optional[Indexer] = None
    retriever: Optional[Retriever] = None


class CodeAnalysisAgent(BaseAgent):
    """
    Agent specialized in analyzing source code for structure, quality,
    complexity, and potential issues.
    """

    agent_type: AgentType = AgentType.CODE_ANALYSIS

    def __init__(
        self,
        model_client: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        retriever: Optional[Retriever] = None,
        memory_store: Optional[MemoryStore] = None,
        config: Optional[Config] = None,
        indexer: Optional[Indexer] = None,
        static_analyzer: Optional[StaticAnalyzer] = None,
        grounding_checker: Optional[GroundingChecker] = None,
        filesystem: Optional[FilesystemTools] = None,
    ) -> None:
        """
        Initialize the agent and record its collaborators.

        Nothing is constructed or loaded here. The Supervisor builds
        every agent during its own startup, so this must stay free of
        model loads, vector store connections, and filesystem checks.

        Every collaborator is optional and injected. Anything supplied
        is used exactly as given, for every repository, and is never
        rebuilt. Anything omitted is built on demand and rooted at the
        repository being analyzed.

        Args:
            model_client: Client used to make model calls. Without one
                the agent runs static-only.
            tool_registry: Registry used to invoke tools.
            retriever: Semantic search over the index.
            memory_store: Long-term memory store.
            config: Optional Config instance. A default is loaded when
                not supplied.
            indexer: Index builder. Built per repository when omitted.
            static_analyzer: The deterministic pass. Built per
                repository when omitted.
            grounding_checker: The verification gate. Built per
                repository when omitted.
            filesystem: Sandboxed file access. Built per repository when
                omitted.
        """
        super().__init__(
            model_client=model_client,
            tool_registry=tool_registry,
            retriever=retriever,
            memory_store=memory_store,
        )
        self.config = config or Config.load()
        self._indexer = indexer
        self._static_analyzer = static_analyzer
        self._grounding_checker = grounding_checker
        self._filesystem = filesystem
        self._bound: Dict[str, _Pipeline] = {}
        self._last_root: Optional[str] = None

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def analyze_repository(
        self,
        repository_path: str = ".",
        question: Optional[str] = None,
        use_rag: bool = True,
        top_k: Optional[int] = None,
    ) -> CodeAnalysisReport:
        """
        Run the full analysis pipeline over a repository.

        The stages, in order: index, static analysis, verify the static
        findings, retrieve context, prompt the model, parse it, verify
        what it proposed, merge, deduplicate, sort.

        Static analysis runs before the model call and its results are
        put into the prompt. That ordering is the point of the design --
        it stops the model spending its attention rediscovering unused
        imports, and it means a model failure costs only the model's
        contribution.

        Note this returns a `CodeAnalysisReport`, where the scaffold
        returned a placeholder `CodeAnalysisResult`. The old shape --
        a summary string and a list of issue strings -- cannot express a
        verified finding with a file, a line range, and quoted evidence.
        `analyze_file` still returns `CodeAnalysisResult`.

        Args:
            repository_path: Repository to analyze. Interpreted relative
                to an injected collaborator's workspace root if there is
                one, otherwise treated as the root itself.
            question: What to look for. Defaults to a general bug hunt.
            use_rag: When False, indexing and retrieval are skipped
                entirely. The model still runs, with static findings but
                no retrieved code.
            top_k: Chunks to retrieve. Defaults to
                `Config.retrieval_top_k`.

        Returns:
            The verified findings and everything about how they were
            reached.

        Raises:
            ValueError: If `repository_path` is empty.
            ToolExecutionError: If the repository does not exist.
            PathOutsideWorkspaceError: If it escapes an injected
                collaborator's workspace.
        """
        started = time.time()
        question = (question or DEFAULT_QUESTION).strip() or DEFAULT_QUESTION

        report = CodeAnalysisReport(
            repository_path=repository_path, question=question
        )
        pipeline = self._bind(repository_path)
        scope = self._scope(pipeline, repository_path)

        # Retrieval exists to fill the prompt. With no model to prompt,
        # indexing a repository would be minutes of embedding work whose
        # only consumer is absent, so it is skipped rather than wasted.
        if use_rag and not self._model_will_run():
            use_rag = False
            logger.info("No model available; skipping indexing and retrieval.")

        logger.info(
            "Analyzing %s (question=%r, rag=%s)", pipeline.root, question, use_rag
        )

        # 1-2. Index, or bring an existing index up to date.
        if use_rag:
            report.index_update = self._sync_index(pipeline, scope, report)

        # 3-4. Deterministic findings, then verify them like any other.
        static_findings = self._run_static(pipeline, scope, report)

        # 5. Context for the model.
        if use_rag and pipeline.retriever is not None:
            report.context = self._gather(
                pipeline, question, scope, top_k, report
            )

        # 6-10. The model half, which is allowed to contribute nothing.
        llm_findings = self._run_model(
            pipeline, question, report.context, static_findings, report
        )

        # 11-13. One ordered, deduplicated set.
        merged, removed = self._merge(static_findings, llm_findings)
        report.findings = merged
        report.duplicates_removed = removed
        report.duration_seconds = time.time() - started

        logger.info("Analysis complete: %s", report.summary())
        return report

    def analyze_query(
        self,
        repository_path: str,
        question: str,
        top_k: Optional[int] = None,
    ) -> CodeAnalysisReport:
        """
        Answer a natural language question about a repository.

        The same pipeline as `analyze_repository`, driven by the user's
        own words. Retrieval is what makes the difference: "where are
        SQL injections possible" and "explain the authentication flow"
        pull completely different code into the prompt, so the model
        sees the parts of the repository the question is actually about.

        Questions that are not bug hunts are expected and supported.
        "Explain the authentication flow" should produce a populated
        `answer` and an empty `findings` list, and that is a complete
        response, not a failure.

        Args:
            repository_path: Repository to analyze.
            question: The natural language question.
            top_k: Chunks to retrieve. Defaults to
                `Config.retrieval_top_k`.

        Returns:
            The answer, plus any verified findings the question turned
            up.

        Raises:
            ValueError: If `question` or `repository_path` is empty.
            ToolExecutionError: If the repository does not exist.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        return self.analyze_repository(
            repository_path=repository_path,
            question=question,
            use_rag=True,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Scaffold interface
    # ------------------------------------------------------------------

    def run(self, repo_path: str) -> Dict[str, str]:
        """
        Simple entry point used by the Supervisor for direct routing.

        Returns a status dict rather than a report because that is what
        the Supervisor and `app/main.py` consume. Failures are caught
        and described here instead of propagating: the Supervisor's
        contract is a dict with a message, and a routing demo should not
        crash because a path does not exist.

        Args:
            repo_path: Path to the repository to analyze.

        Returns:
            A dict with "status" and "message" keys.
        """
        try:
            report = self.analyze_repository(repo_path)
        except (CodebaseAssistantError, ValueError) as exc:
            logger.warning("Analysis of %r failed: %s", repo_path, exc)
            return {
                "status": "error",
                "message": f"Could not analyze {repo_path!r}: {exc}",
            }

        return {"status": "success", "message": report.summary()}

    def handle(self, request: AgentRequest) -> AgentResponse:
        """
        Handle a code analysis request.

        Args:
            request: The request describing what to analyze. The
                repository is read from `context["repository_path"]`,
                falling back to `context["repo_path"]` and then the
                current directory; `instruction` becomes the question.

        Returns:
            A response whose `output` is the CodeAnalysisReport, or
            whose `errors` explain why there is none.
        """
        context = request.context or {}
        repository_path = (
            context.get("repository_path") or context.get("repo_path") or "."
        )

        try:
            report = self.analyze_repository(
                repository_path=repository_path,
                question=request.instruction,
                use_rag=bool(context.get("use_rag", True)),
                top_k=context.get("top_k"),
            )
        except (CodebaseAssistantError, ValueError) as exc:
            logger.warning("Request %s failed: %s", request.task_id, exc)
            return AgentResponse(
                task_id=request.task_id,
                agent_type=self.agent_type,
                success=False,
                output=None,
                errors=[str(exc)],
            )

        return AgentResponse(
            task_id=request.task_id,
            agent_type=self.agent_type,
            success=True,
            output=report,
            errors=list(report.notes),
        )

    def analyze_file(
        self, file_path: str, repository_path: Optional[str] = None
    ) -> CodeAnalysisResult:
        """
        Analyze a single file for issues and metrics.

        Deterministic only. One file is not enough context for the model
        to say anything the static pass cannot, and calling it here
        would make a cheap operation slow and non-repeatable.

        Args:
            file_path: File to analyze, relative to the repository root.
            repository_path: Repository the file belongs to. Defaults to
                the last repository analyzed, so the common sequence --
                analyze a repository, then ask about one of its files --
                works without repeating the path.

        Returns:
            A result whose issues are rendered from verified findings.
        """
        if not file_path or not str(file_path).strip():
            return CodeAnalysisResult(
                summary="No file path given.", issues=[], metrics={}
            )

        try:
            pipeline = self._bind(repository_path or self._default_root())
        except (CodebaseAssistantError, ValueError) as exc:
            return CodeAnalysisResult(
                summary=f"Could not analyze {file_path}: {exc}",
                issues=[],
                metrics={},
            )

        try:
            findings = pipeline.analyzer.analyze_file(file_path)
        except CodebaseAssistantError as exc:
            logger.info("Could not analyze %s: %s", file_path, exc)
            return CodeAnalysisResult(
                summary=f"Could not analyze {file_path}: {exc}",
                issues=[],
                metrics={},
            )

        verified = pipeline.checker.verify_reports(findings)
        counts: Dict[str, Any] = {"findings": len(verified.grounded)}
        for finding in verified.grounded:
            key = f"severity_{finding.severity}"
            counts[key] = counts.get(key, 0) + 1

        return CodeAnalysisResult(
            summary=(
                f"{len(verified.grounded)} verified issue(s) in {file_path}."
                if verified.grounded
                else f"No issues detected in {file_path}."
            ),
            issues=[
                f"L{f.line_start}: [{f.severity}] {f.bug_type} - {f.description}"
                for f in verified.grounded
            ],
            metrics=counts,
        )

    def detect_code_smells(
        self, file_path: str, repository_path: Optional[str] = None
    ) -> List[str]:
        """
        Detect potential code smells within a file.

        Args:
            file_path: File to inspect, relative to the repository root.
            repository_path: Repository the file belongs to. Defaults to
                the last repository analyzed.

        Returns:
            One description per smell, from the static pass's
            deterministic quality checks.
        """
        try:
            pipeline = self._bind(repository_path or self._default_root())
            findings = pipeline.analyzer.check_code_quality(file_path)
        except (CodebaseAssistantError, ValueError) as exc:
            logger.info("Could not inspect %s: %s", file_path, exc)
            return []

        return [
            f"L{f.line_start}: {f.bug_type} - {f.description}" for f in findings
        ]

    def gather_context(self, query: str) -> List[RetrievedChunk]:
        """
        Gather relevant context via the retriever.

        Args:
            query: Query to search for.

        Returns:
            The retrieved chunks, or an empty list if no retriever is
            configured or the index is empty.
        """
        if self.retriever is None:
            logger.warning("gather_context called with no retriever configured.")
            return []

        try:
            return self.retriever.retrieve(query)
        except CodebaseAssistantError as exc:
            logger.warning("Retrieval failed for %r: %s", query, exc)
            return []

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _sync_index(
        self, pipeline: _Pipeline, scope: str, report: CodeAnalysisReport
    ) -> Optional[IndexUpdate]:
        """
        Bring the index up to date, building it if it does not exist.

        `Indexer.update_index` covers both cases: with no manifest every
        file reads as new, so the first call is a full build and every
        later one only touches what changed.

        Args:
            pipeline: The bound collaborators.
            scope: Directory to index, relative to the repository root.
            report: Report to record notes on.

        Returns:
            What the update did, or None if it could not run.
        """
        if pipeline.indexer is None:
            return None

        try:
            update = pipeline.indexer.update_index(scope)
        except CodebaseAssistantError as exc:
            # Embedding model or vector store unavailable. Retrieval
            # will find nothing, the prompt loses its code context, and
            # static analysis carries the run.
            note = f"Indexing failed, continuing without retrieval: {exc}"
            logger.warning(note)
            report.notes.append(note)
            return None

        logger.info("Index: %s", update.summary())
        return update

    def _run_static(
        self, pipeline: _Pipeline, scope: str, report: CodeAnalysisReport
    ) -> List[BugReport]:
        """
        Run the deterministic pass and verify its findings.

        Static findings are verified rather than trusted. They are
        expected to pass -- StaticAnalyzer quotes exact source slices --
        so a rejection here means the file changed between analysis and
        verification, which is worth catching.

        Args:
            pipeline: The bound collaborators.
            scope: Directory to analyze, relative to the repository
                root.
            report: Report to record the static result and notes on.

        Returns:
            The grounded static findings.
        """
        static = pipeline.analyzer.analyze_repository_detailed(scope)
        report.static_report = static
        logger.info("Static analysis: %s", static.summary())

        if not static.findings:
            return []

        # Hash the cited files now, so an edit between this point and
        # verification is detected rather than silently tolerated.
        pipeline.checker.snapshot_reports(static.findings)
        verified = pipeline.checker.verify_reports(static.findings)
        report.rejected.extend(r for r in verified.results if not r.grounded)

        if verified.rejected:
            note = (
                f"{len(verified.rejected)} static finding(s) failed grounding; "
                f"the source may have changed mid-run."
            )
            logger.warning(note)
            report.notes.append(note)

        return verified.grounded

    def _gather(
        self,
        pipeline: _Pipeline,
        question: str,
        scope: str,
        top_k: Optional[int],
        report: CodeAnalysisReport,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the code the question is about.

        Args:
            pipeline: The bound collaborators.
            question: The user's question, used as the query.
            scope: Directory being analyzed, unused for filtering but
                kept for the log line.
            top_k: Chunks to retrieve.
            report: Report to record notes on.

        Returns:
            The retrieved chunks, empty if retrieval was not possible.
        """
        if pipeline.retriever is None:
            return []

        try:
            chunks = pipeline.retriever.retrieve(question, top_k=top_k)
        except CodebaseAssistantError as exc:
            note = f"Retrieval failed, prompting without code context: {exc}"
            logger.warning(note)
            report.notes.append(note)
            return []

        if not chunks:
            note = (
                "Retrieval returned nothing; the model will see the static "
                "findings but no source."
            )
            logger.warning(note)
            report.notes.append(note)
            return []

        logger.info("Retrieved %d chunk(s) for %r in %s", len(chunks), question, scope)
        return chunks

    def _model_will_run(self) -> bool:
        """
        Report whether a model call is going to happen.

        Checked before indexing so the expensive half of the pipeline is
        not paid for when nothing will read its output.

        Returns:
            True if a client is configured and its provider is
            available.
        """
        if self.model_client is None:
            return False

        try:
            return bool(self.model_client.is_available())
        except Exception as exc:
            logger.warning("Provider availability check failed: %s", exc)
            return False

    def _run_model(
        self,
        pipeline: _Pipeline,
        question: str,
        context: Sequence[RetrievedChunk],
        static_findings: Sequence[BugReport],
        report: CodeAnalysisReport,
    ) -> List[BugReport]:
        """
        Prompt the model, parse it, and verify everything it claims.

        Every exit from this method leaves the static findings intact.
        A missing provider, an unreachable endpoint, a response that is
        not JSON, and a response full of invented code are all handled
        the same way: note it, return nothing, let the run continue.

        Args:
            pipeline: The bound collaborators.
            question: The user's question.
            context: Retrieved chunks to show the model.
            static_findings: Findings already confirmed, so the model
                does not repeat them.
            report: Report to record the answer, rejections, and notes.

        Returns:
            The grounded model findings.
        """
        if self.model_client is None:
            report.notes.append(
                "No model client configured; ran deterministic analysis only."
            )
            return []

        if not self._model_will_run():
            report.notes.append(
                "No model provider is available; ran deterministic analysis "
                "only. Inject a configured provider to enable LLM analysis."
            )
            return []

        prompt = self.build_prompt(question, context, static_findings)

        try:
            response = self.model_client.generate(
                [
                    ModelMessage(role="system", content=SYSTEM_PROMPT),
                    ModelMessage(role="user", content=prompt),
                ]
            )
        except Exception as exc:
            # Deliberately broad. A provider talks to the network and can
            # raise anything its transport raises, and no failure out
            # there is worth discarding verified static findings over.
            note = f"Model call failed, keeping static findings only: {exc}"
            logger.warning(note)
            report.notes.append(note)
            return []

        report.model_used = True
        answer, proposed = self.parse_response(response.content)
        report.answer = answer

        if not proposed:
            if not answer:
                note = (
                    "The model's response could not be parsed into findings; "
                    "keeping static findings only."
                )
                logger.warning(note)
                report.notes.append(note)
            else:
                logger.info("Model proposed no findings.")
            return []

        verified = pipeline.checker.verify_reports(proposed)
        report.rejected.extend(r for r in verified.results if not r.grounded)

        for result in verified.results:
            if not result.grounded:
                logger.warning(
                    "Discarded ungrounded model finding at %s:%d-%d (%s): %s",
                    result.file_path,
                    result.line_start,
                    result.line_end,
                    result.status.value,
                    result.reason,
                )

        if verified.rejected:
            note = (
                f"{len(verified.rejected)} of {len(proposed)} model finding(s) "
                f"were discarded as ungrounded."
            )
            report.notes.append(note)

        logger.info(
            "Model findings: %d proposed, %d grounded.",
            len(proposed),
            len(verified.grounded),
        )
        return verified.grounded

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        question: str,
        context: Sequence[RetrievedChunk],
        static_findings: Sequence[BugReport] = (),
    ) -> str:
        """
        Build the user half of the model prompt.

        Public because it is the part of the pipeline worth inspecting
        directly: what the model was shown explains what it returned.

        Code is rendered with a real line-number gutter. Without it the
        model has to guess line numbers, every guess is wrong, and
        grounding rejects the lot -- so the gutter is not decoration,
        it is what makes a verifiable citation possible at all.

        Args:
            question: The user's question.
            context: Retrieved chunks to show.
            static_findings: Findings already confirmed.

        Returns:
            The rendered prompt.
        """
        sections: List[str] = [f"QUESTION\n{question}"]

        if static_findings:
            shown = list(static_findings)[:MAX_STATIC_IN_PROMPT]
            lines = [
                f"- {f.file_path}:{f.line_start} [{f.severity}] "
                f"{f.bug_type}: {f.description}"
                for f in shown
            ]
            if len(static_findings) > len(shown):
                lines.append(f"- ... and {len(static_findings) - len(shown)} more")
            sections.append(
                "KNOWN STATIC FINDINGS (already confirmed -- do not repeat these)\n"
                + "\n".join(lines)
            )

        if context:
            sections.append("CODE CONTEXT\n" + self._render_context(context))
        else:
            sections.append(
                "CODE CONTEXT\n(none retrieved -- if you cannot see the "
                "relevant code, say so and return no findings)"
            )

        sections.append(
            "Answer the question, and report any bug you can see in the code "
            "above that is not already listed. Reply with one JSON object."
        )
        return "\n\n".join(sections)

    @staticmethod
    def _render_context(context: Sequence[RetrievedChunk]) -> str:
        """
        Render retrieved chunks with headers and a line-number gutter.

        Args:
            context: Chunks to render.

        Returns:
            The rendered code blocks.
        """
        blocks: List[str] = []

        for chunk in context:
            meta = chunk.metadata or {}
            start = int(meta.get("line_start", 1) or 1)
            end = int(meta.get("line_end", start) or start)
            name = meta.get("function_name") or meta.get("class_name") or ""
            label = f" ({name})" if name else ""

            numbered = "\n".join(
                f"{start + offset:>5} | {line}"
                for offset, line in enumerate(chunk.content.split("\n"))
            )
            blocks.append(
                f"--- {chunk.source} lines {start}-{end}{label} ---\n{numbered}"
            )

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(self, content: str) -> Tuple[str, List[BugReport]]:
        """
        Turn a model response into an answer and candidate findings.

        Tolerant by design. Models wrap JSON in fences, prefix it with
        "Here is the analysis:", and occasionally return a bare array.
        None of that is worth failing over, and none of it can smuggle
        an unverified finding through -- everything parsed here still
        has to survive grounding.

        A finding with missing or malformed fields is dropped
        individually with a log line, so one bad entry does not cost the
        rest.

        Args:
            content: Raw text from the model.

        Returns:
            The prose answer and the candidate BugReports.
        """
        payload = self._extract_json(content)

        if payload is None:
            logger.warning(
                "Model response contained no JSON object; treating the whole "
                "response as prose."
            )
            return content.strip(), []

        if isinstance(payload, list):
            payload = {"answer": "", "findings": payload}
        if not isinstance(payload, dict):
            return content.strip(), []

        answer = str(payload.get("answer") or "").strip()
        raw = payload.get("findings") or payload.get("bugs") or []
        if not isinstance(raw, list):
            logger.warning("Model returned a non-list `findings` field; ignoring.")
            return answer, []

        findings: List[BugReport] = []
        for index, entry in enumerate(raw):
            report = self._to_bug_report(entry, index)
            if report is not None:
                findings.append(report)

        return answer, findings

    def _to_bug_report(self, entry: Any, index: int) -> Optional[BugReport]:
        """
        Convert one parsed entry into a BugReport.

        Args:
            entry: A single item from the model's `findings` list.
            index: Its position, for the log line.

        Returns:
            The BugReport, or None if the entry is unusable.
        """
        if not isinstance(entry, dict):
            logger.warning("findings[%d] is not an object; dropped.", index)
            return None

        try:
            line_start = int(entry.get("line_start", 0))
            line_end = int(entry.get("line_end", line_start))
        except (TypeError, ValueError):
            logger.warning("findings[%d] has non-numeric line numbers; dropped.", index)
            return None

        severity = str(entry.get("severity", "medium")).strip().lower()
        if severity not in SEVERITY_RANK:
            severity = "medium"

        try:
            confidence = float(entry.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        file_path = str(entry.get("file_path") or "").strip()
        evidence = self._strip_gutter(str(entry.get("evidence") or ""))

        if not file_path or not evidence.strip():
            # Grounding would reject these anyway; dropping them here
            # keeps the rejection log about hallucinations rather than
            # about malformed output.
            logger.warning(
                "findings[%d] has no file path or no evidence; dropped.", index
            )
            return None

        try:
            return BugReport(
                bug_type=str(entry.get("bug_type") or "unspecified").strip(),
                description=str(entry.get("description") or "").strip()
                or "No description given.",
                severity=severity,  # type: ignore[arg-type]
                confidence=min(max(confidence, 0.0), MAX_LLM_CONFIDENCE),
                file_path=file_path,
                function_name=str(entry.get("function_name") or "<module>").strip(),
                line_start=line_start,
                line_end=max(line_end, line_start),
                evidence=evidence,
                suggested_fix=(
                    str(entry["suggested_fix"]).strip()
                    if entry.get("suggested_fix")
                    else None
                ),
                detection_method="llm",
            )
        except Exception as exc:
            logger.warning("findings[%d] could not be built: %s", index, exc)
            return None

    @staticmethod
    def _strip_gutter(evidence: str) -> str:
        """
        Remove the line-number gutter if the model copied it back.

        The gutter is this module's own rendering, so a quote carrying
        it is still a faithful quote of what the model was shown.
        Stripping it is undoing our own formatting, not relaxing the
        check -- the result still has to match the file exactly.

        Only applied when *every* non-blank line carries a gutter, so a
        line of real code that happens to contain a pipe is left alone.

        Args:
            evidence: Evidence as the model returned it.

        Returns:
            The evidence with any gutter removed.
        """
        lines = evidence.split("\n")
        candidates = [line for line in lines if line.strip()]

        if not candidates or not all(_GUTTER.match(line) for line in candidates):
            return evidence

        return "\n".join(_GUTTER.sub("", line) if line.strip() else line
                         for line in lines)

    @staticmethod
    def _extract_json(content: str) -> Optional[Any]:
        """
        Pull a JSON value out of a model response.

        Tries the whole string, then a fenced block, then the first
        balanced object or array found in the text.

        Args:
            content: Raw text from the model.

        Returns:
            The decoded value, or None if there is no JSON in it.
        """
        text = (content or "").strip()
        if not text:
            return None

        for candidate in [text] + [m.strip() for m in _FENCE.findall(text)]:
            try:
                return json.loads(candidate)
            except (ValueError, TypeError):
                continue

        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            if start == -1:
                continue

            depth = 0
            in_string = False
            escaped = False

            for position in range(start, len(text)):
                char = text[position]

                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : position + 1])
                        except (ValueError, TypeError):
                            break

        return None

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def _merge(
        self,
        static_findings: Sequence[BugReport],
        llm_findings: Sequence[BugReport],
    ) -> Tuple[List[BugReport], int]:
        """
        Combine both halves into one ordered, deduplicated set.

        Static findings are seeded first so that when the two halves
        describe the same defect, the deterministic account survives.
        Its category and line range come from the AST rather than from
        a model's reading of it.

        Two findings are the same defect when they cite the same file
        and the same bug type over overlapping lines. Requiring the bug
        type to match is what stops a genuine second bug on a shared
        line from being swallowed.

        Args:
            static_findings: Grounded findings from the static pass.
            llm_findings: Grounded findings from the model.

        Returns:
            The merged findings and how many were dropped as
            duplicates.
        """
        kept: List[BugReport] = []
        removed = 0

        for finding in list(static_findings) + list(llm_findings):
            duplicate = next(
                (other for other in kept if self._same_defect(other, finding)), None
            )
            if duplicate is None:
                kept.append(finding)
                continue

            removed += 1
            logger.debug(
                "Merged duplicate %s at %s:%d into the %s finding.",
                finding.bug_type,
                finding.file_path,
                finding.line_start,
                duplicate.detection_method,
            )

        kept.sort(
            key=lambda f: (
                -SEVERITY_RANK.get(f.severity, 0),
                -f.confidence,
                f.file_path,
                f.line_start,
            )
        )
        return kept, removed

    @staticmethod
    def _same_defect(left: BugReport, right: BugReport) -> bool:
        """
        Decide whether two findings describe the same defect.

        Args:
            left: An already-kept finding.
            right: A candidate.

        Returns:
            True if they cite the same file and bug type over
            overlapping lines.
        """
        return (
            left.file_path == right.file_path
            and left.bug_type.lower() == right.bug_type.lower()
            and left.line_start <= right.line_end
            and right.line_start <= left.line_end
        )

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _bind(self, repository_path: str) -> _Pipeline:
        """
        Resolve the collaborators to use for one repository.

        Injected collaborators are returned untouched, whatever
        repository is named -- the caller wired them deliberately and
        rebuilding them would discard that. Only the gaps are filled,
        rooted at the repository, and the result is cached so repeated
        calls do not reopen the vector store.

        Args:
            repository_path: Repository being analyzed.

        Returns:
            The bound collaborators.

        Raises:
            ValueError: If `repository_path` is empty.
            ToolExecutionError: If the repository does not exist.
        """
        if not repository_path or not str(repository_path).strip():
            raise ValueError("repository_path must be a non-empty string.")

        root = self._root_for(repository_path)
        self._last_root = root

        cached = self._bound.get(root)
        if cached is not None:
            return cached

        filesystem = self._filesystem or self._filesystem_tools(
            root, config=self.config
        )
        pipeline = _Pipeline(
            root=root,
            filesystem=filesystem,
            analyzer=self._static_analyzer
            or StaticAnalyzer(
                workspace_root=root, config=self.config, filesystem=filesystem
            ),
            checker=self._grounding_checker
            or GroundingChecker(
                workspace_root=root, config=self.config, filesystem=filesystem
            ),
            indexer=self._indexer
            or Indexer(
                vector_store_path=self._store_for(root),
                config=self.config,
                workspace_root=root,
            ),
        )

        # The Retriever must read the collection the Indexer wrote, so
        # it adopts that Indexer rather than being pointed at a path.
        pipeline.retriever = self.retriever or Retriever(
            config=self.config, indexer=pipeline.indexer
        )

        self._bound[root] = pipeline
        return pipeline

    def _default_root(self) -> str:
        """
        The repository to use when a caller does not name one.

        Args:
            None.

        Returns:
            The last repository analyzed, or the current directory if
            none has been.
        """
        return self._last_root or "."

    def _root_for(self, repository_path: str) -> str:
        """
        Decide which directory the collaborators should be rooted at.

        An injected collaborator already has a workspace root, and a
        path handed to this agent is then relative to it. Without one,
        the repository is its own root.

        Args:
            repository_path: Repository being analyzed.

        Returns:
            An absolute directory path.
        """
        injected = self._filesystem or self._static_analyzer or self._grounding_checker
        if injected is not None:
            return os.path.abspath(str(injected.workspace_root))
        return os.path.abspath(repository_path)

    def _store_for(self, root: str) -> str:
        """
        Choose a vector store directory for a repository.

        Each repository gets its own subdirectory, keyed by a hash of
        its path. Sharing one collection between repositories would let
        chunks from one answer questions about another, which is a
        correctness bug that would be very hard to see.

        Args:
            root: Absolute repository path.

        Returns:
            The directory the index for this repository lives in.
        """
        digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:12]
        return os.path.join(self.config.chroma_persist_directory, digest)

    @staticmethod
    def _scope(pipeline: _Pipeline, repository_path: str) -> str:
        """
        Express the requested path relative to the bound workspace root.

        Args:
            pipeline: The bound collaborators.
            repository_path: Repository as the caller named it.

        Returns:
            A path relative to the workspace root, or "." for the root
            itself.
        """
        target = os.path.abspath(repository_path)
        if target == pipeline.root:
            return "."

        try:
            relative = os.path.relpath(target, pipeline.root)
        except ValueError:
            return "."

        return "." if relative.startswith("..") else relative.replace("\\", "/")
