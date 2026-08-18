"""
runner.py
=========

BenchmarkRunner: prepare repositories and collect metrics from the
existing Supervisor / agent pipelines.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

from codebase_assistant.config import Config
from codebase_assistant.supervisor import Supervisor

from .datasets import BenchmarkCase, ensure_fixture_repositories
from .export import write_csv_report, write_json_report
from .metrics import (
    extract_analysis_metrics,
    extract_documentation_metrics,
    extract_overall_metrics,
    extract_rag_metrics,
    extract_testing_metrics,
    measure_memory_mb,
)
from .mock_llm import BenchmarkLLMClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class PreparedRepository:
    """A local repository path ready for benchmarking."""

    name: str
    path: str
    temporary: bool = False
    cleanup_root: Optional[str] = None


class BenchmarkRunner:
    """
    Runs analysis / documentation / testing against one or more repos.
    """

    def __init__(
        self,
        *,
        mode: str = "offline",
        results_dir: Optional[Path] = None,
        work_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            mode: ``offline`` injects a deterministic mock LLM; ``live``
                uses the Supervisor's configured providers.
            results_dir: Where JSON/CSV reports are written.
            work_dir: Scratch directory for chroma/memory isolation.
        """
        self.mode = (mode or "offline").strip().lower()
        if self.mode not in {"offline", "live"}:
            raise ValueError("mode must be 'offline' or 'live'")
        self.results_dir = Path(results_dir or DEFAULT_RESULTS_DIR)
        self.work_dir = Path(
            work_dir
            or (self.results_dir / ".work" / uuid.uuid4().hex[:10])
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._temp_roots: List[str] = []

    def run_dataset(self) -> Dict[str, Any]:
        """Run the built-in fixture dataset."""
        cases = ensure_fixture_repositories()
        prepared = [
            PreparedRepository(name=case.name, path=str(case.path.resolve()))
            for case in cases
        ]
        return self.run_repositories(prepared, dataset="builtin")

    def run_reference(self, reference: str) -> Dict[str, Any]:
        """
        Run against a local path or GitHub HTTPS URL.

        Args:
            reference: Local directory or GitHub URL.
        """
        prepared = self._prepare_reference(reference)
        try:
            return self.run_repositories([prepared], dataset="custom")
        finally:
            self.cleanup()

    def run_repositories(
        self,
        repositories: Sequence[PreparedRepository],
        *,
        dataset: str = "custom",
    ) -> Dict[str, Any]:
        """Benchmark each prepared repository and assemble a report."""
        started = time.time()
        benchmark_id = uuid.uuid4().hex
        repo_reports: List[Dict[str, Any]] = []

        for prepared in repositories:
            logger.info("Benchmarking %s (%s)", prepared.name, prepared.path)
            repo_reports.append(self._benchmark_one(prepared))

        report = {
            "benchmark_id": benchmark_id,
            "dataset": dataset,
            "mode": self.mode,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "project_root": str(PROJECT_ROOT),
            "repositories": repo_reports,
            "summary": self._summarize(repo_reports),
            "duration_seconds": round(time.time() - started, 4),
        }
        return report

    def export(
        self,
        report: Dict[str, Any],
        *,
        write_csv: bool = True,
        basename: str = "latest",
    ) -> Dict[str, str]:
        """Write JSON (and optional CSV) under the results directory."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        json_path = write_json_report(report, self.results_dir / f"{basename}.json")
        outputs = {"json": str(json_path)}
        stamped = self.results_dir / f"{report['benchmark_id']}.json"
        write_json_report(report, stamped)
        outputs["json_stamped"] = str(stamped)
        if write_csv:
            csv_path = write_csv_report(report, self.results_dir / f"{basename}.csv")
            outputs["csv"] = str(csv_path)
        return outputs

    def cleanup(self) -> None:
        """Remove temporary clones created for remote URLs."""
        while self._temp_roots:
            root = self._temp_roots.pop()
            shutil.rmtree(root, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_reference(self, reference: str) -> PreparedRepository:
        text = str(reference or "").strip()
        if not text:
            raise ValueError("repository reference is required")

        # Local path first.
        candidate = Path(text).expanduser()
        if candidate.exists():
            return PreparedRepository(
                name=candidate.name or "local_repo",
                path=str(candidate.resolve()),
            )

        # GitHub URL via Supervisor tools (no production code changes).
        supervisor = self._build_supervisor(label="clone")
        github = supervisor.github_tools
        if not github.is_remote_reference(text):
            raise FileNotFoundError(f"Repository not found: {text}")
        github.validate_repository(text)
        temp_root = tempfile.mkdtemp(prefix="benchmark_clone_")
        self._temp_roots.append(temp_root)
        destination = os.path.join(temp_root, "repo")
        github.clone_repository(text, destination)
        return PreparedRepository(
            name=Path(destination).name,
            path=destination,
            temporary=True,
            cleanup_root=temp_root,
        )

    def _benchmark_one(self, prepared: PreparedRepository) -> Dict[str, Any]:
        supervisor = self._build_supervisor(label=prepared.name)
        if self.mode == "offline":
            self._inject_offline_llm(supervisor, prepared.name)

        memory_before = measure_memory_mb()
        events_before = len(supervisor.tracer.get_events())
        per_agent: Dict[str, float] = {}

        # Analysis
        analysis_started = time.perf_counter()
        analysis_response = supervisor.handle_task(
            "analysis: Find likely bugs and correctness problems in this code.",
            prepared.path,
        )
        analysis_seconds = time.perf_counter() - analysis_started
        per_agent["analysis"] = analysis_seconds
        analysis_report = analysis_response.output
        if analysis_report is None:
            analysis_report = type("EmptyReport", (), {})()
        analysis_events = supervisor.tracer.get_events()[events_before:]
        analysis_metrics = extract_analysis_metrics(
            analysis_report,
            analysis_events,
            wall_seconds=analysis_seconds,
        )
        rag_metrics = extract_rag_metrics(analysis_report, analysis_events)

        # Documentation
        doc_mark = len(supervisor.tracer.get_events())
        doc_started = time.perf_counter()
        documentation_response = supervisor.handle_task(
            "documentation README",
            prepared.path,
        )
        doc_seconds = time.perf_counter() - doc_started
        per_agent["documentation"] = doc_seconds
        doc_events = supervisor.tracer.get_events()[doc_mark:]
        documentation_metrics = extract_documentation_metrics(
            documentation_response.output
            or type("EmptyDocs", (), {"summary": "", "abstention": None})(),
            doc_events,
            wall_seconds=doc_seconds,
            success=bool(documentation_response.success),
        )

        # Testing
        test_mark = len(supervisor.tracer.get_events())
        test_started = time.perf_counter()
        testing_response = supervisor.handle_task(
            "testing",
            prepared.path,
        )
        test_seconds = time.perf_counter() - test_started
        per_agent["testing"] = test_seconds
        test_events = supervisor.tracer.get_events()[test_mark:]
        testing_metrics = extract_testing_metrics(
            testing_response.output
            or type(
                "EmptyTests",
                (),
                {"summary": "", "generated_tests": {}, "abstention": None},
            )(),
            test_events,
            wall_seconds=test_seconds,
        )

        all_events = supervisor.tracer.get_events()[events_before:]
        memory_after = measure_memory_mb()
        memory_usage = None
        if memory_before is not None and memory_after is not None:
            memory_usage = max(memory_before, memory_after)
        elif memory_after is not None:
            memory_usage = memory_after

        overall = extract_overall_metrics(
            per_agent_runtime_seconds=per_agent,
            events=all_events,
            memory_usage_mb=memory_usage,
        )

        return {
            "name": prepared.name,
            "path": prepared.path,
            "analysis": analysis_metrics,
            "documentation": documentation_metrics,
            "testing": testing_metrics,
            "rag": rag_metrics,
            "overall": overall,
            "agent_success": {
                "analysis": bool(analysis_response.success),
                "documentation": bool(documentation_response.success),
                "testing": bool(testing_response.success),
            },
        }

    def _build_supervisor(self, label: str) -> Supervisor:
        """Construct an isolated Supervisor for one benchmark case."""
        case_root = self.work_dir / label
        case_root.mkdir(parents=True, exist_ok=True)
        chroma_dir = str(case_root / "chroma")
        memory_dir = str(case_root / "memory")
        # Config.load() reads these env vars; set them so every collaborator
        # (Indexer/Retriever/agents) stays inside the benchmark work dir.
        os.environ["CHROMA_PERSIST_DIR"] = chroma_dir
        os.environ["MEMORY_STORE_PATH"] = memory_dir
        os.environ["WORKSPACE_ROOT"] = str(case_root)
        os.environ["RERANK_ENABLED"] = "false"
        config = Config.load()
        config.workspace_root = str(case_root)
        config.chroma_persist_directory = chroma_dir
        config.memory_store_path = memory_dir
        config.rerank_enabled = False

        if self.mode == "offline":
            with patch.object(Supervisor, "_init_openrouter_provider", return_value=None), \
                patch.object(Supervisor, "_init_ollama_provider", return_value=None):
                return Supervisor(config=config)
        return Supervisor(config=config)

    @staticmethod
    def _inject_offline_llm(supervisor: Supervisor, label: str) -> None:
        """Replace agent model clients with the deterministic mock."""
        client = BenchmarkLLMClient(repo_hint=label)
        for agent in supervisor.agents.values():
            agent.model_client = client

    @staticmethod
    def _summarize(repo_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not repo_reports:
            return {
                "repository_count": 0,
                "mean_total_runtime_seconds": 0.0,
                "total_grounded_findings": 0,
                "mean_abstention_rate_documentation": 0.0,
            }
        totals = [
            float(
                (item.get("overall") or {}).get("total_pipeline_runtime_seconds") or 0.0
            )
            for item in repo_reports
        ]
        grounded = sum(
            int((item.get("analysis") or {}).get("grounded_findings") or 0)
            for item in repo_reports
        )
        abstention = [
            float((item.get("documentation") or {}).get("abstention_rate") or 0.0)
            for item in repo_reports
        ]
        return {
            "repository_count": len(repo_reports),
            "mean_total_runtime_seconds": round(sum(totals) / len(totals), 4),
            "total_grounded_findings": grounded,
            "mean_abstention_rate_documentation": round(
                sum(abstention) / len(abstention), 4
            ),
        }


def format_console_summary(report: Dict[str, Any]) -> str:
    """Render a compact human-readable summary."""
    lines = [
        "Benchmark Summary",
        "=================",
        f"id:    {report.get('benchmark_id')}",
        f"mode:  {report.get('mode')}",
        f"dataset: {report.get('dataset')}",
        f"repos: {len(report.get('repositories') or [])}",
        "",
    ]
    for item in report.get("repositories") or []:
        overall = item.get("overall") or {}
        analysis = item.get("analysis") or {}
        docs = item.get("documentation") or {}
        testing = item.get("testing") or {}
        lines.extend(
            [
                f"[{item.get('name')}]",
                f"  total_runtime_s     {overall.get('total_pipeline_runtime_seconds')}",
                f"  grounded_findings    {analysis.get('grounded_findings')}",
                f"  hallucinations_rej  {analysis.get('hallucinations_rejected')}",
                f"  docs_summary        {docs.get('repository_summary_produced')}",
                f"  docs_abstention     {docs.get('abstention_rate')}",
                f"  tests_generated     {testing.get('generated_test_files')}",
                f"  tests_passed        {testing.get('passed_tests')}",
                f"  trace_events        {overall.get('tracing_event_count')}",
                "",
            ]
        )
    summary = report.get("summary") or {}
    lines.extend(
        [
            "Aggregate",
            f"  mean_total_runtime_s {summary.get('mean_total_runtime_seconds')}",
            f"  total_grounded       {summary.get('total_grounded_findings')}",
        ]
    )
    return "\n".join(lines)
