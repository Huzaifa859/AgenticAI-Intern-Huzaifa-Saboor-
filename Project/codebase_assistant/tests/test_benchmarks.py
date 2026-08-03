"""
test_benchmarks.py
===================

Tests for the evaluation benchmark suite.

Uses mocked Supervisor calls for runner timing/export coverage so the
suite stays offline and does not require embedding downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.export import flatten_repository_rows, write_csv_report, write_json_report
from benchmarks.metrics import (
    extract_analysis_metrics,
    extract_documentation_metrics,
    extract_overall_metrics,
    extract_rag_metrics,
    extract_testing_metrics,
    parse_execution_counts,
)
from benchmarks.runner import BenchmarkRunner, PreparedRepository, format_console_summary
from codebase_assistant.schemas.schemas import (
    AbstentionResult,
    AgentResponse,
    AgentType,
    BugReport,
    DocumentationResult,
    TestingResult as TestingResultModel,
)


def _bug(method: str = "static", confidence: float = 0.9) -> BugReport:
    return BugReport(
        bug_type="unused_import",
        description="unused",
        severity="low",
        confidence=confidence,
        file_path="a.py",
        function_name="<module>",
        line_start=1,
        line_end=1,
        evidence="import os",
        detection_method=method,  # type: ignore[arg-type]
    )


def test_parse_execution_counts() -> None:
    counts = parse_execution_counts(
        "Generated tests.\nExecution: 2 passed, 1 failed, 0 skipped, 0 errors"
    )
    assert counts["passed_tests"] == 2
    assert counts["failed_tests"] == 1
    assert counts["skipped_tests"] == 0


def test_extract_analysis_metrics_counts_and_latencies() -> None:
    report = SimpleNamespace(
        findings=[_bug("static"), _bug("llm")],
        rejected=[object()],
        duration_seconds=1.25,
        model_used=True,
        abstention=None,
        index_update=SimpleNamespace(duration_seconds=0.4, ingestion=SimpleNamespace(chunks_indexed=3)),
        context=[object(), object()],
    )
    events = [
        SimpleNamespace(name="retrieval_finished", duration_ms=12.0, metadata={"chunks": 2}),
        SimpleNamespace(name="model_response", duration_ms=30.0, metadata={}),
    ]
    metrics = extract_analysis_metrics(report, events)
    assert metrics["static_findings"] == 1
    assert metrics["llm_findings"] == 1
    assert metrics["grounded_findings"] == 2
    assert metrics["hallucinations_rejected"] == 1
    assert metrics["repository_indexing_time_seconds"] == 0.4
    assert metrics["retrieval_latency_ms"] == 12.0
    assert metrics["model_latency_ms"] == 30.0


def test_extract_documentation_and_testing_metrics() -> None:
    docs = DocumentationResult(
        file_path="README.md",
        function_name="README",
        summary="# Demo\n\nSummary",
        parameters=[],
        returns="",
        example_usage="",
        abstention=None,
    )
    doc_metrics = extract_documentation_metrics(docs, wall_seconds=0.5, success=True)
    assert doc_metrics["document_length_chars"] == len(docs.summary)
    assert doc_metrics["repository_summary_produced"] is True
    assert doc_metrics["abstention_rate"] == 0.0

    tests = TestingResultModel(
        summary="Execution: 1 passed, 0 failed, 0 skipped",
        generated_tests={"test_a.py": "def test_a():\n    assert True\n"},
        coverage_estimate=0.2,
    )
    test_events = [
        SimpleNamespace(name="pytest_execution_finished", duration_ms=8.0, metadata={})
    ]
    test_metrics = extract_testing_metrics(tests, test_events, wall_seconds=0.7)
    assert test_metrics["generated_test_files"] == 1
    assert test_metrics["passed_tests"] == 1
    assert test_metrics["execution_time_ms"] == 8.0


def test_extract_rag_and_overall_metrics() -> None:
    report = SimpleNamespace(
        index_update=SimpleNamespace(
            duration_seconds=0.2,
            ingestion=SimpleNamespace(chunks_indexed=5),
        ),
        context=[1, 2, 3],
    )
    events = [
        SimpleNamespace(name="retrieval_finished", duration_ms=5.5, metadata={"chunks": 3})
    ]
    rag = extract_rag_metrics(report, events)
    assert rag["chunks_generated"] == 5
    assert rag["retrieved_chunks"] == 3
    overall = extract_overall_metrics(
        per_agent_runtime_seconds={"analysis": 1.0, "documentation": 0.5, "testing": 0.25},
        events=events,
        memory_usage_mb=100.0,
    )
    assert overall["total_pipeline_runtime_seconds"] == 1.75
    assert overall["tracing_event_count"] == 1


def test_json_and_csv_export(tmp_path: Path) -> None:
    report = {
        "benchmark_id": "abc123",
        "mode": "offline",
        "repositories": [
            {
                "name": "demo",
                "path": "/tmp/demo",
                "analysis": {
                    "analysis_latency_seconds": 1.0,
                    "static_findings": 2,
                    "llm_findings": 0,
                    "grounded_findings": 2,
                    "hallucinations_rejected": 0,
                },
                "documentation": {
                    "documentation_generation_time_seconds": 0.2,
                    "document_length_chars": 10,
                    "abstention_rate": 0.0,
                    "repository_summary_produced": True,
                },
                "testing": {
                    "generation_time_seconds": 0.3,
                    "generated_test_files": 1,
                    "passed_tests": 1,
                    "failed_tests": 0,
                },
                "rag": {
                    "indexing_time_seconds": 0.1,
                    "chunks_generated": 4,
                    "retrieved_chunks": 2,
                },
                "overall": {
                    "total_pipeline_runtime_seconds": 1.5,
                    "tracing_event_count": 9,
                    "memory_usage_mb": 50.0,
                },
            }
        ],
        "summary": {"repository_count": 1},
    }
    json_path = write_json_report(report, tmp_path / "latest.json")
    csv_path = write_csv_report(report, tmp_path / "latest.csv")
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["benchmark_id"] == "abc123"
    assert "grounded_findings" in flatten_repository_rows(loaded)[0]
    assert csv_path.exists()
    assert "demo" in csv_path.read_text(encoding="utf-8")


def test_benchmark_runner_timing_and_export(tmp_path: Path) -> None:
    """Runner should time agents, collect metrics, and export JSON."""
    repo = tmp_path / "sample"
    repo.mkdir()
    (repo / "mod.py").write_text("def ok():\n    return 1\n", encoding="utf-8")

    analysis_report = SimpleNamespace(
        findings=[_bug()],
        rejected=[],
        duration_seconds=0.1,
        model_used=False,
        abstention=None,
        index_update=SimpleNamespace(
            duration_seconds=0.05,
            ingestion=SimpleNamespace(chunks_indexed=1),
        ),
        context=[],
    )
    doc_result = DocumentationResult(
        file_path="README.md",
        function_name="README",
        summary="# Sample",
        parameters=[],
        returns="",
        example_usage="",
    )
    test_result = TestingResultModel(
        summary="Execution: 1 passed, 0 failed, 0 skipped",
        generated_tests={"test_mod.py": "def test_ok():\n    assert True\n"},
        coverage_estimate=0.1,
    )

    tracer = MagicMock()
    tracer.get_events.return_value = [
        SimpleNamespace(name="retrieval_finished", duration_ms=1.0, metadata={"chunks": 0}),
        SimpleNamespace(name="model_response", duration_ms=2.0, metadata={}),
        SimpleNamespace(name="pytest_execution_finished", duration_ms=3.0, metadata={}),
    ]

    supervisor = MagicMock()
    supervisor.tracer = tracer
    supervisor.agents = {}
    supervisor.handle_task.side_effect = [
        AgentResponse(
            task_id="a",
            agent_type=AgentType.CODE_ANALYSIS,
            success=True,
            output=analysis_report,
        ),
        AgentResponse(
            task_id="d",
            agent_type=AgentType.DOCUMENTATION,
            success=True,
            output=doc_result,
        ),
        AgentResponse(
            task_id="t",
            agent_type=AgentType.TESTING,
            success=True,
            output=test_result,
        ),
    ]

    runner = BenchmarkRunner(
        mode="offline",
        results_dir=tmp_path / "results",
        work_dir=tmp_path / "work",
    )

    with patch.object(runner, "_build_supervisor", return_value=supervisor), patch.object(
        runner, "_inject_offline_llm"
    ):
        # Fake monotonic clock: each perf_counter pair advances by 0.1s
        clock = {"t": 0.0}

        def _perf() -> float:
            clock["t"] += 0.05
            return clock["t"]

        with patch("benchmarks.runner.time.perf_counter", side_effect=_perf):
            report = runner.run_repositories(
                [PreparedRepository(name="sample", path=str(repo))],
                dataset="test",
            )

    assert report["mode"] == "offline"
    assert len(report["repositories"]) == 1
    item = report["repositories"][0]
    assert item["analysis"]["grounded_findings"] == 1
    assert item["documentation"]["repository_summary_produced"] is True
    assert item["testing"]["passed_tests"] == 1
    assert item["overall"]["total_pipeline_runtime_seconds"] > 0
    assert item["overall"]["per_agent_runtime_seconds"]["analysis"] > 0

    outputs = runner.export(report, write_csv=True)
    assert Path(outputs["json"]).exists()
    assert Path(outputs["csv"]).exists()
    summary = format_console_summary(report)
    assert "Benchmark Summary" in summary
    assert "sample" in summary


def test_unsupported_repository_abstention_metrics() -> None:
    """Documentation abstention should surface as abstention_rate=1.0."""
    result = DocumentationResult(
        file_path="README.md",
        function_name="README",
        summary="",
        parameters=[],
        returns="",
        example_usage="",
        abstention=AbstentionResult(
            reason="Repository contains no supported Python files.",
            confidence=1.0,
            evidence_available=[],
            recommended_next_steps=["Add Python files"],
        ),
    )
    metrics = extract_documentation_metrics(result, success=False)
    assert metrics["abstention_rate"] == 1.0
    assert metrics["repository_summary_produced"] is False
