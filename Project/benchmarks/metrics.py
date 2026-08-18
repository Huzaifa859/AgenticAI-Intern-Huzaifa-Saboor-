"""
metrics.py
==========

Pure metric extraction helpers for benchmark reports.

These functions read existing agent outputs and Tracer events. They do
not call models or mutate pipelines.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

_EXECUTION_RE = re.compile(
    r"Execution:\s*"
    r"(?P<passed>\d+)\s+passed,\s*"
    r"(?P<failed>\d+)\s+failed,\s*"
    r"(?P<skipped>\d+)\s+skipped"
    r"(?:,\s*(?P<errors>\d+)\s+error(?:s)?)?",
    re.IGNORECASE,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_durations_ms(
    events: Sequence[Any], names: Iterable[str]
) -> List[float]:
    wanted = {str(name) for name in names}
    durations: List[float] = []
    for event in events:
        name = getattr(event, "name", None) or (
            event.get("name") if isinstance(event, dict) else None
        )
        if name not in wanted:
            continue
        duration = getattr(event, "duration_ms", None)
        if duration is None and isinstance(event, dict):
            duration = event.get("duration_ms")
        if duration is not None:
            durations.append(_as_float(duration))
    return durations


def _sum_ms(events: Sequence[Any], names: Iterable[str]) -> float:
    return round(sum(_event_durations_ms(events, names)), 3)


def _meta(event: Any, key: str, default: Any = None) -> Any:
    metadata = getattr(event, "metadata", None)
    if metadata is None and isinstance(event, dict):
        metadata = event.get("metadata") or {}
    if not isinstance(metadata, dict):
        return default
    return metadata.get(key, default)


def extract_analysis_metrics(
    report: Any,
    events: Sequence[Any] = (),
    *,
    wall_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Extract Code Analysis metrics from a CodeAnalysisReport + traces.
    """
    findings = list(getattr(report, "findings", None) or [])
    static_findings = [
        item for item in findings if getattr(item, "detection_method", "") == "static"
    ]
    llm_findings = [
        item for item in findings if getattr(item, "detection_method", "") == "llm"
    ]
    rejected = list(getattr(report, "rejected", None) or [])
    duration = _as_float(
        wall_seconds
        if wall_seconds is not None
        else getattr(report, "duration_seconds", 0.0)
    )
    index_update = getattr(report, "index_update", None)
    indexing_time = _as_float(
        getattr(index_update, "duration_seconds", 0.0) if index_update else 0.0
    )
    if indexing_time <= 0:
        indexing_time = _sum_ms(events, ("indexing_finished", "indexing")) / 1000.0

    retrieval_ms = _sum_ms(events, ("retrieval_finished", "retrieval"))
    model_ms = _sum_ms(events, ("model_response", "model_request"))

    abstention = getattr(report, "abstention", None)
    return {
        "analysis_latency_seconds": round(duration, 4),
        "repository_indexing_time_seconds": round(indexing_time, 4),
        "retrieval_latency_ms": retrieval_ms,
        "model_latency_ms": model_ms,
        "total_runtime_seconds": round(duration, 4),
        "static_findings": len(static_findings),
        "llm_findings": len(llm_findings),
        "grounded_findings": len(findings),
        "hallucinations_rejected": len(rejected),
        "model_used": bool(getattr(report, "model_used", False)),
        "abstained": abstention is not None,
        "abstention_reason": getattr(abstention, "reason", None),
    }


def extract_documentation_metrics(
    result: Any,
    events: Sequence[Any] = (),
    *,
    wall_seconds: Optional[float] = None,
    success: bool = False,
) -> Dict[str, Any]:
    """Extract Documentation metrics from a DocumentationResult + traces."""
    summary = str(getattr(result, "summary", "") or "")
    abstention = getattr(result, "abstention", None)
    function_name = str(getattr(result, "function_name", "") or "")
    documented = 0
    if summary.strip() and abstention is None:
        documented = 1
    generation_ms = _sum_ms(
        events,
        ("documentation_finished", "model_response", "model_request"),
    )
    wall = (
        _as_float(wall_seconds)
        if wall_seconds is not None
        else generation_ms / 1000.0
    )
    return {
        "documentation_generation_time_seconds": round(wall, 4),
        "document_length_chars": len(summary),
        "functions_modules_documented": documented,
        "abstention_rate": 1.0 if abstention is not None else 0.0,
        "repository_summary_produced": bool(
            success and summary.strip() and abstention is None
        ),
        "target_name": function_name,
        "abstention_reason": getattr(abstention, "reason", None),
        "model_latency_ms": _sum_ms(events, ("model_response",)),
    }


def parse_execution_counts(summary: str) -> Dict[str, int]:
    """Parse TestingAgent execution summary counters."""
    text = str(summary or "")
    match = _EXECUTION_RE.search(text)
    if not match:
        skipped = 1 if "Execution: skipped" in text else 0
        return {
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": skipped,
            "execution_errors": 1 if "error" in text.lower() and skipped else 0,
        }
    return {
        "passed_tests": int(match.group("passed")),
        "failed_tests": int(match.group("failed")),
        "skipped_tests": int(match.group("skipped")),
        "execution_errors": int(match.group("errors") or 0),
    }


def extract_testing_metrics(
    result: Any,
    events: Sequence[Any] = (),
    *,
    wall_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract Testing metrics from a TestingResult + traces."""
    generated = dict(getattr(result, "generated_tests", None) or {})
    summary = str(getattr(result, "summary", "") or "")
    abstention = getattr(result, "abstention", None)
    counts = parse_execution_counts(summary)
    generation_ms = _sum_ms(
        events, ("testing_finished", "model_response", "model_request")
    )
    execution_ms = _sum_ms(
        events, ("pytest_execution_finished", "pytest_execution_started")
    )
    wall = (
        _as_float(wall_seconds)
        if wall_seconds is not None
        else generation_ms / 1000.0
    )
    return {
        "generation_time_seconds": round(wall, 4),
        "execution_time_ms": execution_ms,
        "generated_test_files": len(generated),
        "passed_tests": counts["passed_tests"],
        "failed_tests": counts["failed_tests"],
        "skipped_tests": counts["skipped_tests"],
        "execution_errors": counts["execution_errors"],
        "abstained": abstention is not None,
        "abstention_reason": getattr(abstention, "reason", None),
        "coverage_estimate": _as_float(getattr(result, "coverage_estimate", 0.0)),
    }


def extract_rag_metrics(
    report: Any = None,
    events: Sequence[Any] = (),
) -> Dict[str, Any]:
    """Extract RAG indexing/retrieval metrics from analysis report + traces."""
    index_update = getattr(report, "index_update", None) if report is not None else None
    ingestion = getattr(index_update, "ingestion", None) if index_update else None
    chunks_generated = int(getattr(ingestion, "chunks_indexed", 0) or 0)
    indexing_time = _as_float(
        getattr(index_update, "duration_seconds", 0.0) if index_update else 0.0
    )
    if indexing_time <= 0:
        indexing_time = _sum_ms(events, ("indexing_finished", "indexing")) / 1000.0

    retrieved = 0
    for event in events:
        name = getattr(event, "name", None) or (
            event.get("name") if isinstance(event, dict) else None
        )
        if name in {"retrieval_finished", "retrieval"}:
            chunks = _meta(event, "chunks", None)
            if chunks is None:
                chunks = _meta(event, "chunk_count", 0)
            retrieved = max(retrieved, int(chunks or 0))

    context = list(getattr(report, "context", None) or []) if report is not None else []
    if context:
        retrieved = max(retrieved, len(context))

    return {
        "indexing_time_seconds": round(indexing_time, 4),
        "chunks_generated": chunks_generated,
        "retrieved_chunks": retrieved,
        "retrieval_latency_ms": _sum_ms(events, ("retrieval_finished", "retrieval")),
    }


def extract_overall_metrics(
    *,
    per_agent_runtime_seconds: Dict[str, float],
    events: Sequence[Any] = (),
    memory_usage_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregate overall pipeline metrics."""
    total = round(sum(_as_float(value) for value in per_agent_runtime_seconds.values()), 4)
    return {
        "total_pipeline_runtime_seconds": total,
        "per_agent_runtime_seconds": {
            key: round(_as_float(value), 4)
            for key, value in per_agent_runtime_seconds.items()
        },
        "memory_usage_mb": (
            round(memory_usage_mb, 3) if memory_usage_mb is not None else None
        ),
        "tracing_event_count": len(list(events or [])),
    }


def measure_memory_mb() -> Optional[float]:
    """Best-effort current process RSS in megabytes."""
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: kilobytes; macOS: bytes.
        if usage > 10**9:
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0
    except Exception:
        return None
