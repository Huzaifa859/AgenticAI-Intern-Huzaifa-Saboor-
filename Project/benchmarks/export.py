"""
export.py
=========

JSON / CSV writers for benchmark results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def ensure_results_dir(path: Path) -> Path:
    """Create the results directory when missing."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_report(report: Dict[str, Any], destination: Path) -> Path:
    """
    Write a deterministic JSON benchmark report.

    Args:
        report: Serializable benchmark payload.
        destination: Output file path.

    Returns:
        The written path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    destination.write_text(text + "\n", encoding="utf-8")
    return destination


def flatten_repository_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten per-repository metrics into CSV-friendly rows."""
    rows: List[Dict[str, Any]] = []
    for item in report.get("repositories", []) or []:
        base = {
            "benchmark_id": report.get("benchmark_id"),
            "mode": report.get("mode"),
            "repository": item.get("name"),
            "repository_path": item.get("path"),
        }
        overall = item.get("overall") or {}
        analysis = item.get("analysis") or {}
        documentation = item.get("documentation") or {}
        testing = item.get("testing") or {}
        rag = item.get("rag") or {}
        row = {
            **base,
            "total_pipeline_runtime_seconds": overall.get(
                "total_pipeline_runtime_seconds"
            ),
            "tracing_event_count": overall.get("tracing_event_count"),
            "memory_usage_mb": overall.get("memory_usage_mb"),
            "analysis_latency_seconds": analysis.get("analysis_latency_seconds"),
            "static_findings": analysis.get("static_findings"),
            "llm_findings": analysis.get("llm_findings"),
            "grounded_findings": analysis.get("grounded_findings"),
            "hallucinations_rejected": analysis.get("hallucinations_rejected"),
            "documentation_generation_time_seconds": documentation.get(
                "documentation_generation_time_seconds"
            ),
            "document_length_chars": documentation.get("document_length_chars"),
            "documentation_abstention_rate": documentation.get("abstention_rate"),
            "repository_summary_produced": documentation.get(
                "repository_summary_produced"
            ),
            "testing_generation_time_seconds": testing.get("generation_time_seconds"),
            "generated_test_files": testing.get("generated_test_files"),
            "passed_tests": testing.get("passed_tests"),
            "failed_tests": testing.get("failed_tests"),
            "rag_indexing_time_seconds": rag.get("indexing_time_seconds"),
            "chunks_generated": rag.get("chunks_generated"),
            "retrieved_chunks": rag.get("retrieved_chunks"),
        }
        rows.append(row)
    return rows


def write_csv_report(report: Dict[str, Any], destination: Path) -> Path:
    """Write a flattened CSV summary of repository metrics."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = flatten_repository_rows(report)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["repository"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return destination
