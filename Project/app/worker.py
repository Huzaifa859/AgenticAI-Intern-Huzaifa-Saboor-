"""
app/worker.py
=============

Run one agent job in a separate process and write a JSON result file.

Used by the Streamlit UI so embedding/LLM memory pressure or a worker
crash cannot take down the Streamlit server.

Optional ``--progress`` writes NDJSON stage lines for live UI updates.

Example:

    python app/worker.py --job analysis --repo examples/demo_repo \\
        --question "Find bugs" --out /tmp/result.json \\
        --progress /tmp/progress.ndjson
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# Keep runtime data outside the project tree so nothing touches watched sources.
_RUNTIME_ROOT = os.path.join(
    tempfile.gettempdir(), "codebase_assistant_streamlit"
)
os.makedirs(_RUNTIME_ROOT, exist_ok=True)
os.environ.setdefault(
    "CHROMA_PERSIST_DIR", os.path.join(_RUNTIME_ROOT, "chroma")
)
os.environ.setdefault(
    "MEMORY_STORE_PATH", os.path.join(_RUNTIME_ROOT, "memory_store")
)

#: Tracer / lifecycle event names → short human progress messages.
_STAGE_MESSAGES: Dict[str, str] = {
    "job_started": "Worker started",
    "job_finished": "Worker finished",
    "job_failed": "Worker failed",
    "indexing": "Indexing repository...",
    "before_ingest": "Preparing repository index...",
    "after_ingest": "Repository index ready",
    "before_model_call": "Calling language model...",
    "after_model_call": "Model response received",
    "model_request": "Calling language model...",
    "model_response": "Model response received",
    "before_agent_run": "Starting agent...",
    "after_agent_run": "Agent finished",
    "documentation_started": "Starting documentation...",
    "documentation_ast_scan_finished": "Scanning symbols for documentation...",
    "documentation_symbol_started": "Documenting symbol...",
    "documentation_symbol_finished": "Finished documenting symbol",
    "documentation_merge_started": "Merging documentation results...",
    "documentation_merge_finished": "Documentation merge complete",
    "documentation_finished": "Documentation finished",
    "documentation_retry_started": "Repairing documentation JSON...",
    "documentation_grounding_started": "Grounding documentation claims...",
    "documentation_grounding_finished": "Documentation grounding finished",
    "testing_started": "Starting test generation...",
    "testing_symbol_generation_finished": "Generated tests for a symbol",
    "testing_merge_completed": "Merging generated tests...",
    "testing_repair_started": "Repairing failing tests...",
    "testing_finished": "Testing finished",
    "analysis_started": "Starting code analysis...",
    "static_analysis": "Running static analysis...",
    "retrieval": "Retrieving code context...",
}


class ProgressWriter:
    """Append NDJSON progress lines for the Streamlit UI to tail."""

    def __init__(self, path: Optional[str]) -> None:
        self.path = (path or "").strip() or None
        self._last_message = ""

    def emit(self, stage: str, message: str = "", **extra: Any) -> None:
        """Write one progress event; never raises into the worker."""
        if not self.path:
            return
        stage_key = str(stage or "progress")
        text = (message or _STAGE_MESSAGES.get(stage_key) or "").strip()
        if not text:
            text = stage_key.replace("_", " ").strip().capitalize() or "Working..."
        # Avoid flooding the UI with identical consecutive lines.
        if text == self._last_message and stage_key not in {
            "job_started",
            "job_finished",
            "job_failed",
        }:
            return
        self._last_message = text
        payload = {
            "ts": time.time(),
            "stage": stage_key,
            "message": text,
        }
        if extra:
            payload["extra"] = {
                key: value
                for key, value in extra.items()
                if value is not None and not callable(value)
            }
        try:
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
        except OSError:
            pass


def _attach_tracer_progress(supervisor: Any, progress: ProgressWriter) -> None:
    """Forward selected Tracer.record calls into the progress file."""
    tracer = getattr(supervisor, "tracer", None)
    if tracer is None or not progress.path:
        return
    original_record = tracer.record

    def record(event_type: Any, name: str, **metadata: Any) -> None:
        original_record(event_type, name, **metadata)
        stage = str(name or "")
        if not stage:
            return
        message = _STAGE_MESSAGES.get(stage)
        if message is None:
            # Keep high-signal agent stages; skip noisy internals.
            interesting = (
                stage.startswith("documentation_")
                or stage.startswith("testing_")
                or stage.startswith("analysis_")
                or stage in _STAGE_MESSAGES
            )
            if not interesting:
                return
            message = stage.replace("_", " ").strip().capitalize()
        symbol = metadata.get("symbol")
        if symbol:
            message = f"{message} ({symbol})"
        progress.emit(stage, message)

    tracer.record = record  # type: ignore[method-assign]


def _analysis_to_dict(report: Any) -> Dict[str, Any]:
    """Slim JSON-ready view of a CodeAnalysisReport (no retrieval context)."""
    findings = []
    for finding in list(getattr(report, "findings", None) or []):
        if hasattr(finding, "model_dump"):
            findings.append(finding.model_dump())
        else:
            findings.append(
                {
                    "bug_type": getattr(finding, "bug_type", ""),
                    "description": getattr(finding, "description", ""),
                    "severity": getattr(finding, "severity", ""),
                    "confidence": float(getattr(finding, "confidence", 0.0)),
                    "file_path": getattr(finding, "file_path", ""),
                    "function_name": getattr(finding, "function_name", ""),
                    "line_start": int(getattr(finding, "line_start", 0) or 0),
                    "line_end": int(getattr(finding, "line_end", 0) or 0),
                    "evidence": getattr(finding, "evidence", ""),
                    "suggested_fix": getattr(finding, "suggested_fix", None),
                    "detection_method": getattr(finding, "detection_method", ""),
                    "metadata": dict(getattr(finding, "metadata", None) or {}),
                }
            )

    abstention = getattr(report, "abstention", None)
    abstention_data = None
    if abstention is not None:
        abstention_data = (
            abstention.model_dump()
            if hasattr(abstention, "model_dump")
            else {
                "reason": getattr(abstention, "reason", ""),
                "confidence": float(getattr(abstention, "confidence", 0.0)),
                "evidence_available": list(
                    getattr(abstention, "evidence_available", None) or []
                ),
                "recommended_next_steps": list(
                    getattr(abstention, "recommended_next_steps", None) or []
                ),
            }
        )

    return {
        "repository_path": getattr(report, "repository_path", ""),
        "question": getattr(report, "question", ""),
        "findings": findings,
        "answer": getattr(report, "answer", "") or "",
        "notes": list(getattr(report, "notes", None) or []),
        "duration_seconds": float(getattr(report, "duration_seconds", 0.0) or 0.0),
        "model_used": bool(getattr(report, "model_used", False)),
        "duplicates_removed": int(getattr(report, "duplicates_removed", 0) or 0),
        "abstention": abstention_data,
    }


def _write(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Codebase Assistant UI worker")
    parser.add_argument(
        "--job",
        required=True,
        choices=("analysis", "documentation", "testing"),
    )
    parser.add_argument("--repo", required=True, help="Prepared repository path")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--progress",
        default="",
        help="Optional NDJSON progress file for live UI updates",
    )
    parser.add_argument("--question", default="Find bugs and potential issues")
    parser.add_argument("--mode", default="")
    parser.add_argument("--file", default="")
    parser.add_argument("--function", default="")
    parser.add_argument("--class-name", default="")
    parser.add_argument("--write-to-disk", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)

    progress = ProgressWriter(args.progress)
    progress.emit("job_started", f"Starting {args.job} job...")

    try:
        from service import (
            build_supervisor,
            run_analysis,
            run_documentation,
            run_testing,
        )

        supervisor = build_supervisor()
        _attach_tracer_progress(supervisor, progress)

        if args.job == "analysis":
            progress.emit("analysis_started", "Starting code analysis...")
            report = run_analysis(
                supervisor, args.repo, question=args.question
            )
            payload = {"ok": True, "job": args.job, "result": _analysis_to_dict(report)}
        elif args.job == "documentation":
            progress.emit("documentation_started", "Starting documentation...")
            result = run_documentation(
                supervisor,
                args.repo,
                mode=args.mode,
                file_path=args.file,
                function_name=args.function,
                class_name=args.class_name,
                write_to_disk=bool(args.write_to_disk),
                replace_existing=bool(args.replace_existing),
            )
            payload = {
                "ok": True,
                "job": args.job,
                "result": result.model_dump(),
            }
        else:
            progress.emit("testing_started", "Starting test generation...")
            result = run_testing(
                supervisor,
                args.repo,
                mode=args.mode,
                file_path=args.file,
                function_name=args.function,
            )
            payload = {
                "ok": True,
                "job": args.job,
                "result": result.model_dump(),
            }
        _write(args.out, payload)
        progress.emit("job_finished", f"{args.job.capitalize()} job complete")
        return 0
    except Exception as exc:  # noqa: BLE001 — always emit a JSON error payload
        progress.emit("job_failed", f"Worker failed: {exc}")
        _write(
            args.out,
            {
                "ok": False,
                "job": args.job,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
