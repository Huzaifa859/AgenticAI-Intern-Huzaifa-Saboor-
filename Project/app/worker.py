"""
app/worker.py
=============

Run one agent job and write a JSON result file.

Used by:
- one-shot CLI: ``python app/worker.py --job ...``
- long-lived ``worker_server.py`` (warm Supervisor + embeddings)

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
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# Keep runtime data outside the project tree so nothing touches watched sources.
from ui_paths import chroma_persist_dir, memory_store_path, streamlit_data_dir  # noqa: E402

_RUNTIME_ROOT = streamlit_data_dir()
os.makedirs(_RUNTIME_ROOT, exist_ok=True)
os.environ.setdefault("CHROMA_PERSIST_DIR", chroma_persist_dir())
os.environ.setdefault("MEMORY_STORE_PATH", memory_store_path())
os.makedirs(os.environ["CHROMA_PERSIST_DIR"], exist_ok=True)
os.makedirs(os.environ["MEMORY_STORE_PATH"], exist_ok=True)

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

_STREAM_STAGES = frozenset(
    {
        "documentation_stream_delta",
        "analysis_stream_delta",
    }
)
_STREAM_MESSAGES: Dict[str, str] = {
    "documentation_stream_delta": "Streaming documentation…",
    "analysis_stream_delta": "Streaming analysis…",
}


class ProgressWriter:
    """Append NDJSON progress lines for the Streamlit UI to tail."""

    #: Coalesce token deltas so the progress file is not one line per token.
    _STREAM_BATCH_SECONDS = 0.04
    #: Flush early when the buffer is already large enough to paint.
    _STREAM_BATCH_CHARS = 48

    def __init__(self, path: Optional[str]) -> None:
        self.path = (path or "").strip() or None
        self._last_message = ""
        self._stream_buf = ""
        self._stream_opened_at = 0.0
        self._stream_stage = "documentation_stream_delta"
        self._stream_message = _STREAM_MESSAGES["documentation_stream_delta"]

    def emit(self, stage: str, message: str = "", **extra: Any) -> None:
        """Write one progress event; never raises into the worker."""
        if not self.path:
            return
        stage_key = str(stage or "progress")
        # Flush any pending stream tokens before a normal stage line
        # so the UI never sees stages arrive ahead of earlier text.
        if stage_key not in _STREAM_STAGES:
            self.flush_stream()
        text = (message or _STAGE_MESSAGES.get(stage_key) or "").strip()
        if not text:
            text = stage_key.replace("_", " ").strip().capitalize() or "Working..."
        # Avoid flooding the UI with identical consecutive stage lines.
        # Stream deltas intentionally repeat the same message while the
        # growing text lives in ``extra["text"]``.
        if text == self._last_message and stage_key not in {
            "job_started",
            "job_finished",
            "job_failed",
            *_STREAM_STAGES,
        }:
            return
        self._last_message = text
        self._write_event(stage_key, text, extra)

    def emit_stream_delta(
        self,
        chunk: str,
        *,
        stage: str = "documentation_stream_delta",
    ) -> None:
        """Buffer a token delta; flush on a short timer."""
        text = str(chunk or "")
        if not text or not self.path:
            return
        stage_key = str(stage or "documentation_stream_delta")
        if stage_key not in _STREAM_STAGES:
            stage_key = "documentation_stream_delta"
        if self._stream_buf and stage_key != self._stream_stage:
            self.flush_stream()
        self._stream_stage = stage_key
        self._stream_message = _STREAM_MESSAGES.get(stage_key, "Streaming…")
        now = time.monotonic()
        if not self._stream_buf:
            self._stream_opened_at = now
        self._stream_buf += text
        if (
            len(self._stream_buf) >= self._STREAM_BATCH_CHARS
            or (now - self._stream_opened_at) >= self._STREAM_BATCH_SECONDS
        ):
            self.flush_stream()

    def flush_stream(self) -> None:
        """Write any buffered stream text immediately."""
        if not self.path or not self._stream_buf:
            return
        chunk = self._stream_buf
        self._stream_buf = ""
        self._stream_opened_at = 0.0
        stage = self._stream_stage or "documentation_stream_delta"
        message = self._stream_message or _STREAM_MESSAGES.get(
            stage,
            "Streaming…",
        )
        self._last_message = message
        self._write_event(stage, message, {"text": chunk})

    def _write_event(
        self,
        stage_key: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "stage": stage_key,
            "message": message,
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


def _attach_tracer_progress(supervisor: Any, progress: ProgressWriter) -> Callable[[], None]:
    """Forward selected Tracer.record calls into the progress file.

    Returns a restore callable that puts the original ``record`` back.
    """
    tracer = getattr(supervisor, "tracer", None)
    if tracer is None or not progress.path:
        return lambda: None
    original_record = tracer.record

    def record(event_type: Any, name: str, **metadata: Any) -> None:
        stage = str(name or "")
        # Stream tokens skip Tracer storage — thousands of per-token
        # events would stall the UI and bloat memory. Progress file only.
        if stage in _STREAM_STAGES:
            chunk = str(metadata.get("text") or "")
            if chunk:
                progress.emit_stream_delta(chunk, stage=stage)
            return
        original_record(event_type, name, **metadata)
        if not stage:
            return
        message = _STAGE_MESSAGES.get(stage)
        if message is None:
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

    def restore() -> None:
        progress.flush_stream()
        tracer.record = original_record  # type: ignore[method-assign]

    return restore


def _finding_to_dict(finding: Any) -> Dict[str, Any]:
    """Normalize a BugReport / finding-like object to a plain dict."""
    if hasattr(finding, "model_dump"):
        return dict(finding.model_dump())
    return {
        "bug_type": getattr(finding, "bug_type", ""),
        "description": getattr(finding, "description", ""),
        "severity": getattr(finding, "severity", ""),
        "confidence": float(getattr(finding, "confidence", 0.0) or 0.0),
        "file_path": getattr(finding, "file_path", ""),
        "function_name": getattr(finding, "function_name", ""),
        "line_start": int(getattr(finding, "line_start", 0) or 0),
        "line_end": int(getattr(finding, "line_end", 0) or 0),
        "evidence": getattr(finding, "evidence", ""),
        "suggested_fix": getattr(finding, "suggested_fix", None),
        "detection_method": getattr(finding, "detection_method", ""),
        "metadata": dict(getattr(finding, "metadata", None) or {}),
    }


def _ungrounded_candidates_from_report(report: Any) -> List[Dict[str, Any]]:
    """Slim view of grounding rejections for the UI."""
    candidates: List[Dict[str, Any]] = []
    for result in list(getattr(report, "rejected", None) or []):
        nested = getattr(result, "report", None)
        base = _finding_to_dict(nested) if nested is not None else {}
        status = getattr(result, "status", "")
        status_value = getattr(status, "value", status)
        match_type = getattr(result, "match_type", "")
        match_value = getattr(match_type, "value", match_type)
        candidates.append(
            {
                "bug_type": base.get("bug_type") or "",
                "description": base.get("description") or "",
                "severity": base.get("severity") or "",
                "confidence": float(base.get("confidence") or 0.0),
                "file_path": getattr(result, "file_path", None)
                or base.get("file_path")
                or "",
                "function_name": base.get("function_name") or "",
                "line_start": int(
                    getattr(result, "line_start", None)
                    or base.get("line_start")
                    or 0
                ),
                "line_end": int(
                    getattr(result, "line_end", None) or base.get("line_end") or 0
                ),
                "evidence": getattr(result, "expected_evidence", None)
                or base.get("evidence")
                or "",
                "actual_source": getattr(result, "actual_source", "") or "",
                "suggested_fix": base.get("suggested_fix"),
                "detection_method": base.get("detection_method") or "",
                "grounding_status": str(status_value or ""),
                "grounding_reason": getattr(result, "reason", "") or "",
                "match_type": str(match_value or ""),
                "found_at_line": getattr(result, "found_at_line", None),
            }
        )
    return candidates


def _analysis_to_dict(report: Any) -> Dict[str, Any]:
    """Slim JSON-ready view of a CodeAnalysisReport (no retrieval context)."""
    findings = [
        _finding_to_dict(finding)
        for finding in list(getattr(report, "findings", None) or [])
    ]

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

    ungrounded = _ungrounded_candidates_from_report(report)
    return {
        "repository_path": getattr(report, "repository_path", ""),
        "question": getattr(report, "question", ""),
        "findings": findings,
        "ungrounded_candidates": ungrounded,
        "answer": getattr(report, "answer", "") or "",
        "notes": list(getattr(report, "notes", None) or []),
        "duration_seconds": float(getattr(report, "duration_seconds", 0.0) or 0.0),
        "model_used": bool(getattr(report, "model_used", False)),
        "duplicates_removed": int(getattr(report, "duplicates_removed", 0) or 0),
        "llm_grounded_count": int(getattr(report, "llm_grounded_count", 0) or 0),
        "abstention": abstention_data,
    }


def _write(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


class JobCancelled(Exception):
    """Raised when a cooperative cancel is requested before/during a job."""


def execute_job(
    *,
    job: str,
    repo: str,
    out: str,
    progress_path: str = "",
    question: str = "Find bugs and potential issues",
    mode: str = "",
    file_path: str = "",
    function_name: str = "",
    class_name: str = "",
    write_to_disk: bool = False,
    replace_existing: bool = False,
    supervisor: Any = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Run one agent job, write ``out``, and return the result payload.

    When ``supervisor`` is provided (warm worker server), it is reused so
    embeddings stay loaded. Otherwise a fresh Supervisor is built.
    """
    progress = ProgressWriter(progress_path)
    progress.emit("job_started", f"Starting {job} job...")

    def _cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    try:
        if _cancelled():
            raise JobCancelled("cancelled by user")

        from service import (
            build_supervisor,
            run_analysis,
            run_documentation,
            run_testing,
        )

        owns_supervisor = supervisor is None
        if owns_supervisor:
            supervisor = build_supervisor()
        restore_tracer = _attach_tracer_progress(supervisor, progress)
        try:
            if _cancelled():
                raise JobCancelled("cancelled by user")

            if job == "analysis":
                progress.emit("analysis_started", "Starting code analysis...")
                report = run_analysis(supervisor, repo, question=question)
                payload: Dict[str, Any] = {
                    "ok": True,
                    "job": job,
                    "result": _analysis_to_dict(report),
                }
            elif job == "documentation":
                progress.emit("documentation_started", "Starting documentation...")
                result = run_documentation(
                    supervisor,
                    repo,
                    mode=mode,
                    file_path=file_path,
                    function_name=function_name,
                    class_name=class_name,
                    write_to_disk=bool(write_to_disk),
                    replace_existing=bool(replace_existing),
                )
                payload = {
                    "ok": True,
                    "job": job,
                    "result": result.model_dump(),
                }
            elif job == "testing":
                progress.emit("testing_started", "Starting test generation...")
                result = run_testing(
                    supervisor,
                    repo,
                    mode=mode,
                    file_path=file_path,
                    function_name=function_name,
                )
                payload = {
                    "ok": True,
                    "job": job,
                    "result": result.model_dump(),
                }
            else:
                raise ValueError(f"Unknown job type: {job!r}")

            if _cancelled():
                raise JobCancelled("cancelled by user")

            _write(out, payload)
            progress.emit("job_finished", f"{job.capitalize()} job complete")
            return payload
        finally:
            restore_tracer()
    except JobCancelled as exc:
        payload = {
            "ok": False,
            "job": job,
            "error": str(exc) or "cancelled by user",
            "cancelled": True,
        }
        _write(out, payload)
        progress.emit("job_failed", "Job cancelled")
        return payload
    except Exception as exc:  # noqa: BLE001 — always emit a JSON error payload
        payload = {
            "ok": False,
            "job": job,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        progress.emit("job_failed", f"Worker failed: {exc}")
        _write(out, payload)
        return payload


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

    payload = execute_job(
        job=args.job,
        repo=args.repo,
        out=args.out,
        progress_path=args.progress,
        question=args.question,
        mode=args.mode,
        file_path=args.file,
        function_name=args.function,
        class_name=args.class_name,
        write_to_disk=bool(args.write_to_disk),
        replace_existing=bool(args.replace_existing),
    )
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
