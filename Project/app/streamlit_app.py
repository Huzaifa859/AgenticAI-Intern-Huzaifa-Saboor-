"""
app/streamlit_app.py
====================

Lightweight Streamlit UI for the Codebase Assistant.

Run from the Project directory:

    streamlit run app/streamlit_app.py --server.fileWatcherType=none

Or double-click / run:

    run_ui.bat

Agent jobs run in a separate ``worker.py`` process so embedding/LLM
memory pressure cannot kill the Streamlit server. Live stage progress
is tailed from an NDJSON progress file; completed runs are kept in a
capped sidebar history.
"""

from __future__ import annotations

import html
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKER_PATH = os.path.join(_APP_DIR, "worker.py")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from codebase_assistant.exceptions.tool_exceptions import (  # noqa: E402
    InvalidRepositoryURLError,
    RepositoryCloneError,
)
from service import (  # noqa: E402
    RepositoryPathError,
    prepare_repository,
    provider_status_message,
)
from ui_export import (  # noqa: E402
    export_filename,
    markdown_for_agent,
    result_to_json,
)
from ui_history import (  # noqa: E402
    append_history,
    clear_history,
    format_history_label,
    format_history_summary,
    load_history,
    make_run_entry,
    summarize_result,
)
from ui_reports import (  # noqa: E402
    render_analysis_report,
    render_documentation_result,
    render_testing_result,
)

DOC_MODES = ("readme", "file", "function", "class")
TEST_MODES = ("repository", "file", "function")
AGENTS = ("Analysis", "Documentation", "Testing")
RESULT_TABS = ("Analysis", "Documentation", "Testing")

#: Approximate pipeline weights (0–1) for the live progress bar.
_STAGE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "analysis": {
        "job_started": 0.04,
        "before_ingest": 0.12,
        "indexing": 0.18,
        "after_ingest": 0.28,
        "analysis_started": 0.32,
        "static_analysis": 0.40,
        "retrieval": 0.48,
        "before_agent_run": 0.52,
        "before_model_call": 0.58,
        "model_request": 0.62,
        "after_model_call": 0.82,
        "model_response": 0.86,
        "after_agent_run": 0.92,
        "job_finished": 1.0,
        "job_failed": 1.0,
    },
    "documentation": {
        "job_started": 0.04,
        "before_ingest": 0.10,
        "indexing": 0.14,
        "after_ingest": 0.20,
        "documentation_started": 0.24,
        "documentation_ast_scan_finished": 0.32,
        "before_model_call": 0.40,
        "model_request": 0.44,
        "documentation_symbol_started": 0.48,
        "documentation_symbol_finished": 0.58,
        "after_model_call": 0.62,
        "model_response": 0.66,
        "documentation_retry_started": 0.70,
        "documentation_merge_started": 0.74,
        "documentation_merge_finished": 0.80,
        "documentation_grounding_started": 0.86,
        "documentation_grounding_finished": 0.92,
        "documentation_finished": 0.96,
        "job_finished": 1.0,
        "job_failed": 1.0,
    },
    "testing": {
        "job_started": 0.04,
        "before_ingest": 0.10,
        "indexing": 0.14,
        "after_ingest": 0.20,
        "testing_started": 0.26,
        "before_model_call": 0.36,
        "model_request": 0.40,
        "testing_symbol_generation_finished": 0.55,
        "after_model_call": 0.60,
        "model_response": 0.64,
        "testing_merge_completed": 0.72,
        "testing_repair_started": 0.82,
        "testing_finished": 0.94,
        "job_finished": 1.0,
        "job_failed": 1.0,
    },
}

_PROGRESS_CSS = """
<style>
@keyframes ca-pulse {
  0%, 100% { opacity: 0.55; }
  50% { opacity: 1; }
}
@keyframes ca-shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
/* Hide Streamlit toolbar "Running" indicator next to Deploy.
   Our job monitor uses a fragment auto-refresh; that widget otherwise
   blinks/resizes beside Deploy and looks like a bug. */
div[data-testid="stStatusWidget"] {
  display: none !important;
}
.ca-progress-wrap {
  margin: 0.35rem 0 0.85rem;
  padding: 0.85rem 1rem 0.95rem;
  border-radius: 12px;
  border: 1px solid rgba(49, 51, 63, 0.14);
  background: linear-gradient(
    120deg,
    #f8fafc 0%,
    #eef2ff 45%,
    #f8fafc 100%
  );
  background-size: 220% 100%;
  animation: ca-shimmer 4.5s ease-in-out infinite;
}
.ca-progress-wrap.ca-done {
  animation: none;
  background: #f0fdf4;
  border-color: rgba(22, 163, 74, 0.28);
}
.ca-progress-wrap.ca-error {
  animation: none;
  background: #fef2f2;
  border-color: rgba(220, 38, 38, 0.28);
}
.ca-progress-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.55rem;
  font-size: 0.92rem;
}
.ca-progress-title {
  font-weight: 600;
  color: #0f172a;
  display: flex;
  align-items: center;
  gap: 0.55rem;
}
.ca-dot {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 999px;
  background: #2563eb;
  box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.15);
  animation: ca-pulse 1.2s ease-in-out infinite;
}
.ca-progress-wrap.ca-done .ca-dot {
  background: #16a34a;
  box-shadow: 0 0 0 4px rgba(22, 163, 74, 0.15);
  animation: none;
}
.ca-progress-wrap.ca-error .ca-dot {
  background: #dc2626;
  box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.15);
  animation: none;
}
.ca-progress-meta {
  color: #64748b;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.ca-stage-line {
  margin-top: 0.45rem;
  color: #334155;
  font-size: 0.88rem;
}
.ca-stage-line strong {
  color: #0f172a;
  font-weight: 600;
}
.ca-history-card {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 10px;
  padding: 0.55rem 0.65rem 0.45rem;
  margin-bottom: 0.45rem;
  background: rgba(255, 255, 255, 0.72);
}
.ca-history-title {
  font-size: 0.84rem;
  font-weight: 600;
  color: #0f172a;
  margin-bottom: 0.15rem;
}
.ca-history-summary {
  font-size: 0.78rem;
  color: #64748b;
  margin-bottom: 0.35rem;
  line-height: 1.35;
}
</style>
"""


def _local_now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _inject_progress_styles() -> None:
    """Inject CSS for the live progress panel and history cards."""
    st.markdown(_PROGRESS_CSS, unsafe_allow_html=True)


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _stage_progress(job: str, stage: str, current: float) -> float:
    """Advance the bar using known stage weights; never move backwards."""
    weights = _STAGE_WEIGHTS.get(job) or {}
    target = float(weights.get(stage, 0.0) or 0.0)
    if target <= 0:
        return min(0.95, max(current, current + 0.015 if current < 0.9 else current))
    return max(current, min(1.0, target))


def _render_progress_panel(
    *,
    job: str,
    fraction: float,
    stage_message: str,
    elapsed: float,
    state: str = "running",
) -> None:
    """
    Draw the live progress header + Streamlit progress bar.

    Uses lightweight Streamlit widgets (not remounted custom HTML) so
    fragment auto-refresh does not blink the page.
    """
    pct = int(round(max(0.0, min(1.0, fraction)) * 100))
    title = {
        "running": f"Running {job.capitalize()}",
        "complete": f"{job.capitalize()} complete",
        "error": f"{job.capitalize()} failed",
    }.get(state, f"Running {job.capitalize()}")
    st.markdown(f"**{title}** · {pct}% · {_format_elapsed(elapsed)}")
    if stage_message:
        st.caption(f"Current stage: {stage_message}")
    st.progress(max(0.0, min(1.0, fraction)))


def _init_state() -> None:
    """Ensure session keys exist and hydrate history from disk once."""
    defaults = {
        "repo_path": None,
        "repo_reference": "",
        "provider_status": "",
        "last_analysis": None,
        "last_documentation": None,
        "last_testing": None,
        "last_doc_target": "",
        "last_error": "",
        "last_agent": "",
        "run_history": None,
        "history_loaded": False,
        "viewing_history_id": None,
        "viewing_history_label": "",
        "active_result_tab": "Analysis",
        "active_job": None,
        "job_log": [],
        "job_show_stages": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.history_loaded:
        st.session_state.run_history = load_history()
        st.session_state.history_loaded = True
    if st.session_state.run_history is None:
        st.session_state.run_history = []
    if st.session_state.active_result_tab not in RESULT_TABS:
        st.session_state.active_result_tab = "Analysis"
    if not st.session_state.provider_status:
        try:
            st.session_state.provider_status = provider_status_message()
        except Exception:
            st.session_state.provider_status = ""


def _clear_results() -> None:
    """Drop cached agent outputs when the repository changes."""
    st.session_state.last_analysis = None
    st.session_state.last_documentation = None
    st.session_state.last_testing = None
    st.session_state.last_doc_target = ""
    st.session_state.last_error = ""
    st.session_state.last_agent = ""
    st.session_state.viewing_history_id = None
    st.session_state.viewing_history_label = ""


def _set_active_tab(agent: str) -> None:
    """Select the result pane matching an agent name."""
    if agent in RESULT_TABS:
        st.session_state.active_result_tab = agent


def _load_repository(reference: str) -> None:
    """Prepare a repository without loading embedding models in-process."""
    ref = (reference or "").strip()
    if not ref:
        st.session_state.last_error = "Enter a local path or GitHub URL."
        return

    previous = st.session_state.repo_reference
    try:
        path = prepare_repository(ref)
    except (RepositoryPathError, InvalidRepositoryURLError, RepositoryCloneError) as exc:
        st.session_state.last_error = str(exc)
        return
    except Exception as exc:  # noqa: BLE001 — surface unexpected prep failures
        st.session_state.last_error = f"Could not prepare repository: {exc}"
        return

    if previous != ref:
        _clear_results()

    st.session_state.repo_reference = ref
    st.session_state.repo_path = path
    st.session_state.last_error = ""


def _tail_progress(
    path: str, offset: int
) -> Tuple[List[Dict[str, str]], int]:
    """Read new NDJSON progress events from ``offset``."""
    if not path or not os.path.isfile(path):
        return [], offset
    try:
        with open(path, "r", encoding="utf-8") as handle:
            handle.seek(offset)
            chunk = handle.read()
            new_offset = handle.tell()
    except OSError:
        return [], offset

    events: List[Dict[str, str]] = []
    for line in chunk.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            events.append({"stage": "progress", "message": text})
            continue
        stage = str(payload.get("stage") or "progress").strip()
        message = str(payload.get("message") or stage or "").strip()
        if message:
            events.append({"stage": stage, "message": message})
    return events, new_offset


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` still looks alive."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            return str(pid) in (completed.stdout or "")
        except OSError:
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill_pid(pid: int) -> None:
    """Force-stop a worker process tree."""
    if pid <= 0:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def _cleanup_paths(*paths: str) -> None:
    for path in paths:
        if not path:
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def _record_history(
    *,
    agent: str,
    target: str,
    ok: bool,
    result: Optional[Dict[str, Any]] = None,
    error: str = "",
    started_at: str,
) -> None:
    """Append one run to session + disk history."""
    entry = make_run_entry(
        agent=agent,
        repo_reference=str(st.session_state.repo_reference or ""),
        target=target,
        ok=ok,
        error=error,
        summary=summarize_result(agent, result, error=error),
        result=result or {},
        started_at=started_at,
        finished_at=_local_now_iso(),
    )
    st.session_state.run_history = append_history(
        list(st.session_state.run_history or []),
        entry,
    )
    st.session_state.viewing_history_id = None
    st.session_state.viewing_history_label = ""


def _apply_success_result(agent: str, result: Dict[str, Any], target: str) -> None:
    """Store a successful agent result and focus its tab."""
    if agent == "Analysis":
        st.session_state.last_analysis = result
    elif agent == "Documentation":
        st.session_state.last_documentation = result
        st.session_state.last_doc_target = target
    elif agent == "Testing":
        st.session_state.last_testing = result
    st.session_state.last_agent = agent
    st.session_state.last_error = ""
    _set_active_tab(agent)


def _start_worker(
    agent: str,
    job: str,
    repo_path: str,
    *,
    target: str,
    **fields: Any,
) -> None:
    """Launch the worker subprocess and park state for the live monitor."""
    fd, out_path = tempfile.mkstemp(prefix="ca_ui_", suffix=".json")
    os.close(fd)
    progress_fd, progress_path = tempfile.mkstemp(prefix="ca_ui_", suffix=".ndjson")
    os.close(progress_fd)
    err_fd, stderr_path = tempfile.mkstemp(prefix="ca_ui_", suffix=".err")
    os.close(err_fd)
    try:
        with open(progress_path, "w", encoding="utf-8"):
            pass
    except OSError:
        pass

    cmd = [
        sys.executable,
        _WORKER_PATH,
        "--job",
        job,
        "--repo",
        repo_path,
        "--out",
        out_path,
        "--progress",
        progress_path,
    ]
    if "question" in fields:
        cmd.extend(["--question", str(fields.get("question") or "")])
    if fields.get("mode"):
        cmd.extend(["--mode", str(fields["mode"])])
    if fields.get("file_path"):
        cmd.extend(["--file", str(fields["file_path"])])
    if fields.get("function_name"):
        cmd.extend(["--function", str(fields["function_name"])])
    if fields.get("class_name"):
        cmd.extend(["--class-name", str(fields["class_name"])])
    if fields.get("write_to_disk"):
        cmd.append("--write-to-disk")
    if fields.get("replace_existing"):
        cmd.append("--replace-existing")

    env = os.environ.copy()
    runtime_root = os.path.join(tempfile.gettempdir(), "codebase_assistant_streamlit")
    env.setdefault("CHROMA_PERSIST_DIR", os.path.join(runtime_root, "chroma"))
    env.setdefault("MEMORY_STORE_PATH", os.path.join(runtime_root, "memory_store"))

    err_handle = open(stderr_path, "w", encoding="utf-8", errors="replace")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=_PROJECT_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=err_handle,
            text=True,
        )
    except Exception:
        err_handle.close()
        _cleanup_paths(out_path, progress_path, stderr_path)
        raise
    finally:
        # Child keeps the inherited handle; parent can close its copy.
        try:
            err_handle.close()
        except OSError:
            pass

    started = time.monotonic()
    st.session_state.job_log = ["Queued — launching isolated worker process…"]
    # Reset stages visibility for each new run; user can hide during the run.
    st.session_state.job_show_stages = True
    st.session_state.active_job = {
        "pid": int(process.pid or 0),
        "job": job,
        "agent": agent,
        "target": target,
        "out_path": out_path,
        "progress_path": progress_path,
        "stderr_path": stderr_path,
        "offset": 0,
        "fraction": 0.02,
        "stage_message": "Starting worker…",
        "started_mono": started,
        "started_at": _local_now_iso(),
        "last_creep": started,
        "last_message": "",
        "cancelled": False,
    }
    st.session_state.last_agent = agent
    st.session_state.last_error = ""
    st.session_state.viewing_history_id = None
    st.session_state.viewing_history_label = ""
    _set_active_tab(agent)


def _finalize_active_job(*, cancelled: bool = False) -> None:
    """Consume worker output (or cancellation) and clear active job state."""
    job_state = st.session_state.get("active_job")
    if not isinstance(job_state, dict):
        return

    agent = str(job_state.get("agent") or "Analysis")
    target = str(job_state.get("target") or "")
    started_at = str(job_state.get("started_at") or _local_now_iso())
    out_path = str(job_state.get("out_path") or "")
    progress_path = str(job_state.get("progress_path") or "")
    stderr_path = str(job_state.get("stderr_path") or "")
    pid = int(job_state.get("pid") or 0)

    try:
        if cancelled:
            _kill_pid(pid)
            message = f"{agent} cancelled by user"
            st.session_state.last_error = message
            _record_history(
                agent=agent,
                target=target,
                ok=False,
                error=message,
                started_at=started_at,
            )
            _set_active_tab(agent)
            return

        payload: Optional[Dict[str, Any]] = None
        if os.path.isfile(out_path):
            try:
                with open(out_path, encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                payload = None

        if payload and payload.get("ok"):
            result = payload.get("result")
            if isinstance(result, dict):
                _apply_success_result(agent, result, target)
                _record_history(
                    agent=agent,
                    target=target,
                    ok=True,
                    result=result,
                    started_at=started_at,
                )
                return
            message = f"{agent} failed: Worker returned no result object."
        else:
            detail = ""
            if payload and payload.get("error"):
                detail = str(payload.get("error"))
            elif os.path.isfile(stderr_path):
                try:
                    with open(stderr_path, encoding="utf-8", errors="replace") as handle:
                        detail = handle.read().strip()
                except OSError:
                    detail = ""
            message = f"{agent} failed: {detail or 'Worker job failed.'}"

        st.session_state.last_error = message
        _record_history(
            agent=agent,
            target=target,
            ok=False,
            error=message,
            started_at=started_at,
        )
        _set_active_tab(agent)
    finally:
        _cleanup_paths(out_path, progress_path, stderr_path)
        st.session_state.active_job = None


def _poll_active_job() -> None:
    """Advance progress state for the in-flight worker; finalize if done."""
    job_state = st.session_state.get("active_job")
    if not isinstance(job_state, dict):
        return

    if job_state.get("cancelled"):
        _finalize_active_job(cancelled=True)
        return

    pid = int(job_state.get("pid") or 0)
    progress_path = str(job_state.get("progress_path") or "")
    events, offset = _tail_progress(progress_path, int(job_state.get("offset") or 0))
    job_state["offset"] = offset
    log = list(st.session_state.job_log or [])
    for event in events:
        message = event.get("message") or ""
        stage = event.get("stage") or "progress"
        if message and message != job_state.get("last_message"):
            log.append(f"→ {message}")
            job_state["last_message"] = message
            job_state["stage_message"] = message
        job_state["fraction"] = _stage_progress(
            str(job_state.get("job") or ""),
            stage,
            float(job_state.get("fraction") or 0.02),
        )
    st.session_state.job_log = log[-40:]

    now = time.monotonic()
    last_creep = float(job_state.get("last_creep") or now)
    fraction = float(job_state.get("fraction") or 0.02)
    if now - last_creep >= 1.2 and fraction < 0.93:
        job_state["fraction"] = min(0.93, fraction + 0.008)
        job_state["last_creep"] = now

    st.session_state.active_job = job_state
    if not _pid_alive(pid):
        # Small grace so the worker can flush result JSON.
        time.sleep(0.15)
        _finalize_active_job(cancelled=False)


@st.fragment(run_every=timedelta(seconds=1.5))
def _render_job_monitor_live() -> None:
    """Live progress panel + Stop control while a worker is running."""
    _poll_active_job()
    job_state = st.session_state.get("active_job")
    if not isinstance(job_state, dict):
        st.rerun()
        return

    job = str(job_state.get("job") or "job")
    elapsed = time.monotonic() - float(job_state.get("started_mono") or time.monotonic())
    top = st.columns([4, 1])
    with top[1]:
        if st.button("Stop run", type="secondary", width="stretch", key="stop_run"):
            job_state["cancelled"] = True
            st.session_state.active_job = job_state
            _finalize_active_job(cancelled=True)
            st.rerun()
            return

    _render_progress_panel(
        job=job,
        fraction=float(job_state.get("fraction") or 0.02),
        stage_message=str(job_state.get("stage_message") or "Working…"),
        elapsed=elapsed,
        state="running",
    )

    # Checkbox state persists across fragment refreshes (unlike expander
    # with expanded=True, which kept forcing itself open).
    st.checkbox(
        "Show pipeline stages",
        key="job_show_stages",
        help="Uncheck to hide the stage log while the job runs.",
    )
    if st.session_state.get("job_show_stages", True):
        log_lines = list(st.session_state.job_log or [])
        st.caption(f"Pipeline stages · {_format_elapsed(elapsed)}")
        if log_lines:
            st.markdown("\n".join(f"- {line}" for line in log_lines[-20:]))
        else:
            st.caption("Waiting for worker stages…")


def _render_job_monitor() -> None:
    """Mount the live monitor only while a job is active."""
    if isinstance(st.session_state.get("active_job"), dict):
        _render_job_monitor_live()


def _run_selected_agent(
    agent: str,
    *,
    question: str,
    doc_mode: str,
    test_mode: str,
    file_path: str,
    function_name: str,
    class_name: str,
    write_to_disk: bool,
    replace_existing: bool,
) -> None:
    """Start the selected agent in a cancellable worker subprocess."""
    if st.session_state.get("active_job"):
        st.session_state.last_error = "A job is already running. Stop it first."
        return

    repo_path = st.session_state.repo_path
    if not repo_path:
        st.session_state.last_error = "Load a repository first."
        return

    try:
        if agent == "Analysis":
            target = question[:80] or "analysis"
            _start_worker(
                agent,
                "analysis",
                repo_path,
                target=target,
                question=question,
            )
        elif agent == "Documentation":
            target = file_path or class_name or function_name or doc_mode or "repository"
            _start_worker(
                agent,
                "documentation",
                repo_path,
                target=str(target),
                mode=doc_mode,
                file_path=file_path,
                function_name=function_name,
                class_name=class_name,
                write_to_disk=write_to_disk,
                replace_existing=replace_existing,
            )
        else:
            target = file_path or function_name or test_mode or "repository"
            _start_worker(
                agent,
                "testing",
                repo_path,
                target=str(target),
                mode=test_mode,
                file_path=file_path,
                function_name=function_name,
            )
    except Exception as exc:  # noqa: BLE001 — show launch failures in the UI
        message = f"{agent} failed: {exc}"
        st.session_state.last_error = message
        _record_history(
            agent=agent,
            target="launch",
            ok=False,
            error=message,
            started_at=_local_now_iso(),
        )
        _set_active_tab(agent)


def _restore_history_entry(entry: Dict[str, Any]) -> None:
    """Load a historical result into the matching result tab."""
    agent = str(entry.get("agent") or "")
    result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
    st.session_state.viewing_history_id = entry.get("id")
    st.session_state.viewing_history_label = (
        f"{format_history_label(entry)} — {format_history_summary(entry)}"
    )
    st.session_state.last_error = ""
    if agent == "Analysis":
        st.session_state.last_analysis = result
        st.session_state.last_agent = "Analysis"
    elif agent == "Documentation":
        st.session_state.last_documentation = result
        st.session_state.last_doc_target = str(entry.get("target") or "")
        st.session_state.last_agent = "Documentation"
    elif agent == "Testing":
        st.session_state.last_testing = result
        st.session_state.last_agent = "Testing"
    _set_active_tab(agent)


def _render_export_buttons(agent: str, result: Any, *, doc_target: str = "") -> None:
    """Markdown + JSON download buttons for the active result."""
    md = markdown_for_agent(agent, result, doc_target=doc_target)
    raw = result_to_json(result)
    col_a, col_b, _ = st.columns([1, 1, 2])
    with col_a:
        st.download_button(
            "Download Markdown",
            data=md,
            file_name=export_filename(agent, "md"),
            mime="text/markdown",
            width="stretch",
            key=f"dl_md_{agent}",
        )
    with col_b:
        st.download_button(
            "Download JSON",
            data=raw,
            file_name=export_filename(agent, "json"),
            mime="application/json",
            width="stretch",
            key=f"dl_json_{agent}",
        )


def _render_history_sidebar() -> None:
    """Draw the run-history expander with local times + summaries."""
    with st.sidebar.expander("Run history", expanded=False):
        history: List[Dict[str, Any]] = list(st.session_state.run_history or [])
        if not history:
            st.caption("No runs yet.")
            return

        newest_first = list(reversed(history))
        for entry in newest_first[:20]:
            entry_id = str(entry.get("id") or "")
            title = html.escape(format_history_label(entry))
            summary = html.escape(format_history_summary(entry))
            st.markdown(
                f"""
<div class="ca-history-card">
  <div class="ca-history-title">{title}</div>
  <div class="ca-history-summary">{summary}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"hist_{entry_id}", width="stretch"):
                _restore_history_entry(entry)
                st.rerun()

        if st.button("Clear history", key="clear_history", width="stretch"):
            st.session_state.run_history = []
            clear_history()
            st.session_state.viewing_history_id = None
            st.session_state.viewing_history_label = ""
            st.rerun()


def _render_sidebar() -> None:
    """Draw controls and return nothing; actions mutate session state."""
    st.sidebar.title("Codebase Assistant")
    st.sidebar.caption("Browse analysis, docs, and test reports")

    reference = st.sidebar.text_input(
        "Repository path or GitHub URL",
        value=st.session_state.repo_reference or "examples/demo_repo",
        help="Local path relative to Project/, absolute path, or HTTPS GitHub URL.",
    )

    busy = bool(st.session_state.get("active_job"))
    if st.sidebar.button("Load repository", width="stretch", disabled=busy):
        with st.spinner("Preparing repository..."):
            _load_repository(reference)

    if st.session_state.repo_path:
        st.sidebar.success(f"Ready: `{st.session_state.repo_path}`")
    if st.session_state.provider_status:
        st.sidebar.caption(st.session_state.provider_status)

    _render_history_sidebar()

    st.sidebar.divider()
    agent = st.sidebar.radio("Agent", AGENTS, index=0, disabled=busy)

    question = "Find bugs and potential issues"
    doc_mode = "readme"
    test_mode = "function"
    file_path = ""
    function_name = ""
    class_name = ""
    write_to_disk = False
    replace_existing = False

    if agent == "Analysis":
        question = st.sidebar.text_area(
            "Question",
            value="Find bugs and potential issues",
            height=80,
            disabled=busy,
        )
    elif agent == "Documentation":
        doc_mode = st.sidebar.selectbox(
            "Documentation mode", DOC_MODES, index=2, disabled=busy
        )
        if doc_mode in {"file", "function", "class"}:
            file_path = st.sidebar.text_input("File path", value="", disabled=busy)
        if doc_mode == "function":
            function_name = st.sidebar.text_input(
                "Function name", value="", disabled=busy
            )
        if doc_mode == "class":
            class_name = st.sidebar.text_input("Class name", value="", disabled=busy)
        write_to_disk = st.sidebar.checkbox(
            "Write documentation to disk", value=False, disabled=busy
        )
        replace_existing = False
        if write_to_disk:
            replace_existing = st.sidebar.checkbox(
                "Replace existing documentation", value=False, disabled=busy
            )
    else:
        test_mode = st.sidebar.selectbox(
            "Testing mode", TEST_MODES, index=2, disabled=busy
        )
        if test_mode in {"file", "function"}:
            file_path = st.sidebar.text_input("File path", value="", disabled=busy)
        if test_mode == "function":
            function_name = st.sidebar.text_input(
                "Function name", value="", disabled=busy
            )

    run_disabled = (not bool(st.session_state.repo_path)) or busy
    if st.sidebar.button(
        "Run",
        type="primary",
        width="stretch",
        disabled=run_disabled,
    ):
        _run_selected_agent(
            agent,
            question=question,
            doc_mode=doc_mode,
            test_mode=test_mode,
            file_path=file_path,
            function_name=function_name,
            class_name=class_name,
            write_to_disk=write_to_disk,
            replace_existing=replace_existing,
        )


def _render_result_pane() -> None:
    """Render the selected result tab content with export actions."""
    selected = st.session_state.get("active_result_tab") or "Analysis"
    st.segmented_control(
        "Results",
        options=list(RESULT_TABS),
        key="active_result_tab",
        label_visibility="collapsed",
    )
    selected = st.session_state.get("active_result_tab") or selected

    if selected == "Analysis":
        if st.session_state.last_analysis is not None:
            _render_export_buttons("Analysis", st.session_state.last_analysis)
            render_analysis_report(st.session_state.last_analysis)
        else:
            st.caption("No analysis result in this session.")
    elif selected == "Documentation":
        if st.session_state.last_documentation is not None:
            _render_export_buttons(
                "Documentation",
                st.session_state.last_documentation,
                doc_target=st.session_state.last_doc_target,
            )
            render_documentation_result(
                st.session_state.last_documentation,
                requested_target=st.session_state.last_doc_target,
            )
        else:
            st.caption("No documentation result in this session.")
    else:
        if st.session_state.last_testing is not None:
            _render_export_buttons("Testing", st.session_state.last_testing)
            render_testing_result(st.session_state.last_testing)
        else:
            st.caption("No testing result in this session.")


def _render_main() -> None:
    """Draw status and the latest report."""
    st.title("Codebase Assistant")
    st.markdown(
        "Load a repository in the sidebar, choose an agent, then browse the report here."
    )

    _render_job_monitor()

    if st.session_state.viewing_history_id:
        st.info(
            f"Viewing historical run: {st.session_state.viewing_history_label}. "
            "Click **Run** for a new live job."
        )

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if not st.session_state.repo_path:
        st.info(
            "Start by loading a repository — try `examples/demo_repo` for a quick demo."
        )
        return

    st.caption(
        f"Repository: `{st.session_state.repo_reference}` → `{st.session_state.repo_path}`"
    )

    has_any = any(
        [
            st.session_state.last_analysis,
            st.session_state.last_documentation,
            st.session_state.last_testing,
        ]
    )
    if st.session_state.get("active_job") and not has_any:
        st.caption("Job in progress — results will appear here when finished.")
        return

    if not has_any:
        st.info("No results yet. Choose an agent and click **Run**.")
        return

    _render_result_pane()


def main() -> None:
    """Streamlit entrypoint."""
    st.set_page_config(
        page_title="Codebase Assistant",
        layout="wide",
    )
    _inject_progress_styles()
    _init_state()
    _render_sidebar()
    _render_main()


if __name__ == "__main__":
    main()
