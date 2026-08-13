"""
app/streamlit_app.py
====================

Lightweight Streamlit UI for the Codebase Assistant.

Run from the Project directory:

    streamlit run app/streamlit_app.py --server.fileWatcherType=none

Or double-click / run:

    run_ui.bat

Agent jobs prefer a long-lived warm ``worker_server`` (embeddings +
Supervisor stay loaded). If that server is unavailable, Streamlit falls
back to a one-shot ``worker.py`` subprocess. Live stage progress is
tailed from an NDJSON progress file; completed runs are kept in a
capped sidebar history. Session conversation memory (repo/targets/short
summaries) is owned by the Streamlit process via ``ui_memory``.
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

from worker_client import (  # noqa: E402
    cancel_job as _cancel_warm_job,
    ensure_worker_server,
    job_status as _warm_job_status,
    submit_job as _submit_warm_job,
)

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
from ui_memory import (  # noqa: E402
    build_streamlit_conversation_memory,
    clear_conversation_memory,
    memory_target,
    record_memory_message,
    record_repository_loaded,
    remembered_repository_reference,
    store_memory_target,
    summarize_analysis_for_memory,
    summarize_documentation_for_memory,
    summarize_testing_for_memory,
)
from ui_paths import chroma_persist_dir, memory_store_path  # noqa: E402
from ui_reports import (  # noqa: E402
    render_analysis_report,
    render_documentation_message,
    render_documentation_result,
    render_testing_result,
)

# Same canonical Chroma/memory bases as worker.py / CLI (explicit env wins).
os.environ.setdefault("CHROMA_PERSIST_DIR", chroma_persist_dir())
os.environ.setdefault("MEMORY_STORE_PATH", memory_store_path())

#: Sidebar widget keys prefilling from ConversationMemory targets.
_SIDEBAR_TARGET_KEYS = (
    "sidebar_file_path",
    "sidebar_function_name",
    "sidebar_class_name",
)

DOC_MODES = ("readme", "file", "function", "class")
TEST_MODES = ("repository", "file", "function")
AGENTS = ("Analysis", "Documentation", "Testing")
RESULT_TABS = ("Analysis", "Documentation", "Testing")


@st.cache_resource(show_spinner="Starting warm worker…")
def _auto_start_worker() -> None:
    """Spawn the warm worker once when Streamlit starts.

    Uses st.cache_resource so this runs exactly once per Streamlit server
    session regardless of how many times the page rerenders. If the worker
    is already running it returns immediately.
    """
    ensure_worker_server()


_auto_start_worker()

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
        "testing_stream_delta": 0.52,
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
@keyframes ca-spin {
  to { transform: rotate(360deg); }
}
/* Hide Streamlit toolbar "Running" indicator next to Deploy. */
div[data-testid="stStatusWidget"] {
  display: none !important;
}
.ca-run-card {
  margin: 0.15rem 0 0.85rem;
  padding: 1rem 1.1rem 1.05rem;
  border-radius: 14px;
  border: 1px solid rgba(15, 23, 42, 0.10);
  /* Solid fill only — no fade/opacity animation (fragment refresh was
     replaying enter animations and making cards twitch/brightness-shift). */
  background: #ffffff;
}
.ca-run-card.ca-done {
  background: #f0fdf4;
  border-color: rgba(22, 163, 74, 0.25);
}
.ca-run-card.ca-error {
  background: #fef2f2;
  border-color: rgba(220, 38, 38, 0.25);
}
.ca-run-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.85rem;
  margin-bottom: 0.7rem;
}
.ca-run-title-row {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  min-width: 0;
}
.ca-spinner {
  width: 1.05rem;
  height: 1.05rem;
  border-radius: 999px;
  border: 2px solid rgba(37, 99, 235, 0.2);
  border-top-color: #2563eb;
  animation: ca-spin 0.85s linear infinite;
  flex: 0 0 auto;
}
.ca-run-card.ca-done .ca-spinner,
.ca-run-card.ca-error .ca-spinner {
  animation: none;
  border-color: transparent;
  background: #16a34a;
  box-shadow: inset 0 0 0 2px #fff;
}
.ca-run-card.ca-error .ca-spinner {
  background: #dc2626;
}
.ca-run-title {
  font-weight: 650;
  font-size: 1.02rem;
  color: #0f172a;
  letter-spacing: -0.01em;
  line-height: 1.25;
}
.ca-run-sub {
  margin-top: 0.2rem;
  color: #475569;
  font-size: 0.88rem;
  line-height: 1.35;
}
.ca-run-sub strong {
  color: #0f172a;
  font-weight: 600;
}
.ca-run-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.2rem;
  color: #64748b;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  font-size: 0.84rem;
}
.ca-run-pct {
  font-size: 1.15rem;
  font-weight: 700;
  color: #1e3a8a;
  letter-spacing: -0.02em;
}
.ca-run-card.ca-done .ca-run-pct { color: #166534; }
.ca-run-card.ca-error .ca-run-pct { color: #991b1b; }
.ca-bar-track {
  position: relative;
  height: 0.55rem;
  border-radius: 999px;
  background: #e2e8f0;
  overflow: hidden;
}
.ca-bar-fill {
  height: 100%;
  border-radius: 999px;
  background: #2563eb;
  transition: width 0.45s ease;
}
.ca-run-card.ca-done .ca-bar-fill {
  background: #16a34a;
}
.ca-run-card.ca-error .ca-bar-fill {
  background: #dc2626;
}
.ca-bar-glow {
  display: none;
}
.ca-stage-panel {
  margin: 0.35rem 0 0.25rem;
  padding: 0.75rem 0.85rem 0.65rem;
  border-radius: 12px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: #f8fafc;
}
.ca-stage-panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.55rem;
  color: #64748b;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.ca-stage-list {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  max-height: 220px;
  overflow-y: auto;
  padding-right: 0.15rem;
}
.ca-stage-item {
  display: flex;
  align-items: flex-start;
  gap: 0.55rem;
  padding: 0.35rem 0.45rem;
  border-radius: 8px;
  color: #334155;
  font-size: 0.88rem;
  line-height: 1.35;
}
.ca-stage-item.ca-active {
  background: #eff6ff;
  color: #0f172a;
  font-weight: 550;
}
.ca-stage-item.ca-done {
  color: #475569;
}
.ca-stage-mark {
  width: 0.55rem;
  height: 0.55rem;
  margin-top: 0.35rem;
  border-radius: 999px;
  flex: 0 0 auto;
  background: #94a3b8;
}
.ca-stage-item.ca-done .ca-stage-mark {
  background: #22c55e;
}
.ca-stage-item.ca-active .ca-stage-mark {
  background: #2563eb;
}
div[data-testid="stProgress"] > div {
  border-radius: 999px !important;
  height: 0.55rem !important;
}
div[data-testid="stProgress"] > div > div {
  border-radius: 999px !important;
  background: linear-gradient(90deg, #1d4ed8, #0f766e) !important;
}
.ca-history-card {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 10px;
  padding: 0.55rem 0.65rem 0.45rem;
  margin-bottom: 0.45rem;
  background: #ffffff;
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
    """Draw the animated live-run progress card + native progress bar."""
    pct = int(round(max(0.0, min(1.0, fraction)) * 100))
    width = max(0.0, min(100.0, float(pct)))
    title = {
        "running": f"Running {job.capitalize()}",
        "complete": f"{job.capitalize()} complete",
        "error": f"{job.capitalize()} failed",
    }.get(state, f"Running {job.capitalize()}")
    wrap = "ca-run-card"
    if state == "complete":
        wrap += " ca-done"
    elif state == "error":
        wrap += " ca-error"
    safe_stage = html.escape(stage_message or "Working…")
    st.markdown(
        f"""
<div class="{wrap}">
  <div class="ca-run-top">
    <div class="ca-run-title-row">
      <div class="ca-spinner" aria-hidden="true"></div>
      <div>
        <div class="ca-run-title">{html.escape(title)}</div>
        <div class="ca-run-sub">Current stage: <strong>{safe_stage}</strong></div>
      </div>
    </div>
    <div class="ca-run-meta">
      <div class="ca-run-pct">{pct}%</div>
      <div>{_format_elapsed(elapsed)} elapsed</div>
    </div>
  </div>
  <div class="ca-bar-track" role="progressbar" aria-valuenow="{pct}" aria-valuemin="0" aria-valuemax="100">
    <div class="ca-bar-fill" style="width: {width:.1f}%;"></div>
    <div class="ca-bar-glow"></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_stage_timeline(log_lines: List[str], *, elapsed: float) -> None:
    """Render recent pipeline stages as a compact animated timeline."""
    lines = [str(line).lstrip("→ ").strip() for line in (log_lines or []) if str(line).strip()]
    items_html: List[str] = []
    if not lines:
        items_html.append(
            '<div class="ca-stage-item ca-active">'
            '<span class="ca-stage-mark"></span>'
            "<span>Waiting for worker stages…</span>"
            "</div>"
        )
    else:
        for index, line in enumerate(lines):
            active = index == len(lines) - 1
            cls = "ca-stage-item ca-active" if active else "ca-stage-item ca-done"
            items_html.append(
                f'<div class="{cls}">'
                f'<span class="ca-stage-mark"></span>'
                f"<span>{html.escape(line)}</span>"
                f"</div>"
            )
    st.markdown(
        f"""
<div class="ca-stage-panel">
  <div class="ca-stage-panel-head">
    <span>Pipeline stages</span>
    <span>{_format_elapsed(elapsed)}</span>
  </div>
  <div class="ca-stage-list">
    {"".join(items_html)}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    """Ensure session keys exist and hydrate history/memory from disk once."""
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
        "doc_stream_text": "",
        "doc_stream_shown": 0,
        "conversation_memory": None,
        "memory_bootstrapped": False,
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
    if not st.session_state.memory_bootstrapped or st.session_state.conversation_memory is None:
        st.session_state.conversation_memory = build_streamlit_conversation_memory()
        st.session_state.memory_bootstrapped = True
        _ensure_sidebar_target_defaults()
        if not st.session_state.repo_reference:
            remembered = remembered_repository_reference(
                st.session_state.conversation_memory
            )
            if remembered and "repo_input" not in st.session_state:
                st.session_state.repo_input = remembered


def _ensure_sidebar_target_defaults() -> None:
    """Prefill docs/testing sidebar keys from memory once per session."""
    remembered = memory_target(st.session_state.get("conversation_memory"))
    mapping = {
        "sidebar_file_path": remembered["file_path"],
        "sidebar_function_name": remembered["function_name"],
        "sidebar_class_name": remembered["class_name"],
    }
    for key, value in mapping.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_sidebar_target_keys() -> None:
    """Drop sidebar target widget keys so empty memory does not re-prefill."""
    for key in _SIDEBAR_TARGET_KEYS:
        st.session_state.pop(key, None)


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
    record_repository_loaded(
        st.session_state.get("conversation_memory"),
        ref,
        path,
    )


def _user_instruction_for_job(job_state: Dict[str, Any]) -> str:
    """Short user-turn text for ConversationMemory (CLI-parity style)."""
    agent = str(job_state.get("agent") or "")
    if agent == "Analysis":
        question = str(job_state.get("question") or "").strip()
        return f"Analysis: {question}" if question else "Analysis"
    if agent == "Documentation":
        parts = ["Documentation"]
        mode = str(job_state.get("mode") or "").strip()
        if mode:
            parts.append(f"mode={mode}")
        file_path = str(job_state.get("file_path") or "").strip()
        function_name = str(job_state.get("function_name") or "").strip()
        class_name = str(job_state.get("class_name") or "").strip()
        if file_path:
            parts.append(f"file={file_path}")
        if function_name:
            parts.append(f"function={function_name}")
        if class_name:
            parts.append(f"class={class_name}")
        return " ".join(parts)
    if agent == "Testing":
        parts = ["Testing"]
        mode = str(job_state.get("mode") or "").strip()
        if mode:
            parts.append(f"mode={mode}")
        file_path = str(job_state.get("file_path") or "").strip()
        function_name = str(job_state.get("function_name") or "").strip()
        if file_path:
            parts.append(f"file={file_path}")
        if function_name:
            parts.append(f"function={function_name}")
        return " ".join(parts)
    return agent or "Run"


def _record_job_memory(
    job_state: Dict[str, Any],
    *,
    ok: bool,
    result: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> None:
    """Record short user/assistant turns after a Load/Run finishes."""
    memory = st.session_state.get("conversation_memory")
    if memory is None:
        return

    agent = str(job_state.get("agent") or "")
    record_memory_message(memory, "user", _user_instruction_for_job(job_state))

    if not ok:
        record_memory_message(memory, "assistant", (error or f"{agent} failed.").strip())
        return

    payload = result if isinstance(result, dict) else {}
    if agent == "Analysis":
        summary = summarize_analysis_for_memory(payload)
    elif agent == "Documentation":
        store_memory_target(
            memory,
            file_path=str(job_state.get("file_path") or ""),
            function_name=str(job_state.get("function_name") or ""),
            class_name=str(job_state.get("class_name") or ""),
        )
        summary = summarize_documentation_for_memory(payload)
    elif agent == "Testing":
        store_memory_target(
            memory,
            file_path=str(job_state.get("file_path") or ""),
            function_name=str(job_state.get("function_name") or ""),
        )
        summary = summarize_testing_for_memory(payload)
    else:
        summary = f"{agent} completed."
    record_memory_message(memory, "assistant", summary)


def _tail_progress(
    path: str, offset: int
) -> Tuple[List[Dict[str, Any]], int]:
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

    events: List[Dict[str, Any]] = []
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
        if not message:
            continue
        event: Dict[str, Any] = {"stage": stage, "message": message}
        # Stream deltas carry token text in ``extra`` — keep it so the
        # live documentation box can grow while the job is still running.
        extra = payload.get("extra")
        if isinstance(extra, dict) and extra:
            event["extra"] = extra
        events.append(event)
    return events, new_offset


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` still looks alive (not exited / not a zombie)."""
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
    # Reap our own exited children so they don't linger as zombies. On Linux,
    # os.kill(pid, 0) still succeeds for zombies, which previously left the UI
    # stuck at 100% after "job complete".
    try:
        waited_pid, _status = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("State:"):
                    # Z = zombie (exited, not yet reaped)
                    return "Z" not in line.split()[1:2]
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _worker_exited(job_state: Dict[str, Any]) -> bool:
    """Return True when the one-shot worker subprocess has exited."""
    if str(job_state.get("transport") or "") == "warm":
        return False
    process = job_state.get("process")
    if isinstance(process, subprocess.Popen):
        try:
            if process.poll() is not None:
                return True
        except OSError:
            return True
    return not _pid_alive(int(job_state.get("pid") or 0))


def _warm_job_finished(job_state: Dict[str, Any]) -> bool:
    """Return True when the warm worker reports a terminal job status."""
    if str(job_state.get("transport") or "") != "warm":
        return False
    job_id = str(job_state.get("job_id") or "")
    if not job_id:
        return False
    try:
        status = str(_warm_job_status(job_id).get("status") or "")
    except Exception:
        # If the server vanished, fall back to result-file readiness.
        return _result_ready(job_state)
    job_state["warm_status"] = status
    return status in {"done", "error", "cancelled"}


def _result_ready(job_state: Dict[str, Any]) -> bool:
    """True when the worker wrote a finished result payload we can consume."""
    out_path = str(job_state.get("out_path") or "")
    if not out_path or not os.path.isfile(out_path):
        return False
    try:
        with open(out_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and ("ok" in payload)


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
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
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


def _start_oneshot_worker(
    *,
    job: str,
    repo_path: str,
    out_path: str,
    progress_path: str,
    stderr_path: str,
    fields: Dict[str, Any],
    env: Dict[str, str],
) -> subprocess.Popen:
    """Launch the legacy one-shot worker.py subprocess."""
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

    err_handle = open(stderr_path, "w", encoding="utf-8", errors="replace")
    try:
        return subprocess.Popen(
            cmd,
            cwd=_PROJECT_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=err_handle,
            text=True,
        )
    finally:
        try:
            err_handle.close()
        except OSError:
            pass


def _start_worker(
    agent: str,
    job: str,
    repo_path: str,
    *,
    target: str,
    **fields: Any,
) -> None:
    """Submit a job to the warm worker (or one-shot fallback) and monitor it."""
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

    env = os.environ.copy()
    env.setdefault("CHROMA_PERSIST_DIR", chroma_persist_dir())
    env.setdefault("MEMORY_STORE_PATH", memory_store_path())

    transport = "oneshot"
    process: Optional[subprocess.Popen] = None
    job_id = ""
    stage_message = "Starting worker…"
    log_line = "Queued — launching isolated worker process…"

    health = ensure_worker_server(env=env)
    if health is not None and health.get("model_loaded"):
        try:
            submitted = _submit_warm_job(
                {
                    "job": job,
                    "repo": repo_path,
                    "out": out_path,
                    "progress": progress_path,
                    "question": str(fields.get("question") or ""),
                    "mode": str(fields.get("mode") or ""),
                    "file": str(fields.get("file_path") or ""),
                    "function": str(fields.get("function_name") or ""),
                    "class_name": str(fields.get("class_name") or ""),
                    "write_to_disk": bool(fields.get("write_to_disk")),
                    "replace_existing": bool(fields.get("replace_existing")),
                }
            )
            job_id = str(submitted.get("id") or "")
            transport = "warm"
            stage_message = "Warm worker ready…"
            log_line = "Queued — using warm worker (embeddings already loaded)…"
        except Exception:
            transport = "oneshot"

    if transport != "warm":
        try:
            process = _start_oneshot_worker(
                job=job,
                repo_path=repo_path,
                out_path=out_path,
                progress_path=progress_path,
                stderr_path=stderr_path,
                fields=fields,
                env=env,
            )
        except Exception:
            _cleanup_paths(out_path, progress_path, stderr_path)
            raise
        stage_message = "Starting worker…"
        log_line = "Queued — launching isolated worker process…"

    started = time.monotonic()
    st.session_state.job_log = [log_line]
    # Reset stages visibility for each new run; user can hide during the run.
    st.session_state.job_show_stages = True
    st.session_state.doc_stream_text = ""
    st.session_state.doc_stream_shown = 0
    st.session_state.active_job = {
        "transport": transport,
        "job_id": job_id,
        "pid": int(process.pid or 0) if process is not None else 0,
        # Keep the Popen handle so we can poll()/wait() and reap zombies on Linux.
        "process": process,
        "job": job,
        "agent": agent,
        "target": target,
        "question": str(fields.get("question") or ""),
        "mode": str(fields.get("mode") or ""),
        "file_path": str(fields.get("file_path") or ""),
        "function_name": str(fields.get("function_name") or ""),
        "class_name": str(fields.get("class_name") or ""),
        "out_path": out_path,
        "progress_path": progress_path,
        "stderr_path": stderr_path,
        "offset": 0,
        "fraction": 0.02,
        "stage_message": stage_message,
        "started_mono": started,
        "started_at": _local_now_iso(),
        "last_creep": started,
        "last_message": "",
        "cancelled": False,
        "terminal_stage": False,
        "warm_status": "",
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
    process = job_state.get("process")

    try:
        if cancelled:
            warm_id = str(job_state.get("job_id") or "")
            if str(job_state.get("transport") or "") == "warm" and warm_id:
                try:
                    _cancel_warm_job(warm_id)
                except Exception:
                    pass
            if isinstance(process, subprocess.Popen):
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    _kill_pid(pid)
            elif pid > 0:
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
            _record_job_memory(job_state, ok=False, error=message)
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
                _record_job_memory(job_state, ok=True, result=result)
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
        _record_job_memory(job_state, ok=False, error=message)
        _set_active_tab(agent)
    finally:
        if isinstance(process, subprocess.Popen):
            try:
                if process.poll() is None:
                    # Result is already on disk; don't hang the UI on slow
                    # interpreter teardown (common with embedding libs).
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        try:
                            process.wait(timeout=2)
                        except Exception:
                            pass
                else:
                    try:
                        process.wait(timeout=0.1)
                    except Exception:
                        pass
            except Exception:
                _kill_pid(pid)
        elif pid > 0 and sys.platform != "win32":
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass
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

    progress_path = str(job_state.get("progress_path") or "")
    events, offset = _tail_progress(progress_path, int(job_state.get("offset") or 0))
    job_state["offset"] = offset
    log = list(st.session_state.job_log or [])
    stream_buf = str(st.session_state.get("doc_stream_text") or "")
    for event in events:
        message = event.get("message") or ""
        stage = event.get("stage") or "progress"
        if stage in {
            "documentation_stream_delta",
            "analysis_stream_delta",
            "testing_stream_delta",
        }:
            extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
            chunk = str((extra or {}).get("text") or "")
            if chunk:
                stream_buf += chunk
            # Keep a single stage line; don't spam the timeline per token.
            if message and message != job_state.get("last_message"):
                log.append(f"→ {message}")
                job_state["last_message"] = message
                job_state["stage_message"] = message
            job_state["fraction"] = _stage_progress(
                str(job_state.get("job") or ""),
                "model_request",
                float(job_state.get("fraction") or 0.02),
            )
            continue
        if stage in {"job_finished", "job_failed"}:
            job_state["terminal_stage"] = True
        if message and message != job_state.get("last_message"):
            log.append(f"→ {message}")
            job_state["last_message"] = message
            job_state["stage_message"] = message
        job_state["fraction"] = _stage_progress(
            str(job_state.get("job") or ""),
            stage,
            float(job_state.get("fraction") or 0.02),
        )
    st.session_state.doc_stream_text = stream_buf
    st.session_state.job_log = log[-40:]

    now = time.monotonic()
    last_creep = float(job_state.get("last_creep") or now)
    fraction = float(job_state.get("fraction") or 0.02)
    if now - last_creep >= 1.2 and fraction < 0.93:
        job_state["fraction"] = min(0.93, fraction + 0.008)
        job_state["last_creep"] = now

    st.session_state.active_job = job_state
    # Finalize when the one-shot process exits, the warm server reports a
    # terminal status, or the worker already wrote the result JSON.
    if (
        _worker_exited(job_state)
        or _warm_job_finished(job_state)
        or (job_state.get("terminal_stage") and _result_ready(job_state))
    ):
        # Small grace so the worker can flush result JSON.
        time.sleep(0.15)
        warm_status = str(job_state.get("warm_status") or "")
        _finalize_active_job(cancelled=warm_status == "cancelled")


@st.fragment(run_every=timedelta(milliseconds=100))
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
        _render_stage_timeline(
            list(st.session_state.job_log or [])[-12:],
            elapsed=elapsed,
        )

    if job in {"documentation", "analysis", "testing"}:
        full = str(st.session_state.get("doc_stream_text") or "")
        # Free OpenRouter models often buffer the whole SSE body and
        # deliver it in one burst. Reveal gradually so the live panel
        # still looks like streaming instead of a sudden dump.
        shown = int(st.session_state.get("doc_stream_shown") or 0)
        if shown > len(full):
            shown = len(full)
        if shown < len(full):
            remaining = len(full) - shown
            step = 64 if remaining < 320 else min(360, max(96, remaining // 10))
            shown = min(len(full), shown + step)
        st.session_state.doc_stream_shown = shown
        visible = full[:shown]
        if job == "analysis" and visible.strip():
            visible = f"```json\n{visible}"
        render_documentation_message(
            visible,
            live=True,
            key=f"{job}_stream_assistant",
        )


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

    if agent == "Documentation":
        if doc_mode == "function" and not function_name.strip():
            st.session_state.last_error = "Enter a function name for function mode."
            return
        if doc_mode == "class" and not class_name.strip():
            st.session_state.last_error = "Enter a class name for class mode."
            return
        if doc_mode == "file" and not file_path.strip():
            st.session_state.last_error = "Enter a file path for file mode."
            return
    elif agent == "Testing":
        if test_mode == "function" and not function_name.strip():
            st.session_state.last_error = "Enter a function name for function mode."
            return
        if test_mode == "file" and not file_path.strip():
            st.session_state.last_error = "Enter a file path for file mode."
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


def _render_memory_sidebar() -> None:
    """Compact Session memory panel (conversational context, not run history)."""
    with st.sidebar.expander("Session memory", expanded=False):
        memory = st.session_state.get("conversation_memory")
        if memory is None:
            st.caption("Memory unavailable.")
            return

        meta = getattr(memory, "metadata", {}) or {}
        repo_ref = str(meta.get("repository_reference") or "").strip()
        repo_path = str(meta.get("repository_path") or "").strip()
        target = memory_target(memory)

        if repo_ref or repo_path:
            if repo_ref:
                st.caption(f"Repo: `{repo_ref}`")
            if repo_path:
                st.caption(f"Path: `{repo_path}`")
        else:
            st.caption("No repository remembered.")

        target_bits = []
        if target["file_path"]:
            target_bits.append(f"file=`{target['file_path']}`")
        if target["function_name"]:
            target_bits.append(f"function=`{target['function_name']}`")
        if target["class_name"]:
            target_bits.append(f"class=`{target['class_name']}`")
        if target_bits:
            st.caption("Last target: " + " · ".join(target_bits))

        turns = memory.get_history(limit=8)
        if turns:
            st.markdown("**Recent turns**")
            for message in turns:
                role = str(getattr(message, "role", "") or "message")
                content = str(getattr(message, "content", "") or "").strip()
                if len(content) > 160:
                    content = content[:157].rstrip() + "..."
                st.caption(f"{role}: {content}")
        else:
            st.caption("No conversation turns yet.")

        if st.button("Clear memory", key="clear_memory", width="stretch"):
            clear_conversation_memory(memory)
            _clear_sidebar_target_keys()
            st.session_state.pop("repo_input", None)
            remembered = remembered_repository_reference(memory)
            if not remembered:
                st.session_state.repo_input = "examples/demo_repo"
            _ensure_sidebar_target_defaults()
            st.rerun()


def _render_sidebar() -> None:
    """Draw controls and return nothing; actions mutate session state."""
    st.sidebar.title("Codebase Assistant")
    st.sidebar.caption("Browse analysis, docs, and test reports")

    _ensure_sidebar_target_defaults()
    if "repo_input" not in st.session_state:
        remembered = remembered_repository_reference(
            st.session_state.get("conversation_memory")
        )
        st.session_state.repo_input = (
            st.session_state.repo_reference
            or remembered
            or "examples/demo_repo"
        )

    busy = bool(st.session_state.get("active_job"))

    # A form batches the text input and the submit click into one atomic
    # rerun, so a value typed just before clicking can never be dropped
    # (Streamlit otherwise only commits a bare text_input on blur/Enter,
    # which races with a fast click on the button next to it).
    with st.sidebar.form("repo_load_form", clear_on_submit=False):
        reference = st.text_input(
            "Repository path or GitHub URL",
            key="repo_input",
            help="Local path relative to Project/, absolute path, or HTTPS GitHub URL.",
        )
        load_clicked = st.form_submit_button(
            "Load repository", width="stretch", disabled=busy
        )
    if load_clicked:
        with st.spinner("Preparing repository..."):
            _load_repository(reference)

    if st.session_state.repo_path:
        st.sidebar.success(f"Ready: `{st.session_state.repo_path}`")

    _render_history_sidebar()
    _render_memory_sidebar()

    st.sidebar.divider()
    agent = st.sidebar.radio("Agent", AGENTS, index=0, disabled=busy)

    doc_mode = "readme"
    test_mode = "function"
    write_to_disk = False
    replace_existing = False

    # Mode selectors and the write-to-disk toggle must live outside the
    # form below: they conditionally reveal other widgets (e.g. picking
    # "class" mode reveals the Class name box), and forms only apply
    # widget changes on submit, which would make that reveal lag a
    # submission behind.
    if agent == "Documentation":
        doc_mode = st.sidebar.selectbox(
            "Documentation mode", DOC_MODES, index=2, disabled=busy
        )
        write_to_disk = st.sidebar.checkbox(
            "Write documentation to disk", value=False, disabled=busy
        )
        replace_existing = False
        if write_to_disk:
            replace_existing = st.sidebar.checkbox(
                "Replace existing documentation", value=False, disabled=busy
            )
    elif agent == "Testing":
        test_mode = st.sidebar.selectbox(
            "Testing mode", TEST_MODES, index=2, disabled=busy
        )

    # Widgets for a target field that isn't shown in the current agent/mode
    # must not leak a stale value from a previous agent/mode into this
    # request: Streamlit keeps a widget's session_state entry forever once
    # set, even after the widget stops being rendered. Drop any key whose
    # widget is not part of the current agent/mode before it can be read.
    active_target_keys: set[str] = set()
    if agent == "Documentation":
        if doc_mode in {"file", "function", "class"}:
            active_target_keys.add("sidebar_file_path")
        if doc_mode == "function":
            active_target_keys.add("sidebar_function_name")
        if doc_mode == "class":
            active_target_keys.add("sidebar_class_name")
    elif agent == "Testing":
        if test_mode in {"file", "function"}:
            active_target_keys.add("sidebar_file_path")
        if test_mode == "function":
            active_target_keys.add("sidebar_function_name")
    for key in _SIDEBAR_TARGET_KEYS:
        if key not in active_target_keys:
            st.session_state.pop(key, None)

    run_disabled = (not bool(st.session_state.repo_path)) or busy

    # A form batches every free-text field with the Run click into one
    # atomic rerun, removing the widget-value race where a fast click could
    # otherwise submit a stale/blank value typed just before it.
    with st.sidebar.form("run_request_form", clear_on_submit=False):
        question = "Find bugs and potential issues"
        if agent == "Analysis":
            question = st.text_area(
                "Question",
                value="Find bugs and potential issues",
                height=80,
                disabled=busy,
            )
        elif agent == "Documentation":
            if doc_mode in {"file", "function", "class"}:
                st.text_input(
                    "File path", key="sidebar_file_path", disabled=busy
                )
            if doc_mode == "function":
                st.text_input(
                    "Function name", key="sidebar_function_name", disabled=busy
                )
            if doc_mode == "class":
                st.text_input(
                    "Class name", key="sidebar_class_name", disabled=busy
                )
        else:
            if test_mode in {"file", "function"}:
                st.text_input(
                    "File path", key="sidebar_file_path", disabled=busy
                )
            if test_mode == "function":
                st.text_input(
                    "Function name", key="sidebar_function_name", disabled=busy
                )

        run_clicked = st.form_submit_button(
            "Run",
            type="primary",
            width="stretch",
            disabled=run_disabled,
        )

    # Keys may be unbound when the active mode hides those widgets.
    file_path = str(st.session_state.get("sidebar_file_path") or "")
    function_name = str(st.session_state.get("sidebar_function_name") or "")
    class_name = str(st.session_state.get("sidebar_class_name") or "")

    if run_clicked:
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
    active = st.session_state.get("active_job")
    active_job = str(active.get("job") or "") if isinstance(active, dict) else ""
    if active_job == "documentation":
        st.caption(
            "Assistant reply is streaming above — the finished answer keeps "
            "the same message layout."
        )
        return
    if active_job == "analysis":
        st.caption(
            "Analysis is streaming above — when it finishes, findings appear "
            "in the usual report layout."
        )
        return

    if st.session_state.get("active_job") and not has_any:
        st.caption("Job in progress — watch the live output above.")
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
