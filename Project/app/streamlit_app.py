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
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
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
from ui_history import (  # noqa: E402
    append_history,
    clear_history,
    format_history_label,
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
  0%, 100% { opacity: 0.55; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.15); }
}
@keyframes ca-shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
div[data-testid="stStatusWidget"] {
  border: 1px solid rgba(49, 51, 63, 0.16);
  border-radius: 12px;
  padding: 0.15rem 0.35rem 0.35rem;
  background: linear-gradient(
    135deg,
    rgba(250, 250, 252, 0.95) 0%,
    rgba(241, 245, 249, 0.9) 100%
  );
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
</style>
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _inject_progress_styles() -> None:
    """Inject CSS once per page render for the live progress panel."""
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
        # Unknown stage: nudge forward slightly so the bar still feels alive.
        return min(0.95, max(current, current + 0.015 if current < 0.9 else current))
    return max(current, min(1.0, target))


def _render_progress_panel(
    *,
    placeholder: Any,
    job: str,
    fraction: float,
    stage_message: str,
    elapsed: float,
    state: str = "running",
) -> None:
    """Draw the animated progress header + Streamlit progress bar."""
    pct = int(round(max(0.0, min(1.0, fraction)) * 100))
    title = {
        "running": f"Running {job.capitalize()}",
        "complete": f"{job.capitalize()} complete",
        "error": f"{job.capitalize()} failed",
    }.get(state, f"Running {job.capitalize()}")
    wrap_class = "ca-progress-wrap"
    if state == "complete":
        wrap_class += " ca-done"
    elif state == "error":
        wrap_class += " ca-error"
    safe_stage = html.escape(stage_message or "")
    stage_html = (
        f'<div class="ca-stage-line">Current stage: <strong>{safe_stage}</strong></div>'
        if safe_stage
        else ""
    )
    with placeholder.container():
        st.markdown(
            f"""
<div class="{wrap_class}">
  <div class="ca-progress-head">
    <div class="ca-progress-title"><span class="ca-dot"></span>{title}</div>
    <div class="ca-progress-meta">{pct}% · {_format_elapsed(elapsed)}</div>
  </div>
  {stage_html}
</div>
""",
            unsafe_allow_html=True,
        )
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.history_loaded:
        st.session_state.run_history = load_history()
        st.session_state.history_loaded = True
    if st.session_state.run_history is None:
        st.session_state.run_history = []
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


def _run_worker(job: str, repo_path: str, **fields: Any) -> Dict[str, Any]:
    """
    Run one agent job in a subprocess with live progress updates.

    Raises:
        RuntimeError: When the worker exits without a usable result file.
    """
    fd, out_path = tempfile.mkstemp(prefix="ca_ui_", suffix=".json")
    os.close(fd)
    progress_fd, progress_path = tempfile.mkstemp(prefix="ca_ui_", suffix=".ndjson")
    os.close(progress_fd)
    # Start empty so the UI can open the file immediately.
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

    process = subprocess.Popen(
        cmd,
        cwd=_PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    offset = 0
    last_message = ""
    fraction = 0.02
    stage_message = "Starting worker…"
    started = time.monotonic()
    last_creep = started
    payload: Optional[Dict[str, Any]] = None
    panel = st.empty()
    try:
        _render_progress_panel(
            placeholder=panel,
            job=job,
            fraction=fraction,
            stage_message=stage_message,
            elapsed=0.0,
            state="running",
        )
        with st.status("Pipeline stages", expanded=True) as status:
            status.write("Queued — launching isolated worker process…")
            while True:
                events, offset = _tail_progress(progress_path, offset)
                for event in events:
                    message = event.get("message") or ""
                    stage = event.get("stage") or "progress"
                    if message and message != last_message:
                        status.write(f"→ {message}")
                        last_message = message
                        stage_message = message
                    fraction = _stage_progress(job, stage, fraction)

                # Soft creep while a long stage (e.g. model call) is in flight.
                now = time.monotonic()
                if now - last_creep >= 1.2 and fraction < 0.93:
                    fraction = min(0.93, fraction + 0.008)
                    last_creep = now

                # Refresh every tick so elapsed time / bar stay live.
                _render_progress_panel(
                    placeholder=panel,
                    job=job,
                    fraction=fraction,
                    stage_message=stage_message,
                    elapsed=now - started,
                    state="running",
                )
                status.update(
                    label=f"{stage_message} · {_format_elapsed(now - started)}",
                    state="running",
                )

                code = process.poll()
                if code is not None:
                    events, offset = _tail_progress(progress_path, offset)
                    for event in events:
                        message = event.get("message") or ""
                        stage = event.get("stage") or "progress"
                        if message and message != last_message:
                            status.write(f"→ {message}")
                            last_message = message
                            stage_message = message
                        fraction = _stage_progress(job, stage, fraction)
                    break
                time.sleep(0.28)

            stdout, stderr = process.communicate(timeout=5)
            elapsed = time.monotonic() - started
            if not os.path.isfile(out_path):
                detail = (stderr or stdout or "").strip()
                _render_progress_panel(
                    placeholder=panel,
                    job=job,
                    fraction=fraction,
                    stage_message=stage_message or "Worker exited unexpectedly",
                    elapsed=elapsed,
                    state="error",
                )
                status.update(label=f"{job.capitalize()} failed", state="error")
                raise RuntimeError(
                    detail
                    or f"Worker exited with code {process.returncode} and wrote no result."
                )
            with open(out_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("ok"):
                fraction = 1.0
                stage_message = f"{job.capitalize()} finished"
                _render_progress_panel(
                    placeholder=panel,
                    job=job,
                    fraction=fraction,
                    stage_message=stage_message,
                    elapsed=elapsed,
                    state="complete",
                )
                status.update(
                    label=f"{job.capitalize()} complete · {_format_elapsed(elapsed)}",
                    state="complete",
                )
            else:
                _render_progress_panel(
                    placeholder=panel,
                    job=job,
                    fraction=max(fraction, 0.2),
                    stage_message=str(payload.get("error") or "Job failed"),
                    elapsed=elapsed,
                    state="error",
                )
                status.update(label=f"{job.capitalize()} failed", state="error")
    finally:
        for path in (out_path, progress_path):
            try:
                os.remove(path)
            except OSError:
                pass

    if not isinstance(payload, dict):
        raise RuntimeError("Worker returned an invalid payload.")
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error") or "Worker job failed."))
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Worker returned no result object.")
    return result


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
        finished_at=_utc_now_iso(),
    )
    st.session_state.run_history = append_history(
        list(st.session_state.run_history or []),
        entry,
    )
    st.session_state.viewing_history_id = None
    st.session_state.viewing_history_label = ""


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
    """Dispatch the selected agent via worker subprocess and store JSON."""
    repo_path = st.session_state.repo_path
    if not repo_path:
        st.session_state.last_error = "Load a repository first."
        return

    st.session_state.last_error = ""
    st.session_state.last_agent = agent
    st.session_state.viewing_history_id = None
    st.session_state.viewing_history_label = ""
    started_at = _utc_now_iso()

    target = "repository"
    try:
        if agent == "Analysis":
            target = question[:80] or "analysis"
            result = _run_worker(
                "analysis",
                repo_path,
                question=question,
            )
            st.session_state.last_analysis = result
            _record_history(
                agent=agent,
                target=target,
                ok=True,
                result=result,
                started_at=started_at,
            )
        elif agent == "Documentation":
            target = file_path or class_name or function_name or doc_mode or "repository"
            result = _run_worker(
                "documentation",
                repo_path,
                mode=doc_mode,
                file_path=file_path,
                function_name=function_name,
                class_name=class_name,
                write_to_disk=write_to_disk,
                replace_existing=replace_existing,
            )
            st.session_state.last_documentation = result
            st.session_state.last_doc_target = target
            _record_history(
                agent=agent,
                target=str(target),
                ok=True,
                result=result,
                started_at=started_at,
            )
        else:
            target = file_path or function_name or test_mode or "repository"
            result = _run_worker(
                "testing",
                repo_path,
                mode=test_mode,
                file_path=file_path,
                function_name=function_name,
            )
            st.session_state.last_testing = result
            _record_history(
                agent=agent,
                target=str(target),
                ok=True,
                result=result,
                started_at=started_at,
            )
    except Exception as exc:  # noqa: BLE001 — show agent failures in the UI
        message = f"{agent} failed: {exc}"
        st.session_state.last_error = message
        _record_history(
            agent=agent,
            target=str(target),
            ok=False,
            error=message,
            started_at=started_at,
        )


def _restore_history_entry(entry: Dict[str, Any]) -> None:
    """Load a historical result into the matching result tab."""
    agent = str(entry.get("agent") or "")
    result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
    st.session_state.viewing_history_id = entry.get("id")
    st.session_state.viewing_history_label = format_history_label(entry)
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


def _render_history_sidebar() -> None:
    """Draw the run-history expander."""
    with st.sidebar.expander("Run history", expanded=False):
        history: List[Dict[str, Any]] = list(st.session_state.run_history or [])
        if not history:
            st.caption("No runs yet.")
            return

        newest_first = list(reversed(history))
        for entry in newest_first[:20]:
            label = format_history_label(entry)
            entry_id = str(entry.get("id") or "")
            if st.button(label, key=f"hist_{entry_id}", width="stretch"):
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

    if st.sidebar.button("Load repository", width="stretch"):
        with st.spinner("Preparing repository..."):
            _load_repository(reference)

    if st.session_state.repo_path:
        st.sidebar.success(f"Ready: `{st.session_state.repo_path}`")
    if st.session_state.provider_status:
        st.sidebar.caption(st.session_state.provider_status)

    _render_history_sidebar()

    st.sidebar.divider()
    agent = st.sidebar.radio("Agent", AGENTS, index=0)

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
        )
    elif agent == "Documentation":
        doc_mode = st.sidebar.selectbox("Documentation mode", DOC_MODES, index=2)
        if doc_mode in {"file", "function", "class"}:
            file_path = st.sidebar.text_input("File path", value="")
        if doc_mode == "function":
            function_name = st.sidebar.text_input("Function name", value="")
        if doc_mode == "class":
            class_name = st.sidebar.text_input("Class name", value="")
        write_to_disk = st.sidebar.checkbox("Write documentation to disk", value=False)
        replace_existing = False
        if write_to_disk:
            replace_existing = st.sidebar.checkbox(
                "Replace existing documentation", value=False
            )
    else:
        test_mode = st.sidebar.selectbox("Testing mode", TEST_MODES, index=2)
        if test_mode in {"file", "function"}:
            file_path = st.sidebar.text_input("File path", value="")
        if test_mode == "function":
            function_name = st.sidebar.text_input("Function name", value="")

    run_disabled = not bool(st.session_state.repo_path)
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


def _render_main() -> None:
    """Draw status and the latest report."""
    st.title("Codebase Assistant")
    st.markdown(
        "Load a repository in the sidebar, choose an agent, then browse the report here."
    )

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
    if not has_any:
        st.info("No results yet. Choose an agent and click **Run**.")
        return

    tabs = st.tabs(["Analysis", "Documentation", "Testing"])

    with tabs[0]:
        if st.session_state.last_analysis is not None:
            render_analysis_report(st.session_state.last_analysis)
        else:
            st.caption("No analysis result in this session.")

    with tabs[1]:
        if st.session_state.last_documentation is not None:
            render_documentation_result(
                st.session_state.last_documentation,
                requested_target=st.session_state.last_doc_target,
            )
        else:
            st.caption("No documentation result in this session.")

    with tabs[2]:
        if st.session_state.last_testing is not None:
            render_testing_result(st.session_state.last_testing)
        else:
            st.caption("No testing result in this session.")


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
