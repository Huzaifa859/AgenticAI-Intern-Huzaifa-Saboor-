"""
app/streamlit_app.py
====================

Lightweight Streamlit UI for the Codebase Assistant.

Run from the Project directory:

    streamlit run app/streamlit_app.py --server.fileWatcherType=none

Or double-click / run:

    run_ui.bat

Agent jobs run in a separate ``worker.py`` process so embedding/LLM
memory pressure cannot kill the Streamlit server.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

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
from ui_reports import (  # noqa: E402
    render_analysis_report,
    render_documentation_result,
    render_testing_result,
)

DOC_MODES = ("readme", "file", "function", "class")
TEST_MODES = ("repository", "file", "function")
AGENTS = ("Analysis", "Documentation", "Testing")


def _init_state() -> None:
    """Ensure session keys exist."""
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
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


def _run_worker(job: str, repo_path: str, **fields: Any) -> Dict[str, Any]:
    """
    Run one agent job in a subprocess and return the JSON payload.

    Raises:
        RuntimeError: When the worker exits without a usable result file.
    """
    fd, out_path = tempfile.mkstemp(prefix="ca_ui_", suffix=".json")
    os.close(fd)
    cmd = [
        sys.executable,
        _WORKER_PATH,
        "--job",
        job,
        "--repo",
        repo_path,
        "--out",
        out_path,
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

    try:
        completed = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if not os.path.isfile(out_path):
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                detail
                or f"Worker exited with code {completed.returncode} and wrote no result."
            )
        with open(out_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    finally:
        try:
            os.remove(out_path)
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

    try:
        if agent == "Analysis":
            st.session_state.last_analysis = _run_worker(
                "analysis",
                repo_path,
                question=question,
            )
        elif agent == "Documentation":
            target = file_path or class_name or function_name or "repository"
            st.session_state.last_documentation = _run_worker(
                "documentation",
                repo_path,
                mode=doc_mode,
                file_path=file_path,
                function_name=function_name,
                class_name=class_name,
                write_to_disk=write_to_disk,
                replace_existing=replace_existing,
            )
            st.session_state.last_doc_target = target
        else:
            st.session_state.last_testing = _run_worker(
                "testing",
                repo_path,
                mode=test_mode,
                file_path=file_path,
                function_name=function_name,
            )
    except Exception as exc:  # noqa: BLE001 — show agent failures in the UI
        st.session_state.last_error = f"{agent} failed: {exc}"


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
        with st.spinner(
            f"Running {agent} in a background worker (can take several minutes)..."
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
    _init_state()
    _render_sidebar()
    _render_main()


if __name__ == "__main__":
    main()
