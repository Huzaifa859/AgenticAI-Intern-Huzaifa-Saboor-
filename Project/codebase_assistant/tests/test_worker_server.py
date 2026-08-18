"""
test_worker_server.py
=====================

Unit tests for the warm worker HTTP API. execute_job and model preload are
mocked so CI stays offline and fast.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _PROJECT_ROOT / "app"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import worker_server  # noqa: E402
from worker import execute_job  # noqa: E402


def _http_json(method: str, url: str, body: Dict[str, Any] | None = None) -> tuple[int, Dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            raw = resp.read().decode("utf-8")
            return int(resp.status), json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"error": raw}
        return int(exc.code), payload


@pytest.fixture()
def warm_server(tmp_path: Path):
    """Start WorkerHandler on an ephemeral port with a fake warm Supervisor."""
    state = worker_server.WorkerState()
    state.supervisor = MagicMock(name="Supervisor")
    state.model_loaded = True

    with patch.object(worker_server, "STATE", state):
        server = ThreadingHTTPServer(("127.0.0.1", 0), worker_server.WorkerHandler)
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        try:
            yield {"base": base, "state": state, "tmp": tmp_path}
        finally:
            server.shutdown()
            server.server_close()


def test_health_reports_model_loaded(warm_server) -> None:
    code, payload = _http_json("GET", f"{warm_server['base']}/health")
    assert code == 200
    assert payload["ok"] is True
    assert payload["model_loaded"] is True
    assert payload["busy"] is False
    assert payload["pid"] > 0


def test_submit_job_runs_mocked_execute_job(warm_server) -> None:
    out = warm_server["tmp"] / "result.json"
    progress = warm_server["tmp"] / "progress.ndjson"
    out.write_text("{}", encoding="utf-8")
    progress.write_text("", encoding="utf-8")

    fake_payload = {
        "ok": True,
        "job": "analysis",
        "result": {"findings": []},
    }

    def _fake_execute_job(**kwargs: Any) -> Dict[str, Any]:
        out.write_text(json.dumps(fake_payload), encoding="utf-8")
        return fake_payload

    with patch("worker.execute_job", side_effect=_fake_execute_job) as mock_exec:
        code, submitted = _http_json(
            "POST",
            f"{warm_server['base']}/jobs",
            {
                "job": "analysis",
                "repo": str(warm_server["tmp"] / "repo"),
                "out": str(out),
                "progress": str(progress),
                "question": "Find bugs",
            },
        )
        assert code == 202
        job_id = submitted["id"]
        assert job_id
        assert submitted["status"] == "queued"

        deadline = time.monotonic() + 5.0
        status = ""
        while time.monotonic() < deadline:
            _, status_payload = _http_json(
                "GET", f"{warm_server['base']}/jobs/{job_id}"
            )
            status = str(status_payload.get("status") or "")
            if status in {"done", "error", "cancelled"}:
                break
            time.sleep(0.05)

        assert status == "done"
        mock_exec.assert_called_once()
        assert mock_exec.call_args.kwargs["job"] == "analysis"
        assert mock_exec.call_args.kwargs["supervisor"] is warm_server["state"].supervisor


def test_submit_rejects_when_busy(warm_server) -> None:
    warm_server["state"].busy = True
    code, payload = _http_json(
        "POST",
        f"{warm_server['base']}/jobs",
        {
            "job": "analysis",
            "repo": "/tmp/repo",
            "out": "/tmp/out.json",
        },
    )
    assert code == 409
    assert "busy" in str(payload.get("error") or "").lower()


def test_execute_job_returns_expected_shape(tmp_path: Path) -> None:
    """Thin offline check that execute_job writes the shared result payload."""
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "out.json"

    fake_report = MagicMock()
    with patch("service.build_supervisor") as mock_build, patch(
        "service.run_analysis", return_value=fake_report
    ), patch("worker._analysis_to_dict", return_value={"findings": []}), patch(
        "worker._attach_tracer_progress", return_value=lambda: None
    ):
        mock_build.return_value = MagicMock()
        payload = execute_job(
            job="analysis",
            repo=str(repo),
            out=str(out),
            question="Find bugs",
        )

    assert payload["ok"] is True
    assert payload["job"] == "analysis"
    assert "result" in payload
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["ok"] is True
    assert saved["job"] == "analysis"
