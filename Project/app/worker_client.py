"""
app/worker_client.py
====================

HTTP client helpers for the long-lived warm worker server.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_WORKER_SERVER_PATH = os.path.join(_APP_DIR, "worker_server.py")

DEFAULT_WORKER_URL = "http://127.0.0.1:8765"
DEFAULT_HEALTH_TIMEOUT_S = 120.0


def worker_base_url() -> str:
    return (os.environ.get("CA_WORKER_URL") or DEFAULT_WORKER_URL).rstrip("/")


def _request_json(
    method: str,
    url: str,
    body: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 5.0,
) -> Tuple[int, Dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
            if not isinstance(payload, dict):
                payload = {"ok": False, "error": "invalid JSON response"}
            return int(resp.status), payload
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"ok": False, "error": raw or str(exc)}
        if not isinstance(payload, dict):
            payload = {"ok": False, "error": str(exc)}
        return int(exc.code), payload


def health(timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    try:
        code, payload = _request_json(
            "GET", f"{worker_base_url()}/health", timeout=timeout
        )
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    if code != 200 or not payload.get("ok"):
        return None
    return payload


def spawn_worker_server(
    *,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    """Start worker_server.py detached; caller waits on /health."""
    port = int(os.environ.get("CA_WORKER_PORT", "8765"))
    host = os.environ.get("CA_WORKER_HOST", "127.0.0.1")
    cmd = [
        sys.executable,
        _WORKER_SERVER_PATH,
        "--host",
        host,
        "--port",
        str(port),
    ]
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    # Detach on Windows so Streamlit exit doesn't always kill it mid-session;
    # Docker entrypoint owns the process instead.
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(
        cmd,
        cwd=_PROJECT_ROOT,
        env=child_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def ensure_worker_server(
    *,
    env: Optional[Dict[str, str]] = None,
    timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
) -> Optional[Dict[str, Any]]:
    """
    Return health payload when a warm worker is ready.

    Spawns ``worker_server.py`` if needed. Returns None on failure.
    """
    current = health()
    if current is not None and current.get("model_loaded"):
        return current
    if current is not None and not current.get("model_loaded"):
        # Server up but still preloading.
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            time.sleep(1.0)
            current = health()
            if current is not None and current.get("model_loaded"):
                return current
        return current

    try:
        spawn_worker_server(env=env)
    except OSError:
        return None

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(1.0)
        current = health()
        if current is not None and current.get("model_loaded"):
            return current
    return health()


def submit_job(body: Dict[str, Any], *, timeout: float = 10.0) -> Dict[str, Any]:
    code, payload = _request_json(
        "POST", f"{worker_base_url()}/jobs", body, timeout=timeout
    )
    if code not in {200, 202}:
        raise RuntimeError(payload.get("error") or f"submit failed ({code})")
    if not payload.get("id"):
        raise RuntimeError("submit response missing job id")
    return payload


def job_status(job_id: str, *, timeout: float = 5.0) -> Dict[str, Any]:
    code, payload = _request_json(
        "GET", f"{worker_base_url()}/jobs/{job_id}", timeout=timeout
    )
    if code != 200:
        raise RuntimeError(payload.get("error") or f"status failed ({code})")
    return payload


def cancel_job(job_id: str, *, timeout: float = 5.0) -> Dict[str, Any]:
    code, payload = _request_json(
        "POST",
        f"{worker_base_url()}/jobs/{job_id}/cancel",
        {},
        timeout=timeout,
    )
    if code != 200:
        raise RuntimeError(payload.get("error") or f"cancel failed ({code})")
    return payload
