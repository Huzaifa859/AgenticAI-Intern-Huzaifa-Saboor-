"""
Start the warm worker server, wait until embeddings are loaded, then run Streamlit.

Used as the Docker image CMD. Only Streamlit's :8501 is published; the worker
binds to 127.0.0.1 inside the container.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional


def _worker_url() -> str:
    return (os.environ.get("CA_WORKER_URL") or "http://127.0.0.1:8765").rstrip("/")


def _health() -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"{_worker_url()}/health", timeout=2.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def main() -> int:
    host = os.environ.get("CA_WORKER_HOST", "127.0.0.1")
    port = os.environ.get("CA_WORKER_PORT", "8765")
    wait_s = int(os.environ.get("CA_WORKER_READY_TIMEOUT", "180"))

    print(f"[entrypoint] starting warm worker on {_worker_url()}", flush=True)
    worker = subprocess.Popen(
        [
            sys.executable,
            "app/worker_server.py",
            "--host",
            host,
            "--port",
            str(port),
        ],
    )

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if worker.poll() is not None:
            print(
                "[entrypoint] warm worker exited before ready; "
                "continuing with Streamlit (one-shot fallback)",
                flush=True,
            )
            break
        health = _health()
        if health and health.get("model_loaded"):
            print(f"[entrypoint] warm worker ready (pid {worker.pid})", flush=True)
            break
        time.sleep(1.0)
    else:
        print(
            "[entrypoint] warm worker preload timed out; "
            "Streamlit will retry / fall back on first Run",
            flush=True,
        )

    streamlit_cmd = [
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--server.headless=true",
        "--server.fileWatcherType=none",
    ]
    os.execvp(streamlit_cmd[0], streamlit_cmd)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
