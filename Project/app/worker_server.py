"""
app/worker_server.py
====================

Long-lived localhost HTTP worker that keeps Supervisor + embeddings warm
across Streamlit jobs.

    python app/worker_server.py
    # listens on http://127.0.0.1:8765

Endpoints:
  GET  /health
  POST /jobs
  GET  /jobs/{id}
  POST /jobs/{id}/cancel
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from ui_paths import chroma_persist_dir, memory_store_path, streamlit_data_dir  # noqa: E402

_RUNTIME_ROOT = streamlit_data_dir()
os.makedirs(_RUNTIME_ROOT, exist_ok=True)
os.environ.setdefault("CHROMA_PERSIST_DIR", chroma_persist_dir())
os.environ.setdefault("MEMORY_STORE_PATH", memory_store_path())
os.makedirs(os.environ["CHROMA_PERSIST_DIR"], exist_ok=True)
os.makedirs(os.environ["MEMORY_STORE_PATH"], exist_ok=True)

logger = logging.getLogger("worker_server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class WorkerState:
    """Process-wide warm Supervisor and single-flight job tracking."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.supervisor: Any = None
        self.model_loaded = False
        self.busy = False
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}

    def preload(self) -> None:
        """Build Supervisor once and load the embedding model into memory."""
        from codebase_assistant.config import Config
        from codebase_assistant.rag.embeddings import EmbeddingGenerator
        from service import build_supervisor

        logger.info("Building Supervisor...")
        self.supervisor = build_supervisor()
        config = getattr(self.supervisor, "config", None) or Config.load()
        logger.info(
            "Preloading embedding model %r...",
            getattr(config, "embedding_model_name", "embedding"),
        )
        started = time.perf_counter()
        EmbeddingGenerator(config=config).load_model()
        self.model_loaded = True
        logger.info(
            "Embedding model ready in %.1fs",
            time.perf_counter() - started,
        )

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "busy": self.busy,
            "model_loaded": self.model_loaded,
            "pid": os.getpid(),
            "jobs_tracked": len(self.jobs),
        }

    def submit(self, body: Dict[str, Any]) -> Dict[str, Any]:
        job = str(body.get("job") or "").strip()
        repo = str(body.get("repo") or "").strip()
        out = str(body.get("out") or "").strip()
        if job not in {"analysis", "documentation", "testing"}:
            raise ValueError("job must be analysis|documentation|testing")
        if not repo or not out:
            raise ValueError("repo and out are required")

        with self.lock:
            if self.busy:
                raise RuntimeError("worker is busy")
            if self.supervisor is None:
                raise RuntimeError("worker not ready")
            job_id = uuid.uuid4().hex
            cancel_event = threading.Event()
            record = {
                "id": job_id,
                "status": "queued",
                "job": job,
                "repo": repo,
                "out": out,
                "progress": str(body.get("progress") or ""),
                "error": "",
                "created_at": time.time(),
                "finished_at": None,
            }
            self.jobs[job_id] = record
            self._cancel_flags[job_id] = cancel_event
            self.busy = True

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, body, cancel_event),
            name=f"ca-job-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return {"id": job_id, "status": "queued"}

    def _run_job(
        self,
        job_id: str,
        body: Dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        from worker import execute_job

        record = self.jobs[job_id]
        record["status"] = "running"
        try:
            payload = execute_job(
                job=str(body.get("job") or ""),
                repo=str(body.get("repo") or ""),
                out=str(body.get("out") or ""),
                progress_path=str(body.get("progress") or ""),
                question=str(
                    body.get("question") or "Find bugs and potential issues"
                ),
                mode=str(body.get("mode") or ""),
                file_path=str(body.get("file") or body.get("file_path") or ""),
                function_name=str(
                    body.get("function") or body.get("function_name") or ""
                ),
                class_name=str(
                    body.get("class_name") or body.get("class-name") or ""
                ),
                write_to_disk=bool(body.get("write_to_disk")),
                replace_existing=bool(body.get("replace_existing")),
                supervisor=self.supervisor,
                should_cancel=cancel_event.is_set,
            )
            if payload.get("cancelled") or cancel_event.is_set():
                record["status"] = "cancelled"
                record["error"] = str(payload.get("error") or "cancelled")
            elif payload.get("ok"):
                record["status"] = "done"
            else:
                record["status"] = "error"
                record["error"] = str(payload.get("error") or "job failed")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            record["status"] = "error"
            record["error"] = str(exc)
        finally:
            record["finished_at"] = time.time()
            with self.lock:
                self.busy = False

    def job_status(self, job_id: str) -> Dict[str, Any]:
        record = self.jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return {
            "id": record["id"],
            "status": record["status"],
            "job": record["job"],
            "error": record.get("error") or "",
            "out": record.get("out") or "",
            "progress": record.get("progress") or "",
        }

    def cancel(self, job_id: str) -> Dict[str, Any]:
        record = self.jobs.get(job_id)
        if record is None:
            raise KeyError(job_id)
        event = self._cancel_flags.get(job_id)
        if event is not None:
            event.set()
        if record["status"] in {"queued", "running"}:
            record["status"] = "cancelled"
            record["error"] = record.get("error") or "cancelled by user"
        return self.job_status(job_id)


STATE = WorkerState()


class WorkerHandler(BaseHTTPRequestHandler):
    """Minimal JSON HTTP API for the warm worker."""

    server_version = "CodebaseAssistantWorker/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/health":
            self._send_json(200, STATE.health())
            return
        if path.startswith("/jobs/"):
            job_id = path.split("/", 2)[-1]
            try:
                self._send_json(200, STATE.job_status(job_id))
            except KeyError:
                self._send_json(404, {"ok": False, "error": "unknown job"})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return

        if path == "/jobs":
            try:
                result = STATE.submit(body)
                self._send_json(202, result)
            except RuntimeError as exc:
                self._send_json(409, {"ok": False, "error": str(exc)})
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                logger.exception("submit failed")
                self._send_json(500, {"ok": False, "error": str(exc)})
            return

        if path.startswith("/jobs/") and path.endswith("/cancel"):
            parts = path.strip("/").split("/")
            # jobs/{id}/cancel
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                job_id = parts[1]
                try:
                    self._send_json(200, STATE.cancel(job_id))
                except KeyError:
                    self._send_json(404, {"ok": False, "error": "unknown job"})
                return

        self._send_json(404, {"ok": False, "error": "not found"})


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [worker_server] %(message)s",
    )
    STATE.preload()
    server = ThreadingHTTPServer((host, port), WorkerHandler)
    logger.info("Warm worker listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        server.server_close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Warm Streamlit worker server")
    parser.add_argument(
        "--host",
        default=os.environ.get("CA_WORKER_HOST", DEFAULT_HOST),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CA_WORKER_PORT", str(DEFAULT_PORT))),
    )
    args = parser.parse_args(argv)
    run_server(host=args.host, port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
