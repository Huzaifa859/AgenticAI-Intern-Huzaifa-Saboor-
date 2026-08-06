"""
app/ui_history.py
=================

Helpers for capped Streamlit run history (session + local JSONL).
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

HISTORY_LIMIT = 20
RUNTIME_ROOT = os.path.join(
    tempfile.gettempdir(), "codebase_assistant_streamlit"
)
HISTORY_PATH = os.path.join(RUNTIME_ROOT, "ui_run_history.jsonl")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def history_path() -> str:
    """Return the on-disk history file path."""
    return HISTORY_PATH


def load_history(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load the newest ``HISTORY_LIMIT`` runs from JSONL (oldest→newest)."""
    target = path or HISTORY_PATH
    if not os.path.isfile(target):
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with open(target, encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("id"):
                    entries.append(item)
    except OSError:
        return []
    if len(entries) > HISTORY_LIMIT:
        entries = entries[-HISTORY_LIMIT:]
    return entries


def save_history(
    entries: List[Dict[str, Any]],
    path: Optional[str] = None,
) -> None:
    """Rewrite history JSONL with at most ``HISTORY_LIMIT`` newest runs."""
    target = path or HISTORY_PATH
    capped = list(entries or [])[-HISTORY_LIMIT:]
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            for item in capped:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    except OSError:
        pass


def clear_history(path: Optional[str] = None) -> None:
    """Delete the on-disk history file."""
    target = path or HISTORY_PATH
    try:
        if os.path.isfile(target):
            os.remove(target)
    except OSError:
        pass


def make_run_entry(
    *,
    agent: str,
    repo_reference: str,
    target: str,
    ok: bool,
    error: str = "",
    summary: str = "",
    result: Optional[Dict[str, Any]] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one history entry dict."""
    return {
        "id": uuid.uuid4().hex,
        "started_at": started_at or _utc_now_iso(),
        "finished_at": finished_at or _utc_now_iso(),
        "agent": agent,
        "repo_reference": repo_reference or "",
        "target": target or "",
        "ok": bool(ok),
        "error": error or "",
        "summary": summary or "",
        "result": dict(result or {}),
    }


def append_history(
    entries: List[Dict[str, Any]],
    entry: Dict[str, Any],
    path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Append ``entry``, cap to ``HISTORY_LIMIT``, persist, and return the list.

    Returned list is oldest→newest.
    """
    updated = list(entries or [])
    updated.append(entry)
    if len(updated) > HISTORY_LIMIT:
        updated = updated[-HISTORY_LIMIT:]
    save_history(updated, path=path)
    return updated


def summarize_result(agent: str, result: Optional[Dict[str, Any]], error: str = "") -> str:
    """Build a short one-line summary for the history list."""
    if error:
        return (error or "Failed")[:120]
    data = result or {}
    if agent == "Analysis":
        findings = data.get("findings") or []
        abstention = data.get("abstention") or {}
        if abstention:
            return f"Abstained: {abstention.get('reason') or 'no reason'}"[:120]
        return f"{len(findings)} finding(s)"
    if agent == "Documentation":
        abstention = data.get("abstention") or {}
        if abstention:
            return f"Abstained: {abstention.get('reason') or 'no reason'}"[:120]
        summary = (data.get("summary") or "").strip().replace("\n", " ")
        return (summary[:100] + "…") if len(summary) > 100 else (summary or "Documentation ready")
    if agent == "Testing":
        abstention = data.get("abstention") or {}
        if abstention:
            return f"Abstained: {abstention.get('reason') or 'no reason'}"[:120]
        tests = data.get("generated_tests") or {}
        return f"{len(tests)} test file(s)"
    return "Completed"


def format_history_label(entry: Dict[str, Any]) -> str:
    """Sidebar label for one history row."""
    agent = entry.get("agent") or "?"
    repo = entry.get("repo_reference") or "?"
    if len(repo) > 28:
        repo = "…" + repo[-27:]
    stamp = (entry.get("finished_at") or entry.get("started_at") or "")[-8:]
    status = "ok" if entry.get("ok") else "fail"
    return f"{agent} · {repo} · {stamp} · {status}"
