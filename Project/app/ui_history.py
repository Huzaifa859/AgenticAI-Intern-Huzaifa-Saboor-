"""
app/ui_history.py
=================

Helpers for capped Streamlit run history (session + local JSONL).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ui_paths import run_history_path

HISTORY_LIMIT = 20


def _local_now_iso() -> str:
    """Current local device time with timezone offset (no microseconds)."""
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def history_path() -> str:
    """Return the on-disk history file path (respects RUN_HISTORY_PATH)."""
    return run_history_path()


def parse_history_time(value: Any) -> Optional[datetime]:
    """Parse a stored ISO timestamp into an aware datetime when possible."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        # Legacy naive UTC stamps from earlier builds.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def format_local_time(value: Any) -> str:
    """
    Format a stored timestamp in the user's local device timezone.

    Example: ``Aug 06, 2026 05:43:12 AM``
    """
    dt = parse_history_time(value)
    if dt is None:
        return "unknown time"
    local = dt.astimezone()
    return local.strftime("%b %d, %Y %I:%M:%S %p")


def load_history(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load the newest ``HISTORY_LIMIT`` runs from JSONL (oldest→newest)."""
    target = path or history_path()
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
    target = path or history_path()
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
    target = path or history_path()
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
    """Build one history entry dict (timestamps in local device time)."""
    return {
        "id": uuid.uuid4().hex,
        "started_at": started_at or _local_now_iso(),
        "finished_at": finished_at or _local_now_iso(),
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
    """Compact one-line label (agent · local time · status)."""
    agent = entry.get("agent") or "?"
    stamp = format_local_time(entry.get("finished_at") or entry.get("started_at"))
    status = "ok" if entry.get("ok") else "fail"
    return f"{agent} · {stamp} · {status}"


def format_history_summary(entry: Dict[str, Any]) -> str:
    """Secondary history line with result summary."""
    summary = str(entry.get("summary") or "").strip()
    if summary:
        return summary[:140]
    repo = str(entry.get("repo_reference") or "").strip()
    target = str(entry.get("target") or "").strip()
    bits = [bit for bit in (repo, target) if bit]
    return " · ".join(bits) if bits else "No summary"
