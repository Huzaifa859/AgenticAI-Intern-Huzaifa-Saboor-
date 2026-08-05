"""
app/ui_reports.py
=================

Streamlit-friendly renderers for agent result payloads (JSON dicts).

Pure presentation — does not call agents or mutate reports.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Mapping, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st


def _parse_execution_counts(summary: str) -> Dict[str, int]:
    """Parse pytest pass/fail/skip/error counts from a testing summary."""
    text = summary or ""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    match = re.search(
        r"(\d+)\s+passed,\s+(\d+)\s+failed,\s+(\d+)\s+skipped,\s+(\d+)\s+errors",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        counts["passed"] = int(match.group(1))
        counts["failed"] = int(match.group(2))
        counts["skipped"] = int(match.group(3))
        counts["errors"] = int(match.group(4))
    return counts


def _detect_writeback_note(summary: str) -> str:
    """Extract a short write-back note from a documentation summary."""
    text = summary or ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("write-back"):
            return stripped
    if "Write-back" in text:
        match = re.search(r"Write-back[^\n]*", text)
        if match:
            return match.group(0).strip()
    return ""


def _as_dict(value: Any) -> Dict[str, Any]:
    """Normalize pydantic models / plain objects into a dict."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _severity_counts(findings: List[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for finding in findings:
        severity = str(finding.get("severity") or "").lower()
        if severity in counts:
            counts[severity] += 1
    return counts


def _render_stats(items: List[tuple[str, str]]) -> None:
    """Render compact stats without Streamlit Metric widgets."""
    cols = st.columns(len(items))
    for column, (label, value) in zip(cols, items):
        column.markdown(f"**{label}**<br>{value}", unsafe_allow_html=True)


def _render_abstention(abstention: Optional[Mapping[str, Any]]) -> None:
    """Render an abstention block when present."""
    if not abstention:
        return
    st.warning(f"**Abstained:** {abstention.get('reason') or '(no reason)'}")
    steps = list(abstention.get("recommended_next_steps") or [])
    if steps:
        st.markdown("**Recommended next steps**")
        for step in steps:
            st.markdown(f"- {step}")


def render_analysis_report(report: Any) -> None:
    """Render an analysis report dict in the Streamlit main pane."""
    data = _as_dict(report)
    findings = [dict(item) for item in list(data.get("findings") or [])]
    severity = _severity_counts(findings)

    st.subheader("Analysis report")
    _render_stats(
        [
            ("Findings", str(len(findings))),
            ("High", str(severity["high"])),
            ("Medium", str(severity["medium"])),
            ("Low", str(severity["low"])),
        ]
    )

    st.caption(
        f"Duration: {float(data.get('duration_seconds') or 0.0):.1f}s · "
        f"Model used: {'yes' if data.get('model_used') else 'no'} · "
        f"Duplicates removed: {int(data.get('duplicates_removed') or 0)}"
    )

    if data.get("question"):
        st.markdown(f"**Question:** {data['question']}")

    notes = list(data.get("notes") or [])
    if notes:
        with st.expander("Notes", expanded=not findings):
            for note in notes:
                st.markdown(f"- {note}")

    _render_abstention(data.get("abstention"))

    answer = (data.get("answer") or "").strip()
    if answer:
        with st.expander("Model answer", expanded=False):
            st.markdown(answer)

    if not findings:
        st.info("No verified findings.")
        return

    table_rows: List[Dict[str, object]] = []
    for finding in findings:
        table_rows.append(
            {
                "severity": finding.get("severity"),
                "confidence": round(float(finding.get("confidence") or 0.0), 2),
                "file": finding.get("file_path"),
                "lines": f"{finding.get('line_start')}-{finding.get('line_end')}",
                "type": finding.get("bug_type"),
                "method": finding.get("detection_method"),
                "summary": str(finding.get("description") or "")[:120],
            }
        )
    st.dataframe(table_rows, width="stretch", hide_index=True)

    st.markdown("### Finding details")
    for index, finding in enumerate(findings, start=1):
        title = (
            f"{index}. [{finding.get('severity')}] {finding.get('bug_type')} — "
            f"{finding.get('file_path')}:{finding.get('line_start')}-"
            f"{finding.get('line_end')}"
        )
        with st.expander(title, expanded=index == 1):
            st.markdown(finding.get("description") or "(no description)")
            st.caption(
                f"Confidence: {float(finding.get('confidence') or 0.0):.2f} · "
                f"Method: {finding.get('detection_method')} · "
                f"Function: {finding.get('function_name') or '(n/a)'}"
            )
            if finding.get("evidence"):
                st.markdown("**Evidence**")
                st.code(str(finding.get("evidence")), language="python")
            if finding.get("suggested_fix"):
                st.markdown("**Suggested fix**")
                st.markdown(str(finding.get("suggested_fix")))
            meta = dict(finding.get("metadata") or {})
            if meta.get("evidence_relocated"):
                st.caption(
                    "Evidence relocated from "
                    f"{meta.get('original_lines')} → {meta.get('relocated_lines')}"
                )


def render_documentation_result(
    result: Any,
    *,
    requested_target: str = "",
) -> None:
    """Render a documentation result dict in the Streamlit main pane."""
    data = _as_dict(result)
    st.subheader("Documentation result")

    target = (
        requested_target
        or data.get("file_path")
        or data.get("function_name")
        or "(repository)"
    )
    write_note = _detect_writeback_note(str(data.get("summary") or ""))
    grounded = "No" if data.get("abstention") else "Yes"

    st.markdown(
        f"**Target:** `{target}` · **Grounded:** {grounded} · "
        f"**Written:** {write_note or '(not written)'}"
    )

    _render_abstention(data.get("abstention"))

    body = str(data.get("summary") or "").strip()
    if write_note and write_note in body:
        body = body.replace(write_note, "").rstrip()

    if body:
        st.markdown(body)
    else:
        st.info("Empty documentation summary.")

    parameters = list(data.get("parameters") or [])
    if parameters:
        st.markdown("### Parameters")
        st.dataframe(parameters, width="stretch", hide_index=True)

    if data.get("returns"):
        st.markdown("### Returns")
        st.markdown(str(data.get("returns")))

    if data.get("example_usage"):
        st.markdown("### Example")
        st.code(str(data.get("example_usage")), language="python")


def render_testing_result(result: Any) -> None:
    """Render a testing result dict in the Streamlit main pane."""
    data = _as_dict(result)
    st.subheader("Testing result")

    summary = str(data.get("summary") or "")
    counts = _parse_execution_counts(summary)
    generated = dict(data.get("generated_tests") or {})
    names = sorted(generated.keys())
    coverage_pct = float(data.get("coverage_estimate") or 0.0) * 100.0

    _render_stats(
        [
            ("Test files", str(len(names))),
            ("Passed", str(counts["passed"])),
            ("Failed", str(counts["failed"])),
            ("Coverage", f"{coverage_pct:.0f}%"),
        ]
    )

    if counts["skipped"] or counts["errors"]:
        st.caption(
            f"Skipped: {counts['skipped']} · Errors: {counts['errors']}"
        )

    if summary:
        with st.expander("Summary", expanded=True):
            st.markdown(summary)

    _render_abstention(data.get("abstention"))

    if not names:
        st.info("No generated test files.")
        return

    st.markdown("### Generated tests")
    for name in names:
        source = generated.get(name) or ""
        label = os.path.basename(name) or name
        with st.expander(label, expanded=len(names) == 1):
            st.caption(name)
            st.code(str(source).rstrip() or "(empty file)", language="python")
