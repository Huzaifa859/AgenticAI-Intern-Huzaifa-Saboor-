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
from typing import Any, Dict, List, Mapping, Optional, Sequence

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

_REPORT_CSS = """
<style>
.ca-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0.35rem 0 0.75rem;
}
.ca-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.22rem 0.65rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
  border: 1px solid transparent;
  font-variant-numeric: tabular-nums;
}
.ca-chip-pass {
  color: #166534;
  background: #dcfce7;
  border-color: rgba(22, 101, 52, 0.18);
}
.ca-chip-fail {
  color: #991b1b;
  background: #fee2e2;
  border-color: rgba(153, 27, 27, 0.18);
}
.ca-chip-skip {
  color: #92400e;
  background: #fef3c7;
  border-color: rgba(146, 64, 14, 0.18);
}
.ca-chip-error {
  color: #9f1239;
  background: #ffe4e6;
  border-color: rgba(159, 18, 57, 0.18);
}
.ca-chip-neutral {
  color: #334155;
  background: #e2e8f0;
  border-color: rgba(51, 65, 85, 0.14);
}
.ca-chip-cov {
  color: #1e3a8a;
  background: #dbeafe;
  border-color: rgba(30, 58, 138, 0.16);
}
</style>
"""

def _ensure_report_styles() -> None:
    """Inject chip styles for the current page render."""
    st.markdown(_REPORT_CSS, unsafe_allow_html=True)


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

    for key, pattern in (
        ("passed", r"(\d+)\s+passed"),
        ("failed", r"(\d+)\s+failed"),
        ("skipped", r"(\d+)\s+skipped"),
        ("errors", r"(\d+)\s+errors?"),
    ):
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            counts[key] = int(found.group(1))
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


def _extract_write_path(write_note: str) -> str:
    """Best-effort path extraction from a write-back note."""
    text = write_note or ""
    # Common shapes: Write-back: path/to/file.py  OR wrote to `path`
    for pattern in (
        r"[`'\"]([^`'\"]+)[`'\"]",
        r"(?:to|into|at|:)\s+([^\s].+)$",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".")
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


def _finding_search_blob(finding: Mapping[str, Any]) -> str:
    """Lowercased haystack for free-text finding search."""
    parts = [
        finding.get("severity"),
        finding.get("bug_type"),
        finding.get("description"),
        finding.get("file_path"),
        finding.get("function_name"),
        finding.get("evidence"),
        finding.get("suggested_fix"),
        finding.get("detection_method"),
    ]
    return " ".join(str(part or "") for part in parts).lower()


def _filter_findings(
    findings: Sequence[Mapping[str, Any]],
    *,
    severities: Sequence[str],
    query: str,
) -> List[Dict[str, Any]]:
    """Filter findings by severity set and free-text query."""
    allowed = {str(item).lower() for item in severities}
    needle = (query or "").strip().lower()
    filtered: List[Dict[str, Any]] = []
    for finding in findings:
        item = dict(finding)
        severity = str(item.get("severity") or "").lower()
        if allowed and severity not in allowed:
            continue
        if needle and needle not in _finding_search_blob(item):
            continue
        filtered.append(item)
    return filtered


def _render_stats(items: List[tuple[str, str]]) -> None:
    """Render compact stats without Streamlit Metric widgets."""
    cols = st.columns(len(items))
    for column, (label, value) in zip(cols, items):
        column.markdown(f"**{label}**<br>{value}", unsafe_allow_html=True)


def _render_status_chips(
    *,
    passed: int,
    failed: int,
    skipped: int = 0,
    errors: int = 0,
    coverage_pct: Optional[float] = None,
    files: Optional[int] = None,
) -> None:
    """Colored pass/fail (and related) status chips."""
    _ensure_report_styles()
    chips: List[str] = []
    if files is not None:
        chips.append(
            f'<span class="ca-chip ca-chip-neutral">Files {int(files)}</span>'
        )
    chips.append(f'<span class="ca-chip ca-chip-pass">Passed {int(passed)}</span>')
    chips.append(f'<span class="ca-chip ca-chip-fail">Failed {int(failed)}</span>')
    if skipped:
        chips.append(
            f'<span class="ca-chip ca-chip-skip">Skipped {int(skipped)}</span>'
        )
    if errors:
        chips.append(
            f'<span class="ca-chip ca-chip-error">Errors {int(errors)}</span>'
        )
    if coverage_pct is not None:
        chips.append(
            f'<span class="ca-chip ca-chip-cov">Coverage {coverage_pct:.0f}%</span>'
        )
    st.markdown(
        f'<div class="ca-chip-row">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


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


def _is_long_log(text: str) -> bool:
    """True when a pytest/summary log should start collapsed."""
    content = text or ""
    return len(content) > 500 or content.count("\n") >= 12


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

    st.markdown("### Filter findings")
    filter_cols = st.columns([2, 3])
    with filter_cols[0]:
        selected_severities = st.multiselect(
            "Severity",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
            key="analysis_severity_filter",
            help="Show only findings at the selected severity levels.",
        )
    with filter_cols[1]:
        search_query = st.text_input(
            "Search",
            value="",
            key="analysis_findings_search",
            placeholder="file, function, bug type, description…",
            help="Case-insensitive match across finding fields.",
        )

    filtered = _filter_findings(
        findings,
        severities=selected_severities or [],
        query=search_query,
    )
    if not selected_severities:
        st.info("Select at least one severity to show findings.")
        return

    st.caption(f"Showing {len(filtered)} of {len(findings)} finding(s).")
    if not filtered:
        st.warning("No findings match the current severity/search filters.")
        return

    table_rows: List[Dict[str, object]] = []
    for finding in filtered:
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
    for index, finding in enumerate(filtered, start=1):
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
    write_path = _extract_write_path(write_note)
    grounded = "No" if data.get("abstention") else "Yes"

    st.markdown(f"**Target:** `{target}` · **Grounded:** {grounded}")

    if write_note:
        if write_path:
            st.success(
                f"Written to disk successfully: `{write_path}`\n\n{write_note}"
            )
        else:
            st.success(f"Written to disk successfully.\n\n{write_note}")
    else:
        st.info("Not written to disk — preview only (enable write-back in the sidebar).")

    _render_abstention(data.get("abstention"))

    body = str(data.get("summary") or "").strip()
    if write_note and write_note in body:
        body = body.replace(write_note, "").rstrip()

    if body:
        st.markdown("### Documentation")
        st.markdown(body)
        st.caption("Use the copy icon on the block below to copy the full text.")
        st.code(body, language="markdown")
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

    _render_status_chips(
        passed=counts["passed"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        errors=counts["errors"],
        coverage_pct=coverage_pct,
        files=len(names),
    )

    if summary:
        expanded = not _is_long_log(summary)
        label = "Pytest log / summary"
        if _is_long_log(summary):
            label += " (collapsed — long output)"
        with st.expander(label, expanded=expanded):
            st.code(summary, language="text")

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
