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
import html

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
/* ChatGPT-style assistant prose — theme-aware so dark mode stays readable. */
.ca-doc-fallback {
  margin: 0.35rem 0 0.5rem;
  padding: 0.15rem 0.05rem;
  color: var(--text-color, #e8eaed);
  font-size: 0.98rem;
  line-height: 1.7;
  max-width: 52rem;
}
.ca-doc-fallback .ca-doc-msg-body {
  color: inherit;
  font-size: inherit;
  line-height: inherit;
  min-height: 1.5rem;
  /* Let the page grow so the viewport can follow streaming text.
     An inner max-height + overflow trap keeps new tokens off-screen. */
  overflow: visible;
}
.ca-doc-msg-body > :first-child { margin-top: 0; }
.ca-doc-msg-body > :last-child { margin-bottom: 0; }
.ca-doc-msg-body p { margin: 0 0 0.85rem; }
.ca-doc-msg-body h1,
.ca-doc-msg-body h2,
.ca-doc-msg-body h3,
.ca-doc-msg-body h4 {
  color: var(--text-color, #f3f4f6);
  font-weight: 650;
  line-height: 1.3;
  margin: 1.15rem 0 0.5rem;
}
.ca-doc-msg-body h1 { font-size: 1.35rem; }
.ca-doc-msg-body h2 { font-size: 1.18rem; }
.ca-doc-msg-body h3 { font-size: 1.05rem; }
.ca-doc-msg-body ul,
.ca-doc-msg-body ol {
  margin: 0 0 0.85rem;
  padding-left: 1.35rem;
}
.ca-doc-msg-body li { margin: 0.2rem 0; }
.ca-doc-msg-body code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.86em;
  background: var(--secondary-background-color, #1f2937);
  color: var(--text-color, #e5e7eb);
  border-radius: 5px;
  padding: 0.1rem 0.35rem;
}
.ca-doc-msg-body pre {
  margin: 0 0 0.95rem;
  padding: 0.85rem 0.95rem;
  border-radius: 10px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  background: var(--secondary-background-color, #111827);
  overflow-x: auto;
}
.ca-doc-msg-body pre code {
  background: transparent;
  padding: 0;
  color: inherit;
  font-size: 0.84rem;
  line-height: 1.55;
  white-space: pre-wrap;
}
.ca-doc-msg-body strong { color: inherit; font-weight: 650; }
.ca-doc-msg-body table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 0.95rem;
  font-size: 0.92rem;
}
.ca-doc-msg-body th,
.ca-doc-msg-body td {
  border: 1px solid rgba(148, 163, 184, 0.25);
  padding: 0.45rem 0.6rem;
  text-align: left;
  vertical-align: top;
}
.ca-doc-msg-body th {
  background: var(--secondary-background-color, #1f2937);
  font-weight: 650;
}
.ca-doc-placeholder {
  color: var(--text-color, #9ca3af);
  opacity: 0.75;
  margin: 0;
}
.ca-doc-caret {
  display: inline-block;
  width: 0.45rem;
  height: 1.05em;
  margin-left: 1px;
  border-radius: 1px;
  background: var(--text-color, #e5e7eb);
  vertical-align: -0.15em;
  animation: ca-doc-caret 1s steps(1, end) infinite;
}
@keyframes ca-doc-caret {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
</style>
"""

def _ensure_report_styles() -> None:
    """Inject chip / documentation message styles for the current page render."""
    st.markdown(_REPORT_CSS, unsafe_allow_html=True)


def _render_inline_markdown(text: str) -> str:
    """Escape text and apply a small inline markdown subset."""
    raw = text or ""
    if not raw:
        return ""
    pieces: List[str] = []
    cursor = 0
    pattern = re.compile(
        r"(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*]+\*|_[^_]+_)"
    )
    for match in pattern.finditer(raw):
        if match.start() > cursor:
            pieces.append(html.escape(raw[cursor : match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            pieces.append(f"<code>{html.escape(token[1:-1])}</code>")
        elif token.startswith("**") or token.startswith("__"):
            pieces.append(f"<strong>{html.escape(token[2:-2])}</strong>")
        else:
            pieces.append(f"<em>{html.escape(token[1:-1])}</em>")
        cursor = match.end()
    if cursor < len(raw):
        pieces.append(html.escape(raw[cursor:]))
    return "".join(pieces)


def documentation_body_html(text: str, *, live: bool = False) -> str:
    """
    Convert documentation markdown into safe HTML for the shared message UI.

    Supports headings, lists, fenced code, paragraphs, and light inline marks.
    Unclosed fences mid-stream are closed so partial replies still render.
    """
    source = text or ""
    if source.count("```") % 2 == 1:
        source = source + "\n```"
    if not source.strip():
        caret = (
            '<span class="ca-doc-caret" aria-hidden="true"></span>' if live else ""
        )
        return (
            f'<p class="ca-doc-placeholder">Waiting for first token…{caret}</p>'
        )

    blocks: List[str] = []
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    index = 0
    paragraph: List[str] = []
    list_kind = ""
    list_items: List[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        body = " ".join(paragraph).strip()
        if body:
            blocks.append(f"<p>{_render_inline_markdown(body)}</p>")
        paragraph = []

    def flush_list() -> None:
        nonlocal list_kind, list_items
        if not list_items:
            list_kind = ""
            return
        tag = "ol" if list_kind == "ol" else "ul"
        items = "".join(f"<li>{item}</li>" for item in list_items)
        blocks.append(f"<{tag}>{items}</{tag}>")
        list_kind = ""
        list_items = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            flush_list()
            lang = stripped[3:].strip()
            index += 1
            code_lines: List[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            code = html.escape("\n".join(code_lines))
            lang_attr = f' class="language-{html.escape(lang)}"' if lang else ""
            blocks.append(f"<pre><code{lang_attr}>{code}</code></pre>")
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1))
            blocks.append(
                f"<h{level}>{_render_inline_markdown(heading.group(2))}</h{level}>"
            )
            index += 1
            continue

        bullet = re.match(r"^[-*+]\s+(.+)$", stripped)
        numbered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if bullet or numbered:
            flush_paragraph()
            kind = "ol" if numbered else "ul"
            if list_kind and list_kind != kind:
                flush_list()
            list_kind = kind
            item_text = (numbered or bullet).group(1)  # type: ignore[union-attr]
            list_items.append(_render_inline_markdown(item_text))
            index += 1
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            index += 1
            continue

        flush_list()
        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    flush_list()

    body = "".join(blocks) if blocks else f"<p>{_render_inline_markdown(source)}</p>"
    if live:
        # Attach caret to the last text-bearing block without nesting invalid HTML.
        caret = '<span class="ca-doc-caret" aria-hidden="true"></span>'
        for tag in ("</p>", "</li>", "</h4>", "</h3>", "</h2>", "</h1>", "</code></pre>"):
            if body.endswith(tag):
                if tag == "</code></pre>":
                    return body[: -len(tag)] + caret + tag
                return body[: -len(tag)] + caret + tag
        return body + caret
    return body


def _follow_streaming_viewport(*, tick: int = 0) -> None:
    """Keep the newest streamed text visible in Streamlit's main scroll pane.

    Streamlit scrolls ``[data-testid=\"stMain\"]``, not ``window``. ``st.html``
    injects into the parent document (unlike ``components.html`` iframes). A
    single interval watches for ``#ca-doc-stream-end`` and scrolls while it
    exists so fragment remounts / delayed paints still follow the stream.
    """
    # tick changes force Streamlit to accept a new HTML payload periodically.
    st.html(
        f"""
<div id="ca-scroll-tick" data-tick="{int(tick)}" style="display:none" aria-hidden="true"></div>
<script>
(function () {{
  function scrollRoot() {{
    var end = document.getElementById("ca-doc-stream-end");
    if (!end) return false;
    var root =
      document.querySelector('[data-testid="stMain"]') ||
      document.querySelector('[data-testid="stAppViewContainer"]') ||
      document.querySelector("section.main");
    if (root) {{
      root.scrollTop = root.scrollHeight;
      return true;
    }}
    if (typeof end.scrollIntoView === "function") {{
      end.scrollIntoView({{ behavior: "auto", block: "end", inline: "nearest" }});
    }}
    return true;
  }}
  scrollRoot();
  if (!window.__caDocFollowTimer) {{
    window.__caDocFollowTimer = setInterval(function () {{
      if (!scrollRoot()) {{
        clearInterval(window.__caDocFollowTimer);
        window.__caDocFollowTimer = null;
      }}
    }}, 120);
  }}
}})();
</script>
""",
        unsafe_allow_javascript=True,
    )


def render_documentation_message(
    text: str,
    *,
    live: bool = False,
    title: str = "Documentation",
    key: Optional[str] = None,
) -> None:
    """
    Render live or final documentation as one ChatGPT-style assistant message.

    Always paints one accumulated markdown body via ``st.markdown`` (stable
    Streamlit path). Optional HTML component is skipped — path-based custom
    components were mounting blank/zero-height frames in this app.
    """
    del title, key  # Call-site compatibility; chrome stays intentionally minimal.
    _ensure_report_styles()
    body = documentation_body_html(text, live=live)
    # Sentinel sits after the message so scroll helpers can find the bottom.
    sentinel = (
        '<div id="ca-doc-stream-end" aria-hidden="true"></div>' if live else ""
    )
    st.markdown(
        f'<div class="ca-doc-fallback"><div class="ca-doc-msg-body">{body}</div></div>'
        f"{sentinel}",
        unsafe_allow_html=True,
    )
    if live:
        _follow_streaming_viewport(tick=len(text or ""))


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


def _render_ungrounded_candidates(candidates: List[Dict[str, Any]]) -> None:
    """Render findings that failed grounding (never mixed into verified)."""
    st.markdown("### Unverified (failed grounding)")
    st.warning(
        "These candidates were **discarded by grounding** — evidence did not "
        "match the source at the cited lines. They are **not** verified bugs."
    )
    st.caption(f"{len(candidates)} ungrounded candidate(s).")
    for index, item in enumerate(candidates, start=1):
        title = (
            f"{index}. [{item.get('severity') or '?'}] "
            f"{item.get('bug_type') or 'candidate'} — "
            f"{item.get('file_path')}:{item.get('line_start')}-"
            f"{item.get('line_end')}"
        )
        with st.expander(title, expanded=index == 1):
            st.markdown(item.get("description") or "(no description)")
            st.caption(
                f"Method: {item.get('detection_method') or '(n/a)'} · "
                f"Status: {item.get('grounding_status') or '(n/a)'} · "
                f"Match: {item.get('match_type') or '(n/a)'}"
            )
            if item.get("grounding_reason"):
                st.markdown(f"**Why rejected:** {item.get('grounding_reason')}")
            if item.get("found_at_line") is not None:
                st.caption(f"Evidence found near line: {item.get('found_at_line')}")
            if item.get("evidence"):
                st.markdown("**Claimed evidence**")
                st.code(str(item.get("evidence")), language="python")
            if item.get("actual_source"):
                st.markdown("**Actual source at cited range**")
                st.code(str(item.get("actual_source")), language="python")
            if item.get("suggested_fix"):
                st.markdown("**Suggested fix (unverified)**")
                st.markdown(str(item.get("suggested_fix")))


def render_analysis_report(report: Any) -> None:
    """Render an analysis report dict in the Streamlit main pane."""
    data = _as_dict(report)
    findings = [dict(item) for item in list(data.get("findings") or [])]
    candidates = [
        dict(item) for item in list(data.get("ungrounded_candidates") or [])
    ]
    severity = _severity_counts(findings)

    st.subheader("Analysis report")
    _render_stats(
        [
            ("Verified", str(len(findings))),
            ("High", str(severity["high"])),
            ("Medium", str(severity["medium"])),
            ("Low", str(severity["low"])),
            ("Ungrounded", str(len(candidates))),
        ]
    )

    st.caption(
        f"Duration: {float(data.get('duration_seconds') or 0.0):.1f}s · "
        f"Model used: {'yes' if data.get('model_used') else 'no'} · "
        f"Duplicates removed: {int(data.get('duplicates_removed') or 0)} · "
        f"LLM grounded: {int(data.get('llm_grounded_count') or 0)}"
    )

    if "analysis_show_ungrounded" not in st.session_state:
        st.session_state.analysis_show_ungrounded = False
    show_ungrounded = st.checkbox(
        "Show ungrounded candidates",
        key="analysis_show_ungrounded",
        help=(
            "Also show LLM/static findings that failed grounding. "
            "They stay separate from verified findings."
        ),
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

    st.markdown("### Verified findings")
    if not findings:
        st.info("No verified findings.")
    else:
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
        else:
            st.caption(f"Showing {len(filtered)} of {len(findings)} verified finding(s).")
            if not filtered:
                st.warning("No findings match the current severity/search filters.")
            else:
                table_rows: List[Dict[str, object]] = []
                for finding in filtered:
                    table_rows.append(
                        {
                            "severity": finding.get("severity"),
                            "confidence": round(
                                float(finding.get("confidence") or 0.0), 2
                            ),
                            "file": finding.get("file_path"),
                            "lines": (
                                f"{finding.get('line_start')}-"
                                f"{finding.get('line_end')}"
                            ),
                            "type": finding.get("bug_type"),
                            "method": finding.get("detection_method"),
                            "summary": str(finding.get("description") or "")[:120],
                        }
                    )
                st.dataframe(table_rows, width="stretch", hide_index=True)

                st.markdown("### Finding details")
                for index, finding in enumerate(filtered, start=1):
                    title = (
                        f"{index}. [{finding.get('severity')}] "
                        f"{finding.get('bug_type')} — "
                        f"{finding.get('file_path')}:{finding.get('line_start')}-"
                        f"{finding.get('line_end')}"
                    )
                    with st.expander(title, expanded=index == 1):
                        st.markdown(
                            finding.get("description") or "(no description)"
                        )
                        st.caption(
                            f"Confidence: "
                            f"{float(finding.get('confidence') or 0.0):.2f} · "
                            f"Method: {finding.get('detection_method')} · "
                            f"Function: {finding.get('function_name') or '(n/a)'}"
                        )
                        if finding.get("evidence"):
                            st.markdown("**Evidence**")
                            st.code(
                                str(finding.get("evidence")), language="python"
                            )
                        if finding.get("suggested_fix"):
                            st.markdown("**Suggested fix**")
                            st.markdown(str(finding.get("suggested_fix")))
                        meta = dict(finding.get("metadata") or {})
                        if meta.get("evidence_relocated"):
                            st.caption(
                                "Evidence relocated from "
                                f"{meta.get('original_lines')} → "
                                f"{meta.get('relocated_lines')}"
                            )

    if show_ungrounded:
        if candidates:
            _render_ungrounded_candidates(candidates)
        else:
            st.markdown("### Unverified (failed grounding)")
            st.caption("No ungrounded candidates in this run.")
    elif candidates:
        st.caption(
            f"{len(candidates)} ungrounded candidate(s) hidden — "
            "enable **Show ungrounded candidates** above to inspect them."
        )


def render_documentation_result(
    result: Any,
    *,
    requested_target: str = "",
) -> None:
    """Render a documentation result dict in the Streamlit main pane."""
    data = _as_dict(result)
    st.subheader("Documentation result")

    # Show what was requested alongside what the backend actually returned
    # instead of only the pre-run guess: they can legitimately differ (e.g.
    # a targeted request that falls back to a broader scope), and hiding
    # that mismatch made incorrect scoping invisible in the UI.
    requested_display = requested_target or "(repository)"
    returned_target = (
        data.get("function_name") or data.get("file_path") or "repository"
    )
    write_note = _detect_writeback_note(str(data.get("summary") or ""))
    write_path = _extract_write_path(write_note)
    grounded = "No" if data.get("abstention") else "Yes"

    st.markdown(
        f"**Requested:** `{requested_display}` · **Returned:** `{returned_target}` · "
        f"**Grounded:** {grounded}"
    )

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
        render_documentation_message(
            body,
            live=False,
            title="Documentation",
            key="doc_stream_assistant",
        )
    else:
        st.info("Empty documentation summary.")


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
