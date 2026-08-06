"""
app/ui_export.py
================

Build downloadable Markdown / JSON exports for Streamlit result tabs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return {}


def result_to_json(result: Any) -> str:
    """Pretty-print a result payload as JSON text."""
    return json.dumps(_as_dict(result), indent=2, ensure_ascii=False)


def analysis_to_markdown(report: Any) -> str:
    """Render an analysis report as Markdown."""
    data = _as_dict(report)
    findings = list(data.get("findings") or [])
    candidates = list(data.get("ungrounded_candidates") or [])
    lines: List[str] = [
        "# Analysis report",
        "",
        f"- Question: {data.get('question') or '(none)'}",
        f"- Verified findings: {len(findings)}",
        f"- Ungrounded candidates: {len(candidates)}",
        f"- Duration: {float(data.get('duration_seconds') or 0.0):.1f}s",
        f"- Model used: {'yes' if data.get('model_used') else 'no'}",
        "",
    ]
    answer = (data.get("answer") or "").strip()
    if answer:
        lines.extend(["## Model answer", "", answer, ""])
    notes = list(data.get("notes") or [])
    if notes:
        lines.extend(["## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    abstention = data.get("abstention") or {}
    if abstention:
        lines.extend(
            [
                "## Abstained",
                "",
                str(abstention.get("reason") or "(no reason)"),
                "",
            ]
        )
    if findings:
        lines.extend(["## Findings", ""])
        for index, finding in enumerate(findings, start=1):
            lines.append(
                f"### {index}. [{finding.get('severity')}] "
                f"{finding.get('bug_type')} — "
                f"{finding.get('file_path')}:{finding.get('line_start')}-"
                f"{finding.get('line_end')}"
            )
            lines.append("")
            lines.append(str(finding.get("description") or "(no description)"))
            lines.append("")
            if finding.get("evidence"):
                lines.extend(
                    [
                        "**Evidence**",
                        "",
                        "```python",
                        str(finding.get("evidence")),
                        "```",
                        "",
                    ]
                )
            if finding.get("suggested_fix"):
                lines.extend(
                    ["**Suggested fix**", "", str(finding.get("suggested_fix")), ""]
                )
    if candidates:
        lines.extend(
            [
                "## Unverified (failed grounding)",
                "",
                "These were discarded by grounding and are NOT verified bugs.",
                "",
            ]
        )
        for index, item in enumerate(candidates, start=1):
            lines.extend(
                [
                    f"### {index}. [{item.get('severity') or '?'}] "
                    f"{item.get('bug_type') or 'candidate'} — "
                    f"{item.get('file_path')}:{item.get('line_start')}-"
                    f"{item.get('line_end')}",
                    "",
                    str(item.get("description") or "(no description)"),
                    "",
                    f"- Status: {item.get('grounding_status') or '(n/a)'}",
                    f"- Reason: {item.get('grounding_reason') or '(n/a)'}",
                    "",
                ]
            )
            if item.get("evidence"):
                lines.extend(
                    [
                        "**Claimed evidence**",
                        "",
                        "```python",
                        str(item.get("evidence")),
                        "```",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def documentation_to_markdown(
    result: Any, *, requested_target: str = ""
) -> str:
    """Render a documentation result as Markdown."""
    data = _as_dict(result)
    target = (
        requested_target
        or data.get("file_path")
        or data.get("function_name")
        or "(repository)"
    )
    lines: List[str] = [
        "# Documentation result",
        "",
        f"- Target: `{target}`",
        "",
        str(data.get("summary") or "").strip() or "_Empty documentation summary._",
        "",
    ]
    parameters = list(data.get("parameters") or [])
    if parameters:
        lines.extend(["## Parameters", ""])
        for item in parameters:
            if isinstance(item, Mapping):
                name = item.get("name") or "?"
                desc = item.get("description") or ""
                lines.append(f"- `{name}`: {desc}")
            else:
                lines.append(f"- {item}")
        lines.append("")
    if data.get("returns"):
        lines.extend(["## Returns", "", str(data.get("returns")), ""])
    if data.get("example_usage"):
        lines.extend(
            [
                "## Example",
                "",
                "```python",
                str(data.get("example_usage")),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def testing_to_markdown(result: Any) -> str:
    """Render a testing result as Markdown."""
    data = _as_dict(result)
    generated = dict(data.get("generated_tests") or {})
    lines: List[str] = [
        "# Testing result",
        "",
        f"- Test files: {len(generated)}",
        f"- Coverage estimate: {float(data.get('coverage_estimate') or 0.0) * 100.0:.0f}%",
        "",
        "## Summary",
        "",
        str(data.get("summary") or "").strip() or "_No summary._",
        "",
    ]
    if generated:
        lines.extend(["## Generated tests", ""])
        for name in sorted(generated.keys()):
            lines.extend(
                [
                    f"### `{name}`",
                    "",
                    "```python",
                    str(generated.get(name) or "").rstrip() or "(empty file)",
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def markdown_for_agent(
    agent: str,
    result: Any,
    *,
    doc_target: str = "",
) -> str:
    """Dispatch Markdown export by agent name."""
    if agent == "Analysis":
        return analysis_to_markdown(result)
    if agent == "Documentation":
        return documentation_to_markdown(result, requested_target=doc_target)
    if agent == "Testing":
        return testing_to_markdown(result)
    return result_to_json(result)


def export_filename(agent: str, ext: str) -> str:
    """Stable download filename for an agent export."""
    slug = {
        "Analysis": "analysis",
        "Documentation": "documentation",
        "Testing": "testing",
    }.get(agent, "report")
    return f"codebase_assistant_{slug}.{ext}"
