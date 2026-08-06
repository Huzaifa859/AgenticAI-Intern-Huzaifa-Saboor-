"""
report_formatter.py
===================

Terminal-friendly formatting helpers for CodeAnalysisReport output.

Pure presentation -- reads a report and returns strings. Does not
modify the report or any pipeline data.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from codebase_assistant.agents.code_analysis_agent import CodeAnalysisReport
from codebase_assistant.schemas.schemas import BugReport, DocumentationResult, TestingResult

# Avoid pytest collecting TestingResult when this module is imported in tests.
TestingResult.__test__ = False

BANNER_WIDTH = 65
SEVERITY_ORDER: Tuple[str, ...] = ("high", "medium", "low")
DEFAULT_EVIDENCE_WIDTH = 72
DEFAULT_TEXT_WIDTH = 78

# ANSI styles. Disabled automatically when color is off.
_STYLES: Dict[str, str] = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}

_SEVERITY_STYLE: Dict[str, str] = {
    "high": "red",
    "medium": "yellow",
    "low": "green",
}


def terminal_width(default: int = DEFAULT_TEXT_WIDTH) -> int:
    """
    Best-effort terminal width for wrapping and tables.

    Args:
        default: Width to use when the terminal size is unknown.

    Returns:
        A sensible column width in characters.
    """
    try:
        columns = shutil.get_terminal_size(fallback=(default, 24)).columns
    except OSError:
        columns = default
    return max(40, min(columns, 120))


def color_enabled(explicit: Optional[bool] = None) -> bool:
    """
    Decide whether ANSI color should be used.

    Color is off when explicitly disabled, when stdout is not a TTY,
    or when NO_COLOR is set in the environment.

    Args:
        explicit: Force color on (True) or off (False). None means
            auto-detect.

    Returns:
        True if color escape codes may be emitted.
    """
    if explicit is False:
        return False
    if explicit is True:
        return True
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def style(text: str, name: str, *, use_color: bool) -> str:
    """
    Apply a named ANSI style when color is enabled.

    Args:
        text: Text to style.
        name: Style key from `_STYLES`.
        use_color: Whether to emit escape codes.

    Returns:
        Styled text, or plain text when color is off.
    """
    if not use_color or name not in _STYLES:
        return text
    return f"{_STYLES[name]}{text}{_STYLES['reset']}"


def severity_label(severity: str, *, use_color: bool) -> str:
    """
    Render a severity name for section headers.

    Args:
        severity: "high", "medium", or "low".
        use_color: Whether to colorize the label.

    Returns:
        An upper-case severity label.
    """
    label = severity.upper()
    tone = _SEVERITY_STYLE.get(severity.lower(), "white")
    return style(label, tone, use_color=use_color)


def banner(title: str, width: int = BANNER_WIDTH) -> str:
    """
    Render a section banner.

    Args:
        title: Banner title.
        width: Total banner width.

    Returns:
        A boxed banner string.
    """
    line = "=" * width
    return f"{line}\n{title}\n{line}"


def wrap_block(
    text: str,
    width: int,
    *,
    initial_prefix: str = "",
    subsequent_prefix: str = "",
) -> str:
    """
    Wrap prose or code for terminal display.

    Blank lines in the input are preserved. Each non-empty line is
    wrapped independently so indented evidence stays readable.

    Args:
        text: Text to wrap.
        width: Maximum line width including prefixes.
        initial_prefix: Prefix for the first line of each paragraph.
        subsequent_prefix: Prefix for continuation lines.

    Returns:
        The wrapped text.
    """
    if not text:
        return initial_prefix.rstrip()

    usable = max(20, width - len(subsequent_prefix))
    lines: List[str] = []

    for raw in text.split("\n"):
        if not raw.strip():
            lines.append("")
            continue

        wrapped = textwrap.fill(
            raw,
            width=usable,
            initial_indent=initial_prefix,
            subsequent_indent=subsequent_prefix,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.append(wrapped)

    return "\n".join(lines)


def align_row(label: str, value: str, label_width: int = 24) -> str:
    """
    Align a label/value pair in two columns.

    Args:
        label: Left column text.
        value: Right column text.
        label_width: Width reserved for the label.

    Returns:
        A single aligned line.
    """
    return f"{label:<{label_width}}{value}"


def format_table(
    rows: Sequence[Tuple[str, str]],
    *,
    label_width: int = 28,
    divider: str = "-",
) -> str:
    """
    Render a simple two-column summary table.

    Args:
        rows: (label, value) pairs.
        label_width: Width of the label column.
        divider: Character repeated under the header.

    Returns:
        The rendered table.
    """
    if not rows:
        return ""

    header = align_row("Metric", "Value", label_width=label_width)
    rule = align_row(divider * label_width, divider * 12, label_width=label_width)
    body = [align_row(label, value, label_width=label_width) for label, value in rows]
    return "\n".join([header, rule, *body])


def count_rejected_hallucinations(report: CodeAnalysisReport) -> int:
    """
    Count grounding rejections from model-proposed findings.

    Args:
        report: Completed analysis report.

    Returns:
        Number of rejected LLM findings.
    """
    return sum(
        1
        for result in report.rejected
        if result.report is not None and result.report.detection_method == "llm"
    )


def group_findings_by_severity(
    findings: Sequence[BugReport],
) -> Dict[str, List[BugReport]]:
    """
    Group findings by severity, highest first.

    Args:
        findings: Verified findings to group.

    Returns:
        A mapping of severity to findings sorted by file and line.
    """
    grouped: Dict[str, List[BugReport]] = {name: [] for name in SEVERITY_ORDER}
    other: List[BugReport] = []

    for finding in findings:
        key = finding.severity.lower()
        if key in grouped:
            grouped[key].append(finding)
        else:
            other.append(finding)

    for items in grouped.values():
        items.sort(key=lambda f: (f.file_path, f.line_start, f.bug_type))

    if other:
        grouped["other"] = sorted(
            other, key=lambda f: (f.file_path, f.line_start, f.bug_type)
        )

    return grouped


def format_line_range(finding: BugReport) -> str:
    """
    Format a finding's line range.

    Args:
        finding: The finding to format.

    Returns:
        A line number or inclusive range string.
    """
    if finding.line_start == finding.line_end:
        return str(finding.line_start)
    return f"{finding.line_start}-{finding.line_end}"


def format_finding(
    finding: BugReport,
    *,
    index: int,
    width: int,
    use_color: bool,
) -> str:
    """
    Format one verified finding.

    Args:
        finding: The finding to render.
        index: One-based index within its severity group.
        width: Terminal width for wrapping.
        use_color: Whether to use ANSI color.

    Returns:
        A multi-line finding block.
    """
    indent = "  "
    label_width = 13
    lines = finding.line_start if finding.line_start == finding.line_end else (
        f"{finding.line_start}-{finding.line_end}"
    )

    header = (
        f"{indent}[{index}] "
        f"{style(finding.bug_type, 'bold', use_color=use_color)}  "
        f"{finding.file_path}:{lines}  "
        f"{finding.detection_method}  "
        f"conf={finding.confidence:.2f}"
    )

    fields = [
        ("File", finding.file_path),
        ("Lines", lines),
        ("Type", finding.bug_type),
        ("Method", finding.detection_method),
        ("Confidence", f"{finding.confidence:.2f}"),
        ("Function", finding.function_name),
    ]

    body: List[str] = [header, ""]
    for label, value in fields:
        body.append(f"{indent}{label + ':':<{label_width}}{value}")

    body.append(f"{indent}Description:")
    body.append(
        wrap_block(
            finding.description,
            width,
            initial_prefix=indent + "  ",
            subsequent_prefix=indent + "  ",
        )
    )
    body.append(f"{indent}Evidence:")
    body.append(
        wrap_block(
            finding.evidence,
            width,
            initial_prefix=indent + "  | ",
            subsequent_prefix=indent + "  | ",
        )
    )

    if finding.suggested_fix:
        body.append(f"{indent}Suggested fix:")
        body.append(
            wrap_block(
                finding.suggested_fix,
                width,
                initial_prefix=indent + "  ",
                subsequent_prefix=indent + "  ",
            )
        )

    return "\n".join(body)


def format_findings_section(
    report: CodeAnalysisReport,
    *,
    width: int,
    use_color: bool,
) -> str:
    """
    Format all findings grouped by severity.

    Empty severity groups are omitted.

    Args:
        report: Completed analysis report.
        width: Terminal width for wrapping.
        use_color: Whether to use ANSI color.

    Returns:
        The findings section, or a no-findings message.
    """
    if not report.findings:
        return "\nNo verified findings."

    grouped = group_findings_by_severity(report.findings)
    sections: List[str] = ["", "Findings", "-" * min(width, BANNER_WIDTH)]

    order = list(SEVERITY_ORDER) + (["other"] if "other" in grouped else [])

    for severity in order:
        items = grouped.get(severity, [])
        if not items:
            continue

        title = severity_label(severity, use_color=use_color)
        count = len(items)
        noun = "finding" if count == 1 else "findings"
        sections.append("")
        sections.append(f"{title}  ({count} {noun})")
        sections.append("-" * min(width, BANNER_WIDTH))

        for index, finding in enumerate(items, start=1):
            sections.append(
                format_finding(
                    finding,
                    index=index,
                    width=width,
                    use_color=use_color,
                )
            )

    return "\n".join(sections)


def format_header(report: CodeAnalysisReport, rejected_llm: int) -> str:
    """
    Format the report header block.

    Args:
        report: Completed analysis report.
        rejected_llm: Rejected hallucination count.

    Returns:
        The header section.
    """
    rows = [
        ("Repository", report.repository_path),
        ("Duration", f"{report.duration_seconds:.2f}s"),
        ("Static findings", str(len(report.static_findings))),
        ("LLM findings", str(len(report.llm_findings))),
        ("Rejected hallucinations", str(rejected_llm)),
    ]

    parts = [banner("Code Analysis Report"), ""]
    parts.extend(align_row(label, value) for label, value in rows)
    return "\n".join(parts)


def format_notes(report: CodeAnalysisReport) -> str:
    """
    Format optional notes and the model answer.

    Args:
        report: Completed analysis report.

    Returns:
        Notes section, or an empty string.
    """
    parts: List[str] = []

    if report.answer:
        parts.extend(["", "Model answer:", wrap_block(report.answer, terminal_width())])

    if report.notes:
        parts.append("")
        parts.append("Notes:")
        for note in report.notes:
            parts.append(f"  - {note}")

    return "\n".join(parts)


def format_summary_table(
    report: CodeAnalysisReport,
    rejected_llm: int,
) -> str:
    """
    Format the summary statistics table.

    Args:
        report: Completed analysis report.
        rejected_llm: Rejected hallucination count.

    Returns:
        The summary section.
    """
    rows: List[Tuple[str, str]] = [
        ("Total verified findings", str(len(report.findings))),
        ("  static", str(len(report.static_findings))),
        ("  llm", str(len(report.llm_findings))),
        ("Rejected (all)", str(len(report.rejected))),
        ("Rejected (hallucinated)", str(rejected_llm)),
        ("Duplicates merged", str(report.duplicates_removed)),
        ("Model used", str(report.model_used)),
    ]

    if report.static_report is not None:
        rows.extend(
            [
                ("Files analyzed", str(report.static_report.files_analyzed)),
                ("Files skipped", str(len(report.static_report.skipped))),
                ("Files failed", str(len(report.static_report.failed))),
            ]
        )

    if report.index_update is not None:
        rows.append(("Index update", report.index_update.summary()))

    rows.append(("Context chunks", str(len(report.context))))

    by_severity = report.by_severity()
    for severity, count in by_severity.items():
        rows.append((f"Severity: {severity}", str(count)))

    by_type: Dict[str, int] = {}
    for finding in report.findings:
        by_type[finding.bug_type] = by_type.get(finding.bug_type, 0) + 1
    for bug_type, count in sorted(by_type.items(), key=lambda item: -item[1]):
        rows.append((f"Type: {bug_type}", str(count)))

    body = format_table(rows)
    return f"\n{banner('Summary Statistics')}\n\n{body}"


def format_ungrounded_section(report: CodeAnalysisReport) -> str:
    """
    Format rejected grounding candidates for terminal display.

    These are never mixed into verified findings.
    """
    if not report.rejected:
        return ""
    lines = [
        banner("Unverified (failed grounding)"),
        "",
        "These candidates were discarded by grounding and are NOT verified bugs.",
        "",
    ]
    for index, result in enumerate(report.rejected, start=1):
        nested = result.report
        bug_type = getattr(nested, "bug_type", None) or "candidate"
        severity = getattr(nested, "severity", None) or "?"
        method = getattr(nested, "detection_method", None) or "?"
        description = getattr(nested, "description", None) or "(no description)"
        status = getattr(result.status, "value", result.status)
        lines.extend(
            [
                f"{index}. [{severity}] {bug_type} — "
                f"{result.file_path}:{result.line_start}-{result.line_end}",
                f"   Method: {method} · Status: {status}",
                f"   Why: {result.reason}",
                f"   {description}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def format_report(
    report: CodeAnalysisReport,
    *,
    width: Optional[int] = None,
    color: Optional[bool] = None,
) -> str:
    """
    Format a complete CodeAnalysisReport for terminal output.

    Args:
        report: Completed analysis report.
        width: Optional terminal width override.
        color: Optional color override. None means auto-detect.

    Returns:
        The full formatted report as one string.
    """
    term_width = width or terminal_width()
    use_color = color_enabled(color)
    rejected_llm = count_rejected_hallucinations(report)

    sections = [
        format_header(report, rejected_llm),
        format_notes(report),
        format_findings_section(report, width=term_width, use_color=use_color),
        format_summary_table(report, rejected_llm),
    ]
    show_ungrounded = False
    try:
        from codebase_assistant.config import Config

        show_ungrounded = bool(Config.load().analysis_show_ungrounded)
    except Exception:
        show_ungrounded = False
    if show_ungrounded:
        ungrounded = format_ungrounded_section(report)
        if ungrounded:
            sections.append(ungrounded)
    return "\n".join(part for part in sections if part)


def print_report(
    report: CodeAnalysisReport,
    *,
    width: Optional[int] = None,
    color: Optional[bool] = None,
) -> None:
    """
    Print a formatted CodeAnalysisReport to stdout.

    Args:
        report: Completed analysis report.
        width: Optional terminal width override.
        color: Optional color override. None means auto-detect.
    """
    print(format_report(report, width=width, color=color))


def format_documentation_result(result: DocumentationResult) -> str:
    """
    Format a DocumentationResult for terminal display.

    Args:
        result: Documentation agent output.

    Returns:
        A readable multi-section string.
    """
    lines = [
        "=" * BANNER_WIDTH,
        "Documentation Result",
        "=" * BANNER_WIDTH,
        "",
        "Summary",
        "-" * BANNER_WIDTH,
        (result.summary or "(empty)").strip() or "(empty)",
        "",
        "Function / Module",
        "-" * BANNER_WIDTH,
        f"File:      {result.file_path or '(none)'}",
        f"Name:      {result.function_name or '(none)'}",
        "",
        "Parameters",
        "-" * BANNER_WIDTH,
    ]

    if result.parameters:
        for index, param in enumerate(result.parameters, start=1):
            if not isinstance(param, dict):
                lines.append(f"  [{index}] {param}")
                continue
            name = str(param.get("name") or "")
            ptype = str(param.get("type") or "")
            description = str(param.get("description") or "")
            label = name or f"param_{index}"
            type_part = f" ({ptype})" if ptype else ""
            desc_part = f": {description}" if description else ""
            lines.append(f"  [{index}] {label}{type_part}{desc_part}")
    else:
        lines.append("  (none)")

    lines.extend(
        [
            "",
            "Returns",
            "-" * BANNER_WIDTH,
            (result.returns or "(none)").strip() or "(none)",
            "",
            "Example usage",
            "-" * BANNER_WIDTH,
            (result.example_usage or "(none)").strip() or "(none)",
            "",
        ]
    )
    return "\n".join(lines)


def print_documentation_result(result: DocumentationResult) -> None:
    """Print a formatted DocumentationResult to stdout."""
    print(format_documentation_result(result))


def format_testing_result(
    result: TestingResult, *, include_source: bool = False
) -> str:
    """
    Format a TestingResult for terminal display.

    Args:
        result: Testing agent output.
        include_source: When True, append full generated test modules.

    Returns:
        A readable multi-section string.
    """
    lines = [
        "=" * BANNER_WIDTH,
        "Testing Result",
        "=" * BANNER_WIDTH,
        "",
        "Summary",
        "-" * BANNER_WIDTH,
        (result.summary or "(empty)").strip() or "(empty)",
        "",
        "Coverage estimate",
        "-" * BANNER_WIDTH,
        f"{float(result.coverage_estimate):.2f}",
        "",
        "Generated test filenames",
        "-" * BANNER_WIDTH,
    ]

    filenames = sorted(result.generated_tests.keys())
    if filenames:
        for name in filenames:
            lines.append(f"  - {name}")
    else:
        lines.append("  (none)")

    if include_source and result.generated_tests:
        lines.extend(["", "Generated test source", "-" * BANNER_WIDTH])
        for name in filenames:
            code = result.generated_tests.get(name) or ""
            lines.append(f"\n--- {name} ---")
            lines.append(code.rstrip() or "(empty file)")
            lines.append("")

    lines.append("")
    return "\n".join(lines)


def print_testing_result(
    result: TestingResult, *, include_source: bool = False
) -> None:
    """Print a formatted TestingResult to stdout."""
    print(format_testing_result(result, include_source=include_source))
