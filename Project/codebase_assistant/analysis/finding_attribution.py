"""
finding_attribution.py
======================

Attribution and status annotations for Code Analysis findings.

Grounding and documentation checks classify findings; they do not remove
them. Display labels are stored on ``BugReport.metadata`` so schemas stay
stable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from ..schemas.schemas import BugReport

FOUND_BY_STATIC = "Static Analysis"
FOUND_BY_LLM = "LLM"
FOUND_BY_BOTH = "Static + LLM"

GROUNDING_GROUNDED = "grounded"
GROUNDING_UNGROUNDED = "ungrounded"

META_FOUND_BY = "found_by"
META_GROUNDING_STATUS = "grounding_status"
META_GROUNDING_REASON = "grounding_reason"
META_ALREADY_DOCUMENTED = "already_documented"

_DOC_STRING = re.compile(
    r"(?s)([ruRU]{0,2})('''.*?'''|\"\"\".*?\"\"\")"
)
_LINE_COMMENT = re.compile(r"(?m)^\s*#.*?$")
_EXPLICIT_BUG_DOC = re.compile(
    r"(?is)(?:^|[^\w])("
    r"bug\s*:|bugs\s*:|FIXME|XXX|TODO\s*:|"
    r"known\s+bug|known\s+issue|incorrectly|wrongly|broken\s+because"
    r")(?:\b|(?=\s)|$)"
)


def set_found_by(finding: BugReport, found_by: str) -> None:
    """Stamp Found-by attribution and align ``detection_method``."""
    finding.metadata[META_FOUND_BY] = found_by
    if found_by == FOUND_BY_BOTH:
        finding.detection_method = "hybrid"
    elif found_by == FOUND_BY_STATIC:
        finding.detection_method = "static"
    elif found_by == FOUND_BY_LLM:
        finding.detection_method = "llm"


def set_grounding_status(
    finding: BugReport,
    *,
    grounded: bool,
    reason: str = "",
) -> None:
    """Record grounded/ungrounded classification without dropping the finding."""
    finding.metadata[META_GROUNDING_STATUS] = (
        GROUNDING_GROUNDED if grounded else GROUNDING_UNGROUNDED
    )
    if reason:
        finding.metadata[META_GROUNDING_REASON] = reason


def set_already_documented(finding: BugReport, documented: bool) -> None:
    """Mark findings whose defect is already called out in nearby docs."""
    if documented:
        finding.metadata[META_ALREADY_DOCUMENTED] = True
    else:
        finding.metadata.pop(META_ALREADY_DOCUMENTED, None)


def found_by_label(finding: BugReport) -> str:
    """Human-readable Found-by label with detection_method fallback."""
    explicit = str(finding.metadata.get(META_FOUND_BY) or "").strip()
    if explicit:
        return explicit
    method = (finding.detection_method or "").lower()
    if method == "hybrid":
        return FOUND_BY_BOTH
    if method == "static":
        return FOUND_BY_STATIC
    if method == "llm":
        return FOUND_BY_LLM
    return method or "Unknown"


def grounding_label(finding: BugReport) -> Optional[str]:
    """Return ``Grounded evidence`` / ``Ungrounded evidence``, or None."""
    status = str(finding.metadata.get(META_GROUNDING_STATUS) or "").lower()
    if status == GROUNDING_GROUNDED:
        return "Grounded evidence"
    if status == GROUNDING_UNGROUNDED:
        return "Ungrounded evidence"
    return None


def attribution_lines(finding: BugReport) -> List[str]:
    """
    Combinable display lines for a finding.

    Example::

        Found by: Static + LLM
        Grounded evidence
        Already documented in the code.
    """
    lines = [f"Found by: {found_by_label(finding)}"]
    grounded = grounding_label(finding)
    if grounded:
        lines.append(grounded)
    if finding.metadata.get(META_ALREADY_DOCUMENTED):
        lines.append("Already documented in the code.")
    return lines


def _documentation_regions(source_text: str) -> List[str]:
    regions: List[str] = []
    for match in _DOC_STRING.finditer(source_text or ""):
        regions.append(match.group(0))
    for match in _LINE_COMMENT.finditer(source_text or ""):
        regions.append(match.group(0))
    return regions


def _significant_tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "this",
        "that",
        "with",
        "when",
        "from",
        "not",
        "it",
        "as",
        "by",
        "at",
        "bug",
        "bugs",
        "issue",
        "issues",
        "known",
        "todo",
        "fixme",
        "xxx",
        "incorrectly",
        "wrongly",
        "broken",
        "because",
    }
    return {
        token
        for token in re.findall(r"[a-z_][a-z0-9_]{2,}", (text or "").lower())
        if token not in stop
    }


def is_already_documented_in_source(
    source_text: str,
    *,
    description: str,
    bug_type: str = "",
) -> bool:
    """
    True when a docstring/comment explicitly discusses the same concrete defect.

    Ordinary API docs without an explicit bug/FIXME-style callout do not
    qualify.
    """
    finding_tokens = _significant_tokens(f"{bug_type} {description}")
    if not finding_tokens:
        return False

    for region in _documentation_regions(source_text):
        if not _EXPLICIT_BUG_DOC.search(region):
            continue
        overlap = finding_tokens & _significant_tokens(region)
        if len(overlap) >= 2 or (
            len(overlap) == 1 and len(finding_tokens) <= 3
        ):
            return True
    return False


def merge_attribution(primary: BugReport, secondary: BugReport) -> None:
    """
    Fold ``secondary`` into ``primary`` when both describe the same defect.

    Preserves Static + LLM found-by, prefers grounded over ungrounded, and
    keeps an already-documented flag if either side has it.
    """
    sources = {found_by_label(primary), found_by_label(secondary)}
    methods = {
        (primary.detection_method or "").lower(),
        (secondary.detection_method or "").lower(),
    }
    if (
        FOUND_BY_BOTH in sources
        or methods >= {"static", "llm"}
        or "hybrid" in methods
        or sources >= {FOUND_BY_STATIC, FOUND_BY_LLM}
    ):
        set_found_by(primary, FOUND_BY_BOTH)
    elif FOUND_BY_STATIC in sources or "static" in methods:
        set_found_by(primary, FOUND_BY_STATIC)
    elif FOUND_BY_LLM in sources or "llm" in methods:
        set_found_by(primary, FOUND_BY_LLM)

    statuses = {
        str(primary.metadata.get(META_GROUNDING_STATUS) or "").lower(),
        str(secondary.metadata.get(META_GROUNDING_STATUS) or "").lower(),
    }
    if GROUNDING_GROUNDED in statuses:
        reason = str(
            primary.metadata.get(META_GROUNDING_REASON)
            or secondary.metadata.get(META_GROUNDING_REASON)
            or ""
        )
        set_grounding_status(primary, grounded=True, reason=reason)
    elif GROUNDING_UNGROUNDED in statuses:
        reason = str(
            primary.metadata.get(META_GROUNDING_REASON)
            or secondary.metadata.get(META_GROUNDING_REASON)
            or ""
        )
        set_grounding_status(primary, grounded=False, reason=reason)

    if primary.metadata.get(META_ALREADY_DOCUMENTED) or secondary.metadata.get(
        META_ALREADY_DOCUMENTED
    ):
        set_already_documented(primary, True)

    if not (primary.evidence or "").strip() and (secondary.evidence or "").strip():
        primary.evidence = secondary.evidence


def annotate_batch_from_verification(
    findings: Sequence[BugReport],
    results: Sequence[Any],
    *,
    found_by: str,
) -> List[BugReport]:
    """
    Apply Found-by + grounding annotations for a verified batch.

    Every input finding is returned; grounding never filters the list.
    """
    by_id: Dict[int, Any] = {}
    for result in results:
        nested = getattr(result, "report", None)
        if nested is not None:
            by_id[id(nested)] = result

    annotated: List[BugReport] = []
    for finding in findings:
        set_found_by(finding, found_by)
        result = by_id.get(id(finding))
        if result is None:
            annotated.append(finding)
            continue
        grounded = bool(getattr(result, "grounded", False))
        reason = str(getattr(result, "reason", "") or "")
        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)
        if status_value and not reason:
            reason = str(status_value)
        set_grounding_status(finding, grounded=grounded, reason=reason)
        annotated.append(finding)
    return annotated
