"""
report_builder.py
==================

Defines ReportBuilder helpers for labelling and abstention.

Verified findings are assembled by CodeAnalysisAgent; this module owns
the proposal's explicit abstention path so "cannot determine" is
distinguishable from "no bugs found".
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..schemas.schemas import AbstentionResult, BugReport

#: Default next steps when callers do not supply their own.
_DEFAULT_NEXT_STEPS = (
    "Add or point at the relevant source files.",
    "Ask a more specific question about a file or function.",
    "Ensure the repository index can retrieve the target code.",
)


class ReportBuilder:
    """
    Builds abstention records and filters low-confidence findings.
    """

    def __init__(self, min_confidence: float = 0.4) -> None:
        """
        Initialize the ReportBuilder.

        Args:
            min_confidence: Findings below this confidence are withheld
                in favor of abstention when nothing stronger remains.
        """
        self.min_confidence = min_confidence

    def from_static_finding(self, finding: BugReport) -> Optional[BugReport]:
        """
        Pass through a grounded static BugReport when confidence is enough.

        Args:
            finding: Already-constructed BugReport from static analysis.

        Returns:
            The finding, or None when it falls below ``min_confidence``.
        """
        if finding is None:
            return None
        if float(getattr(finding, "confidence", 0.0) or 0.0) < self.min_confidence:
            return None
        return finding

    def from_llm_finding(self, finding: BugReport) -> Optional[BugReport]:
        """
        Pass through a grounded LLM BugReport when confidence is enough.

        Args:
            finding: Already-constructed BugReport from the model path.

        Returns:
            The finding, or None when it falls below ``min_confidence``.
        """
        if finding is None:
            return None
        if float(getattr(finding, "confidence", 0.0) or 0.0) < self.min_confidence:
            return None
        return finding

    def merge(
        self, static_reports: List[BugReport], llm_reports: List[BugReport]
    ) -> List[BugReport]:
        """
        Combine static and LLM findings, dropping low-confidence items.

        Args:
            static_reports: Reports from the deterministic pass.
            llm_reports: Reports proposed by the model.

        Returns:
            Concatenated reports that clear ``min_confidence``.
        """
        merged: List[BugReport] = []
        for report in list(static_reports or []) + list(llm_reports or []):
            if float(getattr(report, "confidence", 0.0) or 0.0) >= self.min_confidence:
                merged.append(report)
        return merged

    def abstain(
        self,
        reason: str,
        *,
        confidence: float = 1.0,
        evidence_available: Optional[Sequence[str]] = None,
        recommended_next_steps: Optional[Sequence[str]] = None,
    ) -> AbstentionResult:
        """
        Build an explicit abstention record.

        Args:
            reason: Why the determination could not be made.
            confidence: Confidence in the abstention decision.
            evidence_available: Evidence that was present but insufficient.
            recommended_next_steps: Suggested follow-up actions.

        Returns:
            An AbstentionResult suitable for attaching to agent outputs.
        """
        steps = list(recommended_next_steps or _DEFAULT_NEXT_STEPS)
        evidence = [str(item) for item in (evidence_available or []) if str(item).strip()]
        return AbstentionResult(
            reason=str(reason or "No grounded evidence was found.").strip()
            or "No grounded evidence was found.",
            confidence=min(max(float(confidence), 0.0), 1.0),
            evidence_available=evidence,
            recommended_next_steps=steps,
        )

    def filter_confident(self, findings: Sequence[BugReport]) -> List[BugReport]:
        """
        Keep only findings at or above ``min_confidence``.

        Args:
            findings: Candidate grounded findings.

        Returns:
            Filtered findings list.
        """
        return [
            finding
            for finding in findings or []
            if float(getattr(finding, "confidence", 0.0) or 0.0) >= self.min_confidence
        ]
