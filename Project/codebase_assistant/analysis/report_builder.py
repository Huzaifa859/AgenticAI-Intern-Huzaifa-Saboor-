"""
report_builder.py
==================

Defines ReportBuilder, which assembles verified findings into the
BugReport objects the user finally sees.

This is where the proposal's labelling rules are enforced: every report
carries a confidence score and a detection method (`static`, `llm`,
`hybrid`, or `dynamic`) so a reader can weight static-confirmed issues
above LLM-only suggestions.

TODO: Implement real construction and merging. Note that the proposal's
"explicit abstention over guessing" path has no schema yet — BugReport
cannot represent "cannot determine", and an empty result list does not
distinguish clean code from insufficient context.

TODO: Add an abstention schema and return it from `abstain()`.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..schemas.schemas import BugReport


class ReportBuilder:
    """
    Builds and merges BugReport objects from analysis findings.
    """

    def __init__(self, min_confidence: float = 0.0) -> None:
        """
        Initialize the ReportBuilder.

        Args:
            min_confidence: Reports below this confidence are withheld
                in favor of abstention.
        """
        self.min_confidence = min_confidence

    def from_static_finding(self, finding: Any) -> Optional[BugReport]:
        """
        Build a report from a deterministic static analysis finding.

        Args:
            finding: Raw finding produced by StaticAnalyzer.

        Returns:
            A BugReport labelled `detection_method="static"`, or None
            (placeholder always returns None).

        TODO: Implement real construction.
        """
        # TODO: implement real static finding conversion
        return None

    def from_llm_finding(self, finding: Any) -> Optional[BugReport]:
        """
        Build a report from an LLM-proposed finding.

        Args:
            finding: Raw finding produced by the Code Analysis Agent.

        Returns:
            A BugReport labelled `detection_method="llm"`, or None
            (placeholder always returns None).

        TODO: Implement real construction. Anything built here must
        still pass GroundingChecker before reaching the user.
        """
        # TODO: implement real LLM finding conversion
        return None

    def merge(self, static_reports: List[BugReport], llm_reports: List[BugReport]) -> List[BugReport]:
        """
        Combine static and LLM findings into one deduplicated list.

        Args:
            static_reports: Reports from the deterministic pass.
            llm_reports: Reports proposed by the model.

        Returns:
            The merged report list (placeholder empty list).

        TODO: Implement real merging. Where both passes describe the
        same issue, the result should be relabelled `hybrid` and its
        confidence raised.
        """
        # TODO: implement real report merging
        return []

    def abstain(self, reason: str) -> None:
        """
        Record that the system cannot determine whether a bug exists.

        Args:
            reason: Why the determination could not be made (e.g.
                insufficient retrieved context, confidence below
                threshold).

        Returns:
            Nothing yet — there is no schema to return.

        TODO: Introduce an abstention schema and return it here, so
        "cannot determine" is distinguishable from "no bugs found".
        """
        # TODO: implement real abstention once a schema exists
        return None
