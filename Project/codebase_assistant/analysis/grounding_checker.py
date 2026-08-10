"""
grounding_checker.py
=====================

Defines GroundingChecker, the verification gate every bug report must
pass before a user ever sees it.

This is the proposal's core hallucination mitigation: a report claims
that specific code exists at a specific file and line range, and this
module reads the real source and confirms it. Claims whose quoted
evidence does not match are discarded rather than downgraded.

Verification is purely mechanical — no model is involved — so it is
equally applicable to static findings and LLM-proposed ones.

TODO: Implement real verification, including a normalization policy for
whitespace and line-ending differences that should not count as a
mismatch.
"""

from __future__ import annotations

from typing import List

from ..schemas.schemas import BugReport


class GroundingChecker:
    """
    Verifies that a bug report's quoted evidence exists in the real
    source at the claimed location.
    """

    def __init__(self, workspace_root: str = ".") -> None:
        """
        Initialize the GroundingChecker.

        Args:
            workspace_root: Root directory claimed file paths are
                resolved against.
        """
        self.workspace_root = workspace_root

    def verify(self, report: BugReport) -> bool:
        """
        Verify a single bug report against the source it cites.

        Args:
            report: The BugReport to verify.

        Returns:
            True if the report's evidence is grounded (placeholder
            always returns False).

        TODO: Read the cited file, extract the claimed line range, and
        compare it against `report.evidence`.
        """
        # TODO: implement real grounding verification
        return False

    def verify_snippet(self, file_path: str, line_start: int, line_end: int, evidence: str) -> bool:
        """
        Verify that a snippet appears at a given location in a file.

        Args:
            file_path: Path to the file the snippet is claimed to be in.
            line_start: First line of the claimed range.
            line_end: Last line of the claimed range.
            evidence: Exact code the claim quotes.

        Returns:
            True if the snippet matches (placeholder always returns
            False).

        TODO: Implement the real comparison, handling out-of-range line
        numbers and unreadable files as a failed check rather than an
        error.
        """
        # TODO: implement real snippet verification
        return False

    def filter_reports(self, reports: List[BugReport]) -> List[BugReport]:
        """
        Drop every report that fails verification.

        Args:
            reports: Candidate reports, from either the static pass or
                the LLM.

        Returns:
            Only the reports whose evidence was verified (placeholder
            empty list).

        TODO: Implement real filtering, and log each discarded claim so
        the false-positive rate can be measured against the evaluation
        benchmark.
        """
        # TODO: implement real report filtering
        return []
