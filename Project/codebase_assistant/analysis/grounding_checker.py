"""
grounding_checker.py
=====================

The verification gate every bug report must pass before a user sees it.

This is the proposal's core hallucination mitigation: a report claims
that specific code exists at a specific file and line range, and this
module reads the real source and confirms it. Claims whose quoted
evidence does not match are discarded rather than downgraded.

Verification is purely mechanical -- no model is involved, nothing is
inferred, and the same input always produces the same verdict -- so it
applies equally to static findings and LLM-proposed ones. A finding from
StaticAnalyzer and a finding invented by a model are put through exactly
the same check, which is the point: the checker has no way to tell them
apart and no reason to.

A failed check is not an error. An unverifiable claim is data -- it is
what a hallucination looks like -- so it comes back as a
`GroundingResult` carrying the reason, not as a raised exception.
`GroundingVerificationError` is reserved for the case where the check
itself could not be carried out.

When quoted evidence exists in the cited file but the line numbers are
wrong, the checker may deterministically relocate the finding to the
matching span. Relocation never invents evidence and never uses fuzzy
or semantic matching -- only exact and mechanical normalizations.
"""

from __future__ import annotations

import hashlib
import io
import logging
import textwrap
import tokenize
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import Config
from ..exceptions.base import CodebaseAssistantError
from ..exceptions.tool_exceptions import PathOutsideWorkspaceError
from ..schemas.schemas import BugReport
from ..tools.filesystem_tools import FilesystemTools
from ..tracing.events import TraceEventType
from ..tracing.tracer import Tracer

logger = logging.getLogger(__name__)

#: Occurrence: (line_start, line_end, match_type) — 1-based inclusive.
_Occurrence = Tuple[int, int, "MatchType"]


class GroundingStatus(str, Enum):
    """
    Why a report passed or failed verification.

    A string enum so a status survives JSON serialization into a trace
    or a notebook table without conversion.
    """

    VERIFIED = "verified"
    FILE_MISSING = "file_missing"
    FILE_UNREADABLE = "file_unreadable"
    INVALID_LINE_RANGE = "invalid_line_range"
    EMPTY_EVIDENCE = "empty_evidence"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    SOURCE_CHANGED = "source_changed"


class MatchType(str, Enum):
    """
    How closely the evidence matched the source.

    Recorded even on success, so a caller that wants byte-for-byte
    certainty can tell an exact match from a tolerated one without
    re-running the check in strict mode.
    """

    EXACT = "exact"
    NORMALIZED = "normalized"
    DEDENTED = "dedented"
    COMMENT_STRIPPED = "comment_stripped"
    NONE = "none"


_MATCH_PRIORITY = {
    MatchType.EXACT: 0,
    MatchType.NORMALIZED: 1,
    MatchType.DEDENTED: 2,
    MatchType.COMMENT_STRIPPED: 3,
}


@dataclass
class GroundingResult:
    """
    The verdict on a single report.

    Attributes:
        grounded: Whether the claim is supported by the real source.
        status: The specific reason, machine-readable.
        reason: The same reason in a sentence, for logs and users.
        expected_evidence: What the report claimed the source says.
        actual_source: What the source actually says at the claimed
            range. Empty when the file could not be read or the range
            does not exist.
        match_type: How closely the two matched.
        file_path: File the claim cites.
        line_start: First line of the claimed range.
        line_end: Last line of the claimed range.
        found_at_line: Where the evidence actually appears, when it is
            present in the file but not at the claimed range. This is
            what separates a stale report from an invented one.
        source_changed: True when the file's hash differs from the
            snapshot taken at analysis time. On its own this does not
            reject a report -- see `verify_report`.
        report: The report that was verified, when there was one.
        metadata: Mechanical annotations (e.g. relocation details).
    """

    grounded: bool
    status: GroundingStatus
    reason: str
    expected_evidence: str = ""
    actual_source: str = ""
    match_type: MatchType = MatchType.NONE
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    found_at_line: Optional[int] = None
    source_changed: bool = False
    report: Optional[BugReport] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        """
        Render a one-line verdict.

        Returns:
            A readable summary suitable for logs or a notebook cell.
        """
        verdict = "GROUNDED" if self.grounded else "REJECTED"
        location = f"{self.file_path}:{self.line_start}-{self.line_end}"
        return f"[{verdict}] {location} ({self.status.value}) {self.reason}"


@dataclass
class VerificationSummary:
    """
    The outcome of verifying a batch of reports.

    Attributes:
        grounded: Reports that passed, in their original order.
        rejected: Reports that failed, in their original order.
        results: Every GroundingResult, one per input report, in order.
    """

    grounded: List[BugReport] = field(default_factory=list)
    rejected: List[BugReport] = field(default_factory=list)
    results: List[GroundingResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Number of reports verified."""
        return len(self.results)

    @property
    def rejection_rate(self) -> float:
        """
        Share of reports that failed verification.

        Returns:
            A ratio between 0.0 and 1.0, or 0.0 for an empty batch.
        """
        if not self.results:
            return 0.0
        return len(self.rejected) / len(self.results)

    def rejections_by_status(self) -> Dict[str, int]:
        """
        Count rejections by failure mode.

        Returns:
            A mapping of status value to count, most common first.
        """
        counts: Dict[str, int] = {}
        for result in self.results:
            if result.grounded:
                continue
            counts[result.status.value] = counts.get(result.status.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    def summary(self) -> str:
        """
        Render a one-line summary of the batch.

        Returns:
            A readable summary suitable for logs or a notebook cell.
        """
        parts = [
            f"{len(self.grounded)}/{self.total} grounded",
            f"{len(self.rejected)} rejected",
        ]
        breakdown = self.rejections_by_status()
        if breakdown:
            parts.append(
                "("
                + ", ".join(f"{name}: {count}" for name, count in breakdown.items())
                + ")"
            )
        return " ".join(parts)


class GroundingChecker:
    """
    Verifies that a bug report's quoted evidence exists in the real
    source at the claimed location.
    """

    def __init__(
        self,
        workspace_root: str = ".",
        config: Optional[Config] = None,
        filesystem: Optional[FilesystemTools] = None,
        strict: bool = False,
        snapshot: Optional[Dict[str, str]] = None,
        tracer: Optional[Tracer] = None,
    ) -> None:
        """
        Initialize the GroundingChecker.

        Args:
            workspace_root: Root directory claimed file paths are
                resolved against. Kept as the leading parameter to match
                the scaffold's original signature.
            config: Optional Config instance. A default is loaded when
                not supplied.
            filesystem: Optional FilesystemTools. Built from the config
                and workspace root when omitted, so the sandbox check
                and the size ceiling are enforced in one place rather
                than restated here.
            strict: When True only a byte-for-byte match grounds a
                report. When False, differences that cannot change what
                the code does -- line endings and trailing whitespace --
                are tolerated and recorded. See `_compare`.
            snapshot: Optional mapping of file path to content hash,
                captured when the reports were produced. Enables
                staleness detection; see `capture_snapshot`.
            tracer: Optional Tracer for relocation / grounding events.

        Raises:
            ToolExecutionError: If the workspace root does not exist.
        """
        self.workspace_root = workspace_root
        self.config = config or Config.load()
        self.filesystem = filesystem or FilesystemTools(
            workspace_root=workspace_root, config=self.config
        )
        self.strict = strict
        self.snapshot: Dict[str, str] = dict(snapshot or {})
        self.tracer = tracer

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_report(
        self,
        report: BugReport,
        source_cache: Optional[Dict[str, Optional[str]]] = None,
    ) -> GroundingResult:
        """
        Verify a single report against the source it cites.

        The checks run in the order they can fail, cheapest first: empty
        evidence needs no file at all, a missing file makes the range
        meaningless, and an invalid range makes the comparison
        impossible. Each stage returns a specific status rather than a
        bare False, because "the model quoted nothing" and "the model
        quoted something that is not there" are different failures and
        only the second is a hallucination.

        Args:
            report: The report to verify.
            source_cache: Optional path-to-source map reused across a
                batch, so a file cited by twenty findings is read once.
                A None value marks a file already known to be
                unreadable.

        Returns:
            The verdict, including both the expected evidence and the
            actual source so a caller can show the difference.
        """
        expected = report.evidence or ""

        if not expected.strip():
            return self._reject(
                report,
                GroundingStatus.EMPTY_EVIDENCE,
                "The report quotes no evidence, so there is nothing to "
                "verify against the source.",
                expected,
            )

        try:
            source = self._read(report.file_path, source_cache)
        except FileNotFoundError:
            return self._reject(
                report,
                GroundingStatus.FILE_MISSING,
                f"The cited file {report.file_path!r} does not exist.",
                expected,
            )
        except CodebaseAssistantError as exc:
            return self._reject(
                report,
                GroundingStatus.FILE_UNREADABLE,
                f"The cited file {report.file_path!r} could not be read: {exc}",
                expected,
            )

        lines = source.split("\n")
        changed = self._has_changed(report.file_path, source)
        range_error = self._validate_range(
            report.line_start, report.line_end, len(lines)
        )

        actual = ""
        if range_error is None:
            actual = "\n".join(lines[report.line_start - 1 : report.line_end])
            match_type = self._compare(expected, actual)
            if match_type is not MatchType.NONE:
                # The claim holds at the cited range. A file that has
                # changed elsewhere does not invalidate it -- the
                # evidence is verifiable right now.
                if changed:
                    logger.info(
                        "Grounded but %s changed since analysis: %s:%d-%d",
                        report.file_path,
                        report.file_path,
                        report.line_start,
                        report.line_end,
                    )
                return self._accept(report, actual, match_type, changed)

        # Line numbers are wrong or out of range, but the quote may still
        # exist elsewhere in this file. Relocate deterministically when
        # the evidence is found; never invent evidence.
        repaired = self._relocate_evidence(report, expected, source)
        if repaired is not None:
            return repaired

        if range_error is not None:
            return self._reject(
                report,
                GroundingStatus.INVALID_LINE_RANGE,
                range_error,
                expected,
            )

        if changed:
            return self._reject(
                report,
                GroundingStatus.SOURCE_CHANGED,
                f"{report.file_path} has changed since analysis and the cited "
                f"code is no longer present anywhere in it.",
                expected,
                actual=actual,
                source_changed=True,
            )

        return self._reject(
            report,
            GroundingStatus.EVIDENCE_MISMATCH,
            f"The quoted evidence does not appear at "
            f"{report.file_path}:{report.line_start}-{report.line_end}, or "
            f"anywhere else in the file.",
            expected,
            actual=actual,
        )

    def verify_reports(self, reports: Sequence[BugReport]) -> VerificationSummary:
        """
        Verify a batch of reports, keeping the two outcomes separate.

        Files are read once per batch. The cache lives only for the
        duration of this call and is deliberately not kept on the
        instance: a checker that remembered file contents between calls
        would be unable to notice the source changing, which is one of
        the things it exists to detect.

        Args:
            reports: Candidate reports, from either the static pass or
                the LLM.

        Returns:
            The grounded reports, the rejected reports, and the full
            result for each.
        """
        summary = VerificationSummary()
        source_cache: Dict[str, Optional[str]] = {}

        for report in reports:
            result = self.verify_report(report, source_cache=source_cache)
            summary.results.append(result)

            if result.grounded:
                # Prefer the (possibly line-repaired) report from the
                # verdict so callers see updated line numbers.
                summary.grounded.append(result.report or report)
                logger.debug("Grounded: %s", result.summary())
            else:
                summary.rejected.append(report)
                logger.warning("Rejected: %s", result.summary())

        if summary.rejected:
            logger.info("Grounding check: %s", summary.summary())

        self._trace(
            "grounding_completed",
            success=True,
            total=summary.total,
            grounded=len(summary.grounded),
            rejected=len(summary.rejected),
            relocated=sum(
                1
                for result in summary.results
                if result.grounded and result.metadata.get("evidence_relocated")
            ),
        )
        return summary

    def verify_evidence(
        self, file_path: str, line_start: int, line_end: int, evidence: str
    ) -> GroundingResult:
        """
        Verify a snippet without constructing a BugReport.

        The same check `verify_report` runs, for callers holding a
        location and a quote but not yet a report -- an agent checking
        its own draft before committing to it, for example.

        Args:
            file_path: File the snippet is claimed to be in.
            line_start: First line of the claimed range.
            line_end: Last line of the claimed range.
            evidence: Exact code the claim quotes.

        Returns:
            The verdict, with `report` left as None.
        """
        probe = BugReport(
            bug_type="grounding_probe",
            description="Ad-hoc snippet verification.",
            severity="low",
            confidence=0.0,
            file_path=file_path,
            function_name="<unknown>",
            line_start=line_start,
            line_end=line_end,
            evidence=evidence,
            detection_method="static",
        )
        result = self.verify_report(probe)
        result.report = None
        return result

    def filter_reports(self, reports: Sequence[BugReport]) -> List[BugReport]:
        """
        Drop every report that fails verification.

        The convenience form of `verify_reports` for callers that only
        want the survivors. Each discarded claim is logged, so the
        rejections are still recoverable from the log even though they
        are not returned.

        Args:
            reports: Candidate reports, from either the static pass or
                the LLM.

        Returns:
            Only the reports whose evidence was verified.
        """
        return self.verify_reports(reports).grounded

    # ------------------------------------------------------------------
    # Scaffold interface (boolean forms)
    # ------------------------------------------------------------------

    def verify(self, report: BugReport) -> bool:
        """
        Verify a single bug report against the source it cites.

        The scaffold's original boolean form. `verify_report` returns
        the same verdict with the reason attached and should be
        preferred; this exists so existing callers keep working.

        Args:
            report: The BugReport to verify.

        Returns:
            True if the report's evidence is grounded.
        """
        return self.verify_report(report).grounded

    def verify_snippet(
        self, file_path: str, line_start: int, line_end: int, evidence: str
    ) -> bool:
        """
        Verify that a snippet appears at a given location in a file.

        The scaffold's original boolean form of `verify_evidence`. Out
        of range line numbers and unreadable files come back as False
        rather than raising.

        Args:
            file_path: Path to the file the snippet is claimed to be in.
            line_start: First line of the claimed range.
            line_end: Last line of the claimed range.
            evidence: Exact code the claim quotes.

        Returns:
            True if the snippet matches.
        """
        return self.verify_evidence(
            file_path, line_start, line_end, evidence
        ).grounded

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def capture_snapshot(self, file_paths: Iterable[str]) -> Dict[str, str]:
        """
        Hash a set of files so later edits can be detected.

        Relocation detection catches the common case -- code moved, so
        the line numbers drifted -- but not the case where a file was
        edited and the cited lines coincidentally still read the same.
        Taking hashes at analysis time closes that gap.

        Call this when the reports are produced, then hand the result to
        the checker that verifies them.

        Args:
            file_paths: Files to hash. Unreadable ones are skipped, on
                the grounds that a file which cannot be read now will be
                caught by verification anyway.

        Returns:
            A mapping of file path to content hash, also stored on the
            instance.
        """
        snapshot: Dict[str, str] = {}

        for path in file_paths:
            try:
                snapshot[path] = self._hash(self.filesystem.read_file(path))
            except CodebaseAssistantError as exc:
                logger.debug("Could not snapshot %s: %s", path, exc)

        self.snapshot = snapshot
        return snapshot

    def snapshot_reports(self, reports: Sequence[BugReport]) -> Dict[str, str]:
        """
        Hash every file cited by a batch of reports.

        Args:
            reports: Reports whose files should be hashed.

        Returns:
            A mapping of file path to content hash, also stored on the
            instance.
        """
        return self.capture_snapshot(
            sorted({report.file_path for report in reports})
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _compare(self, expected: str, actual: str) -> MatchType:
        """
        Compare quoted evidence against real source.

        Exact first. In non-strict mode further mechanical tiers are
        tried: line endings / trailing whitespace, shared leading
        indent, then comment+whitespace token signatures. No fuzzy or
        semantic matching.

        Args:
            expected: Evidence quoted by the report.
            actual: Source text at the claimed range.

        Returns:
            The tier at which the two matched, or NONE.
        """
        if expected == actual:
            return MatchType.EXACT
        if self.strict:
            return MatchType.NONE

        if self._normalize(expected) == self._normalize(actual):
            return MatchType.NORMALIZED
        if self._dedent(expected) == self._dedent(actual):
            return MatchType.DEDENTED

        expected_sig = self._token_signature(expected)
        actual_sig = self._token_signature(actual)
        if (
            expected_sig is not None
            and actual_sig is not None
            and expected_sig == actual_sig
        ):
            return MatchType.COMMENT_STRIPPED
        return MatchType.NONE

    def _relocate_evidence(
        self,
        report: BugReport,
        expected: str,
        source: str,
    ) -> Optional[GroundingResult]:
        """
        Repair line numbers when evidence exists elsewhere in the file.

        Args:
            report: Finding with inaccurate lines (mutated on success).
            expected: Quoted evidence.
            source: Full file contents.

        Returns:
            A grounded result after repair, or None if evidence is absent.
        """
        original_start = int(report.line_start or 0)
        original_end = int(report.line_end or original_start)

        self._trace(
            "evidence_search_started",
            success=True,
            file=report.file_path,
            original_lines=[original_start, original_end],
        )

        occurrences = self._find_evidence_occurrences(expected, source)
        if not occurrences:
            self._trace(
                "evidence_relocation_failed",
                success=False,
                file=report.file_path,
                original_lines=[original_start, original_end],
                reason="evidence_not_found",
            )
            return None

        chosen = self._select_occurrence(occurrences, original_start)
        new_start, new_end, match_type = chosen
        lines = source.split("\n")
        actual = "\n".join(lines[new_start - 1 : new_end])

        # Prefer keeping the caller's evidence string; acceptance still
        # requires a mechanical match at the repaired range.
        accepted_match = self._compare(expected, actual)
        if accepted_match is MatchType.NONE:
            self._trace(
                "evidence_relocation_failed",
                success=False,
                file=report.file_path,
                original_lines=[original_start, original_end],
                relocated_lines=[new_start, new_end],
                reason="post_repair_mismatch",
            )
            return None

        relocation_meta = {
            "evidence_relocated": True,
            "original_lines": [original_start, original_end],
            "relocated_lines": [new_start, new_end],
            "relocation_match_type": match_type.value,
        }
        report.line_start = new_start
        report.line_end = new_end
        report.metadata.update(relocation_meta)

        self._trace(
            "evidence_relocated",
            success=True,
            file=report.file_path,
            original_lines=[original_start, original_end],
            relocated_lines=[new_start, new_end],
            match_type=match_type.value,
        )
        logger.info(
            "Relocated evidence for %s from %d-%d to %d-%d (%s).",
            report.file_path,
            original_start,
            original_end,
            new_start,
            new_end,
            match_type.value,
        )

        changed = self._has_changed(report.file_path, source)
        result = self._accept(
            report,
            actual,
            accepted_match,
            changed,
            metadata=relocation_meta,
        )
        result.found_at_line = new_start
        return result

    def _find_evidence_occurrences(
        self, expected: str, source: str
    ) -> List[_Occurrence]:
        """
        Find every mechanical match of evidence inside source.

        Exact / whitespace / dedent matches use a fixed line span.
        Comment-stripped matches allow a bounded larger span so
        comment-only lines between tokens can still align.
        """
        lines = source.split("\n")
        if not lines:
            return []

        evidence_lines = expected.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        span = max(1, len(evidence_lines))
        found: List[_Occurrence] = []
        seen = set()

        for start in range(len(lines)):
            end_index = start + span
            if end_index > len(lines):
                break
            candidate = "\n".join(lines[start:end_index])
            match_type = self._compare_without_comment_tier(expected, candidate)
            if match_type is MatchType.NONE:
                continue
            key = (start + 1, end_index, match_type)
            if key in seen:
                continue
            seen.add(key)
            found.append((start + 1, end_index, match_type))

        if self.strict:
            return found

        target_sig = self._token_signature(expected)
        if target_sig is None or not target_sig:
            return found

        max_span = max(span + 20, span * 3)
        for start in range(len(lines)):
            for length in range(span, max_span + 1):
                end_index = start + length
                if end_index > len(lines):
                    break
                candidate = "\n".join(lines[start:end_index])
                candidate_sig = self._token_signature(candidate)
                if candidate_sig is None or candidate_sig != target_sig:
                    continue
                key = (start + 1, end_index, MatchType.COMMENT_STRIPPED)
                if key in seen:
                    continue
                # Skip if an exact/ws match already covers this start.
                if any(item[0] == start + 1 and item[2] != MatchType.COMMENT_STRIPPED for item in found):
                    continue
                seen.add(key)
                found.append((start + 1, end_index, MatchType.COMMENT_STRIPPED))

        return found

    def _compare_without_comment_tier(
        self, expected: str, actual: str
    ) -> MatchType:
        """Like `_compare` but stops before comment-stripped matching."""
        if expected == actual:
            return MatchType.EXACT
        if self.strict:
            return MatchType.NONE
        if self._normalize(expected) == self._normalize(actual):
            return MatchType.NORMALIZED
        if self._dedent(expected) == self._dedent(actual):
            return MatchType.DEDENTED
        return MatchType.NONE

    @staticmethod
    def _select_occurrence(
        occurrences: Sequence[_Occurrence], original_start: int
    ) -> _Occurrence:
        """
        Choose one occurrence deterministically.

        Prefer tighter match tiers, then proximity to the original
        line_start, then earlier line numbers.
        """
        anchor = original_start if original_start > 0 else 1

        def sort_key(item: _Occurrence) -> Tuple[int, int, int, int, int]:
            start, end, match_type = item
            return (
                _MATCH_PRIORITY.get(match_type, 99),
                abs(start - anchor),
                end - start,
                start,
                end,
            )

        return sorted(occurrences, key=sort_key)[0]

    @staticmethod
    def _token_signature(text: str) -> Optional[Tuple[Tuple[int, str], ...]]:
        """
        Build a comment- and whitespace-insensitive token signature.

        Returns:
            A tuple of (token_type, token_string) pairs, or None when
            the snippet cannot be tokenized.
        """
        unified = text.replace("\r\n", "\n").replace("\r", "\n")
        if not unified.strip():
            return None
        try:
            tokens: List[Tuple[int, str]] = []
            for tok in tokenize.generate_tokens(io.StringIO(unified).readline):
                if tok.type in (
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                    tokenize.ENCODING,
                    tokenize.ENDMARKER,
                ):
                    continue
                if tok.type == tokenize.STRING:
                    tokens.append((tok.type, tok.string))
                else:
                    tokens.append((tok.type, tok.string))
            return tuple(tokens) if tokens else None
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return None

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Strip differences that cannot change the code's behavior.

        Args:
            text: Text to normalize.

        Returns:
            The text with line endings unified and trailing whitespace
            removed from each line and from the whole.
        """
        unified = text.replace("\r\n", "\n").replace("\r", "\n")
        return "\n".join(line.rstrip() for line in unified.split("\n")).strip("\n")

    @classmethod
    def _dedent(cls, text: str) -> str:
        """
        Normalize, then remove the common leading indentation.

        Args:
            text: Text to dedent.

        Returns:
            The normalized text with its shared left margin removed.
        """
        return textwrap.dedent(cls._normalize(text))

    def _locate(self, expected: str, source: str) -> Optional[int]:
        """
        Find where the evidence actually appears in a file.

        Compatibility helper over `_find_evidence_occurrences`.

        Args:
            expected: Evidence quoted by the report.
            source: The file's full contents.

        Returns:
            The 1-based line where the best occurrence starts, or None.
        """
        occurrences = self._find_evidence_occurrences(expected, source)
        if not occurrences:
            return None
        return self._select_occurrence(occurrences, 1)[0]

    @staticmethod
    def _validate_range(
        line_start: int, line_end: int, total_lines: int
    ) -> Optional[str]:
        """
        Check a claimed line range against the real file.

        BugReport types its line numbers as plain ints, so nothing stops
        a caller -- or a model -- from producing a range that is
        negative, inverted, or past the end of the file. Each is caught
        here with a specific message rather than being allowed to slice
        into an empty or unexpected result.

        Args:
            line_start: First line of the claimed range.
            line_end: Last line of the claimed range.
            total_lines: Number of lines in the file.

        Returns:
            A description of the problem, or None if the range is
            usable.
        """
        if line_start < 1:
            return (
                f"Line range starts at {line_start}; line numbers are 1-based, "
                f"so this range cannot exist."
            )
        if line_end < line_start:
            return (
                f"Line range is inverted: it ends at {line_end} but starts at "
                f"{line_start}."
            )
        if line_start > total_lines:
            return (
                f"Line range starts at {line_start} but the file has only "
                f"{total_lines} line(s)."
            )
        if line_end > total_lines:
            return (
                f"Line range ends at {line_end} but the file has only "
                f"{total_lines} line(s)."
            )
        return None

    def _has_changed(self, file_path: str, source: str) -> bool:
        """
        Check a file against the snapshot taken at analysis time.

        Args:
            file_path: File to check.
            source: Its current contents.

        Returns:
            True if a hash was recorded for this file and no longer
            matches. False when no snapshot was taken, since absence of
            evidence is not evidence of change.
        """
        recorded = self.snapshot.get(file_path)
        return recorded is not None and recorded != self._hash(source)

    @staticmethod
    def _hash(source: str) -> str:
        """
        Hash file contents for change detection.

        Args:
            source: The file's contents.

        Returns:
            A hex SHA-256 digest.
        """
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _read(
        self, file_path: str, cache: Optional[Dict[str, Optional[str]]]
    ) -> str:
        """
        Read a cited file through FilesystemTools.

        Direct file I/O is deliberately avoided: routing through
        FilesystemTools means a report citing a path outside the
        workspace is rejected by the same sandbox check that governs
        every other read, rather than by a rule restated here.

        Args:
            file_path: File to read, relative to the workspace root.
            cache: Optional batch cache. A None entry marks a file
                already found to be unreadable.

        Returns:
            The file's contents.

        Raises:
            FileNotFoundError: If the file does not exist.
            PathOutsideWorkspaceError: If the path escapes the
                workspace.
            ToolExecutionError: If the file exists but cannot be read.
            FileTooLargeError: If the file exceeds the size ceiling.
        """
        if not file_path or not str(file_path).strip():
            raise FileNotFoundError("The report cites no file path.")

        if cache is not None and file_path in cache:
            cached = cache[file_path]
            if cached is None:
                raise FileNotFoundError(f"File not found: {file_path!r}")
            return cached

        try:
            if not self.filesystem.file_exists(file_path):
                if cache is not None:
                    cache[file_path] = None
                raise FileNotFoundError(f"File not found: {file_path!r}")
            source = self.filesystem.read_file(file_path)
        except PathOutsideWorkspaceError:
            # A path outside the sandbox is indistinguishable from a
            # missing one as far as the claim is concerned: it cannot be
            # verified, so the report does not survive.
            if cache is not None:
                cache[file_path] = None
            raise FileNotFoundError(
                f"File is outside the workspace: {file_path!r}"
            ) from None

        if cache is not None:
            cache[file_path] = source
        return source

    @staticmethod
    def _accept(
        report: BugReport,
        actual: str,
        match_type: MatchType,
        source_changed: bool,
        metadata: Optional[Dict[str, object]] = None,
    ) -> GroundingResult:
        """
        Build the verdict for a report that passed.

        Args:
            report: The verified report.
            actual: Source text at the claimed range.
            match_type: How closely the evidence matched.
            source_changed: Whether the file has changed since the
                snapshot.
            metadata: Optional mechanical annotations.

        Returns:
            A grounded GroundingResult.
        """
        detail = {
            MatchType.EXACT: "matches the source exactly",
            MatchType.NORMALIZED: (
                "matches the source apart from line endings or trailing "
                "whitespace"
            ),
            MatchType.DEDENTED: (
                "matches the source apart from the quote's leading indentation"
            ),
            MatchType.COMMENT_STRIPPED: (
                "matches the source apart from comments or insignificant "
                "whitespace"
            ),
        }[match_type]

        reason = (
            f"Evidence {detail} at {report.file_path}:"
            f"{report.line_start}-{report.line_end}."
        )
        if source_changed:
            reason += " Note: the file has changed since analysis."
        if metadata and metadata.get("evidence_relocated"):
            original = metadata.get("original_lines")
            reason += f" Line numbers were repaired from {original}."

        return GroundingResult(
            grounded=True,
            status=GroundingStatus.VERIFIED,
            reason=reason,
            expected_evidence=report.evidence,
            actual_source=actual,
            match_type=match_type,
            file_path=report.file_path,
            line_start=report.line_start,
            line_end=report.line_end,
            source_changed=source_changed,
            report=report,
            metadata=dict(metadata or {}),
        )

    def _trace(self, name: str, **metadata: object) -> None:
        """Record a grounding event; never raises."""
        if self.tracer is None:
            return
        try:
            self.tracer.record(
                TraceEventType.AGENT_RUN,
                name,
                component="GroundingChecker",
                **metadata,
            )
        except Exception as exc:
            logger.warning("GroundingChecker tracing failed for %r: %s", name, exc)

    @staticmethod
    def _reject(
        report: BugReport,
        status: GroundingStatus,
        reason: str,
        expected: str,
        actual: str = "",
        found_at_line: Optional[int] = None,
        source_changed: bool = False,
    ) -> GroundingResult:
        """
        Build the verdict for a report that failed.

        Args:
            report: The rejected report.
            status: The failure mode.
            reason: Why it failed, in a sentence.
            expected: Evidence the report quoted.
            actual: Source text at the claimed range, when there was
                any.
            found_at_line: Where the evidence really is, when it was
                found elsewhere.
            source_changed: Whether the file has changed since the
                snapshot.

        Returns:
            A rejected GroundingResult.
        """
        return GroundingResult(
            grounded=False,
            status=status,
            reason=reason,
            expected_evidence=expected,
            actual_source=actual,
            match_type=MatchType.NONE,
            file_path=report.file_path,
            line_start=report.line_start,
            line_end=report.line_end,
            found_at_line=found_at_line,
            source_changed=source_changed,
            report=report,
        )
