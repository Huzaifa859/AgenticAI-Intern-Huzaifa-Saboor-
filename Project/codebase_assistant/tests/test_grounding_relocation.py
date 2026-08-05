"""
test_grounding_relocation.py
============================

Deterministic evidence relocation for GroundingChecker.

Covers exact matches, repaired line numbers, whitespace/comment
normalization, duplicates, missing evidence, and tracing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from codebase_assistant.analysis.grounding_checker import (
    GroundingChecker,
    GroundingStatus,
    MatchType,
)
from codebase_assistant.schemas.schemas import BugReport
from codebase_assistant.tracing.tracer import Tracer


def _write(repo: Path, relative: str, content: str) -> str:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return relative.replace("\\", "/")


def _report(
    file_path: str,
    evidence: str,
    line_start: int,
    line_end: Optional[int] = None,
    **overrides: Any,
) -> BugReport:
    payload = dict(
        bug_type="test_bug",
        description="test finding",
        severity="medium",
        confidence=0.8,
        file_path=file_path,
        function_name="f",
        line_start=line_start,
        line_end=line_end if line_end is not None else line_start,
        evidence=evidence,
        detection_method="llm",
    )
    payload.update(overrides)
    return BugReport(**payload)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return tmp_path


def test_exact_line_match_unchanged(repo: Path) -> None:
    relative = _write(
        repo,
        "mod.py",
        "def f():\n    return 1\n",
    )
    evidence = "    return 1"
    report = _report(relative, evidence, line_start=2, line_end=2)
    checker = GroundingChecker(workspace_root=str(repo))

    result = checker.verify_report(report)

    assert result.grounded is True
    assert result.match_type == MatchType.EXACT
    assert report.line_start == 2
    assert report.metadata.get("evidence_relocated") is not True
    assert result.metadata.get("evidence_relocated") is not True


def test_shifted_lines_repaired_successfully(repo: Path) -> None:
    relative = _write(
        repo,
        "wallet.py",
        "class Wallet:\n"
        "    def withdraw(self, amount):\n"
        "        balance = balance - amount\n"
        "        return balance\n",
    )
    evidence = "        balance = balance - amount"
    report = _report(relative, evidence, line_start=10, line_end=10, confidence=0.73)
    checker = GroundingChecker(workspace_root=str(repo))

    result = checker.verify_report(report)

    assert result.grounded is True
    assert report.line_start == 3
    assert report.line_end == 3
    assert report.confidence == 0.73
    assert report.metadata["evidence_relocated"] is True
    assert report.metadata["original_lines"] == [10, 10]
    assert report.metadata["relocated_lines"] == [3, 3]
    assert result.metadata["evidence_relocated"] is True


def test_whitespace_only_differences_repaired(repo: Path) -> None:
    relative = _write(repo, "ws.py", "value = 1\n")
    # Trailing spaces in evidence; real file has none.
    report = _report(relative, "value = 1   ", line_start=5, line_end=5)
    checker = GroundingChecker(workspace_root=str(repo))

    result = checker.verify_report(report)

    assert result.grounded is True
    assert report.line_start == 1
    assert report.metadata["evidence_relocated"] is True


def test_comment_only_differences_repaired(repo: Path) -> None:
    relative = _write(
        repo,
        "comments.py",
        "def f():\n    x = 1  # keep me\n    return x\n",
    )
    report = _report(relative, "    x = 1", line_start=9, line_end=9)
    checker = GroundingChecker(workspace_root=str(repo))

    result = checker.verify_report(report)

    assert result.grounded is True
    assert report.line_start == 2
    assert report.metadata["evidence_relocated"] is True
    assert result.match_type in {
        MatchType.COMMENT_STRIPPED,
        MatchType.NORMALIZED,
        MatchType.DEDENTED,
        MatchType.EXACT,
    }


def test_relocated_evidence_retained_in_batch(repo: Path) -> None:
    relative = _write(
        repo,
        "batch.py",
        "a = 1\nb = 2\nc = 3\n",
    )
    good = _report(relative, "b = 2", line_start=99, line_end=99)
    bad = _report(relative, "missing_call()", line_start=1, line_end=1)
    checker = GroundingChecker(workspace_root=str(repo))

    summary = checker.verify_reports([good, bad])

    assert len(summary.grounded) == 1
    assert summary.grounded[0].line_start == 2
    assert summary.grounded[0].evidence == "b = 2"
    assert len(summary.rejected) == 1


def test_missing_evidence_rejected(repo: Path) -> None:
    relative = _write(repo, "ok.py", "def f():\n    return 1\n")
    report = _report(relative, "    totally_invented()", line_start=2)
    checker = GroundingChecker(workspace_root=str(repo))

    result = checker.verify_report(report)

    assert result.grounded is False
    assert result.status == GroundingStatus.EVIDENCE_MISMATCH


def test_wrong_file_rejected(repo: Path) -> None:
    _write(repo, "real.py", "def f():\n    return 1\n")
    report = _report("missing.py", "    return 1", line_start=2)
    checker = GroundingChecker(workspace_root=str(repo))

    result = checker.verify_report(report)

    assert result.grounded is False
    assert result.status == GroundingStatus.FILE_MISSING


def test_duplicate_snippets_choose_closest_deterministically(repo: Path) -> None:
    relative = _write(
        repo,
        "dup.py",
        "x = 1\n"
        "y = 2\n"
        "x = 1\n"
        "z = 3\n"
        "x = 1\n",
    )
    # Claim near the middle occurrence.
    report = _report(relative, "x = 1", line_start=4, line_end=4)
    checker = GroundingChecker(workspace_root=str(repo))

    result = checker.verify_report(report)

    assert result.grounded is True
    assert report.line_start == 3
    assert report.metadata["evidence_relocated"] is True


def test_tracing_emitted_for_relocation(repo: Path) -> None:
    relative = _write(repo, "trace.py", "alpha = 1\nbeta = 2\n")
    report = _report(relative, "beta = 2", line_start=50, line_end=50)
    tracer = Tracer(run_id="grounding-relocation")
    checker = GroundingChecker(workspace_root=str(repo), tracer=tracer)

    summary = checker.verify_reports([report])

    assert summary.grounded
    names = [event.name for event in tracer.get_events()]
    assert "evidence_search_started" in names
    assert "evidence_relocated" in names
    assert "grounding_completed" in names
    relocated = next(e for e in tracer.get_events() if e.name == "evidence_relocated")
    assert relocated.metadata["file"] == relative
    assert relocated.metadata["original_lines"] == [50, 50]
    assert relocated.metadata["relocated_lines"] == [2, 2]
    assert relocated.success is True


def test_tracing_emitted_on_relocation_failure(repo: Path) -> None:
    relative = _write(repo, "trace_fail.py", "alpha = 1\n")
    report = _report(relative, "missing = 0", line_start=1, line_end=1)
    tracer = Tracer(run_id="grounding-fail")
    checker = GroundingChecker(workspace_root=str(repo), tracer=tracer)

    result = checker.verify_report(report)

    assert result.grounded is False
    names = [event.name for event in tracer.get_events()]
    assert "evidence_search_started" in names
    assert "evidence_relocation_failed" in names


def test_system_prompt_requires_accurate_citations() -> None:
    from codebase_assistant.agents.code_analysis_agent import SYSTEM_PROMPT

    assert "Report only findings supported by the repository" in SYSTEM_PROMPT
    assert "Never invent files, functions, variables, or evidence" in SYSTEM_PROMPT
    assert "If you are uncertain, omit the finding" in SYSTEM_PROMPT
    assert '"findings"' in SYSTEM_PROMPT
