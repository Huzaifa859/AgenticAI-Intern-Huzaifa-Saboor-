"""
test_week6_integration.py
===========================

End-to-end integration test for the Week 6 analysis pipeline.

Exercises:

    Supervisor
    -> CodeAnalysisAgent
    -> StaticAnalyzer
    -> GroundingChecker

No LLM provider is required. Retrieval and indexing are disabled so the
run stays deterministic and fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codebase_assistant.agents.code_analysis_agent import (
    CodeAnalysisAgent,
    CodeAnalysisReport,
)
from codebase_assistant.analysis.grounding_checker import GroundingChecker
from codebase_assistant.config import Config
from codebase_assistant.schemas.schemas import AgentType
from codebase_assistant.supervisor import Supervisor

# Bug categories planted in the temporary repository.
REQUIRED_BUG_TYPES = {
    "unused_import",
    "undefined_variable",
    "syntax_error",
    "unreachable_code",
}


@pytest.fixture
def buggy_repository(tmp_path: Path) -> Path:
    """
    Build a temporary repository containing several deterministic defects.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to the repository root.
    """
    (tmp_path / "src").mkdir()

    (tmp_path / "src" / "clean.py").write_text(
        '"""A well-behaved module."""\n\n'
        "def add(a, b):\n"
        '    """Return the sum."""\n'
        "    return a + b\n",
        encoding="utf-8",
    )

    (tmp_path / "src" / "issues.py").write_text(
        '"""Module with several static-analysis defects."""\n\n'
        "import os\n"
        "import json\n\n\n"
        "def compute_total(items, tax_rate):\n"
        '    """Sum items and apply tax."""\n'
        "    subtotal = 0\n"
        "    for item in items:\n"
        "        subtotal += item\n"
        "    return subtotal * (1 + tax_rate)\n"
        "    print('done')\n\n\n"
        "def report(values):\n"
        '    """Report a total."""\n'
        "    total = compute_total(values)\n"
        "    return total\n\n\n"
        "def risky():\n"
        '    """Call something undefined."""\n'
        "    return undefined_helper()\n",
        encoding="utf-8",
    )

    (tmp_path / "src" / "broken.py").write_text(
        "def ok():\n"
        "    return 1\n\n\n"
        "def broken(:\n"
        "    this is not valid python\n",
        encoding="utf-8",
    )

    return tmp_path


def test_week6_pipeline_produces_grounded_report(buggy_repository: Path) -> None:
    """
    Run the Week 6 pipeline on a seeded repository without an LLM.

    Verifies that the Supervisor wires the CodeAnalysisAgent, static
    analysis produces findings for each planted defect category, and
    every surviving finding passes the grounding check.
    """
    repo = str(buggy_repository.resolve())

    supervisor = Supervisor(config=Config.load())
    agent = supervisor.agents[AgentType.CODE_ANALYSIS]
    assert isinstance(agent, CodeAnalysisAgent)

    report = agent.analyze_repository(
        repository_path=repo,
        question="Find likely bugs.",
        use_rag=False,
    )

    assert isinstance(report, CodeAnalysisReport)
    assert report.repository_path == repo
    assert report.duration_seconds >= 0.0
    assert report.model_used is False
    assert report.static_report is not None
    assert report.static_report.files_analyzed >= 2

    assert report.findings, "expected at least one verified finding"
    assert len(report.static_findings) == len(report.findings)
    assert len(report.llm_findings) == 0

    found_types = {finding.bug_type for finding in report.findings}
    missing = REQUIRED_BUG_TYPES - found_types
    assert not missing, f"missing expected bug types: {sorted(missing)}"

    checker = GroundingChecker(workspace_root=repo)
    verification = checker.verify_reports(report.findings)
    assert len(verification.grounded) == len(report.findings)
    assert verification.rejected == []

    for finding in report.findings:
        assert finding.detection_method == "static"
        assert finding.evidence.strip()
        assert finding.file_path
        assert finding.line_start >= 1
        assert finding.line_end >= finding.line_start
        assert finding.confidence > 0.0

    for result in verification.results:
        assert result.grounded is True
        assert result.actual_source == finding_evidence_slice(
            repo, result.file_path, result.line_start, result.line_end
        )


def finding_evidence_slice(
    repo_root: str, file_path: str, line_start: int, line_end: int
) -> str:
    """
    Read the exact source slice a finding claims to quote.

    Args:
        repo_root: Repository root directory.
        file_path: Relative path inside the repository.
        line_start: First line of the slice.
        line_end: Last line of the slice.

    Returns:
        The source text at the requested range.
    """
    source = (Path(repo_root) / file_path).read_text(encoding="utf-8")
    lines = source.split("\n")
    return "\n".join(lines[line_start - 1 : line_end])
