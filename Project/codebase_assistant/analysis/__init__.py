"""
analysis
========

The grounded bug-detection pipeline described in the proposal's "Bug
Detection Approach & Hallucination Mitigation" section.

This package is deliberately separate from `agents/`. The Code Analysis
Agent decides *what* to investigate; this package decides *what is true*
and is answerable on its own, without a model in the loop.

The pipeline runs in three stages:

1. StaticAnalyzer — deterministic `pyflakes`/`ast` pass that runs before
   any LLM involvement and cannot hallucinate.
2. GroundingChecker — mechanically verifies that the code quoted as
   evidence really exists at the claimed file and line, discarding any
   claim that fails.
3. ReportBuilder — assembles verified findings into BugReport objects,
   labelling each with its detection method and confidence.

NOTE: Placeholder only. No static analysis, verification, or report
assembly is implemented yet.
"""

from .grounding_checker import GroundingChecker
from .report_builder import ReportBuilder
from .static_analyzer import StaticAnalyzer

__all__ = ["StaticAnalyzer", "GroundingChecker", "ReportBuilder"]
