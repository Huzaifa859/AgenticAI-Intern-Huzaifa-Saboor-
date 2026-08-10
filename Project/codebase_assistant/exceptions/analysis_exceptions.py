"""
analysis_exceptions.py
=======================

Errors raised by the `analysis` package — the static pass, the
grounding check, and report assembly.

TODO: Raise these from `analysis/` once it is implemented.
"""

from __future__ import annotations

from .base import CodebaseAssistantError


class AnalysisError(CodebaseAssistantError):
    """Base class for every analysis-stage failure."""


class StaticAnalysisError(AnalysisError):
    """The deterministic static pass could not complete on a file."""


class SourceParseError(AnalysisError):
    """
    A file could not be parsed into an AST.

    Distinct from a *reported* syntax error: this means analysis itself
    could not proceed, whereas a detected syntax error is a legitimate
    finding returned as a BugReport.
    """


class GroundingVerificationError(AnalysisError):
    """
    The grounding check could not be carried out.

    Raised when verification is impossible (for example the cited file
    has become unreadable), not when a claim simply fails to match —
    an unverified claim is discarded rather than raised.
    """


class InsufficientContextError(AnalysisError):
    """
    Not enough retrieved context to reach a conclusion.

    Backs the proposal's "explicit abstention over guessing" path.
    """
