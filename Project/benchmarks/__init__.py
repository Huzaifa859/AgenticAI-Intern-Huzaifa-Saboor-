"""
benchmarks
==========

Reproducible evaluation suite for Codebase Assistant.

Measures the existing Supervisor / agent pipelines without changing
their production behavior. See ``README.md`` for usage.
"""

from .metrics import (
    extract_analysis_metrics,
    extract_documentation_metrics,
    extract_overall_metrics,
    extract_rag_metrics,
    extract_testing_metrics,
)
from .runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "extract_analysis_metrics",
    "extract_documentation_metrics",
    "extract_testing_metrics",
    "extract_rag_metrics",
    "extract_overall_metrics",
]
