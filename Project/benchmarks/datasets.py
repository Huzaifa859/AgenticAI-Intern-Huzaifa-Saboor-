"""
datasets.py
===========

Built-in benchmark repository descriptors and fixture materialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"


@dataclass(frozen=True)
class BenchmarkCase:
    """One repository case in the benchmark dataset."""

    name: str
    path: Path
    description: str
    expect_findings: bool
    supported: bool = True


def ensure_fixture_repositories() -> List[BenchmarkCase]:
    """
    Ensure built-in fixture repositories exist and return their cases.

    Creates missing fixtures under ``examples/`` so the suite is
    self-contained after clone.
    """
    demo = EXAMPLES_ROOT / "demo_repo"
    clean = EXAMPLES_ROOT / "clean_repo"
    unsupported = EXAMPLES_ROOT / "unsupported_repo"
    medium = EXAMPLES_ROOT / "medium_repo"

    _write_if_missing(
        clean / "math_utils.py",
        '"""Clean helpers with no seeded defects."""\n\n'
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n"
        "    return a * b\n",
    )
    _write_if_missing(
        unsupported / "README.md",
        "# Unsupported fixture\n\nMarkdown only; no Python sources.\n",
    )
    _write_if_missing(
        unsupported / "notes.txt",
        "This repository intentionally has no .py files.\n",
    )
    _write_medium_repo(medium)

    cases = [
        BenchmarkCase(
            name="demo_repo",
            path=demo,
            description="Seeded buggy demo repository.",
            expect_findings=True,
            supported=True,
        ),
        BenchmarkCase(
            name="medium_repo",
            path=medium,
            description="Medium multi-module Python package.",
            expect_findings=True,
            supported=True,
        ),
        BenchmarkCase(
            name="clean_repo",
            path=clean,
            description="Small clean Python repository with almost no findings.",
            expect_findings=False,
            supported=True,
        ),
        BenchmarkCase(
            name="unsupported_repo",
            path=unsupported,
            description="Non-Python repository for abstention / unsupported coverage.",
            expect_findings=False,
            supported=False,
        ),
    ]
    return [case for case in cases if case.path.exists()]


def _write_if_missing(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _write_medium_repo(root: Path) -> None:
    """Create a slightly larger multi-file package when absent."""
    package = root / "samplelib"
    _write_if_missing(
        package / "__init__.py",
        '"""Sample library used by the benchmark suite."""\n',
    )
    _write_if_missing(
        package / "ops.py",
        "def normalize(values):\n"
        "    total = sum(values) or 1\n"
        "    return [value / total for value in values]\n\n"
        "def clamp(value, low=0, high=1):\n"
        "    if value < low:\n"
        "        return low\n"
        "    if value > high:\n"
        "        return high\n"
        "    return value\n",
    )
    _write_if_missing(
        package / "io_utils.py",
        "from typing import Iterable\n\n"
        "def join_lines(lines: Iterable[str]) -> str:\n"
        "    return '\\n'.join(lines)\n\n"
        "def read_label(path: str) -> str:\n"
        "    unused = 123\n"
        "    return path.split('/')[-1]\n",
    )
    _write_if_missing(
        package / "service.py",
        "from .ops import clamp, normalize\n\n"
        "class Scorer:\n"
        "    def score(self, values):\n"
        "        weights = normalize(values)\n"
        "        return clamp(sum(weights))\n\n"
        "def broken_call():\n"
        "    return clamp(0.5, 0, 1, 2)\n",
    )
    _write_if_missing(
        root / "README.md",
        "# medium_repo\n\nMulti-module fixture for benchmark timing.\n",
    )
