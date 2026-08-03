#!/usr/bin/env python3
"""
run_benchmark.py
================

CLI entry point for the Codebase Assistant evaluation benchmark.

Examples (from the Project/ directory):

    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py examples/demo_repo
    python benchmarks/run_benchmark.py https://github.com/user/repo
    python benchmarks/run_benchmark.py --dataset
    python benchmarks/run_benchmark.py examples/demo_repo --mode live
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.runner import BenchmarkRunner, format_console_summary  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible evaluation benchmark against local or "
            "GitHub repositories using the existing Supervisor pipelines."
        )
    )
    parser.add_argument(
        "repository",
        nargs="?",
        help=(
            "Local path or GitHub HTTPS URL. Omit with --dataset to run "
            "the built-in fixture set."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Run the built-in benchmark dataset under examples/.",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "live"),
        default="offline",
        help=(
            "offline (default) uses a deterministic mock LLM for "
            "reproducible docs/testing metrics; live uses configured providers."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "benchmarks" / "results"),
        help="Directory for JSON/CSV reports (default: benchmarks/results).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV export.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable info logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.dataset and not args.repository:
        # Default convenience: built-in dataset.
        args.dataset = True

    runner = BenchmarkRunner(
        mode=args.mode,
        results_dir=Path(args.results_dir),
    )
    try:
        if args.dataset and args.repository:
            print(
                "Use either --dataset or a repository reference, not both.",
                file=sys.stderr,
            )
            return 2
        if args.dataset:
            report = runner.run_dataset()
        else:
            report = runner.run_reference(args.repository)
        outputs = runner.export(report, write_csv=not args.no_csv)
    finally:
        runner.cleanup()

    print(format_console_summary(report))
    print()
    print("Reports written:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
