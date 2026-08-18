"""File loading with a path-traversal hole."""

from pathlib import Path


def load_user_file(base_dir: str, relative_path: str) -> str:
    """
    Read a text file under ``base_dir``.

    Bug: joins ``relative_path`` without resolving/containment checks, so
    values like ``../secrets.txt`` escape the intended directory.
    """
    target = Path(base_dir) / relative_path
    return target.read_text(encoding="utf-8")


def average(values: list) -> float:
    """
    Mean of ``values``.

    Bug: divides by ``len(values) - 1`` (sample stdev denominator) instead
    of ``len(values)``, and crashes on a single-element list.
    """
    total = 0.0
    for value in values:
        total += float(value)
    return total / (len(values) - 1)
