from typing import Iterable


def join_lines(lines: Iterable[str]) -> str:
    return "\n".join(lines)


def read_label(path: str) -> str:
    unused = 123
    return path.split("/")[-1]
