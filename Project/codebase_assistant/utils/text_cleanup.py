"""
text_cleanup.py
================

Small shared cleanups for freeform documentation markdown.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

#: JS-style placeholders some free models emit inside prose/code samples.
_OBJECT_OBJECT_JUNK = re.compile(
    r",?\s*\[object Object\]\s*,?",
    flags=re.IGNORECASE,
)

#: Fancy Unicode -> ASCII. Also used to derive common mojibake forms.
_UNICODE_TO_ASCII: Tuple[Tuple[str, str], ...] = (
    ("\u2010", "-"),
    ("\u2011", "-"),
    ("\u2012", "-"),
    ("\u2013", "-"),
    ("\u2014", "-"),
    ("\u2015", "-"),
    ("\u2212", "-"),
    ("\u00ad", "-"),
    ("\u00a0", " "),
    ("\u2026", "..."),
    ("\u2018", "'"),
    ("\u2019", "'"),
    ("\u201c", '"'),
    ("\u201d", '"'),
    ("\u2192", "->"),
    ("\u2190", "<-"),
    ("├──", "|--"),
    ("└──", "+--"),
    ("├─", "|-"),
    ("└─", "+-"),
    ("│", "|"),
    ("─", "-"),
    ("┼", "+"),
    ("┬", "+"),
    ("┴", "+"),
    ("┌", "+"),
    ("┐", "+"),
    ("└", "+"),
    ("┘", "+"),
    ("├", "|"),
    ("┤", "|"),
)


def _mojibake_forms(text: str) -> Tuple[str, ...]:
    """Return common mis-decodings of a UTF-8 string (cp1252 / latin-1)."""
    raw = text.encode("utf-8")
    forms = []
    for encoding in ("cp1252", "latin-1"):
        try:
            forms.append(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return tuple(dict.fromkeys(forms))


def _build_replacement_map() -> Dict[str, str]:
    """Map both proper Unicode and mojibake spellings to ASCII."""
    mapping: Dict[str, str] = {}
    for src, dst in _UNICODE_TO_ASCII:
        mapping[src] = dst
        for form in _mojibake_forms(src):
            if form and form != src:
                mapping[form] = dst
    mapping["façade"] = "facade"
    mapping["Façade"] = "Facade"
    for form in _mojibake_forms("façade"):
        mapping[form] = "facade"
    for form in _mojibake_forms("Façade"):
        mapping[form] = "Facade"
    # Longer keys first so ├── wins over ├ / ─ pieces.
    return dict(sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True))


_REPLACEMENTS = _build_replacement_map()


def sanitize_documentation_text(text: str) -> str:
    """
    Strip model junk and normalize fancy / mojibake punctuation to ASCII.

    Safe for both stored agent output and Streamlit display rendering.
    """
    cleaned = text or ""
    cleaned = _OBJECT_OBJECT_JUNK.sub("\n", cleaned)
    cleaned = re.sub(r"\[object Object\]", "", cleaned, flags=re.IGNORECASE)
    for src, dst in _REPLACEMENTS.items():
        if src in cleaned:
            cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
