"""
text_cleanup.py
================

Small shared cleanups for freeform documentation markdown.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

#: JS ``[object Object]`` placeholders. Models emit several spellings:
#: bracketed, markdown-escaped ``\[object Object\]``, NBSP/ZWSP inside,
#: optional ``(url)`` after a markdown link, or the bare ``object Object``
#: token that Streamlit markdown then renders as visible "object Object".
_OBJECT_OBJECT_TOKEN = (
    r"(?:\\?\[\s*)?object[\s\u00a0\u200b\u200c\u200d\u2060]+Object"
    r"(?:\s*\\?\])?(?:\s*\([^)]*\))?"
)
_OBJECT_OBJECT_JUNK = re.compile(
    rf",?\s*(?:{_OBJECT_OBJECT_TOKEN}\s*)+,?",
    flags=re.IGNORECASE,
)
_OBJECT_OBJECT_TOKEN_ONLY = re.compile(
    _OBJECT_OBJECT_TOKEN,
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
    ("\u2122", "(TM)"),
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


#: Invisible / format chars some models insert inside ``[object Object]``.
_INVISIBLE_CHARS = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")

#: Entity-encoded ``[object Object]`` (after HTML escaping of brackets).
_OBJECT_OBJECT_ENTITY = re.compile(
    r",?\s*(?:&#91;|&lsqb;|&lbrack;)\s*object\s+Object\s*"
    r"(?:&#93;|&rsqb;|&rbrack;)\s*,?",
    flags=re.IGNORECASE,
)


def strip_object_object_junk(text: str) -> str:
    """
    Remove JS-style ``[object Object]`` placeholders from model text.

    Kept separate from :func:`sanitize_documentation_text` so the live
    streaming buffer (accumulated across chunk boundaries for all three
    agents) can be cleaned without the heavier punctuation/mojibake pass.
    Handles single, consecutive, escaped, entity-encoded, and stray
    occurrences, including junk that sits inside fenced code samples.
    """
    if not text:
        return text
    cleaned = _INVISIBLE_CHARS.sub("", text)
    for _ in range(16):
        prev = cleaned
        # Comma-aware patterns first so ",[object Object]," is removed as a
        # unit. Bare literal replaces would leave orphan commas behind.
        cleaned = _OBJECT_OBJECT_JUNK.sub("\n", cleaned)
        cleaned = re.sub(
            r",?\s*\\?\[\s*object[ \t\xa0]+Object\s*\\?\]\s*,?",
            "\n",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = _OBJECT_OBJECT_ENTITY.sub("\n", cleaned)
        cleaned = _OBJECT_OBJECT_TOKEN_ONLY.sub("", cleaned)
        if cleaned == prev:
            break
    return cleaned


def sanitize_documentation_text(text: str) -> str:
    """
    Strip model junk and normalize fancy / mojibake punctuation to ASCII.

    Safe for both stored agent output and Streamlit display rendering.
    """
    cleaned = strip_object_object_junk(text or "")
    for src, dst in _REPLACEMENTS.items():
        if src in cleaned:
            cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
