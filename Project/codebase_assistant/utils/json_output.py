"""
json_output.py
==============

Deterministic helpers for LLM JSON responses.

Used by Analysis / Documentation / Testing before Pydantic / schema
construction. Does not invent fields — only strips harmless Markdown
fences and extracts JSON values already present in the text.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: OpenRouter / OpenAI-compatible JSON object mode.
JSON_OBJECT_RESPONSE_FORMAT: Dict[str, str] = {"type": "json_object"}

#: Fenced ```json ... ``` or ``` ... ``` block.
_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def json_schema_response_format(
    name: str, schema: Dict[str, Any], *, strict: bool = False
) -> Dict[str, Any]:
    """
    Build an OpenRouter ``response_format`` for JSON Schema mode.

    Prefer :data:`JSON_OBJECT_RESPONSE_FORMAT` when broad model
    compatibility matters more than strict schema enforcement.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": schema,
        },
    }


def strip_markdown_json_fences(text: str) -> str:
    """
    Remove a surrounding Markdown code fence when present.

    If multiple fences exist, prefer the first ``json`` fence, else the
    first fenced block. Bare text is returned unchanged (stripped).
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    json_fence = re.search(
        r"```json\s*([\s\S]*?)```", raw, flags=re.IGNORECASE
    )
    if json_fence:
        return (json_fence.group(1) or "").strip()

    any_fence = _FENCE.search(raw)
    if any_fence:
        return (any_fence.group(1) or "").strip()
    return raw


def extract_json_value(text: str) -> Tuple[Optional[Any], str]:
    """
    Decode a JSON value from model text.

    Order: whole string (after fence strip) → first balanced ``{...}``
    or ``[...]``. Returns ``(value, "")`` on success or
    ``(None, error)`` on failure.
    """
    cleaned = strip_markdown_json_fences(text)
    if not cleaned:
        return None, "Model response was empty."

    try:
        return json.loads(cleaned), ""
    except json.JSONDecodeError as exc:
        last_error = f"JSON decode error: {exc}"

    for opener, closer in (("{", "}"), ("[", "]")):
        value, err = _balanced_slice(cleaned, opener, closer)
        if value is not None:
            return value, ""
        if err:
            last_error = err
    return None, last_error


def extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Decode a JSON object from model text.

    Returns ``(object, "")`` or ``(None, error)``. Non-object JSON roots
    are rejected.
    """
    value, error = extract_json_value(text)
    if value is None:
        return None, error
    if not isinstance(value, dict):
        return None, "JSON root value was not an object."
    return value, ""


def _balanced_slice(
    text: str, opener: str, closer: str
) -> Tuple[Optional[Any], str]:
    start = text.find(opener)
    if start < 0:
        return None, ""

    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                snippet = text[start : position + 1]
                try:
                    return json.loads(snippet), ""
                except json.JSONDecodeError as exc:
                    return None, f"JSON decode error: {exc}"
    return None, f"Unbalanced JSON starting with {opener!r}."


def log_json_parse_outcome(
    *,
    agent: str,
    stage: str,
    success: bool,
    error: str = "",
) -> None:
    """Emit a structured log line for measuring JSON reliability."""
    if success:
        logger.info("json_parse agent=%s stage=%s success=true", agent, stage)
    else:
        logger.warning(
            "json_parse agent=%s stage=%s success=false error=%s",
            agent,
            stage,
            (error or "unknown")[:300],
        )
