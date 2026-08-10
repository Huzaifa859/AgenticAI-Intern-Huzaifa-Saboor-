"""
test_json_output.py
====================

Unit tests for deterministic LLM JSON cleanup and extraction helpers.
"""

from __future__ import annotations

import json
import logging

from codebase_assistant.utils.json_output import (
    JSON_OBJECT_RESPONSE_FORMAT,
    extract_json_object,
    extract_json_value,
    json_schema_response_format,
    log_json_parse_outcome,
    strip_markdown_json_fences,
)


def test_strip_markdown_json_fences() -> None:
    """Fenced JSON should reduce to the inner payload only."""
    fenced = '```json\n{"answer": "ok", "findings": []}\n```'
    assert strip_markdown_json_fences(fenced) == '{"answer": "ok", "findings": []}'


def test_strip_plain_fences() -> None:
    """Generic fences without a language tag are also stripped."""
    fenced = '```\n{"summary": "docs"}\n```'
    assert strip_markdown_json_fences(fenced) == '{"summary": "docs"}'


def test_strip_leaves_bare_json_unchanged() -> None:
    """Bare JSON is returned stripped without alteration."""
    bare = '  {"a": 1}  '
    assert strip_markdown_json_fences(bare) == '{"a": 1}'


def test_extract_json_object_from_fenced_payload() -> None:
    """Fence cleanup plus decode yields a dict for valid fenced JSON."""
    payload, error = extract_json_object(
        'Here you go:\n```json\n{"file_path": "a.py", "summary": "ok"}\n```\n'
    )
    assert error == ""
    assert payload == {"file_path": "a.py", "summary": "ok"}


def test_extract_json_value_accepts_array_root() -> None:
    """Analysis may return a bare findings array."""
    value, error = extract_json_value("[{\"bug_type\": \"x\"}]")
    assert error == ""
    assert isinstance(value, list)
    assert value[0]["bug_type"] == "x"


def test_extract_json_object_rejects_array_root() -> None:
    """Documentation/Testing require an object root."""
    payload, error = extract_json_object("[1, 2]")
    assert payload is None
    assert "not an object" in error


def test_extract_fails_on_invalid_json() -> None:
    """Malformed text returns a decode error, not a guessed object."""
    payload, error = extract_json_object("not-json {{{")
    assert payload is None
    assert error


def test_json_object_response_format_constant() -> None:
    """Shared constant matches OpenRouter json_object mode."""
    assert JSON_OBJECT_RESPONSE_FORMAT == {"type": "json_object"}


def test_json_schema_response_format_helper() -> None:
    """Schema helper wraps a JSON Schema for OpenRouter."""
    fmt = json_schema_response_format(
        "demo",
        {"type": "object", "properties": {"a": {"type": "string"}}},
        strict=False,
    )
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "demo"
    assert fmt["json_schema"]["schema"]["type"] == "object"


def test_log_json_parse_outcome_emits_structured_fields(
    caplog: logging.LogCaptureFixture,
) -> None:
    """Reliability logs include agent, stage, and success."""
    caplog.set_level(logging.INFO)
    log_json_parse_outcome(agent="documentation", stage="initial", success=True)
    log_json_parse_outcome(
        agent="documentation",
        stage="repair",
        success=False,
        error="JSON decode error",
    )
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "json_parse agent=documentation stage=initial success=true" in text
    assert "json_parse agent=documentation stage=repair success=false" in text
    assert "JSON decode error" in text


def test_round_trip_valid_structured_object() -> None:
    """Valid structured output parses without repair."""
    body = {
        "summary": "ok",
        "generated_tests": {"test_a.py": "def test_a():\n    assert True\n"},
        "coverage_estimate": 0.5,
    }
    payload, error = extract_json_object(json.dumps(body))
    assert error == ""
    assert payload == body
