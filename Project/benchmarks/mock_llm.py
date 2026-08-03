"""
mock_llm.py
===========

Deterministic offline LLM client for reproducible benchmarks.

Does not modify production providers. The runner injects this client
into Supervisor agents when ``--mode offline`` is selected.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from codebase_assistant.schemas.schemas import ModelMessage, ModelResponse


class BenchmarkLLMClient:
    """
    Available fake LLM client with stable JSON responses.

    Response selection is based on prompt keywords so the same client
    can serve analysis, documentation, and testing agents.
    """

    def __init__(self, repo_hint: str = "benchmark") -> None:
        self.repo_hint = repo_hint
        self.calls: List[Dict[str, Any]] = []

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[ModelMessage],
        **kwargs: Any,
    ) -> ModelResponse:
        joined = " ".join(str(getattr(message, "content", "")) for message in messages)
        lowered = joined.lower()
        self.calls.append({"kwargs": dict(kwargs), "chars": len(joined)})

        if "pytest" in lowered or "unit test" in lowered or "generated_tests" in lowered:
            payload = {
                "summary": f"Generated benchmark tests for {self.repo_hint}.",
                "generated_tests": {
                    "test_benchmark_smoke.py": (
                        "def test_benchmark_smoke():\n"
                        "    assert 1 + 1 == 2\n"
                    )
                },
                "coverage_estimate": 0.25,
            }
        elif "readme" in lowered or "documentationresult" in lowered or "docstring" in lowered:
            payload = {
                "file_path": "README.md",
                "function_name": "README",
                "summary": (
                    f"# {self.repo_hint}\n\n"
                    "Benchmark documentation summary generated offline."
                ),
                "parameters": [],
                "returns": "",
                "example_usage": "",
            }
        else:
            payload = {
                "answer": "No additional LLM findings beyond the static pass.",
                "findings": [],
            }
        return ModelResponse(content=json.dumps(payload), usage={}, raw={"mock": True})
