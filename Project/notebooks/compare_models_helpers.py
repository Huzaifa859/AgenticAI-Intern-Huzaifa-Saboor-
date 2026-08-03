"""
compare_models_helpers.py
=========================

Evaluation-only helpers for the multi-model comparison notebook.

Does not modify production agent pipelines. Builds isolated Supervisors
and injects a single-model OpenRouter client per comparison run.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

from codebase_assistant.config import Config
from codebase_assistant.models.model_client import LLMClient
from codebase_assistant.models.providers.openrouter_provider import OpenRouterProvider
from codebase_assistant.schemas.schemas import ModelMessage, ModelResponse
from codebase_assistant.supervisor import Supervisor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Display name -> OpenRouter slug.
COMPARISON_MODELS: Dict[str, str] = {
    "Claude Sonnet 4": "anthropic/claude-sonnet-4",
    "Llama 3.1 8B Instruct": "meta-llama/llama-3.1-8b-instruct",
    "Gemma 3 27B IT": "google/gemma-3-27b-it",
    "Nemotron Nano 9B V2": "nvidia/nemotron-nano-9b-v2",
}

DEFAULT_REPO = PROJECT_ROOT / "examples" / "demo_repo"
ALT_REPO = PROJECT_ROOT / "examples" / "medium_repo"


class SingleModelOpenRouter(OpenRouterProvider):
    """
    OpenRouter provider that never walks the production fallback chain.

    Evaluation-only subclass so each notebook row is attributable to one
    model slug.
    """

    def _model_chain(self, primary: str) -> List[str]:
        model = str(primary or self.model or "").strip()
        return [model] if model else []


class UsageTrackingClient(LLMClient):
    """LLMClient wrapper that accumulates token usage and call latency."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reset_usage()

    def reset_usage(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.latencies_ms: List[float] = []
        self.errors: List[str] = []

    def generate(
        self, messages: Sequence[ModelMessage], **options: Any
    ) -> ModelResponse:
        started = time.perf_counter()
        try:
            response = super().generate(messages, **options)
        except Exception as exc:
            self.errors.append(str(exc))
            raise
        finally:
            self.latencies_ms.append((time.perf_counter() - started) * 1000.0)

        usage = getattr(response, "usage", None) or {}
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        total = int(usage.get("total_tokens") or (prompt + completion))
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.call_count += 1
        return response

    def usage_snapshot(self) -> Dict[str, Any]:
        latency = sum(self.latencies_ms)
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_latency_ms": round(latency, 3),
            "model_calls": self.call_count,
            "errors": list(self.errors),
        }


@dataclass
class ModelRunResult:
    """Metrics collected for one model across analysis/docs/testing."""

    display_name: str
    model_slug: str
    available: bool = True
    skip_reason: str = ""
    total_runtime_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    grounded_findings: int = 0
    hallucinations_rejected: int = 0
    documentation_length: int = 0
    generated_test_files: int = 0
    tests_passed: int = 0
    abstentions: int = 0
    analysis_runtime_seconds: float = 0.0
    documentation_runtime_seconds: float = 0.0
    testing_runtime_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        if not self.available:
            return {
                "Model": self.display_name,
                "Slug": self.model_slug,
                "Available": False,
                "Skip reason": self.skip_reason,
                "Runtime (s)": None,
                "Prompt tokens": None,
                "Completion tokens": None,
                "Total tokens": None,
                "Latency (ms)": None,
                "Grounded findings": None,
                "Hallucinations rejected": None,
                "Docs length": None,
                "Test files": None,
                "Tests passed": None,
                "Abstentions": None,
            }
        return {
            "Model": self.display_name,
            "Slug": self.model_slug,
            "Available": True,
            "Skip reason": "",
            "Runtime (s)": round(self.total_runtime_seconds, 3),
            "Prompt tokens": self.prompt_tokens,
            "Completion tokens": self.completion_tokens,
            "Total tokens": self.total_tokens,
            "Latency (ms)": round(self.latency_ms, 1),
            "Grounded findings": self.grounded_findings,
            "Hallucinations rejected": self.hallucinations_rejected,
            "Docs length": self.documentation_length,
            "Test files": self.generated_test_files,
            "Tests passed": self.tests_passed,
            "Abstentions": self.abstentions,
        }


def resolve_repository(choice: str = "demo") -> Path:
    """Return an absolute path to the shared comparison repository."""
    normalized = (choice or "demo").strip().lower()
    if normalized in {"medium", "medium_repo", "examples/medium_repo"}:
        path = ALT_REPO
    else:
        path = DEFAULT_REPO
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Comparison repository not found: {path}")
    return path


def probe_model(model_slug: str, config: Optional[Config] = None) -> Optional[str]:
    """
    Return a skip reason when the model cannot be used, else None.
    """
    cfg = config or Config.load()
    if not cfg.openrouter_api_key:
        return "OPENROUTER_API_KEY is not configured."
    provider = SingleModelOpenRouter(model=model_slug, config=cfg)
    if not provider.is_available():
        return "OpenRouter availability probe failed for this credential/network."
    return None


def build_supervisor_for_model(
    model_slug: str,
    *,
    work_dir: Path,
    config: Optional[Config] = None,
) -> tuple[Supervisor, UsageTrackingClient]:
    """
    Build an isolated Supervisor and inject a pinned-model tracking client.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    chroma = str(work_dir / "chroma")
    memory = str(work_dir / "memory")
    os.environ["CHROMA_PERSIST_DIR"] = chroma
    os.environ["MEMORY_STORE_PATH"] = memory
    os.environ["WORKSPACE_ROOT"] = str(work_dir)
    os.environ["RERANK_ENABLED"] = "false"

    cfg = config or Config.load()
    cfg.chroma_persist_directory = chroma
    cfg.memory_store_path = memory
    cfg.workspace_root = str(work_dir)
    cfg.rerank_enabled = False
    cfg.openrouter_model = model_slug
    cfg.claude_model = model_slug
    cfg.model_name = model_slug

    with patch.object(Supervisor, "_init_openrouter_provider", return_value=None), patch.object(
        Supervisor, "_init_ollama_provider", return_value=None
    ):
        supervisor = Supervisor(config=cfg)

    provider = SingleModelOpenRouter(model=model_slug, config=cfg)
    client = UsageTrackingClient(
        model_name=model_slug,
        max_tokens=cfg.max_tokens,
        provider=provider,
        config=cfg,
    )
    for agent in supervisor.agents.values():
        agent.model_client = client
    return supervisor, client


def _parse_passed_tests(summary: str) -> int:
    text = str(summary or "")
    marker = "passed"
    if "Execution:" not in text:
        return 0
    # "Execution: 1 passed, 0 failed, ..."
    try:
        section = text.split("Execution:", 1)[1].strip()
        number = section.split("passed", 1)[0].strip().split()[-1]
        return int(number)
    except Exception:
        return 0


def run_model_comparison(
    display_name: str,
    model_slug: str,
    repository: Path,
    *,
    work_root: Optional[Path] = None,
) -> ModelRunResult:
    """
    Run analysis, documentation, and testing for one pinned model.
    """
    skip = probe_model(model_slug)
    if skip:
        return ModelRunResult(
            display_name=display_name,
            model_slug=model_slug,
            available=False,
            skip_reason=skip,
        )

    root = Path(work_root or (PROJECT_ROOT / "notebooks" / ".compare_work"))
    case_dir = root / uuid.uuid4().hex[:10]
    supervisor, client = build_supervisor_for_model(model_slug, work_dir=case_dir)
    repo = str(repository.resolve())
    result = ModelRunResult(display_name=display_name, model_slug=model_slug)
    pipeline_started = time.perf_counter()

    # Analysis
    client.reset_usage()
    started = time.perf_counter()
    analysis_response = supervisor.handle_task(
        "analysis: Find likely bugs and correctness problems in this code.",
        repo,
    )
    result.analysis_runtime_seconds = time.perf_counter() - started
    analysis_usage = client.usage_snapshot()
    report = analysis_response.output
    if report is not None:
        result.grounded_findings = len(getattr(report, "findings", []) or [])
        result.hallucinations_rejected = len(getattr(report, "rejected", []) or [])
        if getattr(report, "abstention", None) is not None:
            result.abstentions += 1
    else:
        result.notes.append("Analysis returned no report object.")

    # Documentation
    client.reset_usage()
    started = time.perf_counter()
    documentation_response = supervisor.handle_task("documentation README", repo)
    result.documentation_runtime_seconds = time.perf_counter() - started
    docs_usage = client.usage_snapshot()
    docs = documentation_response.output
    if docs is not None:
        result.documentation_length = len(str(getattr(docs, "summary", "") or ""))
        if getattr(docs, "abstention", None) is not None:
            result.abstentions += 1
    else:
        result.notes.append("Documentation returned no result object.")

    # Testing
    client.reset_usage()
    started = time.perf_counter()
    testing_response = supervisor.handle_task("testing", repo)
    result.testing_runtime_seconds = time.perf_counter() - started
    test_usage = client.usage_snapshot()
    tests = testing_response.output
    if tests is not None:
        generated = dict(getattr(tests, "generated_tests", None) or {})
        result.generated_test_files = len(generated)
        result.tests_passed = _parse_passed_tests(str(getattr(tests, "summary", "") or ""))
        if getattr(tests, "abstention", None) is not None:
            result.abstentions += 1
    else:
        result.notes.append("Testing returned no result object.")

    result.total_runtime_seconds = time.perf_counter() - pipeline_started
    result.prompt_tokens = (
        analysis_usage["prompt_tokens"]
        + docs_usage["prompt_tokens"]
        + test_usage["prompt_tokens"]
    )
    result.completion_tokens = (
        analysis_usage["completion_tokens"]
        + docs_usage["completion_tokens"]
        + test_usage["completion_tokens"]
    )
    result.total_tokens = (
        analysis_usage["total_tokens"]
        + docs_usage["total_tokens"]
        + test_usage["total_tokens"]
    )
    result.latency_ms = (
        analysis_usage["model_latency_ms"]
        + docs_usage["model_latency_ms"]
        + test_usage["model_latency_ms"]
    )
    for bucket_name, bucket in (
        ("analysis", analysis_usage),
        ("documentation", docs_usage),
        ("testing", test_usage),
    ):
        if bucket.get("errors"):
            result.notes.append(f"{bucket_name} errors: {bucket['errors']}")
    return result


def results_to_rows(results: Sequence[ModelRunResult]) -> List[Dict[str, Any]]:
    """Convert run results to table rows."""
    return [item.to_row() for item in results]


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    """Render a GitHub-flavored markdown table."""
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append("" if value is None else str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def qualitative_discussion(results: Sequence[ModelRunResult]) -> str:
    """Build a short written comparison from collected metrics."""
    available = [item for item in results if item.available]
    if not available:
        return (
            "No models were available for comparison. Configure "
            "`OPENROUTER_API_KEY` and re-run the notebook."
        )

    strongest_analysis = max(
        available, key=lambda item: (item.grounded_findings, -item.hallucinations_rejected)
    )
    best_docs = max(available, key=lambda item: item.documentation_length)
    best_tests = max(
        available, key=lambda item: (item.tests_passed, item.generated_test_files)
    )
    fastest = min(available, key=lambda item: item.total_runtime_seconds)
    cheapest = min(
        available,
        key=lambda item: item.total_tokens if item.total_tokens > 0 else 10**12,
    )
    # Overall: prefer grounded findings, then tests passed, then lower runtime.
    overall = max(
        available,
        key=lambda item: (
            item.grounded_findings,
            item.tests_passed,
            item.documentation_length,
            -item.total_runtime_seconds,
        ),
    )

    lines = [
        f"- **Strongest analysis:** {strongest_analysis.display_name} "
        f"({strongest_analysis.grounded_findings} grounded findings, "
        f"{strongest_analysis.hallucinations_rejected} hallucinations rejected).",
        f"- **Best documentation:** {best_docs.display_name} "
        f"({best_docs.documentation_length} characters).",
        f"- **Best tests:** {best_tests.display_name} "
        f"({best_tests.tests_passed} passed across "
        f"{best_tests.generated_test_files} generated file(s)).",
        f"- **Fastest model:** {fastest.display_name} "
        f"({fastest.total_runtime_seconds:.2f}s total runtime).",
        f"- **Cheapest model (by total tokens):** {cheapest.display_name} "
        f"({cheapest.total_tokens} tokens).",
        f"- **Overall recommendation:** {overall.display_name} - best balance of "
        f"grounded findings, executable tests, and documentation length on this repo.",
    ]
    return "\n".join(lines)
