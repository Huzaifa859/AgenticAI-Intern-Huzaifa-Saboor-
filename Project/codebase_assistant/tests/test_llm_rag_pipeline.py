"""
test_llm_rag_pipeline.py
==========================

Week 7 Phase 1 Step 4b — integration tests for the full LLM + RAG
analysis pipeline:

    Repository
    -> Indexer
    -> Retriever
    -> Prompt builder
    -> OpenRouterProvider (HTTP mocked)
    -> parse_response()
    -> GroundingChecker
    -> merged CodeAnalysisReport

Only OpenRouter network calls are mocked. Indexing, embeddings,
retrieval, static analysis, prompting, parsing, grounding, and merging
all execute for real. No Internet access and no real OpenRouter key
are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

from codebase_assistant.agents.code_analysis_agent import (
    CodeAnalysisAgent,
    CodeAnalysisReport,
)
from codebase_assistant.config import Config
from codebase_assistant.models.model_client import LLMClient
from codebase_assistant.models.providers.openrouter_provider import OpenRouterProvider

PROVIDER_GET = "codebase_assistant.models.providers.openrouter_provider.requests.get"
PROVIDER_POST = "codebase_assistant.models.providers.openrouter_provider.requests.post"
PROVIDER_SLEEP = "codebase_assistant.models.providers.openrouter_provider.time.sleep"

WALLET_RELATIVE = "wallet.py"
QUESTION = "Find bugs in withdraw where balance is not validated before subtracting amount."


def _source_slice(repo: Path, relative: str, line_start: int, line_end: int) -> str:
    """Return the exact source text for a 1-based inclusive line range."""
    lines = (repo / relative).read_text(encoding="utf-8").split("\n")
    return "\n".join(lines[line_start - 1 : line_end])


def _http_response(status_code: int, payload: Optional[dict] = None) -> MagicMock:
    """Build a mock requests.Response for OpenRouter."""
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.text = json.dumps(payload) if payload is not None else ""
    if payload is None:
        response.json.side_effect = ValueError("no json")
    else:
        response.json.return_value = payload
    return response


def _chat_payload(content: str) -> dict:
    """OpenRouter chat-completion body wrapping assistant content."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }


def _finding_payload(
    *,
    file_path: str,
    line_start: int,
    line_end: int,
    evidence: str,
    bug_type: str = "missing_validation",
    description: str = "Amount is subtracted without checking available funds.",
) -> str:
    """Serialize a valid model JSON response with one finding."""
    return json.dumps(
        {
            "answer": "Withdraw subtracts without validating the balance.",
            "findings": [
                {
                    "bug_type": bug_type,
                    "description": description,
                    "severity": "high",
                    "confidence": 0.9,
                    "file_path": file_path,
                    "function_name": "withdraw",
                    "line_start": line_start,
                    "line_end": line_end,
                    "evidence": evidence,
                    "suggested_fix": "Reject the withdrawal when amount > balance.",
                }
            ],
        }
    )


@pytest.fixture(scope="module")
def analysis_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Small repository with one static defect and one logic bug for the LLM.

    Static analysis should catch the unused import. The missing balance
    check is a correctness issue meant for the mocked LLM finding.
    """
    root = tmp_path_factory.mktemp("llm_rag_repo")

    (root / "wallet.py").write_text(
        '"""Wallet helpers with an intentional validation gap."""\n'
        "\n"
        "import os\n"
        "\n"
        "\n"
        "def withdraw(balance, amount):\n"
        '    """Withdraw amount from balance without validating funds."""\n'
        "    balance = balance - amount\n"
        "    return balance\n",
        encoding="utf-8",
    )

    (root / "helpers.py").write_text(
        '"""Supporting helpers used to give retrieval more surface area."""\n'
        "\n"
        "\n"
        "def format_money(value):\n"
        '    """Format a numeric amount for display."""\n'
        '    return f"${value:.2f}"\n',
        encoding="utf-8",
    )

    return root


@pytest.fixture(scope="module")
def chroma_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared Chroma persistence root for this module's pipeline runs."""
    return tmp_path_factory.mktemp("llm_rag_chroma")


@pytest.fixture
def pipeline_config(chroma_root: Path) -> Config:
    """Config isolated from local .env keys and with a temp vector store."""
    return Config(
        openrouter_api_key="sk-test-integration",
        chroma_persist_directory=str(chroma_root),
        retrieval_top_k=4,
        max_tokens=512,
    )


@pytest.fixture
def analysis_agent(pipeline_config: Config) -> CodeAnalysisAgent:
    """CodeAnalysisAgent wired to a real OpenRouterProvider + LLMClient."""
    provider = OpenRouterProvider(
        api_key="sk-test-integration",
        model="google/gemma-3-27b-it",
        max_tokens=512,
        timeout=5.0,
        config=pipeline_config,
    )
    client = LLMClient(provider=provider, config=pipeline_config)
    return CodeAnalysisAgent(model_client=client, config=pipeline_config)


def _patch_openrouter_success(content: str):
    """
    Patch OpenRouter GET/POST for a successful availability + completion.

    Returns a context manager; the mock POST records call payloads for
    prompt assertions.
    """
    get_patch = patch(PROVIDER_GET, return_value=_http_response(200, {"data": []}))
    post_patch = patch(PROVIDER_POST)
    sleep_patch = patch(PROVIDER_SLEEP)
    return get_patch, post_patch, sleep_patch, content


def _run_with_llm_content(
    agent: CodeAnalysisAgent,
    repo: Path,
    content: str,
) -> tuple[CodeAnalysisReport, MagicMock]:
    """Execute analyze_repository with a mocked OpenRouter completion body."""
    get_patch, post_patch, sleep_patch, body = _patch_openrouter_success(content)
    with get_patch, sleep_patch, post_patch as mock_post:
        mock_post.return_value = _http_response(200, _chat_payload(body))
        report = agent.analyze_repository(
            repository_path=str(repo.resolve()),
            question=QUESTION,
            use_rag=True,
        )
    return report, mock_post


def test_full_pipeline_indexes_retrieves_prompts_and_grounds(
    analysis_agent: CodeAnalysisAgent,
    analysis_repo: Path,
) -> None:
    """
    Exercise every pipeline stage with a grounded mocked LLM finding.

    Covers indexing, embeddings, retrieval, prompt context, valid LLM
    findings, GroundingChecker acceptance, and static+LLM merge.
    """
    evidence = _source_slice(analysis_repo, WALLET_RELATIVE, 8, 8)
    assert evidence == "    balance = balance - amount"

    content = _finding_payload(
        file_path=WALLET_RELATIVE,
        line_start=8,
        line_end=8,
        evidence=evidence,
    )

    report, mock_post = _run_with_llm_content(analysis_agent, analysis_repo, content)

    assert isinstance(report, CodeAnalysisReport)
    assert report.model_used is True

    # Repository indexing occurs
    assert report.index_update is not None
    assert report.index_update.ingestion.files_indexed >= 1 or report.index_update.added or (
        report.index_update.unchanged
        and report.index_update.ingestion.chunks_indexed >= 0
    )
    # Embeddings created (chunks written on first run; later runs may be unchanged)
    total_chunks = report.index_update.ingestion.chunks_indexed
    if report.index_update.added or report.index_update.modified:
        assert total_chunks >= 1
    else:
        # Incremental no-op still proves a prior indexed store exists for retrieval.
        assert report.index_update.unchanged

    # Retriever returns context
    assert report.context, "expected retrieved chunks for the withdraw question"
    context_sources = {chunk.source.replace("\\", "/") for chunk in report.context}
    assert any(WALLET_RELATIVE in source for source in context_sources)

    # Prompt contains retrieved context
    assert mock_post.call_count >= 1
    request_json = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get(
        "json"
    )
    assert request_json is not None
    user_messages = [
        message["content"]
        for message in request_json["messages"]
        if message["role"] == "user"
    ]
    assert user_messages, "expected a user prompt message"
    prompt = user_messages[0]
    assert "CODE CONTEXT" in prompt
    assert "withdraw" in prompt.lower()
    assert "balance = balance - amount" in prompt

    # Mocked LLM returns valid findings and GroundingChecker accepts them
    assert report.llm_findings, "expected at least one grounded LLM finding"
    llm = report.llm_findings[0]
    assert llm.file_path.replace("\\", "/") == WALLET_RELATIVE
    assert llm.evidence == evidence
    assert llm.detection_method == "llm"
    assert not any(
        result.file_path.replace("\\", "/") == WALLET_RELATIVE
        and result.line_start == 8
        and not result.grounded
        for result in report.rejected
    )

    # Static and LLM findings merge correctly
    assert report.static_findings, "expected static findings (unused import)"
    assert any(f.bug_type == "unused_import" for f in report.static_findings)
    assert len(report.findings) == len(report.static_findings) + len(report.llm_findings)
    methods = {f.detection_method for f in report.findings}
    assert methods == {"static", "llm"}


def test_hallucinated_findings_are_rejected(
    analysis_agent: CodeAnalysisAgent,
    analysis_repo: Path,
) -> None:
    """Findings whose evidence does not match source must not reach findings."""
    content = _finding_payload(
        file_path=WALLET_RELATIVE,
        line_start=8,
        line_end=8,
        evidence="    totally_invented_balance_check()",
        bug_type="hallucinated_bug",
        description="Invented code that does not exist in the repository.",
    )

    report, _ = _run_with_llm_content(analysis_agent, analysis_repo, content)

    assert report.model_used is True
    assert report.context, "pipeline should still retrieve before the model call"
    assert all(f.bug_type != "hallucinated_bug" for f in report.findings)
    assert report.rejected, "expected GroundingChecker to reject the hallucination"
    assert any(not result.grounded for result in report.rejected)
    assert report.static_findings, "static findings must survive hallucinated LLM output"
    assert len(report.llm_findings) == 0


def test_provider_unavailable_falls_back_to_static_only(
    pipeline_config: Config,
    analysis_repo: Path,
) -> None:
    """Missing credentials disable the model half and keep static analysis."""
    provider = OpenRouterProvider(
        api_key="",
        config=pipeline_config,
        timeout=5.0,
    )
    client = LLMClient(provider=provider, config=pipeline_config)
    agent = CodeAnalysisAgent(model_client=client, config=pipeline_config)

    # No OpenRouter HTTP should be required when the key is absent.
    with patch(PROVIDER_GET) as mock_get, patch(PROVIDER_POST) as mock_post:
        report = agent.analyze_repository(
            repository_path=str(analysis_repo.resolve()),
            question=QUESTION,
            use_rag=True,
        )

    mock_get.assert_not_called()
    mock_post.assert_not_called()

    assert report.model_used is False
    assert len(report.llm_findings) == 0
    assert report.static_findings
    assert all(f.detection_method == "static" for f in report.findings)
    assert any("No model provider is available" in note for note in report.notes)


def test_malformed_model_response_falls_back_to_static_only(
    analysis_agent: CodeAnalysisAgent,
    analysis_repo: Path,
) -> None:
    """Unusable OpenRouter bodies keep verified static findings only."""
    with (
        patch(PROVIDER_GET, return_value=_http_response(200, {"data": []})),
        patch(PROVIDER_SLEEP),
        patch(PROVIDER_POST) as mock_post,
    ):
        # Successful HTTP status but body missing choices → ModelResponseError
        mock_post.return_value = _http_response(200, {"id": "bad", "choices": []})
        report = analysis_agent.analyze_repository(
            repository_path=str(analysis_repo.resolve()),
            question=QUESTION,
            use_rag=True,
        )

    assert mock_post.call_count >= 1
    assert report.model_used is False
    assert len(report.llm_findings) == 0
    assert report.static_findings
    assert all(f.detection_method == "static" for f in report.findings)
    assert any("Model call failed" in note for note in report.notes)


def test_static_and_llm_findings_merge_without_dropping_either(
    analysis_agent: CodeAnalysisAgent,
    analysis_repo: Path,
) -> None:
    """Distinct static and LLM defects both survive the merge step."""
    evidence = _source_slice(analysis_repo, WALLET_RELATIVE, 8, 8)
    content = _finding_payload(
        file_path=WALLET_RELATIVE,
        line_start=8,
        line_end=8,
        evidence=evidence,
        bug_type="missing_validation",
    )

    report, _ = _run_with_llm_content(analysis_agent, analysis_repo, content)

    static_types = {f.bug_type for f in report.static_findings}
    llm_types = {f.bug_type for f in report.llm_findings}

    assert "unused_import" in static_types
    assert "missing_validation" in llm_types
    assert len(report.findings) >= 2
    assert report.duplicates_removed == 0
