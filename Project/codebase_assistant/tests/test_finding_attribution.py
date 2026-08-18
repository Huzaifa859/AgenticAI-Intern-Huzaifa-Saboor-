"""
test_finding_attribution.py
===========================

Attribution, annotate-only grounding, merge, and documentation status
for Code Analysis findings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

from codebase_assistant.agents.code_analysis_agent import CodeAnalysisAgent
from codebase_assistant.analysis.finding_attribution import (
    FOUND_BY_BOTH,
    FOUND_BY_LLM,
    FOUND_BY_STATIC,
    META_ALREADY_DOCUMENTED,
    META_FOUND_BY,
    META_GROUNDING_STATUS,
    attribution_lines,
)
from codebase_assistant.config import Config
from codebase_assistant.schemas.schemas import BugReport, ModelResponse, RetrievedChunk


def _mock_client(content: str) -> MagicMock:
    client = MagicMock()
    client.is_available.return_value = True
    response = ModelResponse(content=content, usage={}, raw={})
    client.generate.return_value = response

    def _stream(_messages, on_chunk=None, **_kwargs):
        if on_chunk is not None and content:
            on_chunk(content)
        return response

    client.generate_stream.side_effect = _stream
    return client


def _mock_retriever(chunks: Optional[List[RetrievedChunk]] = None) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = list(chunks or [])
    retriever.vector_store_path = "./.codebase_assistant/chroma"
    retriever.config = MagicMock()
    retriever.vector_db = MagicMock()
    return retriever


def _bug(
    *,
    bug_type: str,
    description: str,
    file_path: str,
    line_start: int,
    line_end: int,
    evidence: str,
    detection_method: str,
    confidence: float = 0.9,
) -> BugReport:
    return BugReport(
        bug_type=bug_type,
        description=description,
        severity="high",
        confidence=confidence,
        file_path=file_path,
        function_name="target",
        line_start=line_start,
        line_end=line_end,
        evidence=evidence,
        detection_method=detection_method,  # type: ignore[arg-type]
        metadata={},
    )


def _llm_payload(finding: dict) -> str:
    return json.dumps(
        {
            "answer": "Analysis complete.",
            "findings": [finding],
        }
    )


def test_static_only_finding_attributed_static() -> None:
    """Case 1: Static-only → Found by Static Analysis."""
    agent = CodeAnalysisAgent(model_client=None, retriever=None)
    static = [
        _bug(
            bug_type="unused_import",
            description="imported name is never used",
            file_path="mod.py",
            line_start=1,
            line_end=1,
            evidence="import unused",
            detection_method="static",
        )
    ]
    merged, removed = agent._merge(static, [])
    assert removed == 0
    assert len(merged) == 1
    assert merged[0].metadata[META_FOUND_BY] == FOUND_BY_STATIC
    assert "Found by: Static Analysis" in attribution_lines(merged[0])


def test_llm_only_grounded_finding(tmp_path: Path) -> None:
    """Case 2: LLM-only grounded → LLM + Grounded, still visible."""
    source = "def target(x):\n    return x + 1\n"
    (tmp_path / "mod.py").write_text(source, encoding="utf-8")
    evidence = "    return x + 1"
    payload = _llm_payload(
        {
            "bug_type": "off_by_one",
            "description": "adds one when identity was intended",
            "severity": "high",
            "confidence": 0.9,
            "file_path": "mod.py",
            "function_name": "target",
            "line_start": 2,
            "line_end": 2,
            "evidence": evidence,
        }
    )
    chunk = RetrievedChunk(
        source="mod.py",
        content=source,
        score=0.9,
        metadata={"file_path": "mod.py", "line_start": 1, "line_end": 2},
    )
    agent = CodeAnalysisAgent(
        model_client=_mock_client(payload),
        retriever=_mock_retriever([chunk]),
        config=Config(grounding_enabled=True, output_cache_enabled=False),
    )

    with patch.object(agent, "_sync_index", return_value=None), patch.object(
        agent, "_run_static", return_value=[]
    ):
        report = agent.analyze_repository(str(tmp_path), use_rag=True)

    llm = [f for f in report.findings if f.bug_type == "off_by_one"]
    assert llm
    finding = llm[0]
    assert finding.metadata[META_FOUND_BY] == FOUND_BY_LLM
    assert finding.metadata[META_GROUNDING_STATUS] == "grounded"
    lines = attribution_lines(finding)
    assert "Found by: LLM" in lines
    assert "Grounded evidence" in lines


def test_llm_only_ungrounded_finding_still_visible(tmp_path: Path) -> None:
    """Case 3: LLM-only ungrounded → LLM + Ungrounded, still in findings."""
    source = "def target(x):\n    return x\n"
    (tmp_path / "mod.py").write_text(source, encoding="utf-8")
    payload = _llm_payload(
        {
            "bug_type": "invented_bug",
            "description": "claims a defect with fabricated evidence",
            "severity": "high",
            "confidence": 0.9,
            "file_path": "mod.py",
            "function_name": "target",
            "line_start": 2,
            "line_end": 2,
            "evidence": "    totally_invented_call()",
        }
    )
    chunk = RetrievedChunk(
        source="mod.py",
        content=source,
        score=0.9,
        metadata={"file_path": "mod.py", "line_start": 1, "line_end": 2},
    )
    agent = CodeAnalysisAgent(
        model_client=_mock_client(payload),
        retriever=_mock_retriever([chunk]),
        config=Config(grounding_enabled=True, output_cache_enabled=False),
    )

    with patch.object(agent, "_sync_index", return_value=None), patch.object(
        agent, "_run_static", return_value=[]
    ):
        report = agent.analyze_repository(str(tmp_path), use_rag=True)

    finding = next(f for f in report.findings if f.bug_type == "invented_bug")
    assert finding.metadata[META_FOUND_BY] == FOUND_BY_LLM
    assert finding.metadata[META_GROUNDING_STATUS] == "ungrounded"
    lines = attribution_lines(finding)
    assert "Found by: LLM" in lines
    assert "Ungrounded evidence" in lines
    assert report.abstention is None


def test_matching_static_and_llm_merge_to_static_plus_llm() -> None:
    """Case 4: matching Static + LLM → one Static + LLM finding."""
    agent = CodeAnalysisAgent(model_client=None, retriever=None)
    static = [
        _bug(
            bug_type="unused_import",
            description="imported name is never used",
            file_path="mod.py",
            line_start=1,
            line_end=1,
            evidence="import unused",
            detection_method="static",
        )
    ]
    llm = [
        _bug(
            bug_type="unused_import",
            description="unused import detected by model",
            file_path="mod.py",
            line_start=1,
            line_end=1,
            evidence="import unused",
            detection_method="llm",
        )
    ]
    static[0].metadata[META_GROUNDING_STATUS] = "grounded"
    llm[0].metadata[META_GROUNDING_STATUS] = "grounded"
    static[0].metadata[META_FOUND_BY] = FOUND_BY_STATIC
    llm[0].metadata[META_FOUND_BY] = FOUND_BY_LLM

    merged, removed = agent._merge(static, llm)
    assert removed == 1
    assert len(merged) == 1
    assert merged[0].metadata[META_FOUND_BY] == FOUND_BY_BOTH
    assert merged[0].detection_method == "hybrid"
    assert "Found by: Static + LLM" in attribution_lines(merged[0])
    assert "Grounded evidence" in attribution_lines(merged[0])


def test_documented_finding_still_visible_with_flag(tmp_path: Path) -> None:
    """Case 5: documented finding stays visible with Already documented."""
    source = (
        "def target(prices):\n"
        '    """Bug: off-by-one — last price is never discounted."""\n'
        "    out = []\n"
        "    for index in range(len(prices) - 1):\n"
        "        out.append(prices[index])\n"
        "    return out\n"
    )
    (tmp_path / "mod.py").write_text(source, encoding="utf-8")
    evidence = "    for index in range(len(prices) - 1):"
    payload = _llm_payload(
        {
            "bug_type": "off_by_one",
            "description": "off-by-one loop skips the last price so it is never discounted",
            "severity": "high",
            "confidence": 0.9,
            "file_path": "mod.py",
            "function_name": "target",
            "line_start": 4,
            "line_end": 4,
            "evidence": evidence,
        }
    )
    chunk = RetrievedChunk(
        source="mod.py",
        content=source,
        score=0.9,
        metadata={"file_path": "mod.py", "line_start": 1, "line_end": 6},
    )
    agent = CodeAnalysisAgent(
        model_client=_mock_client(payload),
        retriever=_mock_retriever([chunk]),
        config=Config(grounding_enabled=True, output_cache_enabled=False),
    )

    with patch.object(agent, "_sync_index", return_value=None), patch.object(
        agent, "_run_static", return_value=[]
    ):
        report = agent.analyze_repository(str(tmp_path), use_rag=True)

    finding = next(f for f in report.findings if f.bug_type == "off_by_one")
    assert finding.metadata.get(META_ALREADY_DOCUMENTED) is True
    assert "Already documented in the code." in attribution_lines(finding)


def test_static_llm_documented_preserves_all_attributes() -> None:
    """Case 6: Static + LLM + documented → all three attributes preserved."""
    agent = CodeAnalysisAgent(model_client=None, retriever=None)
    static = [
        _bug(
            bug_type="off_by_one",
            description="off-by-one loop skips the last price",
            file_path="mod.py",
            line_start=4,
            line_end=4,
            evidence="for index in range(len(prices) - 1):",
            detection_method="static",
        )
    ]
    llm = [
        _bug(
            bug_type="off_by_one",
            description="off-by-one loop skips the last price",
            file_path="mod.py",
            line_start=4,
            line_end=4,
            evidence="for index in range(len(prices) - 1):",
            detection_method="llm",
        )
    ]
    static[0].metadata[META_FOUND_BY] = FOUND_BY_STATIC
    static[0].metadata[META_GROUNDING_STATUS] = "grounded"
    static[0].metadata[META_ALREADY_DOCUMENTED] = True
    llm[0].metadata[META_FOUND_BY] = FOUND_BY_LLM
    llm[0].metadata[META_GROUNDING_STATUS] = "grounded"

    merged, removed = agent._merge(static, llm)
    assert removed == 1
    finding = merged[0]
    lines = attribution_lines(finding)
    assert "Found by: Static + LLM" in lines
    assert "Grounded evidence" in lines
    assert "Already documented in the code." in lines


def test_different_static_and_llm_findings_remain_separate() -> None:
    """Case 7: different bugs stay separate with respective sources."""
    agent = CodeAnalysisAgent(model_client=None, retriever=None)
    static = [
        _bug(
            bug_type="unused_import",
            description="imported name is never used",
            file_path="mod.py",
            line_start=1,
            line_end=1,
            evidence="import unused",
            detection_method="static",
        )
    ]
    llm = [
        _bug(
            bug_type="off_by_one",
            description="loop skips the last element",
            file_path="mod.py",
            line_start=10,
            line_end=10,
            evidence="for i in range(len(items) - 1):",
            detection_method="llm",
        )
    ]
    static[0].metadata[META_FOUND_BY] = FOUND_BY_STATIC
    llm[0].metadata[META_FOUND_BY] = FOUND_BY_LLM

    merged, removed = agent._merge(static, llm)
    assert removed == 0
    assert len(merged) == 2
    by_type = {f.bug_type: f for f in merged}
    assert by_type["unused_import"].metadata[META_FOUND_BY] == FOUND_BY_STATIC
    assert by_type["off_by_one"].metadata[META_FOUND_BY] == FOUND_BY_LLM


def test_unsupported_finding_without_file_still_rejected_at_parse(
    tmp_path: Path,
) -> None:
    """Case 8: existing parse validation still drops unusable fabricated entries."""
    (tmp_path / "mod.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    agent = CodeAnalysisAgent(
        model_client=_mock_client("{}"),
        config=Config(grounding_enabled=True),
    )
    _answer, findings = agent.parse_response(
        json.dumps(
            {
                "answer": "Invented.",
                "findings": [
                    {
                        "bug_type": "invented",
                        "description": "no path",
                        "severity": "high",
                        "confidence": 0.9,
                        "file_path": "",
                        "line_start": 1,
                        "line_end": 1,
                        "evidence": "not real",
                    }
                ],
            }
        ),
        workspace_root=str(tmp_path),
    )
    assert findings == []


def test_analysis_enables_grounding_by_default(tmp_path: Path) -> None:
    """Code Analysis binds a grounding checker with annotation enabled."""
    from codebase_assistant.analysis.grounding_checker import GroundingChecker

    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    agent = CodeAnalysisAgent(model_client=None, retriever=None)
    pipeline = agent._bind(str(tmp_path))
    assert isinstance(pipeline.checker, GroundingChecker)
    assert pipeline.checker.enabled is True
