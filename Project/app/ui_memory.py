"""
app/ui_memory.py
================

Shared ConversationMemory helpers for CLI and Streamlit (CLI parity).

Stores short turns and target metadata only — never full reports or
generated test sources.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Mapping, Optional

from codebase_assistant.memory.conversation_memory import ConversationMemory
from codebase_assistant.memory.memory_store import MemoryStore
from codebase_assistant.schemas.schemas import (
    DocumentationResult,
    ModelMessage,
    TestingResult,
)

#: Soft cap so memory never persists full reports or generated source.
MAX_MEMORY_CONTENT_CHARS = 400

STREAMLIT_CONVERSATION_ID = "streamlit_default"
STREAMLIT_RUNTIME_ROOT = os.path.join(
    tempfile.gettempdir(), "codebase_assistant_streamlit"
)
STREAMLIT_MEMORY_STORE_PATH = os.path.join(STREAMLIT_RUNTIME_ROOT, "memory_store")


def memory_target(memory: Optional[ConversationMemory]) -> Dict[str, str]:
    """Read last documentation/testing target fields from memory metadata."""
    if memory is None:
        return {"file_path": "", "function_name": "", "class_name": ""}
    meta = getattr(memory, "metadata", {}) or {}
    return {
        "file_path": str(meta.get("last_file_path") or ""),
        "function_name": str(meta.get("last_function_name") or ""),
        "class_name": str(meta.get("last_class_name") or ""),
    }


def store_memory_target(
    memory: Optional[ConversationMemory],
    *,
    file_path: str = "",
    function_name: str = "",
    class_name: str = "",
) -> None:
    """Persist the latest target selection for follow-up turns."""
    if memory is None:
        return
    if file_path:
        memory.metadata["last_file_path"] = file_path
    if function_name:
        memory.metadata["last_function_name"] = function_name
    if class_name:
        memory.metadata["last_class_name"] = class_name


def record_memory_message(
    memory: Optional[ConversationMemory],
    role: str,
    content: str,
) -> None:
    """
    Append one short turn to ConversationMemory when available.

    Uses ``add_message`` so summarization and MemoryStore persistence run
    automatically. Content is truncated so full agent reports never enter
    memory.
    """
    if memory is None:
        return
    text = (content or "").strip()
    if not text:
        return
    if len(text) > MAX_MEMORY_CONTENT_CHARS:
        text = text[: MAX_MEMORY_CONTENT_CHARS - 3].rstrip() + "..."
    memory.add_message(ModelMessage(role=role, content=text))


def record_repository_loaded(
    memory: Optional[ConversationMemory],
    reference: str,
    repository_path: str,
) -> None:
    """Record the selected repository for this session."""
    if memory is None:
        return
    memory.metadata["repository_reference"] = reference
    memory.metadata["repository_path"] = repository_path
    record_memory_message(memory, "user", f"Repository: {reference}")
    record_memory_message(memory, "assistant", "Repository loaded.")


def summarize_analysis_for_memory(report: object) -> str:
    """
    Build a short analysis summary for ConversationMemory.

    Accepts a CodeAnalysisReport-like object or a Streamlit worker dict.
    """
    if isinstance(report, Mapping):
        findings = list(report.get("findings") or [])
        llm_grounded = int(report.get("llm_grounded_count") or 0)
        ungrounded = list(report.get("ungrounded_candidates") or [])
        finding_label = "finding" if len(findings) == 1 else "findings"
        parts = [f"{len(findings)} verified {finding_label}."]
        if llm_grounded:
            parts.append(f"{llm_grounded} grounded LLM finding(s).")
        if ungrounded:
            parts.append(f"{len(ungrounded)} ungrounded candidate(s).")
        return " ".join(parts)

    static_count = len(getattr(report, "static_findings", []) or [])
    llm_count = len(getattr(report, "llm_findings", []) or [])
    # Fallback when only merged findings exist on the object.
    if static_count == 0 and llm_count == 0:
        merged = len(getattr(report, "findings", []) or [])
        if merged:
            label = "finding" if merged == 1 else "findings"
            return f"{merged} verified {label}."
    static_label = "finding" if static_count == 1 else "findings"
    llm_label = "finding" if llm_count == 1 else "findings"
    return (
        f"{static_count} static {static_label}. "
        f"{llm_count} grounded LLM {llm_label}."
    )


def summarize_documentation_for_memory(result: object) -> str:
    """Build a short documentation summary for ConversationMemory."""
    if isinstance(result, DocumentationResult):
        name = (result.function_name or "").strip()
        if not name or name.upper() == "README":
            return "README generated."
        return f"Documentation generated for {name}."

    if isinstance(result, Mapping):
        name = str(result.get("function_name") or "").strip()
        file_path = str(result.get("file_path") or "").strip()
        if name and name.upper() != "README":
            return f"Documentation generated for {name}."
        if file_path:
            return f"Documentation generated for {os.path.basename(file_path)}."
        summary = str(result.get("summary") or "").strip().replace("\n", " ")
        if summary:
            return (summary[:120] + "…") if len(summary) > 120 else summary
        return "Documentation generated."
    return "Documentation generated."


def summarize_testing_for_memory(result: object) -> str:
    """Build a short testing summary for ConversationMemory."""
    if isinstance(result, TestingResult):
        module_count = len(result.generated_tests or {})
        module_label = "module" if module_count == 1 else "modules"
        return f"Generated tests for {module_count} {module_label}."

    if isinstance(result, Mapping):
        generated = result.get("generated_tests") or {}
        module_count = len(generated) if isinstance(generated, Mapping) else 0
        module_label = "module" if module_count == 1 else "modules"
        return f"Generated tests for {module_count} {module_label}."
    return "Tests generated."


def build_streamlit_conversation_memory() -> ConversationMemory:
    """
    Create ConversationMemory for the Streamlit UI process.

    Uses a fixed conversation id and the streamlit runtime MemoryStore path
    so snapshots survive script reloads. ``model_client`` is None so the UI
    process never loads an LLM just for summarization.
    """
    os.makedirs(STREAMLIT_MEMORY_STORE_PATH, exist_ok=True)
    store = MemoryStore(storage_path=STREAMLIT_MEMORY_STORE_PATH)
    return ConversationMemory(
        model_client=None,
        memory_store=store,
        conversation_id=STREAMLIT_CONVERSATION_ID,
    )


def clear_conversation_memory(memory: Optional[ConversationMemory]) -> None:
    """Clear in-memory turns/metadata and delete the persisted snapshot."""
    if memory is None:
        return
    memory.metadata.clear()
    memory.clear_history()
    store = memory.memory_store
    if store is not None:
        try:
            store.delete(memory.conversation_id)
        except Exception:
            pass


def remembered_repository_reference(memory: Optional[ConversationMemory]) -> str:
    """Return persisted repository_reference from memory metadata, if any."""
    if memory is None:
        return ""
    meta = getattr(memory, "metadata", {}) or {}
    return str(meta.get("repository_reference") or "").strip()
