# `codebase_assistant` package

This is the importable Python package for **Codebase Assistant**.

For the full project overview, architecture diagram, tech stack, model
choice rationale, and run instructions, see the
[project root README](../README.md).

## Package layout

- `supervisor.py` — top-level orchestrator; routes tasks to agents.
- `agents/` — `CodeAnalysisAgent`, `DocumentationAgent`, `TestingAgent`,
  all built on `BaseAgent`.
- `tools/` — `ToolRegistry`, `GitHubTools`, `FilesystemTools`.
- `rag/` — `Ingestor`, `Chunker`, `EmbeddingGenerator`, `VectorDB`
  (plus `Indexer`/`Retriever` façades).
- `memory/` — `MemoryStore` (long-term), `ConversationMemory` (short-term).
- `models/` — `LLMClient`, the abstraction over future model providers.
- `schemas/` — shared Pydantic models (`BugReport`, `DocumentationResult`,
  `TestGenerationResult`, and internal agent/tool schemas).

Everything here currently returns placeholder data — see the root
README's "Explicitly NOT implemented yet" section for the full list.
