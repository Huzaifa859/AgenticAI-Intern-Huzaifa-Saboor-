# Codebase Assistant

A multi-agent assistant that ingests a GitHub repository and helps a
developer understand, debug, test, and document it through natural
conversation — delivered as a structured Jupyter notebook.

> **Status: scaffold only.** Every agent, tool, and pipeline stage in
> this repo currently returns placeholder data. No real LLM calls, no
> RAG, no MCP yet — see [Explicitly NOT implemented yet](#explicitly-not-implemented-yet).
> This README documents the target architecture the scaffold is built
> toward, per the project proposal (Week 5, Arbisoft Internship
> Program 2026).

## Project Overview

Developers frequently inherit or join codebases they don't
understand — legacy repos, open-source projects, or a teammate's
unfinished work. Understanding architecture, finding bugs, writing
missing tests, and documenting undocumented code are all slow, manual
tasks.

**Codebase Assistant** targets developers onboarding onto an
unfamiliar codebase (new hires, open-source contributors, students
inheriting a legacy project) who need to quickly understand
structure, locate bugs, and generate missing tests/docs. It's built
around three specialized agents coordinated by a Supervisor:

- **Code Analysis Agent** (primary) — explains modules, traces logic,
  finds likely bugs via a grounded static-analysis-first pipeline
  (never hallucinates a bug without evidence that's mechanically
  verified against the real source).
- **Documentation Agent** (secondary) — generates docstrings/README
  sections for undocumented code.
- **Testing Agent** (secondary) — generates and *actually executes*
  pytest unit tests for target functions, reporting real pass/fail.

Scope for the initial version: **Python only**, up to ~100 source
files / ~20,000 LOC, public GitHub repos (no auth needed), binary
files and Jupyter notebooks as *input* are skipped/out of scope.

## Folder Structure

```
.
├── Project.ipynb                    # Mentor-facing notebook: import → supervisor → run task → show output
├── app/
│   └── main.py                      # Standalone script demo of the same end-to-end flow
├── docs/
│   ├── architecture.excalidraw      # Editable architecture diagram (open at excalidraw.com)
│   └── architecture.png             # Rendered preview of the diagram
└── codebase_assistant/               # The importable Python package
    ├── main.py                       # Placeholder CLI entry point
    ├── config.py                     # App configuration
    ├── supervisor.py                 # Top-level orchestrator (routes tasks to agents)
    ├── agents/
    │   ├── base.py                   # BaseAgent abstract class
    │   ├── code_analysis_agent.py    # CodeAnalysisAgent
    │   ├── documentation_agent.py    # DocumentationAgent
    │   └── testing_agent.py          # TestingAgent
    ├── tools/
    │   ├── registry.py               # ToolRegistry (register/get/list/call tools)
    │   ├── github_tools.py           # GitHubTools (clone_repository, validate_repository, ...)
    │   └── filesystem_tools.py       # FilesystemTools (read_file, list_files, search_codebase)
    ├── rag/
    │   ├── ingest.py                 # Ingestor — pipeline entry point
    │   ├── chunker.py                # Chunker — splits content into chunks
    │   ├── embeddings.py             # EmbeddingGenerator — text/code -> vectors
    │   ├── vectordb.py               # VectorDB — store/query embeddings
    │   ├── indexer.py                # Indexer — higher-level indexing façade
    │   └── retriever.py              # Retriever — higher-level retrieval façade
    ├── memory/
    │   ├── memory_store.py           # Long-term MemoryStore
    │   └── conversation_memory.py    # Short-term ConversationMemory (list-backed, no persistence)
    ├── models/
    │   └── model_client.py           # LLMClient (Claude -> OpenRouter -> Llama -> Ollama, future)
    └── schemas/
        └── schemas.py                # Pydantic models: BugReport, DocumentationResult,
                                       # TestGenerationResult, and internal agent/tool schemas
```

## Architecture Diagram

See [`docs/architecture.excalidraw`](docs/architecture.excalidraw) for
the full editable diagram (open at [excalidraw.com](https://excalidraw.com)
via File → Open, or drag the file onto the canvas). A rendered PNG is
at [`docs/architecture.png`](docs/architecture.png) for quick viewing
without Excalidraw.

```
                 User
                   │
            Project Notebook
                   │
              Supervisor
      ┌────────────┼────────────┐
      │            │            │
 Code Agent   Docs Agent   Test Agent
      │            │            │
      └───────Tool Registry─────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
  GitHub      Filesystem     LLM Client
                   │
       Future RAG Pipeline (planned)
   Ingest → Chunk → Embed → ChromaDB
                   │
             Memory Layer
```

## Tech Stack

| Layer | Choice |
|---|---|
| LLMs | **Claude** (via OpenRouter) for code analysis/bug-finding; local **Llama 3** (via **Ollama**) for documentation generation |
| RAG / retrieval | **ChromaDB** as the vector store, `sentence-transformers` (`all-mpnet-base-v2`) for embeddings, AST-aware chunking (one chunk per function/class) |
| Agent layer | Supervisor/worker pattern (this repo), extended with the Code Analysis, Documentation, and Testing workers |
| Tool access | **MCP** — filesystem tools (`read_file`, `list_files`, `search_codebase`) exposed as MCP tools/resources |
| Structured output | **Pydantic** — `BugReport`, `DocumentationResult`, `TestGenerationResult` validated before anything is written to disk |
| Static analysis | `pyflakes` + Python's `ast` module — deterministic first pass, ahead of any LLM bug-finding |
| Ingestion | `GitPython` / `git clone` + `ast` for function/class-level chunking |
| Test validation | `pytest`, generated tests actually executed via subprocess |
| Interface | Single Jupyter notebook (`Project.ipynb`), staged: ingest → analyze → document → test → report |
| Deployment | Docker + `docker-compose`, bundling the Jupyter environment and an Ollama container |

## Why Two Models (Claude + Llama)

The proposal deliberately splits model duties rather than using one
model everywhere:

- **Claude (via OpenRouter)** handles code analysis and bug-finding —
  the highest-stakes, correctness-critical path, where quality and
  reasoning depth matter most and false positives are costly.
- **Local Llama 3 (via Ollama)** handles documentation generation — a
  lower-stakes, higher-volume task (docstrings for every undocumented
  function) where a free, local model is "good enough" and avoids
  per-call API cost.

This is a genuine cost/quality split rather than an arbitrary choice
of a second provider: it keeps the expensive, high-quality model
reserved for the task that most needs it, while routine documentation
generation runs locally at no marginal cost. It also lets the project
demonstrate real multi-model routing (Week 7: "Claude vs. local llama3
side-by-side comparison on the same documentation task").

## How to Run the Scaffold

Everything below runs **today**, but every result is a placeholder —
there's no real repo cloning, LLM call, or retrieval happening yet.
This only proves the wiring (Supervisor → routing → Agent → response)
works end-to-end.

**1. Install dependencies**

```bash
pip install -r codebase_assistant/requirements.txt
```

**2. Run the notebook** (what your mentor will open)

```bash
jupyter notebook Project.ipynb
```

Run all cells top to bottom: it imports the package, creates the
`Supervisor`, dispatches one fake task, and prints the placeholder
response.

**3. Or run the standalone script demo**

```bash
python app/main.py
```

Expected output:

```
Supervisor started.
Routing task...
Running Code Analysis Agent...
Placeholder code analysis completed.
Finished.
```

## Design Notes

- **Supervisor** wires up all shared subsystems (model client, tool
  registry, RAG, memory) and owns the three specialized agents. It
  routes tasks by keyword today (`route_task`); this is the seam
  where real LLM-driven routing will go later.
- **Agents** (`CodeAnalysisAgent`, `DocumentationAgent`,
  `TestingAgent`) all inherit from `BaseAgent` and share two
  interfaces: the simple `run(repo_path) -> dict` used by the
  Supervisor's current routing, and the richer
  `handle(AgentRequest) -> AgentResponse` for future structured use.
- **ToolRegistry** is the single place tools are registered and
  invoked, decoupling agents from concrete tool implementations
  (GitHub, filesystem, and eventually MCP-exposed tools).
- **RAG** (`Ingestor` → `Chunker` → `EmbeddingGenerator` → `VectorDB`,
  plus the higher-level `Indexer`/`Retriever` façades) and **Memory**
  (`MemoryStore` + `ConversationMemory`) are separate concerns: RAG is
  about codebase content retrieval, Memory is about durable
  facts/preferences and short-term conversation history.
- **schemas.py** defines all cross-cutting Pydantic models.
  `BugReport`, `DocumentationResult`, and `TestGenerationResult` are
  field-for-field matches to the proposal's structured output schemas.

## Explicitly NOT Implemented Yet

- Embeddings / vector search (no ChromaDB, no sentence-transformers calls)
- Real RAG retrieval (ingest/chunk/embed/store are all stubs)
- Real LLM calls (Claude, OpenRouter, Llama, Ollama — none wired up)
- MCP (Model Context Protocol) integration
- Real GitHub API / filesystem I/O (tools print a message and return fake data)
- Real static analysis (pyflakes/ast bug detection)
- CLI argument parsing

## Next Steps

1. Implement `LLMClient.generate` with real Claude (OpenRouter) + Ollama/Llama routing.
2. Implement `FilesystemTools` read/write/list/search operations for real.
3. Implement `GitHubTools.clone_repository` / `validate_repository` for real.
4. Implement the RAG pipeline: `Ingestor` → `Chunker` → `EmbeddingGenerator` (sentence-transformers) → `VectorDB` (ChromaDB).
5. Implement `Supervisor.handle_goal` task decomposition and LLM-driven routing.
6. Add the static analysis pass (`pyflakes` + `ast`) ahead of any LLM bug-finding.
7. Add MCP server integration to `ToolRegistry`.
