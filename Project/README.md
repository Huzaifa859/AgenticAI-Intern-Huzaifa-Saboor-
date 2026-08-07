# AI Software Engineering Assistant

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![RAG](https://img.shields.io/badge/RAG-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![MCP](https://img.shields.io/badge/MCP-stdio%20%2B%20local-purple.svg)](#mcp-architecture)

A multi-agent AI assistant that helps developers understand, debug, document, and test Python repositories.

It ingests a local path or public GitHub URL, indexes the codebase with RAG, and routes work through a Supervisor to specialized agents — Code Analysis, Documentation, and Testing — while grounding LLM claims against real source text.

> **Problem it solves:** inherited or unfamiliar codebases are slow to audit by hand. This project turns repository analysis, documentation drafts, and pytest generation into a single, structured multi-agent pipeline with retrieval, tracing, and memory.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Multi-Agent Architecture](#multi-agent-architecture)
- [RAG Pipeline](#rag-pipeline)
- [MCP Architecture](#mcp-architecture)
- [Conversation Memory](#conversation-memory)
- [Tracing](#tracing)
- [Supported Models](#supported-models)
- [GitHub Integration](#github-integration)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [CLI Usage](#cli-usage)
- [Web UI](#web-ui)
- [Screenshots](#screenshots)
- [Example Outputs](#example-outputs)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [License](#license)

---

## Features

| Area | Capability |
|---|---|
| Analysis | Repository analysis with static (`pyflakes` + `ast`) and optional LLM passes |
| Documentation | Documentation generation grounded in retrieved code context |
| Testing | Pytest test generation with local execution of generated tests |
| Repositories | Local paths and GitHub HTTPS URLs |
| RAG | Chunking, embeddings, ChromaDB vector store, semantic retrieval |
| Indexing | Incremental indexing via content-hash manifest |
| Quality | Grounding checker rejects hallucinated evidence before it reaches the user |
| Models | OpenRouter with automatic model fallback; optional local Ollama |
| Retrieval | Optional cross-encoder reranking (disabled by default) |
| Memory | Short-term conversation memory with automatic summarization and disk persistence |
| Observability | End-to-end tracing with ordered events and JSON export |
| Tools | Shared `ToolRegistry` for filesystem and GitHub tools |
| MCP | In-process MCP foundation plus official **stdio** server for external hosts (`analysis_run`, `documentation_run`, `testing_run`, `goal_run`) |
| Interfaces | Interactive CLI menu, non-interactive `--agent` mode, Streamlit web UI, and Jupyter demo notebook |

---

## Architecture

```mermaid
flowchart TD
    CLI[CLI / Streamlit / Notebook / MCP Client]
    CLI --> Supervisor[Supervisor]
    Supervisor --> Routing[Agent Routing]
    Routing --> CAA[Code Analysis Agent]
    Routing --> DA[Documentation Agent]
    Routing --> TA[Testing Agent]
    CAA --> Shared[Shared Components]
    DA --> Shared
    TA --> Shared
    Shared --> Retriever[Retriever]
    Shared --> Indexer[Indexer]
    Shared --> Registry[Tool Registry]
    Shared --> Memory[Conversation Memory]
    Shared --> Tracer[Tracing]
    Shared --> Providers[OpenRouter / Ollama Providers]
```

The Supervisor owns shared services and dispatches each request to the correct agent. Agents do not invent a second routing layer — CLI and MCP both call the same Supervisor pipelines.

Editable architecture references also live under [`docs/architecture.excalidraw`](docs/architecture.excalidraw) and [`docs/architecture.png`](docs/architecture.png).

---

## Multi-Agent Architecture

### Supervisor

Top-level orchestrator. It prepares the repository, wires providers and tools, routes `handle_task(...)` / `handle_goal(...)` to agents, aggregates responses, and records lifecycle traces.

### Code Analysis Agent

Runs the bug-finding pipeline:

1. Optional index update when a model is available
2. Deterministic static analysis
3. Grounding of static findings
4. Retrieval + LLM analysis (when a provider is available)
5. Grounding of LLM findings
6. Merge, deduplicate, and return a `CodeAnalysisReport`

Without a configured LLM, analysis still completes using static findings only.

### Documentation Agent

Retrieves relevant context for a target (for example `README` or a module) and asks the model to produce a structured `DocumentationResult` (summary, parameters, returns, example usage).

### Testing Agent

Generates pytest tests for a target module/function, writes them to a temporary location, executes them with `pytest`, and returns a `TestingResult` that includes generated sources plus an execution summary.

### ToolRegistry

Canonical registry of callable tools. Filesystem helpers, GitHub helpers, and MCP agent tools are registered here so agents and the MCP server resolve capabilities through one interface.

---

## RAG Pipeline

```mermaid
flowchart TD
    Repo[Repository]
    Repo --> Indexer[Indexer]
    Indexer --> Chunker[Chunker]
    Chunker --> Embed[Embedding Generator]
    Embed --> Store[Vector Store / ChromaDB]
    Store --> Retriever[Retriever]
    Retriever --> Prompt[Prompt Builder]
    Prompt --> LLM[LLM Provider]
    LLM --> Ground[Grounding Checker]
    Ground --> Report[Structured Report]
```

**How it works**

1. **Indexer** walks the repository (respecting size/file caps and ignore directories) and updates the index incrementally.
2. **Chunker** produces AST-aware chunks for Python and section chunks for Markdown/text.
3. **Embeddings** are generated with `sentence-transformers` (`all-mpnet-base-v2` by default).
4. **Vector store** persists chunks in ChromaDB.
5. **Retriever** returns the top-k semantically similar chunks (optional cross-encoder rerank).
6. **Prompt builder** packs retrieved context into grounded prompts for the agents.
7. **LLM** proposes findings or documentation/tests.
8. **Grounding checker** verifies quoted evidence against the real source before results are accepted.

---

## MCP Architecture

The MCP layer is another frontend for the existing Supervisor — it does not reimplement agent logic or routing.

There are **two transports**:

| Transport | Who uses it | How |
|---|---|---|
| **Local / in-process** | Unit tests, notebook-style embedding | `MCPServer` + `MCPClient` keyed by `host:port` in the same Python process |
| **Stdio (official MCP)** | Cursor, Claude Desktop, other MCP hosts | `python -m codebase_assistant.mcp` — FastMCP JSON-RPC on stdin/stdout |

```mermaid
flowchart TD
    Host[Cursor_or_ClaudeDesktop]
    CLI["python -m codebase_assistant.mcp"]
    Bridge[stdio FastMCP bridge]
    Local[MCPServer in-process]
    Supervisor[Supervisor]
    Registry[Tool Registry]
    Host -->|stdio JSON-RPC| CLI
    CLI --> Bridge
    Bridge -->|invoke_tool| Local
    Local --> Supervisor
    Supervisor --> Registry
    Registry --> Agents[Code Analysis / Documentation / Testing]
```

**Exposed agent tools (stdio protocol names)**

| Stdio tool | Local registry name | Args | Result |
|---|---|---|---|
| `analysis_run` | `analysis.run` | `repository`, optional `question` | `CodeAnalysisReport` (JSON) |
| `documentation_run` | `documentation.run` | `repository`, optional `target` | `DocumentationResult` (JSON) |
| `testing_run` | `testing.run` | `repository`, optional `target` | `TestingResult` (JSON) |
| `goal_run` | `goal.run` | `repository`, `goal` | Ordered `AgentResponse` list (JSON) |

By default stdio exposes the four agent tools only. Set `MCP_MIRROR_REGISTRY_TOOLS=1` to also mirror filesystem/GitHub registry tools (underscored names + `arguments_json` kwargs).

### Run the stdio server

```bash
cd Project
pip install -r codebase_assistant/requirements.txt   # includes mcp>=1.28,<2
python -m codebase_assistant.mcp --help
python -m codebase_assistant.mcp --list-tools
python -m codebase_assistant.mcp                 # blocks; hosts attach via stdio
# Or: run_mcp.bat
```

Logging goes to **stderr**. Do not print app progress on stdout — that channel is the MCP wire.

### Example host config (Cursor / Claude Desktop)

Point the host at the Project directory and your Python interpreter:

```json
{
  "mcpServers": {
    "codebase-assistant": {
      "command": "python",
      "args": ["-m", "codebase_assistant.mcp"],
      "cwd": "C:/path/to/Project",
      "env": {
        "OPENROUTER_API_KEY": "your_key_here"
      }
    }
  }
}
```

Use an absolute `cwd` so imports and `examples/demo_repo` resolve correctly. Long agent runs may hit host tool timeouts — prefer targeted `documentation_run` / `testing_run` calls in interactive hosts.

The in-process server still owns a Supervisor instance, mirrors its `ToolRegistry`, and records MCP request/response/tool traces. HTTP / SSE MCP and a Docker MCP service are not included yet.

---

## Conversation Memory

| Component | Role |
|---|---|
| `ConversationMemory` | Short-term turn history used during a CLI/session run |
| `MemoryStore` | Persistent on-disk store for conversation snapshots |
| Summarization | When history exceeds the configured message cap, older turns are condensed via the LLM into a system summary |
| Persistence | After each update (and after successful summarization), state is saved through `MemoryStore` |

If the provider is unavailable during summarization, history is left unchanged so no turns are lost.

---

## Tracing

`Tracer` records ordered lifecycle events for a run.

**What gets traced**

- CLI start / agent selection
- Supervisor routing and agent runs
- Ingestion / retrieval / model / tool calls
- Memory summarization
- MCP request, tool invoked, and MCP response (with duration and success)

**Event model**

- Typed categories (`lifecycle`, `agent_run`, `retrieval`, `model_call`, `tool_call`, `memory`, `error`, …)
- Monotonic sequence numbers for deterministic ordering
- Optional duration, success flag, component name, and error text

**Export**

```python
supervisor.tracer.export("trace.json")
```

Writes deterministic JSON with `run_id`, ordered `events`, and an aggregate `summary`.

---

## Supported Models

### OpenRouter

Primary remote provider. Used for analysis and as the general LLM backend when `OPENROUTER_API_KEY` is configured.

Default primary model: `nvidia/nemotron-3-ultra-550b-a55b:free`.

### Nemotron 3 Ultra (free)

Reached through OpenRouter (`nvidia/nemotron-3-ultra-550b-a55b:free`). Default primary model for code analysis and grounded bug finding. Free-tier rate limits apply (see OpenRouter docs).

### Gemma

OpenRouter fallback candidate: `google/gemma-3-27b-it`.

### Llama

OpenRouter fallback candidate: `meta-llama/llama-3.1-8b-instruct`.

Also available locally via Ollama as `llama3` (default Ollama model).

### Nemotron Nano

OpenRouter fallback candidate: `nvidia/nemotron-nano-9b-v2`.

### Ollama

Local provider for models served at `OLLAMA_BASE_URL` (default `http://localhost:11434`). Default model: `llama3`. Used when a local documentation/runtime path is preferred.

### Automatic model fallback

`OpenRouterProvider` retries across a fixed fallback chain when the API returns selected recoverable statuses (for example payment/model-unavailable cases). Authentication and malformed-request failures do **not** walk the chain — they fail fast.

Fallback order:

1. Primary model (default Nemotron 3 Ultra free)
2. Gemma 3 27B IT
3. Llama 3.1 8B Instruct
4. Nemotron Nano 9B

---

## GitHub Integration

| Capability | Behavior |
|---|---|
| Supported URLs | HTTPS GitHub repository URLs |
| Local paths | Existing directories on disk |
| Clone | Validated, then cloned with GitPython (CLI `git` fallback) |
| Temporary workspace | Remote repos are cloned into a temp directory for the run |
| Cleanup | Temporary clones are removed when the CLI/MCP session finishes |
| REST reads | `get_file_contents`, `list_issues`, `list_pull_requests` |
| REST writes | `create_branch`, `commit_file`, `create_pull_request` (require `GITHUB_TOKEN`) |

Public clone/validate works without a token. Authenticated GitHub write operations require `GITHUB_TOKEN`.

---

## Installation

All commands assume you are inside the `Project/` directory of this repository.

### 1. Clone the repository

```bash
git clone https://github.com/Huzaifa859/AgenticAI-Intern-Huzaifa-Saboor-.git
cd AgenticAI-Intern-Huzaifa-Saboor-/Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r codebase_assistant/requirements.txt
```

> First RAG indexing download may pull the embedding model (~400 MB).

### 4. Create `.env`

Create `Project/.env`:

```bash
OPENROUTER_API_KEY=your_key_here
# Optional:
# GITHUB_TOKEN=ghp_...
# OLLAMA_BASE_URL=http://localhost:11434
# RERANK_ENABLED=false
```

### 5. Run

```bash
# Interactive menu
python app/main.py .

# Non-interactive analysis
python app/main.py . --agent analysis --question "Find security bugs"

# Streamlit web UI (same .env / provider setup as the CLI)
streamlit run app/streamlit_app.py
```

---

## Environment Variables

Variables below are loaded by `Config.load()` (from `Project/.env` and the process environment).

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | For LLM features | unset | OpenRouter API key |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Local Ollama service URL |
| `GITHUB_TOKEN` | For authenticated GitHub API writes | unset | GitHub personal access token |
| `WORKSPACE_ROOT` | No | `.` | Default workspace root |
| `CHROMA_PERSIST_DIR` | No | `./.codebase_assistant/chroma` | ChromaDB persistence directory |
| `MEMORY_STORE_PATH` | No | `./.codebase_assistant/memory_store` | Persistent conversation memory path |
| `RETRIEVAL_TOP_K` | No | `8` | Chunks returned per retrieval query |
| `RERANK_ENABLED` | No | `false` | Enable cross-encoder reranking |
| `RERANK_MODEL_NAME` | No | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `RERANK_CANDIDATES` | No | `24` | Candidate pool size before rerank |
| `LOG_LEVEL` | No | `INFO` | Logging threshold |
| `DOCUMENTATION_LENIENT` | No | `true` | Keep imperfect documentation LLM text (with warnings) instead of emptying on JSON/grounding failures. Set `false` for strict abstention. |
| `TESTING_LENIENT` | No | `true` | Salvage pytest source from non-JSON testing model output instead of abstaining with an empty suite. Set `false` for strict abstention. |
| `ANALYSIS_SHOW_UNGROUNDED` | No | `false` | When `true`, the CLI analysis report also prints findings that failed grounding as unverified candidates. The Streamlit UI has a separate checkbox for the same view. |

Model identifiers (`openrouter_model`, `claude_model`, `ollama_model`) are Config defaults (`nvidia/nemotron-3-ultra-550b-a55b:free` and `llama3`) unless changed in code/configuration objects.

---

## CLI Usage

### Interactive menu

```bash
python app/main.py .
```

Omitting `--agent` prepares the repository once and opens a menu for Code Analysis, Documentation, or Testing.

### Analyze

```bash
python app/main.py . --agent analysis --question "Find security bugs"
```

### Generate documentation

```bash
python app/main.py . --agent documentation
```

### Generate tests

```bash
python app/main.py . --agent testing
```

### Run all agents

```bash
python app/main.py . --agent all
```

### GitHub URL

```bash
python app/main.py https://github.com/owner/repo --agent analysis
```

The URL is validated, cloned to a temporary workspace, analyzed, then cleaned up on exit.

### Non-interactive mode

Passing `--agent` runs the selected pipeline once and exits. Color can be forced or disabled:

```bash
python app/main.py codebase_assistant --agent analysis --no-color
```

### Notebook demo

```bash
jupyter notebook Project.ipynb
```

---

## Web UI

A Streamlit frontend wraps the same Supervisor pipeline as the CLI. Provider configuration (`.env`, OpenRouter → Ollama failover) is unchanged.

```bash
cd Project
# Preferred launcher (file-watcher off + prefers Chrome):
run_ui.bat

# Or:
streamlit run app/streamlit_app.py --server.fileWatcherType=none
```

`Project/.streamlit/config.toml` sets `server.headless = false` so Streamlit auto-opens a browser tab on startup (uses your OS default browser; `run_ui.bat` prefers Chrome when installed). It also disables the file watcher (Chroma writes used to restart the app mid-run).

Agent jobs run in a separate `app/worker.py` process so embedding/LLM memory use cannot kill the Streamlit server.

While a job runs, the UI polls an NDJSON progress file from the worker and shows **live stage updates** with a progress bar (indexing, model call, grounding, pytest, and similar). Use **Stop run** to cancel a long job. This is stage progress only — not LLM token streaming into the chat pane.

Completed, failed, and cancelled runs are appended to a capped **Run history** (sidebar expander, newest first, max 20). Each row shows your **local device time** plus a short result summary. History is kept in `st.session_state` and persisted to `%TEMP%/codebase_assistant_streamlit/ui_run_history.jsonl` so it survives Streamlit script reloads. Selecting an entry restores its result and auto-opens the matching result pane. Each result pane can **Download Markdown** or **Download JSON**.

Separately, the UI keeps **Session memory** (same `ConversationMemory` + `MemoryStore` pattern as the CLI): last repository reference/path, docs/testing target fields, and short Load/Run summaries. Memory lives in the Streamlit process (not the worker), prefills sidebar defaults, and persists under `%TEMP%/codebase_assistant_streamlit/memory_store` with conversation id `streamlit_default`. It is not a chat pane and is not injected into agent LLM prompts. **Run history** remains the result browser; **Session memory** is conversational/session context only. Use **Clear memory** in the sidebar expander to reset both in-memory state and the disk snapshot.

On Analysis reports, enable **Show ungrounded candidates** to inspect findings that failed grounding. They appear in a separate **Unverified** section and are never mixed into verified bugs.

In the sidebar:

1. Enter a local path (for example `examples/demo_repo`) or a GitHub HTTPS URL and click **Load repository**
2. Choose **Analysis**, **Documentation**, or **Testing**
3. For docs/tests, set mode and optional file/function/class targeting (defaults may come from Session memory)
4. Click **Run** — watch stage progress (or **Stop run**), then browse the focused result pane
5. Open **Run history** to revisit prior runs or clear the list
6. Open **Session memory** to review remembered repo/targets/short turns, or clear memory

Orchestration lives in `app/service.py`; presentation helpers live in `app/ui_reports.py`; history helpers live in `app/ui_history.py`; conversation-memory helpers live in `app/ui_memory.py`; export helpers live in `app/ui_export.py`. Agent logic stays inside `codebase_assistant/`.

---

## Screenshots

Add real screenshots under [`docs/images/`](docs/images/) when available. Placeholder captions:

### CLI menu

![CLI menu](docs/images/cli-menu.png)

_Interactive agent selection after repository preparation._

### Analysis output

![Analysis output](docs/images/analysis-output.png)

_Severity-grouped findings with grounded evidence snippets._

### Documentation output

![Documentation output](docs/images/documentation-output.png)

_Structured documentation result for a target module or README._

### Testing output

![Testing output](docs/images/testing-output.png)

_Generated pytest sources and local execution summary._

> Image files are not bundled yet. See [`docs/images/README.md`](docs/images/README.md) for suggested filenames.

---

## Example Outputs

### CodeAnalysisReport (shortened)

```json
{
  "repository_path": "/tmp/demo-repo",
  "question": "Find likely bugs and correctness problems in this code.",
  "model_used": true,
  "duplicates_removed": 1,
  "duration_seconds": 4.82,
  "findings": [
    {
      "bug_type": "unused_import",
      "severity": "low",
      "confidence": 0.95,
      "file_path": "math_utils.py",
      "function_name": "<module>",
      "line_start": 1,
      "line_end": 1,
      "evidence": "from typing import Optional",
      "detection_method": "static",
      "description": "'typing.Optional' imported but unused"
    }
  ],
  "notes": []
}
```

### DocumentationResult (shortened)

```json
{
  "file_path": "math_utils.py",
  "function_name": "add",
  "summary": "Return the sum of two numbers.",
  "parameters": [
    {"name": "a", "type": "int | float", "description": "First addend."},
    {"name": "b", "type": "int | float", "description": "Second addend."}
  ],
  "returns": "The numeric sum of a and b.",
  "example_usage": "add(2, 3)  # 5"
}
```

### TestingResult (shortened)

```json
{
  "summary": "Generated tests for add.\n\nExecution: 1 passed, 0 failed.",
  "generated_tests": {
    "test_math_utils.py": "from math_utils import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"
  },
  "coverage_estimate": 0.5
}
```

---

## Testing

From `Project/`:

```bash
PYTHONPATH=. python -m pytest codebase_assistant/tests -q
```

| Suite | Focus |
|---|---|
| Unit tests | Agents, tools, providers, schemas, memory, tracing |
| Integration tests | Week 6 end-to-end static analysis pipeline on a seeded repo |
| Mocked LLM tests | Documentation/testing/analysis with fake providers (no network) |
| RAG / retrieval tests | Indexing and retrieval behavior where covered |
| MCP tests | In-process lifecycle/tools + stdio bridge name mapping / forwarding |
| GitHub tests | REST read/write helpers with mocked HTTP |

Providers are mocked in automated tests; Supervisor and agent pipelines run for real where the suite requires it.

---

## Project Structure

```
Project/
├── app/
│   ├── main.py                 # CLI entry point
│   ├── report_formatter.py     # Terminal report formatting
│   ├── service.py              # Shared non-interactive orchestration
│   ├── ui_reports.py           # Streamlit report renderers
│   ├── ui_history.py           # Capped run-history load/save helpers
│   ├── ui_memory.py            # Shared ConversationMemory helpers (CLI + Streamlit)
│   ├── ui_export.py            # Markdown/JSON download helpers
│   ├── worker.py               # Isolated agent job subprocess (+ progress NDJSON)
│   └── streamlit_app.py        # Streamlit web UI entry point
├── run_ui.bat                  # Launch Streamlit UI
├── run_mcp.bat                 # Launch MCP stdio server
├── Project.ipynb               # End-to-end notebook demo
├── docs/
│   ├── architecture.excalidraw
│   ├── architecture.png
│   └── images/                 # Screenshot drop zone
└── codebase_assistant/
    ├── config.py
    ├── supervisor.py
    ├── agents/                 # Analysis, Documentation, Testing
    ├── analysis/               # StaticAnalyzer, GroundingChecker
    ├── rag/                    # Chunk → Embed → Store → Retrieve
    ├── tools/                  # Filesystem, GitHub, ToolRegistry
    ├── models/                 # LLMClient + OpenRouter/Ollama providers
    ├── memory/                 # ConversationMemory + MemoryStore
    ├── tracing/                # Tracer + TraceEvent
    ├── mcp/                    # In-process MCP + stdio FastMCP bridge
    ├── schemas/
    ├── tests/
    └── requirements.txt
```

---

## Current Limitations

- Static and LLM analysis focus on **Python** repositories (Markdown/text are indexed, but bug detection targets `.py`).
- Repository ingestion respects configured ceilings (default: 100 files, 20k LOC, 500 KB per file).
- Generated tests can fail or need manual refinement for complex modules.
- Cross-encoder reranking is optional and **disabled by default**.
- MCP **HTTP / SSE** remote transport is not implemented (stdio + in-process only).
- Long MCP agent runs may hit host client timeouts.
- GitHub **write** operations require authentication via `GITHUB_TOKEN`.
- Docker files exist as deployment scaffolding and are not yet a polished one-command production stack.
- Supervisor routing is keyword/goal based, not full LLM task planning.

---

## Future Work

- LLM-driven planning and task decomposition in the Supervisor
- Broader language support beyond Python bug detection
- Richer coverage measurement for generated tests
- Production-ready Docker / Compose deployment (Streamlit and/or Jupyter + Ollama)
- Streamlit LLM token streaming (job cancel/kill already shipped)
- CI integration for repository checks on pull requests
- Deeper semantic code search and explanation skills
- Remote MCP transport (HTTP / SSE) and optional Docker MCP service

---

## License

No `LICENSE` file is currently published in this repository. Treat the project as source-available for internship/portfolio review unless the repository owner adds an explicit license.
