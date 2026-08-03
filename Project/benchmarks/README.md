# Benchmark Suite

Reproducible evaluation harness for the AI Software Engineering Assistant.

The suite measures the **existing** Supervisor / agent pipelines. It does not
modify production agent logic.

## How to run

From the `Project/` directory:

```bash
# Built-in dataset (demo, medium, clean, unsupported)
python benchmarks/run_benchmark.py --dataset

# Single local repository
python benchmarks/run_benchmark.py examples/demo_repo

# GitHub URL (cloned to a temporary workspace, then cleaned up)
python benchmarks/run_benchmark.py https://github.com/owner/repo

# Live providers (requires OPENROUTER_API_KEY / optional Ollama)
python benchmarks/run_benchmark.py examples/demo_repo --mode live
```

Default mode is **`offline`**: a deterministic mock LLM is injected into the
Supervisor agents so documentation and testing metrics stay reproducible
without network access. Static analysis still runs for real.

Outputs land in `benchmarks/results/`:

- `latest.json` — full report
- `latest.csv` — flattened summary
- `<benchmark_id>.json` — stamped copy

## Built-in dataset

| Case | Path | Purpose |
|---|---|---|
| `demo_repo` | `examples/demo_repo` | Seeded bugs for analysis recall-style checks |
| `medium_repo` | `examples/medium_repo` | Multi-module package for timing / indexing load |
| `clean_repo` | `examples/clean_repo` | Almost no findings |
| `unsupported_repo` | `examples/unsupported_repo` | No Python files (abstention path) |

Fixtures are created automatically if missing.

## Metrics collected

### Code Analysis
- `analysis_latency_seconds` / `total_runtime_seconds`
- `repository_indexing_time_seconds`
- `retrieval_latency_ms`, `model_latency_ms`
- `static_findings`, `llm_findings`, `grounded_findings`
- `hallucinations_rejected`
- abstention flags when present

### Documentation
- `documentation_generation_time_seconds`
- `document_length_chars`
- `functions_modules_documented`
- `abstention_rate`
- `repository_summary_produced`

### Testing
- `generation_time_seconds`, `execution_time_ms`
- `generated_test_files`
- `passed_tests`, `failed_tests`, `skipped_tests`, `execution_errors`

### RAG
- `indexing_time_seconds`
- `chunks_generated`
- `retrieved_chunks`
- `retrieval_latency_ms`

### Overall
- `total_pipeline_runtime_seconds`
- `per_agent_runtime_seconds`
- `memory_usage_mb` (best-effort; `psutil` or `resource` when available)
- `tracing_event_count`

## How metrics are calculated

1. The runner builds an isolated `Supervisor` (dedicated Chroma / memory paths).
2. It calls the real `handle_task` pipelines for analysis, documentation, and testing.
3. Wall-clock times are measured around each call.
4. Stage latencies (indexing / retrieval / model / pytest) are summed from
   Tracer event `duration_ms` values emitted by the existing agents.
5. Finding counts come from `CodeAnalysisReport` fields (`findings`,
   `rejected`, detection methods).
6. Test pass/fail counters are parsed from `TestingResult.summary`
   (`Execution: N passed, ...`) produced by the Testing Agent.
7. Results are serialized to deterministic JSON (`sort_keys=True`).

## Limitations

- Offline mode does **not** evaluate live LLM quality; it evaluates pipeline
  timing, static analysis, grounding bookkeeping, RAG wiring, and execution
  of generated (mock) tests.
- Live mode depends on provider availability, rate limits, and non-deterministic
  model output — treat it as exploratory, not a regression gate.
- Memory usage is best-effort and may be `null` on platforms without
  `psutil` / `resource` RSS support.
- Cross-encoder reranking is disabled during benchmarks for stable timing.
- GitHub URL runs require network access and `git` / GitPython.

## Tests

```bash
PYTHONPATH=. python -m pytest codebase_assistant/tests/test_benchmarks.py -q
```
