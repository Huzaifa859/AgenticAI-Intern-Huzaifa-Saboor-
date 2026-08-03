# Notebooks

## Model_Comparison.ipynb

Side-by-side evaluation of multiple OpenRouter models on the same repository
and the same Codebase Assistant tasks (analysis, documentation, testing).

### Run

From the `Project/` directory:

```bash
pip install matplotlib pandas
jupyter notebook notebooks/Model_Comparison.ipynb
```

Requires `OPENROUTER_API_KEY` in `Project/.env`.

### Notes

- Evaluation helpers live in `compare_models_helpers.py`.
- Production agent pipelines are not modified.
- Each model is pinned (no OpenRouter fallback chain) for fair attribution.
- Unavailable models are skipped with a recorded reason.
