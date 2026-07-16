# AgenticAI Intern — Huzaifa Saboor

Weekly assignments completed as part of the Agentic AI internship.

## Repository Structure

```
AgenticAI-Intern-Huzaifa-Saboor/
├── README.md
├── prompts.md
├── Week1/          # MNIST digit classifier
├── Week2/          # OpenRouter CLI chat client
└── Week3/          # RAG pipeline, embedding comparison, structured output & validation
```

## Prerequisites

- Python 3.10 or higher
- `pip` and a virtual environment tool (`venv` or `uv`)
- Jupyter (Notebook, JupyterLab, or VS Code Jupyter extension)
- An [OpenRouter](https://openrouter.ai) API key — required for Week 2 and Week 3

## Installation

```bash
git clone <repo-url>
cd AgenticAI-Intern-Huzaifa-Saboor
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

Each week has its own dependencies — install them from inside the week's folder.

---

## Week 1 — MNIST Classifier

Trains and evaluates multiple classifiers (Decision Tree, Naive Bayes, SVM, Random Forest) on the MNIST digit dataset. Includes hyperparameter tuning via GridSearchCV, class-wise metrics, and both raw and normalized confusion matrices.

```bash
cd Week1
pip install -r requirements.txt
jupyter notebook mnist_classifier.ipynb
```

Run all cells top to bottom. Expected outputs:
- Macro-F1 bar plot comparing all four classifiers
- GridSearchCV best parameters printed to output
- `classification_report` with per-digit precision, recall, and F1
- Raw and normalized confusion matrix heatmaps

---

## Week 2 — OpenRouter CLI Chat Client

Implements a reusable OpenRouter API client (`OpenRouterClient`, `OpenRouterAPIError`, `load_api_key()`) with shared constants in `CONSTANTS.py`. The client is imported directly into Week 3.

```bash
cd Week2
pip install -r requirements.txt
cp .env.example .env        # add your OPENROUTER_API_KEY
jupyter notebook ChatApp.ipynb
```

Run all cells top to bottom, then pick a model at the startup prompt. In-chat commands: `/model`, `/history`, `/clear`, `exit`/`quit`.

Expected output: an interactive chat session with colored formatting, response time, and token usage after each reply.

---

## Week 3 — RAG Pipeline, Embedding Comparison, Structured Output & Validation

Single notebook covering five tasks, each building on the previous one.

**Setup** — copy these files from `Week2/` into `Week3/` before running:
- `ChatApp.ipynb`
- `CONSTANTS.py`
- `.env`

Place source PDFs in `Week3/data/` (any PDFs work — no filenames are hardcoded).

```bash
cd Week3
pip install pypdf sentence-transformers chromadb import-ipynb pandas pydantic requests python-dotenv jupyter
jupyter notebook Week3_Assign.ipynb
```

Restart the kernel and Run All top to bottom.

| Task | What it does |
|---|---|
| 1 — RAG Demo | PDF ingestion → chunking → embeddings → ChromaDB → retrieval → answer generation |
| 2 — Embedding Comparison | Compares `all-MiniLM-L6-v2` vs `all-mpnet-base-v2` on retrieval quality and speed |
| 3 — Structured Output | LLM output validated against a Pydantic `Answer` schema, saved as JSON |
| 4 — Validation Tests | pytest-style tests for valid and invalid schema outputs |
| 5 — Hallucination Report | See `Week3/Task5_Hallucination_Report.md` |

Expected outputs: retrieved chunks and generated answers printed inline, `comparison_df` and `speed_summary` tables, `outputs/answer_N.json` files, and a test summary (e.g. `8/8 tests passed`).

---

## Environment Variables

Week 2 and Week 3 require an OpenRouter API key:

```
OPENROUTER_API_KEY=your_key_here
```

`.env` is not committed. A `.env.example` with the expected variable name is provided in `Week2/`.

---

## Prompts Log

See [`prompts.md`](./prompts.md) for AI prompts used during each week's work.
