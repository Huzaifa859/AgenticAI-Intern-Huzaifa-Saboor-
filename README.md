# AgenticAI Intern — Huzaifa Saboor

Weekly assignments completed as part of the Agentic AI internship. Each folder is self-contained with its own notebook(s) and setup instructions.

## Repository Structure

```
AgenticAI-Intern-Huzaifa-Saboor/
├── README.md
├── prompts.md              # log of prompts used with AI assistance, week by week
├── requirements.txt         # dependencies for Week 1 (MNIST classifier)
├── MNISTclassifier/         # Week 1 — MNIST digit classifier
├── Week2-CLIChatApp/        # Week 2 — OpenRouter CLI chat client
└── Week3Assignment/         # Week 3 — RAG pipeline, embedding comparison, structured output & validation
```

## Prerequisites

- Python 3.10 or higher
- `pip` and a virtual environment tool (`venv` or `uv`)
- Jupyter (Notebook, JupyterLab, or the VS Code Jupyter extension) to run the `.ipynb` files
- An [OpenRouter](https://openrouter.ai) API key — required for **Week 2** and **Week 3**, both of which call OpenRouter for LLM generation

## Installation

```bash
git clone https://github.com/Huzaifa859/AgenticAI-Intern-Huzaifa-Saboor-.git
cd AgenticAI-Intern-Huzaifa-Saboor
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

Each week installs its own dependencies (shown below) rather than one global install.

---

## Week 1 — MNIST Classifier (`MNISTclassifier/`)

Trains and evaluates multiple classifiers (Decision Tree, Naive Bayes, SVM, Random Forest) on the MNIST digit dataset. Includes hyperparameter tuning via GridSearchCV, class-wise metrics, and both raw and normalized confusion matrices.

```bash
cd MNISTclassifier
pip install -r ../requirements.txt
jupyter notebook mnist_classifier.ipynb
```

Run all cells top to bottom. Expected outputs:
- Macro-F1 bar plot comparing all four classifiers
- GridSearchCV best parameters printed to output
- `classification_report` with per-digit precision, recall, and F1
- Raw and normalized confusion matrix heatmaps

---

## Week 2 — OpenRouter CLI Chat Client (`Week2-CLIChatApp/`)

`ChatApp.ipynb` implements a reusable OpenRouter API client — `OpenRouterClient`, `OpenRouterAPIError`, `load_api_key()`, and a `MODELS` dict — with shared constants centralized in `CONSTANTS.py`. This client is imported directly into Week 3 rather than reimplemented.

**Setup & run:**
```bash
cd Week2-CLIChatApp
pip install -r requirements.txt
cp .env.example .env        # then add your OPENROUTER_API_KEY
jupyter notebook ChatApp.ipynb
```

Run all cells top to bottom, then pick a model at the startup prompt. Available in-chat commands: `/model` (switch models), `/history` (view conversation), `/clear` (reset), `exit`/`quit`.

**Expected output:** an interactive chat session printed in the notebook cell output, with colored formatting, response time, and token usage shown after each reply.

---

## Week 3 — RAG Pipeline, Embedding Comparison, Structured Output & Validation (`Week3Assignment/`)

Single notebook (`Week3_Assign.ipynb`) covering five tasks, each reusing the previous task's functions rather than duplicating them:

| Task | What it does |
|---|---|
| 1 — RAG Demo | PDF ingestion → chunking → embeddings → ChromaDB storage → top-k retrieval → grounded answer generation via the Week 2 OpenRouter client |
| 2 — Embedding Comparison | Compares `all-MiniLM-L6-v2` vs `all-mpnet-base-v2` on the same chunks: retrieval quality, similarity scores, and speed |
| 3 — Structured Output Pipeline | LLM output constrained to a Pydantic `Answer` schema (question/answer/confidence/sources), validated before saving |
| 4 — Validation Tests | pytest-style tests covering valid and invalid schema outputs |
| 5 — Hallucination Report | `Task5_Hallucination_Report.md` — observed hallucination scenarios, detection methods, mitigations |

**Setup:** the notebook imports `ChatApp.ipynb` via `import_ipynb`, so these must be copied into `Week3Assignment/` first:
- `ChatApp.ipynb`
- `CONSTANTS.py`
- `.env` (with your API key, copied over from Week 2)

Place source PDFs in `Week3Assignment/data/` (any PDFs work — no filenames are hardcoded).

**Install & run:**
```bash
cd Week3Assignment
pip install pypdf sentence-transformers chromadb import-ipynb pandas pydantic requests python-dotenv jupyter
jupyter notebook Week3_Assign.ipynb
```

Restart the kernel and **Run All** top to bottom — later tasks depend on objects created earlier in the notebook.

**Expected outputs:**
- Task 1 — retrieved chunks (source/page/similarity) and a generated answer, printed inline
- Task 2 — `comparison_df` and `speed_summary` tables, printed inline
- Task 3 — `Week3Assignment/outputs/answer_N.json` files, plus the validated JSON printed inline
- Task 4 — `PASSED`/`FAILED` per test, plus a summary count (e.g. `8/8 tests passed`)
- Task 5 — see `Week3Assignment/Task5_Hallucination_Report.md`

---

## Environment Variables

Week 2 and Week 3 both require:
```
OPENROUTER_API_KEY=your_key_here
```
`.env` is not committed (see `.gitignore`); a `.env.example` is provided in `Week2-CLIChatApp/` showing the expected variable name.

---

## Prompts Log

See [`prompts.md`](./prompts.md) for the prompts used with AI assistance during each week's work, and what was reviewed or modified afterward.