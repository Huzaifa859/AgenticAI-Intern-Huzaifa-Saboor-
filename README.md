# AgenticAI Intern — Huzaifa Saboor

This repository contains my weekly assignments completed during the Agentic AI Internship. Each week's work is organized into its own folder with the required notebooks, supporting files, and setup instructions.

## Repository Structure

```
AgenticAI-Intern-Huzaifa-Saboor-/
├── README.md
├── prompts.md
├── requirements.txt
├── MNISTclassifier/
├── Week2-CLIChatApp/
└── Week3Assignment/
```

| Folder | Description |
|--------|-------------|
| `MNISTclassifier/` | Week 1 – MNIST classifier and model evaluation |
| `Week2-CLIChatApp/` | Week 2 – OpenRouter CLI Chat Application |
| `Week3Assignment/` | Week 3 – RAG pipeline, embedding comparison, structured outputs, and validation |

---

## Prerequisites

- Python 3.10 or higher
- `pip`
- A virtual environment tool (`venv`)
- Jupyter Notebook, JupyterLab, or the VS Code Jupyter extension
- An OpenRouter API key (required for Week 2 and Week 3)

---

## Repository Setup

```bash
git clone https://github.com/Huzaifa859/AgenticAI-Intern-Huzaifa-Saboor-.git
cd AgenticAI-Intern-Huzaifa-Saboor-
python -m venv .venv
```

Activate the virtual environment:

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

To reproduce any assignment, navigate to the corresponding week's folder, install its dependencies, and run the notebook from top to bottom.

---

# Week 1 — MNIST Classifier

**Folder:** `MNISTclassifier/`

This assignment trains and evaluates multiple machine learning classifiers on the MNIST handwritten digit dataset. It includes model comparison, hyperparameter tuning using GridSearchCV, and evaluation using classification metrics and confusion matrices.

### Run

```bash
cd MNISTclassifier
pip install -r ../requirements.txt
jupyter notebook mnist_classifier.ipynb
```

Run all notebook cells from top to bottom.

### Expected Output

- Model comparison metrics
- Hyperparameter tuning results
- Classification report
- Raw and normalized confusion matrices

---

# Week 2 — OpenRouter CLI Chat Application

**Folder:** `Week2-CLIChatApp/`

This assignment implements a reusable OpenRouter API client and an interactive notebook-based CLI chat application supporting multiple models, conversation history, response timing, and token usage.

### Setup

Create a `.env` file from `.env.example` and add your OpenRouter API key.

```
OPENROUTER_API_KEY=your_api_key_here
```

### Run

```bash
cd Week2-CLIChatApp
pip install -r requirements.txt
jupyter notebook ChatApp.ipynb
```

Run all notebook cells from top to bottom.

### Expected Output

- Interactive chat session
- Model switching
- Conversation history
- Response time and token usage statistics

---

# Week 3 — RAG Pipeline & Structured Output

**Folder:** `Week3Assignment/`

This assignment contains five tasks:

- RAG demo using PDF documents and ChromaDB
- Embedding model comparison
- Structured JSON output using Pydantic
- Validation tests
- Hallucination analysis and report

### Before Running

Copy the following files from `Week2-CLIChatApp/` into `Week3Assignment/`:

- `ChatApp.ipynb`
- `CONSTANTS.py`
- `.env`

Place the PDF documents inside:

```
Week3Assignment/data/
```

### Run

```bash
cd Week3Assignment
pip install pypdf sentence-transformers chromadb import-ipynb pandas pydantic requests python-dotenv jupyter
jupyter notebook Week3_Assign.ipynb
```

Restart the kernel and run all notebook cells from top to bottom.

### Expected Output

- Retrieved document chunks and generated answers
- Embedding comparison tables
- Validated JSON output files
- Validation test results
- Hallucination report

---

# Environment Variables

Week 2 and Week 3 require an OpenRouter API key.

```
OPENROUTER_API_KEY=your_api_key_here
```

The `.env` file is intentionally excluded from version control. A `.env.example` file is provided in `Week2-CLIChatApp/`.

---

# Prompts Log

The prompts used throughout the internship are documented in `prompts.md`.