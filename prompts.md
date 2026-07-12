## week 1 — ai/ml foundations

---

### setting up the environment

**prompt:**
"how do i install uv and set up a venv with it and then install numpy pandas and scikit-learn"

**result:**
got the commands, ran them, everything installed fine. venv is active and packages are working.

---

### loading mnist

**prompt:**
"give me the code to load mnist using fetch_openml and check the shape of X and y"

**result:**
got the fetch_openml call, printed X.shape and y.shape. X is (70000, 784) and y is (70000,). labels came back as strings which i didnt expect.

---

### fixing string labels

**prompt:**
"why are my mnist labels strings and how do i convert them to integers"

**result:**
got y.astype(int), turned it into encode_labels() function so its reusable and testable.

---

### writing the classifier

**prompt:**
"give me the code to train a random forest classifier on mnist and print accuracy precision and recall"

**result:**
got the full pipeline, ran it, accuracy came out around 0.97. added macro averaging for precision and recall since there are 10 classes.

---

### plotting the confusion matrix

**prompt:**
"how do i plot a confusion matrix using seaborn for a 10 class classifier and show the counts inside each cell"

**result:**
got sns.heatmap with annot=True and fmt="d". changed annot=False to True after seeing the plot was unreadable without numbers.

---

### extracting utility functions

**prompt:**
"which parts of my classifier.py code should i extract into separate functions to make it easier to unit test"

**result:**
got normalize_pixels, encode_labels and split_data as suggestions. made sense since each one does one thing and can be tested with dummy data without loading mnist.

---

### if __name__ == "__main__"

**prompt:**
"my tests were downloading mnist every time i ran pytest, how do i stop that"

**result:**
got the if __name__ == '__main__' fix. wrapped the pipeline in it and tests stopped triggering the download.

---

### setting up ruff

**prompt:**
"how do i install ruff and run it on my classifier.py file"

**result:**
ran ruff check classifier.py, it flagged a couple of unused import warnings. fixed them and got a clean pass.

---

### writing the unit tests

**prompt:**
"give me pytest unit tests for these three functions: normalize_pixels, encode_labels, split_data"

**result:**
got 6 tests total, two per function. checked range and exact values for normalize, dtype and values for encode, shapes and total count for split. all 6 passing.

### normalized confusion matrix

**prompt:**
"add a normalized confusion matrix, basically divide each row by its total so values are between 0 and 1 instead of raw counts. plot both the normal one and normalized one so we can compare"

**result:**
got a plot_confusion_matrix function with a normalize flag. when true it row divides by class totals. both versions plotted at the end, normalized one is way easier to read for spotting which digits get confused.

---

### class wise metrics

**prompt:**
"add classificationreport from sklearn to get precision recall and f1 for each digit separately instead of just the overall averages"

**result:**
got classification_report call after evaluation. can now see exactly which digits like 4 and 9 or 3 and 5 are harder to classify instead of just looking at one overall number.

---

### comparing multiple classifiers

**prompt:**
"compare multiple classifiers like decision tree, naive bayes, svm and random forest. train all of them and plot a barplot of their macro-f1 scores to see which one actually performs better instead of just guessing"

**result:**
got all four classifiers benchmarked on a 10k subsample since svm is too slow on full data. barplot sorted by macro-f1 with scores annotated on each bar. random forest came out on top which justified keeping it.

---

### hyperparameter tuning

**prompt:**
"add gridsearchcv to tune random forest hyperparameters like max_depth, n_estimators, min_samples_split etc. use cv=3 and macro-f1 as scoring metric then retrian the best model on full data"

**result:**
got gridsearchcv set up with the param grid and n_jobs=-1 to use all cores. best params printed after search then used to retrain on the full 56k training set. slight improvement in macro-f1 over default params.

# week 2 — llm foundations

**prompt:**
"should i download ollama on my laptop which has integrated gpu"

**result:**
Yes, you can absolutely install Ollama on a laptop with only an integrated GPU. Ollama can run models on the CPU if you don't have a dedicated NVIDIA GPU.

---

**prompt:**
"i have 16gb ram.what will i be able to run"

**result:**
With 16 GB RAM, you should be able to run both Llama 3 8B and Mistral 7B using Ollama, even if your laptop only has an integrated GPU.

---

**prompt:**
"do i run these commands ollama run llama3 and ollama run mistral in virtual env"

**result:**
No. You do not run the Ollama commands inside your Python virtual environment.

---

**prompt:**
"Compare outputs of 3 models on the same prompt via OpenRouter.How to do this tell me step by step"

**result:**
gpt told me to sign up for OpenRouter and select any three models and use the same prompt on them and gave me step by step walkthrough

---

**prompt:**
"give me three free models exact name which i should use"

**result:**
gave me names of three models i can use for without buying credits

---

**prompt:**
this is my week 2 assignment for internship

Install Ollama and run llama3 + mistral locally

Compare outputs of 3 models on the same prompt via OpenRouter

Build a CLI chat app using the OpenRouter API (Python or Node)

Document a model comparison: speed, quality, use-case fit

Update prompts.md

i have done first and second part

now for third task

i want to add three models to the python code where the user inputs which model he/she wants to use and then give me code for the ✅ Build a CLI chat app using the OpenRouter API (Python or Node) preferably python and then give me Document a model comparison: speed, quality, use-case fit. My mentor expects me to use every feature so add everything relevant in both code and document. Well formatted and commented code and give me back the code only not the python file

give me a prompt for this

**result:**
recieved a long prompt for building the ChatApp with many details regarding features,error handling and api key management.

---

**prompt:**
"modify the cli chat app so that it supports these three models: llama 3.1 8b instruct, gemma 3 27b it and nemotron nano. the user should be able to choose which model to use and switch between them during runtime."

**result:**
The CLI chat application was updated to support multiple LLMs through OpenRouter. A model selection menu was added, allowing users to choose between Llama 3.1 8B Instruct, Gemma 3 27B IT and Nemotron Nano at startup and switch models during runtime.

---

**prompt:**
"add api key management using a .env file and make sure the project follows good secrets hygiene practices."

**result:**
API key management was implemented using a `.env` file and the `python-dotenv` package. A `.env.example` template file was created, and `.gitignore` was updated to prevent sensitive API keys from being committed to GitHub.

---

**prompt:**
"im using openrouter for my assignment. add response timing, token usage statistics and retry logic for rate limit errors."

**result:**
The application was enhanced with response timing metrics, token usage tracking and automatic retry logic with exponential backoff to handle rate limit errors and improve reliability.

---

**prompt:**
"im submitting this project for an internship. improve the code quality by adding type hints, comments, docstrings and better project structure."

**result:**
The codebase was refactored to include type hints, comprehensive comments, function docstrings, improved error handling and a more organized project structure to meet professional development standards.

---

**prompt:**
"create a markdown document comparing multiple llms based on speed, quality and use case fit. include tables and conclusions."

**result:**
A detailed markdown report was created comparing multiple LLMs based on response speed, output quality and use-case suitability. The document includes comparison tables, analysis sections and conclusions derived from testing.

## week 3 — rag pipeline assignment

---

### building the rag notebook

**prompt:**
"Build a RAG demo over PDFs using ChromaDB, sentence-transformers, and pypdf,
reusing the OpenRouter client from ChatApp.ipynb. [full spec of requirements]"

**result:**
Claude built the full pipeline in one pass — PDF discovery, chunking,
embedding, ChromaDB storage, retrieval, and the reused OpenRouter call —
and tested it end-to-end before I reviewed it.

---

### simplifying the notebook

**prompt:**
"make the markdown text and comments briefer, it looks too complex"

**result:**
Condensed all markdown explanations and docstrings while keeping the same
logic.
---

### cleaning up task 1 formatting

**prompt:**
asked to shrink the markdown text and code comments in my week3 notebook since it looked too dense/complex for my mentor to review — wanted headers condensed and comments trimmed without touching the actual logic.

**result:**
got the notebook back with shorter markdown sections and one-line comments/docstrings instead of full paragraph explanations. diffed it against my original to confirm the functions and logic were untouched, just the formatting.

---

### task 2 — embedding model comparison

**prompt:**
gave a detailed spec: compare two sentence-transformers models on retrieval quality, reuse task 1's chunk_records so the only thing changing is the embedding model, build one chromadb collection per model, run the same eval questions against both, and output a comparison table with specific columns (question, model, retrieved source, similarity score, retrieval time, generated answer, manual relevance rating).

**result:**
got build_and_time_collection() and evaluate_model_on_question(), plus a pandas comparison_df and a speed_summary table. reviewed the functions to make sure task 1 code wasn't duplicated. still had to actually run it myself, fill in the manual relevance ratings by hand, and write the analysis based on my real numbers since that part's a judgment call, not something to generate.

---

### task 3 — structured output pipeline

**prompt:**
spec'd out an llm → json → validate → save pipeline. gave the pydantic schema fields i wanted (question, answer, confidence, sources), asked for it split into build_prompt / generate_json / validate_output / save_json, and was explicit that the existing openrouter client shouldn't be touched.

**result:**
got the full pipeline plus a demo showing the validation-failure path on purpose. reviewed it and ran it — hit a UnicodeEncodeError on save_json on my windows machine (cp1252 default encoding couldn't handle a character in one of the answers). fixed that myself by adding encoding="utf-8" to the write_text call.

---

### task 4 — validation tests

**prompt:**
asked for pytest-style tests covering both valid and invalid llm outputs specifically — missing required field, wrong datatype, confidence out of the 0–1 range, sources not being a list, completely invalid json, and extra unexpected fields — organized as reusable test_ functions that reuse task 3's validate_output instead of rewriting it.

**result:**
got a TEST_CASES dict plus individual test_ functions and a small runner (since actual pytest doesn't execute notebook cells directly). double-checked the extra-fields case myself since my Answer schema doesn't forbid extra fields by default — pydantic just silently ignores them — so that case documents that behavior instead of faking it as a failure.

---

### task 5 — hallucination report

**prompt:**
asked for a markdown report on where the llm hallucinated in my pipeline and how i actually caught it, tied to my real implementation, with sections for scenarios observed, detection methods, and mitigation techniques.

**result:**
got a report grounded in my actual pipeline — retrieval inspection, similarity scores, pydantic validation, the task 4 tests. read through it and checked the scenarios against what i'd actually seen while testing before keeping them in.

---

### gitignore review

**prompt:**
asked if my existing .gitignore was missing anything for this kind of project.

**result:**
got suggestions — mypy cache, logs/coverage, env file variants — plus a flag about whether outputs/ should stay tracked so my saved json answers are visible for review. turned out i'd already left it untracked, so no change needed there, just added the extra entries.