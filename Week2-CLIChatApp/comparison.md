# Model Comparison Report: Llama 3.1 8B Instruct vs. Gemma 3 27B IT vs. Nemotron Nano 9B V2

**Internship Project — Week 2: AI/LLM Model Comparison**
**Author:** *Huzaifa Saboor*
**Date:** *2/7/2026*

> **How to use this report:** This document is fully structured and ready to
> commit to GitHub. Cells marked `_TBD_` are the metrics you should fill in
> after running your own tests with `openrouter_chat.py` — timing and
> quality scores depend on your specific hardware, network, and prompts, so
> they're left for you to measure rather than guessed. Everything else
> (model descriptions, prompts, methodology, and analysis framework) is
> complete and based on each model's published documentation.
>
> **Note on model selection:** the original assignment listed
> `nvidia/llama-3.1-nemotron-nano-8b-v1:free`, but that model has since
> been retired from OpenRouter (confirmed via a 404 "model not found"
> response and an empty search result on openrouter.ai/models). It has
> been replaced here with its direct successor, `nvidia/nemotron-nano-9b-v2:free`,
> which is NVIDIA's next-generation Nemotron Nano model.

---

## 1. Introduction

### 1.1 Purpose of the Experiment

This experiment evaluates and compares three large language models (LLMs)
available through OpenRouter — **Llama 3.1 8B Instruct**, **Gemma 3 27B
IT**, and **Nemotron Nano 9B V2** — across speed, output quality, and
practical use-case fit. The goal is to build hands-on intuition for how
model size, architecture, and provider-side tuning affect real-world
chatbot behavior, and to produce a reference that can inform model
selection for future projects.

### 1.2 What Is OpenRouter?

[OpenRouter](https://openrouter.ai) is a unified API gateway that provides
access to hundreds of LLMs from many different providers (Meta, Google,
NVIDIA, Anthropic, OpenAI, Mistral, and more) through a single,
OpenAI-compatible endpoint. Instead of managing separate API keys, SDKs,
and request formats for each provider, a developer sends requests to
`https://openrouter.ai/api/v1/chat/completions` and simply changes the
`model` field to switch providers. OpenRouter also handles routing between
multiple hosting providers for the same model, exposes consistent
pricing/usage data, and offers free-tier models (like the `:free` Nemotron
variant used here) for experimentation without cost.

---

## 2. Models Evaluated

| Model | Identifier | Publisher | Parameters | Context Window |
|---|---|---|---|---|
| Llama 3.1 8B Instruct | `meta-llama/llama-3.1-8b-instruct` | Meta | 8B | 128K tokens |
| Gemma 3 27B IT | `google/gemma-3-27b-it` | Google DeepMind | ~27.4B | 128K tokens |
| Nemotron Nano 9B V2 | `nvidia/nemotron-nano-9b-v2:free` | NVIDIA | 9B | 128K tokens |

**Llama 3.1 8B Instruct** is Meta's compact, general-purpose
instruction-tuned model from the Llama 3.1 family — designed to be fast
and inexpensive while remaining competent at everyday chat, summarization,
and light coding tasks.

**Gemma 3 27B IT** is Google's largest Gemma 3 variant, a multimodal
(text + image input, text output) model trained on Gemini technology. It
supports over 140 languages and includes function-calling and structured
output support, trading some speed for a significantly larger parameter
count and stronger reasoning/coding capability.

**Nemotron Nano 9B V2** is NVIDIA's next-generation Nemotron Nano model,
trained from scratch (rather than derived from an existing Llama
checkpoint like its predecessor) as a unified reasoning and non-reasoning
model. By default it produces a reasoning trace before its final answer;
this behavior can be controlled via the system prompt if only the final
answer is needed. It offers a free tier on OpenRouter, making it
attractive for lightweight, cost-sensitive applications — this internship
uses the `:free` endpoint.

---

## 3. Experimental Setup

### 3.1 Hardware Specifications

| Component | Value |
|---|---|
| Machine | *Lenovo Ideapad Slim 5 16IRL8* |
| CPU | *i7-13700H , 2400Mhz , 14 Core(s)* |
| RAM | *16 GB* |
| GPU (if used for local Ollama models) | *Intel Iris(R) XE Graphics* |
| OS | *Windows 11* |

### 3.2 Local Environment

- Python version: *3.13.5*
- Key packages: `requests`, `python-dotenv` (see `requirements.txt`)
- Ollama version (from Week 2 local task): *0.31.1*
- Network: *PTCL, 30 Mbps* — network latency directly
  affects OpenRouter response times, so it's worth noting here.

### 3.3 OpenRouter Environment

- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Auth: Bearer token via `OPENROUTER_API_KEY` (see [API Key Management](#api-key-management-recap))
- Client: this project's custom `ChatApp.py` CLI (Part 1 of this
  submission)
- Request format: default OpenRouter routing (no pinned provider), no
  system prompt unless noted (e.g., Nemotron reasoning trace behavior)

### 3.4 Evaluation Methodology

Each model was sent the same five prompts (see [Section 4](#4-prompts-used)),
one per category, in a fresh conversation (history cleared between tests to
avoid context-length bias). For each response, the following was recorded:

1. **Response time** — measured by `ChatApp.py` itself (wall-clock
   time from request sent to response received), displayed after every
   reply.
2. **Token usage** — `prompt_tokens` / `completion_tokens` / `total_tokens`
   as reported by the OpenRouter API response.
3. **Qualitative scoring** — each response manually rated 1–5 on the
   criteria in [Section 5](#5-evaluation-metrics), using the same rubric
   across all three models for fairness.

All three models were tested in the same session/day where possible to
minimize variance from provider-side load fluctuations.

---

## 4. Prompts Used

| Category | Prompt |
|---|---|
| **Explanation** | "Explain how DNS (Domain Name System) works, as if teaching a first-year computer science student." |
| **Coding** | "Write a Python function that takes a list of integers and returns the two numbers that sum to a given target, along with a brief explanation of the time complexity." |
| **Reasoning** | "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Explain your reasoning step by step." |
| **Summarization** | "Summarize the following paragraph in 2 sentences: *LeBron Raymone James Sr. (/ləˈbrɒn/ lə-BRON;[1] born December 30, 1984) is an American professional basketball player who most recently played for the Los Angeles Lakers of the National Basketball Association (NBA). Nicknamed "King James", he is the NBA's all-time leading scorer and has won four NBA championships from 10 NBA Finals appearances, including eight consecutive appearances between 2011 and 2018.[2] He has won three Olympic gold medals as a member of the U.S. national team, and is widely considered to be one of the greatest basketball players of all time and one of the greatest athletes in modern history.[a]

In addition to ranking fourth in NBA career assists and sixth in NBA career steals, James holds several individual honors, including four NBA MVP awards, four Finals MVP awards, the Rookie of the Year award, three All-Star Game MVP awards, the inaugural NBA Cup MVP, and the Olympics MVP in the 2024 Summer Olympics. A record 22-time All-Star and 21-time All-NBA selection (including a record 13 First Team selections), he has also made six All-Defensive Teams. The oldest active player in the NBA, he holds records for the most seasons played with 23, the most games played, the most minutes played, and the most field goals made in league history.
*." |
| **Creative Writing** | "Write a short (100-word) story about a robot who discovers music for the first time." |

> Tip: Keep the exact same prompt text across all three models so the
> comparison is apples-to-apples. Paste your actual summarization source
> paragraph into the placeholder above before committing.

---

## 5. Evaluation Metrics

Each response was scored on a **1 (poor) – 5 (excellent)** scale across:

| Metric | Description |
|---|---|
| Response Speed | Time from request to full response |
| Accuracy | Factual correctness of the answer |
| Reasoning Quality | Logical soundness, step-by-step clarity |
| Coding Ability | Correctness, readability, and efficiency of code |
| Creativity | Originality and quality of creative output |
| Instruction Following | Did it do exactly what was asked? |
| Response Completeness | Did it fully answer, or cut off / omit parts? |
| Cost Efficiency | Value relative to token cost (see OpenRouter pricing) |
| Ease of Use | Consistency, formatting, need for follow-up prompting |
| Overall Quality | Holistic impression |

---

## 6. Comparison Tables

### 6.1 Response Time by Prompt Category (seconds)

| Category | Llama 3.1 8B Instruct | Gemma 3 27B IT | Nemotron Nano 9B V2 |
|---|---|---|---|
| Explanation | _TBD_ | _TBD_ | _TBD_ |
| Coding | _TBD_ | _TBD_ | _TBD_ |
| Reasoning | _TBD_ | _TBD_ | _TBD_ |
| Summarization | _TBD_ | _TBD_ | _TBD_ |
| Creative Writing | _TBD_ | _TBD_ | _TBD_ |
| **Average** | **_TBD_** | **_TBD_** | **_TBD_** |

### 6.2 Token Usage by Prompt Category (total tokens)

| Category | Llama 3.1 8B Instruct | Gemma 3 27B IT | Nemotron Nano 9B V2 |
|---|---|---|---|
| Explanation | _TBD_ | _TBD_ | _TBD_ |
| Coding | _TBD_ | _TBD_ | _TBD_ |
| Reasoning | _TBD_ | _TBD_ | _TBD_ |
| Summarization | _TBD_ | _TBD_ | _TBD_ |
| Creative Writing | _TBD_ | _TBD_ | _TBD_ |

### 6.3 Quality Scores (1–5 scale)

| Metric | Llama 3.1 8B Instruct | Gemma 3 27B IT | Nemotron Nano 9B V2 |
|---|---|---|---|
| Accuracy | _TBD_ | _TBD_ | _TBD_ |
| Reasoning Quality | _TBD_ | _TBD_ | _TBD_ |
| Coding Ability | _TBD_ | _TBD_ | _TBD_ |
| Creativity | _TBD_ | _TBD_ | _TBD_ |
| Instruction Following | _TBD_ | _TBD_ | _TBD_ |
| Response Completeness | _TBD_ | _TBD_ | _TBD_ |
| Ease of Use | _TBD_ | _TBD_ | _TBD_ |
| **Overall Quality** | **_TBD_** | **_TBD_** | **_TBD_** |

### 6.4 Published Model Specs (reference, not measured)

| Spec | Llama 3.1 8B Instruct | Gemma 3 27B IT | Nemotron Nano 9B V2 |
|---|---|---|---|
| Parameters | 8B | ~27.4B | 9B |
| Context Window | 128K | 128K | 128K |
| Modality | Text only | Text + image input | Text only |
| Reasoning Mode | Standard | Standard | Unified model; reasoning trace on by default, toggleable via system prompt |
| Free Tier on OpenRouter | No (paid) | No (paid) | Yes (`:free` variant) |
| Local Deployment | Ollama-friendly | Larger footprint | Compact — suited to local/edge deployment |

---

## 7. Detailed Analysis

### 7.1 Llama 3.1 8B Instruct

**Strengths**
- Fast and lightweight — the smallest of the three by parameter count.
- Solid general-purpose instruction following for everyday chat tasks.
- Well-documented, widely supported (also runnable locally via Ollama,
  consistent with the Week 1 task).

**Weaknesses**
- As an 8B model, reasoning depth and coding accuracy on harder problems
  lag behind the larger Gemma 3 27B.
- Less specialized than Nemotron Nano for explicit step-by-step reasoning.

**Best Use Cases:** general chat assistants, FAQ bots, low-latency
applications, and any scenario where cost and speed matter more than
frontier-level reasoning.

**Performance Observations:** *[Fill in after testing — e.g., "Consistently
the fastest of the three, but occasionally gave shorter/less detailed
answers on the reasoning prompt."]*

### 7.2 Gemma 3 27B IT

**Strengths**
- Largest model in this comparison (~27.4B parameters), which typically
  translates to stronger reasoning, coding, and instruction-following.
- Multimodal — can accept image input in addition to text (not exercised
  in this text-only CLI, but relevant for future extensions).
- Strong multilingual support (140+ languages) and structured-output /
  function-calling support.

**Weaknesses**
- Larger model size generally means higher latency and token cost than
  the two smaller models.
- Overkill for simple queries where a smaller model would suffice.

**Best Use Cases:** coding assistance, complex reasoning, multilingual
applications, and tasks that would benefit from future image-input
support.

**Performance Observations:** *[Fill in after testing — e.g., "Produced the
most complete and well-structured code, but had the highest average
response time."]*

### 7.3 Nemotron Nano 9B V2

**Strengths**
- Trained from scratch by NVIDIA as a unified reasoning and non-reasoning
  model — it can produce a step-by-step reasoning trace before its final
  answer, or skip straight to the answer when configured to via the
  system prompt.
- Free tier available on OpenRouter, making it ideal for
  budget-conscious development and testing.
- Compact (9B parameters) relative to Gemma 3 27B, keeping latency and
  cost low while still targeting stronger reasoning than a standard
  instruction-tuned model of similar size.

**Weaknesses**
- Because reasoning traces are on by default, responses can run longer
  (and use more output tokens) than a same-size non-reasoning model unless
  explicitly configured otherwise.
- As a free-tier endpoint, it may be subject to stricter rate limits or
  variable latency versus paid models.
- Being a newer model family (superseding the original Llama-3.1-derived
  Nemotron Nano 8B v1), published third-party benchmarks are less
  extensive than for more established models.

**Best Use Cases:** reasoning-heavy tasks on a budget, RAG pipelines,
tool-calling/agentic workflows, and prototyping where cost must stay at
or near zero.

**Performance Observations:** *[Fill in after testing — e.g., "Reasoning
prompt answer included a visible reasoning trace by default, which was
more thorough than Llama 3.1 8B's direct answer."]*

---

## 8. Speed Analysis

*[Fill in based on Section 6.1 data. Suggested structure:]*

- Fastest model overall: _TBD_
- Slowest model overall: _TBD_
- Category with the largest speed gap between models: _TBD_
- Hypothesis for observed differences (parameter count, provider load,
  free-tier vs. paid routing, reasoning-trace overhead, network
  conditions): _TBD_

## 9. Quality Analysis

*[Fill in based on Section 6.3 data. Suggested structure:]*

- Highest-scoring model overall: _TBD_
- Model with the best coding output: _TBD_
- Model with the best reasoning output: _TBD_
- Any surprising results (e.g., a smaller model outperforming a larger one
  on a specific task): _TBD_

## 10. Use-Case Fit Analysis

| Use Case | Best-Fit Model | Rationale |
|---|---|---|
| General Chat | Llama 3.1 8B Instruct | Fast, capable, low cost for everyday conversation |
| Coding | Gemma 3 27B IT | Larger model, generally stronger code generation/reasoning |
| Reasoning | Nemotron Nano 9B V2 | Unified model with a built-in reasoning trace by default |
| Summarization | *[fill in based on your results]* | *[TBD]* |
| Creative Tasks | *[fill in based on your results]* | *[TBD]* |
| Lightweight Applications | Nemotron Nano 9B V2 or Llama 3.1 8B Instruct | Both are compact, low-cost models suited to local/edge or budget-conscious deployment |

---

## 11. Final Recommendations

For real-world application development, consider:

1. **Start small, scale up only when needed.** Llama 3.1 8B Instruct or
   Nemotron Nano 9B V2 are reasonable defaults for most chat and
   lightweight-reasoning applications; reserve Gemma 3 27B IT for tasks
   that specifically demand deeper reasoning, coding accuracy, or
   multilingual/multimodal input.
2. **Match the model to the task, not the other way around.** Route
   reasoning-heavy or tool-calling workloads to Nemotron Nano (letting its
   default reasoning trace run), coding/complex-analysis workloads to
   Gemma 3 27B, and general conversational traffic to Llama 3.1 8B.
3. **Take advantage of OpenRouter's flexibility.** Because switching
   models is a one-line change (as demonstrated by the `/model` command in
   this project's CLI app), it's practical to A/B test models in
   production or route dynamically based on request type and cost budget.
4. **Watch free-tier limitations — and model deprecations.** The `:free`
   Nemotron endpoint is great for development, but production systems
   should verify rate limits and consider a paid tier or self-hosted
   deployment for reliability. It's also worth periodically checking that
   pinned model IDs are still active on OpenRouter; providers do retire
   older model versions (as happened with the original Nemotron Nano 8B
   v1 used in earlier drafts of this project).

## 12. Conclusion

This experiment compared three OpenRouter-hosted models — Llama 3.1 8B
Instruct, Gemma 3 27B IT, and Nemotron Nano 9B V2 — across a shared set of
prompts spanning explanation, coding, reasoning, summarization, and
creative writing. Combined with a custom Python CLI chat client built for
this internship (see `ChatApp.py`), the exercise reinforced two
practical lessons: first, that model size and specialization trade off
against speed and cost in predictable ways, and second, that a
provider-agnostic gateway like OpenRouter makes it straightforward to
evaluate and switch between models as application requirements evolve —
including recovering gracefully when a specific model ID is retired, as
happened during this project. The completed data in Section 6 should
guide model selection for future project phases, balancing response
quality against latency and budget constraints.

---

### API Key Management Recap

This project's companion CLI application (`ChatApp.py`) loads
`OPENROUTER_API_KEY` from a local `.env` file via `python-dotenv`, and
`.env` is excluded from version control via `.gitignore`. No API key is
hardcoded anywhere in this repository.
