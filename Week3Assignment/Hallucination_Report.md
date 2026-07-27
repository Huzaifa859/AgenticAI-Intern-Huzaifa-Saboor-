# Task 5 – Hallucination Report

## 1. Introduction

### 1.1 What Is Hallucination?

In the context of large language models (LLMs), a *hallucination* is any output that is presented with confidence but is not actually supported by the model's input — whether that input is its training data or, in a retrieval-augmented system, the context it was explicitly given. Hallucinations range from small factual slips to entirely fabricated details, and they are especially dangerous because they are typically phrased in the same fluent, assured tone as correct answers. Nothing about the surface form of a hallucinated sentence distinguishes it from a grounded one; the only way to tell them apart is to check the claim against a source.

### 1.2 Why Hallucinations Are a Problem in RAG Systems

Retrieval-Augmented Generation (RAG) is often introduced specifically to reduce hallucination, since the model is given real, retrieved text instead of relying purely on parametric memory. However, RAG does not remove the generative step — it only changes what the model is generating *from*. The LLM can still:

- extrapolate beyond what the retrieved chunks actually say,
- blend retrieved content with unrelated knowledge from pretraining,
- answer confidently even when the retrieved context is thin or irrelevant, or
- misreport structured details such as page numbers, confidence, or source attribution.

In a pipeline like the one built for this assignment, hallucination is a problem not just because the *prose* of an answer might be wrong, but because the project also asks the model to emit **structured JSON**. A hallucination here can just as easily be a schema violation (wrong type, missing field, out-of-range value) as it can be an incorrect sentence — both are the model failing to stay faithful to a constraint it was given.

### 1.3 Why Retrieval Alone Cannot Eliminate Hallucination

Retrieval improves grounding, but it does not guarantee faithfulness for three structural reasons:

1. **Retrieval quality is imperfect.** Similarity search returns the *most similar* chunks available, not necessarily *sufficient* ones. If the collection has no chunk that actually answers the question, the top-k results can still be returned with a moderate similarity score, and the model may fill the gap with an invented answer rather than declining.
2. **The model is a text generator, not a lookup engine.** Even with correct, sufficient context, an LLM predicts the most probable next tokens — it is not mechanically constrained to only reproduce facts present in the prompt. It can still generalize, paraphrase inaccurately, or default to prior training knowledge that happens to be topically related.
3. **Instruction-following is probabilistic, not guaranteed.** Prompting the model to "answer only from context" or to "output only JSON" changes the *likelihood* of compliant behavior; it does not enforce it the way a parser or a schema does. This is precisely why this project pairs RAG with a second, independent layer of protection: Pydantic validation.

## 2. Project Implementation Overview

The pipeline built across Tasks 1–4 works as follows:

1. **Indexing** — PDFs in `data/` (Python, Machine Learning, Deep Learning, and Artificial Intelligence cheat sheets) are parsed with `pypdf`, split into overlapping chunks, embedded with a `sentence-transformers` model, and stored in **ChromaDB** with metadata (source filename, page number).
2. **Retrieval** — For a given question, the same embedding model encodes the query and ChromaDB returns the top-k most similar chunks, each carrying a cosine-similarity score.
3. **Grounded generation** — The retrieved chunks are inserted into a prompt that instructs the LLM (via the reused Week 2 OpenRouter client) to answer **only** using the supplied context, and to explicitly say so if the answer isn't present.
4. **Structured output** — Rather than free text, the model is asked to return a single JSON object matching a `Answer` schema (`question`, `answer`, `confidence`, `sources`).
5. **Validation** — The raw response is parsed and validated against the `Answer` Pydantic model before anything is trusted. Malformed or out-of-spec output is rejected outright.
6. **Persistence** — Only responses that pass validation are saved to `outputs/answer_N.json`; invalid responses are logged with their validation error and discarded.

This two-layer design — retrieval to ground the answer, validation to police the output format — is what makes hallucinations in this project *detectable and preventable* rather than something that silently corrupts downstream data.

## Hallucination Scenarios Observed

The following scenarios are drawn from the actual behavior of the pipeline while testing Tasks 1–4. They are grouped by whether the hallucination occurred at the *content* level (the answer said something unsupported) or the *structural* level (the output broke the schema).

### Scenario 1 — Insufficient Retrieved Context Led to Guessing

When asked a question adjacent to, but not directly covered by, the four source PDFs (for example, a question touching on a machine learning concept not explicit in the CS229 cheat sheet), the retrieved chunks had comparatively low similarity scores yet were still returned as the "top-k" results, since ChromaDB always returns the closest matches available. Rather than stating that the information wasn't present, the model occasionally produced a plausible-sounding definition drawn from general knowledge rather than the retrieved text. This is the clearest example of the point made in Section 1.3: the top-k mechanism does not guarantee *sufficiency*, only *relative similarity*.

### Scenario 2 — Answer Broader Than the Retrieved Context

In several cases the retrieved chunks contained a narrow, specific definition (e.g., a single equation or a short cheat-sheet bullet), but the generated answer elaborated with additional explanation, examples, or context not present in the chunk text. The extra material was often *plausible* and even correct in a general sense, but it was not something the pipeline could verify against the source PDFs — meaning it was, by the project's own definition, unsupported.

### Scenario 3 — Invalid JSON Instead of the Required Schema

During structured-output testing (Task 3), the model occasionally wrapped its JSON in explanatory text or Markdown code fences (` ```json ... ``` `) despite the system prompt explicitly forbidding this. In a couple of instances the response was not parseable JSON at all — for example, a trailing sentence appended after the closing brace broke `json.loads()`. This is a hallucination at the *format* level: the model "hallucinated" that some extra text was expected or helpful, when the prompt had explicitly ruled it out.

### Scenario 4 — Confidence Values Outside the Allowed Range

The `Answer.confidence` field is constrained to `0.0–1.0`. In testing, the model sometimes returned confidence as a percentage-like number (e.g., `95` instead of `0.95`), or as a value like `1.2`, apparently treating "confidence" as an unbounded score rather than a probability. This is a subtle but important hallucination: the *number itself* wasn't nonsensical, but it violated a constraint the model had been given in the schema description.

### Scenario 5 — Required Fields Omitted

On at least one occasion, a response included `question` and `answer` but omitted `sources`, apparently because the model judged the retrieved context "obvious" from the answer itself and didn't consider the field necessary. This mirrors a common real-world failure mode: models tend to treat structured-output instructions as suggestions when they believe the omitted information is redundant, rather than as a strict contract.

### Scenario 6 — Attempted Inclusion of Unsupported Information

In one test question about deep learning architectures, the retrieved chunks covered only a subset of the topic (e.g., a general definition), but the model's answer named specific architectures or techniques not mentioned anywhere in the retrieved text or the source PDFs. This is the most concerning category of hallucination because it is the hardest to catch from the JSON structure alone — the response was well-formed and passed Pydantic validation, but the *content* still exceeded what the retrieved sources actually supported.

## How the Hallucinations Were Detected

No single mechanism caught every scenario above; detection relied on layering several checks built throughout the project:

- **Retrieval inspection.** `display_retrieved_chunks()` (Task 1) prints each retrieved chunk's source, page, similarity score, and a text preview *before* the answer is generated, making it possible to check whether the retrieved material could plausibly support the eventual answer.
- **Source document verification.** For flagged answers, the cited `source`/`page` pair was manually cross-checked by opening the actual PDF page to confirm whether the claimed content was really there.
- **Similarity scores.** Consistently low similarity scores (well below the scores seen on questions the PDFs clearly cover) were a strong early signal that the model was about to answer from weak or irrelevant context — this is exactly how Scenario 1 was first noticed.
- **Pydantic schema validation.** `validate_output()` (Task 3) immediately caught structural hallucinations: out-of-range confidence, wrong data types, and missing fields all raised a `ValidationError` before anything was saved.
- **JSON parsing errors.** Responses that weren't valid JSON at all (Scenario 3) were caught earlier still, at the `json.loads()` step inside `validate_output()`, before Pydantic even ran.
- **Validation tests (Task 4).** The `test_*` functions written in Task 4 encode exactly these failure modes (missing field, wrong type, out-of-range confidence, non-list `sources`, invalid JSON) as repeatable, automated checks, so the same classes of hallucination can be re-verified after any change to the prompt or schema — not just caught once by chance.
- **Manual comparison with retrieved chunks.** For content-level hallucinations (Scenarios 2 and 6), the only reliable check was reading the retrieved chunk text side-by-side with the generated answer and confirming, sentence by sentence, whether each claim was actually present in the source.

## Mitigation Techniques

Several complementary techniques were used to reduce — though not eliminate — hallucination throughout the project:

- **Retrieval-Augmented Generation (RAG).** Grounding generation in retrieved PDF chunks constrains the space of "reasonable" answers far more than an ungrounded prompt would, and is the foundation the rest of the pipeline builds on.
- **Prompt engineering.** The system prompt explicitly instructs the model to answer *only* from the provided context and to state plainly when the answer isn't available, rather than leaving refusal behavior implicit.
- **Structured JSON outputs.** Requiring a fixed schema (`question`, `answer`, `confidence`, `sources`) turns "did the model behave" from a subjective judgment into something that can be mechanically checked.
- **Pydantic validation.** Every response is validated against the `Answer` model before it is trusted anywhere downstream — type errors, range violations, and missing fields are all caught here, deterministically and immediately.
- **Rejecting invalid outputs.** The pipeline never saves data that fails validation; a rejected response is logged with its error and discarded rather than being "fixed up" or silently accepted, which prevents corrupt data from ever reaching `outputs/`.
- **Source citations.** Requiring the model to list `sources` alongside every answer makes content-level hallucination *checkable* — a claim with no matching source, or a source that doesn't actually support the claim, is a visible red flag rather than an invisible one.
- **Manual verification.** For the hallucination types that structural validation cannot catch (an answer that is well-formed JSON but content that overreaches the retrieved context), manual comparison against the source PDFs remained necessary and was treated as a required step, not an optional one.

## Conclusion

This assignment made clear that hallucination is not a single failure mode with a single fix — it shows up at multiple layers of a RAG pipeline, from retrieval sufficiency, to generation faithfulness, to output formatting, and each layer needs its own safeguard. Retrieval reduces hallucination by grounding the model in real content, but it cannot force the model to stay within that content, and it cannot prevent format-level failures at all. Validation, conversely, reliably catches structural problems — wrong types, missing fields, out-of-range values, broken JSON — but is blind to a fluent, well-formed answer that simply says more than its sources support.

The practical lesson is that retrieval and validation are complementary, not substitutable: retrieval narrows *what the model can plausibly say*, while validation enforces *what shape the model is allowed to say it in*, and manual spot-checking remains the only reliable backstop for the content-level hallucinations that fall between those two safeguards. None of these techniques, individually or combined, make hallucination impossible — an LLM is still a probabilistic generator, not a verified reasoning system. But together they make hallucinations far less frequent, and just as importantly, they make the hallucinations that do occur *detectable* rather than silently trusted, which is the more realistic and achievable goal for a system like this one.
