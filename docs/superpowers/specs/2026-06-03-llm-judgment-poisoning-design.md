# LLM Judgment Poisoning via Document Haystacking

## Threat Model

A producing party in litigation responds to a discovery request by over-producing real, topically adjacent but non-responsive documents alongside the truly responsive ones. This is legal and routine. The requesting party's AI review tool — Relativity aiR (GPT-4 per document), Everlaw Deep Dive (RAG over chunked corpus), or similar — processes the polluted corpus.

Two damage channels:

1. **Retrieval slot waste** — hay docs score high on vector similarity, consuming spots in the top-k retrieval window and displacing truly relevant docs the LLM never sees.
2. **Judgment degradation** — hay docs that share the LLM's context window with truly relevant docs blur the relevance boundary, causing misclassification of documents the LLM would otherwise get right.

The experiment decomposes these two effects and measures them independently.

### Hypothesis

Topically similar hay causes more LLM judgment errors than random off-topic dilution at the same volume. The strategic similarity — not volume alone — is the attack.

## Experimental Conditions

### Document sets (reuse existing pipeline)

| Condition | Selection method | Purpose |
|-----------|-----------------|---------|
| Baseline | TREC-judged relevant docs only | Clean reference |
| Haystacked A | Broad collection from same custodians | Strategic over-production |
| Haystacked B | Keyword-aware selection | More targeted over-production |
| Dilution control | Random off-topic docs from uninvolved custodians | Volume-only control |

Condition C (embedding-optimized) deferred to a follow-up.

### Scales

- Small (3N), medium (6N), large (11N) — where N is the number of relevant docs.

### Data

- Enron corpus (~57K emails, 7 custodians)
- TREC Legal Track topics with ground-truth relevance judgments
- Start with topic 303 (lobbying, 194 relevant, 38 key) for initial validation
- Expand to topic 301 (oil/gas, 44 relevant, 8 key) after pipeline works

## Three Measurement Modes

### Mode 1: End-to-end

Full pipeline. Embed the condition corpus, retrieve top-50 by vector similarity, send to LLM for relevance coding. Measures total damage (retrieval + judgment effects tangled together).

### Mode 2: Retrieval audit

Count how many of the top-50 retrieval slots are occupied by hay docs vs. relevant docs vs. other. No LLM calls needed — purely an analysis of the existing dense retrieval rankings. Quantifies slot waste.

### Mode 3: Ablation (fixed-window judgment test)

Remove the retrieval variable. Force all key docs into the context window, inject hay docs alongside them at controlled ratios:

- 0:1 (clean — key docs only)
- 1:1 (equal hay and key docs)
- 3:1 (3x hay)
- 5:1 (5x hay)

If the LLM's accuracy drops, it can only be because the hay docs confused its judgment — not because relevant docs were displaced from the window.

The ablation is the core contribution. It proves that haystacking doesn't just waste retrieval budget — it actively misleads the LLM even when relevant docs are guaranteed present.

### Decomposition

- End-to-end = retrieval effect + judgment effect
- Ablation = judgment effect only
- Retrieval effect = end-to-end minus ablation

## LLM Judge Design

### Two review paradigms

**Individual review (Relativity aiR-style):** Each document sent to the LLM alone with the review prompt. No other documents in context. The LLM returns binary relevant/not-relevant plus a confidence score (0-1). In this paradigm, hay docs can't influence judgment on other docs — the attack is purely retrieval slot waste.

**Batch review (Everlaw Deep Dive-style):** Top-50 docs sent together in one context window. The LLM codes each document in the batch. Hay docs in the batch can shift the LLM's calibration of what "relevant" means, causing misclassification of co-located relevant docs.

### Review prompt

Mirrors a real Relativity aiR prompt — a natural language description of what's responsive, derived from the TREC topic statement:

> Review this document for responsiveness to the following request: [TREC topic statement]. Classify as RELEVANT or NOT RELEVANT. Provide a confidence score from 0.0 to 1.0.

For batch mode, the prompt asks the LLM to code each document, returning structured output: list of (doc_id, judgment, confidence).

Same prompt across all LLMs and all conditions. The only variable is document content.

### LLM selection

| Model | Rationale |
|-------|-----------|
| Claude Sonnet 4 | Cheap, fast iteration |
| GPT-4o | Relativity's actual backend (Azure OpenAI) |
| Open-source (Llama 3.1 70B or Qwen 2.5 72B) | Via Together/Fireworks API; tests whether open models are more or less vulnerable |

## Metrics

### Per-document judgment metrics

- **Precision** — of docs the LLM calls relevant, fraction that actually are (TREC ground truth)
- **Recall** — of truly relevant docs in the window, fraction correctly identified
- **F1** — harmonic mean
- **False positive rate** — how often the LLM calls hay docs relevant (wasted downstream review)
- **False negative rate** — how often the LLM misses truly relevant docs (the real damage)

### Key comparison metric

**Judgment degradation** = false negative rate under haystacking minus false negative rate under baseline.

Compare haystacked A/B vs. dilution at the same scale to isolate the strategic effect.

### Per-condition aggregates

- **Slot infiltration rate** — fraction of top-50 retrieval slots consumed by hay docs (mode 2, no LLM)
- **Effective recall** — end-to-end: of all key docs in the corpus, how many does retrieve + judge correctly surface?
- **Cost multiplier** — LLM calls wasted on hay vs. baseline

### Ablation-specific

- **Judgment accuracy curve** — accuracy as a function of hay-to-relevant ratio (0:1, 1:1, 3:1, 5:1)
- **Confidence calibration** — does the LLM's confidence on relevant docs drop when hay is present?

### Statistical framework (reuse existing)

- Bootstrap confidence intervals on all metrics
- Wilcoxon signed-rank test for paired comparisons (haystacked vs. dilution at same scale)
- Cohen's d for effect size
- Bonferroni correction for multiple comparisons

## Pipeline Architecture

### Data flow

```
Existing:  corpus -> conditions -> embed -> rank (top-50)
                                               |
New:                                     LLM judge stage
                                          |           |
                                individual review   batch review
                                          |           |
                                     metrics / comparison
                                          |
                                   ablation (fixed-window)
```

### New modules

**`src/llm_judge.py`** — Core LLM interface.
- `judge_individual(doc_text, query, model) -> (judgment, confidence)`
- `judge_batch(doc_ids, doc_texts, query, model) -> list[(doc_id, judgment, confidence)]`
- Pluggable backends: Anthropic SDK, OpenAI SDK, Together/Fireworks
- Rate limiting, retries, structured response parsing
- Disk cache keyed on (model, doc_id_hash, query_hash, paradigm) — reruns are free

**`src/llm_experiment.py`** — Experiment orchestration.
- `run_end_to_end(condition, llm_model, retrieval_model, k=50)` — retrieve, judge, compute metrics
- `run_ablation(key_docs, hay_docs, query, model, ratios=[0,1,3,5])` — fixed window, vary hay ratio
- `run_retrieval_audit(condition, retrieval_model, k=50)` — count hay/relevant/other in top-k

**`src/llm_metrics.py`** — Judgment-specific metrics.
- precision, recall, F1, false negative/positive rates
- confidence calibration statistics
- judgment degradation (delta vs. baseline)

**`run_llm_evaluation.py`** — Top-level experiment script.
- Default: 1 topic x 1 LLM x 1 scale (cheap validation)
- `--expand` for full matrix

### Unchanged modules

`enron_parse.py`, `enron_corpus.py`, `enron_download.py`, `trec_loader.py`, `build_conditions.py`, `embed.py`, `rank.py`, `stats.py`, `detect.py`

### Caching

Every LLM call cached to disk. The cache key is (model, document content hash, query hash, paradigm). Expanding the experiment matrix (adding models, scales, topics) only incurs cost for new combinations. Reruns are free.

## Execution Plan

### Phase 1: Validate pipeline

- 1 topic (303) x 1 LLM (Claude Sonnet) x 1 scale (medium) x both paradigms
- Verify metrics make sense, prompt works, caching works
- Estimate cost for full expansion

### Phase 2: Ablation

- Run fixed-window judgment test across all hay ratios
- This is the core result — does topical hay degrade LLM accuracy more than random dilution?

### Phase 3: Expand

- Add GPT-4o and open-source model
- Add topic 301
- Add remaining scales
- Full statistical analysis

## What This Does NOT Cover

- Prompt injection attacks (out of scope — well-studied separately)
- TAR/active learning poisoning (linear classifier attacks are well-studied)
- Condition C / embedding-optimized hay (deferred)
- Commercial embedding models (API keys not set up)
- Mitigation design (future work after attack is demonstrated)
