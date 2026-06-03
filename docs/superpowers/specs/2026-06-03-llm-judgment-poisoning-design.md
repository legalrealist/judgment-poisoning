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

## Related Work

The underlying phenomenon — topically similar noise degrades LLM accuracy — is established across several research threads. This experiment applies those findings to a specific high-stakes domain (eDiscovery) where the attack is already a routine litigation tactic.

**RAG robustness to irrelevant context:**
- Wu et al. (COLM 2024) — LLMs are more easily misled by semantically related irrelevant information than by unrelated noise. Directly supports the hypothesis that topical hay is worse than random dilution. [arXiv:2404.03302](https://arxiv.org/abs/2404.03302)
- Amiraz et al. (ACL 2025) — "The Distracting Effect": quantifies how irrelevant retrieved passages distract RAG answer generation, identifies "hard distracting passages" (topically similar but irrelevant). [arXiv:2505.06914](https://arxiv.org/abs/2505.06914)
- Chroma "Context Rot" (2025) — 20-50% accuracy drops across 18 models (GPT-4.1, Claude Opus 4, Gemini 2.5 Pro) as context grows. Coherent distractors are worse than shuffled ones — structural plausibility hurts. This is architectural, not a training gap. [research.trychroma.com/context-rot](https://research.trychroma.com/context-rot)
- Lee et al. (2026) — "Lost in the Noise": up to 80% performance drop with contextual distractors, even in reasoning models. Inverse scaling: more test-time compute makes noisy performance worse. [arXiv:2601.07226](https://arxiv.org/abs/2601.07226)

**LLM-as-judge bias:**
- Shan et al. (2024) — LLMs judging relevance in batches exhibit threshold priming: preceding documents shift the relevance threshold for later ones. Tested on GPT-3.5, GPT-4, LLaMA2. [arXiv:2409.16022](https://arxiv.org/abs/2409.16022)
- Yu et al. (2026) — LLM judges systematically overrate passages that don't satisfy the information need, driven by passage length and surface lexical cues. [arXiv:2602.17170](https://arxiv.org/abs/2602.17170)
- Alaofi et al. (SIGIR-AP 2024) — LLMs can be fooled into labelling documents as relevant via keyword stuffing. [arXiv:2501.17969](https://arxiv.org/abs/2501.17969)

**Positional bias in long context:**
- Liu et al. (2023) — "Lost in the Middle": U-shaped attention bias, 30%+ accuracy drop for middle-positioned information. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)

**Gap:** No prior work tests these effects in the eDiscovery context, where strategic over-production is legal, routine, and the requesting party's AI review tool is the target.

## What This Does NOT Cover

- Prompt injection attacks (out of scope — well-studied separately)
- TAR/active learning poisoning (linear classifier attacks are well-studied)
- Condition C / embedding-optimized hay (deferred)
- Commercial embedding models (API keys not set up)
- Mitigation design (future work after attack is demonstrated)
