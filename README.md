# Judgment Stability in LLM-Based Document Review

**Finding:** LLM classification stability depends on whether you ask for expert judgment or observable textual criteria.

## The result

When classifying litigation documents as KEY (smoking gun) or ROUTINE (responsive but unremarkable), prompt framing determines reliability:

| Prompt type | Instability rate | Batch context effect |
|---|---|---|
| Strategic ("would a lawyer flag this?") | 60-95% | Unmeasurable (baseline too noisy) |
| Textual criteria ("contains an admission...") | 3-7% | None detected |

Binary relevance classification (RELEVANT vs. NOT RELEVANT) was 98% stable — the instability is specific to unconstrained salience judgments.

## Two prompt styles

**Strategic prompt** — asks the LLM to simulate expert judgment:
> KEY: A document that could change the outcome of the case. Would a trial lawyer flag this for the jury?

**Textual-criteria prompt** — asks the LLM to identify observable features:
> KEY: Contains one or more of: a direct admission of wrongdoing, a decision to take a specific action, a first-person account, instructions to others, an attempt to conceal or misrepresent information.

Same task, same documents, same model. The textual-criteria prompt produces 93-97% stability across batch contexts.

## Why

"Would a lawyer flag this?" is a latent judgment — it requires case theory, litigation strategy, and context the LLM doesn't have. The answer varies each time.

"Does this contain an admission?" is an observable predicate — the answer is in the text. The LLM reads text reliably.

## Data

- **Corpus:** EDRM Enron v2, ~638K emails, 149 custodians
- **Ground truth:** TREC Legal Track Interactive 2010 (topics 301-304)
- **Model:** Claude Sonnet 4

## Repo structure

The experiment pipeline runs in steps via `run_experiment.py`:

```
python run_experiment.py --step download --custodians allen-p.zip ...
python run_experiment.py --step parse
python run_experiment.py --step conditions --topic 301
python run_experiment.py --step embed --model openai/text-embedding-3-large
```

Key modules:

- `src/llm_judge.py` — LLM-based document judgment with individual and batch paradigms, disk-cached
- `src/llm_experiment.py` — Experiment orchestration: retrieval audit, end-to-end evaluation, ablation (hay:key ratio scaling)
- `src/enron_parse.py`, `src/enron_corpus.py` — Enron email corpus parsing and loading
- `src/trec_loader.py` — TREC qrels ground truth loading
- `src/build_conditions.py` — Experimental condition construction (baseline, haystacked variants, dilution control)
- `src/embed.py` — Multi-model embedding interface (OpenAI, Cohere, Voyage, Google, Jina, Contriever, BGE, E5)

The repo also contains the retrieval-layer experiments (embedding displacement, BM25 comparison) that motivated the stability finding — the haystacking attack failed at the judgment layer, which led to the prompt-framing discovery.
