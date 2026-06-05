"""Stability experiment: strategic vs. textual-criteria prompts for KEY/ROUTINE classification.

Measures whether the same document receives the same classification across
different batch contexts (routine filler, topically adjacent hay, off-topic hay).
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from src.llm_judge import JudgmentResult, _call_llm, _parse_json, _cache_key, _load_cache, _save_cache

CACHE_DIR = Path("llm_cache")

STRATEGIC_PROMPT_INDIVIDUAL = """This document has been confirmed relevant to the following legal request. Classify it as KEY or ROUTINE.

REQUEST: {query}

- KEY: A document that could change the outcome of the case. Would a trial lawyer flag this for the jury?
- ROUTINE: Responsive but would not independently affect the case outcome.

DOCUMENT:
{doc_text}

Respond with JSON only: {{"judgment": "KEY" or "ROUTINE", "confidence": 0.0-1.0}}"""

TEXTUAL_CRITERIA_PROMPT_INDIVIDUAL = """This document has been confirmed relevant to the following legal request. Classify it as KEY or ROUTINE.

REQUEST: {query}

- KEY: Contains one or more of: a direct admission of wrongdoing or knowledge of wrongdoing; a decision to take a specific action related to the requested activity; a first-person account of participating in the activity; instructions or directives to others; an attempt to conceal, destroy, or misrepresent information.
- ROUTINE: Responsive but does not contain any of the above.

DOCUMENT:
{doc_text}

Respond with JSON only: {{"judgment": "KEY" or "ROUTINE", "confidence": 0.0-1.0}}"""

STRATEGIC_PROMPT_BATCH = """Each document below has been confirmed relevant to the following legal request. Classify each as KEY or ROUTINE.

REQUEST: {query}

- KEY: A document that could change the outcome of the case. Would a trial lawyer flag this for the jury?
- ROUTINE: Responsive but would not independently affect the case outcome.

{documents}

Respond with JSON only — an array of objects, one per document, in the same order:
[{{"doc_id": "...", "judgment": "KEY" or "ROUTINE", "confidence": 0.0-1.0}}, ...]"""

TEXTUAL_CRITERIA_PROMPT_BATCH = """Each document below has been confirmed relevant to the following legal request. Classify each as KEY or ROUTINE.

REQUEST: {query}

- KEY: Contains one or more of: a direct admission of wrongdoing or knowledge of wrongdoing; a decision to take a specific action related to the requested activity; a first-person account of participating in the activity; instructions or directives to others; an attempt to conceal, destroy, or misrepresent information.
- ROUTINE: Responsive but does not contain any of the above.

{documents}

Respond with JSON only — an array of objects, one per document, in the same order:
[{{"doc_id": "...", "judgment": "KEY" or "ROUTINE", "confidence": 0.0-1.0}}, ...]"""

PROMPT_STYLES = {
    "strategic": {
        "individual": STRATEGIC_PROMPT_INDIVIDUAL,
        "batch": STRATEGIC_PROMPT_BATCH,
    },
    "textual_criteria": {
        "individual": TEXTUAL_CRITERIA_PROMPT_INDIVIDUAL,
        "batch": TEXTUAL_CRITERIA_PROMPT_BATCH,
    },
}


@dataclass
class StabilityResult:
    target_doc: str
    baseline_judgment: str
    judgments_by_condition: dict[str, str]
    flipped: bool


@dataclass
class StabilityReport:
    prompt_style: str
    topic_id: str
    n_targets: int
    n_flipped: int
    flip_rate: float
    per_condition_flip: dict[str, int]
    results: list[StabilityResult]


def classify_individual(
    doc_id: str,
    doc_text: str,
    query: str,
    model: str,
    prompt_style: str = "textual_criteria",
    cache_dir: Path | None = None,
) -> JudgmentResult:
    cache_dir = cache_dir or CACHE_DIR
    paradigm = f"stability_individual_{prompt_style}"
    key = _cache_key(model, doc_text, query, paradigm)
    cached = _load_cache(cache_dir, key)
    if cached:
        result = cached[0]
        result.doc_id = doc_id
        return result

    template = PROMPT_STYLES[prompt_style]["individual"]
    prompt = template.format(query=query, doc_text=doc_text)
    response = _call_llm(prompt, model)
    parsed = _parse_json(response)
    result = JudgmentResult(
        doc_id=doc_id,
        judgment=parsed["judgment"],
        confidence=float(parsed["confidence"]),
    )
    _save_cache(cache_dir, key, [result])
    return result


def classify_batch(
    doc_ids: list[str],
    doc_texts: list[str],
    query: str,
    model: str,
    prompt_style: str = "textual_criteria",
    cache_dir: Path | None = None,
) -> list[JudgmentResult]:
    cache_dir = cache_dir or CACHE_DIR
    combined_text = "\n\n".join(
        f"--- DOCUMENT {doc_id} ---\n{text}" for doc_id, text in zip(doc_ids, doc_texts)
    )
    paradigm = f"stability_batch_{prompt_style}"
    key = _cache_key(model, combined_text, query, paradigm)
    cached = _load_cache(cache_dir, key)
    if cached:
        for i, r in enumerate(cached):
            r.doc_id = doc_ids[i]
        return cached

    template = PROMPT_STYLES[prompt_style]["batch"]
    prompt = template.format(query=query, documents=combined_text)
    response = _call_llm(prompt, model)
    parsed = _parse_json(response)

    response_map = {}
    for entry in parsed:
        response_map[entry.get("doc_id", "")] = entry

    results = []
    for doc_id in doc_ids:
        if doc_id in response_map:
            entry = response_map[doc_id]
            results.append(JudgmentResult(
                doc_id=doc_id,
                judgment=entry["judgment"],
                confidence=float(entry["confidence"]),
            ))
        else:
            results.append(JudgmentResult(
                doc_id=doc_id, judgment="ROUTINE", confidence=0.0,
            ))

    _save_cache(cache_dir, key, results)
    return results


def build_paired_batches(
    target_doc_ids: list[str],
    corpus: dict,
    relevant_doc_ids: set[str],
    non_relevant_doc_ids: set[str],
    topic_id: str,
    query: str,
    batch_size: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Build paired batch contexts for stability testing.

    For each target doc, creates three batches:
    - control_routine: target + randomly selected relevant docs (routine filler)
    - adjacent_hay: target + topically adjacent non-relevant docs
    - offtopic_hay: target + off-topic non-relevant docs
    """
    rng = random.Random(seed)

    routine_pool = sorted(relevant_doc_ids - set(target_doc_ids))
    non_relevant_pool = sorted(non_relevant_doc_ids & set(corpus.keys()))

    batches = []
    for target_id in target_doc_ids:
        target_text = corpus[target_id].to_text() if hasattr(corpus[target_id], 'to_text') else str(corpus[target_id])
        filler_count = batch_size - 1

        for condition in ["control_routine", "adjacent_hay", "offtopic_hay"]:
            if condition == "control_routine":
                pool = [d for d in routine_pool if d != target_id]
                selected = rng.sample(pool, min(filler_count, len(pool)))
            elif condition == "adjacent_hay":
                selected = rng.sample(non_relevant_pool, min(filler_count, len(non_relevant_pool)))
            else:
                selected = rng.sample(non_relevant_pool, min(filler_count, len(non_relevant_pool)))

            docs = [{"id": target_id, "t": target_text, "is_key": True}]
            for filler_id in selected:
                filler_text = corpus[filler_id].to_text() if hasattr(corpus[filler_id], 'to_text') else str(corpus[filler_id])
                docs.append({"id": filler_id, "t": filler_text, "is_key": False})

            rng.shuffle(docs)

            batches.append({
                "tid": topic_id,
                "stmt": query,
                "condition": condition,
                "target_doc": target_id,
                "docs": docs,
            })

    return batches


def run_baseline_scoring(
    doc_ids: list[str],
    corpus: dict,
    query: str,
    model: str,
    prompt_style: str = "textual_criteria",
    cache_dir: Path | None = None,
) -> list[JudgmentResult]:
    """Classify all confirmed-relevant docs individually as KEY/ROUTINE."""
    results = []
    for i, doc_id in enumerate(doc_ids):
        doc_text = corpus[doc_id].to_text() if hasattr(corpus[doc_id], 'to_text') else str(corpus[doc_id])
        result = classify_individual(doc_id, doc_text, query, model, prompt_style, cache_dir)
        results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  Scored {i + 1}/{len(doc_ids)} docs")
    return results


def run_stability_test(
    batches: list[dict],
    baseline_judgments: dict[str, str],
    query: str,
    model: str,
    prompt_style: str = "textual_criteria",
    cache_dir: Path | None = None,
) -> list[StabilityResult]:
    """Evaluate each batch and compare target doc classification to baseline."""
    target_results: dict[str, dict[str, str]] = {}

    for i, batch in enumerate(batches):
        target_id = batch["target_doc"]
        condition = batch["condition"]
        doc_ids = [d["id"] for d in batch["docs"]]
        doc_texts = [d["t"] for d in batch["docs"]]

        judgments = classify_batch(doc_ids, doc_texts, query, model, prompt_style, cache_dir)
        judgment_map = {j.doc_id: j.judgment for j in judgments}
        target_judgment = judgment_map.get(target_id, "UNKNOWN")

        if target_id not in target_results:
            target_results[target_id] = {}
        target_results[target_id][condition] = target_judgment

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(batches)} batches")

    results = []
    for target_id, condition_judgments in target_results.items():
        baseline = baseline_judgments.get(target_id, "UNKNOWN")
        flipped = any(j != baseline for j in condition_judgments.values())
        results.append(StabilityResult(
            target_doc=target_id,
            baseline_judgment=baseline,
            judgments_by_condition=condition_judgments,
            flipped=flipped,
        ))

    return results


def compute_stability_report(
    results: list[StabilityResult],
    prompt_style: str,
    topic_id: str,
) -> StabilityReport:
    n_flipped = sum(1 for r in results if r.flipped)
    per_condition: dict[str, int] = {}
    for r in results:
        for cond, judgment in r.judgments_by_condition.items():
            if judgment != r.baseline_judgment:
                per_condition[cond] = per_condition.get(cond, 0) + 1

    return StabilityReport(
        prompt_style=prompt_style,
        topic_id=topic_id,
        n_targets=len(results),
        n_flipped=n_flipped,
        flip_rate=n_flipped / len(results) if results else 0.0,
        per_condition_flip=per_condition,
        results=results,
    )
