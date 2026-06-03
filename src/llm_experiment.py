"""Experiment orchestration for LLM judgment poisoning."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.rank import rank_documents
from src.llm_judge import JudgmentResult, judge_individual, judge_batch
from src.llm_metrics import (
    precision as p_fn,
    recall as r_fn,
    f1 as f1_fn,
    false_negative_rate as fnr_fn,
    false_positive_rate as fpr_fn,
    slot_infiltration_rate,
    confidence_stats,
)


@dataclass
class RetrievalAuditResult:
    k: int
    top_k_ids: list[str]
    n_key_in_top_k: int
    n_hay_in_top_k: int
    n_other_in_top_k: int
    slot_infiltration: float


@dataclass
class ParadigmResult:
    precision: float
    recall: float
    f1: float
    false_negative_rate: float
    false_positive_rate: float
    mean_confidence_on_key: float
    judgments: list[JudgmentResult]


@dataclass
class EndToEndResult:
    retrieval_audit: RetrievalAuditResult
    individual: ParadigmResult
    batch: ParadigmResult
    n_docs_judged: int


@dataclass
class AblationResult:
    ratio_results: list[dict]


def run_retrieval_audit(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    key_doc_ids: set[str],
    hay_doc_ids: set[str],
    k: int = 50,
) -> RetrievalAuditResult:
    ranked = rank_documents(doc_ids, doc_embeddings, query_embedding)
    top_k = ranked[:k]
    n_key = sum(1 for d in top_k if d in key_doc_ids)
    n_hay = sum(1 for d in top_k if d in hay_doc_ids)
    n_other = len(top_k) - n_key - n_hay
    return RetrievalAuditResult(
        k=k, top_k_ids=top_k,
        n_key_in_top_k=n_key, n_hay_in_top_k=n_hay, n_other_in_top_k=n_other,
        slot_infiltration=slot_infiltration_rate(top_k, hay_doc_ids),
    )


def _compute_paradigm_result(
    judgments: list[JudgmentResult],
    key_doc_ids: set[str],
    hay_doc_ids: set[str],
) -> ParadigmResult:
    stats = confidence_stats(judgments, key_doc_ids)
    return ParadigmResult(
        precision=p_fn(judgments, key_doc_ids),
        recall=r_fn(judgments, key_doc_ids),
        f1=f1_fn(judgments, key_doc_ids),
        false_negative_rate=fnr_fn(judgments, key_doc_ids),
        false_positive_rate=fpr_fn(judgments, key_doc_ids, hay_doc_ids),
        mean_confidence_on_key=stats["mean_confidence_on_key_docs"],
        judgments=judgments,
    )


def run_end_to_end(
    doc_ids: list[str],
    doc_texts: dict[str, str],
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    key_doc_ids: set[str],
    hay_doc_ids: set[str],
    query: str,
    model: str,
    k: int = 50,
    cache_dir: Path | None = None,
) -> EndToEndResult:
    audit = run_retrieval_audit(doc_ids, doc_embeddings, query_embedding, key_doc_ids, hay_doc_ids, k)
    top_k_ids = audit.top_k_ids
    top_k_texts = [doc_texts[d] for d in top_k_ids]

    individual_judgments = [
        judge_individual(d, doc_texts[d], query, model, cache_dir)
        for d in top_k_ids
    ]
    batch_judgments = judge_batch(top_k_ids, top_k_texts, query, model, cache_dir)

    return EndToEndResult(
        retrieval_audit=audit,
        individual=_compute_paradigm_result(individual_judgments, key_doc_ids, hay_doc_ids),
        batch=_compute_paradigm_result(batch_judgments, key_doc_ids, hay_doc_ids),
        n_docs_judged=len(top_k_ids),
    )


def run_ablation(
    key_doc_ids: list[str],
    key_doc_texts: list[str],
    hay_doc_ids: list[str],
    hay_doc_texts: list[str],
    query: str,
    model: str,
    ratios: list[int] | None = None,
    cache_dir: Path | None = None,
    seed: int = 42,
) -> AblationResult:
    if ratios is None:
        ratios = [0, 1, 3, 5]

    ratio_results = []

    for ratio in ratios:
        n_hay_wanted = len(key_doc_ids) * ratio
        n_hay = min(n_hay_wanted, len(hay_doc_ids))

        if n_hay > 0:
            indices = list(range(len(hay_doc_ids)))
            rng = random.Random(seed)
            rng.shuffle(indices)
            selected = indices[:n_hay]
            selected_ids = [hay_doc_ids[i] for i in selected]
            selected_texts = [hay_doc_texts[i] for i in selected]
        else:
            selected_ids = []
            selected_texts = []

        window_ids = list(key_doc_ids) + selected_ids
        window_texts = list(key_doc_texts) + selected_texts

        # Shuffle window so key docs aren't always first (controls positional bias)
        combined = list(zip(window_ids, window_texts))
        rng_shuffle = random.Random(seed + ratio)
        rng_shuffle.shuffle(combined)
        window_ids, window_texts = zip(*combined) if combined else ([], [])
        window_ids = list(window_ids)
        window_texts = list(window_texts)

        judgments = judge_batch(window_ids, window_texts, query, model, cache_dir, no_chunk=True)

        key_set = set(key_doc_ids)
        hay_set = set(selected_ids)
        stats = confidence_stats(judgments, key_set)

        ratio_results.append({
            "ratio": ratio,
            "n_key_in_window": len(key_doc_ids),
            "n_hay_in_window": n_hay,
            "n_total_in_window": len(window_ids),
            "precision": p_fn(judgments, key_set),
            "recall": r_fn(judgments, key_set),
            "f1": f1_fn(judgments, key_set),
            "false_negative_rate": fnr_fn(judgments, key_set),
            "false_positive_rate": fpr_fn(judgments, key_set, hay_set),
            "mean_confidence_on_key": stats["mean_confidence_on_key_docs"],
        })

    return AblationResult(ratio_results=ratio_results)
