# src/metrics.py
"""Evaluation metrics for retrieval experiments."""

import numpy as np


def recall_at_k(ranked: list[str], key_docs: set[str], k: int) -> float:
    if not key_docs:
        return 0.0
    top_k = set(ranked[:k])
    return len(top_k & key_docs) / len(key_docs)


def mrr(ranked: list[str], key_docs: set[str]) -> float:
    for i, doc_id in enumerate(ranked):
        if doc_id in key_docs:
            return 1.0 / (i + 1)
    return 0.0


def displacement(
    baseline_ranked: list[str],
    attack_ranked: list[str],
    key_docs: set[str],
) -> float:
    def _rank_of(ranked, doc_id):
        try:
            return ranked.index(doc_id) + 1
        except ValueError:
            return len(ranked) + 1

    displacements = []
    for doc_id in key_docs:
        base_rank = _rank_of(baseline_ranked, doc_id)
        attack_rank = _rank_of(attack_ranked, doc_id)
        displacements.append(attack_rank - base_rank)

    return float(np.mean(displacements)) if displacements else 0.0


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
) -> tuple[float, float]:
    values_arr = np.array(values)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(values_arr, size=len(values_arr), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = 1 - ci
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)
