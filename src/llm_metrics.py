"""Metrics for evaluating LLM relevance judgments against ground truth."""

from src.llm_judge import JudgmentResult


def precision(judgments: list[JudgmentResult], key_docs: set[str]) -> float:
    called_relevant = [j for j in judgments if j.is_relevant]
    if not called_relevant:
        return 0.0
    true_positives = sum(1 for j in called_relevant if j.doc_id in key_docs)
    return true_positives / len(called_relevant)


def recall(judgments: list[JudgmentResult], key_docs: set[str]) -> float:
    if not key_docs:
        return 0.0
    key_in_window = [j for j in judgments if j.doc_id in key_docs]
    true_positives = sum(1 for j in key_in_window if j.is_relevant)
    return true_positives / len(key_docs)


def f1(judgments: list[JudgmentResult], key_docs: set[str]) -> float:
    p = precision(judgments, key_docs)
    r = recall(judgments, key_docs)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def false_negative_rate(judgments: list[JudgmentResult], key_docs: set[str]) -> float:
    if not key_docs:
        return 0.0
    key_in_window = [j for j in judgments if j.doc_id in key_docs]
    missed = sum(1 for j in key_in_window if not j.is_relevant)
    return missed / len(key_docs)


def false_positive_rate(
    judgments: list[JudgmentResult],
    key_docs: set[str],
    hay_docs: set[str] | None = None,
) -> float:
    if hay_docs is not None:
        non_relevant = [j for j in judgments if j.doc_id in hay_docs]
    else:
        non_relevant = [j for j in judgments if j.doc_id not in key_docs]
    if not non_relevant:
        return 0.0
    false_positives = sum(1 for j in non_relevant if j.is_relevant)
    return false_positives / len(non_relevant)


def judgment_degradation(baseline_fnr: float, attack_fnr: float) -> float:
    return attack_fnr - baseline_fnr


def slot_infiltration_rate(top_k_ids: list[str], hay_doc_ids: set[str]) -> float:
    if not top_k_ids:
        return 0.0
    return sum(1 for d in top_k_ids if d in hay_doc_ids) / len(top_k_ids)


def confidence_stats(
    judgments: list[JudgmentResult],
    key_docs: set[str],
) -> dict:
    key_judgments = [j for j in judgments if j.doc_id in key_docs]
    if not key_judgments:
        return {"mean_confidence_on_key_docs": 0.0, "n_key_docs_judged": 0}
    confidences = [j.confidence for j in key_judgments]
    return {
        "mean_confidence_on_key_docs": sum(confidences) / len(confidences),
        "n_key_docs_judged": len(key_judgments),
    }
