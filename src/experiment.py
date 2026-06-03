# src/experiment.py
"""Run retrieval experiments across conditions, models, queries, and scale levels."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.rank import rank_documents
from src.metrics import recall_at_k, mrr, displacement, bootstrap_ci


@dataclass
class ExperimentResult:
    condition: str
    model: str
    topic_id: str
    query: str
    scale: str
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ranked_doc_ids: list[str]


def run_single_experiment(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    key_doc_ids: set[str],
    condition: str = "",
    model: str = "",
    topic_id: str = "",
    query: str = "",
    scale: str = "",
) -> ExperimentResult:
    ranked = rank_documents(doc_ids, doc_embeddings, query_embedding)
    return ExperimentResult(
        condition=condition,
        model=model,
        topic_id=topic_id,
        query=query,
        scale=scale,
        recall_at_5=recall_at_k(ranked, key_doc_ids, k=5),
        recall_at_10=recall_at_k(ranked, key_doc_ids, k=10),
        recall_at_20=recall_at_k(ranked, key_doc_ids, k=20),
        mrr=mrr(ranked, key_doc_ids),
        ranked_doc_ids=ranked,
    )


def run_full_experiment(
    conditions: dict[str, dict],
    query_embeddings: dict[str, np.ndarray],
    model: str,
    topic_id: str,
    scale: str,
) -> list[ExperimentResult]:
    results = []
    for condition_name, cond_data in conditions.items():
        for query_text, query_emb in query_embeddings.items():
            result = run_single_experiment(
                doc_ids=cond_data["doc_ids"],
                doc_embeddings=cond_data["doc_embeddings"],
                query_embedding=query_emb,
                key_doc_ids=cond_data["key_doc_ids"],
                condition=condition_name,
                model=model,
                topic_id=topic_id,
                query=query_text,
                scale=scale,
            )
            results.append(result)
    return results


def save_results(results: list[ExperimentResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for r in results:
        data.append({
            "condition": r.condition,
            "model": r.model,
            "topic_id": r.topic_id,
            "query": r.query,
            "scale": r.scale,
            "recall_at_5": r.recall_at_5,
            "recall_at_10": r.recall_at_10,
            "recall_at_20": r.recall_at_20,
            "mrr": r.mrr,
        })
    with open(output_dir / "results.json", "w") as f:
        json.dump(data, f, indent=2)
