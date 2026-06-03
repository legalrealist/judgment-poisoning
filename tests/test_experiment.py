# tests/test_experiment.py
import pytest
import json
import numpy as np
from src.experiment import run_single_experiment, run_full_experiment, save_results, ExperimentResult


def test_run_single_experiment():
    doc_ids = ["key_1", "key_2", "hay_1", "hay_2", "hay_3"]
    doc_embeddings = np.array([
        [0.9, 0.1],  # key_1
        [0.8, 0.2],  # key_2
        [0.1, 0.9],  # hay_1
        [0.2, 0.8],  # hay_2
        [0.3, 0.7],  # hay_3
    ])
    query_embedding = np.array([1.0, 0.0])
    key_doc_ids = {"key_1", "key_2"}

    result = run_single_experiment(
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
        query_embedding=query_embedding,
        key_doc_ids=key_doc_ids,
    )

    assert isinstance(result, ExperimentResult)
    assert result.recall_at_5 == 1.0
    assert result.recall_at_10 == 1.0
    assert result.mrr == 1.0
    assert result.ranked_doc_ids[0] == "key_1"
    assert result.ranked_doc_ids[1] == "key_2"


def test_run_full_experiment():
    doc_ids = ["key_1", "hay_1", "hay_2"]
    doc_embeddings = np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8]])
    query_emb = np.array([1.0, 0.0])

    conditions = {
        "baseline": {
            "doc_ids": ["key_1"],
            "doc_embeddings": np.array([[0.9, 0.1]]),
            "key_doc_ids": {"key_1"},
        },
        "haystacked": {
            "doc_ids": doc_ids,
            "doc_embeddings": doc_embeddings,
            "key_doc_ids": {"key_1"},
        },
    }
    query_embeddings = {"oil drilling": query_emb}

    results = run_full_experiment(
        conditions=conditions,
        query_embeddings=query_embeddings,
        model="test-model",
        topic_id="301",
        scale="small",
    )

    assert len(results) == 2
    baseline_result = [r for r in results if r.condition == "baseline"][0]
    haystacked_result = [r for r in results if r.condition == "haystacked"][0]
    assert baseline_result.recall_at_5 == 1.0
    assert haystacked_result.recall_at_5 == 1.0
    assert haystacked_result.model == "test-model"
    assert haystacked_result.topic_id == "301"


def test_save_results(tmp_path):
    results = [
        ExperimentResult(
            condition="baseline", model="test", topic_id="301",
            query="test query", scale="small",
            recall_at_5=1.0, recall_at_10=1.0, recall_at_20=1.0,
            mrr=1.0, ranked_doc_ids=["a", "b"],
        )
    ]
    save_results(results, tmp_path)
    with open(tmp_path / "results.json") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["recall_at_5"] == 1.0
    assert "ranked_doc_ids" not in data[0]
