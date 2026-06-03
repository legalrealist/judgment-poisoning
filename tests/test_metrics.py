# tests/test_metrics.py
import pytest
import numpy as np
from src.metrics import recall_at_k, mrr, displacement, bootstrap_ci
from src.rank import rank_documents


def test_recall_at_5_perfect():
    ranked = ["key_1", "key_2", "key_3", "other_1", "other_2"]
    key_docs = {"key_1", "key_2", "key_3"}
    assert recall_at_k(ranked, key_docs, k=5) == 1.0


def test_recall_at_5_partial():
    ranked = ["key_1", "other_1", "key_2", "other_2", "other_3", "key_3"]
    key_docs = {"key_1", "key_2", "key_3"}
    assert recall_at_k(ranked, key_docs, k=5) == pytest.approx(2 / 3)


def test_recall_at_5_none():
    ranked = ["other_1", "other_2", "other_3", "other_4", "other_5", "key_1"]
    key_docs = {"key_1"}
    assert recall_at_k(ranked, key_docs, k=5) == 0.0


def test_mrr():
    ranked = ["other_1", "other_2", "key_1", "key_2", "other_3"]
    key_docs = {"key_1", "key_2"}
    assert mrr(ranked, key_docs) == pytest.approx(1 / 3)


def test_mrr_first_position():
    ranked = ["key_1", "other_1", "other_2"]
    key_docs = {"key_1"}
    assert mrr(ranked, key_docs) == 1.0


def test_displacement():
    baseline_ranked = ["key_1", "key_2", "other_1"]
    attack_ranked = ["other_1", "other_2", "key_1", "other_3", "key_2"]
    key_docs = {"key_1", "key_2"}
    assert displacement(baseline_ranked, attack_ranked, key_docs) == pytest.approx(2.5)


def test_rank_documents():
    doc_ids = ["a", "b", "c"]
    doc_embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7, 0.7],
    ])
    query_embedding = np.array([1.0, 0.0])
    ranked = rank_documents(doc_ids, doc_embeddings, query_embedding)
    assert ranked[0] == "a"
    assert ranked[1] == "c"
    assert ranked[2] == "b"


def test_bootstrap_ci():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    low, high = bootstrap_ci(values, n_resamples=1000, ci=0.95)
    assert low < high
    assert low >= 0.0
    assert high <= 0.6
