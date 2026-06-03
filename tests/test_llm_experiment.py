"""Tests for LLM experiment orchestration."""

import numpy as np
import pytest
from unittest.mock import patch

from src.llm_judge import JudgmentResult
from src.llm_experiment import (
    run_retrieval_audit, RetrievalAuditResult,
    run_end_to_end, EndToEndResult,
    run_ablation, AblationResult,
)


# --- Task 3: Retrieval Audit ---

def test_retrieval_audit_basic():
    doc_ids = ["k1", "k2", "h1", "h2", "h3", "o1"]
    doc_embeddings = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.0, 1.0],
    ])
    query_embedding = np.array([1.0, 0.0])
    key_doc_ids = {"k1", "k2"}
    hay_doc_ids = {"h1", "h2", "h3"}

    result = run_retrieval_audit(
        doc_ids=doc_ids, doc_embeddings=doc_embeddings, query_embedding=query_embedding,
        key_doc_ids=key_doc_ids, hay_doc_ids=hay_doc_ids, k=4,
    )
    assert isinstance(result, RetrievalAuditResult)
    assert result.k == 4
    assert result.n_key_in_top_k == 2
    assert result.n_hay_in_top_k == 2
    assert result.n_other_in_top_k == 0
    assert result.slot_infiltration == pytest.approx(0.5)


def test_retrieval_audit_no_hay():
    doc_ids = ["k1", "k2", "o1"]
    doc_embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query_embedding = np.array([1.0, 0.0])
    result = run_retrieval_audit(
        doc_ids=doc_ids, doc_embeddings=doc_embeddings, query_embedding=query_embedding,
        key_doc_ids={"k1", "k2"}, hay_doc_ids=set(), k=3,
    )
    assert result.slot_infiltration == 0.0
    assert result.n_key_in_top_k == 2


def test_retrieval_audit_returns_ranked_ids():
    doc_ids = ["a", "b", "c"]
    doc_embeddings = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    query_embedding = np.array([1.0, 0.0])
    result = run_retrieval_audit(
        doc_ids=doc_ids, doc_embeddings=doc_embeddings, query_embedding=query_embedding,
        key_doc_ids=set(), hay_doc_ids=set(), k=3,
    )
    assert result.top_k_ids[0] == "b"


# --- Task 4: End-to-End ---

def _mock_judge_individual(doc_id, doc_text, query, model, cache_dir=None):
    is_key = doc_id.startswith("k")
    return JudgmentResult(
        doc_id=doc_id,
        judgment="RELEVANT" if is_key else "NOT RELEVANT",
        confidence=0.9 if is_key else 0.8,
    )


def _mock_judge_batch(doc_ids, doc_texts, query, model, cache_dir=None):
    return [_mock_judge_individual(d, t, query, model) for d, t in zip(doc_ids, doc_texts)]


@patch("src.llm_experiment.judge_individual", side_effect=_mock_judge_individual)
@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_end_to_end_basic(mock_batch, mock_individual):
    doc_ids = ["k1", "k2", "h1", "h2"]
    texts = {"k1": "Key doc 1", "k2": "Key doc 2", "h1": "Hay 1", "h2": "Hay 2"}
    doc_embeddings = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.7, 0.3], [0.6, 0.4],
    ])
    query_embedding = np.array([1.0, 0.0])

    result = run_end_to_end(
        doc_ids=doc_ids, doc_texts=texts, doc_embeddings=doc_embeddings,
        query_embedding=query_embedding, key_doc_ids={"k1", "k2"},
        hay_doc_ids={"h1", "h2"}, query="test query", model="claude-sonnet-4", k=4,
    )
    assert isinstance(result, EndToEndResult)
    assert result.individual.recall == 1.0
    assert result.batch.recall == 1.0
    assert result.individual.false_negative_rate == 0.0


@patch("src.llm_experiment.judge_individual", side_effect=_mock_judge_individual)
@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_end_to_end_with_k(mock_batch, mock_individual):
    doc_ids = ["k1", "h1", "h2", "h3"]
    texts = {"k1": "Key", "h1": "Hay1", "h2": "Hay2", "h3": "Hay3"}
    doc_embeddings = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0],
    ])
    query_embedding = np.array([1.0, 0.0])

    result = run_end_to_end(
        doc_ids=doc_ids, doc_texts=texts, doc_embeddings=doc_embeddings,
        query_embedding=query_embedding, key_doc_ids={"k1"},
        hay_doc_ids={"h1", "h2", "h3"}, query="test", model="claude-sonnet-4", k=2,
    )
    assert result.n_docs_judged == 2


# --- Task 5: Ablation ---

@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_ablation_clean(mock_batch):
    result = run_ablation(
        key_doc_ids=["k1", "k2"], key_doc_texts=["Key 1", "Key 2"],
        hay_doc_ids=["h1", "h2", "h3", "h4"], hay_doc_texts=["Hay 1", "Hay 2", "Hay 3", "Hay 4"],
        query="test query", model="claude-sonnet-4", ratios=[0],
    )
    assert len(result.ratio_results) == 1
    r = result.ratio_results[0]
    assert r["ratio"] == 0
    assert r["n_hay_in_window"] == 0
    assert r["n_key_in_window"] == 2


@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_ablation_with_ratio(mock_batch):
    result = run_ablation(
        key_doc_ids=["k1", "k2"], key_doc_texts=["Key 1", "Key 2"],
        hay_doc_ids=["h1", "h2", "h3", "h4", "h5", "h6"],
        hay_doc_texts=["Hay 1", "Hay 2", "Hay 3", "Hay 4", "Hay 5", "Hay 6"],
        query="test query", model="claude-sonnet-4", ratios=[3],
    )
    r = result.ratio_results[0]
    assert r["ratio"] == 3
    assert r["n_hay_in_window"] == 6
    assert r["n_key_in_window"] == 2


@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_ablation_multiple_ratios(mock_batch):
    result = run_ablation(
        key_doc_ids=["k1"], key_doc_texts=["Key 1"],
        hay_doc_ids=["h1", "h2", "h3", "h4", "h5"],
        hay_doc_texts=["H1", "H2", "H3", "H4", "H5"],
        query="test", model="claude-sonnet-4", ratios=[0, 1, 3, 5],
    )
    assert len(result.ratio_results) == 4
    assert [r["ratio"] for r in result.ratio_results] == [0, 1, 3, 5]


@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_ablation_caps_at_available_hay(mock_batch):
    result = run_ablation(
        key_doc_ids=["k1"], key_doc_texts=["Key 1"],
        hay_doc_ids=["h1", "h2"], hay_doc_texts=["H1", "H2"],
        query="test", model="claude-sonnet-4", ratios=[5],
    )
    r = result.ratio_results[0]
    assert r["n_hay_in_window"] == 2  # only 2 available
