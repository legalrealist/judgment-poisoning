"""Tests for LLM judgment metrics."""

import pytest

from src.llm_judge import JudgmentResult
from src.llm_metrics import (
    precision,
    recall,
    f1,
    false_negative_rate,
    false_positive_rate,
    judgment_degradation,
    slot_infiltration_rate,
    confidence_stats,
)


def _make_judgments(specs: list[tuple[str, str, float]]) -> list[JudgmentResult]:
    return [JudgmentResult(doc_id=d, judgment=j, confidence=c) for d, j, c in specs]


def test_perfect_precision():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "NOT RELEVANT", 0.8),
    ])
    key_docs = {"d1"}
    assert precision(judgments, key_docs) == 1.0


def test_perfect_recall():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "RELEVANT", 0.85),
    ])
    key_docs = {"d1", "d2"}
    assert recall(judgments, key_docs) == 1.0


def test_precision_with_false_positive():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "RELEVANT", 0.7),
        ("d3", "NOT RELEVANT", 0.8),
    ])
    key_docs = {"d1"}
    assert precision(judgments, key_docs) == 0.5


def test_recall_with_false_negative():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "NOT RELEVANT", 0.6),
    ])
    key_docs = {"d1", "d2"}
    assert recall(judgments, key_docs) == 0.5


def test_f1_score():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "RELEVANT", 0.7),
        ("d3", "NOT RELEVANT", 0.6),
    ])
    key_docs = {"d1", "d3"}
    p = precision(judgments, key_docs)
    r = recall(judgments, key_docs)
    assert f1(judgments, key_docs) == pytest.approx(2 * p * r / (p + r))


def test_false_negative_rate():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "NOT RELEVANT", 0.6),
        ("d3", "NOT RELEVANT", 0.5),
    ])
    key_docs = {"d1", "d2", "d3"}
    assert false_negative_rate(judgments, key_docs) == pytest.approx(2 / 3)


def test_false_positive_rate_on_hay():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("h1", "RELEVANT", 0.7),
        ("h2", "NOT RELEVANT", 0.8),
    ])
    key_docs = {"d1"}
    hay_docs = {"h1", "h2"}
    assert false_positive_rate(judgments, key_docs, hay_docs) == 0.5


def test_judgment_degradation():
    baseline_fnr = 0.1
    attack_fnr = 0.4
    assert judgment_degradation(baseline_fnr, attack_fnr) == pytest.approx(0.3)


def test_slot_infiltration_rate():
    top_k_ids = ["d1", "h1", "h2", "d2", "h3"]
    hay_doc_ids = {"h1", "h2", "h3"}
    assert slot_infiltration_rate(top_k_ids, hay_doc_ids) == pytest.approx(3 / 5)


def test_confidence_stats():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "RELEVANT", 0.7),
        ("d3", "NOT RELEVANT", 0.3),
    ])
    key_docs = {"d1", "d2"}
    stats = confidence_stats(judgments, key_docs)
    assert stats["mean_confidence_on_key_docs"] == pytest.approx(0.8)
    assert stats["n_key_docs_judged"] == 2


def test_precision_no_relevant_calls():
    judgments = _make_judgments([
        ("d1", "NOT RELEVANT", 0.9),
    ])
    assert precision(judgments, {"d1"}) == 0.0


def test_recall_no_key_docs():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
    ])
    assert recall(judgments, set()) == 0.0
