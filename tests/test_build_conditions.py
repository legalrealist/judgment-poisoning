import pytest
import numpy as np
from src.build_conditions import (
    build_baseline,
    build_haystacked_a,
    build_haystacked_b,
    build_haystacked_c,
    build_dilution_control,
    ConditionSet,
)
from src.trec_loader import TopicJudgments
from src.enron_parse import EnronEmail


def _make_email(doc_id, custodian, body, subject="test"):
    return EnronEmail(
        doc_id=doc_id,
        from_addr=f"{custodian}@enron.com",
        to_addr="other@enron.com",
        subject=subject,
        body=body,
        date_sent="2001-01-01",
        custodian=custodian,
    )


def _make_corpus():
    """Build a small test corpus with known properties."""
    return {
        "key_1": _make_email("key_1", "smith", "California energy pricing fraud scheme"),
        "key_2": _make_email("key_2", "smith", "Hiding losses from auditors in special purpose entities"),
        "resp_1": _make_email("resp_1", "smith", "Q3 earnings report attached"),
        "resp_2": _make_email("resp_2", "jones", "Board meeting minutes from October"),
        "smith_boring_1": _make_email("smith_boring_1", "smith", "Lunch plans for Friday"),
        "smith_boring_2": _make_email("smith_boring_2", "smith", "Office supplies order"),
        "smith_boring_3": _make_email("smith_boring_3", "smith", "Parking pass renewal"),
        "kw_1": _make_email("kw_1", "adams", "California office lease renewal"),
        "kw_2": _make_email("kw_2", "baker", "Energy sector market report Q2"),
        "kw_3": _make_email("kw_3", "clark", "Pricing update for standard contracts"),
        "offtopic_1": _make_email("offtopic_1", "zzz", "Holiday party planning committee"),
        "offtopic_2": _make_email("offtopic_2", "yyy", "New recycling bins in kitchen"),
        "offtopic_3": _make_email("offtopic_3", "xxx", "Softball team signup"),
    }


def _make_judgments():
    tj = TopicJudgments(topic_id="201")
    tj.highly_relevant = {"key_1", "key_2"}
    tj.relevant = {"resp_1", "resp_2"}
    tj.non_relevant = {
        "smith_boring_1", "smith_boring_2", "smith_boring_3",
        "kw_1", "kw_2", "kw_3",
        "offtopic_1", "offtopic_2", "offtopic_3",
    }
    return tj


def test_build_baseline():
    corpus = _make_corpus()
    judgments = _make_judgments()
    condition = build_baseline(corpus, judgments)
    assert isinstance(condition, ConditionSet)
    assert "key_1" in condition.doc_ids
    assert "key_2" in condition.doc_ids
    assert "resp_1" in condition.doc_ids
    assert "resp_2" in condition.doc_ids
    assert "smith_boring_1" not in condition.doc_ids


def test_build_haystacked_a():
    corpus = _make_corpus()
    judgments = _make_judgments()
    baseline = build_baseline(corpus, judgments)
    condition = build_haystacked_a(corpus, judgments, baseline, hay_count=3)
    for doc_id in baseline.doc_ids:
        assert doc_id in condition.doc_ids
    smith_hay = [d for d in condition.doc_ids if d.startswith("smith_boring")]
    assert len(smith_hay) > 0
    assert "offtopic_1" not in condition.doc_ids


def test_build_haystacked_b():
    corpus = _make_corpus()
    judgments = _make_judgments()
    baseline = build_baseline(corpus, judgments)
    condition = build_haystacked_b(corpus, judgments, baseline, hay_count=3)
    for doc_id in baseline.doc_ids:
        assert doc_id in condition.doc_ids
    kw_hay = [d for d in condition.doc_ids if d.startswith("kw_")]
    assert len(kw_hay) > 0


def test_build_haystacked_c():
    corpus = _make_corpus()
    judgments = _make_judgments()
    baseline = build_baseline(corpus, judgments)

    # Create fake embeddings: key docs near [1,0], kw docs near [0.8,0.2], offtopic near [0,1]
    embeddings = {
        "key_1": [0.95, 0.05],
        "key_2": [0.9, 0.1],
        "resp_1": [0.7, 0.3],
        "resp_2": [0.6, 0.4],
        "smith_boring_1": [0.5, 0.5],
        "smith_boring_2": [0.4, 0.6],
        "smith_boring_3": [0.3, 0.7],
        "kw_1": [0.85, 0.15],
        "kw_2": [0.8, 0.2],
        "kw_3": [0.75, 0.25],
        "offtopic_1": [0.05, 0.95],
        "offtopic_2": [0.1, 0.9],
        "offtopic_3": [0.15, 0.85],
    }

    condition = build_haystacked_c(corpus, judgments, baseline, hay_count=3, embeddings=embeddings)

    for doc_id in baseline.doc_ids:
        assert doc_id in condition.doc_ids
    hay_ids = set(condition.doc_ids) - set(baseline.doc_ids)
    assert len(hay_ids) == 3
    # Hay should be the docs most similar to key docs (near [1,0])
    # kw_1 (0.85,0.15), kw_2 (0.8,0.2), kw_3 (0.75,0.25) should be selected
    for doc_id in hay_ids:
        assert doc_id in {"kw_1", "kw_2", "kw_3"}


def test_build_dilution_control_excludes_relevant_custodians():
    corpus = _make_corpus()
    judgments = _make_judgments()
    baseline = build_baseline(corpus, judgments)
    condition = build_dilution_control(corpus, judgments, baseline, hay_count=3)
    for doc_id in baseline.doc_ids:
        assert doc_id in condition.doc_ids
    hay_ids = set(condition.doc_ids) - set(baseline.doc_ids)
    for doc_id in hay_ids:
        assert corpus[doc_id].custodian not in {"smith", "jones"}
