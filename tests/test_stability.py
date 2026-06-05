"""Tests for the stability experiment module."""

import json
from pathlib import Path
from unittest.mock import patch

from src.stability import (
    classify_individual,
    classify_batch,
    build_paired_batches,
    run_baseline_scoring,
    run_stability_test,
    compute_stability_report,
    PROMPT_STYLES,
    StabilityResult,
)
from src.llm_judge import JudgmentResult


class FakeEmail:
    def __init__(self, text):
        self._text = text

    def to_text(self):
        return self._text


def _mock_llm_individual(prompt, model):
    if "trial lawyer" in prompt:
        return '{"judgment": "KEY", "confidence": 0.6}'
    return '{"judgment": "KEY", "confidence": 0.95}'


def _mock_llm_batch(prompt, model):
    if "DOCUMENT doc1" in prompt or "doc1" in prompt:
        docs = []
        for line in prompt.split("\n"):
            if line.startswith("--- DOCUMENT "):
                doc_id = line.split("--- DOCUMENT ")[1].split(" ---")[0]
                docs.append(doc_id)
        results = [{"doc_id": d, "judgment": "KEY", "confidence": 0.9} for d in docs]
        return json.dumps(results)
    return '[]'


class TestPromptStyles:
    def test_both_styles_exist(self):
        assert "strategic" in PROMPT_STYLES
        assert "textual_criteria" in PROMPT_STYLES

    def test_strategic_has_lawyer_language(self):
        assert "trial lawyer" in PROMPT_STYLES["strategic"]["individual"]
        assert "trial lawyer" in PROMPT_STYLES["strategic"]["batch"]

    def test_textual_criteria_has_observable_predicates(self):
        assert "admission of wrongdoing" in PROMPT_STYLES["textual_criteria"]["individual"]
        assert "admission of wrongdoing" in PROMPT_STYLES["textual_criteria"]["batch"]

    def test_both_classify_key_routine(self):
        for style in PROMPT_STYLES.values():
            assert "KEY" in style["individual"]
            assert "ROUTINE" in style["individual"]
            assert "KEY" in style["batch"]
            assert "ROUTINE" in style["batch"]


class TestClassifyIndividual:
    @patch("src.stability._call_llm")
    def test_returns_judgment(self, mock_llm, tmp_path):
        mock_llm.return_value = '{"judgment": "KEY", "confidence": 0.92}'
        result = classify_individual("doc1", "text", "query", "claude-sonnet-4",
                                     "textual_criteria", tmp_path)
        assert result.doc_id == "doc1"
        assert result.judgment == "KEY"
        assert result.confidence == 0.92

    @patch("src.stability._call_llm")
    def test_caches_result(self, mock_llm, tmp_path):
        mock_llm.return_value = '{"judgment": "ROUTINE", "confidence": 0.85}'
        classify_individual("doc1", "text", "query", "claude-sonnet-4",
                            "textual_criteria", tmp_path)
        classify_individual("doc1", "text", "query", "claude-sonnet-4",
                            "textual_criteria", tmp_path)
        assert mock_llm.call_count == 1

    @patch("src.stability._call_llm")
    def test_different_styles_use_different_prompts(self, mock_llm, tmp_path):
        mock_llm.return_value = '{"judgment": "KEY", "confidence": 0.9}'
        classify_individual("doc1", "text", "query", "claude-sonnet-4",
                            "strategic", tmp_path)
        classify_individual("doc1", "text", "query", "claude-sonnet-4",
                            "textual_criteria", tmp_path)
        assert mock_llm.call_count == 2
        assert "trial lawyer" in mock_llm.call_args_list[0][0][0]
        assert "admission of wrongdoing" in mock_llm.call_args_list[1][0][0]


class TestClassifyBatch:
    @patch("src.stability._call_llm")
    def test_returns_judgments_for_all_docs(self, mock_llm, tmp_path):
        mock_llm.return_value = json.dumps([
            {"doc_id": "d1", "judgment": "KEY", "confidence": 0.9},
            {"doc_id": "d2", "judgment": "ROUTINE", "confidence": 0.8},
        ])
        results = classify_batch(["d1", "d2"], ["t1", "t2"], "query",
                                 "claude-sonnet-4", "textual_criteria", tmp_path)
        assert len(results) == 2
        assert results[0].judgment == "KEY"
        assert results[1].judgment == "ROUTINE"


class TestBuildPairedBatches:
    def test_produces_three_conditions_per_target(self):
        corpus = {f"doc{i}": FakeEmail(f"text {i}") for i in range(50)}
        targets = ["doc0", "doc1"]
        relevant = set(f"doc{i}" for i in range(20))
        non_relevant = set(f"doc{i}" for i in range(20, 50))

        batches = build_paired_batches(
            targets, corpus, relevant, non_relevant, "303", "query", batch_size=10,
        )
        assert len(batches) == 6  # 2 targets x 3 conditions
        conditions = [b["condition"] for b in batches]
        assert conditions.count("control_routine") == 2
        assert conditions.count("adjacent_hay") == 2
        assert conditions.count("offtopic_hay") == 2

    def test_batch_contains_target(self):
        corpus = {f"doc{i}": FakeEmail(f"text {i}") for i in range(50)}
        batches = build_paired_batches(
            ["doc0"], corpus, set(f"doc{i}" for i in range(20)),
            set(f"doc{i}" for i in range(20, 50)), "303", "query", batch_size=10,
        )
        for batch in batches:
            target_in_batch = [d for d in batch["docs"] if d["id"] == batch["target_doc"]]
            assert len(target_in_batch) == 1
            assert target_in_batch[0]["is_key"] is True

    def test_batch_size_respected(self):
        corpus = {f"doc{i}": FakeEmail(f"text {i}") for i in range(100)}
        batches = build_paired_batches(
            ["doc0"], corpus, set(f"doc{i}" for i in range(30)),
            set(f"doc{i}" for i in range(30, 100)), "303", "query", batch_size=20,
        )
        for batch in batches:
            assert len(batch["docs"]) == 20


class TestStabilityMeasurement:
    def test_no_flip_when_consistent(self):
        results = [StabilityResult(
            target_doc="doc1",
            baseline_judgment="KEY",
            judgments_by_condition={
                "control_routine": "KEY",
                "adjacent_hay": "KEY",
                "offtopic_hay": "KEY",
            },
            flipped=False,
        )]
        report = compute_stability_report(results, "textual_criteria", "303")
        assert report.flip_rate == 0.0
        assert report.n_flipped == 0

    def test_flip_detected(self):
        results = [StabilityResult(
            target_doc="doc1",
            baseline_judgment="KEY",
            judgments_by_condition={
                "control_routine": "KEY",
                "adjacent_hay": "ROUTINE",
                "offtopic_hay": "KEY",
            },
            flipped=True,
        )]
        report = compute_stability_report(results, "strategic", "303")
        assert report.flip_rate == 1.0
        assert report.n_flipped == 1
        assert report.per_condition_flip["adjacent_hay"] == 1

    def test_mixed_stability(self):
        results = [
            StabilityResult("d1", "KEY", {"c": "KEY", "a": "KEY", "o": "KEY"}, False),
            StabilityResult("d2", "KEY", {"c": "ROUTINE", "a": "KEY", "o": "KEY"}, True),
            StabilityResult("d3", "ROUTINE", {"c": "ROUTINE", "a": "ROUTINE", "o": "ROUTINE"}, False),
            StabilityResult("d4", "KEY", {"c": "ROUTINE", "a": "ROUTINE", "o": "KEY"}, True),
        ]
        report = compute_stability_report(results, "strategic", "303")
        assert report.n_targets == 4
        assert report.n_flipped == 2
        assert report.flip_rate == 0.5
