# tests/test_stats.py
import pytest
import numpy as np
from src.stats import (
    wilcoxon_test,
    cohens_d,
    bonferroni_correct,
    compare_conditions,
)


def test_cohens_d_large_effect():
    group_a = [0.1, 0.15, 0.12, 0.08, 0.11]
    group_b = [0.9, 0.85, 0.88, 0.92, 0.87]
    d = cohens_d(group_a, group_b)
    assert abs(d) > 2.0


def test_cohens_d_no_effect():
    group_a = [0.5, 0.5, 0.5, 0.5]
    group_b = [0.5, 0.5, 0.5, 0.5]
    d = cohens_d(group_a, group_b)
    assert abs(d) < 0.01


def test_wilcoxon_significant():
    group_a = [0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.14, 0.13]
    group_b = [0.8, 0.9, 0.85, 0.82, 0.88, 0.81, 0.84, 0.83]
    stat, p = wilcoxon_test(group_a, group_b)
    assert p < 0.05


def test_bonferroni_correct():
    p_values = [0.01, 0.04, 0.06]
    corrected = bonferroni_correct(p_values, n_comparisons=9)
    assert corrected[0] == pytest.approx(0.09)
    assert corrected[1] == pytest.approx(0.36)
    assert corrected[2] == pytest.approx(0.54)


def test_bonferroni_caps_at_1():
    p_values = [0.5]
    corrected = bonferroni_correct(p_values, n_comparisons=9)
    assert corrected[0] == 1.0


def test_compare_conditions():
    haystacked = [0.2, 0.0, 0.4, 0.2, 0.0, 0.2, 0.0, 0.4, 0.2, 0.0]
    dilution = [0.6, 0.8, 0.6, 0.4, 0.8, 0.6, 0.8, 0.6, 0.4, 0.6]
    result = compare_conditions(haystacked, dilution)
    assert "mean_diff" in result
    assert "cohens_d" in result
    assert "wilcoxon_p" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert result["mean_diff"] < 0
