"""Statistical analysis for comparing experimental conditions."""

import numpy as np
from scipy.stats import wilcoxon

from src.metrics import bootstrap_ci


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d effect size. Positive d means group_b > group_a."""
    a = np.array(group_a)
    b = np.array(group_b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled_std)


def wilcoxon_test(group_a: list[float], group_b: list[float]) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    a = np.array(group_a)
    b = np.array(group_b)
    diff = b - a
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        return 0.0, 1.0
    stat, p = wilcoxon(nonzero)
    return float(stat), float(p)


def bonferroni_correct(p_values: list[float], n_comparisons: int) -> list[float]:
    """Bonferroni correction, capped at 1.0."""
    return [min(p * n_comparisons, 1.0) for p in p_values]


def compare_conditions(
    condition_a_values: list[float],
    condition_b_values: list[float],
) -> dict:
    """Compare two conditions. Convention: positive mean_diff/cohens_d means a > b."""
    a = np.array(condition_a_values)
    b = np.array(condition_b_values)
    diffs = (a - b).tolist()
    ci_lower, ci_upper = bootstrap_ci(diffs)
    _, p = wilcoxon_test(condition_a_values, condition_b_values)
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "cohens_d": cohens_d(condition_b_values, condition_a_values),
        "wilcoxon_p": p,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
