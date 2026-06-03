import pytest
import numpy as np
from src.detect import (
    topical_density_score,
    embedding_distribution_stats,
    custodian_entropy,
)


def test_topical_density_higher_for_clustered():
    clustered = np.array([[0.95, 0.05], [0.9, 0.1], [0.92, 0.08]])
    scattered = np.array([[1, 0], [0, 1], [-1, 0]])
    key_embedding = np.array([1.0, 0.0])
    dense_score = topical_density_score(clustered, key_embedding)
    scatter_score = topical_density_score(scattered, key_embedding)
    assert dense_score > scatter_score


def test_embedding_distribution_stats():
    hay_embeddings = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    key_embeddings = np.array([[1.0, 0.0]])
    stats = embedding_distribution_stats(hay_embeddings, key_embeddings)
    assert "mean_similarity" in stats
    assert "std_similarity" in stats
    assert "max_similarity" in stats
    assert stats["mean_similarity"] > 0


def test_custodian_entropy():
    uniform = ["a", "b", "c", "d", "a", "b", "c", "d"]
    skewed = ["a", "a", "a", "a", "a", "a", "b", "c"]
    assert custodian_entropy(uniform) > custodian_entropy(skewed)
