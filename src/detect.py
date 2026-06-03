"""Detection experiment: can a defender distinguish A/B from C?

Metrics:
- Topical density: how tightly clustered are hay docs around key doc topics?
- Embedding distribution: similarity distribution of hay to key docs
- Custodian entropy: how spread across custodians is the hay?
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def topical_density_score(hay_embeddings: np.ndarray, key_embedding: np.ndarray) -> float:
    """Mean cosine similarity of hay documents to a key document centroid."""
    key_2d = key_embedding.reshape(1, -1)
    sims = cosine_similarity(hay_embeddings, key_2d).flatten()
    return float(np.mean(sims))


def embedding_distribution_stats(hay_embeddings: np.ndarray, key_embeddings: np.ndarray) -> dict:
    """Statistics on the similarity distribution between hay and key docs."""
    sims = cosine_similarity(hay_embeddings, key_embeddings)
    max_per_hay = sims.max(axis=1)
    return {
        "mean_similarity": float(np.mean(max_per_hay)),
        "std_similarity": float(np.std(max_per_hay)),
        "max_similarity": float(np.max(max_per_hay)),
        "min_similarity": float(np.min(max_per_hay)),
    }


def custodian_entropy(custodians: list[str]) -> float:
    """Shannon entropy of custodian distribution."""
    if not custodians:
        return 0.0
    from collections import Counter
    counts = Counter(custodians)
    total = len(custodians)
    probs = np.array([c / total for c in counts.values()])
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))
