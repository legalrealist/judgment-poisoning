# src/rank.py
"""Rank documents by cosine similarity to a query embedding."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def rank_documents(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> list[str]:
    query_2d = query_embedding.reshape(1, -1)
    sims = cosine_similarity(doc_embeddings, query_2d).flatten()
    ranked_indices = np.argsort(sims)[::-1]
    return [doc_ids[i] for i in ranked_indices]
