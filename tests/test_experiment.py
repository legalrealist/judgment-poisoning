# tests/test_experiment.py
import pytest
import numpy as np
from src.experiment import run_single_experiment, ExperimentResult


def test_run_single_experiment():
    doc_ids = ["key_1", "key_2", "hay_1", "hay_2", "hay_3"]
    doc_embeddings = np.array([
        [0.9, 0.1],  # key_1
        [0.8, 0.2],  # key_2
        [0.1, 0.9],  # hay_1
        [0.2, 0.8],  # hay_2
        [0.3, 0.7],  # hay_3
    ])
    query_embedding = np.array([1.0, 0.0])
    key_doc_ids = {"key_1", "key_2"}

    result = run_single_experiment(
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
        query_embedding=query_embedding,
        key_doc_ids=key_doc_ids,
    )

    assert isinstance(result, ExperimentResult)
    assert result.recall_at_5 == 1.0
    assert result.recall_at_10 == 1.0
    assert result.mrr == 1.0
    assert result.ranked_doc_ids[0] == "key_1"
    assert result.ranked_doc_ids[1] == "key_2"
