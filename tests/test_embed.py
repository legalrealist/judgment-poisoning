import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.embed import Embedder, get_embedder, SUPPORTED_MODELS


def test_supported_models_list():
    """All 9 models are listed."""
    assert len(SUPPORTED_MODELS) == 9
    assert "openai/text-embedding-3-large" in SUPPORTED_MODELS
    assert "voyage/voyage-law-2" in SUPPORTED_MODELS
    assert "open/contriever" in SUPPORTED_MODELS


def test_embedder_interface():
    """Embedder has embed_texts method returning numpy array."""

    class FakeEmbedder(Embedder):
        def _embed_uncached(self, texts: list[str]) -> np.ndarray:
            return np.random.randn(len(texts), 128)

    emb = FakeEmbedder(model_name="test")
    result = emb.embed_texts(["hello", "world"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 128)


def test_embedder_caches_to_disk(tmp_path):
    """Embedder caches embeddings to disk and reloads them."""

    class FakeEmbedder(Embedder):
        def __init__(self):
            super().__init__(model_name="test", cache_dir=tmp_path)
            self.call_count = 0

        def _embed_uncached(self, texts: list[str]) -> np.ndarray:
            self.call_count += 1
            return np.ones((len(texts), 4))

    emb = FakeEmbedder()
    doc_ids = ["doc_1", "doc_2"]
    texts = ["hello", "world"]

    result1 = emb.embed_documents(doc_ids, texts)
    assert emb.call_count == 1

    result2 = emb.embed_documents(doc_ids, texts)
    assert emb.call_count == 1  # no additional API call
    np.testing.assert_array_equal(result1, result2)
