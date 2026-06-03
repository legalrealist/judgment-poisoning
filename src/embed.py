"""Multi-model embedding interface with disk caching.

Supports 9 models:
- Commercial: OpenAI large/small, Cohere v3, Voyage law-2, Gemini, Jina v3
- Open-source: Contriever, BGE-large, E5-mistral
"""

from abc import ABC, abstractmethod
from pathlib import Path
import hashlib

import numpy as np

SUPPORTED_MODELS = [
    "openai/text-embedding-3-large",
    "openai/text-embedding-3-small",
    "cohere/embed-v3",
    "voyage/voyage-law-2",
    "google/gemini-embedding",
    "jina/embeddings-v3",
    "open/contriever",
    "open/bge-large-en-v1.5",
    "open/e5-mistral-7b-instruct",
]


class Embedder(ABC):
    """Abstract base class for embedding models with disk caching."""

    def __init__(self, model_name: str, cache_dir: Path | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("embeddings") / model_name.replace("/", "_")

    def embed_documents(self, doc_ids: list[str], texts: list[str]) -> np.ndarray:
        """Embed documents with disk caching keyed by doc_ids."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(doc_ids)
        if cache_path.exists():
            return np.load(cache_path)
        embeddings = self._embed_uncached(texts)
        np.save(cache_path, embeddings)
        return embeddings

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts without caching (for queries)."""
        return self._embed_uncached(texts)

    @abstractmethod
    def _embed_uncached(self, texts: list[str]) -> np.ndarray: ...

    def _cache_path(self, doc_ids: list[str]) -> Path:
        """Deterministic cache path from sorted doc IDs."""
        key = hashlib.sha256("\n".join(sorted(doc_ids)).encode()).hexdigest()[:16]
        return self.cache_dir / f"{key}.npy"


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3 models."""

    def __init__(self, model_name: str = "openai/text-embedding-3-large", **kwargs):
        super().__init__(model_name, **kwargs)
        self._model = model_name.split("/")[1]

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI

        client = OpenAI()
        all_embeddings = []
        for i in range(0, len(texts), 2048):
            batch = texts[i : i + 2048]
            response = client.embeddings.create(model=self._model, input=batch)
            all_embeddings.extend([d.embedding for d in response.data])
        return np.array(all_embeddings)


class CohereEmbedder(Embedder):
    """Cohere embed-v3 model."""

    def __init__(self, model_name: str = "cohere/embed-v3", **kwargs):
        super().__init__(model_name, **kwargs)

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        import cohere

        client = cohere.Client()
        all_embeddings = []
        for i in range(0, len(texts), 96):
            batch = texts[i : i + 96]
            response = client.embed(
                texts=batch, model="embed-english-v3.0", input_type="search_document"
            )
            all_embeddings.extend(response.embeddings)
        return np.array(all_embeddings)


class VoyageEmbedder(Embedder):
    """Voyage AI voyage-law-2 model."""

    def __init__(self, model_name: str = "voyage/voyage-law-2", **kwargs):
        super().__init__(model_name, **kwargs)

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        import voyageai

        client = voyageai.Client()
        all_embeddings = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            result = client.embed(batch, model="voyage-law-2")
            all_embeddings.extend(result.embeddings)
        return np.array(all_embeddings)


class SentenceTransformerEmbedder(Embedder):
    """Open-source models via sentence-transformers."""

    MODEL_MAP = {
        "open/contriever": "facebook/contriever",
        "open/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
        "open/e5-mistral-7b-instruct": "intfloat/e5-mistral-7b-instruct",
    }

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self._hf_name = self.MODEL_MAP[model_name]
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._hf_name)
        return self._model

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


class GoogleEmbedder(Embedder):
    """Google Gemini embedding model."""

    def __init__(self, model_name: str = "google/gemini-embedding", **kwargs):
        super().__init__(model_name, **kwargs)

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        import google.generativeai as genai

        all_embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/text-embedding-004", content=text
            )
            all_embeddings.append(result["embedding"])
        return np.array(all_embeddings)


class JinaEmbedder(Embedder):
    """Jina embeddings-v3 model."""

    def __init__(self, model_name: str = "jina/embeddings-v3", **kwargs):
        super().__init__(model_name, **kwargs)

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        import os
        import requests

        api_key = os.environ.get("JINA_API_KEY", "")
        all_embeddings = []
        for i in range(0, len(texts), 500):
            batch = texts[i : i + 500]
            response = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "jina-embeddings-v3", "input": batch},
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend([d["embedding"] for d in data["data"]])
        return np.array(all_embeddings)


def get_embedder(model_name: str, cache_dir: Path | None = None) -> Embedder:
    """Factory function to create an embedder by model name."""
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if model_name.startswith("openai/"):
        return OpenAIEmbedder(model_name, **kwargs)
    elif model_name.startswith("cohere/"):
        return CohereEmbedder(model_name, **kwargs)
    elif model_name.startswith("voyage/"):
        return VoyageEmbedder(model_name, **kwargs)
    elif model_name.startswith("google/"):
        return GoogleEmbedder(model_name, **kwargs)
    elif model_name.startswith("jina/"):
        return JinaEmbedder(model_name, **kwargs)
    elif model_name.startswith("open/"):
        return SentenceTransformerEmbedder(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: {SUPPORTED_MODELS}")
