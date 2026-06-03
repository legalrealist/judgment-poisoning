# LLM Judgment Poisoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM judge layer on top of the existing dense retrieval pipeline to measure whether strategic document haystacking degrades LLM relevance judgments in eDiscovery-style review.

**Architecture:** Three new modules (`llm_judge.py`, `llm_metrics.py`, `llm_experiment.py`) layer on top of the existing condition/embed/rank pipeline. `llm_judge.py` handles LLM API calls with disk caching. `llm_metrics.py` computes judgment-specific metrics (precision, recall, FNR, FPR). `llm_experiment.py` orchestrates three measurement modes (end-to-end, retrieval audit, ablation). A new top-level script `run_llm_evaluation.py` drives everything.

**Tech Stack:** Python 3, anthropic SDK (0.93.0), openai SDK (2.31.0), existing numpy/sklearn/scipy stack. Together AI or Fireworks for open-source models (to be installed when needed).

---

### Task 1: LLM Judge — Core Interface with Disk Caching

**Files:**
- Create: `src/llm_judge.py`
- Create: `tests/test_llm_judge.py`

This is the foundation — a function that sends a document + query to an LLM and gets back a relevance judgment. All LLM calls are cached to disk so reruns cost nothing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm_judge.py
"""Tests for LLM judge — uses mocked API calls."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.llm_judge import (
    JudgmentResult,
    judge_individual,
    judge_batch,
    _cache_key,
    _load_cache,
    _save_cache,
)


def test_judgment_result_fields():
    r = JudgmentResult(doc_id="doc1", judgment="RELEVANT", confidence=0.95)
    assert r.doc_id == "doc1"
    assert r.judgment == "RELEVANT"
    assert r.confidence == 0.95
    assert r.is_relevant is True


def test_judgment_result_not_relevant():
    r = JudgmentResult(doc_id="doc2", judgment="NOT RELEVANT", confidence=0.3)
    assert r.is_relevant is False


def test_cache_key_deterministic():
    k1 = _cache_key("claude-sonnet-4", "doc text here", "query text", "individual")
    k2 = _cache_key("claude-sonnet-4", "doc text here", "query text", "individual")
    assert k1 == k2


def test_cache_key_varies_by_model():
    k1 = _cache_key("claude-sonnet-4", "doc text", "query", "individual")
    k2 = _cache_key("gpt-4o", "doc text", "query", "individual")
    assert k1 != k2


def test_cache_key_varies_by_paradigm():
    k1 = _cache_key("claude-sonnet-4", "doc text", "query", "individual")
    k2 = _cache_key("claude-sonnet-4", "doc text", "query", "batch")
    assert k1 != k2


def test_cache_roundtrip(tmp_path):
    result = JudgmentResult(doc_id="d1", judgment="RELEVANT", confidence=0.9)
    key = "abc123"
    _save_cache(tmp_path, key, [result])
    loaded = _load_cache(tmp_path, key)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].doc_id == "d1"
    assert loaded[0].judgment == "RELEVANT"
    assert loaded[0].confidence == 0.9


def test_load_cache_miss(tmp_path):
    assert _load_cache(tmp_path, "nonexistent") is None


MOCK_ANTHROPIC_RESPONSE = {
    "judgment": "RELEVANT",
    "confidence": 0.85,
}


def _mock_anthropic_client():
    client = MagicMock()
    message = MagicMock()
    message.content = [MagicMock(text=json.dumps(MOCK_ANTHROPIC_RESPONSE))]
    client.messages.create.return_value = message
    return client


@patch("src.llm_judge._get_anthropic_client")
def test_judge_individual_anthropic(mock_get_client, tmp_path):
    mock_get_client.return_value = _mock_anthropic_client()
    result = judge_individual(
        doc_id="doc1",
        doc_text="Email about lobbying efforts in California.",
        query="Documents related to lobbying activities.",
        model="claude-sonnet-4",
        cache_dir=tmp_path,
    )
    assert result.doc_id == "doc1"
    assert result.judgment == "RELEVANT"
    assert result.confidence == 0.85


@patch("src.llm_judge._get_anthropic_client")
def test_judge_individual_uses_cache(mock_get_client, tmp_path):
    mock_get_client.return_value = _mock_anthropic_client()
    # First call hits API
    judge_individual("doc1", "text", "query", "claude-sonnet-4", cache_dir=tmp_path)
    # Second call should use cache — reset mock to verify
    mock_get_client.return_value.messages.create.reset_mock()
    result = judge_individual("doc1", "text", "query", "claude-sonnet-4", cache_dir=tmp_path)
    mock_get_client.return_value.messages.create.assert_not_called()
    assert result.judgment == "RELEVANT"


MOCK_BATCH_RESPONSE = [
    {"doc_id": "d1", "judgment": "RELEVANT", "confidence": 0.9},
    {"doc_id": "d2", "judgment": "NOT RELEVANT", "confidence": 0.7},
]


def _mock_anthropic_client_batch():
    client = MagicMock()
    message = MagicMock()
    message.content = [MagicMock(text=json.dumps(MOCK_BATCH_RESPONSE))]
    client.messages.create.return_value = message
    return client


@patch("src.llm_judge._get_anthropic_client")
def test_judge_batch_anthropic(mock_get_client, tmp_path):
    mock_get_client.return_value = _mock_anthropic_client_batch()
    results = judge_batch(
        doc_ids=["d1", "d2"],
        doc_texts=["Lobbying email", "Lunch plans"],
        query="Documents about lobbying.",
        model="claude-sonnet-4",
        cache_dir=tmp_path,
    )
    assert len(results) == 2
    assert results[0].doc_id == "d1"
    assert results[0].is_relevant is True
    assert results[1].doc_id == "d2"
    assert results[1].is_relevant is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_judge.py -v`
Expected: FAIL — `src.llm_judge` does not exist

- [ ] **Step 3: Implement llm_judge.py**

```python
# src/llm_judge.py
"""LLM-based document relevance judgment with disk caching."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path("llm_cache")

INDIVIDUAL_PROMPT = """Review this document for responsiveness to the following request:

REQUEST: {query}

DOCUMENT:
{doc_text}

Respond with JSON only: {{"judgment": "RELEVANT" or "NOT RELEVANT", "confidence": 0.0-1.0}}"""

BATCH_PROMPT = """Review each document for responsiveness to the following request:

REQUEST: {query}

{documents}

Respond with JSON only — an array of objects, one per document, in the same order:
[{{"doc_id": "...", "judgment": "RELEVANT" or "NOT RELEVANT", "confidence": 0.0-1.0}}, ...]"""


@dataclass
class JudgmentResult:
    doc_id: str
    judgment: str
    confidence: float

    @property
    def is_relevant(self) -> bool:
        return self.judgment.upper() == "RELEVANT"


def _cache_key(model: str, doc_text: str, query: str, paradigm: str) -> str:
    content = f"{model}|{doc_text}|{query}|{paradigm}"
    return hashlib.sha256(content.encode()).hexdigest()[:24]


def _load_cache(cache_dir: Path, key: str) -> list[JudgmentResult] | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return [JudgmentResult(**r) for r in data]


def _save_cache(cache_dir: Path, key: str, results: list[JudgmentResult]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    data = [{"doc_id": r.doc_id, "judgment": r.judgment, "confidence": r.confidence} for r in results]
    with open(path, "w") as f:
        json.dump(data, f)


def _get_anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def _get_openai_client():
    import openai
    return openai.OpenAI()


def _call_anthropic(prompt: str, model: str) -> str:
    client = _get_anthropic_client()
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(prompt: str, model: str) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_together(prompt: str, model: str) -> str:
    client = _get_openai_client()
    client.base_url = "https://api.together.xyz/v1"
    client.api_key = os.environ.get("TOGETHER_API_KEY", "")
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


MODEL_BACKENDS = {
    "claude-sonnet-4": ("anthropic", "claude-sonnet-4-20250514"),
    "gpt-4o": ("openai", "gpt-4o"),
    "llama-3.1-70b": ("together", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    "qwen-2.5-72b": ("together", "Qwen/Qwen2.5-72B-Instruct-Turbo"),
}


def _call_llm(prompt: str, model: str) -> str:
    if model not in MODEL_BACKENDS:
        raise ValueError(f"Unknown model: {model}. Supported: {list(MODEL_BACKENDS.keys())}")
    backend, model_id = MODEL_BACKENDS[model]
    if backend == "anthropic":
        return _call_anthropic(prompt, model_id)
    elif backend == "openai":
        return _call_openai(prompt, model_id)
    elif backend == "together":
        return _call_together(prompt, model_id)
    raise ValueError(f"Unknown backend: {backend}")


def _parse_json(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def judge_individual(
    doc_id: str,
    doc_text: str,
    query: str,
    model: str,
    cache_dir: Path | None = None,
) -> JudgmentResult:
    cache_dir = cache_dir or CACHE_DIR
    key = _cache_key(model, doc_text, query, "individual")
    cached = _load_cache(cache_dir, key)
    if cached:
        result = cached[0]
        result.doc_id = doc_id
        return result

    prompt = INDIVIDUAL_PROMPT.format(query=query, doc_text=doc_text)
    response = _call_llm(prompt, model)
    parsed = _parse_json(response)
    result = JudgmentResult(
        doc_id=doc_id,
        judgment=parsed["judgment"],
        confidence=float(parsed["confidence"]),
    )
    _save_cache(cache_dir, key, [result])
    return result


def judge_batch(
    doc_ids: list[str],
    doc_texts: list[str],
    query: str,
    model: str,
    cache_dir: Path | None = None,
) -> list[JudgmentResult]:
    cache_dir = cache_dir or CACHE_DIR
    combined_text = "\n\n".join(
        f"--- DOCUMENT {doc_id} ---\n{text}" for doc_id, text in zip(doc_ids, doc_texts)
    )
    key = _cache_key(model, combined_text, query, "batch")
    cached = _load_cache(cache_dir, key)
    if cached:
        for i, r in enumerate(cached):
            r.doc_id = doc_ids[i]
        return cached

    prompt = BATCH_PROMPT.format(query=query, documents=combined_text)
    response = _call_llm(prompt, model)
    parsed = _parse_json(response)
    results = []
    for i, entry in enumerate(parsed):
        results.append(JudgmentResult(
            doc_id=doc_ids[i],
            judgment=entry["judgment"],
            confidence=float(entry["confidence"]),
        ))
    _save_cache(cache_dir, key, results)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_judge.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_judge.py tests/test_llm_judge.py
git commit -m "feat: LLM judge with disk caching and multi-backend support"
```

---

### Task 2: LLM Metrics — Judgment-Specific Evaluation

**Files:**
- Create: `src/llm_metrics.py`
- Create: `tests/test_llm_metrics.py`

Computes precision, recall, F1, false positive/negative rates, and judgment degradation from LLM judgment results against TREC ground truth.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm_metrics.py
"""Tests for LLM judgment metrics."""

import pytest

from src.llm_judge import JudgmentResult
from src.llm_metrics import (
    precision,
    recall,
    f1,
    false_negative_rate,
    false_positive_rate,
    judgment_degradation,
    slot_infiltration_rate,
    confidence_stats,
)


def _make_judgments(specs: list[tuple[str, str, float]]) -> list[JudgmentResult]:
    return [JudgmentResult(doc_id=d, judgment=j, confidence=c) for d, j, c in specs]


# Perfect case: LLM gets everything right
def test_perfect_precision():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "NOT RELEVANT", 0.8),
    ])
    key_docs = {"d1"}
    assert precision(judgments, key_docs) == 1.0


def test_perfect_recall():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "RELEVANT", 0.85),
    ])
    key_docs = {"d1", "d2"}
    assert recall(judgments, key_docs) == 1.0


# Imperfect cases
def test_precision_with_false_positive():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),      # true positive
        ("d2", "RELEVANT", 0.7),      # false positive (hay)
        ("d3", "NOT RELEVANT", 0.8),  # true negative
    ])
    key_docs = {"d1"}
    assert precision(judgments, key_docs) == 0.5


def test_recall_with_false_negative():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "NOT RELEVANT", 0.6),  # false negative — missed key doc
    ])
    key_docs = {"d1", "d2"}
    assert recall(judgments, key_docs) == 0.5


def test_f1_score():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "RELEVANT", 0.7),      # false positive
        ("d3", "NOT RELEVANT", 0.6),  # false negative
    ])
    key_docs = {"d1", "d3"}
    p = precision(judgments, key_docs)
    r = recall(judgments, key_docs)
    assert f1(judgments, key_docs) == pytest.approx(2 * p * r / (p + r))


def test_false_negative_rate():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "NOT RELEVANT", 0.6),
        ("d3", "NOT RELEVANT", 0.5),
    ])
    key_docs = {"d1", "d2", "d3"}
    assert false_negative_rate(judgments, key_docs) == pytest.approx(2 / 3)


def test_false_positive_rate_on_hay():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),      # key doc
        ("h1", "RELEVANT", 0.7),      # hay — false positive
        ("h2", "NOT RELEVANT", 0.8),  # hay — true negative
    ])
    key_docs = {"d1"}
    hay_docs = {"h1", "h2"}
    assert false_positive_rate(judgments, key_docs, hay_docs) == 0.5


def test_judgment_degradation():
    baseline_fnr = 0.1
    attack_fnr = 0.4
    assert judgment_degradation(baseline_fnr, attack_fnr) == pytest.approx(0.3)


def test_slot_infiltration_rate():
    top_k_ids = ["d1", "h1", "h2", "d2", "h3"]
    hay_doc_ids = {"h1", "h2", "h3"}
    assert slot_infiltration_rate(top_k_ids, hay_doc_ids) == pytest.approx(3 / 5)


def test_confidence_stats():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
        ("d2", "RELEVANT", 0.7),
        ("d3", "NOT RELEVANT", 0.3),
    ])
    key_docs = {"d1", "d2"}
    stats = confidence_stats(judgments, key_docs)
    assert stats["mean_confidence_on_key_docs"] == pytest.approx(0.8)
    assert stats["n_key_docs_judged"] == 2


def test_precision_no_relevant_calls():
    judgments = _make_judgments([
        ("d1", "NOT RELEVANT", 0.9),
    ])
    assert precision(judgments, {"d1"}) == 0.0


def test_recall_no_key_docs():
    judgments = _make_judgments([
        ("d1", "RELEVANT", 0.9),
    ])
    assert recall(judgments, set()) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_metrics.py -v`
Expected: FAIL — `src.llm_metrics` does not exist

- [ ] **Step 3: Implement llm_metrics.py**

```python
# src/llm_metrics.py
"""Metrics for evaluating LLM relevance judgments against ground truth."""

from src.llm_judge import JudgmentResult


def precision(judgments: list[JudgmentResult], key_docs: set[str]) -> float:
    called_relevant = [j for j in judgments if j.is_relevant]
    if not called_relevant:
        return 0.0
    true_positives = sum(1 for j in called_relevant if j.doc_id in key_docs)
    return true_positives / len(called_relevant)


def recall(judgments: list[JudgmentResult], key_docs: set[str]) -> float:
    if not key_docs:
        return 0.0
    key_in_window = [j for j in judgments if j.doc_id in key_docs]
    true_positives = sum(1 for j in key_in_window if j.is_relevant)
    return true_positives / len(key_docs)


def f1(judgments: list[JudgmentResult], key_docs: set[str]) -> float:
    p = precision(judgments, key_docs)
    r = recall(judgments, key_docs)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def false_negative_rate(judgments: list[JudgmentResult], key_docs: set[str]) -> float:
    if not key_docs:
        return 0.0
    key_in_window = [j for j in judgments if j.doc_id in key_docs]
    missed = sum(1 for j in key_in_window if not j.is_relevant)
    return missed / len(key_docs)


def false_positive_rate(
    judgments: list[JudgmentResult],
    key_docs: set[str],
    hay_docs: set[str] | None = None,
) -> float:
    if hay_docs is not None:
        non_relevant = [j for j in judgments if j.doc_id in hay_docs]
    else:
        non_relevant = [j for j in judgments if j.doc_id not in key_docs]
    if not non_relevant:
        return 0.0
    false_positives = sum(1 for j in non_relevant if j.is_relevant)
    return false_positives / len(non_relevant)


def judgment_degradation(baseline_fnr: float, attack_fnr: float) -> float:
    return attack_fnr - baseline_fnr


def slot_infiltration_rate(top_k_ids: list[str], hay_doc_ids: set[str]) -> float:
    if not top_k_ids:
        return 0.0
    return sum(1 for d in top_k_ids if d in hay_doc_ids) / len(top_k_ids)


def confidence_stats(
    judgments: list[JudgmentResult],
    key_docs: set[str],
) -> dict:
    key_judgments = [j for j in judgments if j.doc_id in key_docs]
    if not key_judgments:
        return {"mean_confidence_on_key_docs": 0.0, "n_key_docs_judged": 0}
    confidences = [j.confidence for j in key_judgments]
    return {
        "mean_confidence_on_key_docs": sum(confidences) / len(confidences),
        "n_key_docs_judged": len(key_judgments),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_metrics.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_metrics.py tests/test_llm_metrics.py
git commit -m "feat: LLM judgment metrics — precision, recall, FNR, FPR, slot infiltration"
```

---

### Task 3: LLM Experiment — Retrieval Audit (Mode 2, No LLM Calls)

**Files:**
- Create: `src/llm_experiment.py`
- Create: `tests/test_llm_experiment.py`

Start with retrieval audit — counts how many top-k slots are hay docs vs. relevant docs. No LLM calls, so it's free to run and validates the pipeline integration.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm_experiment.py
"""Tests for LLM experiment orchestration."""

import numpy as np
import pytest

from src.llm_experiment import run_retrieval_audit, RetrievalAuditResult


def test_retrieval_audit_basic():
    doc_ids = ["k1", "k2", "h1", "h2", "h3", "o1"]
    # Embeddings: key docs close to query, hay docs medium, other far
    doc_embeddings = np.array([
        [1.0, 0.0],   # k1 — key
        [0.9, 0.1],   # k2 — key
        [0.7, 0.3],   # h1 — hay
        [0.6, 0.4],   # h2 — hay
        [0.5, 0.5],   # h3 — hay
        [0.0, 1.0],   # o1 — other
    ])
    query_embedding = np.array([1.0, 0.0])
    key_doc_ids = {"k1", "k2"}
    hay_doc_ids = {"h1", "h2", "h3"}

    result = run_retrieval_audit(
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
        query_embedding=query_embedding,
        key_doc_ids=key_doc_ids,
        hay_doc_ids=hay_doc_ids,
        k=4,
    )
    assert isinstance(result, RetrievalAuditResult)
    assert result.k == 4
    assert result.n_key_in_top_k == 2
    assert result.n_hay_in_top_k == 2
    assert result.n_other_in_top_k == 0
    assert result.slot_infiltration == pytest.approx(0.5)


def test_retrieval_audit_no_hay():
    doc_ids = ["k1", "k2", "o1"]
    doc_embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query_embedding = np.array([1.0, 0.0])

    result = run_retrieval_audit(
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
        query_embedding=query_embedding,
        key_doc_ids={"k1", "k2"},
        hay_doc_ids=set(),
        k=3,
    )
    assert result.slot_infiltration == 0.0
    assert result.n_key_in_top_k == 2


def test_retrieval_audit_returns_ranked_ids():
    doc_ids = ["a", "b", "c"]
    doc_embeddings = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    query_embedding = np.array([1.0, 0.0])

    result = run_retrieval_audit(
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
        query_embedding=query_embedding,
        key_doc_ids=set(),
        hay_doc_ids=set(),
        k=3,
    )
    assert result.top_k_ids[0] == "b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_experiment.py -v`
Expected: FAIL — `src.llm_experiment` does not exist

- [ ] **Step 3: Implement retrieval audit in llm_experiment.py**

```python
# src/llm_experiment.py
"""Experiment orchestration for LLM judgment poisoning."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.rank import rank_documents
from src.llm_judge import JudgmentResult, judge_individual, judge_batch
from src.llm_metrics import (
    precision,
    recall,
    f1,
    false_negative_rate,
    false_positive_rate,
    slot_infiltration_rate,
    confidence_stats,
)


@dataclass
class RetrievalAuditResult:
    k: int
    top_k_ids: list[str]
    n_key_in_top_k: int
    n_hay_in_top_k: int
    n_other_in_top_k: int
    slot_infiltration: float


def run_retrieval_audit(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    key_doc_ids: set[str],
    hay_doc_ids: set[str],
    k: int = 50,
) -> RetrievalAuditResult:
    ranked = rank_documents(doc_ids, doc_embeddings, query_embedding)
    top_k = ranked[:k]
    n_key = sum(1 for d in top_k if d in key_doc_ids)
    n_hay = sum(1 for d in top_k if d in hay_doc_ids)
    n_other = len(top_k) - n_key - n_hay
    return RetrievalAuditResult(
        k=k,
        top_k_ids=top_k,
        n_key_in_top_k=n_key,
        n_hay_in_top_k=n_hay,
        n_other_in_top_k=n_other,
        slot_infiltration=slot_infiltration_rate(top_k, hay_doc_ids),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_experiment.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_experiment.py tests/test_llm_experiment.py
git commit -m "feat: retrieval audit — count hay/key/other docs in top-k slots"
```

---

### Task 4: LLM Experiment — End-to-End Mode (Mode 1)

**Files:**
- Modify: `src/llm_experiment.py`
- Modify: `tests/test_llm_experiment.py`

Adds the end-to-end mode: retrieve top-k, send to LLM for individual and batch judgment, compute metrics.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_experiment.py`:

```python
from unittest.mock import patch, MagicMock
from src.llm_experiment import run_end_to_end, EndToEndResult


def _mock_judge_individual(doc_id, doc_text, query, model, cache_dir=None):
    """Key docs get RELEVANT, everything else NOT RELEVANT."""
    is_key = doc_id.startswith("k")
    return JudgmentResult(
        doc_id=doc_id,
        judgment="RELEVANT" if is_key else "NOT RELEVANT",
        confidence=0.9 if is_key else 0.8,
    )


def _mock_judge_batch(doc_ids, doc_texts, query, model, cache_dir=None):
    return [_mock_judge_individual(d, t, query, model) for d, t in zip(doc_ids, doc_texts)]


@patch("src.llm_experiment.judge_individual", side_effect=_mock_judge_individual)
@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_end_to_end_basic(mock_batch, mock_individual):
    doc_ids = ["k1", "k2", "h1", "h2"]
    texts = {"k1": "Key doc 1", "k2": "Key doc 2", "h1": "Hay 1", "h2": "Hay 2"}
    doc_embeddings = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.7, 0.3], [0.6, 0.4],
    ])
    query_embedding = np.array([1.0, 0.0])
    key_doc_ids = {"k1", "k2"}
    hay_doc_ids = {"h1", "h2"}

    result = run_end_to_end(
        doc_ids=doc_ids,
        doc_texts=texts,
        doc_embeddings=doc_embeddings,
        query_embedding=query_embedding,
        key_doc_ids=key_doc_ids,
        hay_doc_ids=hay_doc_ids,
        query="test query",
        model="claude-sonnet-4",
        k=4,
    )
    assert isinstance(result, EndToEndResult)
    # Mock gives perfect accuracy — key docs marked relevant, hay not
    assert result.individual.recall == 1.0
    assert result.batch.recall == 1.0
    assert result.individual.false_negative_rate == 0.0


@patch("src.llm_experiment.judge_individual", side_effect=_mock_judge_individual)
@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_end_to_end_with_k(mock_batch, mock_individual):
    """When k < total docs, only top-k are judged."""
    doc_ids = ["k1", "h1", "h2", "h3"]
    texts = {"k1": "Key", "h1": "Hay1", "h2": "Hay2", "h3": "Hay3"}
    doc_embeddings = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0],
    ])
    query_embedding = np.array([1.0, 0.0])

    result = run_end_to_end(
        doc_ids=doc_ids,
        doc_texts=texts,
        doc_embeddings=doc_embeddings,
        query_embedding=query_embedding,
        key_doc_ids={"k1"},
        hay_doc_ids={"h1", "h2", "h3"},
        query="test",
        model="claude-sonnet-4",
        k=2,
    )
    # Only top-2 are judged: k1 and h1
    assert result.n_docs_judged == 2
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_llm_experiment.py::test_end_to_end_basic -v`
Expected: FAIL — `EndToEndResult` not defined

- [ ] **Step 3: Implement end-to-end mode**

Add to `src/llm_experiment.py`:

```python
@dataclass
class ParadigmResult:
    precision: float
    recall: float
    f1: float
    false_negative_rate: float
    false_positive_rate: float
    mean_confidence_on_key: float
    judgments: list[JudgmentResult]


@dataclass
class EndToEndResult:
    retrieval_audit: RetrievalAuditResult
    individual: ParadigmResult
    batch: ParadigmResult
    n_docs_judged: int


def _compute_paradigm_result(
    judgments: list[JudgmentResult],
    key_doc_ids: set[str],
    hay_doc_ids: set[str],
) -> ParadigmResult:
    from src.llm_metrics import precision as p_fn, recall as r_fn, f1 as f1_fn
    from src.llm_metrics import false_negative_rate as fnr_fn, false_positive_rate as fpr_fn
    stats = confidence_stats(judgments, key_doc_ids)
    return ParadigmResult(
        precision=p_fn(judgments, key_doc_ids),
        recall=r_fn(judgments, key_doc_ids),
        f1=f1_fn(judgments, key_doc_ids),
        false_negative_rate=fnr_fn(judgments, key_doc_ids),
        false_positive_rate=fpr_fn(judgments, key_doc_ids, hay_doc_ids),
        mean_confidence_on_key=stats["mean_confidence_on_key_docs"],
        judgments=judgments,
    )


def run_end_to_end(
    doc_ids: list[str],
    doc_texts: dict[str, str],
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    key_doc_ids: set[str],
    hay_doc_ids: set[str],
    query: str,
    model: str,
    k: int = 50,
    cache_dir: Path | None = None,
) -> EndToEndResult:
    audit = run_retrieval_audit(doc_ids, doc_embeddings, query_embedding, key_doc_ids, hay_doc_ids, k)
    top_k_ids = audit.top_k_ids
    top_k_texts = [doc_texts[d] for d in top_k_ids]

    individual_judgments = [
        judge_individual(d, doc_texts[d], query, model, cache_dir)
        for d in top_k_ids
    ]
    batch_judgments = judge_batch(top_k_ids, top_k_texts, query, model, cache_dir)

    return EndToEndResult(
        retrieval_audit=audit,
        individual=_compute_paradigm_result(individual_judgments, key_doc_ids, hay_doc_ids),
        batch=_compute_paradigm_result(batch_judgments, key_doc_ids, hay_doc_ids),
        n_docs_judged=len(top_k_ids),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_experiment.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_experiment.py tests/test_llm_experiment.py
git commit -m "feat: end-to-end mode — retrieve top-k, judge individual + batch, compute metrics"
```

---

### Task 5: LLM Experiment — Ablation Mode (Mode 3)

**Files:**
- Modify: `src/llm_experiment.py`
- Modify: `tests/test_llm_experiment.py`

The core contribution — fixed-window judgment test that isolates the LLM judgment degradation effect from retrieval displacement.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm_experiment.py`:

```python
from src.llm_experiment import run_ablation, AblationResult


@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_ablation_clean(mock_batch):
    """Ratio 0 = key docs only, no hay."""
    result = run_ablation(
        key_doc_ids=["k1", "k2"],
        key_doc_texts=["Key 1", "Key 2"],
        hay_doc_ids=["h1", "h2", "h3", "h4"],
        hay_doc_texts=["Hay 1", "Hay 2", "Hay 3", "Hay 4"],
        query="test query",
        model="claude-sonnet-4",
        ratios=[0],
    )
    assert len(result.ratio_results) == 1
    r = result.ratio_results[0]
    assert r["ratio"] == 0
    assert r["n_hay_in_window"] == 0
    assert r["n_key_in_window"] == 2


@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_ablation_with_ratio(mock_batch):
    """Ratio 3 = 3x hay per key doc."""
    result = run_ablation(
        key_doc_ids=["k1", "k2"],
        key_doc_texts=["Key 1", "Key 2"],
        hay_doc_ids=["h1", "h2", "h3", "h4", "h5", "h6"],
        hay_doc_texts=["Hay 1", "Hay 2", "Hay 3", "Hay 4", "Hay 5", "Hay 6"],
        query="test query",
        model="claude-sonnet-4",
        ratios=[3],
    )
    r = result.ratio_results[0]
    assert r["ratio"] == 3
    assert r["n_hay_in_window"] == 6  # 2 key * 3 ratio
    assert r["n_key_in_window"] == 2


@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_ablation_multiple_ratios(mock_batch):
    result = run_ablation(
        key_doc_ids=["k1"],
        key_doc_texts=["Key 1"],
        hay_doc_ids=["h1", "h2", "h3", "h4", "h5"],
        hay_doc_texts=["H1", "H2", "H3", "H4", "H5"],
        query="test",
        model="claude-sonnet-4",
        ratios=[0, 1, 3, 5],
    )
    assert len(result.ratio_results) == 4
    ratios_returned = [r["ratio"] for r in result.ratio_results]
    assert ratios_returned == [0, 1, 3, 5]


@patch("src.llm_experiment.judge_batch", side_effect=_mock_judge_batch)
def test_ablation_caps_at_available_hay(mock_batch):
    """If ratio requests more hay than available, use all available."""
    result = run_ablation(
        key_doc_ids=["k1"],
        key_doc_texts=["Key 1"],
        hay_doc_ids=["h1", "h2"],
        hay_doc_texts=["H1", "H2"],
        query="test",
        model="claude-sonnet-4",
        ratios=[5],
    )
    r = result.ratio_results[0]
    assert r["n_hay_in_window"] == 2  # only 2 available
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_llm_experiment.py::test_ablation_clean -v`
Expected: FAIL — `AblationResult` not defined

- [ ] **Step 3: Implement ablation mode**

Add to `src/llm_experiment.py`:

```python
import random


@dataclass
class AblationResult:
    ratio_results: list[dict]


def run_ablation(
    key_doc_ids: list[str],
    key_doc_texts: list[str],
    hay_doc_ids: list[str],
    hay_doc_texts: list[str],
    query: str,
    model: str,
    ratios: list[int] | None = None,
    cache_dir: Path | None = None,
    seed: int = 42,
) -> AblationResult:
    if ratios is None:
        ratios = [0, 1, 3, 5]

    rng = random.Random(seed)
    ratio_results = []

    for ratio in ratios:
        n_hay_wanted = len(key_doc_ids) * ratio
        n_hay = min(n_hay_wanted, len(hay_doc_ids))

        if n_hay > 0:
            indices = list(range(len(hay_doc_ids)))
            rng_copy = random.Random(seed)
            rng_copy.shuffle(indices)
            selected = indices[:n_hay]
            selected_ids = [hay_doc_ids[i] for i in selected]
            selected_texts = [hay_doc_texts[i] for i in selected]
        else:
            selected_ids = []
            selected_texts = []

        window_ids = list(key_doc_ids) + selected_ids
        window_texts = list(key_doc_texts) + selected_texts

        # Shuffle window so key docs aren't always first (controls positional bias)
        combined = list(zip(window_ids, window_texts))
        rng_shuffle = random.Random(seed + ratio)
        rng_shuffle.shuffle(combined)
        window_ids, window_texts = zip(*combined) if combined else ([], [])
        window_ids = list(window_ids)
        window_texts = list(window_texts)

        judgments = judge_batch(window_ids, window_texts, query, model, cache_dir)

        key_set = set(key_doc_ids)
        hay_set = set(selected_ids)
        from src.llm_metrics import (
            precision as p_fn, recall as r_fn, f1 as f1_fn,
            false_negative_rate as fnr_fn, false_positive_rate as fpr_fn,
        )
        stats = confidence_stats(judgments, key_set)

        ratio_results.append({
            "ratio": ratio,
            "n_key_in_window": len(key_doc_ids),
            "n_hay_in_window": n_hay,
            "n_total_in_window": len(window_ids),
            "precision": p_fn(judgments, key_set),
            "recall": r_fn(judgments, key_set),
            "f1": f1_fn(judgments, key_set),
            "false_negative_rate": fnr_fn(judgments, key_set),
            "false_positive_rate": fpr_fn(judgments, key_set, hay_set),
            "mean_confidence_on_key": stats["mean_confidence_on_key_docs"],
        })

    return AblationResult(ratio_results=ratio_results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_experiment.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_experiment.py tests/test_llm_experiment.py
git commit -m "feat: ablation mode — fixed-window judgment test at controlled hay ratios"
```

---

### Task 6: Top-Level Evaluation Script

**Files:**
- Create: `run_llm_evaluation.py`

Wires everything together: loads corpus, conditions, embeddings, runs all three modes, saves results. Default: 1 topic x 1 LLM x 1 scale (cheap validation).

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Run LLM judgment poisoning evaluation.

Usage:
    python run_llm_evaluation.py                          # default: topic 303, claude-sonnet-4, medium
    python run_llm_evaluation.py --model gpt-4o           # single model
    python run_llm_evaluation.py --expand                 # full matrix
    python run_llm_evaluation.py --mode ablation          # ablation only
    python run_llm_evaluation.py --mode retrieval-audit   # no LLM calls
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.enron_corpus import load_parsed_corpus
from src.embed import get_embedder
from src.llm_experiment import run_retrieval_audit, run_end_to_end, run_ablation

CONDITIONS_DIR = Path("corpus/conditions")
ENRON_PARSED = Path("corpus/enron/parsed")
RESULTS_DIR = Path("results/llm")
TOPICS_FILE = Path("corpus/enron/trec_topics/topics.json")

DEFAULT_TOPIC = "303"
DEFAULT_SCALE = "medium"
DEFAULT_MODEL = "claude-sonnet-4"
DEFAULT_RETRIEVAL_MODEL = "open/bge-large-en-v1.5"
CONDITIONS = ["baseline", "haystacked_a", "haystacked_b", "dilution_control"]
ABLATION_RATIOS = [0, 1, 3, 5]


def load_query(topic_id: str) -> str:
    with open(TOPICS_FILE) as f:
        topics = json.load(f)["topics"]
    return topics[topic_id]["statement"]


def load_condition(topic_id: str, scale: str, condition_name: str) -> dict:
    path = CONDITIONS_DIR / topic_id / scale / f"{condition_name}.json"
    with open(path) as f:
        return json.load(f)


def run_retrieval_audit_mode(
    topic_id: str, scale: str, retrieval_model: str, corpus: dict
) -> list[dict]:
    print(f"\n=== Retrieval Audit: topic={topic_id}, scale={scale} ===")
    embedder = get_embedder(retrieval_model)
    query_text = load_query(topic_id)

    results = []
    for cond_name in CONDITIONS:
        cond = load_condition(topic_id, scale, cond_name)
        doc_ids = [d for d in cond["doc_ids"] if d in corpus]
        texts = [corpus[d].to_text() for d in doc_ids]
        embeddings = embedder.embed_documents(doc_ids, texts)
        query_emb = embedder.embed_texts([query_text])[0]

        key_doc_ids = set(cond["metadata"].get("key_doc_ids", []))
        hay_doc_ids = set(cond["metadata"].get("hay_doc_ids", []))

        audit = run_retrieval_audit(doc_ids, embeddings, query_emb, key_doc_ids, hay_doc_ids)
        row = {
            "mode": "retrieval_audit",
            "topic_id": topic_id,
            "scale": scale,
            "condition": cond_name,
            "retrieval_model": retrieval_model,
            "k": audit.k,
            "n_docs": len(doc_ids),
            "n_key_in_top_k": audit.n_key_in_top_k,
            "n_hay_in_top_k": audit.n_hay_in_top_k,
            "n_other_in_top_k": audit.n_other_in_top_k,
            "slot_infiltration": audit.slot_infiltration,
        }
        results.append(row)
        print(f"  {cond_name:20s} key={audit.n_key_in_top_k} hay={audit.n_hay_in_top_k} "
              f"other={audit.n_other_in_top_k} infiltration={audit.slot_infiltration:.2f}")
    return results


def run_end_to_end_mode(
    topic_id: str, scale: str, llm_model: str, retrieval_model: str, corpus: dict
) -> list[dict]:
    print(f"\n=== End-to-End: topic={topic_id}, scale={scale}, llm={llm_model} ===")
    embedder = get_embedder(retrieval_model)
    query_text = load_query(topic_id)

    results = []
    for cond_name in CONDITIONS:
        cond = load_condition(topic_id, scale, cond_name)
        doc_ids = [d for d in cond["doc_ids"] if d in corpus]
        doc_texts = {d: corpus[d].to_text() for d in doc_ids}
        texts_list = [doc_texts[d] for d in doc_ids]
        embeddings = embedder.embed_documents(doc_ids, texts_list)
        query_emb = embedder.embed_texts([query_text])[0]

        key_doc_ids = set(cond["metadata"].get("key_doc_ids", []))
        hay_doc_ids = set(cond["metadata"].get("hay_doc_ids", []))

        e2e = run_end_to_end(
            doc_ids, doc_texts, embeddings, query_emb,
            key_doc_ids, hay_doc_ids, query_text, llm_model,
        )
        for paradigm_name, paradigm in [("individual", e2e.individual), ("batch", e2e.batch)]:
            row = {
                "mode": "end_to_end",
                "paradigm": paradigm_name,
                "topic_id": topic_id,
                "scale": scale,
                "condition": cond_name,
                "llm_model": llm_model,
                "retrieval_model": retrieval_model,
                "n_docs_judged": e2e.n_docs_judged,
                "slot_infiltration": e2e.retrieval_audit.slot_infiltration,
                "precision": paradigm.precision,
                "recall": paradigm.recall,
                "f1": paradigm.f1,
                "false_negative_rate": paradigm.false_negative_rate,
                "false_positive_rate": paradigm.false_positive_rate,
                "mean_confidence_on_key": paradigm.mean_confidence_on_key,
            }
            results.append(row)
        print(f"  {cond_name:20s} "
              f"ind_FNR={e2e.individual.false_negative_rate:.3f} "
              f"bat_FNR={e2e.batch.false_negative_rate:.3f} "
              f"infiltration={e2e.retrieval_audit.slot_infiltration:.2f}")
    return results


def run_ablation_mode(
    topic_id: str, scale: str, llm_model: str, corpus: dict
) -> list[dict]:
    print(f"\n=== Ablation: topic={topic_id}, scale={scale}, llm={llm_model} ===")
    query_text = load_query(topic_id)

    results = []
    for cond_name in CONDITIONS:
        if cond_name == "baseline":
            continue
        cond = load_condition(topic_id, scale, cond_name)
        key_ids = [d for d in cond["metadata"].get("key_doc_ids", []) if d in corpus]
        hay_ids = [d for d in cond["metadata"].get("hay_doc_ids", []) if d in corpus]
        key_texts = [corpus[d].to_text() for d in key_ids]
        hay_texts = [corpus[d].to_text() for d in hay_ids]

        ablation = run_ablation(key_ids, key_texts, hay_ids, hay_texts, query_text, llm_model)
        for rr in ablation.ratio_results:
            row = {
                "mode": "ablation",
                "topic_id": topic_id,
                "scale": scale,
                "condition": cond_name,
                "llm_model": llm_model,
                "ratio": rr["ratio"],
                "n_key_in_window": rr["n_key_in_window"],
                "n_hay_in_window": rr["n_hay_in_window"],
                "precision": rr["precision"],
                "recall": rr["recall"],
                "f1": rr["f1"],
                "false_negative_rate": rr["false_negative_rate"],
                "false_positive_rate": rr["false_positive_rate"],
                "mean_confidence_on_key": rr["mean_confidence_on_key"],
            }
            results.append(row)
            print(f"  {cond_name:20s} ratio={rr['ratio']} "
                  f"FNR={rr['false_negative_rate']:.3f} "
                  f"recall={rr['recall']:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run LLM judgment poisoning evaluation")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Topic ID (default: 303)")
    parser.add_argument("--scale", default=DEFAULT_SCALE, help="Scale (default: medium)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model (default: claude-sonnet-4)")
    parser.add_argument("--retrieval-model", default=DEFAULT_RETRIEVAL_MODEL)
    parser.add_argument("--mode", choices=["all", "retrieval-audit", "end-to-end", "ablation"],
                        default="all", help="Which mode(s) to run")
    parser.add_argument("--expand", action="store_true", help="Run full matrix")
    args = parser.parse_args()

    topics = ["303", "301"] if args.expand else [args.topic]
    models = ["claude-sonnet-4", "gpt-4o"] if args.expand else [args.model]
    scales = ["small", "medium", "large"] if args.expand else [args.scale]

    print("Loading corpus...")
    corpus = load_parsed_corpus(ENRON_PARSED)
    print(f"Loaded {len(corpus)} emails")

    all_results = []

    for topic in topics:
        for scale in scales:
            if args.mode in ("all", "retrieval-audit"):
                all_results.extend(
                    run_retrieval_audit_mode(topic, scale, args.retrieval_model, corpus)
                )
            for model in models:
                if args.mode in ("all", "end-to-end"):
                    all_results.extend(
                        run_end_to_end_mode(topic, scale, model, args.retrieval_model, corpus)
                    )
                if args.mode in ("all", "ablation"):
                    all_results.extend(
                        run_ablation_mode(topic, scale, model, corpus)
                    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "llm_results.json"

    # Merge with existing results rather than overwriting
    existing = []
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)

    # Dedup by a composite key
    def _key(r):
        return (r.get("mode"), r.get("topic_id"), r.get("scale"),
                r.get("condition"), r.get("llm_model", ""),
                r.get("paradigm", ""), r.get("ratio", ""))

    seen = {_key(r) for r in all_results}
    merged = all_results + [r for r in existing if _key(r) not in seen]

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nResults saved to {output_file} ({len(merged)} rows, {len(all_results)} new)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is syntactically correct**

Run: `python -c "import ast; ast.parse(open('run_llm_evaluation.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add run_llm_evaluation.py
git commit -m "feat: top-level LLM evaluation script with all three modes"
```

---

### Task 7: Smoke Test — Retrieval Audit on Real Data (No LLM Calls)

**Files:**
- No new files — runs existing code against real data

Validates the pipeline end-to-end without spending any LLM API credits. This surfaces any data loading or integration issues before we start making API calls.

- [ ] **Step 1: Run retrieval audit**

Run: `python run_llm_evaluation.py --mode retrieval-audit --topic 303 --scale medium`
Expected: Output showing slot infiltration rates for each condition. Haystacked A/B should show higher infiltration than dilution control.

- [ ] **Step 2: Inspect results**

Run: `cat results/llm/llm_results.json | python -m json.tool | head -50`
Expected: JSON with mode=retrieval_audit rows, reasonable slot_infiltration values.

- [ ] **Step 3: Fix any issues found, re-run**

If there are data loading issues (missing conditions, doc IDs not in corpus), fix them before proceeding.

---

### Task 8: First LLM Run — Ablation with Claude Sonnet

**Files:**
- No new files — runs existing code

The first real LLM experiment. Runs the ablation (fixed-window judgment test) with Claude Sonnet on topic 303 at medium scale. This is the cheapest meaningful test.

- [ ] **Step 1: Run ablation**

Run: `python run_llm_evaluation.py --mode ablation --topic 303 --scale medium --model claude-sonnet-4`
Expected: FNR values for each condition at each ratio. Key result: does FNR increase with hay ratio for haystacked conditions more than dilution?

- [ ] **Step 2: Inspect results**

Run: `python -c "
import json
with open('results/llm/llm_results.json') as f:
    data = json.load(f)
for r in data:
    if r['mode'] == 'ablation':
        print(f\"{r['condition']:20s} ratio={r['ratio']} FNR={r['false_negative_rate']:.3f} recall={r['recall']:.3f}\")
"`
Expected: Ablation results showing judgment accuracy degradation curve.

- [ ] **Step 3: Estimate cost for full expansion**

Check the LLM cache directory to count calls and estimate token usage:
Run: `ls llm_cache/ | wc -l && du -sh llm_cache/`

- [ ] **Step 4: Commit results**

```bash
git add results/llm/
git commit -m "results: first ablation — Claude Sonnet, topic 303, medium scale"
```

---

### Task 9: End-to-End Run with Claude Sonnet

**Files:**
- No new files

Runs the full end-to-end pipeline: retrieve top-50, judge with Claude Sonnet in both individual and batch paradigms.

- [ ] **Step 1: Run end-to-end**

Run: `python run_llm_evaluation.py --mode end-to-end --topic 303 --scale medium --model claude-sonnet-4`

- [ ] **Step 2: Analyze results**

Run: `python -c "
import json
with open('results/llm/llm_results.json') as f:
    data = json.load(f)
for r in data:
    if r['mode'] == 'end_to_end':
        print(f\"{r['paradigm']:12s} {r['condition']:20s} FNR={r['false_negative_rate']:.3f} FPR={r['false_positive_rate']:.3f} infiltration={r['slot_infiltration']:.2f}\")
"`

- [ ] **Step 3: Commit results**

```bash
git add results/llm/
git commit -m "results: end-to-end — Claude Sonnet, topic 303, medium scale"
```

---

### Task 10: Expand to GPT-4o

**Files:**
- No new files

Repeat ablation + end-to-end with GPT-4o (Relativity's actual backend). Requires `OPENAI_API_KEY` environment variable.

- [ ] **Step 1: Verify OpenAI key is set**

Run: `python -c "import os; print('OPENAI_API_KEY' in os.environ)"`
Expected: `True`. If `False`, set it before proceeding.

- [ ] **Step 2: Run ablation with GPT-4o**

Run: `python run_llm_evaluation.py --mode ablation --topic 303 --scale medium --model gpt-4o`

- [ ] **Step 3: Run end-to-end with GPT-4o**

Run: `python run_llm_evaluation.py --mode end-to-end --topic 303 --scale medium --model gpt-4o`

- [ ] **Step 4: Compare Claude vs GPT-4o**

Run: `python -c "
import json
with open('results/llm/llm_results.json') as f:
    data = json.load(f)
ablation = [r for r in data if r['mode'] == 'ablation' and r['condition'] == 'haystacked_a']
for r in sorted(ablation, key=lambda x: (x['llm_model'], x['ratio'])):
    print(f\"{r['llm_model']:20s} ratio={r['ratio']} FNR={r['false_negative_rate']:.3f}\")
"`

- [ ] **Step 5: Commit results**

```bash
git add results/llm/
git commit -m "results: GPT-4o ablation + end-to-end, topic 303, medium scale"
```
