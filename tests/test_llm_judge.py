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
    judge_individual("doc1", "text", "query", "claude-sonnet-4", cache_dir=tmp_path)
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
