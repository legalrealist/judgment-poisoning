import pytest
from src.trec_loader import parse_qrels, TopicJudgments


def test_parse_qrels_binary():
    """Parse a TREC qrels file with binary relevance (R/N/B)."""
    qrels_content = """201 0 doc_001 R
201 0 doc_002 N
201 0 doc_003 R
201 0 doc_004 B
202 0 doc_005 R
202 0 doc_006 N"""

    judgments = parse_qrels(qrels_content)

    assert "201" in judgments
    assert "202" in judgments
    topic_201 = judgments["201"]
    assert topic_201.relevant == {"doc_001", "doc_003"}
    assert topic_201.non_relevant == {"doc_002"}
    assert topic_201.broken == {"doc_004"}


def test_parse_qrels_graded():
    """Parse qrels with numeric graded relevance (0/1/2)."""
    qrels_content = """201 0 doc_001 2
201 0 doc_002 0
201 0 doc_003 1
201 0 doc_004 2"""

    judgments = parse_qrels(qrels_content, graded=True)

    topic_201 = judgments["201"]
    assert topic_201.highly_relevant == {"doc_001", "doc_004"}
    assert topic_201.relevant == {"doc_003"}
    assert topic_201.non_relevant == {"doc_002"}


def test_topic_judgments_key_documents():
    """key_documents() returns highly relevant if graded, else all relevant."""
    qrels_graded = """201 0 doc_001 2
201 0 doc_002 1
201 0 doc_003 0"""

    judgments = parse_qrels(qrels_graded, graded=True)
    assert judgments["201"].key_documents() == {"doc_001"}

    qrels_binary = """201 0 doc_001 R
201 0 doc_002 R
201 0 doc_003 N"""

    judgments = parse_qrels(qrels_binary)
    assert judgments["201"].key_documents() == {"doc_001", "doc_002"}
