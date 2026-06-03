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


def test_parse_qrels_numeric_binary():
    """Parse qrels with numeric 0/1 relevance (interactive 2010 format)."""
    qrels_content = """301 0 3.1007403.DKQYPWMOI55PSMBZWFTW5SDVR0GPGHLIB 1
301 0 3.1007404.ABCDEFGHIJ 0
302 0 3.2000001.XYZXYZXYZ 1"""

    judgments = parse_qrels(qrels_content)

    assert "301" in judgments
    assert "302" in judgments
    topic_301 = judgments["301"]
    assert topic_301.relevant == {"3.1007403.DKQYPWMOI55PSMBZWFTW5SDVR0GPGHLIB"}
    assert topic_301.non_relevant == {"3.1007404.ABCDEFGHIJ"}


def test_parse_qrels_five_fields():
    """Parse interactive 2010 format with 5 fields (probability column)."""
    qrels_content = """301 0 3.1007403.ABC 1 0.85
301 0 3.1007404.DEF 0 0.10
301 0 3.1007405.GHI -1 0.50"""

    judgments = parse_qrels(qrels_content)

    topic_301 = judgments["301"]
    assert topic_301.relevant == {"3.1007403.ABC"}
    assert topic_301.non_relevant == {"3.1007404.DEF"}
    # -1 should be skipped (not assessed)
    assert "3.1007405.GHI" not in topic_301.all_assessed()


def test_parse_qrels_learning_format():
    """Parse learning 2010 format (topic:doc_id cost relevance)."""
    qrels_content = """200:3.818877.G3T4II30F0UK 100 1
200:3.818878.AAABBBCCC 100 0
201:3.900000.XYZXYZ 50 1
201:3.900001.ABCABC 50 -1"""

    judgments = parse_qrels(qrels_content)

    assert "200" in judgments
    assert "201" in judgments
    topic_200 = judgments["200"]
    assert topic_200.relevant == {"3.818877.G3T4II30F0UK"}
    assert topic_200.non_relevant == {"3.818878.AAABBBCCC"}
    topic_201 = judgments["201"]
    assert topic_201.relevant == {"3.900000.XYZXYZ"}
    # -1 should be skipped
    assert "3.900001.ABCABC" not in topic_201.all_assessed()


def test_parse_qrels_negative_one_skipped():
    """Documents with relevance -1 are not assessed and should be skipped."""
    qrels_content = """301 0 doc_a 1
301 0 doc_b -1
301 0 doc_c 0"""

    judgments = parse_qrels(qrels_content)

    topic = judgments["301"]
    assert topic.relevant == {"doc_a"}
    assert topic.non_relevant == {"doc_c"}
    assert "doc_b" not in topic.relevant
    assert "doc_b" not in topic.non_relevant
    assert "doc_b" not in topic.all_assessed()
