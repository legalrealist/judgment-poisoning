import pytest
from pathlib import Path

from src.enron_corpus import save_parsed_corpus, load_parsed_corpus
from src.enron_parse import EnronEmail


@pytest.fixture
def sample_emails():
    return [
        EnronEmail(
            doc_id="doc_001",
            from_addr="alice@enron.com",
            to_addr="bob@enron.com",
            subject="Q3 Report",
            body="Please review the attached report.",
            date_sent="2001-09-01T08:00:00Z",
            custodian="alice@enron.com",
            cc="carol@enron.com",
            bcc="",
        ),
        EnronEmail(
            doc_id="doc_002",
            from_addr="bob@enron.com",
            to_addr="alice@enron.com",
            subject="Re: Q3 Report",
            body="Looks good to me.",
            date_sent="2001-09-02T10:00:00Z",
            custodian="bob@enron.com",
            cc="",
            bcc="dave@enron.com",
        ),
    ]


def test_save_load_roundtrip(tmp_path, sample_emails):
    """Emails survive a save/load roundtrip."""
    output_path = tmp_path / "custodian.json"
    save_parsed_corpus(sample_emails, output_path)

    corpus = load_parsed_corpus(tmp_path)

    assert len(corpus) == 2
    assert "doc_001" in corpus
    assert "doc_002" in corpus

    restored = corpus["doc_001"]
    original = sample_emails[0]
    assert restored.doc_id == original.doc_id
    assert restored.from_addr == original.from_addr
    assert restored.to_addr == original.to_addr
    assert restored.subject == original.subject
    assert restored.body == original.body
    assert restored.date_sent == original.date_sent
    assert restored.custodian == original.custodian
    assert restored.cc == original.cc
    assert restored.bcc == original.bcc


def test_save_creates_parent_dirs(tmp_path, sample_emails):
    """save_parsed_corpus creates parent directories if they don't exist."""
    output_path = tmp_path / "a" / "b" / "corpus.json"
    save_parsed_corpus(sample_emails, output_path)
    assert output_path.exists()


def test_load_empty_dir(tmp_path):
    """Loading from an empty directory returns an empty dict."""
    corpus = load_parsed_corpus(tmp_path)
    assert corpus == {}


def test_load_multiple_json_files(tmp_path, sample_emails):
    """load_parsed_corpus merges emails from multiple JSON files."""
    save_parsed_corpus(sample_emails[:1], tmp_path / "file1.json")
    save_parsed_corpus(sample_emails[1:], tmp_path / "file2.json")

    corpus = load_parsed_corpus(tmp_path)
    assert len(corpus) == 2
    assert "doc_001" in corpus
    assert "doc_002" in corpus
