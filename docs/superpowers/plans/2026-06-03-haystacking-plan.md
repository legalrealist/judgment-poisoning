# Haystacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether strategic over-production of authentic documents buries key evidence in dense retrieval, using the Enron email corpus with TREC Legal Track relevance judgments.

**Architecture:** Download and parse the EDRM Enron v2 corpus, load TREC 2009-2011 relevance judgments, build experimental conditions (baseline, A/B/C attackers, dilution control) per topic, embed under 9 models, rank by cosine similarity, compute metrics, run detection experiments.

**Tech Stack:** Python 3.11+, numpy, scipy, scikit-learn (cosine similarity, bootstrap), openai/cohere/voyageai/google-generativeai SDKs (commercial embeddings), sentence-transformers + torch (open-source embeddings), lxml (Enron XML parsing), matplotlib/seaborn (figures)

---

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

```gitignore
# Data (too large for git)
corpus/enron/raw/
corpus/enron/parsed/
embeddings/

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Results (regenerable)
results/

# Environment
.env
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "haystacking"
version = "0.1.0"
description = "Adversarial retrieval through strategic over-production in legal discovery"
requires-python = ">=3.11"

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov"]
```

- [ ] **Step 3: Create requirements.txt**

```
# Core
numpy>=1.26
scipy>=1.12
scikit-learn>=1.4
pandas>=2.2

# Embedding APIs
openai>=1.30
cohere>=5.0
voyageai>=0.3
google-generativeai>=0.5
requests  # for Jina API

# Open-source embeddings
sentence-transformers>=3.0
torch>=2.2

# Enron parsing
lxml>=5.0

# Visualization
matplotlib>=3.8
seaborn>=0.13

# Testing
pytest>=8.0
```

- [ ] **Step 4: Create package structure**

```bash
mkdir -p src tests corpus/enron/raw corpus/enron/parsed corpus/conditions embeddings results
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 5: Initialize virtual environment and install**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml requirements.txt src/__init__.py tests/__init__.py
git commit -m "feat: project scaffolding and dependencies"
```

---

### Task 2: Download and parse Enron corpus

**Files:**
- Create: `src/enron_download.py`
- Create: `src/enron_parse.py`
- Create: `tests/test_enron_parse.py`

- [ ] **Step 1: Write the failing test for email parsing**

```python
# tests/test_enron_parse.py
import pytest
from src.enron_parse import parse_edrm_xml_file, EnronEmail


def test_parse_edrm_xml_extracts_fields():
    """Parse a single EDRM XML file and extract email fields."""
    # We'll create a minimal test fixture inline
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <Root>
        <Batch>
            <Documents>
                <Document DocID="doc_001" DocType="Message" MimeType="message/rfc822">
                    <Tags>
                        <Tag TagName="#From" TagValue="smith@enron.com" />
                        <Tag TagName="#To" TagValue="jones@enron.com" />
                        <Tag TagName="#Subject" TagValue="Re: California pricing" />
                        <Tag TagName="#DateSent" TagValue="2001-08-15T10:30:00Z" />
                    </Tags>
                    <Files>
                        <File FileType="Text">
                            <ExternalFile FileName="doc_001.txt" />
                        </File>
                    </Files>
                </Document>
            </Documents>
        </Batch>
    </Root>"""
    text_files = {"doc_001.txt": "Let's discuss the California pricing strategy for next quarter."}

    emails = parse_edrm_xml_file(xml_content, text_files)

    assert len(emails) == 1
    email = emails[0]
    assert isinstance(email, EnronEmail)
    assert email.doc_id == "doc_001"
    assert email.from_addr == "smith@enron.com"
    assert email.to_addr == "jones@enron.com"
    assert email.subject == "Re: California pricing"
    assert "California pricing strategy" in email.body
    assert email.custodian is None  # set later from directory structure


def test_parse_edrm_xml_skips_non_message():
    """Non-message documents (attachments) are skipped."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <Root>
        <Batch>
            <Documents>
                <Document DocID="attach_001" DocType="File" MimeType="application/pdf">
                    <Tags>
                        <Tag TagName="#FileName" TagValue="report.pdf" />
                    </Tags>
                </Document>
            </Documents>
        </Batch>
    </Root>"""

    emails = parse_edrm_xml_file(xml_content, {})
    assert len(emails) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_enron_parse.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.enron_parse'`

- [ ] **Step 3: Implement EnronEmail dataclass and parser**

```python
# src/enron_parse.py
"""Parse EDRM Enron v2 XML files into structured email objects."""

from dataclasses import dataclass, field
from typing import Optional
from lxml import etree


@dataclass
class EnronEmail:
    doc_id: str
    from_addr: str
    to_addr: str
    subject: str
    body: str
    date_sent: str
    custodian: Optional[str] = None
    cc: str = ""
    bcc: str = ""

    def to_text(self) -> str:
        """Render as plain text for embedding."""
        parts = []
        if self.subject:
            parts.append(f"Subject: {self.subject}")
        if self.from_addr:
            parts.append(f"From: {self.from_addr}")
        if self.to_addr:
            parts.append(f"To: {self.to_addr}")
        if self.date_sent:
            parts.append(f"Date: {self.date_sent}")
        if self.body:
            parts.append("")
            parts.append(self.body)
        return "\n".join(parts)


def parse_edrm_xml_file(
    xml_content: str | bytes, text_files: dict[str, str]
) -> list[EnronEmail]:
    """Parse an EDRM XML file and return a list of EnronEmail objects.

    Args:
        xml_content: Raw XML string or bytes.
        text_files: Mapping of filename -> text content for referenced text files.

    Returns:
        List of EnronEmail objects (messages only, attachments skipped).
    """
    if isinstance(xml_content, str):
        xml_content = xml_content.encode("utf-8")

    root = etree.fromstring(xml_content)
    emails = []

    for doc in root.iter("Document"):
        doc_type = doc.get("DocType", "")
        if doc_type != "Message":
            continue

        doc_id = doc.get("DocID", "")
        tags = {}
        for tag in doc.iter("Tag"):
            tags[tag.get("TagName", "")] = tag.get("TagValue", "")

        # Find the text file reference
        body = ""
        for ext_file in doc.iter("ExternalFile"):
            fname = ext_file.get("FileName", "")
            if fname in text_files:
                body = text_files[fname]
                break

        email = EnronEmail(
            doc_id=doc_id,
            from_addr=tags.get("#From", ""),
            to_addr=tags.get("#To", ""),
            subject=tags.get("#Subject", ""),
            body=body,
            date_sent=tags.get("#DateSent", ""),
            cc=tags.get("#CC", ""),
            bcc=tags.get("#BCC", ""),
        )
        emails.append(email)

    return emails
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_enron_parse.py -v
```

Expected: PASS (both tests)

- [ ] **Step 5: Write download script**

```python
# src/enron_download.py
"""Download EDRM Enron v2 XML from Internet Archive.

The full dataset is 73GB across 159 zip files, organized by custodian.
This script downloads selectively — only custodians needed for selected TREC topics.
"""

import subprocess
import sys
from pathlib import Path

ARCHIVE_BASE = "https://archive.org/download/edrm.enron.email.data.set.v2.xml"
RAW_DIR = Path("corpus/enron/raw")


def list_available_custodians() -> list[str]:
    """Fetch the file listing from Internet Archive and extract custodian zip names."""
    # The archive contains files like: allen-p.zip, arnold-j.zip, etc.
    # Each zip contains one custodian's emails in EDRM XML format
    raise NotImplementedError(
        "Implement after inspecting the actual Internet Archive listing. "
        "Use: curl -s 'https://archive.org/download/edrm.enron.email.data.set.v2.xml/' "
        "to see the file listing."
    )


def download_custodian(custodian_zip: str, output_dir: Path = RAW_DIR) -> Path:
    """Download a single custodian's zip file from Internet Archive.

    Args:
        custodian_zip: Filename like 'allen-p.zip'
        output_dir: Where to save the downloaded file.

    Returns:
        Path to the downloaded zip file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / custodian_zip

    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return output_path

    url = f"{ARCHIVE_BASE}/{custodian_zip}"
    print(f"Downloading {url} ...")
    subprocess.run(
        ["curl", "-L", "-o", str(output_path), url],
        check=True,
    )
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.enron_download <custodian_zip> [custodian_zip ...]")
        print("Example: python -m src.enron_download allen-p.zip arnold-j.zip")
        sys.exit(1)

    for name in sys.argv[1:]:
        download_custodian(name)
```

- [ ] **Step 6: Commit**

```bash
git add src/enron_download.py src/enron_parse.py tests/test_enron_parse.py
git commit -m "feat: Enron corpus download and XML parsing"
```

---

### Task 3: Load TREC Legal Track topics and relevance judgments

**Files:**
- Create: `src/trec_loader.py`
- Create: `tests/test_trec_loader.py`
- Create: `corpus/enron/trec_topics/` (directory for downloaded TREC files)
- Create: `corpus/enron/trec_judgments/` (directory for qrels files)

- [ ] **Step 1: Write the failing test for qrels parsing**

```python
# tests/test_trec_loader.py
import pytest
from src.trec_loader import parse_qrels, TopicJudgments


def test_parse_qrels_binary():
    """Parse a TREC qrels file with binary relevance (R/N/B)."""
    # Standard TREC qrels format: topic_id 0 doc_id relevance
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_trec_loader.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement TREC loader**

```python
# src/trec_loader.py
"""Load TREC Legal Track topics and relevance judgments (qrels)."""

from dataclasses import dataclass, field


@dataclass
class TopicJudgments:
    topic_id: str
    highly_relevant: set[str] = field(default_factory=set)
    relevant: set[str] = field(default_factory=set)
    non_relevant: set[str] = field(default_factory=set)
    broken: set[str] = field(default_factory=set)

    def key_documents(self) -> set[str]:
        """Return the most important documents for this topic.

        If graded relevance is available, return highly_relevant only.
        Otherwise, return all relevant documents.
        """
        if self.highly_relevant:
            return self.highly_relevant
        return self.relevant

    def all_assessed(self) -> set[str]:
        """Return all document IDs that were assessed for this topic."""
        return self.highly_relevant | self.relevant | self.non_relevant | self.broken


def parse_qrels(
    content: str, graded: bool = False
) -> dict[str, TopicJudgments]:
    """Parse a TREC qrels file into TopicJudgments per topic.

    Args:
        content: Raw qrels file content. Format: topic_id iter doc_id relevance
        graded: If True, treat relevance as numeric (0=non-relevant, 1=relevant, 2=highly relevant).
                If False, treat as categorical (R=relevant, N=non-relevant, B=broken).

    Returns:
        Dict mapping topic_id -> TopicJudgments.
    """
    judgments: dict[str, TopicJudgments] = {}

    for line in content.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue

        topic_id, _, doc_id, relevance = parts[0], parts[1], parts[2], parts[3]

        if topic_id not in judgments:
            judgments[topic_id] = TopicJudgments(topic_id=topic_id)

        tj = judgments[topic_id]

        if graded:
            score = int(relevance)
            if score >= 2:
                tj.highly_relevant.add(doc_id)
            elif score == 1:
                tj.relevant.add(doc_id)
            else:
                tj.non_relevant.add(doc_id)
        else:
            if relevance == "R":
                tj.relevant.add(doc_id)
            elif relevance == "N":
                tj.non_relevant.add(doc_id)
            elif relevance == "B":
                tj.broken.add(doc_id)

    return judgments
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_trec_loader.py -v
```

Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add src/trec_loader.py tests/test_trec_loader.py
git commit -m "feat: TREC Legal Track qrels parser with graded relevance support"
```

---

### Task 4: Build experimental conditions

**Files:**
- Create: `src/build_conditions.py`
- Create: `tests/test_build_conditions.py`

This is the core experimental setup. For each TREC topic, build the five conditions (baseline, A, B, C, dilution control) at each scale level.

- [ ] **Step 1: Write the failing test for condition building**

```python
# tests/test_build_conditions.py
import pytest
import numpy as np
from src.build_conditions import (
    build_baseline,
    build_haystacked_a,
    build_haystacked_b,
    build_haystacked_c,
    build_dilution_control,
    ConditionSet,
)
from src.trec_loader import TopicJudgments
from src.enron_parse import EnronEmail


def _make_email(doc_id, custodian, body, subject="test"):
    return EnronEmail(
        doc_id=doc_id,
        from_addr=f"{custodian}@enron.com",
        to_addr="other@enron.com",
        subject=subject,
        body=body,
        date_sent="2001-01-01",
        custodian=custodian,
    )


def _make_corpus():
    """Build a small test corpus with known properties."""
    return {
        # Key documents (highly relevant)
        "key_1": _make_email("key_1", "smith", "California energy pricing fraud scheme"),
        "key_2": _make_email("key_2", "smith", "Hiding losses from auditors in special purpose entities"),
        # Responsive documents
        "resp_1": _make_email("resp_1", "smith", "Q3 earnings report attached"),
        "resp_2": _make_email("resp_2", "jones", "Board meeting minutes from October"),
        # Same-custodian non-responsive (for condition A)
        "smith_boring_1": _make_email("smith_boring_1", "smith", "Lunch plans for Friday"),
        "smith_boring_2": _make_email("smith_boring_2", "smith", "Office supplies order"),
        "smith_boring_3": _make_email("smith_boring_3", "smith", "Parking pass renewal"),
        # Keyword-overlapping non-responsive (for condition B)
        "kw_1": _make_email("kw_1", "adams", "California office lease renewal"),
        "kw_2": _make_email("kw_2", "baker", "Energy sector market report Q2"),
        "kw_3": _make_email("kw_3", "clark", "Pricing update for standard contracts"),
        # Off-topic (for dilution control)
        "offtopic_1": _make_email("offtopic_1", "zzz", "Holiday party planning committee"),
        "offtopic_2": _make_email("offtopic_2", "yyy", "New recycling bins in kitchen"),
        "offtopic_3": _make_email("offtopic_3", "xxx", "Softball team signup"),
    }


def _make_judgments():
    tj = TopicJudgments(topic_id="201")
    tj.highly_relevant = {"key_1", "key_2"}
    tj.relevant = {"resp_1", "resp_2"}
    tj.non_relevant = {
        "smith_boring_1", "smith_boring_2", "smith_boring_3",
        "kw_1", "kw_2", "kw_3",
        "offtopic_1", "offtopic_2", "offtopic_3",
    }
    return tj


def test_build_baseline():
    corpus = _make_corpus()
    judgments = _make_judgments()

    condition = build_baseline(corpus, judgments)

    assert isinstance(condition, ConditionSet)
    # Baseline includes all key + responsive documents
    assert "key_1" in condition.doc_ids
    assert "key_2" in condition.doc_ids
    assert "resp_1" in condition.doc_ids
    assert "resp_2" in condition.doc_ids
    # No non-responsive docs
    assert "smith_boring_1" not in condition.doc_ids


def test_build_haystacked_a():
    corpus = _make_corpus()
    judgments = _make_judgments()
    baseline = build_baseline(corpus, judgments)

    condition = build_haystacked_a(corpus, judgments, baseline, hay_count=3)

    # Should include baseline docs
    for doc_id in baseline.doc_ids:
        assert doc_id in condition.doc_ids
    # Should include docs from same custodians as key docs (smith)
    smith_hay = [d for d in condition.doc_ids if d.startswith("smith_boring")]
    assert len(smith_hay) > 0
    # Should NOT include off-topic docs from unrelated custodians
    assert "offtopic_1" not in condition.doc_ids


def test_build_haystacked_b():
    corpus = _make_corpus()
    judgments = _make_judgments()
    baseline = build_baseline(corpus, judgments)

    condition = build_haystacked_b(corpus, judgments, baseline, hay_count=3)

    # Should include baseline docs
    for doc_id in baseline.doc_ids:
        assert doc_id in condition.doc_ids
    # Should include keyword-overlapping docs
    kw_hay = [d for d in condition.doc_ids if d.startswith("kw_")]
    assert len(kw_hay) > 0


def test_build_dilution_control_excludes_relevant_custodians():
    corpus = _make_corpus()
    judgments = _make_judgments()
    baseline = build_baseline(corpus, judgments)

    condition = build_dilution_control(corpus, judgments, baseline, hay_count=3)

    # Should include baseline docs
    for doc_id in baseline.doc_ids:
        assert doc_id in condition.doc_ids
    # Hay should be from unrelated custodians only
    hay_ids = set(condition.doc_ids) - set(baseline.doc_ids)
    for doc_id in hay_ids:
        assert corpus[doc_id].custodian not in {"smith", "jones"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_build_conditions.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement condition builders**

```python
# src/build_conditions.py
"""Build experimental conditions for each TREC topic.

Five conditions:
- Baseline: key documents + responsive documents
- Haystacked A: baseline + same-custodian emails (broad collection)
- Haystacked B: baseline + keyword-overlapping emails
- Haystacked C: baseline + embedding-similar emails
- Dilution control: baseline + off-topic emails (size-matched to C)
"""

from dataclasses import dataclass, field
from collections import Counter
import re

from src.enron_parse import EnronEmail
from src.trec_loader import TopicJudgments


@dataclass
class ConditionSet:
    name: str
    topic_id: str
    doc_ids: list[str]
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.doc_ids)


def build_baseline(
    corpus: dict[str, EnronEmail],
    judgments: TopicJudgments,
) -> ConditionSet:
    """Build baseline: all key documents + all responsive documents."""
    doc_ids = []
    for doc_id in sorted(judgments.key_documents() | judgments.relevant):
        if doc_id in corpus:
            doc_ids.append(doc_id)

    return ConditionSet(
        name="baseline",
        topic_id=judgments.topic_id,
        doc_ids=doc_ids,
        metadata={"key_doc_ids": sorted(judgments.key_documents() & set(doc_ids))},
    )


def build_haystacked_a(
    corpus: dict[str, EnronEmail],
    judgments: TopicJudgments,
    baseline: ConditionSet,
    hay_count: int,
) -> ConditionSet:
    """Haystacked A: baseline + emails from same custodians (broad collection).

    Selects emails from the same custodians whose mailboxes contain key documents.
    No topic filtering — just 'produce everything from these custodians.'
    """
    key_doc_ids = judgments.key_documents()
    key_custodians = set()
    for doc_id in key_doc_ids:
        if doc_id in corpus and corpus[doc_id].custodian:
            key_custodians.add(corpus[doc_id].custodian)

    baseline_set = set(baseline.doc_ids)
    candidates = [
        doc_id
        for doc_id, email in corpus.items()
        if email.custodian in key_custodians
        and doc_id not in baseline_set
    ]

    hay = candidates[:hay_count]

    return ConditionSet(
        name="haystacked_a",
        topic_id=judgments.topic_id,
        doc_ids=baseline.doc_ids + hay,
        metadata={
            "key_doc_ids": baseline.metadata["key_doc_ids"],
            "hay_doc_ids": hay,
            "selection_method": "broad_collection",
            "custodians": sorted(key_custodians),
        },
    )


def _extract_keywords(emails: list[EnronEmail], top_n: int = 20) -> set[str]:
    """Extract top keywords from a set of emails by term frequency."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very", "just", "because",
        "this", "that", "these", "those", "it", "its", "i", "me", "my", "we",
        "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
        "their", "what", "which", "who", "whom", "how", "when", "where", "why",
        "if", "then", "also", "about", "up", "out", "one", "two",
    }
    word_counts: Counter = Counter()
    for email in emails:
        text = f"{email.subject} {email.body}".lower()
        words = re.findall(r"[a-z]+", text)
        word_counts.update(w for w in words if w not in stop_words and len(w) > 2)

    return {word for word, _ in word_counts.most_common(top_n)}


def build_haystacked_b(
    corpus: dict[str, EnronEmail],
    judgments: TopicJudgments,
    baseline: ConditionSet,
    hay_count: int,
) -> ConditionSet:
    """Haystacked B: baseline + keyword-overlapping emails.

    Extracts keywords from key documents, finds corpus emails sharing those keywords.
    """
    key_emails = [corpus[d] for d in judgments.key_documents() if d in corpus]
    keywords = _extract_keywords(key_emails)

    baseline_set = set(baseline.doc_ids)
    scored = []
    for doc_id, email in corpus.items():
        if doc_id in baseline_set:
            continue
        text = f"{email.subject} {email.body}".lower()
        words = set(re.findall(r"[a-z]+", text))
        overlap = len(words & keywords)
        if overlap > 0:
            scored.append((doc_id, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)
    hay = [doc_id for doc_id, _ in scored[:hay_count]]

    return ConditionSet(
        name="haystacked_b",
        topic_id=judgments.topic_id,
        doc_ids=baseline.doc_ids + hay,
        metadata={
            "key_doc_ids": baseline.metadata["key_doc_ids"],
            "hay_doc_ids": hay,
            "selection_method": "keyword_aware",
            "keywords": sorted(keywords),
        },
    )


def build_haystacked_c(
    corpus: dict[str, EnronEmail],
    judgments: TopicJudgments,
    baseline: ConditionSet,
    hay_count: int,
    embeddings: dict[str, list[float]],
) -> ConditionSet:
    """Haystacked C: baseline + embedding-similar emails.

    For each key document, find the most similar emails by cosine similarity.
    Requires pre-computed embeddings for all documents.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    key_doc_ids = sorted(judgments.key_documents() & set(corpus.keys()))
    baseline_set = set(baseline.doc_ids)

    # Get embeddings for key docs
    key_vecs = np.array([embeddings[d] for d in key_doc_ids])

    # Get embeddings for candidate docs
    candidates = [d for d in corpus if d not in baseline_set and d in embeddings]
    if not candidates:
        return ConditionSet(
            name="haystacked_c",
            topic_id=judgments.topic_id,
            doc_ids=baseline.doc_ids,
            metadata={"key_doc_ids": baseline.metadata["key_doc_ids"], "hay_doc_ids": []},
        )

    cand_vecs = np.array([embeddings[d] for d in candidates])

    # Compute max similarity to any key document for each candidate
    sims = cosine_similarity(cand_vecs, key_vecs)  # (n_candidates, n_keys)
    max_sims = sims.max(axis=1)  # (n_candidates,)

    # Select top-K most similar
    top_indices = np.argsort(max_sims)[::-1][:hay_count]
    hay = [candidates[i] for i in top_indices]

    return ConditionSet(
        name="haystacked_c",
        topic_id=judgments.topic_id,
        doc_ids=baseline.doc_ids + hay,
        metadata={
            "key_doc_ids": baseline.metadata["key_doc_ids"],
            "hay_doc_ids": hay,
            "selection_method": "embedding_optimized",
        },
    )


def build_dilution_control(
    corpus: dict[str, EnronEmail],
    judgments: TopicJudgments,
    baseline: ConditionSet,
    hay_count: int,
) -> ConditionSet:
    """Dilution control: baseline + off-topic emails from unrelated custodians.

    Size-matched to haystacked C. Uses emails from custodians not involved
    in the topic's key or responsive documents.
    """
    involved_custodians = set()
    for doc_id in baseline.doc_ids:
        if doc_id in corpus and corpus[doc_id].custodian:
            involved_custodians.add(corpus[doc_id].custodian)

    baseline_set = set(baseline.doc_ids)
    candidates = [
        doc_id
        for doc_id, email in corpus.items()
        if doc_id not in baseline_set
        and email.custodian not in involved_custodians
    ]

    hay = candidates[:hay_count]

    return ConditionSet(
        name="dilution_control",
        topic_id=judgments.topic_id,
        doc_ids=baseline.doc_ids + hay,
        metadata={
            "key_doc_ids": baseline.metadata["key_doc_ids"],
            "hay_doc_ids": hay,
            "selection_method": "off_topic",
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_build_conditions.py -v
```

Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add src/build_conditions.py tests/test_build_conditions.py
git commit -m "feat: build experimental conditions (baseline, A, B, C, dilution control)"
```

---

### Task 5: Embedding interface with multi-model support

**Files:**
- Create: `src/embed.py`
- Create: `tests/test_embed.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed.py
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
        def embed_texts(self, texts: list[str]) -> np.ndarray:
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

    # First call: computes and caches
    result1 = emb.embed_documents(doc_ids, texts)
    assert emb.call_count == 1

    # Second call: loads from cache
    result2 = emb.embed_documents(doc_ids, texts)
    assert emb.call_count == 1  # no additional API call
    np.testing.assert_array_equal(result1, result2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_embed.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement embedding interface**

```python
# src/embed.py
"""Multi-model embedding interface with disk caching.

Supports 9 models:
- Commercial: OpenAI large/small, Cohere v3, Voyage law-2, Gemini, Jina v3
- Open-source: Contriever, BGE-large, E5-mistral
"""

from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import json

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
    """Base class for embedding models with disk caching."""

    def __init__(self, model_name: str, cache_dir: Path | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("embeddings") / model_name.replace("/", "_")

    def embed_documents(
        self, doc_ids: list[str], texts: list[str]
    ) -> np.ndarray:
        """Embed documents, using cache where available.

        Args:
            doc_ids: Unique identifiers for caching.
            texts: Document texts to embed.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(doc_ids)

        if cache_path.exists():
            return np.load(cache_path)

        embeddings = self._embed_uncached(texts)
        np.save(cache_path, embeddings)
        return embeddings

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts without caching. For queries."""
        return self._embed_uncached(texts)

    @abstractmethod
    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        """Actually compute embeddings. Subclasses implement this."""
        ...

    def _cache_path(self, doc_ids: list[str]) -> Path:
        """Deterministic cache key from sorted doc IDs."""
        key = hashlib.sha256("\n".join(sorted(doc_ids)).encode()).hexdigest()[:16]
        return self.cache_dir / f"{key}.npy"


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str = "openai/text-embedding-3-large", **kwargs):
        super().__init__(model_name, **kwargs)
        self._model = model_name.split("/")[1]

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI

        client = OpenAI()
        # Batch in chunks of 2048 (OpenAI limit)
        all_embeddings = []
        for i in range(0, len(texts), 2048):
            batch = texts[i : i + 2048]
            response = client.embeddings.create(model=self._model, input=batch)
            all_embeddings.extend([d.embedding for d in response.data])
        return np.array(all_embeddings)


class CohereEmbedder(Embedder):
    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        import cohere

        client = cohere.Client()
        # Batch in chunks of 96 (Cohere limit)
        all_embeddings = []
        for i in range(0, len(texts), 96):
            batch = texts[i : i + 96]
            response = client.embed(
                texts=batch,
                model="embed-english-v3.0",
                input_type="search_document",
            )
            all_embeddings.extend(response.embeddings)
        return np.array(all_embeddings)


class VoyageEmbedder(Embedder):
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
    """For open-source models: Contriever, BGE, E5-mistral."""

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
    """Factory: return the right embedder for a model name."""
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_embed.py -v
```

Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add src/embed.py tests/test_embed.py
git commit -m "feat: multi-model embedding interface with disk caching"
```

---

### Task 6: Retrieval ranking and metrics

**Files:**
- Create: `src/rank.py`
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test for metrics**

```python
# tests/test_metrics.py
import pytest
import numpy as np
from src.metrics import recall_at_k, mrr, displacement, bootstrap_ci
from src.rank import rank_documents


def test_recall_at_5_perfect():
    """All key docs in top 5 => recall@5 = 1.0"""
    ranked = ["key_1", "key_2", "key_3", "other_1", "other_2"]
    key_docs = {"key_1", "key_2", "key_3"}
    assert recall_at_k(ranked, key_docs, k=5) == 1.0


def test_recall_at_5_partial():
    """2 of 3 key docs in top 5 => recall@5 = 2/3"""
    ranked = ["key_1", "other_1", "key_2", "other_2", "other_3", "key_3"]
    key_docs = {"key_1", "key_2", "key_3"}
    assert recall_at_k(ranked, key_docs, k=5) == pytest.approx(2 / 3)


def test_recall_at_5_none():
    """No key docs in top 5 => recall@5 = 0.0"""
    ranked = ["other_1", "other_2", "other_3", "other_4", "other_5", "key_1"]
    key_docs = {"key_1"}
    assert recall_at_k(ranked, key_docs, k=5) == 0.0


def test_mrr():
    """MRR is 1/rank of first key document."""
    ranked = ["other_1", "other_2", "key_1", "key_2", "other_3"]
    key_docs = {"key_1", "key_2"}
    assert mrr(ranked, key_docs) == pytest.approx(1 / 3)


def test_mrr_first_position():
    ranked = ["key_1", "other_1", "other_2"]
    key_docs = {"key_1"}
    assert mrr(ranked, key_docs) == 1.0


def test_displacement():
    """Displacement = average rank change of key docs."""
    baseline_ranked = ["key_1", "key_2", "other_1"]  # ranks 1, 2
    attack_ranked = ["other_1", "other_2", "key_1", "other_3", "key_2"]  # ranks 3, 5
    key_docs = {"key_1", "key_2"}
    # key_1: 1->3 = +2, key_2: 2->5 = +3, average = 2.5
    assert displacement(baseline_ranked, attack_ranked, key_docs) == pytest.approx(2.5)


def test_rank_documents():
    """Rank documents by cosine similarity to a query."""
    # 3 docs, 2 dimensions
    doc_ids = ["a", "b", "c"]
    doc_embeddings = np.array([
        [1.0, 0.0],  # a: points right
        [0.0, 1.0],  # b: points up
        [0.7, 0.7],  # c: diagonal
    ])
    query_embedding = np.array([1.0, 0.0])  # query points right

    ranked = rank_documents(doc_ids, doc_embeddings, query_embedding)
    assert ranked[0] == "a"  # most similar to query
    assert ranked[1] == "c"  # second most similar
    assert ranked[2] == "b"  # least similar


def test_bootstrap_ci():
    """Bootstrap CI returns (lower, upper) tuple."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    low, high = bootstrap_ci(values, n_resamples=1000, ci=0.95)
    assert low < high
    assert low >= 0.0
    assert high <= 0.6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement rank.py**

```python
# src/rank.py
"""Rank documents by cosine similarity to a query embedding."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def rank_documents(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> list[str]:
    """Rank documents by cosine similarity to a query.

    Args:
        doc_ids: Document identifiers, parallel to doc_embeddings rows.
        doc_embeddings: (n_docs, embedding_dim) array.
        query_embedding: (embedding_dim,) array.

    Returns:
        doc_ids sorted by descending similarity.
    """
    query_2d = query_embedding.reshape(1, -1)
    sims = cosine_similarity(doc_embeddings, query_2d).flatten()
    ranked_indices = np.argsort(sims)[::-1]
    return [doc_ids[i] for i in ranked_indices]
```

- [ ] **Step 4: Implement metrics.py**

```python
# src/metrics.py
"""Evaluation metrics for retrieval experiments."""

import numpy as np


def recall_at_k(ranked: list[str], key_docs: set[str], k: int) -> float:
    """Fraction of key documents appearing in the top-k ranked results.

    Args:
        ranked: Document IDs sorted by descending relevance.
        key_docs: Set of key document IDs.
        k: Cutoff rank.

    Returns:
        Recall@k in [0.0, 1.0].
    """
    if not key_docs:
        return 0.0
    top_k = set(ranked[:k])
    return len(top_k & key_docs) / len(key_docs)


def mrr(ranked: list[str], key_docs: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first key document.

    Args:
        ranked: Document IDs sorted by descending relevance.
        key_docs: Set of key document IDs.

    Returns:
        MRR in (0.0, 1.0], or 0.0 if no key doc is found.
    """
    for i, doc_id in enumerate(ranked):
        if doc_id in key_docs:
            return 1.0 / (i + 1)
    return 0.0


def displacement(
    baseline_ranked: list[str],
    attack_ranked: list[str],
    key_docs: set[str],
) -> float:
    """Average rank change of key documents between baseline and attack.

    Positive values mean key docs moved down (worse).

    Args:
        baseline_ranked: Ranked doc IDs under baseline condition.
        attack_ranked: Ranked doc IDs under attack condition.
        key_docs: Set of key document IDs.

    Returns:
        Average displacement (positive = key docs ranked lower in attack).
    """
    def _rank_of(ranked, doc_id):
        try:
            return ranked.index(doc_id) + 1  # 1-indexed
        except ValueError:
            return len(ranked) + 1

    displacements = []
    for doc_id in key_docs:
        base_rank = _rank_of(baseline_ranked, doc_id)
        attack_rank = _rank_of(attack_ranked, doc_id)
        displacements.append(attack_rank - base_rank)

    return np.mean(displacements) if displacements else 0.0


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean.

    Args:
        values: Observed values (e.g., recall@5 across queries).
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level (e.g., 0.95 for 95% CI).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    values_arr = np.array(values)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(values_arr, size=len(values_arr), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = 1 - ci
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_metrics.py -v
```

Expected: PASS (all 8 tests)

- [ ] **Step 6: Commit**

```bash
git add src/rank.py src/metrics.py tests/test_metrics.py
git commit -m "feat: retrieval ranking and evaluation metrics with bootstrap CIs"
```

---

### Task 7: Experiment runner

**Files:**
- Create: `src/experiment.py`
- Create: `tests/test_experiment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment.py
import pytest
import numpy as np
from src.experiment import run_single_experiment, ExperimentResult


def test_run_single_experiment():
    """Run a single condition/model/query experiment and get metrics."""
    doc_ids = ["key_1", "key_2", "hay_1", "hay_2", "hay_3"]
    # key docs have high similarity to query, hay docs have low
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
    assert result.recall_at_5 == 1.0  # both key docs in top 5 (only 5 docs)
    assert result.recall_at_10 == 1.0
    assert result.mrr == 1.0  # key_1 is rank 1
    assert result.ranked_doc_ids[0] == "key_1"
    assert result.ranked_doc_ids[1] == "key_2"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_experiment.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement experiment runner**

```python
# src/experiment.py
"""Run retrieval experiments across conditions, models, queries, and scale levels."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.rank import rank_documents
from src.metrics import recall_at_k, mrr, displacement, bootstrap_ci


@dataclass
class ExperimentResult:
    condition: str
    model: str
    topic_id: str
    query: str
    scale: str
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ranked_doc_ids: list[str]


def run_single_experiment(
    doc_ids: list[str],
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    key_doc_ids: set[str],
    condition: str = "",
    model: str = "",
    topic_id: str = "",
    query: str = "",
    scale: str = "",
) -> ExperimentResult:
    """Run a single retrieval experiment and compute metrics.

    Args:
        doc_ids: Document IDs in the condition.
        doc_embeddings: Embeddings for those documents.
        query_embedding: Embedding for the query.
        key_doc_ids: Set of key document IDs (ground truth).
        condition/model/topic_id/query/scale: Metadata for the result.

    Returns:
        ExperimentResult with all metrics.
    """
    ranked = rank_documents(doc_ids, doc_embeddings, query_embedding)

    return ExperimentResult(
        condition=condition,
        model=model,
        topic_id=topic_id,
        query=query,
        scale=scale,
        recall_at_5=recall_at_k(ranked, key_doc_ids, k=5),
        recall_at_10=recall_at_k(ranked, key_doc_ids, k=10),
        recall_at_20=recall_at_k(ranked, key_doc_ids, k=20),
        mrr=mrr(ranked, key_doc_ids),
        ranked_doc_ids=ranked,
    )


def run_full_experiment(
    conditions: dict[str, dict],  # {condition_name: {doc_ids, doc_embeddings, key_doc_ids}}
    query_embeddings: dict[str, np.ndarray],  # {query_text: embedding}
    model: str,
    topic_id: str,
    scale: str,
) -> list[ExperimentResult]:
    """Run all queries against all conditions for one model/topic/scale.

    Returns:
        List of ExperimentResult, one per (condition, query) pair.
    """
    results = []
    for condition_name, cond_data in conditions.items():
        for query_text, query_emb in query_embeddings.items():
            result = run_single_experiment(
                doc_ids=cond_data["doc_ids"],
                doc_embeddings=cond_data["doc_embeddings"],
                query_embedding=query_emb,
                key_doc_ids=cond_data["key_doc_ids"],
                condition=condition_name,
                model=model,
                topic_id=topic_id,
                query=query_text,
                scale=scale,
            )
            results.append(result)
    return results


def save_results(results: list[ExperimentResult], output_dir: Path) -> None:
    """Save experiment results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for r in results:
        data.append({
            "condition": r.condition,
            "model": r.model,
            "topic_id": r.topic_id,
            "query": r.query,
            "scale": r.scale,
            "recall_at_5": r.recall_at_5,
            "recall_at_10": r.recall_at_10,
            "recall_at_20": r.recall_at_20,
            "mrr": r.mrr,
        })
    with open(output_dir / "results.json", "w") as f:
        json.dump(data, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_experiment.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiment.py tests/test_experiment.py
git commit -m "feat: experiment runner for retrieval experiments"
```

---

### Task 8: Statistical analysis

**Files:**
- Create: `src/stats.py`
- Create: `tests/test_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stats.py
import pytest
import numpy as np
from src.stats import (
    wilcoxon_test,
    cohens_d,
    bonferroni_correct,
    compare_conditions,
)


def test_cohens_d_large_effect():
    """Large effect size when distributions don't overlap."""
    group_a = [0.1, 0.15, 0.12, 0.08, 0.11]
    group_b = [0.9, 0.85, 0.88, 0.92, 0.87]
    d = cohens_d(group_a, group_b)
    assert abs(d) > 2.0  # very large effect


def test_cohens_d_no_effect():
    """Near-zero effect size for identical distributions."""
    group_a = [0.5, 0.5, 0.5, 0.5]
    group_b = [0.5, 0.5, 0.5, 0.5]
    d = cohens_d(group_a, group_b)
    assert abs(d) < 0.01


def test_wilcoxon_significant():
    """Wilcoxon detects significant difference."""
    group_a = [0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.14, 0.13]
    group_b = [0.8, 0.9, 0.85, 0.82, 0.88, 0.81, 0.84, 0.83]
    stat, p = wilcoxon_test(group_a, group_b)
    assert p < 0.05


def test_bonferroni_correct():
    """Bonferroni correction multiplies p-values by n_comparisons."""
    p_values = [0.01, 0.04, 0.06]
    corrected = bonferroni_correct(p_values, n_comparisons=9)
    assert corrected[0] == pytest.approx(0.09)
    assert corrected[1] == pytest.approx(0.36)
    assert corrected[2] == pytest.approx(0.54)


def test_bonferroni_caps_at_1():
    """Corrected p-values are capped at 1.0."""
    p_values = [0.5]
    corrected = bonferroni_correct(p_values, n_comparisons=9)
    assert corrected[0] == 1.0


def test_compare_conditions():
    """Compare two conditions across queries and get summary stats."""
    # Simulate recall@5 for haystacked vs dilution across 10 queries
    haystacked = [0.2, 0.0, 0.4, 0.2, 0.0, 0.2, 0.0, 0.4, 0.2, 0.0]
    dilution = [0.6, 0.8, 0.6, 0.4, 0.8, 0.6, 0.8, 0.6, 0.4, 0.6]

    result = compare_conditions(haystacked, dilution)

    assert "mean_diff" in result
    assert "cohens_d" in result
    assert "wilcoxon_p" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert result["mean_diff"] < 0  # haystacked has lower recall
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_stats.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement statistical analysis**

```python
# src/stats.py
"""Statistical analysis for comparing experimental conditions."""

import numpy as np
from scipy.stats import wilcoxon

from src.metrics import bootstrap_ci


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d effect size between two groups.

    Positive d means group_b > group_a.
    """
    a = np.array(group_a)
    b = np.array(group_b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled_std)


def wilcoxon_test(
    group_a: list[float], group_b: list[float]
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test.

    Args:
        group_a: Values under condition A (e.g., recall@5 per query).
        group_b: Values under condition B.

    Returns:
        (statistic, p_value)
    """
    a = np.array(group_a)
    b = np.array(group_b)
    diff = b - a

    # Wilcoxon requires non-zero differences
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        return 0.0, 1.0

    stat, p = wilcoxon(nonzero)
    return float(stat), float(p)


def bonferroni_correct(
    p_values: list[float], n_comparisons: int
) -> list[float]:
    """Bonferroni correction: multiply p-values by number of comparisons.

    Args:
        p_values: Raw p-values.
        n_comparisons: Number of comparisons (e.g., 9 for 9 models).

    Returns:
        Corrected p-values, capped at 1.0.
    """
    return [min(p * n_comparisons, 1.0) for p in p_values]


def compare_conditions(
    condition_a_values: list[float],
    condition_b_values: list[float],
) -> dict:
    """Compare two conditions with full statistical analysis.

    Args:
        condition_a_values: Metric values (e.g., recall@5) per query under condition A.
        condition_b_values: Same metric per query under condition B.

    Returns:
        Dict with mean_diff, cohens_d, wilcoxon_p, ci_lower, ci_upper.
    """
    a = np.array(condition_a_values)
    b = np.array(condition_b_values)
    diffs = (a - b).tolist()

    ci_lower, ci_upper = bootstrap_ci(diffs)
    _, p = wilcoxon_test(condition_a_values, condition_b_values)

    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "cohens_d": cohens_d(condition_b_values, condition_a_values),
        "wilcoxon_p": p,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_stats.py -v
```

Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/stats.py tests/test_stats.py
git commit -m "feat: statistical analysis — Wilcoxon, Cohen's d, Bonferroni, bootstrap CIs"
```

---

### Task 9: Detection experiment

**Files:**
- Create: `src/detect.py`
- Create: `tests/test_detect.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detect.py
import pytest
import numpy as np
from src.detect import (
    topical_density_score,
    embedding_distribution_stats,
    custodian_entropy,
)


def test_topical_density_higher_for_clustered():
    """Tightly clustered docs should have higher topical density."""
    # Clustered: all near [1, 0]
    clustered = np.array([[0.95, 0.05], [0.9, 0.1], [0.92, 0.08]])
    # Scattered: spread across space
    scattered = np.array([[1, 0], [0, 1], [-1, 0]])

    key_embedding = np.array([1.0, 0.0])

    dense_score = topical_density_score(clustered, key_embedding)
    scatter_score = topical_density_score(scattered, key_embedding)
    assert dense_score > scatter_score


def test_embedding_distribution_stats():
    """Returns mean and std of similarities to key docs."""
    hay_embeddings = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    key_embeddings = np.array([[1.0, 0.0]])

    stats = embedding_distribution_stats(hay_embeddings, key_embeddings)
    assert "mean_similarity" in stats
    assert "std_similarity" in stats
    assert "max_similarity" in stats
    assert stats["mean_similarity"] > 0


def test_custodian_entropy():
    """Uniform custodian distribution has higher entropy than skewed."""
    uniform = ["a", "b", "c", "d", "a", "b", "c", "d"]
    skewed = ["a", "a", "a", "a", "a", "a", "b", "c"]

    assert custodian_entropy(uniform) > custodian_entropy(skewed)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_detect.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement detection metrics**

```python
# src/detect.py
"""Detection experiment: can a defender distinguish A/B from C?

Metrics that might distinguish adversarial (C) from normal (A/B) over-production:
- Topical density: how tightly clustered are hay docs around key doc topics?
- Embedding distribution: similarity distribution of hay to key docs
- Custodian entropy: how spread across custodians is the hay?
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def topical_density_score(
    hay_embeddings: np.ndarray,
    key_embedding: np.ndarray,
) -> float:
    """Mean cosine similarity of hay documents to a key document centroid.

    Higher = hay is more tightly clustered around key document topics.

    Args:
        hay_embeddings: (n_hay, dim) embeddings of hay documents.
        key_embedding: (dim,) centroid of key document embeddings.

    Returns:
        Mean similarity score.
    """
    key_2d = key_embedding.reshape(1, -1)
    sims = cosine_similarity(hay_embeddings, key_2d).flatten()
    return float(np.mean(sims))


def embedding_distribution_stats(
    hay_embeddings: np.ndarray,
    key_embeddings: np.ndarray,
) -> dict:
    """Statistics on the similarity distribution between hay and key docs.

    Args:
        hay_embeddings: (n_hay, dim) embeddings.
        key_embeddings: (n_key, dim) embeddings.

    Returns:
        Dict with mean_similarity, std_similarity, max_similarity, min_similarity.
    """
    sims = cosine_similarity(hay_embeddings, key_embeddings)
    max_per_hay = sims.max(axis=1)  # max similarity to any key doc

    return {
        "mean_similarity": float(np.mean(max_per_hay)),
        "std_similarity": float(np.std(max_per_hay)),
        "max_similarity": float(np.max(max_per_hay)),
        "min_similarity": float(np.min(max_per_hay)),
    }


def custodian_entropy(custodians: list[str]) -> float:
    """Shannon entropy of custodian distribution.

    Higher entropy = more evenly distributed across custodians.
    Lower entropy = concentrated in fewer custodians.

    Args:
        custodians: List of custodian names for each hay document.

    Returns:
        Shannon entropy in bits.
    """
    if not custodians:
        return 0.0

    from collections import Counter

    counts = Counter(custodians)
    total = len(custodians)
    probs = np.array([c / total for c in counts.values()])
    # Avoid log(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_detect.py -v
```

Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/detect.py tests/test_detect.py
git commit -m "feat: detection metrics — topical density, embedding distribution, custodian entropy"
```

---

### Task 10: Main pipeline script

**Files:**
- Create: `run_experiment.py`

This is the orchestration script that ties everything together. It runs the full pipeline from the spec.

- [ ] **Step 1: Implement the pipeline**

```python
#!/usr/bin/env python3
"""Haystacking: main experiment pipeline.

Usage:
    python run_experiment.py --step download --custodians allen-p.zip lay-k.zip ...
    python run_experiment.py --step parse
    python run_experiment.py --step conditions --topic 201
    python run_experiment.py --step embed --model openai/text-embedding-3-large
    python run_experiment.py --step rank --topic 201 --model openai/text-embedding-3-large
    python run_experiment.py --step evaluate
    python run_experiment.py --step detect
    python run_experiment.py --step all  # run everything
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.embed import get_embedder, SUPPORTED_MODELS
from src.enron_parse import EnronEmail
from src.trec_loader import parse_qrels
from src.build_conditions import (
    build_baseline,
    build_haystacked_a,
    build_haystacked_b,
    build_haystacked_c,
    build_dilution_control,
)
from src.experiment import run_full_experiment, save_results
from src.stats import compare_conditions, bonferroni_correct
from src.detect import topical_density_score, embedding_distribution_stats, custodian_entropy


CORPUS_DIR = Path("corpus")
ENRON_RAW = CORPUS_DIR / "enron" / "raw"
ENRON_PARSED = CORPUS_DIR / "enron" / "parsed"
CONDITIONS_DIR = CORPUS_DIR / "conditions"
EMBEDDINGS_DIR = Path("embeddings")
RESULTS_DIR = Path("results")

# Scale multipliers: hay_count = multiplier * baseline_size
SCALES = {
    "tight": 0,
    "small": 2,
    "medium": 5,
    "large": 10,
}


def step_download(custodians: list[str]):
    """Download specified custodian zip files from Internet Archive."""
    from src.enron_download import download_custodian

    for name in custodians:
        download_custodian(name)


def step_parse():
    """Parse downloaded Enron XML files into structured email objects."""
    # Implementation depends on actual EDRM XML structure.
    # This will be refined after inspecting the downloaded data.
    print("TODO: Parse EDRM XML files from corpus/enron/raw/ into corpus/enron/parsed/")
    print("Each custodian becomes a JSON file with list of EnronEmail dicts.")


def step_conditions(topic_id: str):
    """Build experimental conditions for a topic at all scale levels."""
    print(f"Building conditions for topic {topic_id}...")

    # Load parsed corpus
    parsed_dir = ENRON_PARSED
    corpus = {}
    for f in parsed_dir.glob("*.json"):
        with open(f) as fh:
            emails = json.load(fh)
            for e in emails:
                corpus[e["doc_id"]] = EnronEmail(**e)

    # Load TREC judgments
    qrels_path = CORPUS_DIR / "enron" / "trec_judgments" / "qrels.txt"
    with open(qrels_path) as f:
        judgments_all = parse_qrels(f.read())

    if topic_id not in judgments_all:
        print(f"Topic {topic_id} not found in qrels. Available: {list(judgments_all.keys())}")
        sys.exit(1)

    judgments = judgments_all[topic_id]
    baseline = build_baseline(corpus, judgments)
    print(f"  Baseline: {len(baseline)} docs ({len(judgments.key_documents())} key)")

    for scale_name, multiplier in SCALES.items():
        if multiplier == 0:
            # Tight = baseline only
            out_dir = CONDITIONS_DIR / topic_id / scale_name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "baseline.json", "w") as f:
                json.dump({"doc_ids": baseline.doc_ids, "metadata": baseline.metadata}, f, indent=2)
            continue

        hay_count = multiplier * len(baseline)
        print(f"  Scale {scale_name}: {hay_count} hay docs")

        cond_a = build_haystacked_a(corpus, judgments, baseline, hay_count)
        cond_b = build_haystacked_b(corpus, judgments, baseline, hay_count)

        # C requires embeddings — use a default model for selection
        # (The actual experiment tests all 9 models for ranking)
        print(f"    Embedding for condition C selection...")
        embedder = get_embedder("openai/text-embedding-3-large")
        all_doc_ids = list(corpus.keys())
        all_texts = [corpus[d].to_text() for d in all_doc_ids]
        all_embeddings = embedder.embed_documents(all_doc_ids, all_texts)
        emb_dict = dict(zip(all_doc_ids, all_embeddings.tolist()))

        cond_c = build_haystacked_c(corpus, judgments, baseline, hay_count, emb_dict)
        control = build_dilution_control(corpus, judgments, baseline, len(cond_c) - len(baseline))

        out_dir = CONDITIONS_DIR / topic_id / scale_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for cond in [baseline, cond_a, cond_b, cond_c, control]:
            with open(out_dir / f"{cond.name}.json", "w") as f:
                json.dump({"doc_ids": cond.doc_ids, "metadata": cond.metadata}, f, indent=2)

    print(f"  Conditions saved to {CONDITIONS_DIR / topic_id}/")


def step_embed(model_name: str):
    """Embed all documents needed for experiments under a specific model."""
    print(f"Embedding with {model_name}...")
    embedder = get_embedder(model_name)

    # Collect all unique doc IDs across all conditions
    all_doc_ids = set()
    for cond_file in CONDITIONS_DIR.rglob("*.json"):
        with open(cond_file) as f:
            data = json.load(f)
            all_doc_ids.update(data["doc_ids"])

    # Load texts
    corpus = {}
    for f in ENRON_PARSED.glob("*.json"):
        with open(f) as fh:
            for e in json.load(fh):
                corpus[e["doc_id"]] = EnronEmail(**e)

    doc_ids = sorted(all_doc_ids & set(corpus.keys()))
    texts = [corpus[d].to_text() for d in doc_ids]

    print(f"  Embedding {len(doc_ids)} documents...")
    embeddings = embedder.embed_documents(doc_ids, texts)
    print(f"  Done. Shape: {embeddings.shape}")


def step_evaluate():
    """Run evaluation across all topics, models, scales, conditions."""
    print("Running full evaluation...")
    # Load all results, compute stats, save summary
    # This is where the bulk of the analysis happens
    # Implementation depends on which topics/models have been embedded
    print("TODO: Iterate over all (topic, model, scale, condition) combinations")
    print("      and compute metrics. Save to results/")


def main():
    parser = argparse.ArgumentParser(description="Haystacking experiment pipeline")
    parser.add_argument("--step", required=True,
                        choices=["download", "parse", "conditions", "embed", "rank", "evaluate", "detect", "all"])
    parser.add_argument("--custodians", nargs="+", help="Custodian zip files to download")
    parser.add_argument("--topic", help="TREC topic ID")
    parser.add_argument("--model", help="Embedding model name")

    args = parser.parse_args()

    if args.step == "download":
        if not args.custodians:
            print("--custodians required for download step")
            sys.exit(1)
        step_download(args.custodians)
    elif args.step == "parse":
        step_parse()
    elif args.step == "conditions":
        if not args.topic:
            print("--topic required for conditions step")
            sys.exit(1)
        step_conditions(args.topic)
    elif args.step == "embed":
        if not args.model:
            print("--model required for embed step")
            sys.exit(1)
        step_embed(args.model)
    elif args.step == "evaluate":
        step_evaluate()
    elif args.step == "detect":
        print("TODO: Run detection experiment")
    elif args.step == "all":
        print("Run steps manually in sequence. See --help for usage.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses arguments correctly**

```bash
python run_experiment.py --help
python run_experiment.py --step download --custodians allen-p.zip lay-k.zip
```

Expected: Help text displays correctly. Download step attempts to fetch files.

- [ ] **Step 3: Commit**

```bash
git add run_experiment.py
git commit -m "feat: main pipeline script with step-by-step execution"
```

---

### Task 11: Exploratory data inspection

**This task has no code to write.** It's the first real interaction with the data.

- [ ] **Step 1: Download the TREC qrels and topic files**

```bash
# TREC 2009 topics and judgments
curl -o corpus/enron/trec_judgments/BatchTopics2009.zip \
  https://trec.nist.gov/data/legal/09/BatchTopics2009.zip
unzip corpus/enron/trec_judgments/BatchTopics2009.zip -d corpus/enron/trec_judgments/

# TREC 2010 qrels (learning task)
curl -o corpus/enron/trec_judgments/qrels.t10legallearn.gz \
  https://trec.nist.gov/data/legal/10/qrels.t10legallearn.gz
gunzip corpus/enron/trec_judgments/qrels.t10legallearn.gz

# TREC 2010 interactive task qrels
curl -o corpus/enron/trec_judgments/qrel_leg_int_2010_msg_post.txt \
  https://trec.nist.gov/data/legal/10/qrel_leg_int_2010_msg_post.txt
```

- [ ] **Step 2: Inspect qrels format and topic descriptions**

```bash
# Check the qrels format
head -20 corpus/enron/trec_judgments/qrels.t10legallearn
head -20 corpus/enron/trec_judgments/qrel_leg_int_2010_msg_post.txt

# Check if graded or binary
sort -k4 corpus/enron/trec_judgments/qrel_leg_int_2010_msg_post.txt | cut -d' ' -f4 | sort -u
```

- [ ] **Step 3: Download one custodian zip and inspect EDRM XML format**

```bash
# Download a small custodian
python run_experiment.py --step download --custodians allen-p.zip

# Unzip and inspect
unzip -l corpus/enron/raw/allen-p.zip | head -30
unzip -o corpus/enron/raw/allen-p.zip -d /tmp/enron-inspect/
ls /tmp/enron-inspect/
# Look at the XML structure
head -100 /tmp/enron-inspect/*/*.xml 2>/dev/null || head -100 /tmp/enron-inspect/*.xml
```

- [ ] **Step 4: Document findings**

Write a brief note about:
- The actual qrels format (binary vs graded? which field names?)
- The actual XML structure (do Tags match our parser assumptions?)
- How doc IDs in qrels map to doc IDs in the XML files
- Which topics have the most highly relevant documents

- [ ] **Step 5: Adapt parser if needed**

Based on the actual data format, adjust `src/enron_parse.py` and `src/trec_loader.py`. Re-run their tests. Commit.

```bash
pytest tests/ -v
git add -A
git commit -m "fix: adapt parsers to actual EDRM XML and TREC qrels format"
```

---

### Task 12: Select topics and build conditions on real data

This task depends on Task 11 findings. The steps are:

- [ ] **Step 1: Select 3-5 TREC topics** based on inspection results. Criteria:
  - Enough highly relevant documents (aim for 5+ per topic)
  - Enough total assessed documents for hay selection
  - Topical diversity

- [ ] **Step 2: Download all custodians needed** for selected topics. Identify which custodians have key documents from the qrels, then download those custodian zips plus a selection of unrelated custodians for the dilution control.

- [ ] **Step 3: Parse all downloaded custodians**

```bash
python run_experiment.py --step parse
```

- [ ] **Step 4: Build conditions for each topic**

```bash
python run_experiment.py --step conditions --topic 201
python run_experiment.py --step conditions --topic 203
# ... repeat for selected topics
```

- [ ] **Step 5: Verify condition sizes and composition**

```python
# Quick sanity check script
import json
from pathlib import Path
for topic_dir in Path("corpus/conditions").iterdir():
    print(f"\n=== Topic {topic_dir.name} ===")
    for scale_dir in sorted(topic_dir.iterdir()):
        print(f"  {scale_dir.name}:")
        for f in sorted(scale_dir.glob("*.json")):
            data = json.load(open(f))
            print(f"    {f.stem}: {len(data['doc_ids'])} docs")
```

- [ ] **Step 6: Commit**

```bash
git add corpus/conditions/ corpus/enron/trec_topics/ corpus/enron/trec_judgments/
git commit -m "feat: select topics, build conditions on real Enron data"
```

---

### Task 13: Run embeddings across all models

- [ ] **Step 1: Set up API keys**

Create a `.env` file (not committed) with:
```
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
VOYAGE_API_KEY=...
GOOGLE_API_KEY=...
JINA_API_KEY=...
```

- [ ] **Step 2: Run embeddings for each model**

Start with OpenAI (fastest API, most likely to work):
```bash
python run_experiment.py --step embed --model openai/text-embedding-3-large
python run_experiment.py --step embed --model openai/text-embedding-3-small
```

Then commercial APIs:
```bash
python run_experiment.py --step embed --model cohere/embed-v3
python run_experiment.py --step embed --model voyage/voyage-law-2
python run_experiment.py --step embed --model google/gemini-embedding
python run_experiment.py --step embed --model jina/embeddings-v3
```

Then open-source (slower, runs locally):
```bash
python run_experiment.py --step embed --model open/contriever
python run_experiment.py --step embed --model open/bge-large-en-v1.5
python run_experiment.py --step embed --model open/e5-mistral-7b-instruct
```

- [ ] **Step 3: Verify embeddings cached**

```bash
ls -la embeddings/
# Should show 9 directories, each with .npy cache files
```

- [ ] **Step 4: Commit cache metadata (not the .npy files)**

```bash
git commit --allow-empty -m "milestone: all 9 embedding models computed and cached"
```

---

### Task 14: Run retrieval experiments and generate results

- [ ] **Step 1: Create query files for each topic**

Based on the TREC topic descriptions, create query JSON files:
```bash
# Example structure
cat > experiments/queries/201.json << 'EOF'
{
  "topic_id": "201",
  "queries": [
    {"text": "<TREC topic statement verbatim>", "category": "trec_original"},
    {"text": "<paraphrased version>", "category": "paraphrase"},
    {"text": "key documents for this topic", "category": "broad"},
    {"text": "most important documents in this production", "category": "review"}
  ]
}
EOF
```

Repeat for each topic with 20-30 queries each.

- [ ] **Step 2: Run the full evaluation**

```bash
python run_experiment.py --step evaluate
```

- [ ] **Step 3: Inspect initial results**

```bash
# Check if key docs are being displaced
python -c "
import json
results = json.load(open('results/summary.json'))
for r in results[:10]:
    print(f'{r[\"condition\"]:20s} {r[\"scale\"]:8s} {r[\"model\"]:35s} recall@5={r[\"recall_at_5\"]:.2f}')
"
```

- [ ] **Step 4: Run detection experiment**

```bash
python run_experiment.py --step detect
```

- [ ] **Step 5: Commit results**

```bash
git add results/ experiments/queries/
git commit -m "feat: full experiment results across all topics, models, scales"
```

---

### Task 15: Generate figures and tables

**Files:**
- Create: `src/visualize.py`

- [ ] **Step 1: Implement visualization**

```python
# src/visualize.py
"""Generate figures and tables for the paper."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def load_results(results_dir: Path = Path("results")) -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    records = []
    for f in results_dir.rglob("results.json"):
        with open(f) as fh:
            records.extend(json.load(fh))
    return pd.DataFrame(records)


def plot_dose_response(df: pd.DataFrame, output_dir: Path):
    """Dose-response curve: recall@5 vs scale for each condition."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_order = ["tight", "small", "medium", "large"]

    for topic_id in df["topic_id"].unique():
        fig, ax = plt.subplots(figsize=(10, 6))
        topic_df = df[df["topic_id"] == topic_id]

        for condition in ["baseline", "haystacked_a", "haystacked_b", "haystacked_c", "dilution_control"]:
            cond_df = topic_df[topic_df["condition"] == condition]
            means = cond_df.groupby("scale")["recall_at_5"].mean().reindex(scale_order)
            ax.plot(scale_order, means.values, marker="o", label=condition)

        ax.set_xlabel("Scale")
        ax.set_ylabel("Recall@5 (mean across queries and models)")
        ax.set_title(f"Dose-Response: Topic {topic_id}")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.savefig(output_dir / f"dose_response_{topic_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_model_comparison(df: pd.DataFrame, output_dir: Path):
    """Heatmap: displacement by model and condition at realistic scale."""
    output_dir.mkdir(parents=True, exist_ok=True)

    large_df = df[df["scale"] == "large"]

    for topic_id in large_df["topic_id"].unique():
        topic_df = large_df[large_df["topic_id"] == topic_id]
        pivot = topic_df.pivot_table(
            values="recall_at_5", index="model", columns="condition", aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
        ax.set_title(f"Recall@5 by Model and Condition — Topic {topic_id}")
        fig.savefig(output_dir / f"model_comparison_{topic_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_gradient(df: pd.DataFrame, output_dir: Path):
    """Bar chart: A vs B vs C vs dilution control at realistic scale."""
    output_dir.mkdir(parents=True, exist_ok=True)

    large_df = df[df["scale"] == "large"]
    means = large_df.groupby("condition")["recall_at_5"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = ["baseline", "dilution_control", "haystacked_a", "haystacked_b", "haystacked_c"]
    colors = ["#2ecc71", "#95a5a6", "#f39c12", "#e67e22", "#e74c3c"]
    values = [means.get(c, 0) for c in conditions]

    ax.bar(conditions, values, color=colors)
    ax.set_ylabel("Recall@5 (mean)")
    ax.set_title("Attacker Gradient: Recall@5 at Realistic Scale")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)
    fig.savefig(output_dir / "gradient.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = load_results()
    output = Path("results/figures")
    plot_dose_response(df, output)
    plot_model_comparison(df, output)
    plot_gradient(df, output)
    print(f"Figures saved to {output}/")
```

- [ ] **Step 2: Generate all figures**

```bash
python -m src.visualize
```

- [ ] **Step 3: Commit**

```bash
git add src/visualize.py results/figures/
git commit -m "feat: visualization — dose-response curves, model heatmaps, gradient chart"
```

---

## Post-plan notes

**Dependency chain:** Tasks 1-6 can be implemented and tested without any data. Task 7-9 likewise. Task 10 ties them together. Task 11 is the first real data interaction and may require revisiting Tasks 2-3 if the actual format differs from assumptions. Tasks 12-15 are sequential and data-dependent.

**Risk: EDRM XML format.** The parser in Task 2 is based on assumed XML structure. The actual format may differ. Task 11 is specifically designed to inspect the real data and adapt.

**Risk: TREC qrels format.** Binary vs graded relevance, doc ID format, and topic availability are unknowns until Task 11.

**Risk: Embedding API costs.** At ~1000 documents × 9 models, costs should be under $5 total. Open-source models are free but slower.

**Total estimated tasks:** 15 tasks, roughly 2-3 days of implementation time.
