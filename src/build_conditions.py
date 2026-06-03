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
    key_doc_ids = judgments.key_documents()
    key_custodians = set()
    for doc_id in key_doc_ids:
        if doc_id in corpus and corpus[doc_id].custodian:
            key_custodians.add(corpus[doc_id].custodian)
    baseline_set = set(baseline.doc_ids)
    candidates = sorted(
        doc_id for doc_id, email in corpus.items()
        if email.custodian in key_custodians and doc_id not in baseline_set
    )
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
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    key_doc_ids = sorted(judgments.key_documents() & set(corpus.keys()))
    baseline_set = set(baseline.doc_ids)
    key_vecs = np.array([embeddings[d] for d in key_doc_ids])
    candidates = [d for d in corpus if d not in baseline_set and d in embeddings]
    if not candidates:
        return ConditionSet(
            name="haystacked_c",
            topic_id=judgments.topic_id,
            doc_ids=baseline.doc_ids,
            metadata={"key_doc_ids": baseline.metadata["key_doc_ids"], "hay_doc_ids": []},
        )
    cand_vecs = np.array([embeddings[d] for d in candidates])
    sims = cosine_similarity(cand_vecs, key_vecs)
    max_sims = sims.max(axis=1)
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
    involved_custodians = set()
    for doc_id in baseline.doc_ids:
        if doc_id in corpus and corpus[doc_id].custodian:
            involved_custodians.add(corpus[doc_id].custodian)
    baseline_set = set(baseline.doc_ids)
    candidates = sorted(
        doc_id for doc_id, email in corpus.items()
        if doc_id not in baseline_set and email.custodian not in involved_custodians
    )
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
