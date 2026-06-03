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
