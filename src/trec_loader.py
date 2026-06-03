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


def _classify(tj: TopicJudgments, doc_id: str, relevance: str, graded: bool) -> None:
    """Classify a document into the appropriate relevance bucket."""
    # Try numeric first
    try:
        score = int(relevance)
    except ValueError:
        score = None

    if score is not None:
        # Skip not-assessed documents
        if score < 0:
            return
        if graded:
            if score >= 2:
                tj.highly_relevant.add(doc_id)
            elif score == 1:
                tj.relevant.add(doc_id)
            else:
                tj.non_relevant.add(doc_id)
        else:
            # Binary numeric: 1=relevant, 0=non-relevant
            if score >= 1:
                tj.relevant.add(doc_id)
            else:
                tj.non_relevant.add(doc_id)
    else:
        # Categorical labels (older TREC years)
        if relevance == "R":
            tj.relevant.add(doc_id)
        elif relevance == "N":
            tj.non_relevant.add(doc_id)
        elif relevance == "B":
            tj.broken.add(doc_id)


def parse_qrels(
    content: str, graded: bool = False
) -> dict[str, TopicJudgments]:
    """Parse a TREC qrels file into TopicJudgments per topic.

    Supports three formats:
      - Standard/interactive: ``topic_id iter doc_id relevance [probability]``
        (4-5 space-separated fields)
      - Learning: ``topic:doc_id cost relevance`` (3 fields, colon in first)
      - Categorical: same as standard but relevance is R/N/B

    Args:
        content: Raw qrels file content.
        graded: If True, treat numeric relevance as graded
                (0=non-relevant, 1=relevant, 2=highly relevant).
                If False, 0=non-relevant, 1=relevant.

    Returns:
        Dict mapping topic_id -> TopicJudgments.
    """
    judgments: dict[str, TopicJudgments] = {}

    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Detect learning format: first field contains a colon (topic:doc_id)
        if ":" in parts[0]:
            topic_id, doc_id = parts[0].split(":", 1)
            if len(parts) < 3:
                continue
            relevance = parts[2]
        else:
            if len(parts) < 4:
                continue
            topic_id = parts[0]
            doc_id = parts[2]
            relevance = parts[3]

        if topic_id not in judgments:
            judgments[topic_id] = TopicJudgments(topic_id=topic_id)

        _classify(judgments[topic_id], doc_id, relevance, graded)

    return judgments
