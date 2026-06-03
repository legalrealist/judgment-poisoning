#!/usr/bin/env python3
"""Run the full haystacking evaluation: embed, rank, measure displacement.

Usage:
    python run_evaluation.py --model open/contriever --topic 303
    python run_evaluation.py --all-models --all-topics
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.enron_corpus import load_parsed_corpus
from src.trec_loader import parse_qrels, TopicJudgments
from src.embed import get_embedder, SUPPORTED_MODELS
from src.experiment import run_single_experiment, ExperimentResult
from src.metrics import displacement

CONDITIONS_DIR = Path("corpus/conditions")
ENRON_PARSED = Path("corpus/enron/parsed")
RESULTS_DIR = Path("results")
TOPICS = ["303", "301"]
SCALES = ["small", "medium", "large"]
OPEN_SOURCE_MODELS = [
    "open/contriever",
    "open/bge-large-en-v1.5",
    "open/e5-mistral-7b-instruct",
]


def load_topic_queries(topic_id: str) -> dict[str, str]:
    """Return query strings for a topic based on TREC topic statements."""
    topics_file = Path("corpus/enron/trec_topics/topics.json")
    with open(topics_file) as f:
        topics = json.load(f)["topics"]

    statement = topics[topic_id]["statement"]
    return {
        "trec_statement": statement,
    }


def run_topic_model(topic_id: str, model_name: str, corpus: dict) -> list[dict]:
    """Run all conditions/scales for one topic and one model."""
    print(f"\n--- Topic {topic_id}, Model {model_name} ---")

    embedder = get_embedder(model_name)
    queries = load_topic_queries(topic_id)

    # Collect all doc IDs across all conditions and scales
    all_doc_ids = set()
    for scale in SCALES:
        scale_dir = CONDITIONS_DIR / topic_id / scale
        if not scale_dir.exists():
            continue
        for cond_file in scale_dir.glob("*.json"):
            with open(cond_file) as f:
                data = json.load(f)
                all_doc_ids.update(data["doc_ids"])

    # Also add baseline from tight
    tight_baseline = CONDITIONS_DIR / topic_id / "tight" / "baseline.json"
    if tight_baseline.exists():
        with open(tight_baseline) as f:
            data = json.load(f)
            all_doc_ids.update(data["doc_ids"])

    # Filter to docs in corpus
    doc_ids = sorted(all_doc_ids & set(corpus.keys()))
    texts = [corpus[d].to_text() for d in doc_ids]

    print(f"  Embedding {len(doc_ids)} documents...")
    all_embeddings = embedder.embed_documents(doc_ids, texts)
    emb_lookup = dict(zip(doc_ids, all_embeddings))

    # Embed queries
    print(f"  Embedding {len(queries)} queries...")
    query_texts = list(queries.values())
    query_names = list(queries.keys())
    query_embeddings = embedder.embed_texts(query_texts)

    results = []

    for scale in SCALES:
        scale_dir = CONDITIONS_DIR / topic_id / scale
        if not scale_dir.exists():
            continue

        # Load baseline for displacement calculation
        baseline_file = scale_dir / "baseline.json"
        if not baseline_file.exists():
            continue

        with open(baseline_file) as f:
            baseline_data = json.load(f)
        key_doc_ids = set(baseline_data["metadata"].get("key_doc_ids", []))

        # Run baseline first to get baseline ranking
        b_ids = [d for d in baseline_data["doc_ids"] if d in emb_lookup]
        b_embs = np.array([emb_lookup[d] for d in b_ids])

        for cond_file in sorted(scale_dir.glob("*.json")):
            cond_name = cond_file.stem
            with open(cond_file) as f:
                cond_data = json.load(f)

            c_ids = [d for d in cond_data["doc_ids"] if d in emb_lookup]
            c_embs = np.array([emb_lookup[d] for d in c_ids])

            for qi, (qname, qtext) in enumerate(queries.items()):
                q_emb = query_embeddings[qi]

                result = run_single_experiment(
                    doc_ids=c_ids,
                    doc_embeddings=c_embs,
                    query_embedding=q_emb,
                    key_doc_ids=key_doc_ids,
                    condition=cond_name,
                    model=model_name,
                    topic_id=topic_id,
                    query=qname,
                    scale=scale,
                )

                # Compute displacement vs baseline
                baseline_result = run_single_experiment(
                    doc_ids=b_ids,
                    doc_embeddings=b_embs,
                    query_embedding=q_emb,
                    key_doc_ids=key_doc_ids,
                )

                disp = displacement(
                    baseline_result.ranked_doc_ids,
                    result.ranked_doc_ids,
                    key_doc_ids,
                )

                row = {
                    "topic_id": topic_id,
                    "model": model_name,
                    "condition": cond_name,
                    "scale": scale,
                    "query": qname,
                    "recall_at_5": result.recall_at_5,
                    "recall_at_10": result.recall_at_10,
                    "recall_at_20": result.recall_at_20,
                    "mrr": result.mrr,
                    "displacement": disp,
                    "n_docs": len(c_ids),
                    "n_key_docs": len(key_doc_ids),
                }
                results.append(row)

                if cond_name != "baseline":
                    print(
                        f"  {scale:6s} {cond_name:20s} "
                        f"R@5={result.recall_at_5:.3f} "
                        f"R@10={result.recall_at_10:.3f} "
                        f"MRR={result.mrr:.3f} "
                        f"disp={disp:+.1f}"
                    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run haystacking evaluation")
    parser.add_argument("--model", help="Single model to run")
    parser.add_argument("--topic", help="Single topic to run")
    parser.add_argument("--all-models", action="store_true", help="Run all open-source models")
    parser.add_argument("--all-topics", action="store_true", help="Run all topics")
    args = parser.parse_args()

    models = OPEN_SOURCE_MODELS if args.all_models else [args.model or OPEN_SOURCE_MODELS[0]]
    topics = TOPICS if args.all_topics else [args.topic or TOPICS[0]]

    print("Loading corpus...")
    corpus = load_parsed_corpus(ENRON_PARSED)
    print(f"Loaded {len(corpus)} emails")

    all_results = []
    for model in models:
        for topic in topics:
            results = run_topic_model(topic, model, corpus)
            all_results.extend(results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file} ({len(all_results)} rows)")

    # Print summary table
    print("\n=== SUMMARY ===")
    import pandas as pd
    df = pd.DataFrame(all_results)
    summary = df.groupby(["topic_id", "model", "condition", "scale"]).agg(
        recall_at_5=("recall_at_5", "mean"),
        displacement=("displacement", "mean"),
    ).round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
