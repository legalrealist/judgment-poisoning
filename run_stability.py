#!/usr/bin/env python3
"""Run the stability experiment: strategic vs. textual-criteria prompts.

Measures classification flip rate across batch contexts for KEY/ROUTINE judgment.

Usage:
    python run_stability.py --step baseline --topic 303
    python run_stability.py --step batches --topic 303
    python run_stability.py --step evaluate --topic 303
    python run_stability.py --step report --topic 303
    python run_stability.py --step all --topic 303
"""

import argparse
import json
import random
import sys
from pathlib import Path

from src.enron_corpus import load_parsed_corpus
from src.trec_loader import parse_qrels
from src.stability import (
    run_baseline_scoring,
    build_paired_batches,
    run_stability_test,
    compute_stability_report,
)

ENRON_PARSED = Path("corpus/enron/parsed")
QRELS_PATH = Path("corpus/enron/trec_judgments/qrels.txt")
TOPICS_FILE = Path("corpus/enron/trec_topics/topics.json")
RESULTS_DIR = Path("results")

DEFAULT_MODEL = "claude-sonnet-4"
DEFAULT_BATCH_SIZE = 20
DEFAULT_N_TARGETS = 30


def load_query(topic_id: str) -> str:
    with open(TOPICS_FILE) as f:
        topics = json.load(f)["topics"]
    return topics[topic_id]["statement"]


def step_baseline(topic_id: str, model: str, prompt_style: str):
    """Score all confirmed-relevant docs individually as KEY/ROUTINE."""
    print(f"Baseline scoring: topic={topic_id}, model={model}, style={prompt_style}")

    corpus = load_parsed_corpus(ENRON_PARSED)
    query = load_query(topic_id)

    with open(QRELS_PATH) as f:
        judgments_all = parse_qrels(f.read())
    judgments = judgments_all[topic_id]

    relevant_ids = sorted(
        (judgments.highly_relevant | judgments.relevant) & set(corpus.keys())
    )
    print(f"  {len(relevant_ids)} confirmed-relevant docs in corpus")

    results = run_baseline_scoring(relevant_ids, corpus, query, model, prompt_style)
    key_count = sum(1 for r in results if r.judgment == "KEY")
    routine_count = sum(1 for r in results if r.judgment == "ROUTINE")
    print(f"  Results: {key_count} KEY, {routine_count} ROUTINE")

    out_dir = RESULTS_DIR / "baseline_scoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"topic_{topic_id}.json"
    data = [{"doc_id": r.doc_id, "judgment": r.judgment, "confidence": r.confidence}
            for r in results]
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {out_file}")


def step_batches(topic_id: str, n_targets: int, batch_size: int, seed: int):
    """Build paired batch contexts for stability testing."""
    print(f"Building paired batches: topic={topic_id}, n_targets={n_targets}")

    corpus = load_parsed_corpus(ENRON_PARSED)
    query = load_query(topic_id)

    baseline_file = RESULTS_DIR / "baseline_scoring" / f"topic_{topic_id}.json"
    if not baseline_file.exists():
        print(f"  Run --step baseline first for topic {topic_id}")
        sys.exit(1)

    with open(baseline_file) as f:
        baseline = json.load(f)
    key_doc_ids = [r["doc_id"] for r in baseline if r["judgment"] == "KEY"]

    with open(QRELS_PATH) as f:
        judgments_all = parse_qrels(f.read())
    judgments = judgments_all[topic_id]
    relevant_ids = (judgments.highly_relevant | judgments.relevant) & set(corpus.keys())
    non_relevant_ids = judgments.non_relevant & set(corpus.keys())

    rng = random.Random(seed)
    targets = rng.sample(key_doc_ids, min(n_targets, len(key_doc_ids)))
    print(f"  Selected {len(targets)} KEY docs as targets")
    print(f"  Relevant pool: {len(relevant_ids)}, Non-relevant pool: {len(non_relevant_ids)}")

    batches = build_paired_batches(
        targets, corpus, relevant_ids, non_relevant_ids,
        topic_id, query, batch_size, seed,
    )
    print(f"  Built {len(batches)} batches ({len(batches) // 3} targets x 3 conditions)")

    out_dir = RESULTS_DIR / "judgment_poisoning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"paired_batches_{topic_id}.json"
    with open(out_file, "w") as f:
        json.dump(batches, f, indent=2)
    print(f"  Saved to {out_file}")


def step_evaluate(topic_id: str, model: str, prompt_style: str):
    """Run batch evaluation and measure stability."""
    print(f"Evaluating stability: topic={topic_id}, model={model}, style={prompt_style}")

    batches_file = RESULTS_DIR / "judgment_poisoning" / f"paired_batches_{topic_id}.json"
    if not batches_file.exists():
        batches_file = RESULTS_DIR / "judgment_poisoning" / "paired_batches.json"
    if not batches_file.exists():
        print(f"  Run --step batches first for topic {topic_id}")
        sys.exit(1)

    with open(batches_file) as f:
        all_batches = json.load(f)
    batches = [b for b in all_batches if b["tid"] == topic_id]
    print(f"  Loaded {len(batches)} batches for topic {topic_id}")

    baseline_file = RESULTS_DIR / "baseline_scoring" / f"topic_{topic_id}.json"
    with open(baseline_file) as f:
        baseline = json.load(f)
    baseline_judgments = {r["doc_id"]: r["judgment"] for r in baseline}

    query = load_query(topic_id)
    results = run_stability_test(batches, baseline_judgments, query, model, prompt_style)
    report = compute_stability_report(results, prompt_style, topic_id)

    print(f"\n  === Stability Report ({prompt_style}) ===")
    print(f"  Targets tested: {report.n_targets}")
    print(f"  Flipped: {report.n_flipped} ({report.flip_rate:.1%})")
    for cond, count in sorted(report.per_condition_flip.items()):
        print(f"    {cond}: {count} flips")

    out_dir = RESULTS_DIR / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"report_{topic_id}_{prompt_style}.json"
    data = {
        "prompt_style": report.prompt_style,
        "topic_id": report.topic_id,
        "model": model,
        "n_targets": report.n_targets,
        "n_flipped": report.n_flipped,
        "flip_rate": report.flip_rate,
        "per_condition_flip": report.per_condition_flip,
        "results": [
            {
                "target_doc": r.target_doc,
                "baseline_judgment": r.baseline_judgment,
                "judgments_by_condition": r.judgments_by_condition,
                "flipped": r.flipped,
            }
            for r in report.results
        ],
    }
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {out_file}")


def step_report(topic_id: str):
    """Print comparison of both prompt styles."""
    print(f"\n=== Stability Comparison: Topic {topic_id} ===\n")

    stability_dir = RESULTS_DIR / "stability"
    for style in ["strategic", "textual_criteria"]:
        report_file = stability_dir / f"report_{topic_id}_{style}.json"
        if not report_file.exists():
            print(f"  {style}: not yet evaluated (run --step evaluate --prompt-style {style})")
            continue

        with open(report_file) as f:
            report = json.load(f)
        print(f"  {style}:")
        print(f"    Flip rate: {report['flip_rate']:.1%} ({report['n_flipped']}/{report['n_targets']})")
        for cond, count in sorted(report.get("per_condition_flip", {}).items()):
            print(f"      {cond}: {count} flips")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run stability experiment")
    parser.add_argument("--step", required=True,
                        choices=["baseline", "batches", "evaluate", "report", "all"])
    parser.add_argument("--topic", required=True, help="TREC topic ID")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model")
    parser.add_argument("--prompt-style", default="textual_criteria",
                        choices=["strategic", "textual_criteria", "both"])
    parser.add_argument("--n-targets", type=int, default=DEFAULT_N_TARGETS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.step in ("baseline", "all"):
        step_baseline(args.topic, args.model, "textual_criteria")

    if args.step in ("batches", "all"):
        step_batches(args.topic, args.n_targets, args.batch_size, args.seed)

    if args.step in ("evaluate", "all"):
        styles = ["strategic", "textual_criteria"] if args.prompt_style == "both" else [args.prompt_style]
        for style in styles:
            step_evaluate(args.topic, args.model, style)

    if args.step in ("report", "all"):
        step_report(args.topic)


if __name__ == "__main__":
    main()
