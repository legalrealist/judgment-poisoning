#!/usr/bin/env python3
"""Run LLM judgment poisoning evaluation.

Usage:
    python run_llm_evaluation.py                          # default: topic 303, claude-sonnet-4, medium
    python run_llm_evaluation.py --model gpt-4o           # single model
    python run_llm_evaluation.py --expand                 # full matrix
    python run_llm_evaluation.py --mode ablation          # ablation only
    python run_llm_evaluation.py --mode retrieval-audit   # no LLM calls
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.enron_corpus import load_parsed_corpus
from src.embed import get_embedder
from src.llm_experiment import run_retrieval_audit, run_end_to_end, run_ablation

CONDITIONS_DIR = Path("corpus/conditions")
ENRON_PARSED = Path("corpus/enron/parsed")
RESULTS_DIR = Path("results/llm")
TOPICS_FILE = Path("corpus/enron/trec_topics/topics.json")

DEFAULT_TOPIC = "303"
DEFAULT_SCALE = "medium"
DEFAULT_MODEL = "claude-sonnet-4"
DEFAULT_RETRIEVAL_MODEL = "open/bge-large-en-v1.5"
CONDITIONS = ["baseline", "haystacked_a", "haystacked_b", "dilution_control"]
ABLATION_RATIOS = [0, 1, 3, 5]


def load_query(topic_id: str) -> str:
    with open(TOPICS_FILE) as f:
        topics = json.load(f)["topics"]
    return topics[topic_id]["statement"]


def load_condition(topic_id: str, scale: str, condition_name: str) -> dict:
    path = CONDITIONS_DIR / topic_id / scale / f"{condition_name}.json"
    with open(path) as f:
        return json.load(f)


def run_retrieval_audit_mode(
    topic_id: str, scale: str, retrieval_model: str, corpus: dict
) -> list[dict]:
    print(f"\n=== Retrieval Audit: topic={topic_id}, scale={scale} ===")
    embedder = get_embedder(retrieval_model)
    query_text = load_query(topic_id)

    results = []
    for cond_name in CONDITIONS:
        cond = load_condition(topic_id, scale, cond_name)
        doc_ids = [d for d in cond["doc_ids"] if d in corpus]
        texts = [corpus[d].to_text() for d in doc_ids]
        embeddings = embedder.embed_documents(doc_ids, texts)
        query_emb = embedder.embed_texts([query_text])[0]

        key_doc_ids = set(cond["metadata"].get("key_doc_ids", []))
        hay_doc_ids = set(cond["metadata"].get("hay_doc_ids", []))

        audit = run_retrieval_audit(doc_ids, embeddings, query_emb, key_doc_ids, hay_doc_ids)
        row = {
            "mode": "retrieval_audit",
            "topic_id": topic_id,
            "scale": scale,
            "condition": cond_name,
            "retrieval_model": retrieval_model,
            "k": audit.k,
            "n_docs": len(doc_ids),
            "n_key_in_top_k": audit.n_key_in_top_k,
            "n_hay_in_top_k": audit.n_hay_in_top_k,
            "n_other_in_top_k": audit.n_other_in_top_k,
            "slot_infiltration": audit.slot_infiltration,
        }
        results.append(row)
        print(f"  {cond_name:20s} key={audit.n_key_in_top_k} hay={audit.n_hay_in_top_k} "
              f"other={audit.n_other_in_top_k} infiltration={audit.slot_infiltration:.2f}")
    return results


def run_end_to_end_mode(
    topic_id: str, scale: str, llm_model: str, retrieval_model: str, corpus: dict
) -> list[dict]:
    print(f"\n=== End-to-End: topic={topic_id}, scale={scale}, llm={llm_model} ===")
    embedder = get_embedder(retrieval_model)
    query_text = load_query(topic_id)

    results = []
    for cond_name in CONDITIONS:
        cond = load_condition(topic_id, scale, cond_name)
        doc_ids = [d for d in cond["doc_ids"] if d in corpus]
        doc_texts = {d: corpus[d].to_text() for d in doc_ids}
        texts_list = [doc_texts[d] for d in doc_ids]
        embeddings = embedder.embed_documents(doc_ids, texts_list)
        query_emb = embedder.embed_texts([query_text])[0]

        key_doc_ids = set(cond["metadata"].get("key_doc_ids", []))
        hay_doc_ids = set(cond["metadata"].get("hay_doc_ids", []))

        e2e = run_end_to_end(
            doc_ids, doc_texts, embeddings, query_emb,
            key_doc_ids, hay_doc_ids, query_text, llm_model,
        )
        for paradigm_name, paradigm in [("individual", e2e.individual), ("batch", e2e.batch)]:
            row = {
                "mode": "end_to_end",
                "paradigm": paradigm_name,
                "topic_id": topic_id,
                "scale": scale,
                "condition": cond_name,
                "llm_model": llm_model,
                "retrieval_model": retrieval_model,
                "n_docs_judged": e2e.n_docs_judged,
                "slot_infiltration": e2e.retrieval_audit.slot_infiltration,
                "precision": paradigm.precision,
                "recall": paradigm.recall,
                "f1": paradigm.f1,
                "false_negative_rate": paradigm.false_negative_rate,
                "false_positive_rate": paradigm.false_positive_rate,
                "mean_confidence_on_key": paradigm.mean_confidence_on_key,
            }
            results.append(row)
        print(f"  {cond_name:20s} "
              f"ind_FNR={e2e.individual.false_negative_rate:.3f} "
              f"bat_FNR={e2e.batch.false_negative_rate:.3f} "
              f"infiltration={e2e.retrieval_audit.slot_infiltration:.2f}")
    return results


def run_ablation_mode(
    topic_id: str, scale: str, llm_model: str, corpus: dict
) -> list[dict]:
    print(f"\n=== Ablation: topic={topic_id}, scale={scale}, llm={llm_model} ===")
    query_text = load_query(topic_id)

    results = []
    for cond_name in CONDITIONS:
        if cond_name == "baseline":
            continue
        cond = load_condition(topic_id, scale, cond_name)
        key_ids = [d for d in cond["metadata"].get("key_doc_ids", []) if d in corpus]
        hay_ids = [d for d in cond["metadata"].get("hay_doc_ids", []) if d in corpus]
        key_texts = [corpus[d].to_text() for d in key_ids]
        hay_texts = [corpus[d].to_text() for d in hay_ids]

        ablation = run_ablation(key_ids, key_texts, hay_ids, hay_texts, query_text, llm_model)
        for rr in ablation.ratio_results:
            row = {
                "mode": "ablation",
                "topic_id": topic_id,
                "scale": scale,
                "condition": cond_name,
                "llm_model": llm_model,
                "ratio": rr["ratio"],
                "n_key_in_window": rr["n_key_in_window"],
                "n_hay_in_window": rr["n_hay_in_window"],
                "precision": rr["precision"],
                "recall": rr["recall"],
                "f1": rr["f1"],
                "false_negative_rate": rr["false_negative_rate"],
                "false_positive_rate": rr["false_positive_rate"],
                "mean_confidence_on_key": rr["mean_confidence_on_key"],
            }
            results.append(row)
            print(f"  {cond_name:20s} ratio={rr['ratio']} "
                  f"FNR={rr['false_negative_rate']:.3f} "
                  f"recall={rr['recall']:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run LLM judgment poisoning evaluation")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Topic ID (default: 303)")
    parser.add_argument("--scale", default=DEFAULT_SCALE, help="Scale (default: medium)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model (default: claude-sonnet-4)")
    parser.add_argument("--retrieval-model", default=DEFAULT_RETRIEVAL_MODEL)
    parser.add_argument("--mode", choices=["all", "retrieval-audit", "end-to-end", "ablation"],
                        default="all", help="Which mode(s) to run")
    parser.add_argument("--expand", action="store_true", help="Run full matrix")
    args = parser.parse_args()

    topics = ["303", "301"] if args.expand else [args.topic]
    models = ["claude-sonnet-4", "gpt-4o"] if args.expand else [args.model]
    scales = ["small", "medium", "large"] if args.expand else [args.scale]

    print("Loading corpus...")
    corpus = load_parsed_corpus(ENRON_PARSED)
    print(f"Loaded {len(corpus)} emails")

    all_results = []

    for topic in topics:
        for scale in scales:
            if args.mode in ("all", "retrieval-audit"):
                all_results.extend(
                    run_retrieval_audit_mode(topic, scale, args.retrieval_model, corpus)
                )
            for model in models:
                if args.mode in ("all", "end-to-end"):
                    all_results.extend(
                        run_end_to_end_mode(topic, scale, model, args.retrieval_model, corpus)
                    )
                if args.mode in ("all", "ablation"):
                    all_results.extend(
                        run_ablation_mode(topic, scale, model, corpus)
                    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "llm_results.json"

    # Merge with existing results rather than overwriting
    existing = []
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)

    def _key(r):
        return (r.get("mode"), r.get("topic_id"), r.get("scale"),
                r.get("condition"), r.get("llm_model", ""),
                r.get("paradigm", ""), r.get("ratio", ""))

    seen = {_key(r) for r in all_results}
    merged = all_results + [r for r in existing if _key(r) not in seen]

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nResults saved to {output_file} ({len(merged)} rows, {len(all_results)} new)")


if __name__ == "__main__":
    main()
