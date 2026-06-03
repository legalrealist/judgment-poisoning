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

SCALES = {
    "tight": 0,
    "small": 2,
    "medium": 5,
    "large": 10,
}


def step_download(custodians: list[str]):
    from src.enron_download import download_custodian
    for name in custodians:
        download_custodian(name)


def step_parse():
    print("TODO: Parse EDRM XML files from corpus/enron/raw/ into corpus/enron/parsed/")
    print("Each custodian becomes a JSON file with list of EnronEmail dicts.")


def step_conditions(topic_id: str):
    print(f"Building conditions for topic {topic_id}...")
    parsed_dir = ENRON_PARSED
    corpus = {}
    for f in parsed_dir.glob("*.json"):
        with open(f) as fh:
            emails = json.load(fh)
            for e in emails:
                corpus[e["doc_id"]] = EnronEmail(**e)

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
            out_dir = CONDITIONS_DIR / topic_id / scale_name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "baseline.json", "w") as f:
                json.dump({"doc_ids": baseline.doc_ids, "metadata": baseline.metadata}, f, indent=2)
            continue

        hay_count = multiplier * len(baseline)
        print(f"  Scale {scale_name}: {hay_count} hay docs")

        cond_a = build_haystacked_a(corpus, judgments, baseline, hay_count)
        cond_b = build_haystacked_b(corpus, judgments, baseline, hay_count)

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
    print(f"Embedding with {model_name}...")
    embedder = get_embedder(model_name)

    all_doc_ids = set()
    for cond_file in CONDITIONS_DIR.rglob("*.json"):
        with open(cond_file) as f:
            data = json.load(f)
            all_doc_ids.update(data["doc_ids"])

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
    print("Running full evaluation...")
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
