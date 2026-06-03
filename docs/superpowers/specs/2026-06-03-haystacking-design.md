# Haystacking: Adversarial Retrieval Through Strategic Over-Production in Legal Discovery

## Design Spec

### Overview

Haystacking is a security research project testing whether strategic over-production of authentic documents in litigation discovery can bury key evidence in the retrieval layer of AI-assisted document review. The attack mechanism exploits dense retrieval systems: by flooding the corpus with topically related documents, an adversary dilutes the semantic signal of damaging evidence.

This is not prompt injection, parser differential, document fabrication, or withholding. Every document in the production is authentic — real business records from the producing party's archives, not adversarially crafted or optimized text. The attack controls which documents surface, not what they say.

The experiment uses the Enron email corpus with TREC Legal Track relevance judgments — real corporate emails with independent expert assessments of relevance.

### Hypothesis

A producing party can cause AI-assisted review tools to omit key documents from retrieval results by over-producing authentic documents that share topic, vocabulary, and entities with the damaging evidence.

### Relationship to prior work

- **PoisonedRAG (USENIX Security 2025):** Demonstrated 90-99% attack success rate on RAG systems by injecting synthetic adversarial texts. Key difference: PoisonedRAG produces unnatural texts with high repetition (0.28 rate) that are potentially detectable via perplexity filters. Haystacking uses exclusively authentic documents — no adversarial optimization, no synthetic content. Whether this makes the attack harder to detect is an empirical question this project tests (see Defenses).
- **DIGA:** Showed dense retrievers have exploitable properties (insensitivity to token order, bias toward influential tokens). Haystacking exploits a different property: dilution through volume.
- **Lying Spreadsheets (Zhu, 2026):** Parser differential attack on XLSX files. Proved category 3 (parser differentials) from the AI attack surface taxonomy. Haystacking proves category 2 (RAG poisoning) via a novel mechanism.
- **TREC Legal Track (2006-2011):** Established the Enron email corpus as the standard benchmark for legal retrieval research, with expert relevance judgments across multiple discovery topics. This project uses the TREC 2009-2011 topics and judgments as ground truth.
- **Attack surface taxonomy:** "When Documents Are the Attack Surface" — https://legalrealist.ai/posts/36-when-ai-is-the-attack-surface/

### Contribution

PoisonedRAG proved corpus poisoning works with synthetic adversarial text that no real attacker could plausibly inject. This project demonstrates the same class of attack using documents that already exist in the adversary's archives, selected through methods litigation teams already use, in a workflow where billions of dollars in legal outcomes depend on which documents surface first.

The contribution is not the retrieval property (dilution reduces precision — known). The contribution is demonstrated exploitability in a real-world high-stakes workflow using realistic attack methods and real documents.

---

## Data source

### Enron email corpus

**EDRM Enron Email Data Set v2** — 1.7 million emails from ~150 Enron employees, released by FERC, curated by EDRM and ZL Technologies. Available on Internet Archive (73GB XML). Every document is an authentic corporate email from a real company involved in real litigation.

### TREC Legal Track relevance judgments

**TREC Legal Track 2009-2011** used the Enron corpus with discovery request topics modeled on real litigation. Professional reviewers assessed documents as responsive or non-responsive for each topic. Where graded relevance is available (highly relevant vs. marginally relevant), the grades are used to identify key documents.

This project uses the TREC assessors' judgments as ground truth. The experimenter does not decide which documents are "smoking guns" — the TREC expert reviewers made those determinations independently.

---

## Experimental design

### Attacker model

The experiment tests three attacker levels representing a gradient from normal litigation behavior to adversarial engineering:

| Attacker | Method | Sophistication | Ethical status |
|---|---|---|---|
| **A: Broad collection** | Produce everything from relevant custodians matching broad search terms | None — any litigation team does this | Normal, happens every case |
| **B: Keyword-aware** | Identify vocabulary in key documents, use broader search terms to pull more documents sharing those keywords | Low — any paralegal can do this | Gray area, arguably standard strategy |
| **C: Embedding optimizer** | Score the collected pool by embedding similarity to key documents, select top-K most similar non-responsive documents | High — requires understanding of retrieval systems | Crosses the line into adversarial engineering |

The gradient is the finding:
- If A already causes measurable displacement, AI-assisted review is fragile to normal litigation behavior (systemic risk)
- If B causes significantly more than A, standard litigation tactics can exploit the vulnerability without technical sophistication (strategic risk)
- If C adds only marginal improvement over B, keyword awareness is sufficient — no embedding knowledge needed (accessibility of the attack)
- If the jump from B to C is detectable, that's the defense story

### Topic selection

Select 3-5 TREC Legal Track topics that have:
- Graded relevance judgments where available (highly relevant vs. marginally relevant)
- Enough highly relevant documents to serve as "key documents" (analogous to smoking guns)
- Enough assessed documents in the broader corpus to support hay selection at scale
- Sufficient topical diversity across selected topics

### Conditions per topic

For each TREC topic, build five conditions:

**Baseline:** Highly relevant documents (per TREC judgments) plus a sample of responsive documents. Represents a tight, well-curated production.

**Haystacked A (broad collection):** Baseline + additional Enron emails from the same custodians whose mailboxes contain the highly relevant documents. No filtering by topic — just "produce everything from these custodians." This is how overinclusive collection works in practice.

**Haystacked B (keyword-aware):** Baseline + Enron emails selected by keyword overlap with the highly relevant documents. Extract key terms from the highly relevant documents, search the broader Enron corpus for emails containing those terms, include regardless of TREC responsiveness judgment. A paralegal could do this today.

**Haystacked C (embedding-optimized):** Baseline + Enron emails selected by embedding similarity to the highly relevant documents. For each highly relevant document, find the K most similar emails in the broader corpus (excluding already-responsive documents). This is the technically sophisticated attack.

**Dilution control:** Baseline + Enron emails from completely unrelated custodians and topics. At each scale level, size-matched to that level's haystacked C condition. Isolates the effect of topical similarity from pure corpus size increase.

### Scale sweep

At each attacker level, test at multiple hay volumes to produce a dose-response curve:

| Scale | Baseline docs | Hay docs added | Approximate total |
|---|---|---|---|
| Tight | N | 0 | N |
| Small | N | 2× baseline | 3N |
| Medium | N | 5× baseline | 6N |
| Large | N | 10× baseline | 11N |

(Exact counts depend on the TREC topics selected and available documents. N = baseline size per topic.)

At each scale level, five conditions are tested:
- Baseline (no hay)
- Haystacked A (broad collection)
- Haystacked B (keyword-aware)
- Haystacked C (embedding-optimized)
- Dilution control (off-topic, size-matched to C at each scale level)

The **primary comparison** is each haystacked condition vs. the size-matched dilution control at the same scale level. This isolates topical similarity from corpus size. If the dilution control causes the same recall drop, the finding is trivial ("bigger corpus = lower recall"). If haystacked conditions displace significantly more, topical similarity is the mechanism.

The **secondary comparison** is across attacker levels (A vs. B vs. C) to show the gradient.

### Queries

**20-30 queries per topic.** Derived from the TREC topic descriptions and expanded with varied phrasing:

*From TREC topic descriptions:*
- Direct use of the TREC topic statement as a query
- Paraphrased versions of the topic statement
- Sub-queries targeting specific aspects of the topic

*Expanded queries:*
- Broad retrieval queries ("key documents for this topic")
- Targeted fact queries ("evidence of [specific claim]")
- Review-style queries ("most important documents in this production")

Final query set should include at least 5 queries written by a practicing lawyer if possible, to reduce experimenter bias in query design.

---

## Embedding models (9 total)

**Commercial APIs:**

| Model | Rationale |
|---|---|
| OpenAI text-embedding-3-large | Academic baseline (used in LegalBench-RAG), likely closest to production eDiscovery platforms |
| OpenAI text-embedding-3-small | Tests whether model size affects vulnerability |
| Cohere embed-v3 | Major commercial alternative for enterprise search |
| Voyage voyage-law-2 | Legal-domain specific, tests whether specialization provides resistance |
| Google Gemini Embedding | #1 on MTEB, #7 on legal benchmarks — interesting divergence |
| Jina embeddings-v3 | Popular commercial embedding API |

**Open-source (run locally):**

| Model | Rationale |
|---|---|
| Contriever | Used in PoisonedRAG, enables direct comparison to prior work |
| BGE-large-en-v1.5 | Most popular open-source embedding model |
| E5-mistral-7b-instruct | Strong open-source performer (Microsoft) |

**Note on eDiscovery platforms:** Research found that no major platform (Relativity, Everlaw, Reveal, Disco, Logikcull) publicly discloses its embedding model. Relativity aiR confirms Azure OpenAI for generative features but not for retrieval. Amazon holds a patent (US12430344B1) that explicitly treats embedding model secrecy as a security feature. Testing across 9 diverse models (commercial + open-source, general + legal-specific, large + small) provides the best available proxy for production systems.

---

## Metrics

| Metric | What it measures |
|---|---|
| **Recall@5** (primary) | How many key documents in top 5 results — maps to "AI picks the 5 most important docs" workflow |
| **Recall@10** | Same for top 10 |
| **Recall@20** | Same for top 20 |
| **MRR** | Average reciprocal rank of first key document — how deep must you scroll |
| **Displacement** | Average rank change of key documents between baseline and haystacked conditions |

**Statistical analysis:** Use bootstrap confidence intervals (10,000 resamples over queries) for each metric at each scale level. Report effect sizes (Cohen's d) for the primary comparison (haystacked vs. dilution control). Use paired Wilcoxon signed-rank tests across queries within each model. Report results both per-model and aggregated across models. Correct for multiple comparisons (Bonferroni across 9 models). Pre-specify alpha = 0.05.

**No optimization loop.** Hay is selected once per attacker level, then frozen. The experiment measures what happens with a realistic one-shot attack, not the theoretical optimum.

---

## Defenses to test

### Detection experiment

Can a defender distinguish normal over-production (A/B) from adversarial engineering (C)?

- **Topical density analysis:** Does condition C show a detectably unusual concentration of documents around specific topics compared to A or B? If C clusters more tightly around key document topics, that's a detection signal.
- **Embedding distribution analysis:** Do the hay documents in C have a detectably different similarity distribution to the key documents than those in A or B? (e.g., suspiciously uniform high similarity)
- **Custodian distribution:** Does C draw from an unusual spread of custodians compared to A (which is custodian-bounded)?
- **Perplexity / naturalness:** Baseline check — all conditions use real documents, so perplexity filters (effective against PoisonedRAG) should show no signal. Confirming this empirically is part of the contribution.

### Mitigation experiment

Can a defender recover key documents even in a haystacked corpus?

- **Redundancy detection:** Does identifying topically similar clusters help surface the signal? (If 50 documents all discuss "client handoffs," the one that discusses "forwarding the client list to a personal email" stands out within the cluster.)
- **Anomaly prompting:** Asking the model to identify the most *unique* or *anomalous* documents instead of the most *important*
- **Consistency checking:** Do contradictions between documents raise flags?
- **Provenance-aware retrieval:** Does weighting by custodian, date, or document type improve key document retrieval in haystacked conditions?

---

## Scope: retrieval layer only

This experiment tests the retrieval layer in isolation — cosine similarity ranking on dense embeddings. Production eDiscovery systems typically have additional layers that could mitigate or defeat a retrieval-layer attack:

- **Cross-encoder reranking** — rerankers that score (query, document) pairs directly are better at distinguishing "discusses client lists routinely" from "discusses stealing client lists" and could rescue smoking guns from lower ranks
- **Summarization / LLM layer** — if an LLM reviews the top-K retrieved documents, it may recognize that the retrieved set is mostly noise and request a wider search, or identify the smoking guns within a mixed set
- **Full-context analysis** — for small enough corpora, an LLM could bypass retrieval entirely and analyze all documents directly

If the attack works at the retrieval layer but gets defeated by reranking or summarization, that's still a finding — it identifies which component of the pipeline is vulnerable. But it means the headline claim is "dense retrieval is vulnerable to haystacking," not "AI-assisted review is vulnerable to haystacking."

The results should be framed accordingly: this is a retrieval-layer vulnerability study. Whether the vulnerability survives into end-to-end eDiscovery workflows is a separate question.

## Secondary LLM test (future work)

A more rigorous follow-up would test the full RAG pipeline: retrieve top-K → rerank → feed to LLM → ask for key documents. This project does not implement that pipeline but suggests it as the natural next step. A preliminary test — uploading selected conditions to Claude Projects / ChatGPT / Gemini and prompting "Identify the 5 most important documents" — can provide directional evidence on whether haystacking survives the summarization layer, but should not be treated as a controlled experiment.

---

## Implementation architecture

```
haystacking/
├── corpus/
│   ├── enron/
│   │   ├── download.py             # fetch EDRM Enron v2 from Internet Archive
│   │   ├── parse.py                # extract emails to plain text with metadata
│   │   ├── trec_topics/            # TREC Legal Track topic definitions
│   │   ├── trec_judgments/         # relevance assessments per topic
│   │   └── build_conditions.py     # construct conditions per topic per attacker level
│   └── conditions/
│       └── {topic_id}/
│           ├── baseline/
│           ├── haystacked_a/
│           ├── haystacked_b/
│           ├── haystacked_c/
│           ├── dilution_control/
│           └── metadata.json       # doc IDs, TREC judgments, selection method
├── embeddings/
│   └── {model_name}/               # cached embeddings per model
├── experiments/
│   ├── queries/
│   │   └── {topic_id}.json         # 20-30 queries per topic
│   ├── embed.py                    # embed all docs + queries across 9 models
│   ├── rank.py                     # cosine similarity ranking per condition
│   ├── evaluate.py                 # compute metrics (Recall@K, MRR, displacement)
│   └── detect.py                   # detection experiment (A/B vs C distinguishability)
├── results/
│   └── {topic_id}/
│       └── {model_name}/
├── paper/
└── README.md
```

**Pipeline:**
1. Download and parse Enron corpus (fetch EDRM v2, extract to plain text)
2. Load TREC Legal Track topics and relevance judgments
3. Select 3-5 topics, identify key documents per topic
4. Build conditions per topic: baseline, A, B, C, dilution control at each scale level
5. Embed all documents under all 9 models
6. Embed queries under all 9 models
7. Rank documents by cosine similarity per condition per model
8. Compute metrics and statistical tests
9. Run detection experiment (A/B vs C distinguishability)
10. Run mitigation experiments
11. Generate comparison tables, figures, and dose-response curves

---

## Limitations

**Enron is a single corpus from a single era.** All documents are emails from one company (2000-2002). No document type diversity (no contracts, depositions, spreadsheets). Language patterns and communication styles differ from modern corporate email. The topics relate to energy trading and regulatory matters, which may not generalize to all litigation domains. Replication on other public corpora or real productions under appropriate ethical protections would strengthen the findings.

**Retrieval layer only.** This experiment tests dense embedding retrieval in isolation. Production eDiscovery platforms add cross-encoder reranking, BM25 hybrid search, metadata filters, and LLM summarization on top of the retrieval layer. Any of these could mitigate or defeat the attack. The results demonstrate a retrieval-layer vulnerability, not an end-to-end workflow vulnerability. Whether haystacking survives through reranking and summarization is the most important open question for follow-up work.

**Scale.** The Enron corpus is large (1.7M emails) but the TREC relevance judgments cover a finite set of assessed documents per topic. The scale sweep can go up to 10× baseline per topic, but "true production scale" testing (50,000+ documents) depends on having enough assessed documents and may require extending beyond the TREC-judged subset.

**Key document selection.** "Smoking guns" are identified by TREC assessor relevance grades, not by independent legal judgment of case significance. This is a reasonable proxy — TREC assessors are professional reviewers — but responsiveness is not the same as case-dispositive importance. The distinction between "highly relevant" and "smoking gun" is acknowledged.

**Attacker realism.** Attacker A (broad collection) and B (keyword-aware) are realistic today. Attacker C (embedding optimization) requires technical sophistication most litigation teams don't currently have, but this capability is becoming accessible as legal AI tools proliferate. The experiment tests all three to show the gradient, not to claim all three are equally likely in practice.

---

## What this project does NOT test

- Real eDiscovery platform responses (no access to Relativity aiR, Everlaw, etc.)
- Actual litigation outcomes
- Whether courts would sanction strategic over-production
- Whether a human reviewer would catch what the AI missed
- Modern document types (contracts, depositions, spreadsheets — Enron is email-only)
- Interaction between haystacking and TAR/active learning workflows

---

## Success criteria

The research succeeds if:
1. **Primary:** At least one haystacked condition (A, B, or C) displaces significantly more than the size-matched dilution control across multiple embedding models. Measured by Recall@5 difference with bootstrap CIs and Wilcoxon tests.
2. The A → B → C gradient shows increasing displacement (more sophisticated selection = more effective attack)
3. The effect increases with scale (dose-response curve)
4. The attack generalizes across embedding model architectures (significant displacement on a majority of the 9 models)
5. The detection experiment can distinguish C from A/B — providing both the attack demonstration and the defense
6. Results replicate across multiple TREC topics (not an artifact of one topic)
