# Haystacking: Adversarial Retrieval Through Strategic Over-Production in Legal Discovery

## Design Spec

### Overview

Haystacking is a security research project demonstrating that strategic over-production of authentic, non-privileged documents in litigation discovery can bury smoking-gun evidence in AI-assisted document review. The attack exploits dense retrieval systems by flooding the corpus with real, topically related documents that dilute the semantic signal of damaging evidence.

This is not prompt injection, parser differential, document fabrication, or withholding. Every document in the production is authentic. The attack controls which documents surface, not what they say.

### Hypothesis

A producing party can cause AI-assisted review tools to omit smoking-gun documents from key document lists by over-producing authentic, non-privileged documents that share topic, vocabulary, and entities with the damaging evidence.

### Relationship to prior work

- **PoisonedRAG (USENIX Security 2025):** Demonstrated 90-99% attack success rate on RAG systems by injecting synthetic adversarial texts. Key difference: PoisonedRAG produces unnatural texts with high repetition (0.28 rate) that are potentially detectable. Haystacking uses exclusively real documents — no synthetic content, no detection signal.
- **DIGA:** Showed dense retrievers have exploitable properties (insensitivity to token order, bias toward influential tokens). Haystacking exploits a different property: dilution through volume.
- **Lying Spreadsheets (Zhu, 2026):** Parser differential attack on XLSX files. Proved category 3 (parser differentials) from the AI attack surface taxonomy. Haystacking proves category 2 (RAG poisoning) via a novel mechanism.
- **AI attack surface taxonomy:** https://legalrealist.ai/posts/36-when-ai-is-the-attack-surface/

### Fictional case

**Meridian Partners v. Sarah Chen** — trade secret misappropriation.

Sarah Chen, a senior account manager at Meridian Partners (a consulting firm), resigns to join a competitor. Meridian claims she took proprietary client lists and relationship data.

**Custodians:** Sarah Chen, her manager, HR department, IT department, CEO of the new company.

### Corpus design

**Total size:** ~200 documents per experimental condition.

**75 clean documents (baseline):**

| Type | Count | Examples |
|---|---|---|
| Emails | ~35 | Internal chains about clients, Chen's resignation, IT access logs |
| Contracts | ~8 | Employment agreement, NDA, IP assignment, non-compete, offer letter |
| Policies | ~6 | Employee handbook excerpts, IT security policy, offboarding checklist |
| Deposition excerpts | ~8 | Chen, her manager, IT admin, HR director |
| Spreadsheets (as text) | ~5 | Client lists, CRM exports, access logs |
| Memos/reports | ~13 | Exit interview, internal investigation, management emails |

**5 smoking guns (included in the clean 75):**

1. **The forwarding email:** Chen forwards client contact spreadsheet to personal email the night before resigning
2. **The planning email:** Email chain with new employer's CEO discussing which Meridian clients to approach first
3. **The admission chat:** Chat message to a friend: "took the whole client list, they'll never notice"
4. **The overlap spreadsheet:** Spreadsheet at new company with 80% overlap with Meridian's client list
5. **The cover-up email:** Chen asks IT to "clean up my email" on her last day

**125 hay documents (intentional haystacking):** ~25 per smoking gun. Authentic, non-privileged documents selected because they share topic, vocabulary, and entities with the corresponding smoking gun. Examples:

- For the forwarding email: routine emails about client handoffs to other account managers, CRM export requests for quarterly reports, standard data transfer procedures
- For the planning email: business development emails discussing prospective clients, competitive landscape memos, industry conference follow-ups
- For the admission chat: casual Slack messages about work, offboarding chatter from other departing employees, social messages mentioning clients in passing
- For the overlap spreadsheet: old CRM exports, quarterly client reports, contact lists for events and mailings
- For the cover-up email: IT cleanup requests from other employees, standard offboarding IT tickets, email archiving policy reminders

**125 hay documents (inadvertent haystacking):** Same count, selected by a simple topical filter: "does this document mention clients, departures, data management, or competitive matters?" No reference to specific smoking guns, no awareness of which documents are damaging. This simulates a producing party that casts a wide net on responsiveness without strategic intent — the kind of over-production that happens routinely when a litigation team errs on the side of inclusion.

**Constraints on all hay documents:**
- Non-privileged — no attorney-client communications, no work product, no litigation strategy
- Authentic to the producing party's files — only documents Chen's former employer would actually have
- Responsive or arguably responsive — plausibly related to the case topics
- No fabricated content — every document must be realistic for the business context

### Embedding models (9 total)

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

### Experimental design

**Three conditions:**
- **Baseline:** 75 clean documents
- **Intentional haystacking:** 75 clean + 125 targeted hay (200 total)
- **Inadvertent haystacking:** 75 clean + 125 untargeted hay (200 total)

**Queries (8-10 total):** Natural-language queries a lawyer would type into an eDiscovery platform:
1. "key documents showing misappropriation of trade secrets"
2. "evidence that Chen intended to take client information"
3. "smoking gun documents in this case"
4. "documents showing theft of proprietary client data"
5. "most important documents for plaintiff's case"
6. (3-5 additional queries with varied phrasing, TBD during implementation)

**Metrics:**

| Metric | What it measures |
|---|---|
| **Recall@5** (primary) | How many smoking guns in top 5 results — maps to "AI picks the 5 most important docs" workflow |
| **Recall@10** | Same for top 10 |
| **Recall@20** | Same for top 20 |
| **MRR** | Average reciprocal rank of first smoking gun — how deep must you scroll |
| **Displacement** | Average rank change of smoking guns between baseline and haystacked conditions |

**No optimization loop.** Hay is selected once based on topical similarity, then frozen. The experiment measures what happens with a realistic one-shot attack, not the theoretical optimum.

### Document format

- **Plain text** for all embedding experiments — clean signal, no format-layer confounds
- **PDF** for the practical demonstration (uploading to platforms) — tests whether extraction pipelines affect results

### Implementation architecture

```
haystacking/
├── corpus/
│   ├── clean/              # 75 base documents
│   ├── hay_intentional/    # 125 targeted hay
│   ├── hay_inadvertent/    # 125 blind saturation hay
│   └── metadata.json       # doc IDs, types, custodians, smoking gun flags
├── embeddings/
│   └── {model_name}/       # cached embeddings per model
├── experiments/
│   ├── queries.json        # the 8-10 test queries
│   ├── embed.py            # embed all docs + queries across 9 models
│   ├── rank.py             # cosine similarity ranking per condition
│   └── evaluate.py         # compute metrics (Recall@K, MRR, displacement)
├── results/
│   └── {model_name}/       # per-model results tables
├── paper/                  # writeup
└── README.md
```

**Pipeline:**
1. Generate corpus (LLM-assisted authoring, human-curated for consistency)
2. Embed all documents under all 9 models
3. Embed queries under all 9 models
4. Rank documents by cosine similarity per condition per model
5. Compute metrics
6. Generate comparison tables and figures

**Document generation:** LLM-assisted but human-curated. Use Claude to draft realistic documents, then review for consistency across the corpus (dates, names, custodian voices, document metadata). The corpus must feel like a real production, not a collection of isolated documents.

### Secondary LLM test

After the embedding experiments, upload all three corpora to Claude Projects / ChatGPT / Gemini and prompt: "Identify the 5 most important documents in this production." Compare key document lists across conditions. This tests whether haystacking survives into the summarization layer, not just the retrieval layer.

### Defenses to test

- Provenance tracking — does flagging production source help?
- Consistency checking — do contradictions between documents raise flags?
- Redundancy detection — does identifying topically similar clusters help surface the signal?
- Asking the model to identify the most *unique* or *anomalous* documents instead of the most *important*

### What this project does NOT test

- Real eDiscovery platform responses (no access to Relativity aiR, Everlaw, etc.)
- Actual litigation outcomes
- Whether courts would sanction strategic over-production
- Optimal hay ratios (we test one ratio, 63% hay)
- Whether a human reviewer would catch what the AI missed

### Success criteria

The research succeeds if:
1. Recall@5 for smoking guns drops measurably between baseline and intentional haystacking across multiple embedding models
2. The effect is present in inadvertent haystacking (even without targeting)
3. The attack generalizes across embedding model architectures
