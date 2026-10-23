# Evaluation plan and experiments

Scientific Paper Search System (Knowledge Graph + RAG) · Roadmap tasks 3.1 and 4.1

**Status:** set-up complete, 2026-10-23 · **labels pending** — no score in this project is
reportable until §2 is filled in by hand.

Two Phase 1 choices are still assumptions: which embedding model retrieves best on this
corpus (`DECIDE-5` picked `bge-small` and asked for this check), and which extractor
should build the graph (`DECIDE-7`, `FR-9`). This document is how both get measured
rather than argued.

---

## 1. Plan

| Step | What | Where |
|---|---|---|
| 1 | Collect ~200 abstracts | [`notebooks/collect_abstracts.py`](../notebooks/collect_abstracts.py) → `data/eval/abstracts.jsonl` |
| 2 | Build the evaluation set: judged queries and entity-annotated abstracts | `data/eval/queries.jsonl`, `data/eval/entities.jsonl` — **by hand** |
| 3 | Search experiment: BM25, MiniLM, SPECTER2 (+ bge-small) | [`notebooks/exp_search.py`](../notebooks/exp_search.py) |
| 4 | Entity extraction experiment: rules, scientific NER, LLM | [`notebooks/exp_extraction.py`](../notebooks/exp_extraction.py) |

The evaluation corpus is deliberately separate from the 1,000-paper prototype corpus.
200 abstracts is small enough to judge thoroughly, and a fixed, committed corpus means a
score can always be recomputed against exactly what was labelled.

## 2. Evaluation set

Format, workflow and annotation guidelines: [`data/eval/README.md`](../data/eval/README.md).
Everything under `data/eval/` is tracked in git, unlike the rest of `data/`.

**Corpus.** The 200 most-cited Semantic Scholar papers matching "retrieval augmented
generation" whose abstract has at least 300 characters. Most-cited makes the papers
recognisable, which makes judging faster and more consistent.

**Queries** (`queries.jsonl`, target 20–30). Five are seeded from the use cases in
[`requirements.md`](requirements.md) §5; the rest are written by the team. Relevance is
judged by **pooling**: every system's top 10 for a query is merged into one candidate list
(`pool.jsonl`) and judged from title and abstract. Papers outside the pool count as not
relevant — the standard TREC assumption, and fair here because every system under test
contributes to the pool.

**Entities** (`entities.jsonl`). 50 abstracts sampled with a fixed seed, annotated with
`Method`, `Dataset` and `Task` spans copied verbatim from the text.

**Why the labels are not generated.** An LLM is one of the systems being compared.
Model-written gold labels would score that model against itself, and would bias every
other system towards agreeing with it. The tooling stops at producing pools and skeletons.

**Agreement.** Both members annotate the same 10 abstracts first. Disagreements there are
gaps in the guidelines and get fixed before the rest is split.

## 3. Search experiment

| System | Model | Query handling | Notes |
|---|---|---|---|
| BM25 | `rank_bm25` Okapi | `\w+` tokens, lowercased | Title + abstract |
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | as-is | The fast baseline in `DECIDE-5` |
| bge-small | `BAAI/bge-small-en-v1.5` | `settings.embedding_query_prefix` | The model `DECIDE-5` pinned — not in the original plan, added because `DECIDE-5` asks for exactly this comparison |
| SPECTER2 | `allenai/specter2_base` + adapters | `specter2_adhoc_query` adapter | Documents use the `specter2` proximity adapter on `title [SEP] abstract`, as the model card prescribes |

**Metrics**, top 10, binary relevance, macro-averaged over judged queries: Recall@10,
MRR@10, nDCG@10. The metric functions carry hand-computed asserts that run before any
table is printed.

**SPECTER2 runs in its own venv** (`.venv-specter2`). The `adapters` library requires
`transformers < 5`, and downgrading the shared venv would constrain the LLM choice in
`DECIDE-7`. The encoder writes vectors to `data/processed/specter2/`; `exp_search.py`
refuses vectors that do not match the current corpus or queries. Rebuild them after every
edit to `queries.jsonl`:

```bash
python3.11 -m venv .venv-specter2 && .venv-specter2/bin/pip install adapters torch  # once
.venv-specter2/bin/python notebooks/encode_specter2.py
```

## 4. Entity extraction experiment

| System | What it is | Notes |
|---|---|---|
| rules | Regular expressions: capitalised names before *dataset/benchmark/corpus*, `*QA` names, mixed-case tokens (*ColBERT*, *Self-RAG*), a list of task phrases | Baseline; high recall on names, no understanding of context |
| scientific NER | GLiNER zero-shot NER, `urchade/gliner_medium-v2.1`, labels `method` / `dataset` / `task`, threshold 0.5 | Chosen over scispaCy: scispaCy is trained on biomedical text, emits a single untyped `ENTITY` label, and would downgrade `numpy` in the shared venv |
| LLM | `Qwen/Qwen2.5-3B-Instruct`, loaded in-process with `transformers` (`NFR-8`), greedy decoding | Model chosen for this experiment only — `DECIDE-7` stays open. `settings.llm_model` overrides it |

**Scoring.** An entity is the pair (type, normalised text); normalisation is case,
whitespace and surrounding punctuation. Micro precision, recall and F1, per type and
overall, over annotated abstracts only. Exact match is strict on purpose: *DPR* versus
*dense passage retriever* counts as a miss, which is the fragmentation `FR-10` has to deal
with downstream anyway.

**The LLM is not schema-constrained yet.** It is asked for JSON and parsed; the script
reports how many outputs failed to parse. `DECIDE-7` requires constrained decoding for
`FR-9`, and that count is the measurement that justifies adding the library.

**Predictions are saved** to `data/processed/extraction/<system>.jsonl` for error
analysis, so annotating more abstracts only needs a re-score, not a reason to guess.

## 5. Results

| | Status |
|---|---|
| Search | Pipeline verified end to end; **no scores** — 5 of 20–30 queries written, none judged |
| Extraction | Pipeline verified end to end; **no scores** — 0 of 50 abstracts annotated |

Fill this section from the script output once the labels exist, with the commit hash of
`data/eval/` the numbers were computed against.

## 6. Relationship to the prototype pipeline

| Question | Answered by | Feeds |
|---|---|---|
| Which embedding model? | §3 | `settings.embedding_model` (`DECIDE-5`), `scripts/build_index.py` |
| Which extractor builds the graph? | §4 | `paper_search.extraction.entities`, `scripts/extract_entities.py` |

KG v1 uses GLiNER as a stand-in extractor ([`kg-schema.md`](kg-schema.md) §4) because it
runs over 1,000 abstracts in minutes on a laptop. Every entity relationship in the graph
records its `extractor`, so swapping in the winner of §4 is a re-run, not a migration.
