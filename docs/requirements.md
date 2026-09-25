# Requirements

Scientific Paper Search System (Knowledge Graph + RAG) · Roadmap task 1.2

**Status:** draft, 2026-09-23 · **Authors:** Tran Ha My, Pham Minh Hieu

This document covers the **whole system**, including the Phase 2 features, so that the
Phase 1 design does not have to be reworked later. Requirements carry stable IDs
(`FR-*`, `NFR-*`, `UC-*`) so `architecture.md`, `kg-schema.md` and the tests can point
back at them.

The six decisions that were open in the first draft were all resolved on 2026-09-24; the
requirements below state the chosen values. The reasoning and the rejected options are in
[`decisions.md`](decisions.md), and [§8](#8-resolved-decisions) summarizes the outcomes.

---

## 1. Target users

| User | Who they are | What they are trying to do |
|---|---|---|
| **U1 — Student** | Undergraduate or master's student starting on a new topic | Get oriented in an unfamiliar field: which papers matter, what the standard methods and datasets are, what a paper says without reading all 15 pages |
| **U2 — Researcher** | Someone already working in the field | Find specific related work, check whether an idea has been tried, compare how several papers approach the same problem, locate papers that use a particular dataset or method |

Both users share the same core frustration: keyword search returns papers that contain
the words but not the answer, and it says nothing about how papers relate to each other.

**Primary user for Phase 1 evaluation:** U2, because their queries are specific enough to
have a checkable ground truth (see `UC-2`, `UC-5`).

---

## 2. Scope limits

Measured by the collection spike in
[`notebooks/spike_data_availability.py`](../notebooks/spike_data_availability.py),
run 2026-09-23 against Semantic Scholar and arXiv.

| Limit | Value | Basis |
|---|---|---|
| **Topic** | Retrieval-Augmented Generation and neighbouring retrieval / QA work | 17,536 papers available on Semantic Scholar — far more than needed |
| **Corpus size** | 1,000 papers with metadata | Roadmap 2.2 target; reachable in a single bulk API call |
| **With full text** | ~640 papers (64%) | 644 of 1,000 sampled had an arXiv ID or an open-access PDF |
| **Metadata only** | ~360 papers (36%) | The remaining 356 — **the system must still index and return these** |
| **Language** | English only | Non-English papers are dropped at collection time |
| **Time span** | No explicit cut-off | The topic is young: ~80% of matches are from 2024–2026 |
| **Modality** | Text only | No figures, tables, equations or images are extracted |

### 2.1 Consequences of the measurements

These three facts come from the spike and directly constrain the design.

**S1 — A third of the corpus has no full text.** Chunking, indexing and retrieval must all
work for a paper whose only text is its title and abstract. Provenance must record whether
a chunk came from `abstract` or from a named section, so that citations stay honest about
what was actually read (`FR-14`, `NFR-6`).

**S2 — PDF access is uneven.** Papers with an arXiv ID download reliably. Links from
`openAccessPdf` often resolve to publisher sites that reject automated requests
(`dl.acm.org` returned HTTP 403 for every attempt in the sample). Resolution order is
therefore: arXiv ID → `openAccessPdf` URL → give up and keep the paper as metadata-only,
with the failure logged (`FR-4`). **Publisher bot protection is not circumvented.**
No scanned PDFs appeared in the sample, so OCR is out of scope.

**S3 — The internal citation graph will be sparse.** With ~80% of the corpus published in
the last two years, most citations point *outside* the collected set, and recent papers
have few incoming citations. This weakens `FR-17` (citation-based expansion). Mitigation
in `FR-3`: one hop of reference snowballing during collection (adopted, `DECIDE-1`).

---

## 3. Functional requirements

### Phase 1 — data, graph, search baseline

| ID | Requirement |
|---|---|
| **FR-1** | Collect paper metadata from the Semantic Scholar API: title, abstract, authors, year, venue, citation count, reference count, external IDs (DOI, arXiv, S2). |
| **FR-2** | Collect paper metadata and PDFs from the arXiv API. |
| **FR-3** | Expand the corpus by one hop of references from the seed papers, keeping those cited by at least 3 seed papers. These carry the `:External` label, are graph-only, and are excluded from the search index. |
| **FR-4** | Download full-text PDFs using the resolution order in `S2`; log every failure with its reason; keep the paper as metadata-only rather than dropping it. |
| **FR-5** | Normalize records from both sources into one common paper schema. |
| **FR-6** | Deduplicate papers appearing in both sources, keyed on DOI, then arXiv ID, then Semantic Scholar ID, then normalized title. |
| **FR-7** | Extract text from PDFs and split it into the canonical sections: Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion, Other. Headings that do not map to one of the first seven go to `Other` rather than being dropped. |
| **FR-8** | Clean extracted text: strip running headers and footers, repair hyphenation across line breaks, and separate the references section from the body. |
| **FR-9** | Extract entities — methods, datasets, tasks, topics — from each paper using spaCy NER and a **locally run open-weights LLM**. Extraction output is constrained to a JSON schema at decode time, not requested in the prompt and repaired afterwards. |
| **FR-10** | Normalize entity names so surface variants collapse to one entity (e.g. "BERT-base" and "BERT"). |
| **FR-11** | Build a Neo4j Knowledge Graph with node types `Paper`, `Author`, `Venue`, `Topic`, `Method`, `Dataset` and relationships `AUTHORED`, `CITES`, `PUBLISHED_IN`, `HAS_TOPIC`, `USES_METHOD`, `USES_DATASET`. Detailed in `kg-schema.md`. |
| **FR-12** | Make graph loading idempotent (`MERGE`, plus uniqueness constraints), so re-running a load never duplicates nodes. |
| **FR-13** | Split paper text into retrieval chunks: by section first, then into overlapping windows within any section that exceeds the embedding model's input limit. Canonical sections are Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion, Other. |
| **FR-14** | Attach provenance to every chunk: paper ID, source (`abstract` or section name), and position within that source. Nothing may enter an index without it. |
| **FR-15** | Build a BM25 keyword index and a FAISS vector index over the chunks, and expose a top-k search function for each. |

### Phase 2 — hybrid retrieval, RAG, chatbot

| ID | Requirement |
|---|---|
| **FR-16** | Fuse BM25 and vector rankings into one result list (Reciprocal Rank Fusion or a weighted sum). |
| **FR-17** | Expand results through the graph: papers sharing a method, dataset, topic, author, or citation link with a strong hit. Effectiveness is bounded by `S3`. |
| **FR-18** | Filter results by year, venue, topic, method, and dataset. |
| **FR-19** | Generate answers from the retrieved context only, using the same locally run model, and refuse to answer — explicitly — when the retrieved context does not support one. |
| **FR-20** | Attach inline citations to every claim, resolving to a paper ID and the specific section the text came from. |
| **FR-21** | Summarize a single paper on request. |
| **FR-22** | Compare how two or more papers approach the same problem. |
| **FR-23** | Recommend related work for a paper or a described idea. |
| **FR-24** | Hold conversation state, and rewrite follow-up questions into standalone queries before retrieval. |
| **FR-25** | Expose a REST API: search, chat, paper detail, graph exploration. |
| **FR-26** | Provide a demo UI: search with filters, chat with clickable citations, and a paper detail page. |

---

## 4. Non-functional requirements

| ID | Attribute | Target | Why this number |
|---|---|---|---|
| **NFR-1** | Search latency (BM25 or FAISS, top-10) | < 1 s, p95, on a 1,000-paper corpus | Interactive search stops feeling interactive past ~1 s |
| **NFR-2** | End-to-end answer latency | First token < 3 s; complete answer < 20 s, p95 | Local GPU inference, answer streamed. Perceived latency is the first token, not the last. The old < 10 s target assumed a hosted API and is not reachable for a 500-token answer on a single GPU |
| **NFR-3** | Faithfulness | ≥ 90% of generated claims supported by the retrieved context | Manual review of a 50-answer sample (~200 claims), both members grading independently; scored 1 / 0.5 / 0 with weighted Cohen's kappa reported |
| **NFR-4** | Citation accuracy | ≥ 95% of citations resolve to a real paper that actually contains the cited content | A wrong citation is worse than no citation for an academic tool |
| **NFR-5** | Refusal behaviour | The system says it does not know rather than guessing, whenever retrieval returns nothing above the relevance threshold | Follows from `FR-19` |
| **NFR-6** | Provenance completeness | 100% of indexed chunks carry paper ID, source and position | `FR-20` is impossible without this; cheap to enforce, expensive to retrofit |
| **NFR-7** | Corpus scale | 1,000 papers, ~640 with full text (§2) | Measured, not assumed |
| **NFR-8** | External services | **No paid or hosted LLM API.** All inference runs on hardware the team controls | Required by the course. Removes the running-cost question entirely and replaces it with a compute budget (`NFR-12`) |
| **NFR-9** | Reproducibility | A clean clone reaches a running Neo4j and a green test suite with `make install && make neo4j-up && make test`; the LLM runtime is installed the same way on both machines | Both members must get identical environments. A self-hosted model makes this harder than an API key did, so the model name and runtime version are pinned, not left to whatever each machine happens to have |
| **NFR-10** | Idempotent pipeline | Every stage can be re-run without corrupting or duplicating its output | Follows from `FR-12`; a pipeline that cannot be re-run cannot be debugged |
| **NFR-11** | Rate-limit compliance | Respect the published rate limits of Semantic Scholar and arXiv; back off on HTTP 429; never work around bot protection | `S2`; also a condition of using these APIs at all |
| **NFR-12** | Hardware | One GPU for LLM inference; embedding and indexing stay CPU-feasible | The GPU bounds the LLM size (`DECIDE-7`). Embedding deliberately stays on CPU with `BAAI/bge-small-en-v1.5` (384 dim) so indexing is not blocked when the GPU is busy or unavailable |

---

## 5. Use cases

Each use case names the requirements it exercises. `UC-1` to `UC-4` are Phase 1 and can be
checked against the search baseline; `UC-5` to `UC-7` need the Phase 2 RAG pipeline.

### UC-1 — Find papers on a topic (U1, U2)

> "retrieval augmented generation for long documents"

**Expected:** top-10 papers ranked by relevance, each showing title, authors, year, venue
and abstract, with an indication of whether full text is available.
**Exercises:** `FR-15`, `NFR-1`. **Success:** at least 5 of the top 10 judged relevant.

### UC-2 — Find papers using a specific dataset or method (U2)

> "papers that evaluate on Natural Questions using a dense retriever"

**Expected:** papers linked to both entities in the graph, answerable by a single Cypher
query without any text search.
**Exercises:** `FR-9`, `FR-10`, `FR-11`. **Success:** returns a non-empty, manually
verified correct set — this is the use case that justifies building a graph at all.

### UC-3 — Explore a paper's neighbourhood (U1, U2)

> Starting from one paper: what does it cite, what cites it, which papers share its
> methods or datasets, what else did its authors write?

**Expected:** a list of connected papers grouped by relationship type.
**Exercises:** `FR-11`, `FR-17`. **Note:** `CITES` results will be thin — see `S3`.

### UC-4 — Filter a result set (U2)

> "hybrid retrieval, 2024 onwards, ACL venues only"

**Expected:** the `UC-1` result list narrowed by structured metadata.
**Exercises:** `FR-18`.

### UC-5 — Ask a question and get a cited answer (U1, U2)

> "What are the main approaches to reducing hallucination in RAG systems?"

**Expected:** a paragraph-level answer in which every claim carries an inline citation
resolving to a paper ID and section. Nothing asserted that the retrieved context does not
support.
**Exercises:** `FR-19`, `FR-20`, `NFR-3`, `NFR-4`.

### UC-6 — Summarize a paper (U1)

> "Summarize the paper on self-RAG"

**Expected:** problem, method, datasets, main result, stated limitations. For a
metadata-only paper, an abstract-based summary that **says so explicitly** rather than
silently giving a thinner answer.
**Exercises:** `FR-21`, `S1`.

### UC-7 — Compare methods (U2)

> "How do dense retrieval and BM25 compare for open-domain QA in these papers?"

**Expected:** a comparison grounded in specific papers, each point cited, with
disagreements between papers surfaced rather than averaged away.
**Exercises:** `FR-22`, `FR-16`.

### UC-8 — Ask a question with no answer in the corpus (U1, U2)

> "What is the melting point of tungsten?"

**Expected:** an explicit statement that the corpus contains nothing relevant. No
fabricated answer, no citation of unrelated papers.
**Exercises:** `FR-19`, `NFR-5`. **Included deliberately:** this is the failure mode that
most damages trust in a research tool.

---

## 6. Out of scope

Recorded so the boundary is deliberate rather than accidental.

- Non-English papers.
- Figures, tables, equations and images — text only.
- OCR for scanned PDFs: none appeared in the sample (§2.1, `S2`).
- Bypassing publisher bot protection or paywalls (`NFR-11`).
- User accounts, saved searches, personalization.
- Live re-crawling: the corpus is collected once, with a recorded data version.
- Full-scale deployment, containerized serving, horizontal scaling. The deliverable is a
  local demo.

---

## 7. Traceability

| Roadmap deliverable | Requirements |
|---|---|
| 2.2 Data collection | `FR-1` – `FR-6`, `NFR-11` |
| 2.3 PDF processing | `FR-4`, `FR-7`, `FR-8` |
| 3.1 Information extraction | `FR-9`, `FR-10` |
| 3.2 Knowledge Graph v1 | `FR-11`, `FR-12` |
| 3.3 Search baseline | `FR-13` – `FR-15`, `NFR-1` |
| 4.1 Baseline evaluation | `UC-1` – `UC-4`, `NFR-1` |
| 5.1 Hybrid retrieval | `FR-16` – `FR-18` |
| 5.2 RAG pipeline | `FR-19`, `FR-20`, `NFR-2` – `NFR-5` |
| 5.3 Chatbot | `FR-21` – `FR-24` |
| 5.4 API and demo UI | `FR-25`, `FR-26` |
| 5.5 Evaluation | `NFR-1` – `NFR-4`, `UC-5` – `UC-8` |

---

## 8. Resolved decisions

All six were resolved on 2026-09-24. Full reasoning, the rejected options and the
amendments made to each recommendation are in [`decisions.md`](decisions.md).

| ID | Outcome | Affects |
|---|---|---|
| **DECIDE-1** | Snowball one hop, keeping references cited by ≥3 seeds; marked with the `:External` label | `FR-3`, `FR-17`, `UC-3` |
| **DECIDE-2** | Section-aware windows, over the eight canonical section names | `FR-13`, `FR-14` |
| **DECIDE-3** | 50 answers, both members grading independently, scored 1 / 0.5 / 0, weighted Cohen's kappa | `NFR-3`, `NFR-4` |
| **DECIDE-4** | ~~Extraction on a hosted API~~ — **superseded 2026-09-25**: the course requires self-hosted inference. Reopened as `DECIDE-7` | `FR-9`, `NFR-8` |
| **DECIDE-5** | `BAAI/bge-small-en-v1.5`, 384 dim, pinned in `config.py` together with its query prefix | `FR-13`, `FR-15`, `NFR-1` |
| **DECIDE-6** | Conventions adopted as proposed; in effect from 2026-09-24 | Roadmap 1.1 |

**DECIDE-7 is open**: which open-weights model, at which size, on which local runtime. See
[`decisions.md`](decisions.md). It replaces `DECIDE-4`, which assumed a hosted API.

Still outstanding, though not a decision: the Semantic Scholar API key, being applied for
as of 2026-09-24. The unauthenticated quota is not sufficient for `FR-3`.

---

## 9. Sign-off

| Member | Reviewed | Date |
|---|---|---|
| Tran Ha My | ☐ | |
| Pham Minh Hieu | ☐ | |
