# Project Roadmap & Task List

This document lists everything that needs to be done to build the **Scientific Paper Search System (Knowledge Graph + RAG)**. The project runs over two semesters as two separate courses. Tick each box when a task is done so the whole team can see progress at a glance.

**Legend:** `[ ]` to do · `[x]` done · **Deliverable** = what must exist at the end of the step

---

## Overview

| Phase | Time | Focus | Main outcome |
|---|---|---|---|
| **Phase 1** (current) | Sep – Dec 2026 | Design, data, Knowledge Graph, search baseline | Documented architecture, KG v1, working BM25 + vector search, Phase 1 report |
| **Phase 2** | Next semester (separate course) | RAG, chatbot, UI, evaluation | End-to-end demo with cited answers, evaluation results, final thesis report |

### Phase 1 timeline

| Month | Focus |
|---|---|
| **September** | Repository setup, requirements, architecture, self-study |
| **October** | KG schema, data collection, PDF processing |
| **November** | Information extraction, KG v1, search baseline |
| **December** | Baseline evaluation, Phase 1 report and presentation |

---

## Phase 1: Foundations (Sep – Dec 2026)

### September: Requirements, Architecture & Self-Study

#### 1.1 Repository setup
- [ ] Create the folder structure (`src/`, `docs/`, `data/`, `tests/`, `notebooks/`)
- [ ] Add `requirements.txt`, `.env.example`, `.gitignore` (ignore `data/`, `.env`, index files, model caches)
- [ ] Add `docker-compose.yml` to run Neo4j locally with the same config for both members
- [ ] Agree on conventions: branch naming, commit messages, pull request review between members
- [ ] Set up a code formatter/linter (e.g. `ruff`, `black`)

**Deliverable:** a repo both members can clone and run Neo4j from with one command.

#### 1.2 Requirements analysis (`docs/requirements.md`)
Covers the **whole system**, including Phase 2 features, so the design supports them from the start.
- [ ] Define the target users (students, researchers) and their main goals
- [ ] Write functional requirements: search, filter, explore graph, Q&A, summarize, compare methods, recommend related work
- [ ] Write non-functional requirements: response time, answer faithfulness, citation accuracy, dataset size, cost limits
- [ ] Write 5–8 use cases with example queries and expected results
- [ ] Define the scope limits (which fields/topics, how many papers, which languages)

**Deliverable:** a requirements document agreed on by both members.

#### 1.3 Architecture design (`docs/architecture.md`)
- [ ] Draw the end-to-end pipeline: collection → extraction → KG → indexing → retrieval → RAG → chatbot
- [ ] Define each module's responsibility, inputs, and outputs
- [ ] Define internal APIs/interfaces between modules (function signatures or REST endpoints)
- [ ] Design **provenance tracking**: every text chunk must link back to paper ID, section, and position, so answers can cite sources
- [ ] Record key design decisions and why (e.g. why Neo4j, why hybrid search, which embedding model)

**Deliverable:** an architecture document with diagrams.

#### 1.4 Self-study
See [`docs/learning-path.md`](docs/learning-path.md) for resources.
- [ ] Requirements & architecture documentation (arc42)
- [ ] Semantic Scholar and arXiv APIs
- [ ] PDF processing with PyMuPDF
- [ ] Knowledge Graph concepts (Stanford CS520)
- [ ] Neo4j & Cypher (GraphAcademy)
- [ ] NER with spaCy and LLM-based extraction
- [ ] Embeddings (Sentence-Transformers) and FAISS
- [ ] BM25 (`rank_bm25`)

---

### October: KG Schema, Data Collection & PDF Processing

#### 2.1 Knowledge Graph model (`docs/kg-schema.md`)
- [ ] Define node types: `Paper`, `Author`, `Venue`, `Topic`, `Method`, `Dataset`
- [ ] Define relationships: `AUTHORED`, `CITES`, `PUBLISHED_IN`, `HAS_TOPIC`, `USES_METHOD`, `USES_DATASET`
- [ ] Define properties for each node and relationship
- [ ] Choose unique keys for deduplication (DOI, arXiv ID, Semantic Scholar ID)
- [ ] Write example Cypher queries the system must support (e.g. "papers using dataset X with method Y")

**Deliverable:** a schema diagram and a list of target queries.

#### 2.2 Data collection (`src/collection/`)
- [ ] Choose a narrow topic for the prototype dataset (e.g. "retrieval-augmented generation")
- [ ] Write a Semantic Scholar client: search, pagination, rate-limit handling, retries
- [ ] Write an arXiv client for metadata and PDF download
- [ ] Collect **500–1,000 papers** with metadata (title, abstract, authors, year, venue, citations, references)
- [ ] Normalize fields into one common format
- [ ] Deduplicate papers that appear in both sources
- [ ] Save raw and processed data (JSON/Parquet) with a data version

**Deliverable:** a clean prototype metadata set.

#### 2.3 PDF processing (`src/collection/pdf/`)
- [ ] Extract text from PDFs with PyMuPDF
- [ ] Split text into sections (Abstract, Introduction, Method, Experiments, Conclusion…)
- [ ] Clean text: remove headers/footers, fix hyphenation, drop references section if needed
- [ ] Log papers that fail extraction

**Deliverable:** sectioned full text for the papers with available PDFs.

---

### November: Information Extraction, KG v1 & Search Baseline

#### 3.1 Information extraction (`src/extraction/`)
- [ ] Try spaCy NER on sample abstracts and review the quality
- [ ] Write LLM prompts to extract methods, datasets, and tasks as structured JSON
- [ ] Normalize entity names (e.g. "BERT-base" and "BERT" → one entity)
- [ ] Manually check a sample of ~50 papers to estimate extraction accuracy

**Deliverable:** extracted entities for the prototype set with an accuracy estimate.

#### 3.2 Knowledge Graph v1 (`src/graph/`)
- [ ] Create constraints and indexes in Neo4j
- [ ] Write loaders using `MERGE` so re-running does not create duplicates
- [ ] Load papers, authors, venues, and citations
- [ ] Load extracted methods, datasets, and topics
- [ ] Test the target Cypher queries from 2.1
- [ ] Record graph statistics (number of nodes and relationships by type)

**Deliverable:** a populated KG v1 that answers the target queries.

#### 3.3 Search baseline (`src/indexing/`, `src/retrieval/`)
- [ ] Split text into chunks and keep the provenance metadata
- [ ] Build a BM25 index on titles and abstracts (then full-text chunks)
- [ ] Generate embeddings with Sentence-Transformers and build a FAISS index
- [ ] Write a simple search function for each method returning top-k results

**Deliverable:** working keyword and vector search.

---

### December: Baseline Evaluation & Phase 1 Report

#### 4.1 Baseline evaluation
- [ ] Create a **test query set** (20–30 queries with the relevant papers marked by hand)
- [ ] Measure BM25 and vector search results (Recall@k, MRR, nDCG)
- [ ] Note weaknesses of each method to motivate hybrid search and RAG in Phase 2

**Deliverable:** baseline scores that Phase 2 will be compared against.

#### 4.2 Phase 1 report
- [ ] Write the Phase 1 report: problem, requirements, architecture, KG model, data, baseline results, Phase 2 plan
- [ ] Prepare presentation slides
- [ ] Prepare a short demo: Cypher queries on KG v1 and baseline search results
- [ ] Clean up code and update README with setup instructions

#### ✅ Phase 1 checkpoint
- [ ] Requirements, architecture, and KG schema documents complete
- [ ] Prototype dataset collected and processed
- [ ] KG v1 loaded in Neo4j
- [ ] BM25 and FAISS baselines working and measured
- [ ] Phase 1 report and presentation done

---

## Phase 2: RAG, Chatbot & Evaluation (Next Semester)

The detailed schedule will be planned at the start of the next semester.

#### 5.1 Hybrid retrieval
- [ ] Combine BM25 and vector scores (e.g. Reciprocal Rank Fusion or weighted sum)
- [ ] Add KG-based retrieval: expand results using citations, shared methods/datasets, and authors
- [ ] Optionally add a reranker (cross-encoder) on top of the merged results

#### 5.2 RAG pipeline
- [ ] Choose the LLM (API or local) based on cost and quality
- [ ] Design prompts that force answers to use only the retrieved context
- [ ] Add inline citations that map to paper IDs and sections
- [ ] Handle "no relevant information found" instead of guessing

#### 5.3 Chatbot
- [ ] Keep conversation history for follow-up questions
- [ ] Rewrite follow-up questions into standalone queries before retrieval
- [ ] Support intents: search, summarize a paper, compare methods, Q&A, recommend related work, explore a topic

#### 5.4 End-to-end API & demo UI
- [ ] FastAPI backend with endpoints for search, chat, paper details, and graph exploration
- [ ] Search page with results and filters (year, venue, topic)
- [ ] Chat interface showing answers with clickable citations
- [ ] Paper detail page; optional graph visualization

#### 5.5 Evaluation
- [ ] Retrieval: compare BM25, vector, hybrid, and hybrid + KG against the Phase 1 baselines
- [ ] Generation: measure faithfulness, answer relevance, and citation accuracy
- [ ] **Ablation study:** remove one component at a time (KG, BM25, vectors, reranker) and measure the impact
- [ ] **Error analysis:** group failure cases and explain their causes
- [ ] Measure response time

#### 5.6 Final thesis
- [ ] Add tests for core modules and finalize the README
- [ ] Write the final thesis report
- [ ] Prepare the defense slides, demo script, and a backup demo video

---

## Team

Both members work across all tasks; the split below shows who leads each area.

| Area | Lead |
|---|---|
| Data collection, PDF processing, Knowledge Graph | Tran Ha My |
| Search, RAG, chatbot, UI | Pham Minh Hieu |
| Requirements, architecture, evaluation, report | Shared |