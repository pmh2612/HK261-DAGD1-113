# Learning Path

Roadmap task 1.4 · linked from [`../Roadmap.md`](../Roadmap.md)

Self-study resources for the stack. Ordered by **when the topic is first needed**, not by
difficulty — the point is to have learned something shortly before using it, not months
before or a week after.

**Lead** marks who needs the topic in depth. The other member should still know enough to
review a pull request in that area (see `decisions.md` DECIDE-6), which is roughly the
"skim" column.

| When needed | Topic | Lead | Blocks |
|---|---|---|---|
| Now (September) | arc42 documentation | Both | 1.2, 1.3 |
| Now (September) | Semantic Scholar & arXiv APIs | My | 2.2 |
| October | PyMuPDF | My | 2.3 |
| October | Knowledge Graph concepts | My | 2.1 |
| October | Neo4j & Cypher | My (Hiếu: skim) | 2.1, 3.2 |
| November | spaCy NER & LLM extraction | My | 3.1 |
| November | Embeddings & FAISS | Hiếu (My: skim) | 3.3 |
| November | BM25 | Hiếu | 3.3 |
| December | Retrieval evaluation metrics | Both | 4.1 |

Depth to aim for: enough to make the design decisions in
[`decisions.md`](decisions.md) and to debug what breaks. Not a course completion.

---

## September

### arc42 — documenting requirements and architecture

**<https://arc42.org/overview>** — the template itself, with the section-by-section
explanation. Free, and short enough to read in one sitting.

arc42 is a 12-section template for architecture documents. You do not have to use all 12 —
for this project, sections 1 (requirements and goals), 3 (scope and context), 5 (building
block view), 6 (runtime view) and 9 (design decisions) carry almost everything. The
"decision record" idea in [`decisions.md`](decisions.md) is arc42 section 9.

What to take from it: an architecture document is not a diagram with captions. It states
what each component is responsible for, what crosses the boundary between components, and
which alternatives were rejected and why. The last part is what gets asked about in a
defence.

### Semantic Scholar API

**<https://api.semanticscholar.org/api-docs/>** — endpoint reference.

Read the Graph API section. Three things matter for task 2.2:

- **`/paper/search/bulk`** returns up to 1,000 results per call with a continuation token.
  This is what the collection spike used.
- **`/paper/batch`** takes up to 500 paper IDs in one POST and is how reference lists get
  fetched. This is the endpoint that gets rate-limited hardest without an API key.
- **The `fields` parameter** controls what comes back. Nested fields use dot notation
  (`references.paperId`, `authors.name`). Asking for fewer fields is faster and less likely
  to be throttled.

Rate limits apply per IP across all unauthenticated users, so a 429 is often somebody
else's traffic, not yours. Back off and retry rather than assuming your request is wrong
(`requirements.md` NFR-11).

### arXiv API

**<https://info.arxiv.org/help/api/user-manual.html>** — user manual.

The `arxiv` Python package in `requirements.txt` wraps this, so the manual matters mainly
for two things: the **query syntax** (field prefixes like `ti:`, `abs:`, `cat:`, and how to
combine them) and the **rate limit** — arXiv asks for roughly one request every 3 seconds,
which the package handles if you set `delay_seconds` and do not go around it.

arXiv PDFs download reliably, which is why `decisions.md` DECIDE-1 resolves PDF access
through arXiv first.

---

## October

### PyMuPDF — PDF text extraction

**<https://pymupdf.readthedocs.io/en/latest/the-basics.html>** — start here, then the
`Page.get_text()` reference.

`get_text()` has several output modes and the choice matters. `"text"` gives plain reading
order. `"dict"` / `"blocks"` give position and font information for each span — which is
how you detect headers, footers and section headings, since those are usually distinguished
by font size or position rather than by any marker in the text.

The hard parts of task 2.3 are not the API:

- **Two-column layouts.** Reading order can interleave columns. Check early on a real paper.
- **Hyphenation.** Words split across lines need rejoining, but not every trailing hyphen is
  a line break (e.g. "state-of-the-art").
- **Where the references section starts.** Needed to avoid indexing the bibliography as if
  it were content.

The collection spike found no scanned PDFs in the sample, so OCR is out of scope
(`requirements.md` §2.1, `S2`).

### Knowledge Graph concepts

**<https://web.stanford.edu/class/cs520/>** — Stanford CS520, *Knowledge Graphs*. Lecture
videos and slides are public.

Watch the lectures on **what a knowledge graph is** and **how knowledge graphs are created**.
The rest (embeddings of graphs, reasoning) is beyond this project.

The useful idea for task 2.1 is **identity**: deciding when two records are the same entity.
That is the whole difficulty of the graph. Two papers with the same title may be a preprint
and its published version; two authors with the same name may be different people; "BERT"
and "BERT-base" may or may not be one method. `requirements.md` FR-6 and FR-10 are both
identity problems.

### Neo4j and Cypher

**<https://graphacademy.neo4j.com/courses/cypher-fundamentals/>** — free, about 2 hours,
runs against a sandbox database in the browser.

Then **<https://neo4j.com/docs/cypher-manual/current/>** as reference.

The three things this project depends on:

- **`MERGE` vs `CREATE`.** `MERGE` matches an existing node or creates it if absent. Loaders
  must use it, or re-running a load duplicates everything (`requirements.md` FR-12).
- **Constraints and indexes.** A uniqueness constraint on the paper ID is what makes `MERGE`
  both correct and fast. Create them before loading, not after.
- **Variable-length patterns** (`-[:CITES*1..2]->`) for the neighbourhood queries in `UC-3`.

The database is already running locally — `make neo4j-up`, then <http://localhost:7474> for
the browser. `make check-neo4j` verifies the connection from Python.

---

## November

### spaCy NER and LLM-based extraction

**<https://course.spacy.io/en>** — free interactive course. Chapters 1–3 are enough.

Then **<https://spacy.io/usage/linguistic-features#named-entities>** for the NER specifics.

Expect the out-of-the-box models to underperform here, and plan for it: spaCy's pretrained
entity types are `PERSON`, `ORG`, `GPE` and similar. It does not know what a *method* or a
*dataset* is, because those categories do not exist in its training data. Running it on an
abstract will find institution names, not "contrastive learning" or "Natural Questions".

That is the reason task 3.1 pairs it with LLM extraction. Useful split: spaCy for the
entity types it was trained on, an LLM prompt returning structured JSON for the
domain-specific ones. `decisions.md` DECIDE-4 covers which model and what it costs.

For the LLM half, the topic to learn is **structured output** — constraining a model to
return valid JSON matching a schema, rather than parsing prose. Every major API supports
this directly; it is far more reliable than asking nicely in the prompt and repairing the
output afterwards.

### Embeddings and Sentence-Transformers

**<https://sbert.net/>** — documentation and usage examples.

Core idea: a sentence embedding maps text to a vector such that similar meanings land close
together, which is what lets search match "how to stop models making things up" against a
paper that only ever says "hallucination mitigation".

Four things to understand before task 3.3:

- **The input limit.** Most models truncate at 512 tokens. Longer input is silently cut —
  no warning. This is the constraint behind `decisions.md` DECIDE-2.
- **Symmetric vs asymmetric search.** Query and document are not the same kind of text. Some
  models (the `bge-*` family) expect a prefix on the query that documents do not get.
  Omitting it degrades results quietly.
- **Normalization and the similarity metric.** Normalize vectors and inner product becomes
  cosine similarity, which is what FAISS's `IndexFlatIP` then computes.
- **Model choice is permanent-ish.** Index-time and query-time embeddings must come from the
  same model; changing it means re-embedding everything (DECIDE-5).

**<https://huggingface.co/spaces/mteb/leaderboard>** — MTEB leaderboard, for comparing
models. Filter to retrieval tasks and English, and weigh model size against the CPU-only
constraint (`NFR-12`).

### FAISS — vector search

**<https://github.com/facebookresearch/faiss/wiki>** — the wiki, especially
*Guidelines to choose an index*.

For a corpus this size the answer is `IndexFlatIP` — exact, exhaustive search. At ~20,000
vectors it is fast enough, and it has no training step and no recall loss. The approximate
indexes (`IVF`, `HNSW`, `PQ`) exist for corpora orders of magnitude larger and trade recall
for speed you do not need here.

Worth knowing they exist, and worth a sentence in the report explaining why an exact index
is the right choice at this scale — that is a design decision, not an omission.

The one piece of bookkeeping: FAISS returns **row numbers**, not your IDs. Keep a separate
list mapping row number to chunk ID and persist it alongside the index, or search results
cannot be traced back to papers (`NFR-6`).

### BM25

**<https://github.com/dorianbrown/rank_bm25>** — the package. The README is short and the
API is three methods.

For the concept, the readable reference is Robertson & Zaragoza, *The Probabilistic
Relevance Framework: BM25 and Beyond* (2009) — the sections on term frequency saturation
and length normalization are the ones worth reading.

What matters in practice:

- **Tokenization is yours to do.** `rank_bm25` takes pre-tokenized lists, not raw strings.
  Lowercasing, punctuation and stopword decisions are all yours, and they affect results.
  Use the same tokenization at index time and query time.
- **`k1` and `b`.** Term-frequency saturation and length normalization. Defaults are fine to
  start; knowing what they do is enough.
- **Why BM25 is still in the system.** It handles exact matches — dataset names, model
  names, acronyms — that embeddings blur together. This complementarity is the argument for
  hybrid retrieval in Phase 2, and the Phase 1 baseline is what demonstrates it.

---

## December

### Retrieval evaluation

Needed for task 4.1. No course required, but the definitions must be exact, because the
Phase 1 baseline numbers are what Phase 2 gets compared against.

- **Recall@k** — of all relevant documents, how many appear in the top k.
- **MRR** (Mean Reciprocal Rank) — 1/rank of the first relevant result, averaged. Cares only
  about the first hit.
- **nDCG@k** — rewards relevant results appearing higher, and supports graded relevance
  rather than binary.

Which to report depends on the use case: `UC-1` (browse a topic) is a Recall@10 question;
`UC-2` (find the paper using dataset X) is an MRR question. Say which and why.

**<https://trec.nist.gov/>** has the standard definitions if a formal citation is wanted.

The harder half is building the test set: 20–30 queries with relevant papers marked by hand.
Both members should judge an overlapping subset so the judgements can be checked against
each other — same reasoning as `decisions.md` DECIDE-3.

---

## Notes

Links were correct when written (2026-09-23). If one has moved, the search terms in the
heading will find it.

When something here turns out to be wrong or missing in practice, edit this file. It is
meant to be the path actually taken, which is also what the Phase 1 report needs to describe.
