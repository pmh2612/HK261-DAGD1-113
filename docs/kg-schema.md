# Knowledge Graph schema

Scientific Paper Search System (Knowledge Graph + RAG) · Roadmap task 2.1

**Status:** v1, 2026-10-23 · implemented in
[`src/paper_search/graph/loader.py`](../src/paper_search/graph/loader.py), loaded by
[`scripts/build_graph.py`](../scripts/build_graph.py)

Covers `FR-11` and `FR-12`. Node and relationship types are the ones `FR-11` fixes; this
document adds properties, keys, and the queries the graph has to answer.

---

## 1. Diagram

```mermaid
flowchart LR
    A[Author] -- AUTHORED<br/>position --> P[Paper]
    P -- CITES --> P2[Paper / :External]
    P -- PUBLISHED_IN --> V[Venue]
    P -- USES_METHOD<br/>extractor --> M[Method]
    P -- USES_DATASET<br/>extractor --> D[Dataset]
    P -- HAS_TOPIC<br/>extractor --> T[Topic]
```

## 2. Nodes

Every node type has exactly one uniqueness constraint, on its key. Loaders `MERGE` on
that key, so re-running a load never duplicates anything (`FR-12`).

| Label | Key (unique) | Other properties | Source |
|---|---|---|---|
| `Paper` | `paper_id` (Semantic Scholar ID) | `title`, `abstract`, `year`, `doi`, `arxiv_id`, `citation_count`, `has_full_text`, `data_version` | collection |
| `Author` | `author_id` | `name` | collection |
| `Venue` | `key` | `name` | collection |
| `Method` | `key` | `name` | extraction |
| `Dataset` | `key` | `name` | extraction |
| `Topic` | `key` | `name` | extraction |

**`:External`** is a second label on a `Paper` node, never a separate node type
(`DECIDE-1`). It marks a reference cited by ≥ 3 seed papers that was pulled in for the
graph only: no chunks, never returned by search (invariant P5). Because it is still a
`:Paper`, the `Paper.paper_id` constraint covers it — a paper first collected as a
reference and later as a seed is the same node, and the loader removes the label.

### Keys

| Node | Key | Why |
|---|---|---|
| `Paper` | Semantic Scholar ID | One ID space for seeds, references and citations. DOI and arXiv ID are kept as properties; they drive deduplication at collection (`FR-6`), not graph identity |
| `Author` | Semantic Scholar author ID, else `"name:" + lowercased name` | Names collide ("Y. Wang"). The name fallback exists only for the few authors S2 has no ID for, and can merge two people — accepted for v1 |
| `Venue` | lowercased, whitespace-collapsed venue string | S2 venue strings are already mostly canonical |
| `Method` / `Dataset` / `Topic` | `normalize_entity(name)` | See §4 |

## 3. Relationships

| Type | From → To | Properties | Notes |
|---|---|---|---|
| `AUTHORED` | `Author` → `Paper` | `position` (0 = first author) | |
| `CITES` | `Paper` → `Paper` | | Only between papers in the graph. References to anything else are dropped — the graph does not hold stub nodes |
| `PUBLISHED_IN` | `Paper` → `Venue` | | Absent when S2 has no venue |
| `USES_METHOD` | `Paper` → `Method` | `extractor` | |
| `USES_DATASET` | `Paper` → `Dataset` | `extractor` | |
| `HAS_TOPIC` | `Paper` → `Topic` | `extractor` | |

**`extractor` on every entity relationship** records which model asserted it. Extraction
is the least reliable stage and is going to be re-run when `DECIDE-7` lands; this is what
makes a rebuild traceable instead of a silent change to the graph.

**`CITES` is thin by construction** (constraint C2): at 1,000 seeds only a small share of
references land inside the corpus. The entity relationships are where the connectivity
comes from, which is why §5's neighbourhood query goes through them too.

## 4. Entities

**What counts as `Topic`.** `FR-9` extracts methods, datasets, tasks and topics, while
`FR-11` has no `Task` node. v1 stores research *tasks* ("open-domain question answering",
"fact verification") as `Topic` — they are what a user means by "papers on X". Broader
fields of study would add little: nearly every paper in a single-topic corpus would share
them.

**Normalization (`FR-10`).** `normalize_entity` lowercases, turns `- _ /` into spaces,
collapses whitespace, strips surrounding punctuation, and drops size/variant suffixes
(`base`, `large`, `small`, `xl`, `uncased`, …). So *BERT-base*, *bert* and *BERT* are one
`Method`; *Self-RAG* and *Self RAG* are one. The first surface form seen becomes `name`.

What it does **not** do: merge an acronym with its expansion (*DPR* vs *dense passage
retriever*) or catch synonyms. That fragmentation is real and will show in the graph
statistics; alias merging is the next step if it hurts the target queries.

**Extractor in v1:** GLiNER zero-shot NER (`urchade/gliner_medium-v2.1`) on the abstract.
`FR-9` specifies an in-process LLM with schema-constrained decoding; that waits on
`DECIDE-7` and on the comparison in
[`notebooks/exp_extraction.py`](../notebooks/exp_extraction.py). GLiNER is a stand-in,
chosen because it is fast enough to run over the whole corpus on a laptop.

## 5. Target queries

The queries the graph has to answer. `scripts/build_graph.py` runs each one after loading,
with parameters picked so the result is known to be non-empty. Keep the two in sync.

**UC-2 — papers using dataset X with method Y.** The query that justifies having a graph:
no text search can answer it reliably.

```cypher
MATCH (:Dataset {key: $dataset})<-[:USES_DATASET]-(p:Paper)
      -[:USES_METHOD]->(:Method {key: $method})
WHERE NOT p:External
RETURN p.title AS title, p.year AS year ORDER BY p.citation_count DESC LIMIT 5
```

**UC-3 — a paper's neighbourhood**, grouped by how each neighbour is connected.

```cypher
MATCH (p:Paper {paper_id: $paper_id})
CALL (p) {
    MATCH (p)-[:CITES]->(o:Paper) RETURN 'cites' AS rel, o
    UNION MATCH (p)<-[:CITES]-(o:Paper) RETURN 'cited by' AS rel, o
    UNION MATCH (p)-[:USES_METHOD|USES_DATASET]->()<-[:USES_METHOD|USES_DATASET]-(o:Paper)
          WHERE NOT o:External RETURN 'shares method/dataset' AS rel, o
    UNION MATCH (p)<-[:AUTHORED]-(:Author)-[:AUTHORED]->(o:Paper)
          WHERE o <> p RETURN 'same author' AS rel, o
}
RETURN rel, count(DISTINCT o) AS papers ORDER BY rel
```

**Most-used datasets** — topic exploration (`FR` graph exploration, Phase 2 UI).

```cypher
MATCH (d:Dataset)<-[:USES_DATASET]-(p:Paper)
RETURN d.name AS dataset, count(p) AS papers ORDER BY papers DESC LIMIT 5
```

**Methods used for a topic** — the starting point for `UC-7` method comparison.

```cypher
MATCH (t:Topic {key: $topic})<-[:HAS_TOPIC]-(p:Paper)-[:USES_METHOD]->(m:Method)
RETURN m.name AS method, count(p) AS papers ORDER BY papers DESC LIMIT 5
```

Search (Phase 2 `FR-17`) must exclude `:External` papers from anything it returns as a
result: filter with `WHERE NOT p:External`, as above.

## 6. KG v1 statistics

_Filled in from the `scripts/build_graph.py` output once v1 is loaded._
