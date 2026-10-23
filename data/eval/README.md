# Evaluation set

Hand-labelled data for the search and entity-extraction experiments. Tracked in git,
unlike the rest of `data/`: the labels are human work and the scores mean nothing
without the exact version they were computed against.

| File | Written by | Contents |
|---|---|---|
| `abstracts.jsonl` | `notebooks/collect_abstracts.py` | 200 most-cited Semantic Scholar papers for "retrieval augmented generation" with an abstract of ≥ 300 characters |
| `queries.jsonl` | seeded by the script, then **people** | `{"qid", "query", "relevant": [paper_id, ...] \| null}` |
| `pool.jsonl` | `notebooks/exp_search.py --pool` | per query, the union of every system's top 10 — the list to judge |
| `entities.jsonl` | sampled by the script, then **people** | 50 abstracts, `"entities": [{"text", "type"}] \| null` |

`null` means "not labelled yet" and is skipped by the scorers. `[]` means "labelled, and
there is nothing" — do not use it as a placeholder.

Do not re-run `collect_abstracts.py --force` once labelling has started: the corpus
changes and every label pointing into it goes stale.

## Workflow

```bash
# 1. add queries to queries.jsonl (target 20-30; five seeds from requirements.md §5)
.venv-specter2/bin/python notebooks/encode_specter2.py   # after every edit to queries
.venv/bin/python notebooks/exp_search.py --pool          # 2. build the judging pools
# 3. judge pool.jsonl, copy relevant paper_ids into queries.jsonl
.venv/bin/python notebooks/exp_search.py                 # 4. score

# 5. annotate entities.jsonl, then
.venv/bin/python notebooks/exp_extraction.py
```

## Judging queries

A paper is **relevant** if a researcher typing that query would want it in the top 10,
judged from title and abstract alone. Judge every candidate in the pool. Papers outside
the pool are treated as not relevant — that is the standard pooling assumption and the
reason every system contributes to the pool.

## Annotating entities

Label only what the abstract **names**, copied exactly as written:

| Type | Label | Do not label |
|---|---|---|
| `Method` | named models, systems, algorithms, techniques: *BERT*, *DPR*, *Self-RAG*, *BM25*, *retrieval-augmented generation* | generic phrases: *our method*, *a novel framework*, *large language models* |
| `Dataset` | named datasets and benchmarks: *Natural Questions*, *HotpotQA*, *MS MARCO* | *a new dataset*, *several benchmarks* |
| `Task` | research tasks: *open-domain question answering*, *fact verification*, *summarization* | goals and properties: *efficiency*, *factuality* |

- An entity mentioned several times is listed once.
- Long form and acronym (*Retrieval-Augmented Generation (RAG)*) → two entries, one each.
- The paper's own new system counts as a `Method`.

Both members should annotate the same 10 abstracts first and compare; disagreements there
are guideline gaps, so fix this table before splitting the rest.
