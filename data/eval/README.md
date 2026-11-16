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

Label with `notebooks/label.py` rather than by editing the JSONL files: it saves after
every finished item, writes atomically, and refuses entity text that is not in the
abstract verbatim.

```bash
L=".venv/bin/python notebooks/label.py"

# Queries (target 20-30; five seeds from requirements.md §5)
$L add "graph-based retrieval augmented generation"      # 1. add queries
.venv-specter2/bin/python notebooks/encode_specter2.py   # 2. after adding queries
.venv/bin/python notebooks/exp_search.py --pool          #    rebuild the judging pools
$L queries                                               # 3. y / n per candidate
.venv/bin/python notebooks/exp_search.py                 # 4. score

# Entities - agreement round first, then split the rest
cp data/eval/entities.jsonl data/eval/entities.hieu.jsonl
cp data/eval/entities.jsonl data/eval/entities.my.jsonl
$L entities --file data/eval/entities.hieu.jsonl         # each annotator, first 10 abstracts
$L entities --file data/eval/entities.my.jsonl
$L agree data/eval/entities.hieu.jsonl data/eval/entities.my.jsonl
# fix the guidelines below where you disagreed, merge into entities.jsonl, then
$L entities                                              # the remaining abstracts
.venv/bin/python notebooks/exp_extraction.py             # score
```

Entity input is one per line: `m: BERT`, `d: HotpotQA`, `t: question answering`. An empty
line finishes the abstract, `none` marks one with no entities, `q` quits.

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
