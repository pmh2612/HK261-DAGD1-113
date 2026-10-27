# BT6 + BT7 — Keyword and semantic search (SciFact sanity check)

Run `20261027-154840`, configs [`bt6_bm25.yaml`](../../configs/bt6_bm25.yaml) and
[`bt7_dense.yaml`](../../configs/bt7_dense.yaml), script
[`scripts/run_search.py`](../../scripts/run_search.py). One run directory per system in
[`experiments/runs/bt6/`](../runs/bt6/) and [`bt7/`](../runs/bt7/), each with its config,
metrics and per-query nDCG@10.

**On our own `tiny` set: not run yet** — the queries and judgments do not exist. They wait
for the supervisor's answer on cross-listed papers, which decides what `tiny` contains
(BT1 report). Everything below is on SciFact.

## Setup

- **SciFact** (BEIR): 5,183 biomedical abstracts, 300 test queries (scientific claims)
  with judgments; 809 train queries. Public numbers exist for it, so it checks our code
  as well as the methods.
- **No tuning on test.** BM25 field and preprocessing are chosen, and k1 / b tuned, on the
  train queries; every number here is on the 300 test queries.
- **Significance:** paired t-test on per-query nDCG@10 (300 pairs). No correction for
  multiple comparisons, so a p near 0.05 is weak evidence.
- Metrics: nDCG@10 (main), Recall@10, Recall@100, MRR@10; index build time and query
  latency on an Apple-silicon laptop (MPS), queries encoded in one batch.

## BT6 — BM25

| Field \ preprocessing | lower | stop | stem | stop + stem |
|---|---:|---:|---:|---:|
| title | 0.411 | 0.426 | 0.451 | 0.466 |
| abstract | 0.636 | 0.646 | 0.667 | 0.677 |
| title + abstract | 0.652 | 0.670 | 0.678 | 0.683 |
| title ×2 + abstract | 0.659 | 0.674 | 0.686 | **0.692** |

nDCG@10, `rank_bm25` BM25Okapi, k1 = 1.5, b = 0.75.

- **Stemming + stopword removal is the one change that clearly matters:** +0.040 over
  plain lowercase on title ×2 + abstract (p = 0.0005).
- **Weighting the title ×2 helps a little:** +0.008 over title + abstract, not
  significant (p = 0.07).
- **Title alone is far worse** (0.41–0.47): the abstract carries the evidence. This backs
  the decision to work from abstracts.
- **Tuning k1 / b did not transfer.** The train grid picked k1 = 2.0, b = 0.5 (train
  nDCG@10 0.702), which scores 0.686 on test, −0.005 against the defaults (p = 0.28).

**Chosen: title ×2 + abstract, stopwords removed, Snowball stemming, default k1 / b.**
nDCG@10 0.692 against 0.665 published for BM25 (Anserini) on SciFact, consistent with the
stemming gain. The `rank_bm25` vs `bm25s` speed comparison waits for `medium`, as planned:
BM25 here takes 0.4 s to index and 3 ms per query.

## BT7 — Dense embeddings

| Model | nDCG@10 | R@10 | R@100 | MRR@10 | Index 5,183 docs | ms / query | p vs bge-base |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BAAI/bge-base-en-v1.5` | **0.742** | **0.871** | **0.961** | **0.707** | 146 s | 2.7 | — |
| `BAAI/bge-small-en-v1.5` | 0.706 | 0.830 | 0.943 | 0.673 | 50 s | 1.0 | 0.0008 |
| `intfloat/e5-small-v2` | 0.696 | 0.815 | 0.931 | 0.666 | 47 s | 1.1 | 0.0014 |
| *BM25 (chosen above)* | 0.692 | 0.819 | 0.926 | 0.656 | 0.4 s | 3.0 | 0.0026 |
| `sentence-transformers/all-MiniLM-L6-v2` | 0.648 | 0.788 | 0.925 | 0.607 | 20 s | 0.6 | < 0.0001 |
| `allenai/specter2` (+ ad-hoc query adapter) | 0.646 | 0.774 | 0.960 | 0.610 | not measured¹ | not measured¹ | < 0.0001 |

¹ SPECTER2 is encoded in a separate venv (`adapters` conflicts with the main one), and
that run was not timed; its 0.0 s in the run files is only loading saved vectors.

- **bge-base is the best system, and the gap to every other is significant** (p ≤ 0.003).
  Its cost is 3× bge-small to index: 28 ms per abstract, so ~10 min for `medium`.
- **bge-small and e5-small are no better than BM25** (+0.014, p = 0.40;
  +0.004, p = 0.82).
- **MiniLM and SPECTER2 are worse than BM25** (−0.043, p = 0.018; −0.046, p = 0.017).
  SPECTER2 was trained for paper-to-paper similarity, not short queries. Its Recall@100
  (0.960) matches bge-base, so it still finds the right papers, it ranks them lower.
- These agree with published SciFact numbers (bge-base ≈ 0.74, MiniLM ≈ 0.65), so the
  pipeline (prefixes, normalization, FAISS) is set up correctly.

## Caveats

- **SciFact is biomedical claim verification, not CS paper search.** Queries are claims,
  not "papers about X". The ranking of methods may change on our own queries; this is a
  sanity check, and the choice is confirmed or overturned on `tiny` and `small`.
- **DECIDE-5 pinned `bge-small-en-v1.5`.** On SciFact `bge-base` beats it by 0.036
  (p = 0.0008). Not changed until our own set agrees.

## Proposal for the next round

Carry three systems into `tiny` / `small`: **bge-base**, **BM25 (title ×2 + abstract,
stop + stem)** and **bge-small** (the current DECIDE-5 pin, a third of bge-base's cost).
These three also feed BT8 hybrid search: BM25 and bge-base disagree on enough queries
for fusion to be worth testing.

**Pool size for judging our queries.** Pooling the top 20 of all 22 systems gave 47–57
candidates per query in a smoke test on `tiny`, about 2,500 judgments for 50 queries.
Pooling only the systems carried forward, plus MiniLM and SPECTER2 for diversity, should
roughly halve that. To decide when the queries are written.
