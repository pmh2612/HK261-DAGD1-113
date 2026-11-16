# BT8 — Hybrid search and rerank (SciFact sanity check)

Run `20261104-134634`, config [`bt8_hybrid.yaml`](../../configs/bt8_hybrid.yaml), script
[`run_hybrid.py`](../../scripts/run_hybrid.py), fusions in
[`retrieval/hybrid.py`](../../src/paper_search/retrieval/hybrid.py). One run directory per
system in [`experiments/runs/bt8/`](../runs/bt8/) with config, metrics and per-query nDCG@10.

**On our own queries: not run yet**, for the same reason as BT6/BT7. **KG-augmented search
(plan method 3): not run** — it needs the BT5 knowledge graph.

## Setup

- **Same evaluation as BT6/BT7:** SciFact, 5,183 abstracts, 300 test queries; anything
  tuned is tuned on the 809 train queries. Paired t-test on per-query nDCG@10.
- **Components:** the systems BT6/BT7 carried forward — BM25 (title ×2 + abstract,
  stopwords + stemming), bge-base, bge-small. Rerun in the same run, and they reproduce
  BT6/BT7 exactly (0.6915, 0.7421, 0.7057).
- **Fusions:**
  - **RRF** — sum of 1 / (60 + rank) over the two retrievers; ranks only, no tuning.
  - **Linear** — α · dense + (1 − α) · BM25, each min-max normalized per query; α from
    0 to 1 in steps of 0.1, chosen on train.
  - **Rerank** — the top 50 re-ordered by a cross-encoder:
    `cross-encoder/ms-marco-MiniLM-L6-v2` (the plan's example; the hub now redirects the
    old `…MiniLM-L-6-v2` name here) and `BAAI/bge-reranker-base`.

## Results

| System | nDCG@10 | R@10 | R@100 | MRR@10 | ms / query | p vs bge-base |
|---|---:|---:|---:|---:|---:|---:|
| **Linear BM25 + bge-base (α = 0.9)** | **0.763** | **0.890** | **0.973** | **0.729** | 12 | **0.0015** |
| bge-base alone | 0.742 | 0.871 | 0.961 | 0.707 | 2.8 | — |
| RRF BM25 + bge-base | 0.742 | 0.872 | 0.970 | 0.707 | 5.9 | 0.98 |
| Linear BM25 + bge-small (α = 0.8) | 0.740 | 0.871 | 0.972 | 0.703 | 9.9 | 0.84 |
| RRF BM25 + bge-small | 0.728 | 0.861 | 0.962 | 0.693 | 4.0 | 0.28 |
| RRF + rerank bge-reranker-base | 0.724 | 0.858 | 0.970 | 0.688 | 1,754 | 0.25 |
| bge-small alone | 0.706 | 0.830 | 0.943 | 0.673 | 0.9 | 0.0008 |
| bge-base + rerank MiniLM | 0.697 | 0.827 | 0.961 | 0.664 | 309 | 0.0021 |
| RRF + rerank MiniLM | 0.693 | 0.829 | 0.970 | 0.658 | 296 | 0.0010 |
| BM25 alone | 0.692 | 0.819 | 0.926 | 0.656 | 2.5 | 0.0026 |

Query time includes the components a system needs. Index build: BM25 0.3 s, bge-small
49 s, bge-base 135 s for 5,183 abstracts.

Other paired comparisons:

| Comparison | ΔnDCG@10 | p |
|---|---:|---:|
| Linear BM25 + bge-small vs bge-small | +0.034 | 0.0012 |
| Linear BM25 + bge-base vs RRF BM25 + bge-base | +0.021 | 0.021 |
| RRF BM25 + bge-small vs bge-small | +0.022 | 0.065 |
| RRF + bge-reranker vs RRF | −0.017 | 0.20 |
| RRF + MiniLM rerank vs RRF | −0.049 | 0.0008 |

Train nDCG@10 by α (BM25 + bge-base): 0.700 at α = 0 (BM25 only), rising steadily to
0.792 at 0.9, then 0.771 at 1.0 (bge-base only). With bge-small the peak is at 0.8.

## Findings

- **Linear fusion is the only method that beats bge-base:** +0.021 (p = 0.0015), and it
  improves every metric, Recall@100 included (0.961 → 0.973). The curve over α is smooth
  with one peak, so the choice of α is not a lucky grid point.
- **A little BM25 goes a long way.** The best mix is 90% dense, 10% BM25: BM25 alone is
  0.05 behind, yet adding a small share of its exact-term matching still helps. At α = 0.5
  (equal weights) the mix is worse than bge-base alone on train (0.753 vs 0.771).
- **RRF does not help here.** It gives both retrievers equal say by rank, and BM25 is
  the clearly weaker one, so RRF ends up level with bge-base (p = 0.98). It is not a
  safe default when one retriever is much weaker than the other.
- **BM25 + bge-small reaches bge-base** (0.740 vs 0.742, p = 0.84) with an index 2.7×
  faster to build. A cheaper option at `medium` if build time matters.
- **Rerankers make things worse.** MiniLM, trained on MS MARCO web questions, drops
  nDCG@10 by 0.045–0.049 (p ≤ 0.0021): scientific claims are not web queries.
  bge-reranker-base also loses (−0.017, not significant) and costs 1.75 s per query,
  about 150× the linear fusion.

## Caveats

- **SciFact again, not our queries.** α = 0.9 is tuned on SciFact claims. Our 50 queries
  are too few to tune α on safely; plan to keep 0.9, or check a coarse grid with
  cross-validation, and report both.
- **One general-purpose reranker each,** at depth 50. A reranker trained on scientific
  text could behave differently; not tried.
- Fusion operates on full score matrices (every query × every document). Fine at
  5,000–20,000 abstracts; a larger corpus would fuse only each retriever's top few hundred.

## Proposal

- **Main search: linear BM25 + bge-base, α = 0.9.** To be confirmed on our own queries.
- **Cheap fallback: linear BM25 + bge-small**, if bge-base's build time becomes a problem
  at `medium`.
- **Drop cross-encoder rerank** for now; revisit only with a scientific-domain reranker.
- **KG-augmented expansion** (plan method 3) next, once BT5 has a graph on the new corpus:
  candidates from papers sharing an entity with the query, fused into the linear mix.
