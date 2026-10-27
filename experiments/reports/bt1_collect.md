# BT1 — Data collection (`tiny`)

Run `20261027-112046`, config [`configs/bt1_collect.yaml`](../../configs/bt1_collect.yaml),
script [`scripts/bt1_collect.py`](../../scripts/bt1_collect.py). Metrics per source in
[`experiments/runs/bt1/`](../runs/bt1/). The metrics files come from a re-run served from
the cache; collection times are from the first, uncached run
([`20261027-112046_network.log`](../runs/bt1/20261027-112046_network.log)).

## Setup

- **Sampling frame:** every arXiv ID listed under `cs.IR`, `cs.CL` or `cs.LG` and
  submitted January–June 2024, cross-lists included: 32,152 listings, **28,330 unique
  papers**. Sorted, shuffled once with seed 42; `tiny` is the first 200, `small` the first
  2,000, `medium` the first 20,000, so `tiny ⊂ small ⊂ medium`.
  Three months (13,536 papers) were too few for `medium`, hence six.
- **arXiv:** the search API (`export.arxiv.org`) returned 429 to every request on
  2026-10-27, so the frame comes from the `/list` pages (allowed by robots.txt, 15 s apart)
  and metadata from OAI-PMH `GetRecord` (3 s apart).
- **Semantic Scholar:** `/paper/batch` looked up by `ARXIV:<id>`, with the API key.
- **Combined:** arXiv record as the base (title, abstract, categories), S2 adds publication
  year, venue, author IDs, DOI, citation count and fields of study
  (`arxiv_oai.merge`).

## Results

Share of the 200 sampled papers that have each field:

| Field | (a) S2 only | (b) arXiv only | (c) combined |
|---|---:|---:|---:|
| abstract | 1.00 | 1.00 | 1.00 |
| year | 1.00 | 1.00 | 1.00 |
| venue | 0.675 | 0.08¹ | 0.68 |
| authors | 1.00 | 1.00 | 1.00 |
| author IDs | 1.00 | 0.00 | 1.00 |
| DOI | 0.945 | 0.09 | 0.945 |
| citation count | 1.00 | 0.00 | 1.00 |
| fields of study | 1.00 | 0.00 | 1.00 |
| arXiv categories | 0.00 | 1.00 | 1.00 |
| mean abstract length (words) | 177.7 | 177.1 | 177.1 |
| collection time, 200 papers | 21 s | 788 s | 809 s |

¹ arXiv's `journal-ref`, a free-text citation string, not a venue name.

Agreement between the two sources:

| | |
|---|---:|
| sampled papers S2 resolves by arXiv ID | 200 / 200 |
| titles equal after normalization | 97.0% (6 differ) |
| DOI conflicts | 0 |
| S2 has no abstract | 0 |
| abstracts identical (similarity ≥ 0.95) | 90.5% (19 differ) |
| median abstract similarity | 1.00 |
| duplicate titles in the sample | 0 |

## Error analysis

1. **S2 gives "arXiv.org" as the venue for 59 of 200 papers** (29.5%). That is where the
   preprint lives, not a venue, and would become one `Venue` node holding a third of the
   graph. Now normalized to no venue; real venue coverage is 67.5%, not 97%.
2. **OAI-PMH `<created>` is not the v1 date.** `2401.06344` has `created` 2026-03-20; 33
   of 200 papers got a 2025 or 2026 year from it. Year is now taken from the arXiv ID
   (`2401` → 2024).
3. **Title differs where the published version was renamed:** `2403.08002` *Towards a
   clinically accessible radiology foundation model…* vs *A clinically accessible small
   multimodal radiology model…*; `2405.10780` adds/drops *and Miniaturized*;
   `2405.06361` LaTeX `$\ell_2$` vs Unicode `ℓ2`.
4. **Abstract differs for the same reason (19 papers).** `2405.10780` similarity 0.02 and
   `2403.08002` 0.05: S2 holds the journal abstract, arXiv the preprint's. The combined
   record keeps the arXiv text.
5. **S2 year is the publication year:** 6 papers are earlier than their arXiv year, e.g.
   `2401.10355` (2019, *Mechanical Systems and Signal Processing*) posted to arXiv in 2024.
   The combined record keeps the S2 year.
6. **Cross-lists bring in other fields.** Only 123 of 200 papers have `cs.IR`, `cs.CL` or
   `cs.LG` as their *primary* category; the rest are primarily `cs.CV` (21), `stat.ML` (7),
   `eess.SP` (6) and others.

## Conclusion

**(c) combined.** Neither source is enough alone: arXiv has no author IDs, citations,
venues or fields of study (needed by BT5); S2 has no arXiv categories and in ~10% of
cases holds a different (published) abstract. Matching by arXiv ID is exact: 200/200, no
DOI conflicts.

## Before `small` — decisions for the team

1. **Keep cross-lists whose primary category is outside the three?** (77/200 now.)
   Keeping them gives a broader corpus; dropping them makes labelling easier. Restricting
   to primary would change the frame, so it has to be decided before `small`.
2. **arXiv OAI is the bottleneck: 3.9 s per paper** — `small` ≈ 2.2 h, `medium` ≈ 22 h.
   S2 takes ~1 min for `medium`. Options: accept it (it runs unattended and caches), or
   take abstracts from S2 and only categories from arXiv (the `/list` pages already show
   them). The second trades ~10% preprint abstracts for published ones; not tested yet.
