# BT1 — Data collection (`tiny`)

Runs `20261107-164502` (`tiny`) and `20261111-065139` (`small`, section below), config [`configs/bt1_collect.yaml`](../../configs/bt1_collect.yaml),
script [`scripts/bt1_collect.py`](../../scripts/bt1_collect.py). Metrics per source in
[`experiments/runs/bt1/`](../runs/bt1/), from an uncached run, so collection times are real.

**Re-run on 2026-11-07 after the supervisor dropped cross-lists** (decision in
[`docs/decisions.md`](../../docs/decisions.md)). The first run, `20261027-112046`, sampled a
frame with cross-lists; its numbers are kept at the end of this report.

## Setup

- **Sampling frame:** every arXiv ID listed under `cs.IR`, `cs.CL` or `cs.LG`, submitted
  January–July 2024, whose **primary** category is one of the three: 32,939 listed papers,
  **21,808 in the frame**. Sorted, shuffled once with seed 42; `tiny` is the first 200,
  `small` the first 2,000, `medium` the first 20,000, so `tiny ⊂ small ⊂ medium`.
  - The primary category comes from the `/list` pages themselves (`primary-subject`), so
    filtering the frame costs no extra OAI calls.
  - Primary-only keeps 66% of listed papers: six months give 18,805, too few for `medium`,
    hence July was added.
  - The new listings were checked against the first run's: the same IDs for every
    category and month of January–June.
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
| venue | 0.67 | 0.125¹ | 0.67 |
| authors | 1.00 | 1.00 | 1.00 |
| author IDs | 1.00 | 0.00 | 1.00 |
| DOI | 0.945 | 0.125 | 0.945 |
| citation count | 1.00 | 0.00 | 1.00 |
| fields of study | 1.00 | 0.00 | 1.00 |
| arXiv categories | 0.00 | 1.00 | 1.00 |
| mean abstract length (words) | 169.9 | 170.6 | 170.6 |
| collection time, 200 papers | 2 s | 2,979 s | 2,981 s |

¹ arXiv's `journal-ref`, a free-text citation string, not a venue name.

Agreement between the two sources:

| | |
|---|---:|
| sampled papers S2 resolves by arXiv ID | 200 / 200 |
| titles equal after normalization | 97.0% (6 differ) |
| DOI conflicts | 0 |
| S2 has no abstract | 0 |
| abstracts identical (similarity ≥ 0.95) | 91.5% (17 differ) |
| median abstract similarity | 1.00 |
| duplicate titles in the sample | 0 |

Primary category of the 200: `cs.LG` 106, `cs.CL` 83, `cs.IR` 11. 128 are also cross-listed
elsewhere. Fields of study (S2): all 200 Computer Science, then Mathematics 29,
Linguistics 25, Engineering 21, Medicine 12.

**arXiv was four times slower than in the first run: 14.9 s per paper** (3.9 s on
2026-10-27). Single records took up to 2 minutes, i.e. the client was backing off on
429s or timeouts; the run does not log which. The `/list` download the same day also
timed out once, which is why `_get` now retries on timeouts and dropped connections too.
At this rate `small` takes about 8 hours instead of 2.2.

## Error analysis

1. **S2 gives "arXiv.org" as the venue for 58 of 200 papers** (29%; 59 in the first
   sample). That is where the preprint lives, not a venue, and would become one `Venue`
   node holding a third of the graph. Normalized to no venue; real venue coverage is 67%.
2. **OAI-PMH `<created>` is not the v1 date** (found in the first sample: `2401.06344` has
   `created` 2026-03-20; 33 of 200 papers got a 2025 or 2026 year from it). Year is taken
   from the arXiv ID (`2401` → 2024); no paper in this sample is dated after 2024.
3. **Title and abstract differ where the published version was revised** (6 titles,
   17 abstracts): S2 holds the published text, arXiv the preprint's. The combined record
   keeps the arXiv text. Examples from the first sample: `2403.08002` renamed on
   publication, abstract similarity 0.05; `2405.06361` LaTeX `$\ell_2$` vs Unicode `ℓ2`.
4. **S2 year is the publication year:** 6 papers are earlier than their arXiv year (the
   first sample's `2401.10355`: 2019 journal paper posted to arXiv in 2024). The combined
   record keeps the S2 year.
5. **`cs.IR` is small:** 11 of 200 papers (5.5%), against 106 `cs.LG`. That is its share
   of the frame (1,278 of 21,808, 5.9%; `cs.CL` 41.4%, `cs.LG` 52.7%), not a sampling
   error, but queries about retrieval itself will have few relevant papers at `tiny`.

## Conclusion

**(c) combined.** Neither source is enough alone: arXiv has no author IDs, citations,
venues or fields of study (needed by BT5); S2 has no arXiv categories and in ~10% of
cases holds a different (published) abstract. Matching by arXiv ID is exact: 200/200, no
DOI conflicts.

## `small` (2,000 papers)

Run `20261111-065139`, same config with `--level small`; the 200 `tiny` papers come from
the cache (`tiny ⊂ small`).

| Field | (a) S2 only | (b) arXiv only | (c) combined |
|---|---:|---:|---:|
| abstract | 1.00 | 1.00 | 1.00 |
| venue | 0.677 | 0.085¹ | 0.68 |
| author IDs | 1.00 | 0.00 | 1.00 |
| DOI | 0.941 | 0.105 | 0.941 |
| citation count, fields of study | 1.00 | 0.00 | 1.00 |
| arXiv categories | 0.00 | 1.00 | 1.00 |
| collection time | 10 s | 12,263 s² | 12,273 s |

S2 resolves 2,000 / 2,000 by arXiv ID; titles agree 98.0%; no DOI conflicts; no duplicate
titles. Same picture as `tiny`, so the conclusion stands.

² Time the machine was awake (`time.monotonic`, which stops while macOS sleeps): **6.9 s
per uncached paper** (1,788 fetched, 212 cached). The run actually took 3.6 days of wall
time because the laptop slept with its lid closed; `caffeinate -i` only prevents idle
sleep. At 6.9 s, `medium` (18,000 more papers) needs about 35 hours of awake time.

### How far S2 abstracts differ from arXiv's (the measurement for option A vs B)

- **92.6% identical** (similarity ≥ 0.95); 147 papers (7.4%) differ. 146 of the 147 have
  a venue: S2 holds the published version, arXiv the preprint.
- **52 papers (2.6%) differ a lot** (similarity < 0.5). Judged from each S2
  abstract's opening lines against the paper's title, 49 are the same paper's abstract
  rewritten for publication (same topic and method, e.g.
  `2403.11020` *Accelerating prototype selection with spatial abstraction*).
- **3 are broken in S2 (0.15%):**
  - `2406.15038`: the abstract is an empty JATS tag, `<jats:p/>`.
  - `2405.04292`: *Mitigating Clickbait: … Spoiler Generation* carries the abstract of
    a different paper (abusive comments in Indic languages).
  - `2406.15291`: truncated to under 30 words.

  The combined record keeps arXiv's abstract, so none of these reaches the data at `small`.

### Proposal for `medium`

**Option B: abstracts from S2 for the 18,000 papers not yet fetched,** keeping the arXiv
abstracts already in the cache for the first 2,000.

- Under 3% of S2 abstracts are materially different, and those are the published
  version of the same paper, which is fine for search and extraction.
- The broken 0.15% can be caught by a check (empty or tag-only text, fewer than 30 words)
  and fetched from arXiv OAI instead; the wrong-paper case needs a title–abstract
  vocabulary check, to be tested on these 2,000 before relying on it.
- Primary category comes from the `/list` pages, which already show every paper's
  subjects, so no OAI call is needed for categories.
- Every record says where its abstract came from, so the mix stays traceable.
- About 35 hours of arXiv calls (A) against minutes (B).

`medium` is not needed yet: the plan runs the 2–3 best methods of each task on `small`
first.

## First run (with cross-lists), for reference

Run `20261027-112046`: frame of January–June 2024 with cross-lists, 28,330 papers. Same
conclusion; coverage within one point of the table above (venue 0.68, DOI 0.945), abstracts
identical 90.5%, arXiv 3.9 s per paper. Only 123 of 200 papers had a primary category among
the three (the rest mainly `cs.CV` 21, `stat.ML` 7, `eess.SP` 6), which led to dropping
cross-lists.
