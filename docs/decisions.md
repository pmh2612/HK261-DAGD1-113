# Decision Record

Scientific Paper Search System (Knowledge Graph + RAG)

Open decisions raised in [`requirements.md`](requirements.md) §8. Each one is written so
that both members can read the tradeoff, disagree with the recommendation, and record what
was actually chosen. Roadmap 1.3 requires design decisions and their reasons to be
recorded — this file is that record.

**How to use:** read the decision, discuss, then fill in the *Decision* box at the end of
the section. Do not delete the options that were rejected; why something was *not* chosen
is the part that is useful in six months and in the Phase 1 report.

| ID | Decision | Blocks | Status |
|---|---|---|---|
| [DECIDE-1](#decide-1--reference-snowballing) | Snowball one hop of references? | Roadmap 2.2 (October) | ☑ **B**, `:External` |
| [DECIDE-2](#decide-2--chunking-strategy) | Chunking strategy | Roadmap 3.3 (November) | ☑ **C** |
| [DECIDE-3](#decide-3--faithfulness-review-protocol) | Faithfulness review protocol | Roadmap 5.5 (Phase 2) | ☑ agreed + weighted kappa |
| [DECIDE-4](#decide-4--which-llm) | Which LLM, hosted or local | Roadmap 3.1 (November) | ⊘ **superseded by DECIDE-7** |
| [DECIDE-5](#decide-5--embedding-model) | Embedding model | Roadmap 3.3 (November) | ☑ `bge-small-en-v1.5` |
| [DECIDE-6](#decide-6--team-conventions) | Team conventions | Any shared code | ☑ adopted |
| [DECIDE-7](#decide-7--which-open-weights-model-and-runtime) | Which open-weights model and local runtime | Roadmap 3.1 (November) | ☐ **open** |

Six were resolved on 2026-09-24 (Tran Ha My, via pull request #1; the reply is kept
verbatim at the end of [`decisions-vi.md`](decisions-vi.md)). Each section below records
the outcome and any amendment made to the recommendation.

**`DECIDE-4` was superseded on 2026-09-25.** The course requires the team to run the model
itself rather than call a hosted API, which removes the option that decision chose. It is
reopened as `DECIDE-7`.

**Also blocking, and not really a decision:** the Semantic Scholar API key. Being applied
for by Tran Ha My as of 2026-09-24. See [§7](#7-not-a-decision-but-do-it-this-week).

---

## DECIDE-1 — Reference snowballing

**Blocks:** Roadmap 2.2 · **Affects:** `FR-3`, `FR-17`, `UC-3`, `NFR-8`

### Context

`requirements.md` §2.1 `S3` predicted a sparse citation graph. It has now been measured
([`notebooks/spike_data_availability.py`](../notebooks/spike_data_availability.py) and a
follow-up batch query, 2026-09-23):

| Measurement | Value |
|---|---|
| References per seed paper (mean / median) | 29.3 / 25.0 |
| Seed papers with zero references recorded | 166 of 1,000 |
| Unique referenced papers, from a 100-paper sample | 2,022 |
| **References landing back inside the seed set** | **12 of 2,319 edges (0.5%)** |

That last row is the whole problem. Scaling the hit rate to a 1,000-paper seed set gives
roughly **1.2 internal `CITES` edges per paper** — a graph of 1,000 nodes with around 1,100
citation edges. Technically not empty; practically useless for `UC-3` ("what cites this
paper?") because most papers will have zero in-corpus citations.

**Important framing:** fetching the reference lists is needed *either way* — that is how
the 1,100 internal edges get created at all, and it costs about 20 batch API calls. The
actual decision is narrower: **do we also add the referenced papers as new nodes?**

### Options

| | A — Seeds only | B — Add frequently-cited references | C — Add everything |
|---|---|---|---|
| **New nodes** | 0 | ~1,500–3,000 (estimate; papers cited by ≥3 seeds) | ~15,000–18,000 |
| **`CITES` edges** | ~1,100 | ~1,100 internal + ~10,000 to the added papers | ~29,000 |
| **Extra PDF / extraction / embedding cost** | none | **none** — added papers are metadata-only | none, same reason |
| **Extra API cost** | none | ~20 batch calls, same as A | same |
| **What it buys** | nothing | Seeds with no direct link connect through a shared foundational paper — exactly what `FR-17` needs | marginally more, mostly one-off references nobody cites twice |
| **What it costs** | `FR-17` and `UC-3` stay weak | graph is ~3x bigger; needs an "is this paper searchable?" flag | graph is ~16x bigger, mostly noise |

The cost column is the surprise. Snowballed papers exist **only to be citation targets** —
they never need a PDF, LLM extraction (`FR-9`) or an embedding (`FR-13`). The expensive
stages still run over the 1,000 seed papers only. Snowballing is therefore almost free.

### Recommendation — **Option B**

Keep referenced papers cited by **≥3 seed papers**. That threshold picks out the
foundational work of the field (the papers everyone cites) and discards the long tail of
one-off references. The exact threshold should be re-tuned once the real distribution over
1,000 seeds is known — the 100-paper sample gives 52 papers at ≥3, but that number does not
scale linearly.

### Consequence if B is chosen

`kg-schema.md` must distinguish the two kinds of `Paper` node, because a snowballed paper
can appear in graph results but cannot be retrieved from the search index — returning one
without saying so would look like a bug.

Proposal: a boolean property `in_corpus` on `Paper` (`true` = seed, searchable;
`false` = snowballed, graph-only), or a second label `:External`. Retrieval filters on it;
`UC-3` shows both but marks them differently.

> **Decision: Option B**, threshold ≥3 seed papers. Snowballed papers are marked with a
> second label **`:External`** on the `Paper` node, not with an `in_corpus` property.
> **Chosen by:** Tran Ha My · **Date:** 2026-09-24
>
> **Amendment to the recommendation:** the recommendation left the marker open between a
> property and a label; a label was chosen. Retrieval must therefore filter on
> `(:Paper)` without `:External` rather than on a property value, and `kg-schema.md` needs
> a constraint that holds across both labels so a paper cannot be duplicated by being
> collected as a reference first and as a seed later. Re-tune the threshold once the real
> distribution over 1,000 seeds is known.

---

## DECIDE-2 — Chunking strategy

**Blocks:** Roadmap 3.3 · **Affects:** `FR-13`, `FR-14`, `FR-20`, `NFR-6`

### Context

Chunk size sets two things at once, and they pull in opposite directions: how precisely a
citation can point at a source, and how well the embedding represents the text. Embedding
models have a hard input limit (commonly 512 tokens); anything longer is silently
truncated, so an over-long chunk quietly loses its tail.

Remember `S1`: ~360 of the 1,000 papers have no full text at all. For those, the only
chunk is the abstract.

### Options

| | A — Whole sections | B — Fixed windows | C — Section-aware windows |
|---|---|---|---|
| **Unit** | one chunk per section | e.g. 512 tokens, 64 overlap, ignoring structure | split by section, then window any section that is too long |
| **Citation reads as** | "Method section of [12]" | "chunk 47 of [12]" | "Method section of [12]" |
| **Embedding quality** | bad for long sections — truncated | even | even |
| **Risk** | a 3,000-word Experiments section is mostly invisible to retrieval | windows cut across section boundaries, mixing two topics in one vector | slightly more code |
| **Metadata-only papers** | abstract = 1 chunk | abstract = 1 chunk | abstract = 1 chunk, `source="abstract"` |

### Recommendation — **Option C**

It is the only option that satisfies both `NFR-6` (provenance good enough to cite) and the
embedding model's input limit. The extra work over B is small: split on section first, then
apply the same windowing inside each section, and carry the section name in the chunk
metadata.

Chunk provenance should therefore be, at minimum:

```
paper_id · source ("abstract" | section name) · window index within that source · char offsets
```

Window size and overlap are tuning parameters, not part of this decision — set them once
`DECIDE-5` fixes the embedding model's input limit.

> **Decision: Option C** — section-aware windows.
> Canonical section names, agreed as the shared contract between PDF splitting
> (`FR-7`) and chunk provenance (`FR-14`):
>
> ```
> Abstract · Introduction · Related Work · Method · Experiments · Results · Conclusion · Other
> ```
>
> `Other` is the catch-all for headings that do not map to the seven — it exists so that
> unmapped text keeps valid provenance instead of being dropped.
> Window size and overlap still to be set once `DECIDE-5` fixed the input limit; 512 tokens
> with 64 overlap is the starting point.
> **Chosen by:** Tran Ha My (section list), Pham Minh Hieu (strategy) · **Date:** 2026-09-24

---

## DECIDE-3 — Faithfulness review protocol

**Blocks:** Roadmap 5.5 · **Affects:** `NFR-3`, `NFR-4`

### Context

`NFR-3` says "≥90% of claims supported by the retrieved context". That number means
nothing until four things are pinned down: what counts as one claim, how many are graded,
who grades them, and what happens when the two graders disagree. Deciding this *after*
seeing the results is how evaluations become unconvincing.

### Recommendation

| Parameter | Proposal | Why |
|---|---|---|
| **Unit** | One atomic claim — a single assertion that can be checked against one source | Grading whole answers hides partial failures |
| **Sample** | 50 answers drawn from the test query set (Roadmap 4.1), ~4 claims each ≈ 200 claims | Enough for a ±7 pp confidence interval at the 90% level; small enough that two people can grade it in an evening |
| **Scale** | supported / partially supported / unsupported | Binary hides the interesting middle case |
| **Graders** | Both members grade the **same** sample independently, blind to each other | One grader's number is an opinion |
| **Agreement** | Report Cohen's kappa alongside the score | Turns "we checked it" into a defensible measurement; also a good line in the thesis |
| **Disagreements** | Discussed and resolved together; resolved labels are the ones reported | |
| **Timing** | Write the rubric **before** Phase 2 generation works | Writing it afterwards invites fitting the rubric to the results |

Same protocol covers `NFR-4` (citation accuracy): for each cited claim, check that the
cited paper exists and that the cited section actually contains the claim.

> **Decision: as proposed** — 50 answers, ~200 claims, both members grading the same
> sample independently.
> **Amended:** the three-point scale is scored **supported = 1 · partially supported = 0.5
> · unsupported = 0**, and agreement is reported as **weighted** Cohen's kappa rather than
> plain kappa.
> **Chosen by:** Tran Ha My · **Date:** 2026-09-24
>
> The amendment is the right call: the scale is ordinal, so "supported vs partially
> supported" is a smaller disagreement than "supported vs unsupported". Plain kappa treats
> both as equally wrong and understates agreement. Use quadratic weights unless there is a
> reason to prefer linear.

---

## DECIDE-4 — Which LLM

> ## ⊘ SUPERSEDED — 2026-09-25
>
> The course requires the team to run the AI themselves rather than call a hosted API.
> Everything below assumed a hosted API and a dollar budget, so the conclusion no longer
> holds: the cost table that made hosted inference obviously right has become irrelevant,
> because the option it compared against is now the only one available.
>
> Kept rather than deleted, because the *reasoning* still transfers. Two points survive
> the change and carry straight into `DECIDE-7`:
>
> - **It is still two decisions, not one.** Extraction is a batch job with no latency
>   requirement; generation is interactive. They can use different models.
> - **Tran Ha My's escalation trigger still applies.** "Step up if entity names come back
>   fragmented" was the right test, and it matters *more* now: fragmentation is the
>   characteristic failure of a small local model, which is what `DECIDE-7` is choosing.
>
> Reopened as [`DECIDE-7`](#decide-7--which-open-weights-model-and-runtime).

**Blocks:** Roadmap 3.1 · **Affects:** `FR-9`, `FR-19`, `NFR-8`, `NFR-12`

### Context

This was written as one decision but it is two, with different answers:

- **Extraction (`FR-9`)** — batch job, runs once over 1,000 papers, no latency requirement,
  needs reliable structured JSON output.
- **Generation (`FR-19`, Phase 2)** — per query, latency matters, runs many times during
  development and evaluation.

The budget ceiling is `NFR-8` (≤50 USD). Estimating against Claude API list prices
(2026-06 pricing — **re-check before committing**, prices move):

**Extraction**, 1,000 papers × ~900 input tokens (prompt + title + abstract) and ~300
output tokens ⇒ ~0.9M input, ~0.3M output:

| Model | Model ID | Input / output per 1M | One-time cost | With Batch API (−50%) |
|---|---|---|---|---|
| Claude Haiku 4.5 | `claude-haiku-4-5` | $1 / $5 | ~$2.40 | **~$1.20** |
| Claude Sonnet 5 | `claude-sonnet-5` | $2 / $10 | ~$4.80 | ~$2.40 |
| Claude Opus 5 | `claude-opus-5` | $5 / $25 | ~$12.00 | ~$6.00 |

**The cost argument for a local model does not survive these numbers.** Extraction is a
one-time job costing somewhere between one and six dollars. A local 7–8B model would save
that, at the price of worse structured-output reliability and several hours of laptop CPU
time — and `FR-10` (entity normalization) gets harder when extraction is noisier.

**Generation** is the real budget item, because it runs repeatedly. At ~4,000 input and
~500 output tokens per query, roughly 2,000 development and evaluation queries cost about
$13 on Haiku 4.5 and about $26 on Sonnet 5 — most of the ceiling in the second case.

### Recommendation

1. **Extraction: hosted, `claude-haiku-4-5`, through the Batch API** (~$1.20). Extraction
   is a narrow, well-specified task, which is where the cheapest tier does best.
2. **Validate before spending.** Run the extraction prompt on 20 papers first and check the
   JSON by hand. If Haiku's output is unreliable, step up to `claude-sonnet-5` — the
   difference is about two dollars, not a budget decision.
3. **Generation: decide at the start of Phase 2**, against whatever budget is left. Two
   cost levers apply there and should be used from the first day: **prompt caching** (the
   system prompt and instructions are identical across queries) and **the Batch API** for
   evaluation runs, which are not latency-sensitive.
4. **Track spend from the first call.** A shared note with the running total; `NFR-8` is
   easy to blow through without noticing.

`NFR-12` (no GPU) does not constrain this — hosted inference runs nowhere near the laptop.
It constrains `DECIDE-5` instead.

> **Decision: `claude-haiku-4-5` through the Batch API** for extraction (`FR-9`).
> Generation (`FR-19`) deferred to the start of Phase 2, as recommended.
> **Escalation trigger, added:** after the 20-paper manual check, step up to
> `claude-sonnet-5` **if entity names come back fragmented** — i.e. the same method or
> dataset extracted under several surface forms that `FR-10` then has to merge.
> **Chosen by:** Tran Ha My · **Date:** 2026-09-24
>
> A sharper trigger than "if the output is unreliable": fragmentation is the failure mode
> that makes extra work downstream, and it is visible in a 20-paper sample.

---

## DECIDE-5 — Embedding model

**Blocks:** Roadmap 3.3 · **Affects:** `FR-13`, `FR-15`, `NFR-1`, `NFR-12`

### Context

Rough corpus size: ~640 full-text papers at maybe 30 chunks each, plus ~1,000 abstracts
≈ **20,000 chunks** to embed on a laptop CPU (`NFR-12`).

Three properties matter, in this order:

1. **Retrieval quality** on scientific English.
2. **CPU throughput** — this is a one-time cost, so even an hour is acceptable; it only
   becomes painful when chunking is re-tuned and everything must be re-embedded.
3. **Vector dimension** — drives FAISS index size and search latency (`NFR-1`).

### Candidates

All available through `sentence-transformers`, which is already in `requirements.txt`.

| Model | Dim | Size | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ~22M | The fast baseline. Weakest quality, but embeds 20k chunks in minutes |
| `BAAI/bge-small-en-v1.5` | 384 | ~33M | Noticeably better retrieval than MiniLM at nearly the same speed. Requires a query prefix ("Represent this sentence for searching relevant passages:") — **forgetting it silently degrades results** |
| `all-mpnet-base-v2` | 768 | ~110M | Strong general-purpose baseline, ~4× slower, 2× the index size |
| `BAAI/bge-base-en-v1.5` | 768 | ~109M | Best quality of the four, same speed and size cost as mpnet |

### Recommendation — `BAAI/bge-small-en-v1.5`

Best quality per CPU-second of the four, and 384 dimensions keeps the FAISS index small and
`NFR-1` comfortable. Two conditions:

- **Measure, do not assume.** Once the test query set from Roadmap 4.1 exists, run
  `bge-small` against `all-MiniLM-L6-v2` and one 768-dim model on the same queries. That
  comparison is a Phase 1 report result, not wasted work — Roadmap 4.1 asks for exactly
  this kind of baseline number.
- **Pin the model name in `config.py`.** Query-time and index-time embeddings must come
  from the same model; changing it means re-embedding everything. Treat it as part of the
  index's identity, and record which model built which index file.

Also note the asymmetry: `bge-*` models expect the *query* to carry a prefix that the
*documents* do not. Wire this into the search function from the start.

> **Decision: `BAAI/bge-small-en-v1.5`**, 384 dimensions. Pinned in `config.py` as
> `settings.embedding_model`.
> **Flagged by Tran Ha My:** the query-side instruction prefix must not be forgotten. It is
> therefore pinned next to the model as `settings.embedding_query_prefix` rather than left
> as a literal in the search function — the failure mode is silent, so it needs to live
> somewhere visible.
> **Chosen by:** Pham Minh Hieu, confirmed by Tran Ha My · **Date:** 2026-09-24
>
> Still to do: the three-model comparison on the test query set (Roadmap 4.1).

---

## DECIDE-6 — Team conventions

**Blocks:** any shared code · **Affects:** Roadmap 1.1 (last open item)

### Context

Two people, one repository, no convention agreed yet. Cheapest decision here and the one
with the shortest deadline: everything pushed before it is settled has to be tidied up
afterwards.

### Proposal

**Branches** — `<type>/<short-kebab-description>`, matching the commit types:

```
feat/semantic-scholar-client      fix/pdf-hyphenation
docs/kg-schema                    chore/pin-embedding-model
```

**Commits** — imperative subject, ≤72 characters, no trailing period. Body explains *why*,
not *what* (the diff already says what). This matches the existing history, so nothing
needs rewriting.

**Pull requests** — no direct pushes to `main` from here on. Every PR reviewed by the other
member; one approval to merge. Roadmap 1.1 lists "pull request review between members" as
a deliverable, so this is graded, not optional. Reviewing code you did not write is also
how each of you stays able to work on the other's half of the system.

**Ownership** — from the README team table, to avoid two people editing the same file:

| Area | Lead | Directories |
|---|---|---|
| Collection, PDF, Knowledge Graph | Tran Ha My | `src/paper_search/collection/`, `extraction/`, `graph/` |
| Search, RAG, chatbot, UI | Pham Minh Hieu | `src/paper_search/indexing/`, `retrieval/`, `rag/`, `app/` |
| Shared | Both | `config.py`, `docs/`, `tests/`, `pyproject.toml` |

Lead means "reviews every change here", not "only person allowed to touch it".

**Before pushing** — `make lint && make test`. Optionally install the hooks so this is
automatic:

```bash
.venv/bin/pre-commit install
```

**Data files are never committed.** `.gitignore` already covers `data/`, `.env` and index
files. If a dataset has to be shared, share the collection script and its data version, not
the output.

> **Decision: adopted as proposed** — branch naming, commit format, pull-request review,
> directory ownership, and the pre-commit hooks.
> **Chosen by:** Tran Ha My · **Date:** 2026-09-24
>
> In effect from 2026-09-24. Pull request #1 was itself the first use of it.

---

## DECIDE-7 — Which open-weights model, and runtime

**Blocks:** Roadmap 3.1 · **Affects:** `FR-9`, `FR-19`, `NFR-2`, `NFR-9`, `NFR-12`
**Replaces:** `DECIDE-4`

### Context

The course requires self-hosted inference — no hosted LLM API. A GPU is available. Two
workloads, as in `DECIDE-4`:

| | Extraction (`FR-9`) | Generation (`FR-19`, Phase 2) |
|---|---|---|
| Shape | batch, 1,000 papers, once | interactive, per query |
| Input | ~400 tokens (title + abstract) | ~4,000 tokens (retrieved context) |
| Output | ~300 tokens of JSON | ~500 tokens of prose |
| Latency | irrelevant — run it overnight | `NFR-2`: first token < 3 s |
| What it needs | schema adherence, not eloquence | grounding and refusal discipline |

### What actually needs deciding

**1. Model size, which VRAM decides.** Roughly, for a 4-bit quantized model:

| VRAM | Feasible size | Note |
|---|---|---|
| 8 GB | 7–8B | Enough for extraction; adequate for generation with a short context |
| 12–16 GB | 13–14B | Comfortable for both |
| 24 GB+ | 30B+ | More than this project needs |

**Unknown: how much VRAM the available GPU has.** Everything else follows from it, so this
is the first thing to find out.

**2. Which model family.** Any current instruction-tuned open-weights family — Qwen, Llama,
Mistral, Gemma — is a reasonable starting point at these sizes. Pick by measurement, not by
reputation: the test is this corpus, not a leaderboard.

**3. Runtime.** The practical options are Ollama (one install, HTTP API, handles GPU
placement and quantized weights, identical setup on both machines — which `NFR-9` asks for)
or llama.cpp directly (more control, more setup). Both support schema-constrained decoding,
which is the feature that matters most here.

### Recommendation

1. **Check the GPU's VRAM first.** It determines everything above.
2. **Ollama**, for `NFR-9`: both members run the same two commands and get the same model.
3. **One model for both workloads to start.** Splitting them is a real option — a small fast
   model for extraction, a larger one for generation — but only adopt it once there is a
   measurement showing one model cannot do both. Two models is two sets of weights, two
   configurations and two things to describe in the report.
4. **Always use schema-constrained decoding for `FR-9`.** Not a preference: it is what makes
   a small model reliable enough for extraction, and it removes the JSON-repair path
   entirely.
5. **Benchmark on 20 papers before the full run**, exactly as Tran Ha My specified in
   `DECIDE-4`. The test is entity fragmentation. If the model returns the same method under
   three surface forms, `FR-10` inherits the mess.
6. **Pin the model name and runtime version in `config.py` and the manifest**, the same way
   `DECIDE-5` pinned the embedding model. Re-running extraction with a different model
   silently produces a different graph.

### Open question worth asking the advisor

Whether "self-hosted" means running open weights (assumed here) or training or fine-tuning
a model. Those are very different amounts of work, and the second would need its own phase
in the roadmap. Worth confirming before November rather than during it.

> **Decision:** GPU VRAM: ______ · model: ______ · runtime: ______
> **One model or two:** ______
> **Chosen by:** ______ **Date:** ______

---

## 7. Not a decision, but do it this week

**Apply for a Semantic Scholar API key** — <https://www.semanticscholar.org/product/api>

The spike hit the shared unauthenticated quota hard. The bulk search endpoint was fine
(1,000 papers in one call), but `/paper/batch`, which is what reference snowballing needs,
returned HTTP 429 on nine consecutive attempts with 25-second backoff before succeeding
once. Roadmap 2.2 needs many such calls in October.

Approval is not instant, so applying is on the critical path for `DECIDE-1` even though it
is not itself a decision. The key goes in `.env` as `SEMANTIC_SCHOLAR_API_KEY`, which
`config.py` already reads.

Whatever is decided above, `NFR-11` stands: back off on 429, respect the published limits,
and do not work around bot protection.
