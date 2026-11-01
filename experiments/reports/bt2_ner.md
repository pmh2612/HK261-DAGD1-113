# BT2 — Entity extraction (SciERC sanity check)

Runs `20261101-100752` (five systems) and `20261101-101104` (the two LLM arms, re-scored
from cache after a parser fix — see *Errors in our own pipeline*). Config
[`bt2_ner.yaml`](../../configs/bt2_ner.yaml), scripts
[`run_ner.py`](../../scripts/run_ner.py), [`train_scibert_ner.py`](../../scripts/train_scibert_ner.py),
[`ner_errors.py`](../../scripts/ner_errors.py). One run directory per system in
[`experiments/runs/bt2/`](../runs/bt2/) with config, metrics and every prediction.

**On our own annotated abstracts: not run yet** — the 50–100 gold abstracts the plan asks
for do not exist. Everything below is on SciERC.

## Setup

- **SciERC** (Luan et al., 2018): 500 abstracts from AI conferences — NLP, vision,
  speech, ML — split 350 train / 50 dev / 100 test, the same split everyone reports on.
  The closest public set to our arXiv cs.CL / cs.LG corpus.
- **Types scored:** Task, Method, Metric, Material — 922 entities in the 100 test
  abstracts (263 / 430 / 71 / 158). Material is SciERC's datasets-and-resources type, the
  plan's Dataset. SciERC also labels OtherScientificTerm and Generic; those are not scored.
- **Exact match:** same characters, same type. **Partial:** overlaps a gold entity of
  the same type. Micro-averaged P / R / F1. Gold is mention-level, so a name an LLM
  returns once counts at every place it occurs in the abstract.
- **No system sees test before scoring.** The dictionary, the SciSpaCy type mapping and
  the few-shot examples come from train; SciBERT is trained on train and its epoch chosen
  on dev.
- **Significance:** paired bootstrap over abstracts, 10,000 resamples, two-sided
  (F1 is not an average over abstracts, so the t-test of BT6/BT7 does not apply).
- Timing on an Apple-silicon laptop (MPS), 16 GB.

## Results

| # | System | P | R | **F1** | Partial F1 | Task | Method | Metric | Material | s / abstract |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | **SciBERT, fine-tuned on SciERC train** | 0.640 | 0.626 | **0.633** | 0.753 | 0.579 | 0.698 | 0.532 | 0.608 | 0.04 |
| + | GLiNER zero-shot (prototype's extractor) | 0.370 | 0.304 | 0.334 | 0.433 | 0.243 | 0.360 | 0.393 | 0.333 | 0.18 |
| 6 | LLM few-shot (Qwen2.5-3B, 3 examples) | 0.421 | 0.273 | 0.332 | 0.457 | 0.325 | 0.356 | 0.342 | 0.280 | 5.73¹ |
| 3 | SciSpaCy `en_core_sci_sm`, typed by head word | 0.204 | 0.219 | 0.211 | 0.479 | 0.173 | 0.230 | 0.342 | 0.179 | 0.02² |
| 5 | LLM zero-shot (Qwen2.5-3B) | 0.409 | 0.105 | 0.167 | 0.311 | 0.162 | 0.147 | 0.245 | 0.187 | 4.66¹ |
| 1 | Dictionary of train entity names | 0.195 | 0.123 | 0.151 | 0.384 | 0.122 | 0.118 | 0.281 | 0.217 | < 0.01 |
| 2 | spaCy `en_core_web_sm` (general NER) | 0.116 | 0.017 | 0.030 | 0.126 | 0.000 | 0.056 | 0.000 | 0.000 | 0.02 |

Per-type columns are exact F1. # is the method number in the experiment plan.
¹ From the uncached generation run ([`20261101-100752_llm_generation.log`](../runs/bt2/20261101-100752_llm_generation.log)); the
run files say 0.00 because they were re-scored from cache. Loading the 3B model takes
25–50 s on top. ² Measured in the SciSpaCy venv; the run files show only loading saved spans.
Cost is 0 for every system: all run locally.

Paired bootstrap on exact F1:

| Comparison | ΔF1 | p |
|---|---:|---:|
| SciBERT vs every other system | +0.30 to +0.60 | < 0.0001 |
| LLM few-shot vs GLiNER | −0.002 | 0.93 |
| LLM few-shot vs LLM zero-shot | +0.164 | < 0.0001 |
| GLiNER vs SciSpaCy | +0.122 | < 0.0001 |
| SciSpaCy vs dictionary | +0.061 | 0.0004 |
| LLM zero-shot vs dictionary | +0.017 | 0.50 |

**SciBERT** trained in ~5 minutes (8 epochs × 37 s); dev exact F1 by epoch 0.416, 0.597,
0.651, 0.637, 0.642, **0.660**, 0.629, 0.649 — epoch 6 kept. Its test F1 is in line with
published SciBERT-based SciERC entity results (about 0.65–0.69 over all six types).

## Findings

- **The fine-tuned model wins by a wide margin:** 0.633, almost twice the next system,
  ahead of every other by 0.30 or more (p < 0.0001), and the fastest neural system.
  Learning SciERC's own boundaries and type conventions from 350 abstracts beats any
  zero- or few-shot approach.
- **A 3B LLM is no better than GLiNER, and 30× slower.** Few-shot 0.332 vs GLiNER 0.334
  (p = 0.93); 5.7 s per abstract against 0.18 s. At `medium` (20,000 abstracts) that is
  about 32 hours against 1 hour.
- **Examples matter more than anything else for the LLM:** zero-shot → few-shot doubles
  F1 (0.167 → 0.332, p < 0.0001). Zero-shot broke the answer format: in 313 of 400 type
  slots it returned one string instead of a list, often several entities joined by
  commas, and 115 of 326 names could not be found in the abstract ("not specified",
  paraphrases, summaries). Few-shot returned valid lists every time; 48 of 589 names were
  still not in the text.
- **Precision is not the LLM's problem, recall is:** both arms have the best precision
  after SciBERT (0.41–0.42) and the worst recall of the neural systems (0.10–0.27). The
  model names the main contributions and skips minor mentions.
- **SciSpaCy finds the spans but cannot type them:** the best partial F1 after SciBERT
  (0.479) and poor exact F1 (0.211). It marks "database" where gold has "IEMOCAP
  database", "text" for "Japanese text" — 271 boundary errors, the most of any system.
- **General-purpose NER is the useless baseline the plan expected:** spaCy's news model
  finds 138 entities in 100 abstracts, nearly all ORG/PRODUCT names.

## Error analysis

From [`ner_errors.py`](../../scripts/ner_errors.py) over the test predictions:

| System | Boundary | Wrong type | Spurious | Missed |
|---|---:|---:|---:|---:|
| SciBERT | 111 | 54 | 160 | 158 |
| GLiNER | 86 | 66 | 324 | 466 |
| LLM few-shot | 94 | 54 | 198 | 463 |
| SciSpaCy | 271 | 24 | 493 | 392 |
| LLM zero-shot | 71 | 18 | 51 | 681 |
| Dictionary | 184 | 8 | 274 | 559 |

Typical cases:

1. **Boundaries are the main residual error even for SciBERT** (exact 0.633 vs partial
   0.753). Gold *"CUHK03 datasets"* and *"scale Market-1501"*, predicted *"large scale
   Market-1501 and CUHK03 datasets"*; gold *"PASCAL VOC 2006 datasets"*, predicted
   *"Savarese et al. 2007 and PASCAL VOC 2006 datasets"*.
2. **Task vs Method is ambiguous in the gold itself.** *"rule application"* is a Method,
   *"Automatic Speech Recognition ( ASR ) System"* a Method, *"weakly supervised
   dependency parser"* a Task, *"LCS-Marine"* (a system) a Task. Every system loses points
   on this line.
3. **Much of the "spurious" output is real entities SciERC types differently.**
   *"SLIPPER"*, a learning algorithm, is Generic; *"loss functions"* and *"human
   transcriptions"* are OtherScientificTerm. Scoring four of six types penalizes them.
4. **The LLM merges and condenses.** Zero-shot returned *"unsupervised methods,
   semi-supervised methods, hidden Markov models (HMMs)"* as one Method and *"novel view
   generation in one-to-one teleconferencing applications"* where gold has two Tasks.
5. **The dictionary only knows train vocabulary** (missed *"expert games"*,
   *"grounding"*), and its majority type is often wrong in context: *"parsing"* is a Task
   in train, a Method in *"interplay between smoothing and parsing results"*.

## Errors in our own pipeline, found and fixed before reporting

- **Bare-string LLM answers were read character by character.** The zero-shot model
  wrote `"Task": "3-D stereo reconstruction"` instead of a list; iterating that string
  produced one "entity" per letter, and the first score was F1 0.005. `parse_answer` now
  keeps a bare string as one name and counts these format slips (`not_a_list`).
- **Tokenized text hid correct LLM answers.** SciERC text has spaces around punctuation
  (*"Networks ( CNN )"*), the LLM writes *"Networks (CNN)"*. `locate` now ignores spacing
  around punctuation: unlocated few-shot names fell from 89 to 48 and F1 rose from 0.315
  to 0.332. Our own corpus is not tokenized like this, so the fix removes an artefact,
  not a real weakness.
- **Loading Qwen with `device_map="mps"` crashed** (segmentation fault while loading the
  weights). The model is now loaded on CPU and moved to the GPU.

## Caveats

- **SciERC is older AI abstracts (some from the 1990s) with its own annotation conventions.** Our arXiv
  2024 abstracts mention LLMs, benchmarks and model names SciERC has never seen. A
  fine-tuned model is also the one most tied to its training conventions, so its lead may
  shrink on our corpus. To be confirmed on the 50–100 abstracts the team annotates.
- **The LLM is the 3B model that fits a 16 GB laptop.** Larger models may do better;
  the result says "a 3B model, prompted, is not enough", not "LLMs are not enough".
- **Not tuned:** GLiNER's threshold (0.5) and labels, the LLM prompt, and the number of
  few-shot examples were fixed before seeing test, and not tuned on dev.
- The SciERC-trained models we checked on Hugging Face have no model card, so they may have
  seen test; SciBERT was trained here instead (plan method 4).

## Proposal for the next round

Carry **SciBERT (fine-tuned)** and **GLiNER** forward, to be compared on our own annotated
abstracts:

- SciBERT is the clear winner on SciERC; the open question is how it transfers to 2024
  arXiv abstracts.
- GLiNER is the best zero-shot option and needs no training data, if SciBERT transfers
  badly.
- The 3B LLM is dropped as an extractor: no better than GLiNER, 30× slower to run at
  `medium`. It stays a candidate for BT3 (relations) and BT4 (canonicalization), where
  reading the whole abstract matters more.

For annotating our own abstracts, SciBERT's predictions can serve as the draft the plan
suggests ("run a model, then correct"), which should cut annotation time.
