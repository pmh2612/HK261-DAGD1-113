"""Entity extractors compared in BT2. Each returns `predict(docs) -> {doc_id: [Span]}`.

Documents come from `paper_search.eval.ner.load_scierc` (text, words, word offsets,
sentence bounds). Heavy libraries are imported inside the builders, so loading this
module - and its tests - stays cheap.
"""

import hashlib
import itertools
import json
import random
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from paper_search.eval.ner import TYPES, Span, word_span

Predict = Callable[[list[dict]], dict[str, list[Span]]]
BIO_LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in "BI"]


def _surface(doc: dict, a: int, b: int) -> str:
    return " ".join(doc["words"][a : b + 1]).lower()


def _majority(counts: Counter) -> str | None:
    """The most frequent type, if it is one BT2 scores (not OtherScientificTerm/Generic)."""
    t = counts.most_common(1)[0][0]
    return t if t in TYPES else None


def locate(text: str, name: str, t: str) -> list[Span]:
    """Every whole-word occurrence of `name` in `text`; case-insensitive only as fallback.

    LLMs return names, gold is mentions: each occurrence is one mention. Spacing around
    punctuation is ignored, since SciERC text is tokenized ("Networks ( CNN )") and an
    LLM writes "Networks (CNN)"; words must still be separated by whitespace.
    """
    tokens = re.findall(r"\w+|[^\w\s]", name)
    if not tokens:
        return []
    body = re.escape(tokens[0])
    for prev, tok in itertools.pairwise(tokens):
        body += (r"\s+" if prev[-1].isalnum() and tok[0].isalnum() else r"\s*") + re.escape(tok)
    for flags in (0, re.IGNORECASE):
        pattern = re.compile(rf"(?<![\w-]){body}(?![\w-])", flags)
        found = [(m.start(), m.end(), t) for m in pattern.finditer(text)]
        if found:
            return found
    return []


# --- 1. dictionary -----------------------------------------------------------------------


def gazetteer(train: list[dict]) -> Predict:
    """Every entity name seen in train, typed by its majority type, longest match first."""
    counts: dict[str, Counter] = {}
    for doc in train:
        for a, b, t in doc["word_entities"]:
            counts.setdefault(_surface(doc, a, b), Counter())[t] += 1
    lexicon = {s: t for s, c in counts.items() if len(s) > 1 and (t := _majority(c))}
    alternation = "|".join(re.escape(s) for s in sorted(lexicon, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w-])(?:{alternation})(?![\w-])")

    def predict(docs: list[dict]) -> dict[str, list[Span]]:
        return {
            d["doc_id"]: [
                (m.start(), m.end(), lexicon[m.group()])
                for m in pattern.finditer(d["text"].lower())
            ]
            for d in docs
        }

    return predict


# --- 2. general-purpose spaCy ------------------------------------------------------------


def spacy_general(model: str, label_map: dict[str, str]) -> Predict:
    """A news-trained NER with a fixed label mapping: the weak reference point."""
    import spacy

    nlp = spacy.load(model)

    def predict(docs: list[dict]) -> dict[str, list[Span]]:
        return {
            d["doc_id"]: [
                (e.start_char, e.end_char, label_map[e.label_])
                for e in nlp(d["text"]).ents
                if e.label_ in label_map
            ]
            for d in docs
        }

    return predict


# --- 3. SciSpaCy spans, typed by head word -----------------------------------------------


def head_typed(train: list[dict], spans: dict[str, list[list[int]]]) -> Predict:
    """Type SciSpaCy's untyped spans by the majority type of their last word in train.

    SciSpaCy runs in its own venv (notebooks/scispacy_spans.py) and hands over spans.
    A head word seen fewer than twice in train, or mostly as OtherScientificTerm/Generic,
    leaves the span out.
    """
    heads: dict[str, Counter] = {}
    for doc in train:
        for _, b, t in doc["word_entities"]:
            heads.setdefault(doc["words"][b].lower(), Counter())[t] += 1
    types = {h: _majority(c) for h, c in heads.items() if c.total() >= 2}

    def predict(docs: list[dict]) -> dict[str, list[Span]]:
        out = {}
        for d in docs:
            out[d["doc_id"]] = [
                (s, e, t)
                for s, e in spans.get(d["doc_id"], [])
                if (t := types.get(d["text"][s:e].split()[-1].lower()))
            ]
        return out

    return predict


# --- GLiNER ------------------------------------------------------------------------------


def gliner(model: str, labels: dict[str, str], threshold: float) -> Predict:
    """Zero-shot NER, one sentence at a time (GLiNER truncates long inputs)."""
    from gliner import GLiNER

    ner = GLiNER.from_pretrained(model)

    def predict(docs: list[dict]) -> dict[str, list[Span]]:
        out = {}
        for d in docs:
            spans = []
            for i, j in d["sentences"]:
                offset = d["starts"][i]
                sentence = " ".join(d["words"][i:j])
                for e in ner.predict_entities(sentence, list(labels), threshold=threshold):
                    spans.append((offset + e["start"], offset + e["end"], labels[e["label"]]))
            out[d["doc_id"]] = spans
        return out

    return predict


# --- 4. fine-tuned SciBERT ---------------------------------------------------------------


def device() -> str:
    """cuda, then Apple silicon, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def bio_labels(doc: dict, i: int, j: int) -> list[str]:
    """BIO tags for the words of sentence i..j. Nested entities: the longest one wins."""
    tags = ["O"] * (j - i)
    entities = [e for e in doc["word_entities"] if e[2] in TYPES and i <= e[0] and e[1] < j]
    for a, b, t in sorted(entities, key=lambda e: e[0] - e[1]):
        if all(tag == "O" for tag in tags[a - i : b - i + 1]):
            tags[a - i] = f"B-{t}"
            for k in range(a - i + 1, b - i + 1):
                tags[k] = f"I-{t}"
    return tags


def decode_bio(tags: list[str]) -> list[tuple[int, int, str]]:
    """(first, last, type) word spans; an I- that does not continue an entity starts one."""
    spans: list[tuple[int, int, str]] = []
    for k, tag in enumerate(tags):
        if tag == "O":
            continue
        prefix, t = tag.split("-", 1)
        if prefix == "I" and spans and spans[-1][1] == k - 1 and spans[-1][2] == t:
            spans[-1] = (spans[-1][0], k, t)
        else:
            spans.append((k, k, t))
    return spans


def tag_docs(model, tokenizer, docs: list[dict]) -> dict[str, list[Span]]:
    """Run a token-classification model sentence by sentence; first sub-token decides."""
    import torch

    out = {}
    for d in docs:
        spans = []
        for i, j in d["sentences"]:
            enc = tokenizer(
                d["words"][i:j], is_split_into_words=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                ids = model(**enc.to(model.device)).logits[0].argmax(-1).tolist()
            tags, seen = ["O"] * (j - i), set()
            for pos, w in enumerate(enc.word_ids()):
                if w is not None and w not in seen:
                    seen.add(w)
                    tags[w] = model.config.id2label[ids[pos]]
            spans += [
                word_span(d["starts"], d["words"], i + a, i + b, t) for a, b, t in decode_bio(tags)
            ]
        out[d["doc_id"]] = spans
    return out


def scibert(model_dir: Path) -> Predict:
    """SciBERT fine-tuned on SciERC train by scripts/train_scibert_ner.py."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device()).eval()
    return lambda docs: tag_docs(model, tokenizer, docs)


# --- 5-6. in-process LLM, zero- or few-shot ----------------------------------------------

PROMPT = """Extract the scientific entities from this abstract.

- Task: a problem or application the work addresses (e.g. machine translation, parsing).
- Method: a method, model, algorithm, system or technique (e.g. CRF, LSTM, our parser).
- Metric: an evaluation metric or measure (e.g. F1, BLEU, accuracy, word error rate).
- Material: a dataset, corpus or data resource (e.g. Penn Treebank, Japanese text).

Copy each entity exactly as it appears in the abstract. Do not invent entities.
Answer with JSON only: {{"Task": [...], "Method": [...], "Metric": [...], "Material": [...]}}

Abstract:
{text}"""


def parse_answer(raw: str) -> tuple[dict[str, list[str]] | None, int]:
    """{type: [names]} from the model's JSON (None if none), plus bare-string count.

    The second value counts types that came back as a bare string instead of the list
    the prompt asks for. A bare string is kept as one name: iterating it would yield
    single characters.
    """
    # ponytail: parse-and-hope, no constrained decoding; invalid_json and not_a_list in
    # the run stats say whether DECIDE-7's schema-constrained decoding is worth adding.
    try:
        parsed = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
    except (AttributeError, ValueError):
        return None, 0
    if not isinstance(parsed, dict):
        return None, 0
    out, not_a_list = {}, 0
    for t in TYPES:
        value = parsed.get(t) or []
        if isinstance(value, str):
            value, not_a_list = [value], not_a_list + 1
        names = (str(n).strip() for n in value) if isinstance(value, list) else ()
        out[t] = list(dict.fromkeys(n for n in names if n))
    return out, not_a_list


def _answer(doc: dict) -> str:
    """The gold answer for a few-shot example, in the format the prompt asks for."""
    names: dict[str, list[str]] = {t: [] for t in TYPES}
    for a, b, t in doc["word_entities"]:
        name = " ".join(doc["words"][a : b + 1])
        if t in TYPES and name not in names[t]:
            names[t].append(name)
    return json.dumps(names)


def llm(model_id: str, train: list[dict], shots: int, cache_dir: Path) -> Predict:
    """Greedy JSON from an instruction-tuned model loaded in-process (DECIDE-7, NFR-8).

    Few-shot examples are fixed (seed 42) train abstracts of ordinary length, shown as
    earlier user/assistant turns. Raw outputs are cached by a hash of model + messages,
    so a re-run costs nothing. Names that cannot be found in the text are counted.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load on CPU, then move: device_map="mps" segfaulted while loading the weights
    # (torch 2.14, transformers 5.16, 2026-11-01).
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    model = model.to(device()).eval()
    pool = [d for d in train if 100 <= len(d["words"]) <= 160]
    examples = random.Random(42).sample(pool, shots)
    history = [
        turn
        for ex in examples
        for turn in (
            {"role": "user", "content": PROMPT.format(text=ex["text"])},
            {"role": "assistant", "content": _answer(ex)},
        )
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {"invalid_json": 0, "not_a_list": 0, "names": 0, "unlocated": 0}

    def generate(messages: list[dict]) -> str:
        key = hashlib.sha256(json.dumps([model_id, messages]).encode()).hexdigest()
        path = cache_dir / f"{key}.txt"
        if path.exists():
            return path.read_text()
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        text = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        path.write_text(text)
        return text

    def predict(docs: list[dict]) -> dict[str, list[Span]]:
        out = {}
        for d in docs:
            raw = generate([*history, {"role": "user", "content": PROMPT.format(text=d["text"])}])
            parsed, not_a_list = parse_answer(raw)
            stats["invalid_json"] += parsed is None
            stats["not_a_list"] += not_a_list
            spans = []
            for t, names in (parsed or {}).items():
                for name in names:
                    found = locate(d["text"], name, t)
                    stats["names"] += 1
                    stats["unlocated"] += not found
                    spans += found
            out[d["doc_id"]] = spans
        return out

    predict.stats = stats
    return predict
