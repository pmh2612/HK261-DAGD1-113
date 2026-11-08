"""Entity-extraction data and metrics for BT2.

An entity is a span `(start, end, type)` in character offsets of the abstract text, so
every system - dictionary, spaCy, GLiNER, SciBERT, LLM - is scored the same way.

SciERC (Luan et al., 2018) is the public sanity check: 500 AI abstracts (350 train /
50 dev / 100 test) annotated with Task, Method, Metric, Material, OtherScientificTerm
and Generic. BT2 scores the first four; Material is SciERC's datasets-and-resources type.
"""

import json
from pathlib import Path

Span = tuple[int, int, str]
TYPES = ("Task", "Method", "Metric", "Material")
_PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}


def load_scierc(path: Path) -> list[dict]:
    """SciERC in the DyGIE format, one abstract per line, as text with character spans.

    Words are joined with single spaces, so `starts[i]` is where word i begins. Entity
    indices in the file are word positions in the whole abstract, inclusive at both ends.
    All six SciERC types are kept; scoring filters to TYPES.
    """
    docs = []
    for line in path.open(encoding="utf-8"):
        d = json.loads(line)
        words = [_PTB.get(w, w) for sentence in d["sentences"] for w in sentence]
        starts, pos = [], 0
        for w in words:
            starts.append(pos)
            pos += len(w) + 1
        bounds, i = [], 0
        for sentence in d["sentences"]:
            bounds.append((i, i + len(sentence)))
            i += len(sentence)
        word_entities = [(a, b, t) for sentence in d["ner"] for a, b, t in sentence]
        docs.append(
            {
                "doc_id": d["doc_key"],
                "text": " ".join(words),
                "words": words,
                "starts": starts,
                "sentences": bounds,
                "word_entities": word_entities,
                "entities": [word_span(starts, words, a, b, t) for a, b, t in word_entities],
            }
        )
    return docs


def word_span(starts: list[int], words: list[str], a: int, b: int, t: str) -> Span:
    """Character span of words a..b (inclusive)."""
    return (starts[a], starts[b] + len(words[b]), t)


def _overlaps(a: Span, b: Span) -> bool:
    return a[2] == b[2] and a[0] < b[1] and b[0] < a[1]


def _prf(correct_pred: int, n_pred: int, found_gold: int, n_gold: int) -> dict:
    p = correct_pred / n_pred if n_pred else 0.0
    r = found_gold / n_gold if n_gold else 0.0
    return {
        "p": round(p, 4),
        "r": round(r, 4),
        "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4),
    }


def score(gold: dict[str, list[Span]], pred: dict[str, list[Span]]) -> dict:
    """Precision / recall / F1 per type and micro-averaged over TYPES.

    exact: same start, end and type. partial: overlaps a span of the same type, counted
    from each side, so precision asks "does this prediction touch a gold entity" and
    recall "is this gold entity touched by a prediction".
    """
    out = {}
    for t in (*TYPES, "all"):

        def keep(spans: list[Span], t: str = t) -> set[Span]:
            return {s for s in spans if s[2] in TYPES and t in ("all", s[2])}

        n_pred = n_gold = exact = part_pred = part_gold = 0
        for doc_id, gold_spans in gold.items():
            g, p = keep(gold_spans), keep(pred.get(doc_id, []))
            n_pred, n_gold, exact = n_pred + len(p), n_gold + len(g), exact + len(g & p)
            part_pred += sum(any(_overlaps(x, y) for y in g) for x in p)
            part_gold += sum(any(_overlaps(y, x) for x in p) for y in g)
        out[t] = {
            "gold": n_gold,
            "pred": n_pred,
            "exact": _prf(exact, n_pred, exact, n_gold),
            "partial": _prf(part_pred, n_pred, part_gold, n_gold),
        }
    return out


def errors(gold: dict[str, list[Span]], pred: dict[str, list[Span]]) -> dict[str, list]:
    """Every prediction and gold entity that is not an exact match, by kind of error.

    boundary: right type, overlapping a gold entity, wrong start or end.
    type: exactly a gold entity's words, wrong type. spurious: touches nothing of that
    type. missed: a gold entity no prediction of any type overlaps.
    """
    out: dict[str, list] = {"boundary": [], "type": [], "spurious": [], "missed": []}
    for doc_id, gold_spans in gold.items():
        g = {s for s in gold_spans if s[2] in TYPES}
        p = {s for s in pred.get(doc_id, []) if s[2] in TYPES}
        for s in sorted(p - g):
            if any(_overlaps(s, x) for x in g):
                out["boundary"].append((doc_id, s))
            elif any(x[:2] == s[:2] for x in g):
                out["type"].append((doc_id, s))
            else:
                out["spurious"].append((doc_id, s))
        for s in sorted(g - p):
            if not any(x[0] < s[1] and s[0] < x[1] for x in p):
                out["missed"].append((doc_id, s))
    return out


def bootstrap_p(
    gold: dict[str, list[Span]],
    pred_a: dict[str, list[Span]],
    pred_b: dict[str, list[Span]],
    rounds: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Exact micro-F1 difference (a - b) and a two-sided paired bootstrap p-value.

    F1 is not an average over abstracts, so a t-test does not apply: resample abstracts
    with replacement, recompute both F1s, and count how often the difference changes
    sign. Counts per abstract are precomputed, so 10,000 rounds take about a second.
    """
    import random

    def counts(pred: dict[str, list[Span]]) -> list[tuple[int, int, int]]:
        rows = []
        for doc_id, gold_spans in gold.items():
            g = {s for s in gold_spans if s[2] in TYPES}
            p = {s for s in pred.get(doc_id, []) if s[2] in TYPES}
            rows.append((len(g & p), len(p), len(g)))
        return rows

    def f1(rows: list[tuple[int, int, int]], idx: list[int]) -> float:
        tp, n_p, n_g = (sum(rows[i][k] for i in idx) for k in range(3))
        return 2 * tp / (n_p + n_g) if n_p + n_g else 0.0

    a, b = counts(pred_a), counts(pred_b)
    every = list(range(len(a)))
    observed = f1(a, every) - f1(b, every)
    rng = random.Random(seed)
    flips = 0
    for _ in range(rounds):
        idx = [rng.randrange(len(a)) for _ in every]
        flips += (f1(a, idx) - f1(b, idx)) * observed <= 0
    return round(observed, 4), round(min(1.0, 2 * flips / rounds), 4)
