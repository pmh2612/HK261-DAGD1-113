"""BT3: relation extraction - does a paper PROPOSE a method or USE an existing one?

SciERC's relations hold between two entities ("X used for Y"); the graph's hold between
a paper and an entity, so SciERC cannot score them. The distinction the plan names for
methods, PROPOSES vs USES, is measured instead, on our corpus:

1. Pairs. 300 (paper, method) pairs from the small corpus (SciBERT's Method spans, BT2),
   seed 42, at most two per paper, generic names skipped. The local LLM labels each as
   proposes / uses, cached in data/annotations/method_roles.jsonl.
2. Methods. (a) the KG's current baseline, every method USES; (b) sentence patterns:
   the name after "we propose / present / introduce ..." in its sentence, or after
   "our", "called", "named", "dubbed"; (c) (b) or the abstract defines the name as a
   new acronym inside such a sentence. Scored on the test half: P / R / F1 for PROPOSES,
   and accuracy.

The LLM labels the pairs, so it is not one of the compared methods here.

Run: .venv/bin/python scripts/bt3_relations.py --level small
"""

import argparse
import json
import random
import re
from datetime import datetime

from paper_search.config import PROJECT_ROOT, settings
from paper_search.eval.llm_label import chooser
from paper_search.extraction.entities import normalize_entity

PAIRS = PROJECT_ROOT / "data/annotations/method_roles.jsonl"
SAMPLE = 300

PROMPT = """Abstract of a computer science paper:

{abstract}

The abstract mentions the method "{name}". Is "{name}" a new method that this paper \
proposes as its own contribution, or an existing method that the paper uses, builds on \
or compares against?

Answer A if the paper proposes it, B if it is an existing method."""

_SUBJECT = r"(?:\bwe\b|\bthis (?:paper|work|study|article)\b)"
_VERB = r"(?:propos|present|introduc|develop|design|devis|build|creat|formulat|offer)\w*"
PROPOSE = re.compile(rf"{_SUBJECT}[^.;]{{0,60}}?\b{_VERB}", re.IGNORECASE)
NAMING = re.compile(r"\b(?:our|called|named|dubbed|termed|coined)\s+$", re.IGNORECASE)
#: Between the verb and the name: a word here means the name is something the proposed
#: method uses or is compared with ("we propose X that uses BERT"), not X itself.
_DETOUR = re.compile(
    r"\b(?:uses?|using|based|with|than|over|like|such|including|via|leverag\w*|employ\w*"
    r"|from|upon|combin\w*|integrat\w*|extend\w*|and|by|on|to|of|for|in|which|that)\b",
    re.IGNORECASE,
)


def sentences(text: str) -> list[str]:
    """Abstract split at sentence-final punctuation followed by a space."""
    return re.split(r"(?<=[.!?])\s+", text)


def patterns(abstract: str, name: str) -> bool:
    """PROPOSES if a sentence names it after a propose verb, or right after a naming word."""
    for s in sentences(abstract):
        at = s.lower().find(name.lower())
        if at < 0:
            continue
        verb = PROPOSE.search(s)
        near = verb and verb.end() <= at and not _DETOUR.search(s[verb.end() : at])
        if near or NAMING.search(s[max(0, at - 12) : at]):
            return True
    return False


def patterns_or_new_acronym(abstract: str, name: str) -> bool:
    """`patterns`, or the name is defined as an acronym in a propose sentence."""
    if patterns(abstract, name):
        return True
    for s in sentences(abstract):
        if PROPOSE.search(s) and re.search(rf"\(\s*{re.escape(name)}\s*\)", s):
            return True
        acronym = re.search(r"\(([A-Z][A-Za-z0-9-]{1,12})\)", name)
        if acronym and PROPOSE.search(s) and acronym.group(0) in s:
            return True
    return False


def prf(pred: list[bool], gold: list[bool]) -> dict:
    """P / R / F1 of PROPOSES, and accuracy."""
    tp = sum(p and g for p, g in zip(pred, gold, strict=True))
    p = tp / max(sum(pred), 1)
    r = tp / max(sum(gold), 1)
    acc = sum(p == g for p, g in zip(pred, gold, strict=True)) / len(gold)
    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(2 * p * r / max(p + r, 1e-9), 3),
        "accuracy": round(acc, 3),
    }


def main() -> None:
    """Sample and label pairs (cached), then score the methods."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", default="small")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    papers = {
        p["paper_id"]: p
        for p in map(json.loads, (settings.processed_dir / f"papers_{args.level}.jsonl").open())
    }
    if not PAIRS.exists():
        found = json.loads(
            (settings.processed_dir / "kg" / f"entities_{args.level}.json").read_text()
        )
        rng = random.Random(42)
        todo = []
        for pid in rng.sample(sorted(found), len(found)):
            methods = sorted({n for t, n in found[pid] if t == "Method" and normalize_entity(n)})
            for name in rng.sample(methods, min(2, len(methods))):
                todo.append({"paper_id": pid, "method": name})
            if len(todo) >= SAMPLE:
                break
        todo = todo[:SAMPLE]
        choose = chooser(args.model)
        text = " ".join
        probs = choose(
            [
                PROMPT.format(
                    abstract=text(papers[r["paper_id"]]["abstract"].split()), name=r["method"]
                )
                for r in todo
            ],
            ["A", "B"],
        )
        with PAIRS.open("w", encoding="utf-8") as f:
            for n, (r, (a, _)) in enumerate(zip(todo, probs, strict=True)):
                row = {
                    **r,
                    "proposes": a > 0.5,
                    "p_proposes": a,
                    "split": "dev" if n % 2 == 0 else "test",
                }
                row["source"] = args.model
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"labelled {len(todo)} pairs into {PAIRS}")

    rows = [json.loads(line) for line in PAIRS.open(encoding="utf-8")]
    methods = {
        "all_uses (KG baseline)": lambda a, n: False,
        "patterns": patterns,
        "patterns_or_new_acronym": patterns_or_new_acronym,
    }
    result = {
        "level": args.level,
        "labeller": rows[0]["source"],
        "pairs": len(rows),
        "proposes": sum(r["proposes"] for r in rows),
        "methods": {},
    }
    for name, f in methods.items():
        for split in ("dev", "test"):
            part = [r for r in rows if r["split"] == split]
            pred = [
                f(" ".join(papers[r["paper_id"]]["abstract"].split()), r["method"]) for r in part
            ]
            result["methods"].setdefault(name, {})[split] = prf(pred, [r["proposes"] for r in part])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = PROJECT_ROOT / "experiments/runs/bt3" / f"{stamp}_{args.level}"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
