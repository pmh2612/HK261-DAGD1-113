"""BT2 error analysis: what kind of mistakes each run makes, with examples.

Reads the predictions.jsonl of BT2 run directories, classifies every non-exact
prediction and missed entity (paper_search.eval.ner.errors), prints the counts and a
few examples of each kind in their sentence context.

Run: .venv/bin/python scripts/ner_errors.py experiments/runs/bt2/<run> [...] [--examples 3]
"""

import argparse
import json
import random
from pathlib import Path

from paper_search.config import settings
from paper_search.eval.ner import errors, load_scierc


def main() -> None:
    """Counts per error kind, then examples."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    test = load_scierc(settings.raw_dir / "scierc" / "test.json")
    gold = {d["doc_id"]: d["entities"] for d in test}
    text = {d["doc_id"]: d["text"] for d in test}

    for run in args.runs:
        rows = [json.loads(line) for line in (run / "predictions.jsonl").open()]
        pred = {r["doc_id"]: [tuple(e[:3]) for e in r["entities"]] for r in rows}
        found = errors(gold, pred)
        print(f"\n== {run.name}: " + ", ".join(f"{k} {len(v)}" for k, v in found.items()))
        rng = random.Random(42)
        for kind, items in found.items():
            for doc_id, (s, e, t) in rng.sample(items, min(args.examples, len(items))):
                truth = [x for x in gold[doc_id] if x[0] < e and s < x[1]]
                context = text[doc_id][max(0, s - 50) : e + 50].replace("\n", " ")
                print(
                    f"  {kind:<9} {t:<8} '{text[doc_id][s:e]}'  gold: "
                    + (", ".join(f"{x[2]} '{text[doc_id][x[0] : x[1]]}'" for x in truth) or "-")
                )
                print(f"            …{context}…")


if __name__ == "__main__":
    main()
