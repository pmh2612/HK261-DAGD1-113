"""Check the LLM's relevance grades on our corpus against a person's, on a sample.

sample: draws 100 graded pairs (40 grade 2, 30 grade 1, 30 grade 0, seed 42) into a CSV
        for a person to fill in the `human` column with 0/1/2. The LLM's grade is left
        out of the file so it cannot sway the person.
score:  compares the filled-in CSV with the LLM's grades: agreement on "relevant"
        (grade 2, the threshold the qrels use), and weighted kappa on 0/1/2.
sample-pairs / score-pairs: BT4's entity-name pairs, 25 the LLM called the same and 25
        different; `human` is 1 (same) or 0 (different).
sample-roles / score-roles: BT3's method roles, 30 the LLM called proposed by the paper
        and 30 existing; `human` is 1 (proposed) or 0 (existing).

Run: .venv/bin/python scripts/spot_check.py sample
     .venv/bin/python scripts/spot_check.py score --annotator my
"""

import argparse
import csv
import json
import random

from sklearn.metrics import cohen_kappa_score, confusion_matrix

from paper_search.config import PROJECT_ROOT, settings

QRELS = PROJECT_ROOT / "data/annotations/retrieval_qrels.jsonl"
SHEET = PROJECT_ROOT / "data/annotations/spot_check.csv"
STRATA = {2: 40, 1: 30, 0: 30}
PAIRS = PROJECT_ROOT / "data/annotations/entity_pairs.jsonl"
PAIR_SHEET = PROJECT_ROOT / "data/annotations/spot_check_pairs.csv"


def grades() -> dict[tuple[str, str], int]:
    """The LLM's 0/1/2 grade of every judged (qid, paper_id)."""
    rows = [json.loads(line) for line in QRELS.open(encoding="utf-8")]
    return {(r["qid"], r["paper_id"]): r["grade"] for r in rows}


def sample() -> None:
    """Write the sheet to fill in; refuses to overwrite one that has answers."""
    if SHEET.exists() and any(r["human"].strip() for r in csv.DictReader(SHEET.open())):
        raise SystemExit(f"{SHEET} already has answers; not overwriting")
    g = grades()
    rng = random.Random(42)
    picked = [
        k
        for grade, n in STRATA.items()
        for k in rng.sample(sorted(k for k in g if g[k] == grade), n)
    ]
    rng.shuffle(picked)
    queries = {
        r["qid"]: r["query"]
        for r in map(json.loads, (PROJECT_ROOT / "data/annotations/queries.jsonl").open())
    }
    papers = {
        p["paper_id"]: p
        for p in map(json.loads, (settings.processed_dir / "papers_small.jsonl").open())
    }
    with SHEET.open("w", newline="", encoding="utf-8-sig") as f:  # -sig: Excel reads UTF-8
        w = csv.writer(f)
        w.writerow(["n", "qid", "paper_id", "query", "title", "abstract", "human"])
        for n, (q, p) in enumerate(picked, 1):
            w.writerow([n, q, p, queries[q], papers[p]["title"], papers[p]["abstract"], ""])
    print(f"wrote {len(picked)} pairs to {SHEET}")


def score(annotator: str) -> None:
    """Agreement between the filled-in sheet and the LLM."""
    g = grades()
    rows = [r for r in csv.DictReader(SHEET.open(encoding="utf-8-sig")) if r["human"].strip()]
    human = [int(r["human"]) for r in rows]
    llm = [g[(r["qid"], r["paper_id"])] for r in rows]
    rel_h, rel_l = [int(x == 2) for x in human], [int(x == 2) for x in llm]
    hit = sum(a and b for a, b in zip(rel_h, rel_l, strict=True))
    result = {
        "annotator": annotator,
        "pairs": len(rows),
        "relevant_kappa": round(cohen_kappa_score(rel_h, rel_l), 3),
        "llm_precision": round(hit / max(sum(rel_l), 1), 3),  # LLM says 2, person agrees
        "llm_recall": round(hit / max(sum(rel_h), 1), 3),  # person says 2, LLM agrees
        "weighted_kappa_012": round(cohen_kappa_score(human, llm, weights="linear"), 3),
        "confusion_human_rows_llm_cols": confusion_matrix(human, llm, labels=[0, 1, 2]).tolist(),
    }
    out = PROJECT_ROOT / f"experiments/runs/judge/spot_check_{annotator}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


#: Yes/no labels: BT4 name pairs (same entity?) and BT3 method roles (proposed?).
#: `n` of each LLM answer are drawn; `keys` identify a row; `context` adds paper text.
BINARY = {
    "pairs": {
        "source": PAIRS,
        "sheet": PAIR_SHEET,
        "label": "same",
        "n": 25,
        "keys": ("type", "a", "b"),
        "context": False,
        "out": "bt4/spot_check_pairs",
    },
    "roles": {
        "source": PROJECT_ROOT / "data/annotations/method_roles.jsonl",
        "sheet": PROJECT_ROOT / "data/annotations/spot_check_roles.csv",
        "label": "proposes",
        "n": 30,
        "keys": ("paper_id", "method"),
        "context": True,
        "out": "bt3/spot_check_roles",
    },
}


def sample_binary(kind: str) -> None:
    """A yes/no sheet to fill in; refuses to overwrite one that has answers."""
    spec = BINARY[kind]
    sheet = spec["sheet"]
    if sheet.exists() and any(r["human"].strip() for r in csv.DictReader(sheet.open())):
        raise SystemExit(f"{sheet} already has answers; not overwriting")
    rows = [json.loads(line) for line in spec["source"].open(encoding="utf-8")]
    rng = random.Random(42)
    picked = [
        r
        for yes in (True, False)
        for r in rng.sample([r for r in rows if r[spec["label"]] == yes], spec["n"])
    ]
    rng.shuffle(picked)
    papers = {}
    if spec["context"]:
        papers = {
            p["paper_id"]: p
            for p in map(json.loads, (settings.processed_dir / "papers_small.jsonl").open())
        }
    extra = ["title", "abstract"] if spec["context"] else []
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["n", *spec["keys"], *extra, "human"])
        for n, r in enumerate(picked, 1):
            ctx = [papers[r["paper_id"]][k] for k in extra]
            w.writerow([n, *(r[k] for k in spec["keys"]), *ctx, ""])
    print(f"wrote {len(picked)} rows to {sheet}")


def score_binary(kind: str, annotator: str) -> None:
    """Agreement between a filled-in yes/no sheet (1/0) and the LLM's answers."""
    spec = BINARY[kind]
    keys = spec["keys"]
    llm_yes = {
        tuple(r[k] for k in keys): r[spec["label"]]
        for r in map(json.loads, spec["source"].open(encoding="utf-8"))
    }
    rows = [
        r for r in csv.DictReader(spec["sheet"].open(encoding="utf-8-sig")) if r["human"].strip()
    ]
    human = [int(r["human"]) for r in rows]
    llm = [int(llm_yes[tuple(r[k] for k in keys)]) for r in rows]
    hit = sum(a and b for a, b in zip(human, llm, strict=True))
    result = {
        "annotator": annotator,
        "rows": len(rows),
        "kappa": round(cohen_kappa_score(human, llm), 3),
        "llm_precision": round(hit / max(sum(llm), 1), 3),  # LLM says yes, person agrees
        "llm_recall": round(hit / max(sum(human), 1), 3),  # person says yes, LLM agrees
    }
    out = PROJECT_ROOT / f"experiments/runs/{spec['out']}_{annotator}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    kinds = [f"{a}-{k}" for k in BINARY for a in ("sample", "score")]
    parser.add_argument("command", choices=["sample", "score", *kinds])
    parser.add_argument("--annotator", default="my")
    args = parser.parse_args()
    action, _, kind = args.command.partition("-")
    if not kind:
        sample() if action == "sample" else score(args.annotator)
    elif action == "sample":
        sample_binary(kind)
    else:
        score_binary(kind, args.annotator)
