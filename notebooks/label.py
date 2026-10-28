"""Terminal labelling for the evaluation set - faster and safer than editing JSONL by hand.

    add "query text"        append a query to queries.jsonl
    queries                 judge each unjudged query's pool, one candidate at a time
    entities [--file F]     annotate abstracts with Method / Dataset / Task
    agree A B               compare two annotators' entity files (the 10-abstract round)

Progress is saved after every finished query or abstract, so quitting (q) loses at most
the item in front of you. Files are replaced atomically: an interrupted write never
leaves a half-written label file. Guidelines: data/eval/README.md.

Run: .venv/bin/python notebooks/label.py <command>
"""

import argparse
import json
import os
import textwrap
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"
TYPES = {"m": "Method", "d": "Dataset", "t": "Task"}


def read_jsonl(path: Path) -> list[dict]:
    """All rows of a JSONL file."""
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write to a temporary file, then rename over the original."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    os.replace(tmp, path)


def show(text: str) -> None:
    """Print wrapped text, indented."""
    print(textwrap.indent(textwrap.fill(text, 96), "    "))


def add(query: str) -> None:
    """Append a new, unjudged query with the next free qid."""
    path = EVAL_DIR / "queries.jsonl"
    rows = read_jsonl(path)
    qid = f"q{max(int(r['qid'][1:]) for r in rows) + 1:02d}"
    write_jsonl(path, [*rows, {"qid": qid, "query": query, "relevant": None}])
    print(f"added {qid}. Before judging it, rebuild the pools:")
    print("  .venv-specter2/bin/python notebooks/encode_specter2.py")
    print("  .venv/bin/python notebooks/exp_search.py --pool")


def judge_queries() -> None:
    """Judge pool candidates y / n; `relevant` is written once a query's pool is done."""
    path = EVAL_DIR / "queries.jsonl"
    abstracts = {r["paper_id"]: r for r in read_jsonl(EVAL_DIR / "abstracts.jsonl")}
    pools = {r["qid"]: r for r in read_jsonl(EVAL_DIR / "pool.jsonl")}
    rows = read_jsonl(path)
    for row in rows:
        if row["relevant"] is not None:
            continue
        pool = pools.get(row["qid"])
        if not pool or pool["query"] != row["query"]:
            print(f"{row['qid']}: no pool for this query yet - run exp_search.py --pool")
            continue
        relevant = []
        cands = pool["candidates"]
        for i, cand in enumerate(cands, 1):
            paper = abstracts[cand["paper_id"]]
            print(f"\n[{row['qid']}] {row['query']}    candidate {i}/{len(cands)}")
            print(f"  {paper['title']} ({paper['year']})")
            show(paper["abstract"])
            answer = ""
            while answer not in {"y", "n", "q"}:
                answer = input("  relevant? [y/n, q = quit] ").strip().lower()
            if answer == "q":
                print("stopped; this query was not saved")
                return
            if answer == "y":
                relevant.append(cand["paper_id"])
        row["relevant"] = relevant
        write_jsonl(path, rows)
        print(f"saved {row['qid']}: {len(relevant)} relevant of {len(cands)}")
    print("all queries with a pool are judged")


def annotate_entities(path: Path) -> None:
    """Type one entity per line as `m: BERT`, `d: HotpotQA` or `t: question answering`."""
    rows = read_jsonl(path)
    todo = [r for r in rows if r["entities"] is None]
    print(f"{len(todo)} of {len(rows)} abstracts left in {path.name}")
    print("one entity per line: m: <method> / d: <dataset> / t: <task>")
    print("empty line = done with this abstract, `none` = it has no entities, q = quit")
    for n, row in enumerate(todo, 1):
        print(f"\n--- {n}/{len(todo)}  {row['paper_id']}")
        show(row["abstract"])
        entities: list[dict] = []
        while True:
            line = input("  > ").strip()
            if line == "q":
                print("stopped; this abstract was not saved")
                return
            if line in {"", "none"}:
                if line == "" and not entities:
                    print("  nothing entered - type `none` if the abstract has no entities")
                    continue
                break
            kind, _, text = line.partition(":")
            text = text.strip()
            if kind.strip().lower() not in TYPES or not text:
                print("  format: m: <name>, d: <name> or t: <name>")
            elif text not in row["abstract"]:
                print(f"  {text!r} is not in the abstract verbatim - copy it exactly")
            else:
                entities.append({"text": text, "type": TYPES[kind.strip().lower()]})
                print(f"  + {entities[-1]['type']}: {text}")
        row["entities"] = entities
        write_jsonl(path, rows)
        print(f"  saved {len(entities)} entities")
    print("all abstracts annotated")


def agree(a: Path, b: Path) -> None:
    """Entity-level F1 between two annotators on the abstracts both have annotated."""
    from exp_extraction import TYPES as ENTITY_TYPES  # same normalization as the scorer
    from exp_extraction import as_set, prf

    rows_a = {r["paper_id"]: r["entities"] for r in read_jsonl(a) if r["entities"] is not None}
    rows_b = {r["paper_id"]: r["entities"] for r in read_jsonl(b) if r["entities"] is not None}
    shared = sorted(rows_a.keys() & rows_b.keys())
    print(f"{len(shared)} abstracts annotated in both\n")
    for typ in (*ENTITY_TYPES, None):
        sa = {(p, *e) for p in shared for e in as_set(rows_a[p]) if typ in (None, e[0])}
        sb = {(p, *e) for p in shared for e in as_set(rows_b[p]) if typ in (None, e[0])}
        # Both empty is agreement, not an F1 of 0 - say n/a rather than mislead.
        f1 = f"{prf(sa, sb)[2]:.2f}" if sa or sb else " n/a"
        print(
            f"  {typ or 'All':<8} F1 {f1}   ({len(sa & sb)} shared, "
            f"{len(sa - sb)} only in {a.name}, {len(sb - sa)} only in {b.name})"
        )
    for p in shared:
        diff = as_set(rows_a[p]) ^ as_set(rows_b[p])
        if diff:
            print(f"\n{p}:")
            for typ, text in sorted(diff):
                who = a.name if (typ, text) in as_set(rows_a[p]) else b.name
                print(f"  {typ:<8} {text:<40} only in {who}")


def main() -> None:
    """Dispatch the subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("add").add_argument("query")
    sub.add_parser("queries")
    ent = sub.add_parser("entities")
    ent.add_argument("--file", type=Path, default=EVAL_DIR / "entities.jsonl")
    agr = sub.add_parser("agree")
    agr.add_argument("a", type=Path)
    agr.add_argument("b", type=Path)
    args = parser.parse_args()

    if args.command == "add":
        add(args.query)
    elif args.command == "queries":
        judge_queries()
    elif args.command == "entities":
        annotate_entities(args.file)
    else:
        agree(args.a, args.b)


if __name__ == "__main__":
    main()
