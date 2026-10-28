"""BT6 / BT7: run the retrieval systems in one or more configs and score them.

Datasets:
    scifact   BEIR SciFact (5,183 abstracts, 300 test queries with judgments), the public
              sanity check. BM25 picks its field / preprocessing and tunes k1, b on the
              train split; every number reported is on the test split.
    tiny, …   data/processed/papers_<level>.jsonl with our own queries and judgments in
              data/annotations/ (queries.jsonl, retrieval_qrels.jsonl).

--pool writes, for every query, the union of each system's top 20 to
data/annotations/retrieval_pool.jsonl - what the judges grade 0/1/2 (TREC pooling).

SPECTER2 needs its own venv. The first run writes its inputs and skips it; then run
    .venv-specter2/bin/python notebooks/encode_specter2.py --dir data/processed/retrieval/<dataset>
and run this again.

Run: .venv/bin/python scripts/run_search.py --dataset scifact \
         --config configs/bt6_bm25.yaml configs/bt7_dense.yaml
"""

import argparse
import csv
import itertools
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import ttest_rel

from paper_search.config import PROJECT_ROOT, settings
from paper_search.eval.metrics import evaluate, ndcg_at_k
from paper_search.retrieval import rankers

ANNOTATIONS = PROJECT_ROOT / "data" / "annotations"
DEPTH = 100  # deep enough for Recall@100
POOL_DEPTH = 20


def jsonl(path: Path) -> list[dict]:
    """All rows of a JSONL file."""
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def load(dataset: str) -> tuple[list[dict], dict, dict, tuple[dict, dict] | None]:
    """Documents, test queries {qid: text}, qrels {qid: {paper_id: grade}}, train split."""
    if dataset == "scifact":
        d = settings.raw_dir / "beir" / "scifact"
        docs = [
            {"paper_id": r["_id"], "title": r["title"], "abstract": r["text"]}
            for r in jsonl(d / "corpus.jsonl")
        ]
        texts = {r["_id"]: r["text"] for r in jsonl(d / "queries.jsonl")}

        def split(name: str) -> tuple[dict, dict]:
            qrels: dict[str, dict[str, int]] = defaultdict(dict)
            with (d / "qrels" / f"{name}.tsv").open() as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
            return {q: texts[q] for q in qrels}, dict(qrels)

        return docs, *split("test"), split("train")

    docs = [
        {"paper_id": p["paper_id"], "title": p["title"], "abstract": p["abstract"]}
        for p in jsonl(settings.processed_dir / f"papers_{dataset}.jsonl")
    ]
    queries, qrels = {}, defaultdict(dict)
    if (ANNOTATIONS / "queries.jsonl").exists():
        queries = {r["qid"]: r["query"] for r in jsonl(ANNOTATIONS / "queries.jsonl")}
    if (ANNOTATIONS / "retrieval_qrels.jsonl").exists():
        for r in jsonl(ANNOTATIONS / "retrieval_qrels.jsonl"):
            qrels[r["qid"]][r["paper_id"]] = r["rel"]
    return docs, queries, dict(qrels), None


def specs(cfg: dict) -> list[tuple[str, dict]]:
    """(system name, parameters) for every system a config asks for."""
    if "bm25" in cfg:
        g = cfg["bm25"]
        return [
            (
                f"bm25_{f}_{p}",
                {"kind": "bm25", "field": f, "preprocess": p, "k1": g["k1"], "b": g["b"]},
            )
            for f, p in itertools.product(g["fields"], g["preprocess"])
        ]
    return [(m["name"], {"kind": "dense", **m}) for m in cfg["models"]]


def build(spec: dict, docs: list[dict], vec_dir: Path) -> rankers.Rank | None:
    """The ranker for one system; None when its precomputed vectors are missing or stale."""
    if spec["kind"] == "bm25":
        return rankers.bm25(docs, spec["field"], spec["preprocess"], spec["k1"], spec["b"])
    if not spec.get("precomputed"):
        return rankers.dense(
            docs, spec["id"], spec.get("query_prefix", ""), spec.get("doc_prefix", "")
        )
    ids_path = vec_dir / "ids.json"
    if not ids_path.exists():
        return None
    ids = json.loads(ids_path.read_text())
    if ids["docs"] != [d["paper_id"] for d in docs]:
        return None
    query_vecs = dict(zip(ids["queries"], np.load(vec_dir / "queries.npy"), strict=True))
    return rankers.precomputed(np.load(vec_dir / "docs.npy"), query_vecs)


def run(rank: rankers.Rank, docs: list[dict], queries: dict[str, str]) -> tuple[dict, float]:
    """{qid: ranked paper_ids} and milliseconds per query."""
    start = time.perf_counter()
    top = rank(list(queries.values()), DEPTH)
    ms = (time.perf_counter() - start) * 1000 / max(len(queries), 1)
    return {
        q: [docs[i]["paper_id"] for i in r if i >= 0] for q, r in zip(queries, top, strict=True)
    }, ms


def main() -> None:
    """Score every system, tune BM25 where a train split exists, write runs and pools."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="tiny")
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--pool", action="store_true")
    args = parser.parse_args()

    docs, queries, qrels, train = load(args.dataset)
    if not queries:
        raise SystemExit(f"no queries for {args.dataset}: write {ANNOTATIONS / 'queries.jsonl'}")
    vec_dir = settings.processed_dir / "retrieval" / args.dataset
    vec_dir.mkdir(parents=True, exist_ok=True)
    # Inputs for notebooks/encode_specter2.py, which cannot import this package.
    (vec_dir / "docs.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (vec_dir / "queries.jsonl").write_text(
        "".join(json.dumps({"qid": q, "query": t}) + "\n" for q, t in queries.items())
    )
    print(f"{args.dataset}: {len(docs)} documents, {len(queries)} queries")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results: dict[str, dict] = {}
    per_query: dict[str, dict[str, float]] = {}
    out_dirs: dict[str, Path] = {}
    pools: dict[str, set[str]] = defaultdict(set)
    for config_path in args.config:
        cfg = yaml.safe_load(config_path.read_text())
        systems = specs(cfg)
        if "bm25" in cfg and train:
            # Choose field + preprocessing on train, then tune k1 / b on train as well.
            train_score = {}
            for name, spec in systems:
                ranked, _ = run(build(spec, docs, vec_dir), docs, train[0])
                train_score[name] = evaluate(ranked, train[1])["ndcg@10"]
            best_name, best = max(systems, key=lambda s: train_score[s[0]])
            grid = []
            for k1, b in itertools.product(cfg["tune"]["k1"], cfg["tune"]["b"]):
                spec = {**best, "k1": k1, "b": b}
                ranked, _ = run(build(spec, docs, vec_dir), docs, train[0])
                grid.append((evaluate(ranked, train[1])["ndcg@10"], k1, b))
            score, k1, b = max(grid)
            print(f"tuned on train: {best_name} k1={k1} b={b} (train nDCG@10 {score:.4f})")
            systems.append(
                (f"{best_name}_tuned", {**best, "k1": k1, "b": b, "train_ndcg@10": score})
            )

        for name, spec in systems:
            start = time.perf_counter()
            rank = build(spec, docs, vec_dir)
            if rank is None:
                print(f"{name}: skipped - run notebooks/encode_specter2.py --dir {vec_dir}")
                continue
            index_s = time.perf_counter() - start
            ranked, ms = run(rank, docs, queries)
            for q, r in ranked.items():
                pools[q] |= set(r[:POOL_DEPTH])
            metrics = {
                **evaluate(ranked, qrels),
                "index_seconds": round(index_s, 2),
                "ms_per_query": round(ms, 2),
            }
            results[name] = metrics
            per_query[name] = {
                q: ndcg_at_k(r, qrels[q], 10)
                for q, r in ranked.items()
                if any(g > 0 for g in qrels.get(q, {}).values())
            }
            out = PROJECT_ROOT / "experiments/runs" / cfg["task"] / f"{stamp}_{args.dataset}_{name}"
            out.mkdir(parents=True)
            (out / "config.yaml").write_text(
                yaml.safe_dump({"dataset": args.dataset, "system": name, **spec}, sort_keys=False)
            )
            out_dirs[name] = out

    # Paired t-test on per-query nDCG@10 against the best system: is the gap more than
    # query-to-query noise? No multiple-comparison correction - read p near 0.05 as weak.
    judged = [n for n in results if results[n].get("queries")]
    if judged:
        top = max(judged, key=lambda n: results[n]["ndcg@10"])
        qs = sorted(per_query[top])
        for name in judged:
            if name != top:
                a = [per_query[top][q] for q in qs]
                b = [per_query[name][q] for q in qs]
                results[name]["p_vs_best"] = round(float(ttest_rel(a, b).pvalue), 4)
    for name, out in out_dirs.items():
        (out / "metrics.json").write_text(json.dumps(results[name], indent=2) + "\n")
        (out / "per_query_ndcg@10.json").write_text(json.dumps(per_query[name]) + "\n")

    print(
        f"\n{'system':<36} {'nDCG@10':>8} {'R@10':>7} {'R@100':>7} {'MRR@10':>7}"
        f" {'ms/q':>7} {'p vs best':>9}"
    )
    for name, m in sorted(results.items(), key=lambda kv: -kv[1].get("ndcg@10", 0)):
        cells = [m.get(k) for k in ("ndcg@10", "recall@10", "recall@100", "mrr@10")]
        print(
            f"{name:<36} "
            + " ".join(f"{c:>7.4f}" if c is not None else "      -" for c in cells)
            + f" {m['ms_per_query']:>7.1f} {m.get('p_vs_best', '-'):>9}"
        )

    if args.pool:
        title = {d["paper_id"]: d["title"] for d in docs}
        rows = [
            {
                "qid": q,
                "query": queries[q],
                "candidates": [{"paper_id": p, "title": title[p]} for p in sorted(pools[q])],
            }
            for q in queries
        ]
        path = ANNOTATIONS / "retrieval_pool.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        sizes = [len(r["candidates"]) for r in rows]
        print(f"\nwrote pools to {path}: {min(sizes)}-{max(sizes)} candidates per query")


if __name__ == "__main__":
    main()
