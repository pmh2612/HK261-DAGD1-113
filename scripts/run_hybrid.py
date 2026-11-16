"""BT8: hybrid search - RRF and linear fusion of BM25 with dense, then cross-encoder rerank.

Components (BM25 and dense scorers) are built once; every system in the config is a
fusion or rerank of their score matrices, scored like BT6/BT7 (scripts/run_search.py).
The linear weight alpha is tuned on the train split where one exists (SciFact) and
reported on test; otherwise the config's `alpha` is used. The component retrievers run
again in the same run, so every p-value compares systems on identical queries.

KG-augmented search (plan method 3) needs the knowledge graph of BT5 and is not here.

Run: .venv/bin/python scripts/run_hybrid.py --dataset scifact --config configs/bt8_hybrid.yaml
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import ttest_rel

from paper_search.config import PROJECT_ROOT
from paper_search.eval.metrics import evaluate, ndcg_at_k
from paper_search.retrieval import hybrid, rankers
from run_search import DEPTH, load  # scripts/ is on sys.path when this file runs


def build(spec: dict, docs: list[dict]) -> rankers.Scorer:
    """One component retriever as a scorer."""
    if spec["kind"] == "bm25":
        return rankers.bm25_scores(docs, spec["field"], spec["preprocess"], spec["k1"], spec["b"])
    return rankers.dense_scores(
        docs, spec["id"], spec.get("query_prefix", ""), spec.get("doc_prefix", "")
    )


def to_run(scores: np.ndarray, docs: list[dict], qids: list[str]) -> dict[str, list[str]]:
    """Score matrix -> {qid: top-DEPTH paper_ids}."""
    top = np.argsort(-scores, axis=1, kind="stable")[:, :DEPTH]
    return {q: [docs[i]["paper_id"] for i in row] for q, row in zip(qids, top, strict=True)}


def main() -> None:
    """Build components, combine them, score every system, write one run dir each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    docs, queries, qrels, train = load(args.dataset)
    qids, texts = list(queries), list(queries.values())
    print(f"{args.dataset}: {len(docs)} documents, {len(queries)} queries")

    test, train_scores, cost = {}, {}, {}
    for name, spec in cfg["components"].items():
        start = time.perf_counter()
        score = build(spec, docs)
        index_s = time.perf_counter() - start
        start = time.perf_counter()
        test[name] = score(texts)
        cost[name] = {
            "index_seconds": round(index_s, 2),
            "ms_per_query": round((time.perf_counter() - start) * 1000 / len(texts), 2),
        }
        if train:
            train_scores[name] = score(list(train[0].values()))
        print(f"component {name}: {cost[name]}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    systems, results, per_query = {}, {}, {}
    for system in cfg["systems"]:
        name, kind = system["name"], system["kind"]
        start = time.perf_counter()
        extra = {}
        if kind == "single":
            scores, parts = test[system["part"]], [system["part"]]
        elif kind == "rrf":
            parts = system["parts"]
            scores = hybrid.rrf(*(test[p] for p in parts), k=system["k"])
        elif kind == "linear":
            parts = [system["sparse"], system["dense"]]
            alpha = system.get("alpha")
            if train:  # tune alpha on train, never on test
                grid = []
                for a in cfg["alphas"]:
                    mixed = hybrid.linear(train_scores[parts[0]], train_scores[parts[1]], a)
                    run = to_run(mixed, docs, list(train[0]))
                    grid.append((evaluate(run, train[1])["ndcg@10"], a))
                best, alpha = max(grid)
                extra = {
                    "alpha": alpha,
                    "train_ndcg@10": best,
                    "train_grid": {a: s for s, a in grid},
                }
            scores = hybrid.linear(test[parts[0]], test[parts[1]], alpha)
        elif kind == "rerank":
            base = systems[system["base"]]
            parts = base["parts"]
            scores = hybrid.rerank(base["scores"], texts, docs, system["model"], system["depth"])
        else:
            raise ValueError(f"unknown kind {kind}")
        own_ms = (time.perf_counter() - start) * 1000 / len(texts)
        systems[name] = {"scores": scores, "parts": parts}

        run = to_run(scores, docs, qids)
        results[name] = {
            **evaluate(run, qrels),
            # Query time: the components it needs plus its own fusion / rerank step.
            "ms_per_query": round(sum(cost[p]["ms_per_query"] for p in parts) + own_ms, 2),
            "own_ms_per_query": round(own_ms, 2),
            **extra,
        }
        per_query[name] = {
            q: ndcg_at_k(r, qrels[q], 10)
            for q, r in run.items()
            if any(g > 0 for g in qrels.get(q, {}).values())
        }
        out = PROJECT_ROOT / "experiments/runs" / cfg["task"] / f"{stamp}_{args.dataset}_{name}"
        out.mkdir(parents=True)
        components = {p: cfg["components"][p] for p in parts}
        (out / "config.yaml").write_text(
            yaml.safe_dump(
                {"dataset": args.dataset, "system": system, "components": components},
                sort_keys=False,
            )
        )
        print(f"{name}: nDCG@10 {results[name]['ndcg@10']}", flush=True)
        systems[name]["out"] = out

    # Paired t-tests on per-query nDCG@10, as in BT6/BT7: against the best system, and
    # against the best single retriever (does fusion help at all?).
    top = max(results, key=lambda n: results[n]["ndcg@10"])
    singles = [s["name"] for s in cfg["systems"] if s["kind"] == "single"]
    best_single = max(singles, key=lambda n: results[n]["ndcg@10"])
    qs = sorted(per_query[top])
    for name in results:
        for ref, key in ((top, "p_vs_best"), (best_single, f"p_vs_{best_single}")):
            if name != ref:
                a = [per_query[ref][q] for q in qs]
                b = [per_query[name][q] for q in qs]
                results[name][key] = round(float(ttest_rel(a, b).pvalue), 4)
        out = systems[name]["out"]
        (out / "metrics.json").write_text(json.dumps(results[name], indent=2) + "\n")
        (out / "per_query_ndcg@10.json").write_text(json.dumps(per_query[name]) + "\n")

    print(
        f"\n{'system':<34} {'nDCG@10':>8} {'R@10':>7} {'R@100':>7} {'MRR@10':>7} {'ms/q':>7}"
        f"  p vs {best_single}"
    )
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["ndcg@10"]):
        print(
            f"{name:<34} {m['ndcg@10']:>8.4f} {m['recall@10']:>7.4f} {m['recall@100']:>7.4f} "
            f"{m['mrr@10']:>7.4f} {m['ms_per_query']:>7.1f}  {m.get(f'p_vs_{best_single}', '-')}"
            + (f"  alpha={m['alpha']}" if "alpha" in m else "")
        )


if __name__ == "__main__":
    main()
