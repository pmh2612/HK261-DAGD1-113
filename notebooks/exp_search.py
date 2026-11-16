"""Search experiment: BM25 vs MiniLM vs bge-small vs SPECTER2 on the 200-abstract eval set.

bge-small is not in the experiment plan but is the model DECIDE-5 pinned, and DECIDE-5
asks for exactly this comparison, so it runs alongside.

Two modes:

    --pool   for every query, write the union of each system's top-10 to
             data/eval/pool.jsonl. Judges read that, then put the relevant paper_ids
             into queries.jsonl. Pooling is how TREC builds judgments without reading
             all 200 abstracts per query - and it judges every system's hits, not one's.
    (none)   score every judged query (relevant != null): Recall@10, MRR@10, nDCG@10.

SPECTER2 vectors come from notebooks/encode_specter2.py (separate venv); re-run it after
editing queries.jsonl.

Run: .venv/bin/python notebooks/exp_search.py [--pool]
"""

import json
import math
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from paper_search.config import PROJECT_ROOT, settings

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
SPECTER2_DIR = PROJECT_ROOT / "data" / "processed" / "specter2"
K = 10


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens; hyphenated terms split ("retrieval-augmented")."""
    return re.findall(r"\w+", text.lower())


def bm25_ranker(docs: list[dict]):
    """Rank function over BM25 on title + abstract."""
    bm25 = BM25Okapi([tokenize(d["title"] + " " + d["abstract"]) for d in docs])
    return lambda queries: [np.argsort(-bm25.get_scores(tokenize(q)))[:K] for q in queries]


def dense_ranker(docs: list[dict], model_name: str, query_prefix: str = ""):
    """Rank function over cosine similarity of sentence-transformers embeddings."""
    model = SentenceTransformer(model_name)
    doc_vecs = model.encode(
        [d["title"] + ". " + d["abstract"] for d in docs], normalize_embeddings=True
    )

    def rank(queries):
        q_vecs = model.encode([query_prefix + q for q in queries], normalize_embeddings=True)
        return [np.argsort(-(doc_vecs @ q))[:K] for q in q_vecs]

    return rank


def specter2_ranker(docs: list[dict]):
    """Rank function over precomputed SPECTER2 vectors; refuses stale ones."""
    ids = json.loads((SPECTER2_DIR / "ids.json").read_text())
    doc_vecs = np.load(SPECTER2_DIR / "docs.npy")
    q_vecs = dict(zip(ids["queries"], np.load(SPECTER2_DIR / "queries.npy"), strict=True))
    if ids["docs"] != [d["paper_id"] for d in docs]:
        sys.exit("SPECTER2 doc vectors are stale - re-run notebooks/encode_specter2.py")

    def rank(queries):
        if missing := [q for q in queries if q not in q_vecs]:
            sys.exit(f"no SPECTER2 vector for {missing} - re-run notebooks/encode_specter2.py")
        return [np.argsort(-(doc_vecs @ q_vecs[q]))[:K] for q in queries]

    return rank


def recall(ranked: list[str], relevant: set[str]) -> float:
    """Share of relevant papers found in the top K."""
    return len(set(ranked[:K]) & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant paper in the top K, else 0."""
    return next((1 / i for i, p in enumerate(ranked[:K], 1) if p in relevant), 0.0)


def ndcg(ranked: list[str], relevant: set[str]) -> float:
    """Binary-relevance nDCG@K."""
    dcg = sum(1 / math.log2(i + 1) for i, p in enumerate(ranked[:K], 1) if p in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), K) + 1))
    return dcg / ideal


def _check_metrics() -> None:
    """Hand-computed cases, so a broken metric cannot produce a report table."""
    assert recall(["a", "b", "c"], {"b", "z"}) == 0.5
    assert mrr(["a", "b", "c"], {"c"}) == 1 / 3 and mrr(["a"], {"z"}) == 0.0
    assert ndcg(["a", "b"], {"a", "b"}) == 1.0
    assert abs(ndcg(["x", "a"], {"a"}) - 1 / math.log2(3)) < 1e-9


def main() -> None:
    """Pool candidates for judging, or score the judged queries."""
    _check_metrics()
    docs = [json.loads(line) for line in (EVAL_DIR / "abstracts.jsonl").open()]
    queries = [json.loads(line) for line in (EVAL_DIR / "queries.jsonl").open()]
    pooling = "--pool" in sys.argv
    if not pooling:
        queries = [q for q in queries if q["relevant"]]
        if not queries:
            sys.exit("no judged queries yet - run with --pool, then fill in `relevant`")

    systems = {
        "BM25": bm25_ranker(docs),
        "MiniLM": dense_ranker(docs, "sentence-transformers/all-MiniLM-L6-v2"),
        "bge-small": dense_ranker(docs, settings.embedding_model, settings.embedding_query_prefix),
        "SPECTER2": specter2_ranker(docs),
    }
    texts = [q["query"] for q in queries]
    runs = {
        name: [[docs[i]["paper_id"] for i in top] for top in rank(texts)]
        for name, rank in systems.items()
    }

    if pooling:
        title = {d["paper_id"]: d["title"] for d in docs}
        with (EVAL_DIR / "pool.jsonl").open("w", encoding="utf-8") as f:
            for n, q in enumerate(queries):
                pool = sorted({p for run in runs.values() for p in run[n]})
                cands = [{"paper_id": p, "title": title[p]} for p in pool]
                f.write(json.dumps({**q, "candidates": cands}, ensure_ascii=False) + "\n")
        print(f"wrote judging pools for {len(queries)} queries to {EVAL_DIR / 'pool.jsonl'}")
        return

    print(f"{len(queries)} judged queries, top-{K}\n")
    print(f"{'system':<10} {'Recall':>7} {'MRR':>7} {'nDCG':>7}")
    for name, run in runs.items():
        rel = [set(q["relevant"]) for q in queries]
        scores = [
            np.mean([metric(r, g) for r, g in zip(run, rel, strict=True)])
            for metric in (recall, mrr, ndcg)
        ]
        print(f"{name:<10} " + " ".join(f"{s:>7.3f}" for s in scores))


if __name__ == "__main__":
    main()
