"""Ranking metrics for BT6-BT8. Every retrieval conclusion rests on these, hence the tests.

qrels give each judged paper a grade: 0 not relevant, 1 partly, 2 highly relevant
(experiment-plan.md). Recall and MRR count any grade >= 1 as relevant; nDCG uses the
grades as linear gains, as trec_eval and BEIR do.
"""

import math
from statistics import mean


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """Share of the relevant papers found in the top k."""
    relevant = {p for p, g in qrels.items() if g > 0}
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """1 / rank of the first relevant paper in the top k, else 0."""
    return next((1 / i for i, p in enumerate(ranked[:k], 1) if qrels.get(p, 0) > 0), 0.0)


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """Graded nDCG: DCG of the ranking over DCG of the best possible ranking."""
    dcg = sum(qrels.get(p, 0) / math.log2(i + 1) for i, p in enumerate(ranked[:k], 1))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    return dcg / sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict:
    """Mean metrics over the queries that have at least one relevant paper.

    A query with no relevant paper has no defined recall or nDCG; it is left out and
    counted, rather than scored 0 and silently dragging every system down alike.
    """
    judged = [q for q in run if any(g > 0 for g in qrels.get(q, {}).values())]
    if not judged:
        return {"queries": 0, "skipped": len(run)}
    return {
        "queries": len(judged),
        "skipped": len(run) - len(judged),
        "ndcg@10": round(mean(ndcg_at_k(run[q], qrels[q], 10) for q in judged), 4),
        "recall@10": round(mean(recall_at_k(run[q], qrels[q], 10) for q in judged), 4),
        "recall@100": round(mean(recall_at_k(run[q], qrels[q], 100) for q in judged), 4),
        "mrr@10": round(mean(mrr_at_k(run[q], qrels[q], 10) for q in judged), 4),
    }
