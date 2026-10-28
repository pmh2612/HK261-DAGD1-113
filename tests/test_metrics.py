"""Hand-computed cases for the ranking metrics."""

import math

from pytest import approx

from paper_search.eval.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_counts_any_positive_grade():
    qrels = {"a": 2, "b": 1, "z": 0}
    assert recall_at_k(["a", "x", "b"], qrels, 2) == 0.5
    assert recall_at_k(["a", "x", "b"], qrels, 3) == 1.0


def test_mrr_is_first_relevant_rank():
    assert mrr_at_k(["x", "z", "b"], {"b": 1, "z": 0}, 10) == approx(1 / 3)
    assert mrr_at_k(["x", "b"], {"b": 1}, 1) == 0.0


def test_ndcg_uses_grades():
    qrels = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], qrels, 10) == approx(1.0)
    swapped = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], qrels, 10) == approx(swapped)
    assert ndcg_at_k(["x", "y"], qrels, 10) == 0.0


def test_evaluate_skips_queries_without_relevant_papers():
    run = {"q1": ["a", "b"], "q2": ["c"], "q3": ["d"]}
    qrels = {"q1": {"a": 1}, "q2": {"c": 0}}
    out = evaluate(run, qrels)
    assert out["queries"] == 1 and out["skipped"] == 2 and out["ndcg@10"] == 1.0
