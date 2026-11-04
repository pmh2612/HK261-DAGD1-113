"""BT8: combine BM25 and dense retrieval, and rerank with a cross-encoder.

Every function maps score matrices (one row per query, one column per document, from
`rankers.bm25_scores` / `rankers.dense_scores`) to a new score matrix, so fusions stack:
rerank the output of RRF, or of a linear mix, the same way as a single retriever.
"""

import numpy as np


def rrf(*scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: sum of 1 / (k + rank) over the retrievers.

    Uses ranks only, so the retrievers' score scales never have to agree. k = 60 is
    the value of the original paper (Cormack et al., 2009) and the usual default.
    """
    fused = np.zeros_like(scores[0], dtype=float)
    for s in scores:
        ranks = np.argsort(np.argsort(-s, axis=1, kind="stable"), axis=1) + 1
        fused += 1.0 / (k + ranks)
    return fused


def minmax(scores: np.ndarray) -> np.ndarray:
    """Each query's scores rescaled to [0, 1]; a query with all-equal scores becomes 0."""
    low = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - low
    return np.divide(scores - low, span, out=np.zeros_like(scores, dtype=float), where=span > 0)


def linear(sparse: np.ndarray, dense: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted mix: alpha * dense + (1 - alpha) * sparse, each min-max normalized per query."""
    return alpha * minmax(dense) + (1 - alpha) * minmax(sparse)


def rerank(
    scores: np.ndarray, queries: list[str], docs: list[dict], model_id: str, depth: int = 50
) -> np.ndarray:
    """Re-order each query's top `depth` documents by a cross-encoder.

    Documents below the top `depth` keep their order underneath. The cross-encoder
    reads query and document together, which a bi-encoder cannot, at the cost of one
    model call per (query, document) pair.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_id)
    out = np.argsort(np.argsort(-scores, axis=1, kind="stable"), axis=1) * -1.0
    for i, q in enumerate(queries):
        top = np.argsort(-scores[i], kind="stable")[:depth]
        ce = model.predict([(q, f"{docs[j]['title']}. {docs[j]['abstract']}") for j in top])
        # Shift the cross-encoder scores above every original rank (-rank <= 0).
        out[i, top] = ce - ce.min() + 1
    return out
