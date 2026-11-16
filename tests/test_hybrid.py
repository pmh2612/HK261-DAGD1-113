"""Hand-computed cases for the BT8 fusions."""

import numpy as np
from pytest import approx

from paper_search.retrieval.hybrid import linear, minmax, rrf


def test_rrf_uses_ranks_not_scales():
    a = np.array([[10.0, 5.0, 1.0]])  # ranks 1, 2, 3
    b = np.array([[0.1, 0.3, 0.2]])  # ranks 3, 1, 2
    fused = rrf(a, b, k=60)
    assert fused[0] == approx([1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62])
    # Multiplying one retriever's scores by 1000 changes nothing.
    assert rrf(a * 1000, b, k=60) == approx(fused)


def test_minmax_and_linear():
    s = np.array([[2.0, 4.0, 6.0], [3.0, 3.0, 3.0]])
    assert minmax(s)[0] == approx([0.0, 0.5, 1.0])
    assert minmax(s)[1] == approx([0.0, 0.0, 0.0])  # no division by zero
    sparse, dense = np.array([[0.0, 10.0]]), np.array([[1.0, 0.0]])
    assert linear(sparse, dense, 0.0)[0] == approx([0.0, 1.0])  # BM25 only
    assert linear(sparse, dense, 1.0)[0] == approx([1.0, 0.0])  # dense only
    assert linear(sparse, dense, 0.25)[0] == approx([0.25, 0.75])


def test_rerank_reorders_only_the_top(monkeypatch):
    import sentence_transformers

    from paper_search.retrieval.hybrid import rerank

    class FakeCrossEncoder:  # scores a pair by the document title length
        def __init__(self, _):
            pass

        def predict(self, pairs):
            return np.array([float(len(doc.split(".")[0])) for _, doc in pairs])

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    docs = [{"title": "t" * n, "abstract": ""} for n in (1, 5, 3, 9)]
    scores = np.array([[4.0, 3.0, 2.0, 1.0]])  # first-stage order: 0, 1, 2, 3
    out = rerank(scores, ["q"], docs, "fake", depth=3)
    # top 3 (docs 0, 1, 2) re-ordered by title length -> 1, 2, 0; doc 3 stays last
    assert list(np.argsort(-out[0])) == [1, 2, 0, 3]
