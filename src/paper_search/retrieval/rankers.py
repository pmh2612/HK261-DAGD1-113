"""Rankers compared in BT6 (BM25 variants) and BT7 (dense embeddings).

Each builder indexes the documents once and returns `rank(queries, k)`: for every query,
document positions best first. The experiment script treats every system alike.
A document is a dict with `title` and `abstract`.
"""

import re
from collections.abc import Callable

import faiss
import numpy as np
import Stemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

Rank = Callable[[list[str], int], list[list[int]]]

#: What BM25 indexes. Repeating the title is the simplest way to weight it x2.
FIELDS: dict[str, Callable[[dict], str]] = {
    "title": lambda d: d["title"],
    "abstract": lambda d: d["abstract"],
    "title_abstract": lambda d: f"{d['title']} {d['abstract']}",
    "title2_abstract": lambda d: f"{d['title']} {d['title']} {d['abstract']}",
}


def analyzer(preprocess: str) -> Callable[[str], list[str]]:
    """Lowercased word tokens, plus "stop" (drop English stopwords) and/or "stem" (Snowball)."""
    stemmer = Stemmer.Stemmer("english") if "stem" in preprocess else None
    stop = "stop" in preprocess

    def analyze(text: str) -> list[str]:
        tokens = re.findall(r"\w+", text.lower())
        if stop:
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        return stemmer.stemWords(tokens) if stemmer else tokens

    return analyze


def bm25(
    docs: list[dict],
    field: str = "title_abstract",
    preprocess: str = "lower",
    k1: float = 1.5,
    b: float = 0.75,
) -> Rank:
    """BM25Okapi over one field, with the same analyzer at index and query time."""
    analyze = analyzer(preprocess)
    index = BM25Okapi([analyze(FIELDS[field](d)) for d in docs], k1=k1, b=b)

    def rank(queries: list[str], k: int) -> list[list[int]]:
        # Stable sort: ties (often all-zero scores) keep document order, run after run.
        return [
            np.argsort(-index.get_scores(analyze(q)), kind="stable")[:k].tolist() for q in queries
        ]

    return rank


def _flat_index(vecs: np.ndarray) -> faiss.Index:
    """Exact inner product on normalized vectors, i.e. cosine (the BT7 reference index)."""
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype(np.float32))
    return index


def dense(docs: list[dict], model_id: str, query_prefix: str = "", doc_prefix: str = "") -> Rank:
    """Sentence-transformers embeddings of "title. abstract", searched with FAISS.

    Prefixes are part of the model, not a tuning knob: e5 expects "query: " / "passage: ",
    bge a search instruction on queries only. Leaving them off degrades results silently.
    """
    model = SentenceTransformer(model_id)
    index = _flat_index(
        model.encode(
            [f"{doc_prefix}{d['title']}. {d['abstract']}" for d in docs],
            normalize_embeddings=True,
            batch_size=32,
        )
    )

    def rank(queries: list[str], k: int) -> list[list[int]]:
        q = model.encode([query_prefix + q for q in queries], normalize_embeddings=True)
        return index.search(q.astype(np.float32), k)[1].tolist()

    return rank


def precomputed(doc_vecs: np.ndarray, query_vecs: dict[str, np.ndarray]) -> Rank:
    """Search vectors encoded elsewhere (SPECTER2 runs in its own venv)."""
    index = _flat_index(doc_vecs)

    def rank(queries: list[str], k: int) -> list[list[int]]:
        q = np.stack([query_vecs[q] for q in queries])
        return index.search(q.astype(np.float32), k)[1].tolist()

    return rank
