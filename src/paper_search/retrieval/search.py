"""Top-k BM25 and vector search over the built indexes (FR-15, Phase 1 baseline).

The two are kept apart on purpose: Roadmap 4.1 measures each alone (architecture §9).
"""

import json
import pickle
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from paper_search.config import settings
from paper_search.indexing.build import bm25_tokenize
from paper_search.models import Chunk, SearchHit, read_records


@lru_cache(maxsize=2)
def _index(version: str) -> dict:
    """Everything one data version's search needs, loaded once."""
    d = settings.processed_dir / version
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest["embedding_model"] != settings.embedding_model:
        # Searching with a different model returns plausible nonsense, not an error.
        raise RuntimeError(
            f"index built with {manifest['embedding_model']}, "
            f"settings.embedding_model is {settings.embedding_model}"
        )
    with (d / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)  # our own file, written by build_index.py
    return {
        "chunks": {c.chunk_id: c for c in read_records(d / "chunks.parquet", Chunk)},
        "faiss": faiss.read_index(str(d / "faiss.index")),
        "faiss_rows": json.loads((d / "faiss_rows.json").read_text()),
        "bm25": bm25["bm25"],
        "bm25_rows": bm25["rows"],
    }


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def search_bm25(query: str, k: int = 10, version: str | None = None) -> list[SearchHit]:
    """Top-k chunks by BM25 score."""
    idx = _index(version or settings.latest_version())
    scores = idx["bm25"].get_scores(bm25_tokenize(query))
    top = np.argsort(-scores)[:k]
    return [SearchHit(idx["chunks"][idx["bm25_rows"][i]], float(scores[i]), "bm25") for i in top]


def search_vector(query: str, k: int = 10, version: str | None = None) -> list[SearchHit]:
    """Top-k chunks by cosine similarity. Applies the query prefix itself (DECIDE-5)."""
    idx = _index(version or settings.latest_version())
    q = _model().encode([settings.embedding_query_prefix + query], normalize_embeddings=True)
    scores, rows = idx["faiss"].search(q, k)
    return [
        SearchHit(idx["chunks"][idx["faiss_rows"][r]], float(s), "vector")
        for s, r in zip(scores[0], rows[0], strict=True)
        if r != -1
    ]
