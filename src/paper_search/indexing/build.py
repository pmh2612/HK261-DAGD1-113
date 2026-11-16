"""Build the FAISS and BM25 indexes over chunks (FR-15), with their row maps (P3)."""

import json
import pickle
import re
from collections.abc import Sequence
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from paper_search.config import settings
from paper_search.models import Chunk


def bm25_tokenize(text: str) -> list[str]:
    """The one tokenizer for BM25, at index and at query time."""
    return re.findall(r"\w+", text.lower())


def index_text(chunk: Chunk, title: str) -> str:
    """What gets indexed: the title carries topic words an abstract window may lack.

    Only the index sees the title; `chunk.text` stays an exact slice of the source (P2).
    """
    return f"{title}. {chunk.text}"


def build_vector_index(chunks: Sequence[Chunk], titles: dict[str, str], out_dir: Path) -> None:
    """Exact inner-product index on normalized vectors (cosine), plus row -> chunk_id."""
    model = SentenceTransformer(settings.embedding_model)
    vecs = model.encode(
        [index_text(c, titles[c.paper_id]) for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    rows = [c.chunk_id for c in chunks]
    assert index.ntotal == len(rows), "P3 violated: FAISS rows and row map differ"
    faiss.write_index(index, str(out_dir / "faiss.index"))
    (out_dir / "faiss_rows.json").write_text(json.dumps(rows))


def build_bm25_index(chunks: Sequence[Chunk], titles: dict[str, str], out_dir: Path) -> None:
    """BM25Okapi over the same index text, pickled with its own row map."""
    bm25 = BM25Okapi([bm25_tokenize(index_text(c, titles[c.paper_id])) for c in chunks])
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "rows": [c.chunk_id for c in chunks]}, f)
