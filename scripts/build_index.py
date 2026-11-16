"""Stage 4: chunk the seed papers and build the BM25 and FAISS indexes (Roadmap 3.3).

Writes chunks.parquet, faiss.index, faiss_rows.json, bm25.pkl and manifest.json to
data/processed/<version>/. Aborts if any chunk breaks a provenance invariant (P1-P4).
`:External` papers are never indexed (P5). Sections stay empty until the PDF stage
lands, so every chunk is an abstract window for now.

Run: .venv/bin/python scripts/build_index.py [--version 2026-10-23]
"""

import argparse
import datetime
import json

from paper_search.config import settings
from paper_search.indexing.build import build_bm25_index, build_vector_index
from paper_search.indexing.chunking import (
    OVERLAP,
    WINDOW,
    check_provenance,
    chunk_paper,
    source_texts,
)
from paper_search.models import Paper, read_records, write_records


def main() -> None:
    """Chunk, verify provenance, index, record what built it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=None)
    version = parser.parse_args().version or settings.latest_version()
    out_dir = settings.processed_dir / version

    papers = [p for p in read_records(out_dir / "papers.parquet", Paper) if not p.is_external]
    indexable = [p for p in papers if p.abstract]
    print(
        f"{len(indexable)} of {len(papers)} seed papers have text to index "
        f"({len(papers) - len(indexable)} without abstract, skipped until PDFs land)"
    )

    chunks = [c for p in indexable for c in chunk_paper(p, [])]
    check_provenance(chunks, {p.paper_id: source_texts(p, []) for p in indexable})
    titles = {p.paper_id: p.title for p in indexable}

    write_records(chunks, out_dir / "chunks.parquet")
    build_bm25_index(chunks, titles, out_dir)
    build_vector_index(chunks, titles, out_dir)
    manifest = {
        "data_version": version,
        "embedding_model": settings.embedding_model,
        "window_tokens": WINDOW,
        "overlap_tokens": OVERLAP,
        "papers": len(indexable),
        "chunks": len(chunks),
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
