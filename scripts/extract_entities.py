"""Stage 2: extract Method / Dataset / Topic entities for every seed paper (Roadmap 3.1).

Reads data/processed/<version>/papers.parquet, writes entities.parquet next to it.
`:External` papers are skipped: they are graph-only and mostly lack an abstract.

Run: .venv/bin/python scripts/extract_entities.py [--version 2026-10-23]
"""

import argparse

from tqdm import tqdm

from paper_search.config import settings
from paper_search.extraction.entities import extract_entities
from paper_search.models import Paper, read_records, write_records


def main() -> None:
    """Extract, write, and print how many of each entity type came out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=None)
    version = parser.parse_args().version or settings.latest_version()
    out_dir = settings.processed_dir / version

    papers = [p for p in read_records(out_dir / "papers.parquet", Paper) if not p.is_external]
    entities = [extract_entities(p, []) for p in tqdm(papers, desc="extracting")]
    write_records(entities, out_dir / "entities.parquet")

    print(f"wrote entities for {len(entities)} papers to {out_dir / 'entities.parquet'}")
    for field in ("methods", "datasets", "topics"):
        mentions = sum(len(getattr(e, field)) for e in entities)
        empty = sum(not getattr(e, field) for e in entities)
        print(f"  {field:<9} {mentions:>5} mentions, {empty:>4} papers with none")


if __name__ == "__main__":
    main()
