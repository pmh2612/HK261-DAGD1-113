"""Stage 1: collect the prototype corpus from Semantic Scholar (Roadmap 2.2).

Writes
    data/raw/<version>/papers.jsonl          seed records exactly as returned
    data/raw/<version>/external.jsonl        snowballed reference records, as returned
    data/raw/<version>/references.parquet    (paper_id, ref_id) for every seed
    data/processed/<version>/papers.parquet  normalized, English, deduplicated; seeds + :External

Run: .venv/bin/python scripts/collect.py [--version 2026-10-23] [--limit 1000]
Without an API key the reference lookup is throttled hard and can take a while.
"""

import argparse
import datetime
import json

import pandas as pd

from paper_search.collection import semantic_scholar as s2
from paper_search.config import settings
from paper_search.models import write_records

TOPIC = "retrieval augmented generation"  # requirements.md §2


def write_jsonl(path, records: list[dict]) -> None:
    """Raw API records, one per line, unmodified."""
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def main() -> None:
    """Search, snowball one hop of references, normalize, write."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=datetime.date.today().isoformat())
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    raw_dir = settings.raw_dir / args.version
    out_dir = settings.processed_dir / args.version
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The search result is written at once and reused on a re-run, so a failure in the
    # slow, throttled reference lookup below does not cost the search too.
    seed_path = raw_dir / "papers.jsonl"
    if seed_path.exists():
        seed_raw = [json.loads(line) for line in seed_path.open(encoding="utf-8")]
        print(f"search: reusing {len(seed_raw)} records from {seed_path}")
    else:
        seed_raw = s2.search_raw(TOPIC, args.limit)
        write_jsonl(seed_path, seed_raw)
        print(f"search: {len(seed_raw)} records")
    seeds = s2.normalize(seed_raw, args.version)
    print(f"normalized: {len(seeds)} English, deduplicated seeds")

    refs = s2.fetch_references([p.paper_id for p in seeds])
    print(
        f"references: {sum(map(len, refs.values()))} edges from {len(refs)} of {len(seeds)} seeds"
    )
    external_ids = s2.select_external(refs)
    external_raw = s2.fetch_raw(external_ids)
    known = {p.paper_id for p in seeds}
    external = [
        p
        for p in s2.normalize(external_raw, args.version, is_external=True)
        if p.paper_id not in known
    ]
    print(f"snowball: {len(external_ids)} references cited by >=3 seeds, {len(external)} kept")

    write_jsonl(raw_dir / "external.jsonl", external_raw)
    pd.DataFrame(
        [(src, dst) for src, dsts in refs.items() for dst in dsts], columns=["paper_id", "ref_id"]
    ).to_parquet(raw_dir / "references.parquet")
    write_records(seeds + external, out_dir / "papers.parquet")

    print(f"\nwrote {len(seeds) + len(external)} papers to {out_dir / 'papers.parquet'}")
    print(f"  seeds with abstract:  {sum(p.abstract is not None for p in seeds)}")
    print(f"  seeds with full text: {sum(p.has_full_text for p in seeds)}")


if __name__ == "__main__":
    main()
