"""BT1: collect one sample level from arXiv and Semantic Scholar, and compare the sources.

Builds the sampling frame (configs/bt1_collect.yaml), takes the configured level, fetches
each paper from arXiv (OAI-PMH) and from Semantic Scholar (batch lookup by arXiv ID), and
writes the merged papers to data/processed/papers_<level>.jsonl. Raw responses are cached
under data/raw/, so a re-run makes no network calls.

Metrics for each source go to experiments/runs/bt1/<timestamp>_<source>/.

Run: .venv/bin/python scripts/bt1_collect.py [--config configs/bt1_collect.yaml]
"""

import argparse
import difflib
import hashlib
import json
import random
import re
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from paper_search.collection import arxiv_oai, semantic_scholar
from paper_search.config import PROJECT_ROOT, settings
from paper_search.models import Paper

FIELDS = (
    "abstract",
    "year",
    "venue",
    "authors",
    "author_ids",
    "doi",
    "citation_count",
    "fields_of_study",
    "categories",
)


def has(p: Paper, field: str) -> bool:
    """Whether the paper carries a usable value for `field`."""
    value = getattr(p, field)
    if field == "author_ids":
        return any(value)
    return value is not None and value != [] and value != ""


def coverage(papers: list[Paper], n: int) -> dict:
    """Share of the n requested papers that have each field; a missing paper has none."""
    return {
        "papers": len(papers),
        "coverage": {f: round(sum(has(p, f) for p in papers) / n, 3) for f in FIELDS},
        "abstract_words_mean": round(
            statistics.mean(len(p.abstract.split()) for p in papers if p.abstract), 1
        )
        if any(p.abstract for p in papers)
        else None,
    }


def title_key(title: str) -> str:
    """Title reduced to lowercase letters and digits, for matching across sources."""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def main() -> None:
    """Frame, sample, fetch, merge, measure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/bt1_collect.yaml")
    cfg = yaml.safe_load(parser.parse_args().config.read_text())
    level, n = cfg["level"], cfg["levels"][cfg["level"]]
    version = f"arxiv-{level}"
    raw = settings.raw_dir / "arxiv"
    log: list[str] = []

    def say(line: str) -> None:
        print(line, flush=True)
        log.append(line)

    frame = sorted(
        {
            i
            for c in cfg["categories"]
            for m in cfg["months"]
            for i in arxiv_oai.list_month(c, m, raw / "listing")
        }
    )
    assert len(frame) >= max(cfg["levels"].values()), f"frame has only {len(frame)} papers"
    order = frame.copy()
    random.Random(cfg["seed"]).shuffle(order)
    ids = order[:n]
    frame_sha = hashlib.sha256("\n".join(order).encode()).hexdigest()[:12]
    say(f"frame {len(frame)} papers (order sha {frame_sha}); level {level}: {n}")

    # (b) arXiv only
    start, from_cache = time.monotonic(), 0
    by_arxiv: dict[str, Paper] = {}
    for i in tqdm(ids, desc="arXiv OAI"):
        xml, cached = arxiv_oai.fetch_record(i, raw / "oai")
        from_cache += cached
        if p := arxiv_oai.to_paper(xml, version):
            by_arxiv[i] = p
    t_arxiv = time.monotonic() - start
    say(f"arXiv: {len(by_arxiv)}/{n} records in {t_arxiv:.0f}s ({from_cache} cached)")

    # (a) Semantic Scholar only
    start = time.monotonic()
    s2_cache = settings.raw_dir / "s2" / f"arxiv_{level}.jsonl"
    s2_cached = s2_cache.exists()
    if s2_cached:
        records = [json.loads(line) for line in s2_cache.open()]
    else:
        records = semantic_scholar.fetch_raw([f"ARXIV:{i}" for i in ids])
        s2_cache.parent.mkdir(parents=True, exist_ok=True)
        s2_cache.write_text("".join(json.dumps(r) + "\n" for r in records))
    t_s2 = time.monotonic() - start
    by_s2 = {
        p.arxiv_id: p
        for p in (semantic_scholar.to_paper(r, version) for r in records)
        if p.arxiv_id in set(ids)
    }
    say(f"Semantic Scholar: {len(by_s2)}/{n} records in {t_s2:.0f}s (cached: {s2_cached})")

    # (c) merged
    merged = [arxiv_oai.merge(by_arxiv[i], by_s2.get(i)) for i in ids if i in by_arxiv]
    no_abstract = [p.paper_id for p in merged if not p.abstract]
    # Principle 2's PDF fallback is for papers the API gives no abstract; arXiv always
    # has one, so it is only built if this ever logs something.
    if no_abstract:
        say(f"dropped {len(no_abstract)} papers without abstract: {no_abstract}")
    merged = [p for p in merged if p.abstract]
    out = settings.processed_dir / f"papers_{level}.jsonl"
    out.write_text("".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in merged))
    say(f"wrote {len(merged)} papers to {out}")

    both = [i for i in ids if i in by_arxiv and i in by_s2]
    with_both_abstracts = [i for i in both if by_s2[i].abstract]
    ratios = [
        difflib.SequenceMatcher(None, by_arxiv[i].abstract, by_s2[i].abstract).ratio()
        for i in with_both_abstracts
    ]
    overlap = {
        "matched": len(both),
        "match_rate": round(len(both) / n, 3),
        "title_agrees": round(
            sum(title_key(by_arxiv[i].title) == title_key(by_s2[i].title) for i in both)
            / max(len(both), 1),
            3,
        ),
        "doi_conflicts": sum(
            bool(by_arxiv[i].doi and by_s2[i].doi and by_arxiv[i].doi != by_s2[i].doi) for i in both
        ),
        "s2_abstract_missing_when_matched": len(both) - len(with_both_abstracts),
        "abstract_similarity_median": round(statistics.median(ratios), 3) if ratios else None,
        "abstract_identical_share": round(sum(r >= 0.95 for r in ratios) / len(ratios), 3)
        if ratios
        else None,
        "duplicate_titles_in_sample": len(merged) - len({title_key(p.title) for p in merged}),
    }

    runs = {
        "arxiv_only": {**coverage(list(by_arxiv.values()), n), "seconds": round(t_arxiv, 1)},
        "s2_only": {**coverage(list(by_s2.values()), n), "seconds": round(t_s2, 1)},
        "combined": {
            **coverage(merged, n),
            "seconds": round(t_arxiv + t_s2, 1),
            "overlap": overlap,
        },
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for method, metrics in runs.items():
        d = PROJECT_ROOT / "experiments/runs/bt1" / f"{stamp}_{method}"
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(yaml.safe_dump({**cfg, "method": method}, sort_keys=False))
        metrics = {
            **metrics,
            "n_requested": n,
            "frame_size": len(frame),
            "frame_sha": frame_sha,
            "network_cached": {"arxiv": from_cache == n, "s2": s2_cached},
        }
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (d / "log.txt").write_text("\n".join(log) + "\n")
        say(f"\n{method}: {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
