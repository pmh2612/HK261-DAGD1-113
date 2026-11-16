"""Collect ~200 abstracts for the evaluation set and write the annotation skeletons.

Writes, under data/eval/ (tracked in git - labels are human work and must be versioned):

- abstracts.jsonl   the corpus both experiments run on
- entities.jsonl    a fixed sample to annotate with Method / Dataset / Task entities;
                    `entities: null` means "not annotated yet" and is skipped when scoring
- queries.jsonl     seeded with the use-case queries from docs/requirements.md §5;
                    `relevant: null` means "not judged yet"

Existing entities.jsonl / queries.jsonl are never overwritten - they hold annotations.
Re-running only refreshes abstracts.jsonl, which would invalidate the labels, so it
refuses to do that too unless --force is given.

Run: .venv/bin/python notebooks/collect_abstracts.py [--force]
"""

import json
import random
import sys
import time

import requests

from paper_search.config import PROJECT_ROOT, settings

TOPIC = "retrieval augmented generation"
S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
TARGET = 200
MIN_ABSTRACT_CHARS = 300  # shorter ones are usually truncated or boilerplate
ENTITY_SAMPLE = 50  # Roadmap 3.1: ~50 papers checked by hand
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

# Written by the team in docs/requirements.md §5 (UC-1, UC-2, UC-4, UC-6, UC-7).
SEED_QUERIES = [
    "retrieval augmented generation for long documents",
    "papers that evaluate on Natural Questions using a dense retriever",
    "hybrid retrieval",
    "self-RAG",
    "dense retrieval and BM25 compared for open-domain QA",
]


def fetch() -> list[dict]:
    """Most-cited papers on the topic that carry a usable abstract."""
    params = {
        "query": TOPIC,
        "fields": "title,abstract,year,externalIds",
        "sort": "citationCount:desc",
    }
    headers = (
        {"x-api-key": settings.semantic_scholar_api_key}
        if settings.semantic_scholar_api_key
        else {}
    )
    papers: list[dict] = []
    while len(papers) < TARGET:
        resp = requests.get(S2_BULK, params=params, headers=headers, timeout=60)
        if resp.status_code == 429:
            print("  429 rate-limited, backing off 10s")
            time.sleep(10)
            continue
        resp.raise_for_status()
        payload = resp.json()
        papers += [
            p for p in payload.get("data", []) if len(p.get("abstract") or "") >= MIN_ABSTRACT_CHARS
        ]
        if not payload.get("token"):
            break
        params["token"] = payload["token"]
        time.sleep(1)
    return papers[:TARGET]


def write_jsonl(path, rows) -> None:
    """One JSON object per line, UTF-8, keys in insertion order."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Fetch the corpus, then write whichever skeletons do not exist yet."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = EVAL_DIR / "abstracts.jsonl"
    if corpus_path.exists() and "--force" not in sys.argv:
        sys.exit(f"{corpus_path} exists; labels refer to it. Pass --force to replace it.")

    papers = fetch()
    rows = [
        {
            "paper_id": p["paperId"],
            "title": p["title"],
            "year": p.get("year"),
            "arxiv_id": (p.get("externalIds") or {}).get("ArXiv"),
            "abstract": p["abstract"],
        }
        for p in papers
    ]
    write_jsonl(corpus_path, rows)
    print(f"wrote {len(rows)} abstracts to {corpus_path}")

    entities_path = EVAL_DIR / "entities.jsonl"
    if not entities_path.exists():
        sample = random.Random(0).sample(rows, min(ENTITY_SAMPLE, len(rows)))
        write_jsonl(
            entities_path,
            [
                {"paper_id": r["paper_id"], "abstract": r["abstract"], "entities": None}
                for r in sample
            ],
        )
        print(f"wrote {len(sample)} papers to annotate to {entities_path}")

    queries_path = EVAL_DIR / "queries.jsonl"
    if not queries_path.exists():
        write_jsonl(
            queries_path,
            [
                {"qid": f"q{i:02d}", "query": q, "relevant": None}
                for i, q in enumerate(SEED_QUERIES, 1)
            ],
        )
        print(f"wrote {len(SEED_QUERIES)} seed queries to {queries_path}")


if __name__ == "__main__":
    main()
