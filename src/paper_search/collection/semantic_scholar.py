"""Semantic Scholar client: search, references, normalization and deduplication.

Covers FR-1, FR-3, FR-5 and FR-6. arXiv metadata (FR-2) is not fetched separately:
Semantic Scholar already carries the arXiv ID, and the arXiv API is only needed for PDFs,
which belong to the PDF stage.
"""

import re
import time
from collections.abc import Iterable, Sequence

import requests

from paper_search.config import settings
from paper_search.models import Paper

API = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,venue,authors,citationCount,referenceCount,externalIds,openAccessPdf"
BATCH_SIZE = 500  # the /paper/batch maximum
#: The API key allows 1 request per second across all endpoints; stay just under it.
MIN_INTERVAL = 1.1

_last_request = 0.0


def _request(method: str, url: str, **kwargs) -> dict | list:
    """One API call, spaced to the rate limit, with exponential backoff on 429 and 5xx (NFR-11)."""
    global _last_request
    headers = (
        {"x-api-key": settings.semantic_scholar_api_key}
        if settings.semantic_scholar_api_key
        else {}
    )
    for attempt in range(12):
        time.sleep(max(0.0, _last_request + MIN_INTERVAL - time.monotonic()))
        _last_request = time.monotonic()
        resp = requests.request(method, url, headers=headers, timeout=120, **kwargs)
        if resp.status_code != 429 and resp.status_code < 500:
            resp.raise_for_status()
            return resp.json()
        time.sleep(min(2**attempt, 60))
    resp.raise_for_status()
    return {}  # unreachable: raise_for_status raised on the last 429/5xx


def search_raw(query: str, limit: int = 1000) -> list[dict]:
    """Bulk search, most-cited first, records exactly as the API returns them."""
    params = {"query": query, "fields": FIELDS, "sort": "citationCount:desc"}
    records: list[dict] = []
    while len(records) < limit:
        payload = _request("GET", f"{API}/paper/search/bulk", params=params)
        records += payload.get("data", [])
        if not payload.get("token"):
            break
        params["token"] = payload["token"]
    return records[:limit]


def fetch_raw(
    paper_ids: Sequence[str], fields: str = FIELDS, batch_size: int = BATCH_SIZE
) -> list[dict]:
    """Batch lookup by ID; IDs Semantic Scholar does not know are left out.

    A batch that still fails after backoff is logged and skipped rather than ending the
    run: partial references still make a graph (architecture §8).
    """
    out: list[dict] = []
    for start in range(0, len(paper_ids), batch_size):
        batch = list(paper_ids[start : start + batch_size])
        try:
            payload = _request(
                "POST", f"{API}/paper/batch", params={"fields": fields}, json={"ids": batch}
            )
        except requests.HTTPError as exc:
            print(f"  batch {start}-{start + len(batch)} skipped: {exc}", flush=True)
            continue
        out += [p for p in payload if p]
        print(f"  batch {start}-{start + len(batch)}: {len(out)} resolved so far", flush=True)
    return out


def fetch_references(paper_ids: Sequence[str]) -> dict[str, list[str]]:
    """paper_id -> IDs of the papers it references (only references S2 can resolve)."""
    return {
        p["paperId"]: [r["paperId"] for r in p.get("references") or [] if r.get("paperId")]
        # Smaller batches: a reference list per paper makes each response heavy, and the
        # unauthenticated quota rejects 500-paper reference batches outright.
        for p in fetch_raw(paper_ids, fields="references.paperId", batch_size=100)
    }


def select_external(refs: dict[str, list[str]], min_citers: int = 3) -> list[str]:
    """References outside the seed set cited by at least `min_citers` seeds (DECIDE-1)."""
    counts: dict[str, int] = {}
    for targets in refs.values():
        for t in set(targets):
            counts[t] = counts.get(t, 0) + 1
    return sorted(t for t, c in counts.items() if c >= min_citers and t not in refs)


def is_english(text: str) -> bool:
    """Crude language check: almost all letters are ASCII (scope limit: English only)."""
    # ponytail: ASCII ratio, misses English-script languages; use a langid model if
    # non-English papers ever show up in the corpus.
    letters = [c for c in text if c.isalpha()]
    return not letters or sum(c.isascii() for c in letters) / len(letters) > 0.9


def to_paper(raw: dict, data_version: str, is_external: bool = False) -> Paper:
    """Normalize one API record into the common schema (FR-5)."""
    ids = raw.get("externalIds") or {}
    pdf_url = (raw.get("openAccessPdf") or {}).get("url") or None
    authors = raw.get("authors") or []
    return Paper(
        paper_id=raw["paperId"],
        title=" ".join((raw.get("title") or "").split()),
        abstract=" ".join(raw["abstract"].split()) if raw.get("abstract") else None,
        year=raw.get("year"),
        venue=raw.get("venue") or None,
        authors=[a.get("name") or "" for a in authors],
        author_ids=[a.get("authorId") for a in authors],
        doi=(ids.get("DOI") or "").lower() or None,
        arxiv_id=ids.get("ArXiv") or None,
        citation_count=raw.get("citationCount") or 0,
        reference_count=raw.get("referenceCount") or 0,
        pdf_url=pdf_url,
        has_full_text=bool(ids.get("ArXiv") or pdf_url),
        is_external=is_external,
        source="semantic_scholar",
        data_version=data_version,
    )


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def deduplicate(papers: Iterable[Paper]) -> list[Paper]:
    """Keep the first paper per DOI, then arXiv ID, then S2 ID, then title (FR-6).

    Input order is citation count descending, so the copy kept is the most-cited one -
    usually the published version rather than its preprint.
    """
    seen: set[tuple[str, str]] = set()
    out = []
    for p in papers:
        keys = {
            ("doi", p.doi),
            ("arxiv", p.arxiv_id),
            ("s2", p.paper_id),
            ("title", _title_key(p.title)),
        }
        keys = {k for k in keys if k[1]}
        if keys & seen:
            continue
        seen |= keys
        out.append(p)
    return out


def normalize(records: Iterable[dict], data_version: str, is_external: bool = False) -> list[Paper]:
    """Raw API records -> English, deduplicated `Paper`s, input order kept."""
    papers = (to_paper(r, data_version, is_external) for r in records)
    return deduplicate(p for p in papers if is_english(p.title + " " + (p.abstract or "")))
