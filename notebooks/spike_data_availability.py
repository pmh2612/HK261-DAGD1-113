"""Spike: can we actually collect 500-1,000 usable papers on one topic?

Produces every measured number quoted in `docs/requirements.md` §2 and
`docs/architecture.md` §1, so each one can be reproduced rather than taken on
trust:

- how many papers exist for the topic
- what share carry an abstract, a DOI, an arXiv ID, an open-access PDF
- **how many have full text reachable at all** (the union - constraint C1)
- how the corpus is distributed over publication years
- **how many references land back inside the seed set** (constraint C2)
- whether the downloadable PDFs have a real text layer PyMuPDF can read

Not production code - it lives in notebooks/ and nothing imports it.
Run: .venv/bin/python notebooks/spike_data_availability.py

The citation-overlap measurement uses `/paper/batch`, which is heavily
throttled without a Semantic Scholar API key; it backs off and retries, so
that part can take several minutes. Set SEMANTIC_SCHOLAR_API_KEY in `.env` to
make it quick.
"""

import collections
import io
import statistics
import time

import arxiv
import fitz  # PyMuPDF
import requests

from paper_search.config import settings

TOPIC = "retrieval augmented generation"
S2 = "https://api.semanticscholar.org/graph/v1"
S2_BULK = f"{S2}/paper/search/bulk"
S2_BATCH = f"{S2}/paper/batch"
S2_FIELDS = (
    "title,abstract,year,venue,authors,citationCount,referenceCount,externalIds,openAccessPdf"
)
TARGET = 1000
PDF_SAMPLE = 8
CITATION_SAMPLE = 100  # seed papers whose reference lists we pull
SNOWBALL_MIN_CITERS = 3  # DECIDE-1 threshold


def _headers() -> dict[str, str]:
    """Send the API key when one is configured - it lifts the rate limit."""
    key = settings.semantic_scholar_api_key
    return {"x-api-key": key} if key else {}


def fetch_semantic_scholar(target: int = TARGET) -> tuple[int, list[dict]]:
    """Page through the bulk search endpoint. Returns (total available, records)."""
    papers: list[dict] = []
    total = 0
    token = None
    while len(papers) < target:
        params = {"query": TOPIC, "fields": S2_FIELDS, "limit": 1000}
        if token:
            params["token"] = token
        resp = requests.get(S2_BULK, params=params, headers=_headers(), timeout=60)
        if resp.status_code == 429:
            print("  429 rate-limited (no API key), backing off 10s")
            time.sleep(10)
            continue
        resp.raise_for_status()
        payload = resp.json()
        total = payload.get("total", 0)
        papers.extend(payload.get("data", []))
        token = payload.get("token")
        print(f"  page: +{len(payload.get('data', []))} (have {len(papers)} of {total})")
        if not token:
            break
        time.sleep(1)
    return total, papers[:target]


def summarize(papers: list[dict]) -> dict[str, int]:
    """Count how many records carry each field we depend on."""
    return {
        "records": len(papers),
        "with abstract": sum(1 for p in papers if p.get("abstract")),
        "with year": sum(1 for p in papers if p.get("year")),
        "with venue": sum(1 for p in papers if p.get("venue")),
        "with authors": sum(1 for p in papers if p.get("authors")),
        "with citationCount": sum(1 for p in papers if p.get("citationCount") is not None),
        "with referenceCount": sum(1 for p in papers if p.get("referenceCount") is not None),
        "with DOI": sum(1 for p in papers if (p.get("externalIds") or {}).get("DOI")),
        "with arXiv ID": sum(1 for p in papers if (p.get("externalIds") or {}).get("ArXiv")),
        "with open-access PDF": sum(1 for p in papers if (p.get("openAccessPdf") or {}).get("url")),
        # The union is the number that matters: either route gives us full text.
        # Neither column above shows it, because the two overlap.
        "FULL TEXT REACHABLE": sum(1 for p in papers if _has_arxiv(p) or _has_oa_pdf(p)),
        "metadata only": sum(1 for p in papers if not _has_arxiv(p) and not _has_oa_pdf(p)),
    }


def _has_arxiv(paper: dict) -> bool:
    """True when the paper has an arXiv ID, which means a reliable PDF."""
    return bool((paper.get("externalIds") or {}).get("ArXiv"))


def _has_oa_pdf(paper: dict) -> bool:
    """True when Semantic Scholar offers an open-access PDF link."""
    return bool((paper.get("openAccessPdf") or {}).get("url"))


def year_distribution(papers: list[dict]) -> None:
    """Print how recent the corpus is - this is what makes the graph sparse."""
    years = collections.Counter(p["year"] for p in papers if p.get("year"))
    recent = sum(c for y, c in years.items() if y >= 2024)
    print("\nPublication years:")
    for year, count in sorted(years.items())[-6:]:
        print(f"  {year}  {count:>4}")
    print(f"  => {recent} of {len(papers)} ({100 * recent / len(papers):.0f}%) from 2024 onwards")


def citation_overlap(papers: list[dict], n: int = CITATION_SAMPLE) -> None:
    """Measure how many references point back into the seed set (constraint C2).

    This is the number behind DECIDE-1: if almost no reference lands inside the
    corpus, citation-based expansion has nothing to expand along.
    """
    ids = [p["paperId"] for p in papers if p.get("paperId")]
    seed_set = set(ids)
    sample = ids[:n]
    counts = [p["referenceCount"] for p in papers if p.get("referenceCount") is not None]
    print(
        f"\nReference counts across {len(counts)} papers: "
        f"mean {statistics.mean(counts):.1f}, median {statistics.median(counts)}, "
        f"{sum(1 for c in counts if c == 0)} with none recorded"
    )

    refs: collections.Counter = collections.Counter()
    resolved = 0
    print(f"\nPulling reference lists for {len(sample)} seed papers "
          f"({'with' if settings.semantic_scholar_api_key else 'WITHOUT'} an API key):")
    for start in range(0, len(sample), 50):
        batch = sample[start : start + 50]
        payload = None
        for attempt in range(12):
            resp = requests.post(
                S2_BATCH,
                params={"fields": "references.paperId"},
                json={"ids": batch},
                headers=_headers(),
                timeout=120,
            )
            if resp.status_code == 200:
                payload = resp.json()
                break
            print(f"    {resp.status_code}, waiting 25s (attempt {attempt + 1}/12)", flush=True)
            time.sleep(25)
        if payload is None:
            print("    gave up on this batch")
            continue
        for paper in payload:
            if not paper:
                continue
            resolved += 1
            for ref in paper.get("references") or []:
                if ref and ref.get("paperId"):
                    refs[ref["paperId"]] += 1
        print(f"    {resolved} papers resolved, {len(refs):,} unique references so far")
        time.sleep(5)

    if not resolved:
        print("  could not resolve any batch - rerun with an API key")
        return

    edges = sum(refs.values())
    inside = sum(count for pid, count in refs.items() if pid in seed_set)
    print(f"\nFrom {resolved} seed papers: {edges:,} reference edges, {len(refs):,} unique targets")
    print(f"  INSIDE the seed set: {inside} edges ({100 * inside / edges:.1f}%)")
    print(f"  => roughly {inside / resolved:.2f} internal CITES per paper at this corpus size")
    for k in (1, 2, SNOWBALL_MIN_CITERS, 5):
        kept = sum(1 for c in refs.values() if c >= k)
        label = "  <- DECIDE-1 threshold" if k == SNOWBALL_MIN_CITERS else ""
        print(f"  references cited by >={k} of the {resolved} seeds: {kept:>5,}{label}")


def check_pdfs(papers: list[dict], n: int = PDF_SAMPLE) -> None:
    """Download a few open-access PDFs and check they have a readable text layer."""
    urls = [
        p["openAccessPdf"]["url"]
        for p in papers
        if (p.get("openAccessPdf") or {}).get("url")
    ][:n]
    print(f"\nPDF check on {len(urls)} open-access links:")
    ok = scanned = failed = 0
    for url in urls:
        try:
            resp = requests.get(url, timeout=45, headers={"User-Agent": "paper-search-spike/0.1"})
            resp.raise_for_status()
            doc = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
            chars = sum(len(page.get_text()) for page in doc)
            if chars > 1000:
                ok += 1
                print(f"  OK       {chars:>7} chars, {doc.page_count:>3}p  {url[:70]}")
            else:
                scanned += 1
                print(f"  NO TEXT  {chars:>7} chars, {doc.page_count:>3}p  {url[:70]}")
        except Exception as exc:  # noqa: BLE001 - spike: any failure is just a failure
            failed += 1
            print(f"  FAILED   {type(exc).__name__}: {str(exc)[:60]}  {url[:60]}")
    print(f"  => readable {ok}, no text layer {scanned}, download failed {failed}")


def check_arxiv(n: int = 100) -> None:
    """See how many results arXiv alone returns for the same topic."""
    print(f"\narXiv search for {TOPIC!r} (first {n}):")
    results = list(arxiv.Client(page_size=100, delay_seconds=3).results(
        arxiv.Search(query=TOPIC, max_results=n, sort_by=arxiv.SortCriterion.Relevance)
    ))
    with_summary = sum(1 for r in results if r.summary)
    print(f"  {len(results)} results, {with_summary} with abstract, all have a PDF URL")


def main() -> None:
    """Run the whole spike and print the numbers."""
    print(f"Semantic Scholar bulk search for {TOPIC!r}:")
    total, papers = fetch_semantic_scholar()
    print(f"\nTotal matching papers on Semantic Scholar: {total:,}")
    print(f"Pulled {len(papers)} records (no API key).\n")
    for label, count in summarize(papers).items():
        pct = 100 * count / max(len(papers), 1)
        print(f"  {label:<22} {count:>5}  ({pct:5.1f}%)")
    year_distribution(papers)
    check_pdfs(papers)
    check_arxiv()
    citation_overlap(papers)


if __name__ == "__main__":
    main()
