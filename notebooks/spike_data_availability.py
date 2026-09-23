"""Throwaway spike: can we actually collect 500-1,000 usable papers on one topic?

Answers the questions `docs/requirements.md` has to state as scope limits:
how many papers exist for the topic, how many carry an abstract, how many
expose a downloadable open-access PDF, and how many of those PDFs have a real
text layer PyMuPDF can read (scanned PDFs are useless to us).

Not production code - it lives in notebooks/ and nothing imports it.
Run: .venv/bin/python notebooks/spike_data_availability.py
"""

import io
import time

import arxiv
import fitz  # PyMuPDF
import requests

TOPIC = "retrieval augmented generation"
S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = (
    "title,abstract,year,venue,authors,citationCount,referenceCount,externalIds,openAccessPdf"
)
TARGET = 1000
PDF_SAMPLE = 8


def fetch_semantic_scholar(target: int = TARGET) -> tuple[int, list[dict]]:
    """Page through the bulk search endpoint. Returns (total available, records)."""
    papers: list[dict] = []
    total = 0
    token = None
    while len(papers) < target:
        params = {"query": TOPIC, "fields": S2_FIELDS, "limit": 1000}
        if token:
            params["token"] = token
        resp = requests.get(S2_BULK, params=params, timeout=60)
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
    }


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
    check_pdfs(papers)
    check_arxiv()


if __name__ == "__main__":
    main()
