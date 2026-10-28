"""arXiv client: monthly listings for the sampling frame, OAI-PMH for metadata (BT1).

The arXiv search API (export.arxiv.org/api) answered every request with 429 when this
was written (2026-10-27), so neither path uses it: robots.txt allows /list pages, and
OAI-PMH GetRecord returns the full arXiv metadata, abstract included.
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import requests

from paper_search.models import Paper

LIST_URL = "https://arxiv.org/list/{category}/{month}"
OAI_URL = "https://oaipmh.arxiv.org/oai"
LIST_DELAY = 15  # robots.txt Crawl-delay
OAI_DELAY = 3  # arXiv's guideline for its APIs: one request every three seconds
PAGE = 2000  # the largest page /list serves
NS = {"a": "http://arxiv.org/OAI/arXiv/"}


def _get(url: str, delay: float, **params) -> str:
    """One polite GET, with backoff on 429 and 5xx."""
    for attempt in range(8):
        time.sleep(delay)
        resp = requests.get(url, params=params, timeout=120)
        if resp.status_code != 429 and resp.status_code < 500:
            resp.raise_for_status()
            return resp.text
        time.sleep(min(10 * 2**attempt, 300))
    resp.raise_for_status()
    return ""  # unreachable: raise_for_status raised on the last 429/5xx


def list_month(category: str, month: str, cache_dir: Path) -> list[str]:
    """The arXiv IDs listed under `category` for `month` ("2024-03"), cross-lists included.

    Only IDs with that month's YYMM prefix are kept, so the frame means "submitted in
    the month" however late a cross-list was added. Cached, since a past month's
    listing is what defines the sample.
    """
    path = cache_dir / f"{category}_{month}.json"
    if path.exists():
        return json.loads(path.read_text())
    ids: list[str] = []
    skip, total = 0, 1
    while skip < total:
        html = _get(
            LIST_URL.format(category=category, month=month), LIST_DELAY, skip=skip, show=PAGE
        )
        total = int(re.search(r"Total of (\d+) entries", html).group(1))
        ids += re.findall(r'href\s*=\s*"/abs/(\d{4}\.\d{4,5})"', html)
        skip += PAGE
    prefix = month[2:4] + month[5:7]
    ids = [i for i in dict.fromkeys(ids) if i.startswith(prefix)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ids))
    return ids


def fetch_record(arxiv_id: str, cache_dir: Path) -> tuple[str, bool]:
    """Raw OAI-PMH record for one paper, and whether it came from the cache."""
    path = cache_dir / f"{arxiv_id}.xml"
    if path.exists():
        return path.read_text(), True
    xml = _get(
        OAI_URL,
        OAI_DELAY,
        verb="GetRecord",
        identifier=f"oai:arXiv.org:{arxiv_id}",
        metadataPrefix="arXiv",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    return xml, False


def to_paper(xml: str, data_version: str) -> Paper | None:
    """Normalize one OAI-PMH record; None when OAI answered with an error instead."""
    meta = ET.fromstring(xml).find(".//a:arXiv", NS)
    if meta is None:
        return None

    def text(tag: str) -> str:
        return " ".join(meta.findtext(f"a:{tag}", "", NS).split())

    authors = [
        " ".join(filter(None, (a.findtext("a:forenames", "", NS), a.findtext("a:keyname", "", NS))))
        for a in meta.findall("a:authors/a:author", NS)
    ]
    arxiv_id = text("id")
    return Paper(
        paper_id=arxiv_id,
        title=text("title"),
        abstract=text("abstract") or None,
        # The ID says when v1 was submitted; OAI's <created> can be a later version's date.
        year=2000 + int(arxiv_id[:2]),
        venue=text("journal-ref") or None,  # a citation string, the closest arXiv has
        authors=authors,
        author_ids=[None] * len(authors),
        doi=next(iter(text("doi").lower().split()), None),  # some records list several
        arxiv_id=arxiv_id,
        citation_count=None,
        reference_count=None,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        has_full_text=True,
        is_external=False,
        source="arxiv",
        data_version=data_version,
        categories=text("categories").split(),
    )


def merge(ax: Paper, s2: Paper | None) -> Paper:
    """The arXiv record, enriched with what only Semantic Scholar has.

    arXiv wins on title, abstract and categories (it is the source of record for a
    preprint); S2 wins on publication year, venue, author IDs, citations and fields.
    """
    if s2 is None:
        return ax
    return replace(
        ax,
        year=s2.year or ax.year,
        venue=s2.venue or ax.venue,
        authors=s2.authors or ax.authors,
        author_ids=s2.author_ids if s2.authors else ax.author_ids,
        doi=ax.doi or s2.doi,
        abstract=ax.abstract or s2.abstract,
        citation_count=s2.citation_count,
        reference_count=s2.reference_count,
        fields_of_study=s2.fields_of_study,
        s2_id=s2.s2_id,
        source="both",
    )
