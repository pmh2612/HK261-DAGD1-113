"""Tests for arXiv OAI-PMH parsing and the arXiv + Semantic Scholar merge."""

from paper_search.collection.arxiv_oai import merge, to_paper
from paper_search.collection.semantic_scholar import to_paper as s2_paper

RECORD = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><GetRecord><record><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/">
  <id>2403.00368</id><created>2026-03-20</created>
  <authors>
    <author><keyname>Bruun</keyname><forenames>Simone Borg</forenames></author>
    <author><keyname>Lioma</keyname></author>
  </authors>
  <title>Recommending Target
     Actions</title>
  <categories>cs.IR cs.LG</categories>
  <doi>10.1145/3606950 10.1145/other</doi>
  <abstract>  Providing personalized
  recommendations.</abstract>
</arXiv></metadata></record></GetRecord></OAI-PMH>"""

ERROR = """<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
<error code="idDoesNotExist">no such record</error></OAI-PMH>"""


def test_to_paper_parses_oai_record():
    p = to_paper(RECORD, "v")
    assert p.paper_id == p.arxiv_id == "2403.00368" and p.year == 2024
    assert p.title == "Recommending Target Actions"
    assert p.abstract == "Providing personalized recommendations."
    assert p.authors == ["Simone Borg Bruun", "Lioma"] and p.author_ids == [None, None]
    assert p.categories == ["cs.IR", "cs.LG"] and p.doi == "10.1145/3606950"
    assert p.citation_count is None and p.source == "arxiv"
    assert to_paper(ERROR, "v") is None


def test_merge_keeps_arxiv_text_and_s2_metadata():
    ax = to_paper(RECORD, "v")
    s2 = s2_paper(
        {
            "paperId": "abc",
            "title": "Other title",
            "abstract": None,
            "year": 2023,
            "venue": "TORS",
            "authors": [{"authorId": "1", "name": "S. Bruun"}],
            "citationCount": 5,
            "externalIds": {"ArXiv": "2403.00368"},
            "s2FieldsOfStudy": [{"category": "Computer Science"}, {"category": "Computer Science"}],
        },
        "v",
    )
    m = merge(ax, s2)
    assert m.paper_id == "2403.00368" and m.title == ax.title and m.abstract == ax.abstract
    assert (m.year, m.venue, m.citation_count, m.s2_id) == (2023, "TORS", 5, "abc")
    assert m.author_ids == ["1"] and m.fields_of_study == ["Computer Science"]
    assert m.categories == ["cs.IR", "cs.LG"] and m.source == "both"
    assert merge(ax, None) == ax
    preprint = s2_paper({"paperId": "x", "venue": "arXiv.org", "externalIds": {}}, "v")
    assert preprint.venue is None
