"""Tests for normalization, deduplication and snowball selection in collection."""

from paper_search.collection.semantic_scholar import (
    deduplicate,
    is_english,
    select_external,
    to_paper,
)
from paper_search.extraction.entities import normalize_entity
from paper_search.models import Paper, read_records, write_records


def raw(pid, title="A Title", doi=None, arxiv=None, **extra):
    ids = {k: v for k, v in (("DOI", doi), ("ArXiv", arxiv)) if v}
    return {"paperId": pid, "title": title, "externalIds": ids, **extra}


def test_to_paper_normalizes_fields():
    p = to_paper(
        raw(
            "s1",
            "  Dense   Retrieval ",
            doi="10.1/ABC",
            abstract="a\n b",
            authors=[{"authorId": "7", "name": "Ann"}, {"authorId": None, "name": "Bo"}],
        ),
        "v1",
    )
    assert p.title == "Dense Retrieval" and p.abstract == "a b" and p.doi == "10.1/abc"
    assert p.author_ids == ["7", None] and not p.has_full_text and p.year is None


def test_deduplicate_keys_on_doi_arxiv_and_title():
    papers = [
        to_paper(raw("s1", "Self-RAG", doi="10.1/x"), "v"),
        to_paper(raw("s2", "Self RAG: learning", doi="10.1/x"), "v"),  # same DOI
        to_paper(raw("s3", "Other", arxiv="2310.1"), "v"),
        to_paper(raw("s4", "Other paper", arxiv="2310.1"), "v"),  # same arXiv ID
        to_paper(raw("s5", "self-rag"), "v"),  # same normalized title as s1
        to_paper(raw("s6", "Distinct"), "v"),
    ]
    assert [p.paper_id for p in deduplicate(papers)] == ["s1", "s3", "s6"]


def test_select_external_needs_three_citers_outside_the_seeds():
    refs = {"a": ["x", "y", "b"], "b": ["x", "y"], "c": ["x", "x", "b"]}
    assert select_external(refs) == ["x"]  # y: 2 citers; b: a seed; x counted once per seed


def test_is_english():
    assert is_english("Retrieval-augmented generation for QA")
    assert not is_english("检索增强生成的研究与应用")


def test_normalize_entity_collapses_surface_variants():
    assert normalize_entity("BERT-base") == normalize_entity("bert") == "bert"
    assert normalize_entity("Self-RAG") == normalize_entity("self rag")
    assert normalize_entity("Natural Questions.") == "natural questions"


def test_parquet_round_trip_keeps_nulls_and_ints(tmp_path):
    paper = to_paper(raw("s1", abstract="x", year=None, authors=[]), "v")
    dated = to_paper(raw("s2", year=2024, authors=[{"authorId": "1", "name": "A"}]), "v")
    write_records([paper, dated], tmp_path / "p.parquet")
    back = read_records(tmp_path / "p.parquet", Paper)
    assert back == [paper, dated] and back[0].year is None and back[1].year == 2024
