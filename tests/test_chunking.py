"""Tests for `paper_search.indexing.chunking` - the provenance invariants (architecture §5.2)."""

import pytest

from paper_search.indexing.chunking import (
    OVERLAP,
    WINDOW,
    check_provenance,
    chunk_paper,
    source_texts,
    windows,
)
from paper_search.models import Paper, Section


def make_paper(abstract):
    return Paper(
        paper_id="p1",
        title="T",
        abstract=abstract,
        year=None,
        venue=None,
        authors=[],
        author_ids=[],
        doi=None,
        arxiv_id=None,
        citation_count=0,
        reference_count=0,
        pdf_url=None,
        has_full_text=False,
        is_external=False,
        source="semantic_scholar",
        data_version="test",
    )


def test_short_text_is_one_window():
    assert windows("dense retrieval works") == [(0, len("dense retrieval works"))]


def test_long_text_windows_overlap_and_cover_everything():
    text = " ".join(f"word{i}" for i in range(2000))
    spans = windows(text)
    assert len(spans) > 1
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert next_start < prev_end  # consecutive windows overlap
    assert WINDOW > OVERLAP


def test_chunks_satisfy_provenance_and_are_stable():
    paper = make_paper(" ".join(f"tok{i}" for i in range(1500)))
    sections = [Section("p1", "Method", 0, "We use BM25."), Section("p1", "Other", 1, "x")]
    chunks = chunk_paper(paper, sections)
    check_provenance(chunks, {"p1": source_texts(paper, sections)})  # P1, P2, P4
    assert {c.source for c in chunks} == {"Abstract", "Method", "Other"}
    assert chunk_paper(paper, sections) == chunks  # P4: same input, same IDs


def test_paper_without_text_is_refused():
    with pytest.raises(ValueError):
        chunk_paper(make_paper(None), [])


def test_check_provenance_catches_a_wrong_slice():
    paper = make_paper("abc def")
    (chunk,) = chunk_paper(paper, [])
    bad = chunk.__class__(**{**chunk.__dict__, "char_start": 1})
    with pytest.raises(AssertionError, match="P2"):
        check_provenance([bad], {"p1": source_texts(paper, [])})
