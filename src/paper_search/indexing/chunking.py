"""Section-aware windowed chunking with provenance (FR-13, FR-14, DECIDE-2 option C)."""

from functools import lru_cache

from transformers import AutoTokenizer

from paper_search.config import SECTION_NAMES, settings
from paper_search.models import Chunk, Paper, Section

#: Tokens per window. bge-small reads 512 including [CLS]/[SEP]; the title is prepended
#: at index time (build.py), so the window leaves room for it rather than be truncated.
WINDOW = 448
OVERLAP = 64


@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(settings.embedding_model)


def windows(text: str) -> list[tuple[int, int]]:
    """Character spans of overlapping WINDOW-token windows covering `text`."""
    offsets = _tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    if not offsets:
        return [(0, len(text))]
    step = WINDOW - OVERLAP
    spans = []
    for start in range(0, max(len(offsets) - OVERLAP, 1), step):
        win = offsets[start : start + WINDOW]
        spans.append((win[0][0], win[-1][1]))
    return spans


def source_texts(paper: Paper, sections: list[Section]) -> dict[str, str]:
    """Map source name -> full text of that source, the text chunk offsets point into.

    Sections sharing a name (two "Other" blocks) are joined, so `source` stays unique.
    """
    texts = {"Abstract": paper.abstract} if paper.abstract else {}
    for s in sorted(sections, key=lambda s: s.order):
        texts[s.name] = texts[s.name] + "\n\n" + s.text if s.name in texts else s.text
    return texts


def chunk_paper(paper: Paper, sections: list[Section]) -> list[Chunk]:
    """Every source of a paper split into windows.

    Raises for a paper with no text at all - those are filtered out before indexing,
    never indexed as zero chunks (architecture §4.2).
    """
    texts = source_texts(paper, sections)
    if not texts:
        raise ValueError(f"{paper.paper_id} has neither abstract nor sections")
    return [
        Chunk(
            chunk_id=f"{paper.paper_id}:{source}:{i}",
            paper_id=paper.paper_id,
            source=source,
            index=i,
            char_start=start,
            char_end=end,
            text=text[start:end],
        )
        for source, text in texts.items()
        for i, (start, end) in enumerate(windows(text))
    ]


def check_provenance(chunks: list[Chunk], texts: dict[str, dict[str, str]]) -> None:
    """Invariants P1, P2 and P4 (architecture §5.2). `texts`: paper_id -> source_texts.

    Raises on the first violation: a silently wrong index is worse than no index.
    """
    seen = set()
    for c in chunks:
        assert c.paper_id and c.source in SECTION_NAMES, f"P1 violated: {c.chunk_id}"
        source = texts[c.paper_id][c.source]
        assert c.text == source[c.char_start : c.char_end], f"P2 violated: {c.chunk_id}"
        assert c.chunk_id == f"{c.paper_id}:{c.source}:{c.index}", f"P4 violated: {c.chunk_id}"
        assert c.chunk_id not in seen, f"P4 violated, duplicate: {c.chunk_id}"
        seen.add(c.chunk_id)
