"""Shared types passed between stages - the contracts in docs/architecture.md §4.1.

Plain frozen dataclasses. Stages exchange them as Parquet files, never by calling each
other's functions.
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Paper:
    """One paper, normalized from whichever source(s) it came from."""

    paper_id: str  # primary key everywhere: S2 ID for the RAG corpus, arXiv ID for BT1 on
    title: str
    abstract: str | None
    year: int | None
    venue: str | None
    authors: list[str]
    #: Semantic Scholar author IDs, parallel to `authors`. Names alone collide, so the
    #: graph keys authors on these (kg-schema.md); None where S2 has no ID.
    author_ids: list[str | None]
    doi: str | None
    arxiv_id: str | None
    citation_count: int | None  # None = the source does not know (arXiv)
    reference_count: int | None
    pdf_url: str | None
    has_full_text: bool  # an arXiv ID or open-access PDF exists (constraint C1)
    is_external: bool  # snowballed reference, graph-only (DECIDE-1)
    source: str  # "semantic_scholar" | "arxiv" | "both"
    data_version: str
    categories: list[str] = field(default_factory=list)  # arXiv, primary first
    fields_of_study: list[str] = field(default_factory=list)  # Semantic Scholar
    s2_id: str | None = None  # where paper_id is an arXiv ID; S2 references use S2 IDs


@dataclass(frozen=True)
class Section:
    """A canonical section of a paper's full text (FR-7)."""

    paper_id: str
    name: str  # one of config.SECTION_NAMES
    order: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit with the provenance a citation needs (FR-14, architecture §5)."""

    chunk_id: str  # deterministic: f"{paper_id}:{source}:{index}"
    paper_id: str
    source: str  # "Abstract", or a section name from config.SECTION_NAMES
    index: int  # window index within that source, 0-based
    char_start: int  # offsets into the source text, for exact quoting
    char_end: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    """One ranked result from a single retriever."""

    chunk: Chunk
    score: float
    retriever: str  # "bm25" | "vector"


@dataclass(frozen=True)
class PaperEntities:
    """Normalized entity names extracted from one paper (FR-9, FR-10)."""

    paper_id: str
    methods: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    extractor: str = ""  # which model produced these - part of the graph's identity


def write_records(records, path: Path) -> None:
    """Dataclass instances -> Parquet.

    pyarrow, not pandas: pandas turns a nullable int column (`year`) into floats with NaN.
    """
    pq.write_table(pa.Table.from_pylist([asdict(r) for r in records]), path)


def read_records(path: Path, cls: type) -> list:
    """Parquet -> instances of the dataclass `cls`, with plain Python values."""
    names = [f.name for f in fields(cls)]
    return [cls(**row) for row in pq.read_table(path, columns=names).to_pylist()]
