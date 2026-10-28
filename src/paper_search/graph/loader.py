"""Neo4j schema and idempotent loaders (FR-11, FR-12). Model: docs/kg-schema.md.

Every write is a MERGE on a uniquely constrained key, so re-running any loader on the
same input changes nothing.
"""

from collections.abc import Iterable

from neo4j import Driver

from paper_search.extraction.entities import normalize_entity
from paper_search.models import Paper, PaperEntities

CONSTRAINTS = {
    "Paper": "paper_id",
    "Author": "author_id",
    "Venue": "key",
    "Topic": "key",
    "Method": "key",
    "Dataset": "key",
}
BATCH = 500


def _batched(rows: list, size: int = BATCH):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def ensure_constraints(driver: Driver) -> None:
    """One uniqueness constraint per node type.

    `:External` papers are `:Paper` nodes too, so the Paper constraint already stops a
    paper being duplicated when it is collected first as a reference and later as a
    seed (DECIDE-1).
    """
    for label, key in CONSTRAINTS.items():
        driver.execute_query(
            f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        )


def load_papers(driver: Driver, papers: Iterable[Paper]) -> None:
    """Paper nodes with their authors and venue. `:External` follows `is_external`."""
    rows = [
        {
            "props": {
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": p.abstract,
                "year": p.year,
                "doi": p.doi,
                "arxiv_id": p.arxiv_id,
                "citation_count": p.citation_count,
                "has_full_text": p.has_full_text,
                "data_version": p.data_version,
            },
            "external": p.is_external,
            "venue": p.venue,
            "venue_key": " ".join(p.venue.lower().split()) if p.venue else None,
            "authors": [
                # S2 author ID when there is one; otherwise the name is all we have.
                {"id": aid or f"name:{name.lower()}", "name": name, "pos": i}
                for i, (name, aid) in enumerate(zip(p.authors, p.author_ids, strict=True))
            ],
        }
        for p in papers
    ]
    for batch in _batched(rows):
        driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (p:Paper {paper_id: row.props.paper_id})
            SET p += row.props
            FOREACH (_ IN CASE WHEN row.external THEN [1] ELSE [] END | SET p:External)
            FOREACH (_ IN CASE WHEN row.external THEN [] ELSE [1] END | REMOVE p:External)
            FOREACH (_ IN CASE WHEN row.venue_key IS NULL THEN [] ELSE [1] END |
                MERGE (v:Venue {key: row.venue_key})
                ON CREATE SET v.name = row.venue
                MERGE (p)-[:PUBLISHED_IN]->(v))
            WITH p, row
            UNWIND row.authors AS au
            MERGE (a:Author {author_id: au.id})
            ON CREATE SET a.name = au.name
            MERGE (a)-[r:AUTHORED]->(p)
            SET r.position = au.pos
            """,
            rows=batch,
        )


def load_citations(driver: Driver, refs: dict[str, list[str]]) -> None:
    """CITES between papers already in the graph; references outside it are dropped."""
    pairs = [{"src": s, "dst": d} for s, dsts in refs.items() for d in dsts]
    for batch in _batched(pairs, 5000):
        driver.execute_query(
            """
            UNWIND $pairs AS pair
            MATCH (a:Paper {paper_id: pair.src}), (b:Paper {paper_id: pair.dst})
            MERGE (a)-[:CITES]->(b)
            """,
            pairs=batch,
        )


def load_entities(
    driver: Driver, entities: Iterable[PaperEntities], aliases: dict[str, str]
) -> None:
    """Method / Dataset / Topic nodes keyed on the normalized name, linked to papers.

    `aliases` maps an acronym key to its long-form key (`acronym_aliases`). The first
    surface form seen becomes the display `name`; the relationship records which
    extractor asserted it, so a rebuild with another model is traceable.

    The entity layer is derived data, so it is dropped and rebuilt whole: otherwise a
    change to normalization would leave the old keys behind as orphan nodes.
    """
    driver.execute_query("MATCH (n:Method|Dataset|Topic) DETACH DELETE n")
    rows = [
        {
            "paper_id": e.paper_id,
            "label": label,
            "rel": rel,
            "key": aliases.get(key, key),
            "name": name,
            "extractor": e.extractor,
        }
        for e in entities
        for label, rel, names in (
            ("Method", "USES_METHOD", e.methods),
            ("Dataset", "USES_DATASET", e.datasets),
            ("Topic", "HAS_TOPIC", e.topics),
        )
        for name in names
        if (key := normalize_entity(name))
    ]
    for label, rel in (
        ("Method", "USES_METHOD"),
        ("Dataset", "USES_DATASET"),
        ("Topic", "HAS_TOPIC"),
    ):
        part = [r for r in rows if r["label"] == label]
        for batch in _batched(part):
            driver.execute_query(
                f"""
                UNWIND $rows AS row
                MATCH (p:Paper {{paper_id: row.paper_id}})
                MERGE (n:{label} {{key: row.key}})
                ON CREATE SET n.name = row.name
                MERGE (p)-[r:{rel}]->(n)
                SET r.extractor = row.extractor
                """,
                rows=batch,
            )
