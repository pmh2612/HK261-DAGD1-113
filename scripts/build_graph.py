"""Stage 3: load KG v1 into Neo4j, print graph statistics, run the target queries.

Reads papers.parquet and entities.parquet from data/processed/<version>/ and
references.parquet from data/raw/<version>/. Idempotent: re-running adds nothing.
The target queries are the ones docs/kg-schema.md §5 commits to.

Run: .venv/bin/python scripts/build_graph.py [--version 2026-10-23]
"""

import argparse
from collections import defaultdict

import pyarrow.parquet as pq
from neo4j import GraphDatabase

from paper_search.config import settings
from paper_search.extraction.entities import acronym_aliases, normalize_entity
from paper_search.graph.loader import (
    ensure_constraints,
    load_citations,
    load_entities,
    load_papers,
)
from paper_search.models import Paper, PaperEntities, read_records

# docs/kg-schema.md §5 - keep the two in sync.
QUERIES = {
    "UC-2 papers using dataset X with method Y": """
        MATCH (:Dataset {key: $dataset})<-[:USES_DATASET]-(p:Paper)
              -[:USES_METHOD]->(:Method {key: $method})
        WHERE NOT p:External
        RETURN p.title AS title, p.year AS year ORDER BY p.citation_count DESC LIMIT 5""",
    "UC-3 neighbourhood of a paper": """
        MATCH (p:Paper {paper_id: $paper_id})
        CALL (p) {
            MATCH (p)-[:CITES]->(o:Paper) RETURN 'cites' AS rel, o
            UNION MATCH (p)<-[:CITES]-(o:Paper) RETURN 'cited by' AS rel, o
            UNION MATCH (p)-[:USES_METHOD|USES_DATASET]->()<-[:USES_METHOD|USES_DATASET]-(o:Paper)
                  WHERE NOT o:External RETURN 'shares method/dataset' AS rel, o
            UNION MATCH (p)<-[:AUTHORED]-(:Author)-[:AUTHORED]->(o:Paper)
                  WHERE o <> p RETURN 'same author' AS rel, o
        }
        RETURN rel, count(DISTINCT o) AS papers ORDER BY rel""",
    "Most-used datasets": """
        MATCH (d:Dataset)<-[:USES_DATASET]-(p:Paper)
        RETURN d.name AS dataset, count(p) AS papers ORDER BY papers DESC LIMIT 5""",
    "Methods used together with a topic": """
        MATCH (t:Topic {key: $topic})<-[:HAS_TOPIC]-(p:Paper)-[:USES_METHOD]->(m:Method)
        RETURN m.name AS method, count(p) AS papers ORDER BY papers DESC LIMIT 5""",
}


def most_common_pair(entities: list[PaperEntities], aliases: dict[str, str]) -> tuple[str, str]:
    """The (dataset, method) pair shared by most papers - a UC-2 example known to exist."""

    def keys(names: list[str]) -> set[str]:
        return {aliases.get(k, k) for n in names if (k := normalize_entity(n))}

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for e in entities:
        for d in keys(e.datasets):
            for m in keys(e.methods):
                counts[d, m] += 1
    return max(counts, key=counts.get)


def main() -> None:
    """Load, then report what the graph holds and what it can answer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=None)
    version = parser.parse_args().version or settings.latest_version()
    out_dir = settings.processed_dir / version

    papers = read_records(out_dir / "papers.parquet", Paper)
    entities = read_records(out_dir / "entities.parquet", PaperEntities)
    aliases = acronym_aliases(p.abstract for p in papers if p.abstract)
    refs: dict[str, list[str]] = defaultdict(list)
    for row in pq.read_table(settings.raw_dir / version / "references.parquet").to_pylist():
        refs[row["paper_id"]].append(row["ref_id"])

    auth = (settings.neo4j_user, settings.neo4j_password)
    with GraphDatabase.driver(settings.neo4j_uri, auth=auth) as driver:
        ensure_constraints(driver)
        load_papers(driver, papers)
        load_citations(driver, refs)
        load_entities(driver, entities, aliases)

        print(f"KG v1 from data version {version}, {len(aliases)} acronym aliases\n\nNodes:")
        for r in driver.execute_query(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY label"
        ).records:
            print(f"  {r['label']:<10} {r['n']:>6}")
        print("Relationships:")
        for r in driver.execute_query(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type"
        ).records:
            print(f"  {r['type']:<13} {r['n']:>6}")

        dataset, method = most_common_pair(entities, aliases)
        top_topic = driver.execute_query(
            "MATCH (t:Topic)<-[:HAS_TOPIC]-(p) "
            "RETURN t.key AS k, count(p) AS n ORDER BY n DESC LIMIT 1"
        ).records[0]["k"]
        top_paper = max((p for p in papers if not p.is_external), key=lambda p: p.citation_count)
        params = {
            "dataset": dataset,
            "method": method,
            "topic": top_topic,
            "paper_id": top_paper.paper_id,
        }
        for name, cypher in QUERIES.items():
            used = {k: v for k, v in params.items() if f"${k}" in cypher}
            print(f"\n{name}  {used or ''}")
            for r in driver.execute_query(cypher, **used).records:
                print("  " + " | ".join(str(v) for v in r.values()))


if __name__ == "__main__":
    main()
