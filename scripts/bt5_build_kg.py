"""BT5: build the knowledge graph for one corpus level and measure it.

Pipeline, each step with its own cache under data/processed/kg/ and data/raw/kg/:
  1. papers   - papers_<level>.jsonl (BT1)
  2. entities - SciBERT fine-tuned on SciERC (BT2's pick) over every abstract
  3. relations - one default relationship per entity type (BT3, method 1 of the plan)
  4. normalization - normalize_entity + acronym aliases (BT4, rule baseline)
  5. citations - Semantic Scholar references; a reference outside the corpus that at
     least 3 corpus papers cite becomes an :External paper (DECIDE-1)
then loads Neo4j (the database is cleared first: one corpus per graph) and records load
time, node / relationship counts, BT4 node counts and ~10 Cypher queries with timings.

Run: .venv/bin/python scripts/bt5_build_kg.py --level small
"""

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import datetime

from neo4j import GraphDatabase

from paper_search.collection import semantic_scholar as s2
from paper_search.config import PROJECT_ROOT, settings
from paper_search.extraction.entities import acronym_aliases, normalize_entity
from paper_search.graph.loader import (
    ensure_constraints,
    load_citations,
    load_entities,
    load_papers,
)
from paper_search.models import Paper, PaperEntities

#: SciERC type -> (node label, relationship, PaperEntities field). BT3 baseline: the
#: relationship follows from the type alone ("we propose X" and "unlike X" both -> USES).
KINDS = {
    "Method": ("Method", "USES", "methods"),
    "Material": ("Dataset", "EVALUATES_ON", "datasets"),
    "Task": ("Task", "ADDRESSES", "tasks"),
    "Metric": ("Metric", "REPORTS", "metrics"),
}
EXTRACTOR = "scibert-scierc"

QUERIES = {
    "most-used datasets": """
        MATCH (d:Dataset)<-[:EVALUATES_ON]-(p:Paper)
        RETURN d.name AS dataset, count(p) AS papers ORDER BY papers DESC LIMIT 5""",
    "most-used methods": """
        MATCH (m:Method)<-[:USES]-(p:Paper)
        RETURN m.name AS method, count(p) AS papers ORDER BY papers DESC LIMIT 5""",
    "most-studied tasks": """
        MATCH (t:Task)<-[:ADDRESSES]-(p:Paper)
        RETURN t.name AS task, count(p) AS papers ORDER BY papers DESC LIMIT 5""",
    "datasets used for task X": """
        MATCH (:Task {key: $task})<-[:ADDRESSES]-(p:Paper)-[:EVALUATES_ON]->(d:Dataset)
        RETURN d.name AS dataset, count(p) AS papers ORDER BY papers DESC LIMIT 5""",
    "methods used with dataset Y": """
        MATCH (:Dataset {key: $dataset})<-[:EVALUATES_ON]-(p:Paper)-[:USES]->(m:Method)
        RETURN m.name AS method, count(p) AS papers ORDER BY papers DESC LIMIT 5""",
    "papers on task X evaluated on dataset Y": """
        MATCH (:Task {key: $task})<-[:ADDRESSES]-(p:Paper)-[:EVALUATES_ON]->
              (:Dataset {key: $dataset})
        RETURN p.title AS title, p.year AS year LIMIT 5""",
    "metrics reported for task X": """
        MATCH (:Task {key: $task})<-[:ADDRESSES]-(p:Paper)-[:REPORTS]->(m:Metric)
        RETURN m.name AS metric, count(p) AS papers ORDER BY papers DESC LIMIT 5""",
    "authors with most papers on task X": """
        MATCH (:Task {key: $task})<-[:ADDRESSES]-(p:Paper)<-[:AUTHORED]-(a:Author)
        RETURN a.name AS author, count(p) AS papers ORDER BY papers DESC LIMIT 5""",
    "papers the corpus cites most": """
        MATCH (p:Paper)<-[:CITES]-(c:Paper) WHERE NOT c:External
        RETURN p.title AS title, p:External AS external, count(c) AS citers
        ORDER BY citers DESC LIMIT 5""",
    "related papers via shared entities": """
        MATCH (p:Paper {paper_id: $paper_id})-[:USES|EVALUATES_ON|ADDRESSES]->(e)
              <-[:USES|EVALUATES_ON|ADDRESSES]-(o:Paper)
        WHERE o <> p
        RETURN o.title AS title, count(e) AS shared ORDER BY shared DESC LIMIT 5""",
    "fields of study": """
        MATCH (f:FieldOfStudy)<-[:HAS_FIELD]-(p:Paper) WHERE NOT p:External
        RETURN f.name AS field, count(p) AS papers ORDER BY papers DESC LIMIT 5""",
}


def extract(papers: list[Paper], cache) -> dict[str, list[list[str]]]:
    """paper_id -> [[SciERC type, surface form], ...], cached."""
    if cache.exists():
        return json.loads(cache.read_text())
    import spacy
    from spacy.lang.char_classes import ALPHA, HYPHENS
    from spacy.util import compile_infix_regex

    from paper_search.extraction.ner import scibert

    nlp = spacy.blank("en")
    # Keep hyphenated words whole, as SciERC's tokens are: split, "pre-training" gave
    # the entity "pre" (2026-11-27).
    hyphen = rf"(?<=[{ALPHA}0-9])(?:{HYPHENS})(?=[{ALPHA}])"
    infixes = [x for x in nlp.Defaults.infixes if x != hyphen]
    nlp.tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
    nlp.add_pipe("sentencizer")
    docs = []
    for p in papers:
        text = " ".join((p.abstract or "").split())  # no whitespace tokens
        d = nlp(text)
        docs.append(
            {
                "doc_id": p.paper_id,
                "text": text,
                "words": [t.text for t in d],
                "starts": [t.idx for t in d],
                "sentences": [(s.start, s.end) for s in d.sents],
            }
        )
    spans = scibert(settings.processed_dir / "models" / "scibert_scierc")(docs)
    text = {d["doc_id"]: d["text"] for d in docs}
    out = {pid: sorted({(t, text[pid][a:b]) for a, b, t in s}) for pid, s in spans.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def citations(papers: list[Paper], level: str) -> tuple[dict[str, list[str]], list[Paper]]:
    """paper_id -> cited paper_ids, and the :External papers, both cached."""
    by_s2 = {p.s2_id: p.paper_id for p in papers if p.s2_id}
    raw = settings.raw_dir / "kg"
    raw.mkdir(parents=True, exist_ok=True)
    ref_cache, ext_cache = raw / f"references_{level}.json", raw / f"external_{level}.json"
    if not ref_cache.exists():
        ref_cache.write_text(json.dumps(s2.fetch_references(list(by_s2))))
    refs = json.loads(ref_cache.read_text())
    if not ext_cache.exists():
        ext_cache.write_text(json.dumps(s2.fetch_raw(s2.select_external(refs, min_citers=3))))
    version = papers[0].data_version
    external = s2.normalize(json.loads(ext_cache.read_text()), version, is_external=True)
    cites = {by_s2[s]: [by_s2.get(t, t) for t in targets] for s, targets in refs.items()}
    return cites, external


def timed(driver, cypher: str, params: dict, runs: int = 3) -> tuple[list, float]:
    """Records and the median wall time over `runs` (the first run is cold)."""
    times, records = [], []
    for _ in range(runs):
        start = time.perf_counter()
        records = driver.execute_query(cypher, **params).records
        times.append((time.perf_counter() - start) * 1000)
    return records, round(statistics.median(times), 1)


def main() -> None:
    """Extract, load, count, query; write experiments/runs/bt5/<stamp>_<level>/."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", default="small")
    level = parser.parse_args().level

    rows = [json.loads(line) for line in (settings.processed_dir / f"papers_{level}.jsonl").open()]
    papers = [Paper(**r) for r in rows]
    start = time.perf_counter()
    found = extract(papers, settings.processed_dir / "kg" / f"entities_{level}.json")
    extract_s = time.perf_counter() - start
    entities = [
        PaperEntities(
            paper_id=pid,
            extractor=EXTRACTOR,
            **{
                attr: sorted({name for t, name in spans if t == scierc})
                for scierc, (_, _, attr) in KINDS.items()
            },
        )
        for pid, spans in found.items()
    ]
    aliases = acronym_aliases(p.abstract for p in papers if p.abstract)

    # BT4: distinct names per type before and after the rule-based normalization.
    bt4 = {}
    for label, _, attr in KINDS.values():
        names = {n for e in entities for n in getattr(e, attr)}
        keys = {k for n in names if (k := normalize_entity(n))}
        bt4[label] = {
            "surface_forms": len(names),
            "after_normalize": len(keys),
            "after_aliases": len({aliases.get(k, k) for k in keys}),
        }

    cites, external = citations(papers, level)
    auth = (settings.neo4j_user, settings.neo4j_password)
    with GraphDatabase.driver(settings.neo4j_uri, auth=auth) as driver:
        with driver.session() as session:  # one corpus per graph
            session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
        start = time.perf_counter()
        ensure_constraints(driver)
        load_papers(driver, papers + external)
        load_citations(driver, cites)
        load_entities(driver, entities, aliases, tuple(KINDS.values()))
        load_s = time.perf_counter() - start

        nodes = {
            r["label"]: r["n"]
            for r in driver.execute_query(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY label"
            ).records
        }
        rels = {
            r["type"]: r["n"]
            for r in driver.execute_query(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type"
            ).records
        }

        # Parameters that exist in this graph: the most-studied task, the dataset most
        # used for it, and a paper linked to both.
        task = Counter(
            k
            for e in entities
            for n in e.tasks
            if (k := normalize_entity(n)) and (k := aliases.get(k, k))
        ).most_common(1)[0][0]
        rec = driver.execute_query(
            QUERIES["datasets used for task X"].replace("d.name AS dataset", "d.key AS dataset"),
            task=task,
        ).records
        dataset = rec[0]["dataset"] if rec else ""
        hit = driver.execute_query(
            "MATCH (:Task {key: $task})<-[:ADDRESSES]-(p:Paper)-[:EVALUATES_ON]->"
            "(:Dataset {key: $dataset}) RETURN p.paper_id AS id LIMIT 1",
            task=task,
            dataset=dataset,
        ).records
        params = {"task": task, "dataset": dataset, "paper_id": hit[0]["id"] if hit else ""}

        queries = {}
        for name, cypher in QUERIES.items():
            used = {k: v for k, v in params.items() if f"${k}" in cypher}
            records, ms = timed(driver, cypher, used)
            queries[name] = {
                "params": used,
                "ms": ms,
                "rows": [[str(v) for v in r.values()] for r in records],
            }

    result = {
        "level": level,
        "papers": len(papers),
        "external_papers": len(external),
        "extractor": EXTRACTOR,
        "extract_seconds": round(extract_s, 1),
        "load_seconds": round(load_s, 1),
        "nodes": nodes,
        "relationships": rels,
        "bt4_rule_normalization": bt4,
        "acronym_aliases": len(aliases),
        "queries": queries,
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = PROJECT_ROOT / "experiments/runs/bt5" / f"{stamp}_{level}"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print(f"{level}: {len(papers)} papers + {len(external)} external, load {load_s:.1f} s")
    print("nodes", nodes, "\nrelationships", rels, "\nBT4", bt4)
    for name, q in queries.items():
        print(f"\n{name} {q['params'] or ''} ({q['ms']} ms)")
        for row in q["rows"]:
            print("  " + " | ".join(row))


if __name__ == "__main__":
    main()
