"""BT4: entity normalization - which extracted names are the same entity?

1. Pairs. Candidate pairs of distinct names of one type, from three generators so the
   pool is not built by any single method it then judges: rapidfuzz token_sort_ratio
   >= 70, bge-small cosine >= 0.80, and acronym <-> long form (acronym_aliases). Names
   equal after lowercasing are left out (trivially the same). A seeded sample, equal
   parts per generator, is labelled same / different by the local LLM and cached in
   data/annotations/entity_pairs.jsonl.
2. Methods. Rule (normalize_entity + acronym aliases, the KG's baseline), rapidfuzz
   ratio at a threshold, embedding cosine at a threshold, and rule-or-fuzzy. Thresholds
   are chosen on the dev half of the pairs, pairwise P/R/F1 reported on the test half,
   plus how many nodes each method leaves when applied to every name (connected
   components of its merges) and the largest group, which shows over-merging.

The LLM labels the pairs, so it is not one of the compared methods here.

Run: .venv/bin/python scripts/bt4_normalize.py --level small
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime

import numpy as np
from rapidfuzz import fuzz, process

from paper_search.config import PROJECT_ROOT, settings
from paper_search.eval.llm_label import chooser
from paper_search.extraction.entities import acronym_aliases, normalize_entity

LABEL = {"Method": "Method", "Material": "Dataset", "Task": "Task", "Metric": "Metric"}
QUOTA = {"Method": 160, "Task": 120, "Dataset": 70, "Metric": 50}  # 400 pairs
GENERATORS = ("fuzzy", "embedding", "acronym")
EMBEDDER = "BAAI/bge-small-en-v1.5"
FUZZY_GRID = list(range(70, 101, 5))
EMBED_GRID = [round(0.80 + 0.02 * i, 2) for i in range(10)]
PAIRS = PROJECT_ROOT / "data/annotations/entity_pairs.jsonl"

PROMPT = """Two names were extracted from abstracts of computer science papers, both tagged \
as a {kind}.

Name A: {a}
Name B: {b}

Do both names refer to the same {kind}, so that a knowledge graph should store them as \
one node? Spelling, capitalisation, plural, an acronym and its full form, and size \
variants of one model (BERT and BERT-base) count as the same. Different models, \
versions, datasets or tasks (GPT-3 and GPT-4, SQuAD and SQuAD 2.0) are different.

Answer Yes or No."""


def names_by_type(level: str) -> dict[str, list[str]]:
    """Distinct surface names per type, one per lowercase form, most frequent first."""
    found = json.loads((settings.processed_dir / "kg" / f"entities_{level}.json").read_text())
    counts: dict[str, Counter] = {t: Counter() for t in LABEL.values()}
    for spans in found.values():
        for t, name in spans:
            counts[LABEL[t]][name] += 1
    out = {}
    for t, c in counts.items():
        seen, names = set(), []
        for name, _ in c.most_common():
            if name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
        out[t] = names
    return out


def key(name: str, aliases: dict[str, str]) -> str:
    """The rule method's node key; "" for a generic name (no node)."""
    k = normalize_entity(name)
    return aliases.get(k, k)


def fuzzy_pairs(names: list[str], cutoff: int) -> set[tuple[int, int]]:
    """Index pairs i < j with token_sort_ratio >= cutoff on lowercase names."""
    low = [n.lower() for n in names]
    m = process.cdist(
        low, low, scorer=fuzz.token_sort_ratio, score_cutoff=cutoff, workers=-1, dtype=np.uint8
    )
    i, j = np.nonzero(m)
    return {(a, b) for a, b in zip(i.tolist(), j.tolist(), strict=True) if a < b}


def embed_pairs(vecs: np.ndarray, cutoff: float) -> set[tuple[int, int]]:
    """Index pairs i < j with cosine >= cutoff, a block of rows at a time."""
    out = set()
    for start in range(0, len(vecs), 1000):
        sim = vecs[start : start + 1000] @ vecs.T
        i, j = np.nonzero(sim >= cutoff)
        out |= {
            (a + start, b) for a, b in zip(i.tolist(), j.tolist(), strict=True) if a + start < b
        }
    return out


def components(n: int, pairs: set[tuple[int, int]]) -> list[int]:
    """Group sizes after merging every pair (union-find)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    return list(Counter(find(x) for x in range(n)).values())


def prf(pred: list[bool], gold: list[bool]) -> dict:
    """Pairwise precision / recall / F1 of "same"."""
    tp = sum(p and g for p, g in zip(pred, gold, strict=True))
    p = tp / max(sum(pred), 1)
    r = tp / max(sum(gold), 1)
    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(2 * p * r / max(p + r, 1e-9), 3),
    }


def main() -> None:
    """Sample and label pairs (cached), then compare the methods."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", default="small")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    papers = [
        json.loads(line) for line in (settings.processed_dir / f"papers_{args.level}.jsonl").open()
    ]
    aliases = acronym_aliases(p["abstract"] for p in papers if p["abstract"])
    names = names_by_type(args.level)
    embedder = SentenceTransformer(EMBEDDER)
    vecs = {
        t: embedder.encode(ns, normalize_embeddings=True, batch_size=256) for t, ns in names.items()
    }

    if not PAIRS.exists():
        rng = random.Random(42)
        todo = []
        for t, ns in names.items():
            by_key: dict[str, list[int]] = {}
            for i, n in enumerate(ns):
                by_key.setdefault(normalize_entity(n), []).append(i)
            found = {
                "fuzzy": fuzzy_pairs(ns, 70),
                "embedding": embed_pairs(vecs[t], 0.80),
                "acronym": {
                    tuple(sorted((a, b)))
                    for short, long in aliases.items()
                    for a in by_key.get(short, [])
                    for b in by_key.get(long, [])
                },
            }
            picked: set[tuple[int, int]] = set()
            order = sorted(found, key=lambda g: len(found[g]))  # smallest pool first
            for k, g in enumerate(order):  # equal shares; a short pool's rest goes on
                share = (QUOTA[t] - len(picked)) // (len(order) - k)
                pool = sorted(found[g] - picked)
                picked |= set(rng.sample(pool, min(share, len(pool))))
            for a, b in sorted(picked):
                todo.append(
                    {
                        "type": t,
                        "a": ns[a],
                        "b": ns[b],
                        "found_by": [g for g in GENERATORS if (a, b) in found[g]],
                    }
                )
        rng.shuffle(todo)
        choose = chooser(args.model)
        probs = choose(
            [PROMPT.format(kind=r["type"].lower(), a=r["a"], b=r["b"]) for r in todo], ["Yes", "No"]
        )
        with PAIRS.open("w", encoding="utf-8") as f:
            for n, (r, (yes, _)) in enumerate(zip(todo, probs, strict=True)):
                row = {
                    **r,
                    "same": yes > 0.5,
                    "p_yes": yes,
                    "split": "dev" if n % 2 == 0 else "test",
                }
                row["source"] = args.model
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"labelled {len(todo)} pairs into {PAIRS}")

    rows = [json.loads(line) for line in PAIRS.open(encoding="utf-8")]
    emb = {
        (r["a"], r["b"]): float(
            embedder.encode([r["a"], r["b"]], normalize_embeddings=True).prod(0).sum()
        )
        for r in rows
    }
    ratio = {(r["a"], r["b"]): fuzz.token_sort_ratio(r["a"].lower(), r["b"].lower()) for r in rows}

    def rule(r: dict) -> bool:
        ka, kb = key(r["a"], aliases), key(r["b"], aliases)
        return bool(ka) and ka == kb

    families = {
        "rule": {None: rule},
        "fuzzy": {t: (lambda r, t=t: ratio[r["a"], r["b"]] >= t) for t in FUZZY_GRID},
        "embedding": {t: (lambda r, t=t: emb[r["a"], r["b"]] >= t) for t in EMBED_GRID},
        "rule_or_embedding": {
            t: (lambda r, t=t: rule(r) or emb[r["a"], r["b"]] >= t) for t in EMBED_GRID
        },
        "rule_or_fuzzy": {
            t: (lambda r, t=t: rule(r) or ratio[r["a"], r["b"]] >= t) for t in FUZZY_GRID
        },
    }
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    methods = {}
    for fam, grid in families.items():
        best = max(
            grid,
            key=lambda t: (prf([grid[t](r) for r in dev], [r["same"] for r in dev])["f1"], t or 0),
        )
        methods[fam] = {
            "threshold": best,
            "dev": prf([grid[best](r) for r in dev], [r["same"] for r in dev]),
            "test": prf([grid[best](r) for r in test], [r["same"] for r in test]),
            # The acronym third of the pool comes from the aliases the rule itself uses,
            # so the rule gets it right by construction: score without it as well.
            "test_without_acronym_pairs": prf(
                [grid[best](r) for r in test if "acronym" not in r["found_by"]],
                [r["same"] for r in test if "acronym" not in r["found_by"]],
            ),
            "test_by_type": {
                t: prf(
                    [grid[best](r) for r in test if r["type"] == t],
                    [r["same"] for r in test if r["type"] == t],
                )
                for t in QUOTA
            },
        }

    # Applied to every name: how many nodes remain, and the largest merged group.
    nodes = {}
    for t, ns in names.items():
        keys = [key(n, aliases) for n in ns]
        kept = [i for i, k in enumerate(keys) if k]
        rule_groups = Counter(keys[i] for i in kept)
        f_t, e_t = methods["fuzzy"]["threshold"], methods["embedding"]["threshold"]
        fz = components(len(ns), fuzzy_pairs(ns, f_t))
        em = components(len(ns), embed_pairs(vecs[t], e_t))
        nodes[t] = {
            "names": len(ns),
            "rule": {"nodes": len(rule_groups), "largest": max(rule_groups.values())},
            f"fuzzy@{f_t}": {"nodes": len(fz), "largest": max(fz)},
            f"embedding@{e_t}": {"nodes": len(em), "largest": max(em)},
        }

    result = {
        "level": args.level,
        "labeller": rows[0]["source"],
        "pairs": len(rows),
        "same": sum(r["same"] for r in rows),
        "pairs_by_type": dict(Counter(r["type"] for r in rows)),
        "methods": methods,
        "nodes_after": nodes,
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = PROJECT_ROOT / "experiments/runs/bt4" / f"{stamp}_{args.level}"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
