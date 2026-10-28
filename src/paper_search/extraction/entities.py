"""Entity extraction and name normalization (FR-9, FR-10).

KG v1 uses GLiNER zero-shot NER as a stand-in extractor. FR-9 asks for an in-process LLM
with schema-constrained decoding; that waits on DECIDE-7 and on the comparison in
notebooks/exp_extraction.py. Every PaperEntities records which extractor made it, so the
graph can be rebuilt when the winner is chosen.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import lru_cache

from gliner import GLiNER

from paper_search.models import Paper, PaperEntities, Section

GLINER_MODEL = "urchade/gliner_medium-v2.1"
#: GLiNER label -> PaperEntities field. Research tasks become Topic nodes (kg-schema.md).
LABELS = {"method": "methods", "dataset": "datasets", "task": "topics"}
THRESHOLD = 0.5

#: Size/variant suffixes that do not make a different method: "BERT-base" is BERT.
_VARIANT = re.compile(r"(?:[\s-](?:base|large|small|tiny|mini|xl|xxl|uncased|cased))+$")
#: A trailing type word adds nothing: "RAG methods" is RAG, "QA tasks" is QA.
_HEAD = re.compile(r"\s(?:methods?|approach(?:es)?|techniques?|tasks?|datasets?)$")
#: A name made only of these words names no particular entity ("existing methods",
#: "four datasets", "training data"), so it gets no node.
GENERIC = set(
    """a an the our this these other existing traditional current previous prior
    conventional standard novel new proposed various several multiple different many
    two three four five six public benchmark benchmarks real world large scale complex
    downstream fixed external user users method methods approach approaches technique
    techniques model models data dataset datasets task tasks document documents query
    queries question questions knowledge base bases training synthetic structured corpus
    text texts information prompt prompts table tables image images database databases
    evaluation source sources language llm llms""".split()  # noqa: SIM905 - easier to edit as prose
)
#: Hyphen plus the Unicode dashes abstracts use ("Retrieval‐Augmented"), for [...] classes.
_DASH = r"‐-―−-"
#: "Long Form (LF)" in running text. The long form is matched against initials later.
_DEFINITION = re.compile(rf"((?:[\w'{_DASH}]+\s+){{1,8}})\(([A-Z][A-Za-z]{{1,9}})\)")


def normalize_entity(name: str) -> str:
    """Canonical key, so surface variants of one entity collapse to one node.

    Returns "" for a generic phrase; callers drop empty keys.
    """
    key = re.sub(r"\([^)]*\)?", " ", name)  # "(RAG)", or "(rag" when the span is cut
    key = re.sub(rf"[_/{_DASH}]", " ", key).lower()
    key = " ".join(key.split()).strip(" .,;:()[]\"'")
    key = _VARIANT.sub("", _HEAD.sub("", key)).strip()
    return "" if all(w in GENERIC for w in key.split()) else key


def acronym_aliases(texts: Iterable[str]) -> dict[str, str]:
    """Acronym key -> long-form key, read off definitions like "In-Context Learning (ICL)".

    The long form is the shortest run of words before the parenthesis whose initials
    spell the acronym. An acronym defined two ways keeps its most frequent long form.
    """
    # ponytail: initials only - "GraphRAG" or "Self-RAG" style names are never merged.
    # Embedding-based clustering if paraphrase fragmentation still shows in the graph.
    seen: dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        for before, abbr in _DEFINITION.findall(text):
            words = re.findall(rf"[^\s{_DASH}]+", before)
            for n in range(1, len(words) + 1):
                if "".join(w[0] for w in words[-n:]).lower() == abbr.lower():
                    short, long = normalize_entity(abbr), normalize_entity(" ".join(words[-n:]))
                    if short and long and short != long:
                        seen[short][long] += 1
                    break
    return {short: longs.most_common(1)[0][0] for short, longs in seen.items()}


@lru_cache(maxsize=1)
def _model() -> GLiNER:
    return GLiNER.from_pretrained(GLINER_MODEL)


def extract_entities(paper: Paper, sections: list[Section]) -> PaperEntities:
    """Entities from the abstract (title as fallback). Sections are unused until PDFs land."""
    text = paper.abstract or paper.title
    found: dict[str, list[str]] = {field: [] for field in LABELS.values()}
    for e in _model().predict_entities(text, list(LABELS), threshold=THRESHOLD):
        names = found[LABELS[e["label"]]]
        if e["text"] not in names:
            names.append(e["text"])
    return PaperEntities(paper_id=paper.paper_id, extractor=GLINER_MODEL, **found)
