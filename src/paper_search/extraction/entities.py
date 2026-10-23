"""Entity extraction and name normalization (FR-9, FR-10).

KG v1 uses GLiNER zero-shot NER as a stand-in extractor. FR-9 asks for an in-process LLM
with schema-constrained decoding; that waits on DECIDE-7 and on the comparison in
notebooks/exp_extraction.py. Every PaperEntities records which extractor made it, so the
graph can be rebuilt when the winner is chosen.
"""

import re
from functools import lru_cache

from gliner import GLiNER

from paper_search.models import Paper, PaperEntities, Section

GLINER_MODEL = "urchade/gliner_medium-v2.1"
#: GLiNER label -> PaperEntities field. Research tasks become Topic nodes (kg-schema.md).
LABELS = {"method": "methods", "dataset": "datasets", "task": "topics"}
THRESHOLD = 0.5

#: Size/variant suffixes that do not make a different method: "BERT-base" is BERT.
_VARIANT = re.compile(r"(?:[\s-](?:base|large|small|tiny|mini|xl|xxl|uncased|cased))+$")


def normalize_entity(name: str) -> str:
    """Canonical key, so surface variants of one entity collapse to one node."""
    # ponytail: string rules only - "DPR" vs "dense passage retriever" stay apart.
    # Alias merging (acronym expansion, embedding clustering) when fragmentation shows up.
    key = " ".join(re.sub(r"[-_/]", " ", name).lower().split()).strip(" .,;:()[]\"'")
    return _VARIANT.sub("", key).strip()


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
