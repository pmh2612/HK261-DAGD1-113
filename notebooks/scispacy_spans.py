"""SciSpaCy entity spans for BT2, run in its own venv (scispacy needs numpy < 2).

Reads DIR/docs.jsonl ({"doc_id", "text"}, written by scripts/run_ner.py) and writes
DIR/scispacy_spans.jsonl ({"doc_id", "spans": [[start, end], ...]}). The spans are
untyped - en_core_sci_sm only marks "ENTITY" - and run_ner.py assigns the types. Like
encode_specter2.py, it imports nothing from paper_search, which is not installed there.

Setup (once):
    python3.11 -m venv .venv-scispacy
    .venv-scispacy/bin/pip install scispacy==0.6.2 click \
        https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
Run:  .venv-scispacy/bin/python notebooks/scispacy_spans.py --dir data/processed/ner/scierc
"""

import argparse
import json
import time
from pathlib import Path

import spacy


def main() -> None:
    """Write one line of character spans per document."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    io_dir = parser.parse_args().dir
    docs = [json.loads(line) for line in (io_dir / "docs.jsonl").open(encoding="utf-8")]
    nlp = spacy.load("en_core_sci_sm")
    start = time.perf_counter()
    with (io_dir / "scispacy_spans.jsonl").open("w", encoding="utf-8") as f:
        for d, parsed in zip(docs, nlp.pipe(d["text"] for d in docs), strict=True):
            spans = [[e.start_char, e.end_char] for e in parsed.ents]
            f.write(json.dumps({"doc_id": d["doc_id"], "spans": spans}) + "\n")
    per_doc = (time.perf_counter() - start) / len(docs)
    print(f"wrote SciSpaCy spans for {len(docs)} documents to {io_dir} ({per_doc:.3f} s/abstract)")


if __name__ == "__main__":
    main()
