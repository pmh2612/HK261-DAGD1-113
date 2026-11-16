"""Encode the eval corpus and queries with SPECTER2, for notebooks/exp_search.py.

SPECTER2 needs the `adapters` library, which pins transformers < 5, so this runs in its
own venv and hands the vectors over as files. It deliberately imports nothing from
paper_search, which is not installed there.

Documents use the `specter2` (proximity) adapter on "title [SEP] abstract"; queries use
the `specter2_adhoc_query` adapter, as the model card prescribes for search.

Setup (once):  python3.11 -m venv .venv-specter2 && .venv-specter2/bin/pip install adapters torch
Run:           .venv-specter2/bin/python notebooks/encode_specter2.py [--dir DIR]
With --dir, reads DIR/docs.jsonl and DIR/queries.jsonl (written by scripts/run_search.py)
and writes the vectors into DIR. Without it, encodes data/eval/ for exp_search.py.
Re-run whenever the documents or queries change - both consumers check.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
OUT_DIR = ROOT / "data" / "processed" / "specter2"


def encode(model, tokenizer, adapter: str, texts: list[str]) -> np.ndarray:
    """CLS embeddings under the given adapter, L2-normalized."""
    model.set_active_adapters(adapter)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = tokenizer(
                texts[i : i + 16],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            out.append(model(**batch).last_hidden_state[:, 0, :].numpy())
    vecs = np.concatenate(out)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def main() -> None:
    """Write docs.npy / queries.npy plus the id order each row corresponds to."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path)
    io_dir = parser.parse_args().dir
    docs_path = io_dir / "docs.jsonl" if io_dir else EVAL_DIR / "abstracts.jsonl"
    queries_path = io_dir / "queries.jsonl" if io_dir else EVAL_DIR / "queries.jsonl"
    out_dir = io_dir or OUT_DIR
    docs = [json.loads(line) for line in docs_path.open()]
    queries = [json.loads(line) for line in queries_path.open()]

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base").eval()
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2")
    model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query")

    doc_texts = [d["title"] + tokenizer.sep_token + d["abstract"] for d in docs]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "docs.npy", encode(model, tokenizer, "specter2", doc_texts))
    np.save(
        out_dir / "queries.npy",
        encode(model, tokenizer, "adhoc_query", [q["query"] for q in queries]),
    )
    (out_dir / "ids.json").write_text(
        json.dumps(
            {"docs": [d["paper_id"] for d in docs], "queries": [q["query"] for q in queries]}
        )
    )
    print(f"encoded {len(docs)} docs and {len(queries)} queries into {out_dir}")


if __name__ == "__main__":
    main()
