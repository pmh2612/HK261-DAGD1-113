"""BT2: run the entity extractors in a config over SciERC test and score them.

SciERC train builds the dictionary and the SciSpaCy type mapping and supplies the
few-shot examples; SciBERT is trained on it separately (scripts/train_scibert_ner.py).
Every number is on the 100 test abstracts, which no system has seen.

SciSpaCy needs its own venv. The first run writes its input and skips it; then run
    .venv-scispacy/bin/python notebooks/scispacy_spans.py --dir data/processed/ner/scierc
and run this again. Systems are loaded one at a time and released, so the 3B LLM fits
next to the rest; --only runs a subset (the LLM arms take the longest).

Run: .venv/bin/python scripts/run_ner.py --config configs/bt2_ner.yaml [--only gazetteer ...]
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import yaml

from paper_search.config import PROJECT_ROOT, settings
from paper_search.eval.ner import TYPES, load_scierc, score
from paper_search.extraction import ner


def build(system: dict, train: list[dict], io_dir: Path) -> ner.Predict | None:
    """The predictor for one config entry; None when its inputs are not there yet."""
    kind = system["kind"]
    if kind == "gazetteer":
        return ner.gazetteer(train)
    if kind == "spacy":
        return ner.spacy_general(system["model"], system["label_map"])
    if kind == "scispacy":
        path = io_dir / "scispacy_spans.jsonl"
        if not path.exists():
            return None
        rows = [json.loads(line) for line in path.open()]
        return ner.head_typed(train, {r["doc_id"]: r["spans"] for r in rows})
    if kind == "gliner":
        return ner.gliner(system["model"], system["labels"], system["threshold"])
    if kind == "scibert":
        model_dir = PROJECT_ROOT / system["model_dir"]
        return ner.scibert(model_dir) if model_dir.exists() else None
    if kind == "llm":
        cache = settings.processed_dir / "ner" / "llm_cache"
        return ner.llm(system["model"], train, system["shots"], cache)
    raise ValueError(f"unknown kind {kind}")


def main() -> None:
    """Predict, score, write one run directory per system, print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    data = settings.raw_dir / "scierc"
    train, test = load_scierc(data / "train.json"), load_scierc(data / "test.json")
    gold = {d["doc_id"]: d["entities"] for d in test}
    io_dir = settings.processed_dir / "ner" / "scierc"
    io_dir.mkdir(parents=True, exist_ok=True)
    (io_dir / "docs.jsonl").write_text(
        "".join(json.dumps({"doc_id": d["doc_id"], "text": d["text"]}) + "\n" for d in test)
    )
    print(f"SciERC test: {len(test)} abstracts, {score(gold, gold)['all']['gold']} entities")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    for system in cfg["systems"]:
        name = system["name"]
        if args.only and name not in args.only:
            continue
        start = time.perf_counter()
        predict = build(system, train, io_dir)
        if predict is None:
            print(f"{name}: skipped - its input is missing (see the docstring)")
            continue
        load_s = time.perf_counter() - start
        start = time.perf_counter()
        pred = predict(test)
        per_doc = (time.perf_counter() - start) / len(test)
        metrics = {
            **score(gold, pred),
            "seconds_per_abstract": round(per_doc, 3),
            "load_seconds": round(load_s, 1),
            **({"llm": predict.stats} if hasattr(predict, "stats") else {}),
        }
        out = PROJECT_ROOT / "experiments/runs" / cfg["task"] / f"{stamp}_scierc_{name}"
        out.mkdir(parents=True)
        (out / "config.yaml").write_text(yaml.safe_dump({"dataset": "scierc", **system}))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        texts = {d["doc_id"]: d["text"] for d in test}
        (out / "predictions.jsonl").write_text(
            "".join(
                json.dumps(
                    {"doc_id": k, "entities": [[s, e, t, texts[k][s:e]] for s, e, t in v]},
                    ensure_ascii=False,
                )
                + "\n"
                for k, v in pred.items()
            )
        )
        rows.append((name, metrics))
        print(f"{name}: exact F1 {metrics['all']['exact']['f1']:.3f} ({per_doc:.2f} s/abstract)")
        del predict
        gc.collect()

    print(
        f"\n{'system':<16} {'P':>6} {'R':>6} {'F1':>6} {'partF1':>7} "
        + " ".join(f"{t:>8}" for t in TYPES)
        + "  s/abs"
    )
    for name, m in sorted(rows, key=lambda r: -r[1]["all"]["exact"]["f1"]):
        e = m["all"]["exact"]
        print(
            f"{name:<16} {e['p']:>6.3f} {e['r']:>6.3f} {e['f1']:>6.3f} "
            f"{m['all']['partial']['f1']:>7.3f} "
            + " ".join(f"{m[t]['exact']['f1']:>8.3f}" for t in TYPES)
            + f"  {m['seconds_per_abstract']:.2f}"
        )


if __name__ == "__main__":
    main()
