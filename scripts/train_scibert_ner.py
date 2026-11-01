"""BT2 method 4: fine-tune SciBERT for NER on SciERC train, keep the best epoch on dev.

The SciERC-trained models on Hugging Face (anlp/SciERC_trained_Scibert and others,
checked 2026-11-01) have no model card, so nothing says they never saw the test split.
Training our own on train only, choosing the epoch on dev, keeps the test score honest.

Writes the model to data/processed/models/scibert_scierc/ with training.json (per-epoch
dev F1 and the settings used).

Run: .venv/bin/python scripts/train_scibert_ner.py
"""

import json
import random
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from paper_search.config import settings
from paper_search.eval.ner import load_scierc, score
from paper_search.extraction.ner import BIO_LABELS, bio_labels, device, tag_docs

BASE = "allenai/scibert_scivocab_uncased"
SETTINGS = {"epochs": 8, "lr": 5e-5, "batch_size": 16, "warmup": 0.1, "seed": 42}


def encode(tokenizer, docs: list[dict]) -> list[dict]:
    """One example per sentence; only a word's first sub-token carries its label."""
    label_id = {label: i for i, label in enumerate(BIO_LABELS)}
    examples = []
    for d in docs:
        for i, j in d["sentences"]:
            enc = tokenizer(d["words"][i:j], is_split_into_words=True, truncation=True)
            tags, labels, seen = bio_labels(d, i, j), [], set()
            for w in enc.word_ids():
                labels.append(-100 if w is None or w in seen else label_id[tags[w]])
                if w is not None:
                    seen.add(w)
            examples.append(
                {"input_ids": torch.tensor(enc["input_ids"]), "labels": torch.tensor(labels)}
            )
    return examples


def batches(examples: list[dict], size: int, rng: random.Random):
    """Shuffled, padded batches."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    for k in range(0, len(order), size):
        chunk = [examples[i] for i in order[k : k + size]]
        ids = pad_sequence([e["input_ids"] for e in chunk], batch_first=True)
        yield {
            "input_ids": ids,
            "attention_mask": (ids != 0).long(),
            "labels": pad_sequence(
                [e["labels"] for e in chunk], batch_first=True, padding_value=-100
            ),
        }


def main() -> None:
    """Train, score dev after every epoch, save the best epoch."""
    torch.manual_seed(SETTINGS["seed"])
    rng = random.Random(SETTINGS["seed"])
    data = settings.raw_dir / "scierc"
    train, dev = load_scierc(data / "train.json"), load_scierc(data / "dev.json")
    out_dir = settings.processed_dir / "models" / "scibert_scierc"

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE,
        num_labels=len(BIO_LABELS),
        id2label=dict(enumerate(BIO_LABELS)),
        label2id={label: i for i, label in enumerate(BIO_LABELS)},
    ).to(device())
    examples = encode(tokenizer, train)
    steps = SETTINGS["epochs"] * -(-len(examples) // SETTINGS["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["lr"], weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(SETTINGS["warmup"] * steps), steps)
    print(f"{len(examples)} training sentences, {steps} steps on {model.device}")

    history, best = [], -1.0
    for epoch in range(1, SETTINGS["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        for batch in batches(examples, SETTINGS["batch_size"], rng):
            loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
            loss.backward()
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        model.eval()
        gold = {d["doc_id"]: d["entities"] for d in dev}
        f1 = score(gold, tag_docs(model, tokenizer, dev))["all"]["exact"]["f1"]
        history.append(
            {"epoch": epoch, "dev_exact_f1": f1, "seconds": round(time.perf_counter() - start)}
        )
        print(history[-1], flush=True)
        if f1 > best:
            best = f1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    best_epoch = max(history, key=lambda h: h["dev_exact_f1"])["epoch"]
    (out_dir / "training.json").write_text(
        json.dumps(
            {"base": BASE, **SETTINGS, "best_epoch": best_epoch, "history": history}, indent=2
        )
    )
    print(f"best epoch {best_epoch}: dev exact F1 {best:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
