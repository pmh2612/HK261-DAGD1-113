"""Entity extraction experiment: rules vs GLiNER vs an in-process LLM.

Runs all three over the abstracts in data/eval/entities.jsonl, writes each system's
predictions to data/processed/extraction/<system>.jsonl for error analysis, and scores
them against the papers a human has annotated (`entities` not null).

Scoring: an entity is the pair (type, normalized text); precision / recall / F1 are
micro-averaged over annotated papers, per type and overall. Exact match after
normalization - "dense passage retriever" vs "DPR" counts as a miss, which is the
fragmentation FR-10 has to solve later anyway.

LLM: settings.llm_model, else Qwen2.5-3B-Instruct - chosen for this experiment only,
DECIDE-7 is still open. Loaded with transformers in-process (NFR-8).

Run: .venv/bin/python notebooks/exp_extraction.py
"""

import gc
import json
import re
import sys

import torch
from gliner import GLiNER
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from paper_search.config import PROJECT_ROOT, settings

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "extraction"
TYPES = ("Method", "Dataset", "Task")
LLM_MODEL = settings.llm_model or "Qwen/Qwen2.5-3B-Instruct"
GLINER_MODEL = "urchade/gliner_medium-v2.1"

# --- rules -------------------------------------------------------------------------------

DATASET_RE = re.compile(
    r"\b([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,3})\s+(?:dataset|benchmark|corpus)\b"
    r"|\b([A-Z][\w-]*QA)\b"  # HotpotQA, TriviaQA, PopQA ...
)
# Tokens with two or more capitals (ColBERT, Self-RAG, GPT-4) are usually system names.
METHOD_RE = re.compile(r"\b((?:[a-z]*[A-Z][\w]*-?)?[A-Z][a-z]*[A-Z][\w-]*)\b")
METHOD_STOP = {"LLM", "LLMs", "NLP", "AI", "QA", "SOTA", "API", "APIs", "GPU", "GPUs"}
TASK_RE = re.compile(
    r"\b((?:open-domain |multi-hop |long-form |extractive |abstractive )?"
    r"(?:question answering|fact verification|fact checking|summarization|dialogue generation"
    r"|text generation|information retrieval|passage retrieval|code generation"
    r"|machine translation|entity linking|slot filling))\b",
    re.IGNORECASE,
)


def rules(abstract: str) -> list[dict]:
    """Regex baseline: capitalized names before 'dataset', *QA names, CamelCase, task phrases."""
    datasets = {a or b for a, b in DATASET_RE.findall(abstract)}
    methods = {m for m in METHOD_RE.findall(abstract) if m not in METHOD_STOP} - datasets
    tasks = set(TASK_RE.findall(abstract))
    return [
        {"text": t, "type": typ}
        for typ, found in (("Method", methods), ("Dataset", datasets), ("Task", tasks))
        for t in sorted(found)
    ]


# --- GLiNER ------------------------------------------------------------------------------


def gliner_extractor():
    """Zero-shot NER with the three entity types as free-text labels."""
    model = GLiNER.from_pretrained(GLINER_MODEL)
    labels = [t.lower() for t in TYPES]

    def extract(abstract: str) -> list[dict]:
        ents = model.predict_entities(abstract, labels, threshold=0.5)
        return [{"text": e["text"], "type": e["label"].capitalize()} for e in ents]

    return extract


# --- LLM ---------------------------------------------------------------------------------

PROMPT = """Extract the named entities from this scientific abstract.

- Method: a named model, algorithm, system or technique (e.g. BERT, DPR, Self-RAG, BM25).
- Dataset: a named dataset or benchmark (e.g. Natural Questions, HotpotQA, MS MARCO).
- Task: a research task (e.g. open-domain question answering, fact verification).

Copy each entity exactly as written in the abstract. Do not invent entities.
Answer with JSON only: {{"Method": [...], "Dataset": [...], "Task": [...]}}

Abstract:
{abstract}"""


def llm_extractor():
    """Greedy-decoded JSON from an instruction-tuned model, parse failures counted."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    # Not device_map="auto": on Apple silicon accelerate misjudges unified memory and
    # offloads layers to disk, which makes generation unusably slow.
    device = settings.llm_device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, dtype=torch.float16, device_map=device
    ).eval()
    failures = []

    def extract(abstract: str) -> list[dict]:
        messages = [{"role": "user", "content": PROMPT.format(abstract=abstract)}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        text = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        # ponytail: parse-and-hope; DECIDE-7 wants schema-constrained decoding for FR-9.
        # The failure count printed below is the number that justifies adding it.
        try:
            parsed = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group())
            return [{"text": str(t), "type": typ} for typ in TYPES for t in parsed.get(typ) or []]
        except (AttributeError, ValueError):
            failures.append(text)
            return []

    extract.failures = failures
    return extract


# --- scoring -----------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Case, whitespace and surrounding punctuation do not make a different entity."""
    return " ".join(text.lower().split()).strip(" .,;:()[]\"'")


def as_set(entities: list[dict]) -> set[tuple[str, str]]:
    """Entities as comparable (type, normalized text) pairs."""
    return {(e["type"], normalize(e["text"])) for e in entities if e["type"] in TYPES}


def prf(pred: set, gold: set) -> tuple[float, float, float]:
    """Precision, recall, F1 of a predicted set against a gold set."""
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def _check() -> None:
    """Hand-computed cases for the parts a table cannot show are wrong."""
    assert normalize("  BERT-base. ") == "bert-base"
    assert prf({1, 2}, {2, 3}) == (0.5, 0.5, 0.5) and prf(set(), {1}) == (0.0, 0.0, 0.0)
    found = as_set(
        rules(
            "We propose Self-RAG and evaluate on the PopQA dataset for "
            "open-domain question answering."
        )
    )
    assert ("Method", "self-rag") in found and ("Dataset", "popqa") in found
    assert ("Task", "open-domain question answering") in found


def main() -> None:
    """Run every extractor over the sample, save predictions, score annotated papers."""
    _check()
    papers = [json.loads(line) for line in (EVAL_DIR / "entities.jsonl").open()]
    annotated = [p for p in papers if p["entities"] is not None]
    # Factories, loaded one at a time and dropped after use: GLiNER and a 3B model side by
    # side do not fit in 16 GB of unified memory next to anything else.
    systems = {"rules": lambda: rules, "gliner": gliner_extractor, "llm": llm_extractor}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = {}
    for name, factory in systems.items():
        extract = factory()
        preds[name] = {p["paper_id"]: extract(p["abstract"]) for p in tqdm(papers, desc=name)}
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for pid, ents in preds[name].items():
                f.write(json.dumps({"paper_id": pid, "entities": ents}, ensure_ascii=False) + "\n")
        print(f"{name}: {sum(map(len, preds[name].values()))} entities on {len(papers)} papers")
        if name == "llm":
            print(f"llm: {len(extract.failures)} of {len(papers)} outputs were not valid JSON")
        del extract
        gc.collect()

    if not annotated:
        sys.exit(f"\nno annotated papers yet - fill in `entities` in {EVAL_DIR / 'entities.jsonl'}")

    print(f"\n{len(annotated)} annotated papers, micro P / R / F1\n")
    print(f"{'system':<8} " + " ".join(f"{t:>20}" for t in (*TYPES, "All")))
    for name in systems:
        cells = []
        for typ in (*TYPES, None):
            pred, gold = set(), set()
            for p in annotated:
                pred |= {
                    (p["paper_id"], *e)
                    for e in as_set(preds[name][p["paper_id"]])
                    if typ in (None, e[0])
                }
                gold |= {(p["paper_id"], *e) for e in as_set(p["entities"]) if typ in (None, e[0])}
            cells.append("{:.2f}/{:.2f}/{:.2f}".format(*prf(pred, gold)))
        print(f"{name:<8} " + " ".join(f"{c:>20}" for c in cells))


if __name__ == "__main__":
    main()
