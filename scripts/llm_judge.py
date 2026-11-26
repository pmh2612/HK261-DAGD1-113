"""Grade (query, paper) pairs 0/1/2 with a local LLM, in place of hand-made qrels.

Pool: for each query, the top `--depth` papers of every run.json under `--runs`
(written by run_hybrid.py / run_search.py). Grade: one forward pass per pair; the model's
next-token probabilities over "0", "1", "2" give the grade, so nothing is generated or
parsed. Every grade is appended to the cache as it is made, so a stopped run resumes.

Where the dataset has its own qrels (SciFact), the script then checks the judge against
them: pair agreement (Cohen's kappa), and whether the systems keep their order when
scored with the LLM's grades instead (Kendall tau). Trust the judge on our corpus only
if that check holds.

Run: .venv/bin/python scripts/llm_judge.py --dataset scifact \
         --runs experiments/runs/bt8/<stamp>_scifact_* --queries 100
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score

from paper_search.config import PROJECT_ROOT, settings
from paper_search.eval.metrics import evaluate
from run_search import jsonl, load  # scripts/ is on sys.path when this file runs

#: What "relevant" means. SciFact's qrels mark abstracts that support or refute a claim,
#: so the check uses that definition; our corpus asks for topical relevance.
GRADES = {
    "scifact": (
        "The query is a scientific claim.",
        "2 = the abstract directly provides evidence that supports or contradicts the claim\n"
        "1 = the abstract is on the same topic but does not directly address the claim\n"
        "0 = the abstract is unrelated to the claim",
    ),
    "default": (
        "The query is a literature search by a computer science researcher.",
        "2 = the paper is directly about what the query asks for\n"
        "1 = the paper is related and partly useful, but not directly about it\n"
        "0 = the paper is not relevant",
    ),
}

PROMPT = """{intro} Judge how relevant the paper is to the query.

Query: {query}

Paper title: {title}
Abstract: {abstract}

{scale}

Answer with a single digit: 0, 1 or 2."""


def pool(runs: list[dict], qids: list[str], depth: int) -> list[tuple[str, str]]:
    """Unique (qid, paper_id) pairs from every run's top `depth`, in a stable order."""
    pairs = dict.fromkeys((q, p) for q in qids for run in runs for p in run[q][:depth])
    return list(pairs)


def judge(model_id: str, dataset: str, queries: dict, docs: dict, batch: int):
    """A function grading a list of pairs: [(grade, [p0, p1, p2]), ...]."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from paper_search.extraction.ner import device

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    # Load on CPU, then move: device_map="mps" segfaults while loading (see ner.llm).
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = model.to(device()).eval()
    digits = [tokenizer.convert_tokens_to_ids(d) for d in "012"]
    intro, scale = GRADES.get(dataset, GRADES["default"])

    def text(qid: str, pid: str) -> str:
        d = docs[pid]
        user = PROMPT.format(
            intro=intro, query=queries[qid], title=d["title"], abstract=d["abstract"], scale=scale
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3-style templates; ignored by others
            tokenize=False,
        )

    def grade(pairs: list[tuple[str, str]]) -> list[tuple[int, list[float]]]:
        out = []
        for i in range(0, len(pairs), batch):
            enc = tokenizer(
                [text(q, p) for q, p in pairs[i : i + batch]], return_tensors="pt", padding=True
            ).to(model.device)
            # Left padding: positions must count real tokens only, or a padded pair is
            # graded differently from the same pair alone.
            positions = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                logits = model(**enc, position_ids=positions).logits[:, -1, digits].float()
            for row in logits.softmax(-1).tolist():
                out.append((max(range(3), key=row.__getitem__), [round(x, 4) for x in row]))
        if model.device.type == "mps":
            # The MPS cache grows with every new input length; on 16 GB it reached 11 GB
            # and swapped to a crawl (2026-11-25). Freeing it costs nothing measurable.
            torch.mps.empty_cache()
        return out

    return grade


def check(grades: dict, gold: dict, runs: dict[str, dict], qids: list[str]) -> dict:
    """The judge against the dataset's own qrels, pair by pair and system by system."""
    pairs = list(grades)
    truth = [int(gold.get(q, {}).get(p, 0) > 0) for q, p in pairs]
    out = {"pairs": len(pairs), "gold_relevant": sum(truth)}
    for t in (1, 2):
        pred = [int(grades[k] >= t) for k in pairs]
        hit = sum(a and b for a, b in zip(pred, truth, strict=True))
        out[f"grade>={t}"] = {
            "kappa": round(cohen_kappa_score(truth, pred), 3),
            "precision": round(hit / max(sum(pred), 1), 3),
            "recall": round(hit / max(sum(truth), 1), 3),
        }
    llm_qrels: dict[str, dict[str, int]] = {}
    for (q, p), g in grades.items():
        llm_qrels.setdefault(q, {})[p] = g
    gold_q = {q: gold[q] for q in qids}
    systems = {}
    for name, run in runs.items():
        sub = {q: run[q] for q in qids}
        systems[name] = {
            "gold_ndcg@10": evaluate(sub, gold_q)["ndcg@10"],
            "llm_ndcg@10": evaluate(sub, llm_qrels).get("ndcg@10"),
        }
    a = [s["gold_ndcg@10"] for s in systems.values()]
    b = [s["llm_ndcg@10"] for s in systems.values()]
    tau = kendalltau(a, b)
    out["kendall_tau"] = round(float(tau.statistic), 3)
    out["kendall_p"] = round(float(tau.pvalue), 4)
    names = list(systems)
    out["best_by_gold"] = names[a.index(max(a))]
    out["best_by_llm"] = names[b.index(max(b))]
    out["systems"] = systems
    return out


def main() -> None:
    """Pool, grade (resuming from the cache), and check against qrels where they exist."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run dirs")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--queries", type=int, help="sample this many queries (seed 42)")
    # >1 is no faster on MPS and padding flipped a borderline grade (2026-11-25): keep 1.
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    docs, queries, gold, _ = load(args.dataset)
    if args.dataset != "scifact":  # on our corpus load() returns this script's own qrels
        gold = {}
    docs = {d["paper_id"]: d for d in docs}
    runs = {
        d.name.split(f"_{args.dataset}_", 1)[-1]: json.loads((d / "run.json").read_text())
        for d in args.runs
    }
    qids = sorted(queries)
    if args.queries:
        qids = sorted(random.Random(42).sample(qids, args.queries))
    pairs = pool(list(runs.values()), qids, args.depth)

    model_tag = args.model.replace("/", "__")
    cache = settings.processed_dir / "judge" / f"{args.dataset}_{model_tag}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["qid"], r["paper_id"]): r["grade"] for r in jsonl(cache)} if cache.exists() else {}
    todo = [k for k in pairs if k not in done]
    print(f"{len(qids)} queries, {len(pairs)} pairs, {len(todo)} to grade", flush=True)

    start = time.perf_counter()
    if todo:
        grade = judge(args.model, args.dataset, queries, docs, args.batch)
        with cache.open("a", encoding="utf-8") as f:
            for i in range(0, len(todo), 50):
                chunk = todo[i : i + 50]
                for (q, p), (g, probs) in zip(chunk, grade(chunk), strict=True):
                    done[(q, p)] = g
                    f.write(json.dumps({"qid": q, "paper_id": p, "grade": g, "p": probs}) + "\n")
                f.flush()
                print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
    seconds = time.perf_counter() - start

    grades = {k: done[k] for k in pairs}
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "depth": args.depth,
        "queries": len(qids),
        "pairs": len(pairs),
        "grade_counts": {g: list(grades.values()).count(g) for g in range(3)},
        "seconds_per_pair": round(seconds / max(len(todo), 1), 3) if todo else None,
    }
    if gold:
        result["check"] = check(grades, gold, runs, qids)
    else:  # our corpus: the grades become the qrels run_search.py / run_hybrid.py read
        path = PROJECT_ROOT / "data/annotations/retrieval_qrels.jsonl"
        if path.exists() and any("source" not in r for r in jsonl(path)):
            raise SystemExit(f"{path} holds hand labels; not overwriting them")
        # Every cached grade of these queries, not just this pool: a later call with
        # other runs or another depth adds to the qrels instead of replacing them.
        # Relevant = grade 2 only. On SciFact grade >= 2 agreed with people (kappa 0.45),
        # grade >= 1 did not (0.18): the 4B judge hands out 1 too freely (2026-11-27).
        with path.open("w", encoding="utf-8") as f:
            for (q, p), g in sorted((k, g) for k, g in done.items() if k[0] in queries):
                row = {"qid": q, "paper_id": p, "rel": int(g == 2), "grade": g}
                row["source"] = args.model
                f.write(json.dumps(row) + "\n")
        result["qrels"] = str(path.relative_to(PROJECT_ROOT))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = PROJECT_ROOT / "experiments/runs/judge" / f"{stamp}_{args.dataset}"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
