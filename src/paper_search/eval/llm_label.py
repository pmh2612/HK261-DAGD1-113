"""Ask a local LLM a multiple-choice question and read its answer probabilities.

One forward pass per prompt: the next-token distribution over the choices (each a single
token, e.g. "0"/"1"/"2" or "Yes"/"No") is the answer, so nothing is generated or parsed.
Loaded in-process (DECIDE-7). Labels data in place of people: relevance (BT6-BT8, see
scripts/llm_judge.py) and entity-name pairs (BT4, scripts/bt4_normalize.py).
"""

from collections.abc import Callable

Choose = Callable[[list[str], list[str]], list[list[float]]]


def chooser(model_id: str) -> Choose:
    """Load the model once; return choose(user prompts, choices) -> probabilities each."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from paper_search.extraction.ner import device

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load on CPU, then move: device_map="mps" segfaults while loading (see ner.llm).
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = model.to(device()).eval()

    def token(choice: str) -> int:
        ids = tokenizer.encode(choice, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"choice {choice!r} is {len(ids)} tokens, not one")
        return ids[0]

    def choose(prompts: list[str], choices: list[str]) -> list[list[float]]:
        ids = [token(c) for c in choices]
        out = []
        # One prompt at a time: batching is no faster on MPS, and left padding flipped a
        # borderline relevance grade (2026-11-25).
        for user in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3-style templates; ignored by others
                tokenize=False,
            )
            enc = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                row = model(**enc).logits[0, -1, ids].float().softmax(-1).tolist()
            out.append([round(x, 4) for x in row])
        if model.device.type == "mps":
            # The MPS cache grows with every new input length; on 16 GB it reached 11 GB
            # and swapped to a crawl (2026-11-25).
            torch.mps.empty_cache()
        return out

    return choose
