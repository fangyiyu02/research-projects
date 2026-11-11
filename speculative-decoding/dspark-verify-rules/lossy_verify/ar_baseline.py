"""Target-only greedy decoding, same prompts / stop tokens / max length / dtype / attention as run.py.
Gives the 1.00x reference for speedup. No drafter is loaded.

    cd /workspace && python -m lossy_verify.ar_baseline --n 200 --out runs/ar

Outputs in --out: config.json, summary.json, samples.jsonl (same fields as run.py where they apply).
The decoding loop mirrors DeepSpec generate_decoding_sample: one prefill forward, then one
forward per token with a DynamicCache, argmax at T=0.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from . import DEEPSPEC_DIR  # noqa: F401  (puts DeepSpec on sys.path)
from deepspec.data.parser import encode_chat_messages
from deepspec.eval.base_evaluator import resolve_stop_token_ids, trim_output_ids
from deepspec.utils import seed_all

from .metrics import grade, load_gsm8k_with_answers

DEFAULT_SEED = 980406


@torch.inference_mode()
def generate_ar(model, input_ids, max_new_tokens, stop_token_ids):
    device = input_ids.device
    n_in = input_ids.shape[1]
    max_length = n_in + max_new_tokens
    output_ids = torch.empty((1, max_length), dtype=torch.long, device=device)
    output_ids[:, :n_in] = input_ids
    position_ids = torch.arange(max_length, device=device).unsqueeze(0)
    cache = DynamicCache()
    stop = torch.tensor(stop_token_ids or [], device=device)

    out = model(input_ids=input_ids, position_ids=position_ids[:, :n_in],
                past_key_values=cache, use_cache=True, logits_to_keep=1)
    cur = n_in
    tok = out.logits[:, -1, :].argmax(dim=-1)
    forwards = 1
    while True:
        output_ids[:, cur] = tok
        cur += 1
        if stop.numel() and bool(torch.isin(tok, stop).item()):
            break
        if cur >= max_length:
            break
        out = model(input_ids=tok.unsqueeze(0), position_ids=position_ids[:, cur - 1:cur],
                    past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1, :].argmax(dim=-1)
        forwards += 1
    output_ids = trim_output_ids(output_ids[:, :cur], n_in, stop_token_ids)
    return output_ids, forwards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", 0)
    model = AutoModelForCausalLM.from_pretrained(
        a.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    tok = AutoTokenizer.from_pretrained(a.target)
    stop_token_ids = resolve_stop_token_ids(model, tok)

    # same sample selection as BaseEvaluator.run_dataset / LossyDSparkEvaluator.run_dataset
    seed_all(a.seed)
    rows = load_gsm8k_with_answers()
    if len(rows) > a.n:
        rng = random.Random(a.seed)
        rows = list(rows)
        rng.shuffle(rows)
        rows = rows[:a.n]

    config = {"rule": "target_only", "n": a.n, "max_new_tokens": a.max_new_tokens, "temperature": 0.0,
              "seed": a.seed, "target": a.target, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    records = []
    for idx, inst in enumerate(rows):
        seed_all(a.seed + idx)
        input_ids = encode_chat_messages(tok, [{"role": "user", "content": inst["turns"][0]}],
                                         add_generation_prompt=True, enable_thinking=False).to(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        ids, forwards = generate_ar(model, input_ids, a.max_new_tokens, stop_token_ids)
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - t0
        gen = ids[0, input_ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        g = grade(text, inst["answer"])
        n_out = int(gen.numel())
        hit = n_out >= a.max_new_tokens and not bool(torch.isin(gen, torch.tensor(stop_token_ids, device=device)).any())
        records.append({"id": inst["id"], "dataset_idx": idx, "gold": inst["answer"], "text": text, **g,
                        "num_output_tokens": n_out, "hit_max_new_tokens": bool(hit),
                        "target_forwards": forwards, "wall_s": wall,
                        "output_ids": [int(x) for x in gen.tolist()]})
        if (idx + 1) % 20 == 0:
            acc = sum(r["correct_fallback"] for r in records) / len(records)
            tps = sum(r["num_output_tokens"] for r in records) / sum(r["wall_s"] for r in records)
            print(f"[target_only] {idx + 1}/{len(rows)} acc_fallback={acc:.3f} tok/s={tps:.1f}", flush=True)

    with (out / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(records)
    tot_out = sum(r["num_output_tokens"] for r in records)
    tot_wall = sum(r["wall_s"] for r in records)
    summary = {"rule": "target_only", "n": n,
               "accuracy_strict": sum(r["correct_strict"] for r in records) / n,
               "accuracy_fallback": sum(r["correct_fallback"] for r in records) / n,
               "hit_max_new_tokens_count": sum(r["hit_max_new_tokens"] for r in records),
               "extract_fallback_count": sum(r["extract_fallback"] for r in records),
               "mean_output_tokens": tot_out / n,
               "target_forwards_per_output_token": sum(r["target_forwards"] for r in records) / tot_out,
               "mean_wall_s": tot_wall / n, "tokens_per_s": tot_out / tot_wall,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
