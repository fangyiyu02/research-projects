"""
run_drift.py — thin wrapper: attaches DriftRecorder, then calls the OFFICIAL
LLaDA generate() for decoding. No generation logic lives in this file.

The official generate.py is imported from the locked LLaDA repo
(LLADA_REPO env var, default /workspace/repos/LLaDA) and never modified.

  python run_drift.py --model /path/to/ultrallada \
      --prompt-file /workspace/niah_single_2.pt --sample-idx 0 \
      --gen-len 512 --heatmap-stride 8 \
      --out /workspace/runs/results/niah_single_2_4k_s0.npz

ADAPT:
  * blocks path (model.model.transformer.blocks)
  * logits handling for 32K+: your chunked lm_head lives inside the model /
    generate path you already use — nothing to change here
  * MASK_ID if your tokenizer differs
"""
import argparse, json, os, sys, time
import numpy as np
import torch

from drift_recorder import DriftRecorder

MASK_ID = 126336  # ADAPT

LLADA_REPO = os.environ.get("LLADA_REPO", "/workspace/repos/LLaDA")
sys.path.insert(0, LLADA_REPO)
try:
    from generate import generate            # official generate.py, called as-is
except ImportError as e:
    raise SystemExit(
        f"could not import generate from {LLADA_REPO} -- "
        "set LLADA_REPO to the locked LLaDA repo path") from e


@torch.no_grad()
def run(a):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(a.device).eval()
    blocks = model.model.transformer.blocks          # ADAPT

    if a.prompt_file.endswith(".pt"):
        obj = torch.load(a.prompt_file, weights_only=True, map_location="cpu")
        ids = (obj[a.sample_idx] if isinstance(obj, (list, tuple)) else obj)
        ids = ids.flatten().long()
    else:
        text = open(a.prompt_file, encoding="utf-8").read()
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(formatted, add_special_tokens=False,
                  return_tensors="pt").input_ids[0]
    if a.prompt_len:
        ids = ids[: a.prompt_len]
    prompt = ids.unsqueeze(0).to(a.device)
    P, G = len(ids), a.gen_len
    steps = a.steps or G

    rec = DriftRecorder(blocks, heatmap_token_stride=a.heatmap_stride,
                        buffers_on_gpu=(a.buffers == "gpu"))
    rec.attach()
    cuda = a.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        model_gb = torch.cuda.memory_allocated() / 2**30

    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = generate(model, prompt, steps=steps, gen_length=G,
                   block_length=a.block_length, temperature=a.temperature,
                   cfg_scale=0.0, remasking="low_confidence", mask_id=MASK_ID)
    if cuda:
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rec.save(a.out)
    rec.detach()
    n_steps = len(rec.logs["step_time"])
    meta = dict(vars(a), n_steps=n_steps, P=P, G=G, steps=steps,
                total_s=round(total_s, 1),
                mean_step_s=round(total_s / max(n_steps, 1), 3),
                model_gb=round(model_gb, 2) if cuda else None,
                peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if cuda else None)
    json.dump(meta, open(a.out.replace(".npz", ".json"), "w"), indent=2)
    print(f"done: {n_steps} steps, {meta['total_s']}s, "
          f"{meta['mean_step_s']}s/step -> {a.out}")
    print(tok.decode(out[0, P:][:80].cpu()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True, help=".pt (RULER dump) or .txt")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--prompt-len", type=int, default=0, help="0 = as-is")
    p.add_argument("--gen-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=0, help="0 = gen-len (official 1 tok/step)")
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--heatmap-stride", type=int, default=0)
    p.add_argument("--buffers", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None,
                   help="default: workspace/runs/results/<prompt>_s<idx>.npz")
    a = p.parse_args()
    if a.out is None:
        stem = os.path.splitext(os.path.basename(a.prompt_file))[0]
        a.out = os.path.join("/workspace", "runs", "results",
                             f"{stem}_s{a.sample_idx}.npz")
    run(a)


if __name__ == "__main__":
    main()
