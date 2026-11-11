# run_quant_exp.py — cd /workspace/dflash 下运行
import argparse
import dflash.benchmark as bench
from quant import quantize_draft_linear

SCHEME = "int2"     # None / "int8" / "int4" / "int2"
LAYERS = (4,)

_orig_load = bench.load_transformers_models
def _patched_load(model_id, draft_id, device):
    target, draft, tokenizer = _orig_load(model_id, draft_id, device)
    if SCHEME:
        draft = quantize_draft_linear(draft, scheme=SCHEME, layers=LAYERS)
        print(f"[quant] scheme={SCHEME}, layers={LAYERS}")
        for i in range(len(draft.layers)):
            print(f"  L{i} q_proj:", type(draft.layers[i].self_attn.q_proj.weight).__name__)
    else:
        print("[quant] scheme=None (BF16 baseline)")
    return target, draft, tokenizer
bench.load_transformers_models = _patched_load

args = argparse.Namespace(
    model="Qwen/Qwen3-8B",
    draft="z-lab/Qwen3-8B-DFlash-b16",
    dataset="gsm8k",
    max_samples=16,          # 冒烟用 2;正式跑改 64
    reasoning="off",
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    max_new_tokens=2048,
    block_size=None,
)
bench._run_transformers(args)