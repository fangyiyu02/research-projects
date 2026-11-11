"""
S3a Step 0 —— 探测 UltraLLaDA 的实际模块结构。
 
在打任何补丁之前必须先跑这个。它只加载到 CPU（meta device），
不占 GPU，几十秒就能跑完。
 
用法:
    python s3a_inspect.py --model ~/hf_cache/UltraLLaDA
"""
import argparse
import inspect
import torch
from transformers import AutoModelForCausalLM, AutoConfig
 
 
def sep(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
 
    # ---------- 1. config 里的关键开关 ----------
    sep("1. CONFIG（决定补丁走哪条路）")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    for k in ["n_layers", "d_model", "n_heads", "n_kv_heads",
              "mlp_hidden_size", "mlp_ratio", "vocab_size", "embedding_size",
              "weight_tying", "scale_logits", "activation_type",
              "block_type", "include_bias", "rope", "rope_theta",
              "max_sequence_length", "flash_attention"]:
        if hasattr(cfg, k):
            print(f"  {k:24s} = {getattr(cfg, k)}")
 
    tying = getattr(cfg, "weight_tying", None)
    print()
    if tying:
        print("  ⚠️  weight_tying=True → lm_head 不是独立模块，")
        print("      补丁 B 需要改 LLaDAModel.forward 而不是替换 ff_out。")
    else:
        print("  ✅ weight_tying=False → lm_head 是独立的 transformer.ff_out，")
        print("      补丁 B 可以直接用 Identity 替换。")
 
    # ---------- 2. 加载到 meta device（不占显存） ----------
    sep("2. 顶层模块树")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    print(f"  外层类型: {type(model).__name__}")
 
    # 逐层往下找 transformer
    node, path = model, "model"
    for _ in range(4):
        kids = dict(node.named_children())
        print(f"  {path}: {list(kids.keys())}")
        if "transformer" in kids:
            path = f"{path}.transformer"
            node = kids["transformer"]
            print(f"  {path}: {list(dict(node.named_children()).keys())}")
            break
        if len(kids) == 1:
            k = list(kids)[0]
            path, node = f"{path}.{k}", kids[k]
        else:
            break
 
    transformer = node
    print(f"\n  → transformer 的访问路径: {path}")
 
    # ---------- 3. lm_head ----------
    sep("3. LM HEAD（补丁 B 的目标）")
    if hasattr(transformer, "ff_out"):
        h = transformer.ff_out
        print(f"  transformer.ff_out = {type(h).__name__}")
        print(f"    in_features  = {getattr(h, 'in_features', '?')}")
        print(f"    out_features = {getattr(h, 'out_features', '?')}")
        print(f"    bias         = {getattr(h, 'bias', None) is not None}")
    else:
        print("  ⚠️ transformer 下没有 ff_out —— 大概率 weight_tying=True")
 
    # ---------- 4. 一个 block 的内部 ----------
    sep("4. BLOCK 内部结构（补丁 A 的目标）")
    blocks = transformer.blocks
    blk = blocks[0] if not isinstance(blocks, torch.nn.ModuleDict) else blocks["0"]
    print(f"  block 类型: {type(blk).__name__}   共 {len(blocks)} 层\n")
    for n, m in blk.named_children():
        inf = getattr(m, "in_features", None)
        outf = getattr(m, "out_features", None)
        shape = f"{inf} -> {outf}" if inf else ""
        print(f"    {n:16s} {type(m).__name__:24s} {shape}")
 
    # ---------- 5. block.forward 源码 ----------
    sep("5. BLOCK.FORWARD 源码（确认 ff_proj/act/ff_out 是不是连续三行调用）")
    try:
        src = inspect.getsource(type(blk).forward)
        print(src)
    except Exception as e:
        print(f"  取不到源码: {e}")
        print("  手动看: python -c \"import inspect, transformers; ...\"")
        print(f"  或直接打开: {type(blk).__module__}")
 
    # ---------- 6. 模型 forward 的尾部 ----------
    sep("6. MODEL.FORWARD 尾部（确认 ff_out 之后还做不做别的）")
    try:
        inner = model
        for _ in range(3):
            kids = dict(inner.named_children())
            if "transformer" in kids:
                break
            if len(kids) == 1:
                inner = list(kids.values())[0]
        src = inspect.getsource(type(inner).forward)
        tail = src.splitlines()
        # 只打含 ff_out / logits / scale 的那些行
        print("  （只显示与 logits 相关的行）")
        for i, line in enumerate(tail):
            if any(w in line for w in ["ff_out", "logits", "weight_tying",
                                       "scale_logits", "wte.weight"]):
                print(f"    {i:4d}| {line}")
    except Exception as e:
        print(f"  取不到源码: {e}")
 
    # ---------- 7. generate.py 里的 dtype 路径 ----------
    sep("7. 提醒")
    print("""
  还要单独确认一次 generate.py 里的 dtype 提升路径：
 
      grep -n "float64\\|softmax\\|gumbel\\|low_confidence" <你的>/generate.py
 
  重点看 `low_confidence` 分支里的 softmax 是不是
      p = F.softmax(logits.to(torch.float64), dim=-1)
  这一行不受 temperature==0 保护，是 fp64 的真正来源。
""")
 
 
if __name__ == "__main__":
    main()