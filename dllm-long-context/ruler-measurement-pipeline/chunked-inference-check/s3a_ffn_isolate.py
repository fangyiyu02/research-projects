"""
ChunkedFFN 隔离验证 —— 判定 A vs B 的分歧是 bug 还是浮点噪声。
 
只测一层 FFN，不跑 generate，20 分钟内出结果。
 
核心逻辑：
  bf16 的 ULP 相对精度约 2^-8 = 3.9e-3，所以 kernel 换了归约顺序，
  相对误差就在 1e-3 ~ 1e-2 量级，看不出对错。
  fp32 的 ULP 是 2^-24 = 6e-8，归约顺序差异被压到 ~1e-6 以下。
  => 在 fp32 下比，如果还差得多，就只可能是逻辑错了。
 
用法:
    python s3a_ffn_isolate.py --model ~/hf_cache/UltraLLaDA
"""
import argparse, copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
 
from s3a_chunked import ChunkedFFN, _find_transformer
 
 
def report(ref, got, tag):
    """打印误差幅度 + 误差的空间分布（分布形态是区分 bug 和噪声的关键）"""
    r, g = ref.float(), got.float()
    d = (r - g).abs()
    scale = r.abs().max().item()
    max_abs = d.max().item()
    rel = max_abs / scale if scale > 0 else 0.0
    frac = (d > 0).float().mean().item()
    print(f"  {tag:22s} max_abs={max_abs:.3e}  rel={rel:.3e}  "
          f"非零元素占比={frac:.4f}")
    return rel, d
 
 
def spatial(d, chunk, L):
    """
    误差沿序列维的分布。
      均匀        -> 每块都换了 kernel，是浮点噪声
      只在块边界  -> 下标 off-by-one
      只在某些块  -> 切片或写回错了
      第一块为 0  -> 参考实现和第一块碰巧同 kernel，其余错
    """
    per_pos = d.amax(dim=-1)[0]                       # [L]
    nb = (L + chunk - 1) // chunk
    print(f"    分 {nb} 块，逐块的 max 误差:")
    for b in range(nb):
        s, e = b * chunk, min((b + 1) * chunk, L)
        print(f"      块{b} [{s:5d}:{e:5d}] (len={e-s:5d})  "
              f"max={per_pos[s:e].max().item():.3e}  "
              f"mean={per_pos[s:e].mean().item():.3e}")
    print("    块边界前后各一个位置:")
    for b in range(1, nb):
        i = b * chunk
        print(f"      pos {i-1:5d}: {per_pos[i-1]:.3e}   "
              f"pos {i:5d}: {per_pos[i]:.3e}")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=4224)   # = 4096 + 128
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()
 
    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda:0").eval()
 
    blk = _find_transformer(model).blocks[args.layer]
    ff_proj, act, ff_out = blk.ff_proj, blk.act, blk.ff_out
    print(f"层 {args.layer}: ff_proj {ff_proj.in_features}->{ff_proj.out_features}, "
          f"act {type(act).__name__}, ff_out {ff_out.in_features}->{ff_out.out_features}")
 
    L = args.seq
    torch.manual_seed(0)
    # 幅度取 0.5，接近真实 hidden state 的量级；换幅度不影响相对误差结论
    x_bf = (torch.randn(1, L, 4096, device="cuda:0") * 0.5).to(torch.bfloat16)
 
    # ================= 1. BF16 =================
    print("\n" + "=" * 66)
    print("1. BF16 —— 预期会有 1 ULP 量级差异，无法判定对错，只看分布形态")
    print("=" * 66)
    with torch.inference_mode():
        ref = ff_out(act(ff_proj(x_bf)))
 
        # 先确认参考实现本身是确定性的
        ref2 = ff_out(act(ff_proj(x_bf)))
        print(f"  参考实现跑两次是否 bitwise 相同: {torch.equal(ref, ref2)}")
 
        for c in [L, 2048, 1024, 512]:
            m = ChunkedFFN(ff_proj, act, ff_out, c)
            got = m(x_bf)
            rel, d = report(ref, got, f"chunk={c}")
            if c < L and rel > 0:
                spatial(d, c, L)
            # 分块本身必须是确定性的
            if not torch.equal(got, m(x_bf)):
                print(f"    ⚠️ chunk={c} 自身两次不一致，先查这个")
 
    # ================= 2. FP32（判据）=================
    print("\n" + "=" * 66)
    print("2. FP32 —— 判据在这里")
    print("     rel < 1e-6  → 分块逻辑正确，bf16 分歧纯属精度")
    print("     rel > 1e-3  → 逻辑有 bug")
    print("=" * 66)
    ff_proj32 = copy.deepcopy(ff_proj).float()
    ff_out32 = copy.deepcopy(ff_out).float()
    x32 = x_bf.float()
 
    verdict = None
    with torch.inference_mode():
        ref32 = ff_out32(act(ff_proj32(x32)))
        for c in [L, 2048, 1024, 512]:
            got32 = ChunkedFFN(ff_proj32, act, ff_out32, c)(x32)
            rel, d = report(ref32, got32, f"fp32 chunk={c}")
            if c == 1024:
                verdict = rel
                if rel > 1e-3:
                    spatial(d, c, L)
 
    # ================= 3. 结论 =================
    print("\n" + "=" * 66)
    if verdict is None:
        print("没拿到 chunk=1024 的结果")
    elif verdict < 1e-6:
        print(f"✅ fp32 rel={verdict:.2e} < 1e-6")
        print("   → 分块逻辑正确。bf16 下的输出分歧是 GEMM kernel 归约顺序")
        print("     差异经 128 步扩散放大所致，不是实现错误。")
        print("   → 下一步：跑 4K NIAH，确认任务层面没退化。")
    elif verdict < 1e-3:
        print(f"⚠️  fp32 rel={verdict:.2e}，介于两者之间。")
        print("   看上面的逐块分布：均匀 → 大概率仍是数值；")
        print("   集中在某几块或边界 → 查切片下标。")
    else:
        print(f"❌ fp32 rel={verdict:.2e} > 1e-3 → 分块逻辑有 bug")
        print("   重点查：尾块（L 不被 chunk 整除时）、out 的写回下标、")
        print("           act 是否对形状有隐含假设。")
    print("=" * 66)
 
 
if __name__ == "__main__":
    main()
