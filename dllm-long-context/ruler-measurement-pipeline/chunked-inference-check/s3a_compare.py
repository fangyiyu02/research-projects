"""
比对 s3a_chunked.py 产出的多个 json，判断 chunk 是否无损。
 
用法:
    python s3a_compare.py s3a_A.json s3a_B.json s3a_C.json
"""
import json, sys
 
 
def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return
    rs = [json.load(open(p)) for p in paths]
 
    # ---------- 0. 输入必须一致 ----------
    print("=" * 62)
    ins = {r["input_sha256"] for r in rs}
    if len(ins) != 1:
        print("❌ 输入就不一样，后面的比对全部无效")
        for r in rs:
            print(f"   {r['variant']}: input_sha={r['input_sha256']}  "
                  f"src={r['prompt_source']}")
        print("\n   → dummy 模式下三次要用同一个 --dummy-seed")
        return
    print(f"✅ 输入一致  input_sha={ins.pop()}  "
          f"L={rs[0]['exact_input_tokens']}")
 
    # ---------- 1. 输出必须一致（核心判据）----------
    print("=" * 62)
    base = rs[0]
    ok = True
    for r in rs[1:]:
        if r["output_sha256"] == base["output_sha256"]:
            continue
        ok = False
        a, b = base["output_token_ids"], r["output_token_ids"]
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        print(f"❌ {base['variant']} vs {r['variant']}: "
              f"{len(diff)}/{len(a)} 个 token 不同")
        for i, x, y in diff[:8]:
            print(f"     pos {i:4d}: {x} -> {y}")
        if len(diff) <= 2:
            print("     → 只差 1-2 个：大概率是 cuBLAS 换 kernel 导致的 1-ULP "
                  "差异撞上近似平局。dummy 随机输入下置信度分布更平，"
                  "更容易出现。换真 prompt 再看一次。")
        else:
            print("     → 大面积不同：实现有 bug，不是浮点问题。")
    if ok:
        print(f"✅ 输出完全一致  sha={base['output_sha256']}  → chunk 无损")
 
    # ---------- 2. 显存与耗时 ----------
    print("=" * 62)
    print(f"{'变体':<6}{'ffn_chunk':>11}{'head_chunk':>12}"
          f"{'peak_alloc':>12}{'peak_resv':>11}{'wall_ms':>10}{'Δwall':>9}")
    t0 = base["wall_ms"]
    for r in rs:
        d = (r["wall_ms"] - t0) / t0 * 100
        print(f"{r['variant']:<6}{str(r['ffn_chunk'] or '-'):>11}"
              f"{str(r['head_chunk'] or '-'):>12}"
              f"{r['peak_allocated_GiB']:>11.2f}G{r['peak_reserved_GiB']:>10.2f}G"
              f"{r['wall_ms']:>10.0f}{d:>8.1f}%")
 
    # ---------- 3. 诊断 float64 路径 ----------
    print("=" * 62)
    pa = {r["variant"]: r["peak_allocated_GiB"] for r in rs}
    if "A" in pa:
        W = 14.94                       # 8.02B × bf16
        logits_related = pa["A"] - W
        L = rs[0]["exact_input_tokens"] + 128
        bf16_logits = L * 126464 * 2 / 2**30
        ratio = logits_related / bf16_logits
        print(f"A 的峰值 {pa['A']:.2f} GiB − 权重 {W:.2f} GiB "
              f"= {logits_related:.2f} GiB 归给 logits 及其他")
        print(f"单个 bf16 logits 张量 = {bf16_logits:.2f} GiB  →  比值 {ratio:.1f}×")
        if ratio > 6:
            print("  ⇒ 有 fp64 路径（2+8+8=18 字节/元素，比值约 9×）")
            print("    128K 上官方实现会到 ~278 GiB，chunked lm_head 是必需品")
        elif ratio > 2.5:
            print("  ⇒ 疑似 fp32 softmax（2+4=6 字节/元素，比值约 3×）")
        else:
            print("  ⇒ 只有 bf16（比值约 2×），128K 官方实现约 62 GiB")
        if "C" in pa:
            print(f"\nA − C = {pa['A'] - pa['C']:.2f} GiB "
                  f"← chunked lm_head 在 {rs[0]['exact_input_tokens']} 上省下的量")
 
 
if __name__ == "__main__":
    main()
