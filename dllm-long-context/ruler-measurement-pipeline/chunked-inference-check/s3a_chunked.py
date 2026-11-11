"""
S3a —— chunked FFN + chunked lm_head 的实现与等价性验证（4K, no-cache）。
 
三个变体：
    A  official   官方实现，不做任何修改
    B  ffn        只加 chunked FFN
    C  ffn+head   chunked FFN + chunked lm_head
 
判据：三者的 output_token_ids 必须完全一致。
 
用法（每个变体单独一个进程跑，避免 allocator 状态互相污染）：
    python s3a_chunked.py --model ~/hf_cache/UltraLLaDA \\
        --dummy-length 4096 --dummy-seed 1234 --variant A
    python s3a_chunked.py ... --variant B
    python s3a_chunked.py ... --variant C
"""
import argparse, hashlib, json, time
from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
 
 
# =====================================================================
# 补丁 A —— chunked FFN
# =====================================================================
class ChunkedFFN(nn.Module):
    """
    按 UltraLLaDA 的真实 SwiGLU 路径沿【序列维】分块执行：
        ff_out(act(ff_proj(x)) * up_proj(x))
 
    原理：FFN 是 position-wise 的，第 i 个位置的输出只依赖第 i 个位置的
    输入，位置之间不交互。所以沿序列维切开，每一行的数值完全不变。
    FLOPs 也一个不少 —— 只是中间那个 [1, L, 24576] 的大张量不再同时存在。
 
    峰值:  L × 24576 × 2B   →   chunk × 24576 × 2B
    128K:  6.0 GiB          →   0.37 GiB (chunk=8192)
    """
 
    def __init__(self, ff_proj, up_proj, act, ff_out, chunk):
        super().__init__()
        self.ff_proj = ff_proj      # Linear(d_model, mlp_hidden): gate
        self.up_proj = up_proj      # Linear(d_model, mlp_hidden): up
        self.act = act              # SiLU(gate)
        self.ff_out = ff_out        # Linear(mlp_hidden, d_model): down
        self.chunk = chunk
 
    def _one(self, xc):
        gate = self.act(self.ff_proj(xc))
        up = self.up_proj(xc)
        return self.ff_out(gate * up)
 
    def forward(self, x):                       # x: [B, L, d_model]
        L = x.shape[1]
        # 短序列：块数为 1，等价于不分块，零开销
        if L <= self.chunk:
            return self._one(x)
 
        # 先分配输出，逐块写入。每轮结束时块内的中间张量失去引用，
        # PyTorch 立刻回收，所以同一时刻只有一块的中间量活着。
        out = torch.empty(
            x.shape[0], L, self.ff_out.out_features,
            dtype=x.dtype, device=x.device,
        )
        for i in range(0, L, self.chunk):
            out[:, i:i + self.chunk] = self._one(x[:, i:i + self.chunk])
        return out
 
 
def _chunked_llama_forward(self, x, attention_bias=None,
                           layer_past=None, use_cache=False):
    """UltraLLaDA LLaDALlamaBlock.forward，只有 FFN 段改为分块。"""
    x_normed = self.attn_norm(x)
    q = self.q_proj(x_normed)
    k = self.k_proj(x_normed)
    v = self.v_proj(x_normed)

    if self._activation_checkpoint_fn is not None:
        att, cache = self._activation_checkpoint_fn(
            self.attention, q, k, v, attention_bias,
            layer_past=layer_past, use_cache=use_cache,
        )
    else:
        att, cache = self.attention(
            q, k, v, attention_bias,
            layer_past=layer_past, use_cache=use_cache,
        )

    x = x + self.dropout(att)
    og_x = x
    if self._activation_checkpoint_fn is not None:
        x = self._activation_checkpoint_fn(self.ff_norm, x)
    else:
        x = self.ff_norm(x)
    x = self.chunked_ffn(x)
    x = self.dropout(x)
    x = og_x + x
    return x, cache


def patch_ffn(model, chunk=8192, verbose=True):
    """
    UltraLLaDA 的 block 是独立 gate/up 投影：
        x, x_up = ff_proj(x), up_proj(x)
        x = act(x) * x_up
        x = ff_out(x)

    因而必须替换 block.forward 的 FFN 段；不能把 ff_proj 单独换成
    完整 FFN，否则原 forward 仍会再执行 up_proj 和逐元素乘法。
 
    ⚠️ 跑 s3a_inspect.py 的第 5 节确认这三行确实是连续调用后再用。
    ⚠️ 这里的 blk.ff_out 是【层内的 down 投影】，
       和 transformer.ff_out（lm_head）是两个不同的东西，别搞混。
    """
    transformer = _find_transformer(model)
    n = 0
    for blk in transformer.blocks:
        if hasattr(blk, "chunked_ffn"):
            continue                                   # 已经打过了

        if not hasattr(blk, "up_proj"):
            raise RuntimeError(
                f"{type(blk).__name__} 没有 up_proj；当前补丁只支持 "
                "UltraLLaDA 的 LLaDALlamaBlock"
            )
        blk.chunked_ffn = ChunkedFFN(
            blk.ff_proj, blk.up_proj, blk.act, blk.ff_out, chunk
        )
        # 参数现在由 chunked_ffn 持有；清掉旧入口，避免重复注册。
        blk.ff_proj = nn.Identity()
        blk.up_proj = nn.Identity()
        blk.act = nn.Identity()
        blk.ff_out = nn.Identity()
        blk.forward = MethodType(_chunked_llama_forward, blk)
        n += 1
    if verbose:
        print(f"[patch] chunked FFN 已应用于 {n} 层, chunk={chunk}")
    return model
 
 
# =====================================================================
# 补丁 B —— chunked lm_head
# =====================================================================
def detach_lm_head(model, verbose=True):
    """
    把 transformer.ff_out（lm_head）摘下来换成 Identity。
    之后 model(x).logits 返回的是 [B, L, d_model] 的隐状态
    （ln_f 之后、投影之前），而不是 [B, L, 126464] 的 logits。
 
    返回 lm_head 的权重，交给 chunked_argmax_conf 用。
 
    ⚠️ 只在 weight_tying=False 时可用（inspect 第 1 节会告诉你）。
    """
    transformer = _find_transformer(model)
    head = transformer.ff_out
    assert isinstance(head, nn.Linear), f"ff_out 不是 Linear: {type(head)}"
    assert head.bias is None, "lm_head 有 bias，chunked_argmax_conf 需要补上"
    W = head.weight                                   # [V, d_model]
    transformer.ff_out = nn.Identity()
    if verbose:
        print(f"[patch] lm_head 已摘下, W.shape={tuple(W.shape)}")
    return W
 
 
@torch.no_grad()
def chunked_argmax_conf(h, W, chunk=512, need_conf=True):
    """
    分块做 lm_head 投影，每块算完立刻塌成 argmax 和置信度，然后丢弃。
 
    原理：softmax / argmax 都沿【vocab 维】（最后一维）算，
    分块沿【序列维】切 —— 两个方向垂直，一行的计算只用到这一行的数据。
    所以逐 bit 等价。
 
    峰值:  L × 126464 × (2+8+8)B  →  chunk × 126464 × 18B
    128K:  278 GiB               →  1.2 GiB (chunk=512, 有 fp64 路径时)
 
    h: [B, L, d_model]   W: [V, d_model]
    返回 x0: [B, L] int64,  x0_p: [B, L] float64
    """
    B, L, _ = h.shape
    x0 = torch.empty(B, L, dtype=torch.long, device=h.device)
    x0_p = torch.empty(B, L, dtype=torch.float64, device=h.device) if need_conf else None
 
    for i in range(0, L, chunk):
        lg = F.linear(h[:, i:i + chunk], W)            # [B, c, V]
        idx = lg.argmax(dim=-1)                        # [B, c]  ← 塌掉
        x0[:, i:i + chunk] = idx
        if need_conf:
            # 保持 fp64，和官方 generate.py 的 low_confidence 分支一致。
            # 改成 fp32 会翻转近似平局，就不再"无损"了。
            p = torch.softmax(lg.to(torch.float64), dim=-1)
            x0_p[:, i:i + chunk] = p.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        del lg                                          # 显式释放
    return x0, x0_p
 
 
def _find_transformer(model):
    node = model
    for _ in range(4):
        kids = dict(node.named_children())
        if "transformer" in kids:
            return kids["transformer"]
        if len(kids) == 1:
            node = list(kids.values())[0]
        else:
            break
    raise RuntimeError("找不到 transformer，请看 s3a_inspect.py 第 2 节的路径")
 
 
# =====================================================================
# generate —— 忠实复刻官方 LLaDA generate.py，只在 logits 那一段分叉
# =====================================================================
def get_num_transfer_tokens(mask_index, steps):
    """每个 denoising step 定稿几个 token —— 与官方完全一致"""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    n = torch.zeros(mask_num.size(0), steps,
                    device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        n[i, :remainder[i]] += 1
    return n
 
 
@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=32,
             temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
             mask_id=126336, lm_head_W=None, head_chunk=512):
    """
    lm_head_W=None  → 走官方路径（model 直接吐完整 logits）
    lm_head_W=张量  → 走 chunked lm_head 路径
    """
    assert temperature == 0.0, "本脚本只支持 temperature=0（确定性解码）"
    assert cfg_scale == 0.0, "cfg_scale>0 会让 batch 翻倍，是另一个显存 regime"
 
    dev = next(model.parameters()).device
    P = prompt.shape[1]
    x = torch.full((1, P + gen_length), mask_id, dtype=torch.long, device=dev)
    x[:, :P] = prompt
 
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
 
    for nb in range(num_blocks):
        blk_mask = (x[:, P + nb * block_length: P + (nb + 1) * block_length] == mask_id)
        n_transfer = get_num_transfer_tokens(blk_mask, steps_per_block)
 
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            out = model(x).logits
 
            if lm_head_W is None:
                # ---- 官方路径 ----
                logits = out                                     # [1, L, V]
                x0 = torch.argmax(logits, dim=-1)                # temperature=0 → 无 gumbel
                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=dev, dtype=torch.float64)
                else:
                    raise NotImplementedError(remasking)
                del logits, out
            else:
                # ---- chunked lm_head 路径 ----
                x0, x0_p = chunked_argmax_conf(
                    out, lm_head_W, chunk=head_chunk,
                    need_conf=(remasking == "low_confidence"),
                )
                if remasking == "random":
                    x0_p = torch.rand(x0.shape, device=dev, dtype=torch.float64)
                del out
 
            # 当前 block 右边的位置不参与本轮定稿
            x0_p[:, P + (nb + 1) * block_length:] = -torch.inf
 
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-torch.inf,
                                                                    dtype=x0_p.dtype,
                                                                    device=dev))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=int(n_transfer[j, i]))
                transfer[j, sel] = True
            x[transfer] = x0[transfer]
 
    return x[:, P:]
 
 
# =====================================================================
# 主流程
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dummy-length", type=int, default=4096,
                    help="内部随机生成的 dummy prompt token 数")
    ap.add_argument("--dummy-seed", type=int, default=1234,
                    help="A/B/C 必须使用相同 seed")
    ap.add_argument("--variant", choices=["A", "B", "C"], required=True)
    ap.add_argument("--ffn-chunk", type=int, default=8192)
    ap.add_argument("--head-chunk", type=int, default=512)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", default=None, help="结果 json 落盘路径")
    args = ap.parse_args()

    if args.dummy_length <= 0:
        ap.error("--dummy-length 必须大于 0")
 
    torch.cuda.set_device(0)                      # GPU1 接了显示输出，固定用 GPU0
 
    # ---- 加载模型 ----
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda:0").eval()
    load_peak = torch.cuda.max_memory_allocated()
    print(f"[load] peak_allocated = {load_peak / 2**30:.2f} GiB")

    # ---- 生成固定的随机 dummy prompt ----
    # 只排除 mask token，避免它被 generate() 当作待生成位置。
    vocab_size = int(getattr(model.config, "vocab_size", 126464))
    mask_id = int(getattr(model.config, "mask_token_id", 126336))
    if vocab_size <= 1:
        raise ValueError(f"非法 vocab_size={vocab_size}")
    if not 0 <= mask_id < vocab_size:
        raise ValueError(f"mask_id={mask_id} 不在 [0, {vocab_size}) 内")

    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(args.dummy_seed)
    prompt = torch.randint(
        0, vocab_size - 1, (1, args.dummy_length),
        generator=cpu_rng, dtype=torch.long,
    )
    # 将 [mask_id, vocab_size-2] 整段右移一位，得到完整词表中除 mask 外
    # 的均匀采样；这样不会改变随机数消耗，三次运行可严格复现。
    prompt += (prompt >= mask_id).to(prompt.dtype)
    prompt = prompt.contiguous()
    input_sha256 = hashlib.sha256(prompt.numpy().tobytes()).hexdigest()[:16]
    prompt_source = (
        f"dummy:length={args.dummy_length},seed={args.dummy_seed},"
        f"vocab_size={vocab_size},excluded_mask_id={mask_id}"
    )
    prompt = prompt.to("cuda:0")
    print(f"[prompt] exact_input_tokens = {prompt.shape[1]}")
    print(f"[prompt] dummy_seed = {args.dummy_seed}")
    print(f"[prompt] input_sha256 = {input_sha256}")
 
    # ---- 打补丁 ----
    W = None
    if args.variant in ("B", "C"):
        patch_ffn(model, chunk=args.ffn_chunk)
    if args.variant == "C":
        W = detach_lm_head(model)
 
    # ---- warmup（丢弃）----
    with torch.inference_mode():
        _ = generate(model, prompt[:, :512], steps=4, gen_length=32,
                     block_length=32, lm_head_W=W, head_chunk=args.head_chunk)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
 
    # ---- 正式跑 ----
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out_ids = generate(
            model, prompt,
            steps=args.steps, gen_length=args.gen_length,
            block_length=args.block_length,
            lm_head_W=W, head_chunk=args.head_chunk,
        )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - t0) / 1e6
 
    ids = out_ids[0].tolist()
    res = {
        "variant": args.variant,
        "exact_input_tokens": int(prompt.shape[1]),
        "input_sha256": input_sha256,
        "prompt_source": prompt_source,
        "dummy_seed": args.dummy_seed,
        "ffn_chunk": args.ffn_chunk if args.variant in ("B", "C") else None,
        "head_chunk": args.head_chunk if args.variant == "C" else None,
        "wall_ms": round(wall_ms, 1),
        "peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "load_peak_GiB": round(load_peak / 2**30, 3),
        "output_sha256": hashlib.sha256(
            ",".join(map(str, ids)).encode()).hexdigest()[:16],
        "output_token_ids": ids,
    }
 
    print("\n" + json.dumps({k: v for k, v in res.items()
                             if k != "output_token_ids"}, indent=2))
    print(f"first 16 tokens: {ids[:16]}")
 
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n已写入 {args.out}")
 
 
if __name__ == "__main__":
    main()
 
