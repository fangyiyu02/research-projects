#!/usr/bin/env python3
"""
S0 environment probe -- UltraLLaDA / PRO 6000 Max-Q
Checks: torch arch support, attention backends, attention memory scaling, GPU P2P.

Usage:  python s0_probe.py            # uses GPU0 (and GPU1 for the P2P test)
Output: human-readable report to stdout; save it into your S0 records.
"""

import time
import torch
import torch.nn.functional as F

# --- UltraLLaDA-8B real config: keep the probe on the true shapes ---
N_HEADS = 32
HEAD_DIM = 128          # d_model 4096 / 32 heads
DTYPE = torch.bfloat16
DEV = "cuda:0"

line = lambda: print("-" * 72)


def sec(title):
    print()
    line()
    print(title)
    line()


# =====================================================================
sec("0. Torch / device")
# =====================================================================
print("torch          :", torch.__version__)
print("torch CUDA     :", torch.version.cuda)
print("arch list      :", torch.cuda.get_arch_list())
print("  -> sm_120 present:", any("120" in a for a in torch.cuda.get_arch_list()))
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i}: {p.name}  cc={p.major}.{p.minor}  "
          f"SMs={p.multi_processor_count}  mem={p.total_memory / 2**30:.2f} GiB")


# =====================================================================
sec("1. flash-attn package (layer 1 + 2)")
# =====================================================================
try:
    import flash_attn
    print("flash_attn version :", flash_attn.__version__)
    from flash_attn import flash_attn_func

    # dLLM attention is BIDIRECTIONAL -> causal=False
    q = torch.randn(1, 2048, N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    o = flash_attn_func(q, q, q, causal=False)
    torch.cuda.synchronize()
    print("kernel run         : OK  ", tuple(o.shape), o.dtype)
    del q, o
except ImportError as e:
    print("flash_attn         : NOT INSTALLED --", e)
except Exception as e:
    print(f"flash_attn         : RUNTIME FAIL -- {type(e).__name__}: {e}")

try:
    from transformers.utils import is_flash_attn_2_available
    print("transformers sees FA2:", is_flash_attn_2_available())
except Exception as e:
    print("transformers check   : skipped --", e)


# =====================================================================
sec("2. PyTorch SDPA backends (layer 3)")
# =====================================================================
B, S = 1, 4096
q = torch.randn(B, N_HEADS, S, HEAD_DIM, dtype=DTYPE, device=DEV)

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    names = ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"]
    for name in names:
        backend = getattr(SDPBackend, name, None)
        if backend is None:
            print(f"{name:22s} n/a in this torch")
            continue
        try:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(q, q, q, is_causal=False)
            torch.cuda.synchronize()
            print(f"{name:22s} OK")
            del out
        except Exception as e:
            print(f"{name:22s} FAIL  {type(e).__name__}: {str(e)[:80]}")
except ImportError:
    print("torch.nn.attention unavailable (torch < 2.3); skipping per-backend probe")

del q
torch.cuda.empty_cache()


# =====================================================================
sec("3. Attention memory scaling (layer 4 -- DECISIVE)")
# =====================================================================
print("Linear growth  -> flash/mem-efficient, O(L)")
print("Quadratic      -> math backend materialised LxL  ==> 128K impossible")
print()
print(f"{'L':>8} {'extra peak (MiB)':>18} {'predicted L^2 (MiB)':>22}")

for S in (2048, 4096, 8192):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q = torch.randn(1, N_HEADS, S, HEAD_DIM, dtype=DTYPE, device=DEV)
    base = torch.cuda.memory_allocated()
    try:
        out = F.scaled_dot_product_attention(q, q, q, is_causal=False)  # default choice
        torch.cuda.synchronize()
        extra = (torch.cuda.max_memory_allocated() - base) / 2**20
        l2 = N_HEADS * S * S * 2 / 2**20
        print(f"{S:>8} {extra:>18.1f} {l2:>22.1f}")
        del out
    except torch.cuda.OutOfMemoryError:
        print(f"{S:>8} {'OOM':>18} {N_HEADS * S * S * 2 / 2**20:>22.1f}")
    del q

torch.cuda.empty_cache()


# =====================================================================
sec("4. GPU0 <-> GPU1 P2P")
# =====================================================================
if torch.cuda.device_count() < 2:
    print("only one visible GPU; skipping")
else:
    print("can_device_access_peer(0,1):", torch.cuda.can_device_access_peer(0, 1))
    print("can_device_access_peer(1,0):", torch.cuda.can_device_access_peer(1, 0))

    # 1 GiB == the PP=2 boundary hidden state at 128K (131072 x 4096 x 2B)
    n = 1024**3 // 2  # bf16 elements
    a = torch.empty(n, dtype=DTYPE, device="cuda:0")
    b = torch.empty(n, dtype=DTYPE, device="cuda:1")

    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()

    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N
    bw = a.numel() * 2 / dt / 1e9
    print(f"\n1 GiB D2D copy: {dt * 1e3:7.2f} ms   {bw:6.1f} GB/s")
    print("  ~22-26 GB/s -> true P2P over Gen4 x16")
    print("  ~11-13 GB/s -> host bounce (no P2P), cost doubles")
    del a, b

torch.cuda.empty_cache()
print("\ndone.")
