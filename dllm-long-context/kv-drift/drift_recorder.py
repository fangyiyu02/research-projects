"""
drift_recorder.py — minimal KV drift measurement for dLLM no-cache decoding.

Measures, per decoding step t and layer l:
  sim_K/sim_V    cosine(X^t, X^{t-1}) per token, averaged        [T, L]
  rel_K/rel_V    ||X^t - X^{t-1}|| / ||X^{t-1}||, averaged       [T, L]
  total_K/total_V cosine(X^t, X^0) vs a permanent step-0 snapshot [T, L]
  heat_K (opt)   per-token adjacent-step K drift, strided        [T, L, N/s]
  step_time / peak_mem_gb                                        [T]

Steps are detected AUTOMATICALLY: every time the first hooked module (layer 0)
fires again, a new forward has begun. So any generation loop works unmodified —
attach(), call the official generate(), then save():

    rec = DriftRecorder(model.model.transformer.blocks)
    rec.attach()
    out = generate(model, prompt, ...)   # official LLaDA generate.py, untouched
    rec.save(path); rec.detach()

No RoPE, no attention needed (per-position rotation is identical across steps,
so cosine on pre-RoPE K equals cosine on post-RoPE K).

ADAPT: projections auto-detected — fused `att_proj` (split by block.fused_dims)
or separate `k_proj`/`v_proj`. If neither exists on your fork, edit attach().
NOTE: run with cfg_scale=0 (default). cfg_scale>0 doubles the batch inside the
forward and would mix conditional/unconditional streams in the statistics.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F


class DriftRecorder:
    def __init__(self, blocks, heatmap_token_stride=0,
                 buffers_on_gpu=True, buffer_dtype=torch.bfloat16,
                 total_token_stride=4):
        self.blocks = list(blocks)
        self.L = len(self.blocks)
        self.stride = heatmap_token_stride
        self.on_gpu = buffers_on_gpu
        self.dtype = buffer_dtype
        self.total_token_stride = total_token_stride
        self.prev = {}     # (layer, 'K'|'V') -> rolling buffer, overwritten per step
        self.first = {}    # (layer, 'K'|'V') -> step-0 snapshot, never overwritten
        self._handles, self._buf = [], None
        self._first_key, self._t_open = None, None
        self.logs = {k: [] for k in ("sim_K", "rel_K", "sim_V", "rel_V",
                                     "total_K", "total_V", "step_time",
                                     "peak_mem_gb")}
        self.heat, self.extra = [], {}

    # ---------------------------------------------------------------- hooks
    def attach(self):
        for li, blk in enumerate(self.blocks):
            if hasattr(blk, "att_proj"):                       # fused QKV
                self._handles.append(
                    blk.att_proj.register_forward_hook(self._fused(li, blk)))
            elif hasattr(blk, "k_proj") and hasattr(blk, "v_proj"):
                self._handles.append(
                    blk.k_proj.register_forward_hook(self._one(li, "K")))
                self._handles.append(
                    blk.v_proj.register_forward_hook(self._one(li, "V")))
            else:  # ADAPT
                raise RuntimeError(f"block {li}: no att_proj / k_proj+v_proj found")

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _fused(self, li, blk):
        @torch.no_grad()
        def fn(m, i, out):
            dq, dk, dv = blk.fused_dims
            _, k, v = out.split([dq, dk, dv], dim=-1)
            self._autostep((li, "K"))
            self._consume(li, "K", k)
            self._consume(li, "V", v)
        return fn

    def _one(self, li, kind):
        @torch.no_grad()
        def fn(m, i, out):
            self._autostep((li, kind))
            self._consume(li, kind, out)
        return fn

    # ------------------------------------------------- automatic step slicing
    def _autostep(self, key):
        if self._first_key is None:
            self._first_key = key
        if key == self._first_key:          # layer 0 fired -> new forward begins
            if self._buf is not None:
                self._flush()
            self._open()
        elif self._buf is None:             # safety: mid-forward without open buf
            self._open()

    def _open(self):
        nan = np.full(self.L, np.nan)
        self._buf = {"sim_K": nan.copy(), "rel_K": nan.copy(),
                     "sim_V": nan.copy(), "rel_V": nan.copy(),
                     "total_K": nan.copy(), "total_V": nan.copy(),
                     "heat": [None] * self.L}
        self._t_open = time.time()

    def _flush(self):
        b, self._buf = self._buf, None
        for k in ("sim_K", "rel_K", "sim_V", "rel_V", "total_K", "total_V"):
            self.logs[k].append(b[k])
        self.logs["step_time"].append(time.time() - self._t_open)
        self.logs["peak_mem_gb"].append(
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available() else np.nan)
        if self.stride:
            self.heat.append(b["heat"])

    def finalize(self):
        if self._buf is not None:
            self._flush()

    # ----------------------------------------------------------------- core
    @torch.no_grad()
    def _consume(self, li, kind, cur):
        cur = cur.detach().float()
        prev = self.prev.get((li, kind))
        if prev is not None:
            p = prev.to(cur.device, non_blocking=True).float()
            cos = F.cosine_similarity(cur, p, dim=-1)          # [B, N] per token
            rel = (cur - p).norm(dim=-1) / (p.norm(dim=-1) + 1e-8)
            self._buf[f"sim_{kind}"][li] = float(cos.mean())
            self._buf[f"rel_{kind}"][li] = float(rel.mean())
            if self.stride and kind == "K":
                self._buf["heat"][li] = (1 - cos[0, ::self.stride]).half().cpu()
        ts = self.total_token_stride
        f0 = self.first.get((li, kind))
        if f0 is None:                       # step 0: keep forever
            f0 = cur[:, ::ts].to(self.dtype)
            self.first[(li, kind)] = f0 if self.on_gpu else f0.cpu()
        else:                                # cumulative drift vs step 0
            f = f0.to(cur.device, non_blocking=True).float()
            self._buf[f"total_{kind}"][li] = float(
                F.cosine_similarity(cur[:, ::ts], f, dim=-1).mean())
        b = cur.to(self.dtype)
        self.prev[(li, kind)] = b if self.on_gpu else b.cpu()

    # ----------------------------------------------------------------- save
    def save(self, path):
        self.finalize()
        out = {k: np.stack(v) for k, v in self.logs.items() if v}
        if self.stride and self.heat:
            ref = next((h for s in self.heat for h in s if h is not None), None)
            if ref is not None:
                out["heat_K"] = np.stack([
                    np.stack([h.numpy() if h is not None
                              else np.full(ref.shape, np.nan, np.float16)
                              for h in s]) for s in self.heat])
        for k, v in self.extra.items():
            out[f"extra_{k}"] = np.asarray(v)
        np.savez_compressed(path, **out)
        return path
