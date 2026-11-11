# KV drift in long-context dLLM decoding

**Question.** A dLLM recomputes every token's K/V at each denoising step, and because attention is
bidirectional, already-decoded tokens keep changing. Elastic-Cache (ICLR 2026) designed its cache refresh
around this "KV drift", but measured it only below ~1.5K tokens. At 4K–64K:

1. What does drift look like, and do the short-context patterns still hold?
2. Are the layer and step axes coupled, or does drift factor into a layer profile × a step profile?

**Definition.** For each layer, compare the K (or V) vector of every position at step t with step t−1. The
recorder logs cosine similarity (`sim_K/V`), relative L2 change (`rel_K/V`), cumulative change vs. step 0
(`total_K/V`), a per-token heat map, step time and peak memory. Each is a `[steps × layers]` field.

## Setup

| Item | Value |
|---|---|
| Model / GPU | UltraLLaDA-8B (BF16), 1× RTX PRO 6000 96 GB |
| Decoding | official LLaDA `generate()`, block 32, low-confidence remasking, T = 0, 512 steps (1 token/step), gen 512 |
| Data | RULER `niah_single_2` (natural essay haystack). Repetitive-noise haystacks were excluded because they artificially suppress drift |
| Lengths | 4K / 8K / 16K / 32K × 10 samples; 64K × 2 samples (indicative only) |
| Cache | none: full recompute every step, so the drift is intrinsic to the model and not a cache-staleness artifact |

Probes only observe. Decoded tokens are identical to an uninstrumented run.

## Results

| Length | n | Coupling (full) | Coupling (excl. opening) | Opening drift | Steady drift | Opening / steady |
|---|---|---|---|---|---|---|
| 4K | 10 | 0.147 | 0.137 | 0.00114 | 0.00040 | 2.9× |
| 8K | 10 | 0.156 | 0.126 | 0.00117 | 0.00037 | 3.2× |
| 16K | 10 | 0.090 | 0.149 | 0.00221 | 0.00032 | 6.9× |
| 32K | 10 | 0.054 | 0.098 | 0.00226 | 0.00024 | 9.5× |
| 64K | 2 | 0.083 | 0.054 | 0.00157 | 0.00018 | 8.7× |

Coupling index = 1 − σ₁² / Σσ², the share of the step × layer field that a rank-1 (layer profile × step profile)
approximation cannot explain.

1. **Layers act like a switch, not a ramp.** Layers 0–15 barely drift; drift jumps around layer 16 and peaks at
   layers 19–22 for K. The gradual "deeper drifts more" slope reported at short context does not hold.
2. **A noisy opening, then a quiet steady state.** The first 20–30 steps dominate. The opening/steady ratio grows
   from 2.9× to 9.5× with length. The hypothesized spikes at 32-step block boundaries do not appear.
3. **Longer context means lazier KV.** Steady-state drift falls monotonically by ~40% from 4K to 32K, and 64K
   follows the trend. Cached K/V stay valid longer exactly where caching matters most.
4. **Coupling exists but is minor.** A layer profile × step profile explains 85–90% of the field. The residual
   sits in the K transition band (layers 16–18) during the opening steps.
5. **K and V share a shape with different parameters.** V drifts 2–3× more than K, its peak sits deeper (layers
   24–25), and its transition is a gradual slope from about layer 13.

Implication: a static, offline-profiled refresh schedule (two-phase, length-adaptive, targeted at the drift
peak, asymmetric for K/V) could avoid attention-weight-based triggers that conflict with FlashAttention.

**Limitations.** 64K has only 2 samples. Results come from one task and one model. Parallel decoding (several
tokens per step) would need recalibration.

## Files

| File | Role |
|---|---|
| `drift_recorder.py` | Library: per-layer hooks that detect step boundaries and accumulate the drift statistics online |
| `run_drift.py` | One sample: load model + frozen prompt, attach probes, call official `generate()`, write `.npz` + `.json` |
| `test_smoke.py` | 5-step fake-model smoke test for hooks, step detection and output format |
| `plot_drift.py` | Per-sample heat maps, layer curves, cumulative curves |
| `coupling_map.py` | SVD of the drift field: measured / rank-1 / residual (coupling map) per length |
| `check_32k.py` | Spike periodicity and coupling index with the opening removed |
| `drift_report_zh.md` | Full report (Chinese) |

The cross-length aggregation scripts mentioned in the report (`analyze_lengths.py`, `check_lengths.py`,
`run_64k.sh`) lived on the rented GPU machine and are not included.
