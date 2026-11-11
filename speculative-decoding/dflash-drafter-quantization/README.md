# DFlash drafter quantization sensitivity

**Question.** The DFlash drafter is only 5 transformer layers. How much weight-only quantization can it take
before acceptance length τ drops, and is any one layer more sensitive than the others?

## Setup

| Item | Value |
|---|---|
| Target / drafter | `Qwen/Qwen3-8B` (BF16, untouched) / `z-lab/Qwen3-8B-DFlash-b16` |
| Code base | official `z-lab/dflash`, transformers backend; `run_quant_exp.py` monkey-patches the model loader |
| What is quantized | drafter `q/k/v/o_proj` + `gate/up/down_proj` weights only (torchao); embedding and lm_head untouched |
| Schemes | INT8 (`Int8WeightOnlyConfig`), INT4 (group 128), INT2 (`IntxWeightOnlyConfig`, group 32) |
| Data | GSM8K, greedy, reasoning off; 64 samples (all-layer runs), 16 samples (per-layer scan) |
| Hardware | 1× RTX 5090 |

## Results

All-layer quantization (64 samples):

| Drafter weights | τ | Speedup vs. AR | 1-token acceptances | Full-block (16) acceptances |
|---|---|---|---|---|
| BF16 | 6.29 | 4.59× | 12.7% | 8.4% |
| INT8 | 6.28 | 4.27× | 12.7% | 8.5% |
| INT4 | 6.24 | 4.37× | 12.3% | 8.2% |
| INT2 | 4.25 | 2.63× | 18.1% | 0.9% |

Per-layer INT2 scan, quantizing one layer at a time (16 samples):

| INT2 layer | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| τ | **5.76** | 6.05 | 5.98 | 6.14 | 6.05 |

**Takeaways**

- Down to INT4, τ and the acceptance histogram barely move. The cliff lies between 2 and 4 bits.
- INT2 damage is selective: full-block acceptances collapse (8.4% → 0.9%) while short acceptances grow.
- L0, the layer closest to the target-feature injection, looks most sensitive. With n = 16 this is
  preliminary.
- Single-layer losses add up to about 1.5 τ, less than the 2.04 τ lost when all five layers are INT2, so the
  error compounds across layers.
- Wall-clock speedup drops at INT8/INT4 (dequantization overhead at batch size 1). The quantized drafter saves
  memory but not time in eager mode.

## Files and how to run

| File | Role |
|---|---|
| `quant.py` | `quantize_draft_linear(model, scheme, layers)`: torchao weight-only quantization of selected drafter layers |
| `run_quant_exp.py` | patches `dflash.benchmark.load_transformers_models` and calls the official benchmark; `SCHEME` / `LAYERS` at the top |
| `run_all.sh` / `run_layers.sh` | BF16 → INT8 → INT4 sweep / per-layer INT2 sweep (edit `SCHEME`/`LAYERS` in place with `sed`) |
| `check_env.py` | prints torch / CUDA / torchao versions before a run |
| `probes.patch` | optional timing probes for `dflash/model.py` (draft vs. verify time per round, enabled with `DFLASH_PROFILE=1`) |

```bash
# from the root of a z-lab/dflash checkout, with these files copied beside it
bash run_all.sh      # all-layer BF16 / INT8 / INT4
bash run_layers.sh   # INT2, one layer at a time
```
