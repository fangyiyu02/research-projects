# Research Projects — Fangyi Yu

Code and results from my research on **LLM inference efficiency** and **AI for hardware design**.
Each folder has its own README with the question, setup, results, and how to run it.

| Project | Area | What is here | Status |
|---|---|---|---|
| [Diffusion-based speculative decoding](speculative-decoding/) | LLM inference | Training-free lossy verify rules for DSpark (block-diffusion drafter, Qwen3-8B target) compared against ASD; DFlash drafter quantization sensitivity | Explored rules trail ASD; results reported as-is |
| [Long-context dLLM inference](dllm-long-context/) | LLM inference | UltraLLaDA-8B / LLaDA-8B profiling pipeline: latency & memory benchmarks, frozen RULER prompt bank, chunked-inference equivalence checks, KV-drift measurement up to 64K | Measurement study, Jul–Sep 2026 |
| [AHA! hardware Trojan](hardware/aha-hardware-trojan/) | AI for hardware | Three LLM-assisted hardware Trojans inserted into the Secworks AES core, with Yosys/OpenSTA (SKY130) area and timing reports | Completed, Apr 2026 |
| [SmartCache](hardware/smartcache/) | AI for hardware | Bayesian-optimization search over cache geometry with a trace-driven simulator, plus a SystemVerilog cache and testbench for the selected design | Hackathon project, Feb 2026 |

## Repository layout

```
speculative-decoding/
  dspark-verify-rules/         Idea A / Idea B verify rules, ASD + strict baselines, results
  dflash-drafter-quantization/ weight-only quantization of the DFlash drafter (torchao)
dllm-long-context/
  latency-profiling/           end-to-end latency / peak-memory benchmark + scoring
  niah-no-cache-baseline/      frozen NIAH experiment on LLaDA-8B-Instruct (no-cache baseline, 512–8K)
  ruler-measurement-pipeline/  S0–S3 environment probe, RULER prompt freezing, measurement protocol
  kv-drift/                    layer × step KV drift recorder and analysis
  retrieval-probe/             controlled single-token retrieval task for layer-wise interventions
hardware/
  aha-hardware-trojan/
  smartcache/
```

## Notes

- GPU experiments ran on rented single-GPU machines (RTX 5090 / RTX PRO 6000). Scripts assume the layout
  described in each project's setup notes (usually `/workspace/...`).
- Upstream code (DeepSpec, DFlash, LLaDA, RULER) is **not** vendored. Where I changed upstream code, the
  change is shipped as a `.patch` against a pinned commit.
- Large artifacts (model weights, datasets, prompt banks, per-sample logs) are not committed; summary tables
  and figures are.
- Some detailed protocol documents are in Chinese (files ending in `_zh.md`); every project README is in English.
