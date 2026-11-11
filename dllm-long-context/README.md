# Long-Context Inference Efficiency of Diffusion LLMs

A measurement study of where time and memory go when a diffusion language model (dLLM) decodes long inputs.
The main model is **UltraLLaDA-8B** (LLaDA extended to 128K context); short-context checks use
**LLaDA-8B-Instruct**. Work done Jul–Sep 2026 with Prof. Bo Yuan (Rutgers).

The code calls the official LLaDA `generate()` (pinned commit) instead of reimplementing decoding.
Instrumentation and memory-policy changes are kept in separate modules, and run scripts only dispatch.

| Folder | Question | Main result |
|---|---|---|
| [`latency-profiling/`](latency-profiling/) | How do end-to-end latency and peak memory scale with input length? | Latency fits a linear + quadratic model in input length, with the quadratic share reaching ~33% at 16K. Operator breakdown at 4K–16K: lm_head ~32%, attention ~15%, FFN ~10% |
| [`ruler-measurement-pipeline/`](ruler-measurement-pipeline/) | Can long-context runs be measured reproducibly, and where does memory go? | Frozen RULER prompt bank + pre-registered protocol. The **logits tensor**, not KV or attention, is ~98% of dynamic memory; a linear memory model predicts peak usage within 0.2%. Chunking `lm_head` (chunk 512) cuts peak memory by ~40% with bitwise-identical output |
| [`niah-no-cache-baseline/`](niah-no-cache-baseline/) | No-cache NIAH baseline on LLaDA-8B-Instruct, 512–8192 tokens | 100% accuracy up to 4096 tokens (the model's trained window), then falls to 63% at 8192; latency grows from 3.1 s to 56 s |
| [`kv-drift/`](kv-drift/) | At 4K–64K, how much does each layer's K/V change between denoising steps, and are the layer and step axes coupled? | Shallow layers barely drift. Drift is concentrated in the first 20–30 steps. Steady-state drift **falls ~40% from 4K to 32K**. Layer × step coupling explains only 10–15% of the field |
| [`retrieval-probe/`](retrieval-probe/) | Controlled single-token retrieval task for layer-wise "remove the critical token" interventions | Data generator and verification only (tokenizer-checked single-token answers at 20/50/80% depth) |

## Folder details

### latency-profiling
`bench_latency.py` loads RULER JSONL samples, warms up, calls the official `generate()`, and records E2E latency
and CUDA peak memory per sample. `scoring.py` scores the predictions against RULER answers. The official
`generate.py` must be importable (placed beside the script or on `PYTHONPATH`).

### ruler-measurement-pipeline
Stages follow a pre-registered protocol. Detailed protocols are in Chinese: `S2_RULER_prompt_protocol_zh.md`
and `S3_measurement_protocol_zh.md`.

| Stage | Files | What it does |
|---|---|---|
| S0 | `s0_probe.py` | GPU arch support, attention backends, attention memory scaling on UltraLLaDA's real shapes |
| S1 | `s1_seed_and_metadata.py` | RNG reset before each run; records config and the SDPA backend actually chosen (MATH fallback is refused) |
| S2 | `s2_freeze_ruler_sources.py`, `s2_generate_niah1.py`, `s2_generate_ruler_accuracy.py`, `s2_validate_ruler_accuracy.py` | Generates RULER prompts once with a pinned tokenizer, stores `input_ids`, and checks length error (≤ 0.2% and ≤ ±8 tokens) and needle-depth layout |
| S3 | `no_cache/`, `s3_measurement_config.template.json` | Same runtime for every method: `no_cache_nochunk` vs. `no_cache_chunked` (sequence-chunked FFN + `lm_head`) |
| S3a | `chunked-inference-check/` | Checks the chunked implementation: module inspection, A/B/C variants (official / chunked FFN / chunked FFN + `lm_head`) must produce identical tokens, plus an FP32 single-layer test to separate bugs from BF16 reduction-order noise |
| env | `bootstrap.sh` | Reproducible vast.ai setup (venv, CUDA 12.8, pinned LLaDA commit and UltraLLaDA revision) |

RULER itself (NVIDIA) is not vendored; the S2 scripts expect a fresh clone at a recorded commit.

### niah-no-cache-baseline
A self-contained experiment with frozen config, a freshly generated haystack, and cryptographic locks on prompts
and conditions. It covers 16 lengths × 11 needle depths × 10 samples = 1,760 prompts, with 1,760/1,760 valid
results. `src/generate_dllm_cache.py` is a deliberately non-runnable adapter stub for the planned dLLM-Cache
condition, which was not run. Summary table and heat map are in `results/`; the 81 MB prompt bank and the
per-sample results are not committed. See the folder's own README for the runbook.

### kv-drift
See [`kv-drift/README.md`](kv-drift/README.md).

### retrieval-probe
`generate_task.py` builds a 4-item price-retrieval task (answers 1–9, each a single LLaDA token; inputs of
501–512 tokens with one trailing mask; evidence at ~20/50/80% depth) and writes `data/pilot.jsonl`,
`data/eval.jsonl` and `data/manifest.json`. Details are in `README_zh.md`.
