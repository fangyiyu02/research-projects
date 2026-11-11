# Fresh NIAH experiment protocol

## Scope

All earlier NIAH prompts, appended result files, timing values, distance
boundaries, and heatmaps are excluded. This experiment creates a new benchmark
and treats the old work as non-input historical material.

## Design

The experimental unit is one immutable prompt. Target total length and
requested needle depth define a grid cell. Each cell has ten independently
seeded prompts. `sample_id` is the blocking key: no-cache and dLLM-Cache must
evaluate exactly the same input IDs.

Prompt order is shuffled once with a recorded seed. Prompt contents use
independent per-sample seeds, so skipping completed sample IDs cannot change
any later prompt. For the strongest speed claim, run an additional smaller
paired timing session after the cache is available, alternating condition order
within prompt pairs to check session/GPU drift.

## Frozen factors

- Model, tokenizer, and both remote-code revisions are 40-character commits.
- BF16, batch size 1.
- Generation length 64, block length 32, 64 denoising steps.
- Temperature 0, CFG 0, low-confidence remasking.
- Fresh synthetic haystack generator version and seed.
- Complete length/depth/sample grid and all prompt seeds.

## Required evidence

Each prompt stores full text, input IDs, hashes, per-sample seed, exact needle
and question token spans, actual length, requested depth, and exact token
distances. Each result stores synchronized latency, throughput, peak allocated
and reserved CUDA memory, output text, recomputable score fields, experiment
and condition lock hashes, revisions, session ID, and environment metadata.

## Primary outputs

- No-cache accuracy heatmap.
- dLLM-Cache accuracy heatmap.
- Cache minus no-cache accuracy heatmap.
- Paired no-cache/cache latency speedup heatmap.
- Peak allocated GPU-memory comparison by sequence length.

