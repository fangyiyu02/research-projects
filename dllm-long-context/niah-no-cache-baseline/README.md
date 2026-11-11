# Fresh LLaDA NIAH → dLLM-Cache experiment

This is a clean experiment. It does not read, append, repair, or plot any
legacy NIAH file. The synthetic haystack, every prompt, both condition outputs,
and every figure are generated inside this directory.

## Directory contract

```text
config/       frozen experiment configuration and cache-condition template
data/         newly generated deterministic haystack
prompts/      newly generated immutable prompt manifest
locks/        cryptographic experiment/condition locks
src/          newly written experiment code
runs/         newly generated no-cache and dLLM-cache results
analysis/     newly generated tables and figures
docs/         protocol and exact runbook
```

Nothing from an older NIAH run should be copied into any of these directories.

## No-cache baseline

Run all commands from this project root.

```bash
python -m pip install -r requirements.txt

python src/prepare_experiment.py --config config/experiment_v1.json
python src/freeze_experiment.py --config config/experiment_v1.json

# Isolated one-sample smoke test; excluded from formal analysis.
python src/run_no_cache.py \
  --experiment-lock locks/experiment_v1.lock.json \
  --output-dir runs/_smoke/no_cache \
  --limit 1

# Formal run. Re-running this exact command resumes by frozen sample_id.
python src/run_no_cache.py \
  --experiment-lock locks/experiment_v1.lock.json \
  --output-dir runs/no_cache

python src/validate_results.py \
  --condition no_cache \
  --experiment-lock locks/experiment_v1.lock.json \
  --results runs/no_cache/results.jsonl \
  --report runs/no_cache/validation_report.json

python src/plot_results.py \
  --experiment-lock locks/experiment_v1.lock.json \
  --no-cache-results runs/no_cache/results.jsonl \
  --output-dir analysis
```

The baseline figure is written to
`analysis/figures/no_cache_accuracy_heatmap.png`.

## dLLM-Cache condition

`src/generate_dllm_cache.py` is an intentionally non-runnable adapter. After
the cache implementation and its parameters are finalized:

1. Copy `config/cache_v1.template.json` to `config/cache_v1.json`.
2. Implement `generate_dllm_cache()` and set `CACHE_IMPLEMENTED = True`.
3. Set the cache config status to `ready` and replace all placeholder values.
4. Freeze that condition:

   ```bash
   python src/freeze_cache_condition.py \
     --experiment-lock locks/experiment_v1.lock.json \
     --cache-config config/cache_v1.json
   ```

5. Run and validate the cache condition:

   ```bash
   python src/run_dllm_cache.py \
     --experiment-lock locks/experiment_v1.lock.json \
     --cache-lock locks/cache_v1.lock.json \
     --output-dir runs/dllm_cache

   python src/validate_results.py \
     --condition dllm_cache \
     --experiment-lock locks/experiment_v1.lock.json \
     --condition-lock locks/cache_v1.lock.json \
     --results runs/dllm_cache/results.jsonl \
     --report runs/dllm_cache/validation_report.json
   ```

6. Generate all paired comparison figures:

   ```bash
   python src/plot_results.py \
     --experiment-lock locks/experiment_v1.lock.json \
     --no-cache-results runs/no_cache/results.jsonl \
     --cache-results runs/dllm_cache/results.jsonl \
     --output-dir analysis
   ```

The comparison command creates the cache accuracy heatmap, accuracy-difference
heatmap, paired speedup heatmap, peak-memory comparison, and CSV summaries.

See `docs/RUNBOOK.md` before starting a formal GPU run.

