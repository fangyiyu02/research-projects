# Formal runbook

## Before preparation

- Use a new directory containing only this project.
- Do not copy old NIAH prompts, results, or plots into it.
- Review `config/experiment_v1.json`; a change after locking requires a new
  experiment version, not `--overwrite`.

## Preparation gate

1. Run `prepare_experiment.py` once.
2. Confirm 1,760 prompts: 16 lengths × 11 depths × 10 samples.
3. Inspect several prompt records and exact token spans.
4. Run `freeze_experiment.py` once and record the printed SHA-256.

## Baseline gate

1. Use `_smoke/no_cache` for a one-sample smoke test.
2. Start the formal `runs/no_cache` directory without a `results.jsonl`.
3. Keep the GPU model, clocks/power policy, and competing workload as stable as
   practical; preserve every session metadata file.
4. Resume only with the same experiment lock and output directory.
5. Require validation status `valid` before plotting.

## Cache gate

1. Do not edit the frozen experiment config, prompts, common runner, or baseline.
2. Implement only the cache adapter and finalize a separate cache config.
3. Freeze the cache condition against the existing experiment lock.
4. Run cache inference on the same sample IDs and validate it.
5. Plot paired comparisons only when both result sets are complete and valid.

