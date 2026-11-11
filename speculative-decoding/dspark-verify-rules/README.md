# Lossy verify rules for DSpark

**Question.** ASD decides whether to accept a mismatched draft token using the target's own regret r_i. DSpark
already ships two drafter-side signals: a **confidence head** c_k (trained on verify outcomes) and a **Markov
head** (previous-token bias). Can either of them serve as the acceptance condition instead, **without using
r_i anywhere**?

| Idea | Rule at a mismatch position k | Code |
|---|---|---|
| **Idea A**: jump the mismatch, let c_k be the verify bar | accept the draft token if c_k ≥ θ, otherwise truncate. No per-block cap, no budget | `lossy_verify/rules.py` (rule `a3`) |
| **Idea B**: replace with E\*, what happens next? | E = draft token, E\* = target argmax. Estimate how much taking E\* instead of E would shift the target's next-position distribution via the Markov-head bias difference: tv = TV(softmax(z), softmax(z + M(E\*) − M(E))). Accept E if tv ≤ θ, otherwise truncate | `idea_b/rule.py` |
| Baselines | **strict** (lossless DSpark verify) and **ASD** at the paper configuration (B = 8, M = 2, g swept) | `lossy_verify/rules.py` (`strict`, `asd`) |

`rules.py` also contains `a1`, `a2` and `p` (variants that keep ASD's cap/budget structure). They were used in
early exploration only and are not part of the reported results.

## Setup

| Item | Value |
|---|---|
| Target / drafter | `Qwen/Qwen3-8B` / `deepseek-ai/dspark_qwen3_8b_block7` |
| Code base | DeepSpec reference implementation (transformers), commit `005e03b` + `deepspec_hooks.patch` |
| Data | GSM8K, 200 problems (DeepSpec's seed-980406 shuffle), max 512 new tokens |
| Decoding | greedy (T = 0), reasoning off, batch size 1 |
| Hardware | 1× RTX 5090, torch 2.9.1+cu128 |
| Speedup | tokens/s relative to target-only greedy decoding (63.3 tok/s, accuracy 92.0%) |

The patch adds one optional hook (`accept_rule`) to `verify_draft_tokens` and exposes the drafter logits and
the full target logits (including the bonus position). With no rule set, the hook does nothing. I checked
this token-for-token: unpatched vs. patched vs. `strict`-through-hook give 20/20 identical outputs, and
Idea B at θ = −1 reproduces strict on all 200/200 samples.

## Results

Full table: [`results/gsm8k_n200_summary.csv`](results/gsm8k_n200_summary.csv).

| Method | Setting | τ | Accuracy | Speedup |
|---|---|---|---|---|
| DSpark strict (lossless) | – | 6.475 | 92.0% | 4.26× |
| ASD | g = 0.25 (paper config) | 7.142 | 90.0% | 4.68× |
| ASD | g = 0.5 | 7.103 | 91.0% | 4.65× |
| Idea A (c_k only) | θ = 0.97 | 6.549 | 91.0% | 4.30× |
| Idea A (c_k only) | θ = 0.85 | 7.218 | 85.5% | 4.73× |
| Idea B (Markov head) | θ = 0.001 | 6.588 | 91.0% | 4.31× |
| Idea B (Markov head) | θ = 0.99 | 7.285 | 83.0% | 4.75× |

**Takeaways**

- At matched τ, ASD keeps clearly higher accuracy than both rules. The confidence head does not replace r_i:
  c_k overlaps heavily between match and mismatch positions (mismatch median ≈ 0.86, match median ≈ 0.998).
- Idea B's tv estimate is strongly bimodal (25% ≤ 0.05, 75% ≥ 0.996). This suggests a scale mismatch between
  the Markov-head bias and the target logits rather than real next-token changes. Calibrating that scale is
  the open follow-up.
- Both rules depend on DSpark-specific heads, so they would not transfer to EAGLE- or DFlash-style drafters.

## Layout and how to run

```
deepspec_hooks.patch   two-file change to DeepSpec (verify hook + logits fields)
lossy_verify/          Idea A + strict/ASD baselines: context, rules, logger, evaluator, metrics, run, sweep, analyze, speedup
idea_b/                Idea B rule + run/sweep (reuses lossy_verify read-only)
results/               summary CSV + τ–accuracy plot
SETUP_zh.md            full from-scratch setup on a fresh GPU machine (Chinese)
```

```bash
# layout: /workspace/{DeepSpec, lossy_verify, idea_b}
cd /workspace/DeepSpec && git checkout 005e03b && git checkout -b lossy-verify-hook && git apply ../deepspec_hooks.patch
cd /workspace
python -m lossy_verify.metrics --build                       # gsm8k gold answers with per-row checks
python lossy_verify/test_rules_cpu.py && python -m idea_b.test_ideab_cpu   # CPU unit tests
mkdir -p runs runs_b
python -m lossy_verify.run --rule strict --n 200 --out runs/strict
python -m lossy_verify.ar_baseline --n 200 --out runs/ar
bash lossy_verify/sweep.sh asd 0 0.1 0.25 0.5 1.0
bash lossy_verify/sweep.sh a3 0.99 0.97 0.95 0.90 0.85
bash idea_b/sweep.sh 0.001 0.01 0.05 0.1 0.3 0.75 0.99
python -m lossy_verify.analyze runs/
python -m lossy_verify.speedup --ar runs/ar --runs runs/strict runs/asd_g* runs/a3_* runs_b/ideab_theta*
```

Each run writes `config.json`, `summary.json`, `samples.jsonl`, `mismatches.jsonl` and `positions.jsonl`
(per-position c_k, r_i, p_T, p_D).
