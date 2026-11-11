"""Idea B dispatcher: one θ, one output directory. Same outputs as lossy_verify/run.py plus tv_est per mismatch.

    cd /workspace && source venv-deepspec/bin/activate
    python -m idea_b.run --theta 0.05 --n 200 --out runs_b/ideab_theta0.05

Fixed: temperature 0, confidence_threshold 0 (scheduler off, recorder on), enable_thinking False, bsz 1.
Outputs in --out: config.json, summary.json, samples.jsonl, mismatches.jsonl, positions.jsonl.
"""
from __future__ import annotations

import os
import sys

# must be set before CUDA initialises
if "--deterministic" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

from lossy_verify import DEEPSPEC_DIR  # noqa: E402
from lossy_verify.evaluator import LossyDSparkEvaluator  # noqa: E402
from lossy_verify.metrics import load_gsm8k_with_answers  # noqa: E402
from .logger import IdeaBLogger  # noqa: E402
from .rule import IdeaBRule  # noqa: E402

DEFAULT_SEED = 980406  # DeepSpec eval.py default


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theta", type=float, required=True, help="TV threshold; accept a mismatch when tv_est <= theta; theta < 0 never accepts")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k"])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="deepseek-ai/dspark_qwen3_8b_block7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG; only if two runs differ")
    ap.add_argument("--no-positions-log", action="store_true", help="skip positions.jsonl")
    return ap.parse_args()


def deepspec_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", DEEPSPEC_DIR, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_rows = load_gsm8k_with_answers()
    logger = IdeaBLogger(out, log_positions=not a.no_positions_log)
    rule = IdeaBRule(theta=a.theta, logger=logger)

    # mirror eval.py's argparse namespace exactly; the evaluator reads these fields
    args = SimpleNamespace(
        target_name_or_path=a.target,
        draft_name_or_path=a.draft,
        max_new_tokens=a.max_new_tokens,
        temperature=0.0,
        confidence_threshold=0.0,
        tensorboard_dir=None,
        step=None,
        seed=a.seed,
        tasks=[(a.dataset, a.n)],
    )
    config = {
        "rule": rule.name,
        "rule_params": rule.params(),
        "n": a.n,
        "dataset": a.dataset,
        "max_new_tokens": a.max_new_tokens,
        "temperature": 0.0,
        "confidence_threshold": 0.0,
        "seed": a.seed,
        "deterministic": a.deterministic,
        "target": a.target,
        "draft": a.draft,
        "deepspec_commit": deepspec_commit(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    assert torch.cuda.device_count() == 1, "run.py is single-GPU; set CUDA_VISIBLE_DEVICES=0"
    ev = LossyDSparkEvaluator(0, args, rule=rule, logger=logger, dataset_rows=dataset_rows)
    rule.bind_draft_model(ev.draft_model)
    try:
        ev.evaluate()  # parent: recorder start/finish + DeepSpec's own tables
    finally:
        logger.close()

    # ---- samples.jsonl ----
    with (out / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for rec in ev.sample_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- summary.json ----
    recs = ev.sample_records
    n = len(recs)
    row = ev.metrics_rows[0] if ev.metrics_rows else {}
    total_out = sum(r["num_output_tokens"] for r in recs)
    total_fwd = sum(r["verify_rounds"] + 1 for r in recs)  # +1 prefill forward per sample
    total_wall = sum(r["wall_s"] for r in recs)
    recorder_rows = None
    if ev.confidence_head_recorder is not None:
        recorder_rows = [dict(r) for r in getattr(ev.confidence_head_recorder, "rows", [])]
    summary = {
        "rule": rule.name,
        "rule_params": rule.params(),
        "n": n,
        # judgement metrics
        "tau": row.get("acceptance_length"),  # accepted per round incl. bonus token
        "accuracy_strict": sum(r["correct_strict"] for r in recs) / max(n, 1),
        "accuracy_fallback": sum(r["correct_fallback"] for r in recs) / max(n, 1),
        # DeepSpec's own columns
        "draft_tokens_per_proposal": row.get("draft_tokens_per_proposal"),
        "verify_rate": row.get("verify_rate"),
        "accept_rates_by_position": row.get("accept_rates_by_position"),
        # explanation metrics
        "extract_fallback_count": sum(r["extract_fallback"] for r in recs),
        "hit_max_new_tokens_count": sum(r["hit_max_new_tokens"] for r in recs),
        "mean_output_tokens": total_out / max(n, 1),
        "target_forwards_per_output_token": total_fwd / max(total_out, 1),  # implementation-free speedup proxy
        "mean_wall_s": total_wall / max(n, 1),
        "tokens_per_s": total_out / max(total_wall, 1e-9),
        "mismatches_examined": logger.total_examined,
        "mismatches_accepted": logger.total_accepted,
        "accepted_mismatches_per_100_tokens": 100.0 * logger.total_accepted / max(total_out, 1),
        # recorder calibration is only meaningful when the prefix label is the strict one
        "recorder_valid": False,
        "confidence_recorder": recorder_rows,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("confidence_recorder", "accept_rates_by_position")},
                     indent=2, default=str), flush=True)
    ev.clean_up()


if __name__ == "__main__":
    main()
