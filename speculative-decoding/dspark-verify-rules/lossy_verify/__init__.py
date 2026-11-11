"""lossy_verify — Idea A experiment package. Lives beside DeepSpec, never inside it.

Layout (see 代码方案 §2):
    context.py    VerifyContext: tensors handed out by the verify hook + derived quantities
    rules.py      LossyRule base (three gates: single-point / per-block M / per-request B) + 6 rules
    logger.py     MismatchLogger: mismatches.jsonl + positions.jsonl
    evaluator.py  LossyDSparkEvaluator(Qwen3DSparkEvaluator): per-sample grading + timing
    metrics.py    gsm8k gold answers, \\boxed{} extraction, correctness
    run.py        dispatcher (one rule, one parameter set, one output dir)
    sweep.sh      loop run.py over a parameter list
    analyze.py    tables + histograms over a directory of runs

DeepSpec is imported from $DEEPSPEC_DIR (default: ../DeepSpec relative to this package).
"""
from __future__ import annotations

import os
import sys

DEEPSPEC_DIR = os.environ.get(
    "DEEPSPEC_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DeepSpec")),
)
if not os.path.isdir(os.path.join(DEEPSPEC_DIR, "deepspec")):
    raise ImportError(
        f"DeepSpec not found at {DEEPSPEC_DIR}. Set DEEPSPEC_DIR or place lossy_verify/ beside DeepSpec/."
    )
if DEEPSPEC_DIR not in sys.path:
    sys.path.insert(0, DEEPSPEC_DIR)
