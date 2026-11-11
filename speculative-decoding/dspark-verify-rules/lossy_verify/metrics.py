"""gsm8k gold answers + answer extraction + grading. Extraction rules are fixed here for all arms.

Build the answer file once (needs network, `datasets` is in DeepSpec's requirements):
    python -m lossy_verify.metrics --build
It writes lossy_verify/data/gsm8k_with_answers.jsonl, aligned row-by-row with DeepSpec's
eval_datasets/gsm8k.jsonl and asserted by question text (not assumed).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from . import DEEPSPEC_DIR

DATA_DIR = Path(__file__).resolve().parent / "data"
GSM8K_WITH_ANSWERS = DATA_DIR / "gsm8k_with_answers.jsonl"


# ---- gold answers -----------------------------------------------------------
def build_gsm8k_with_answers(deepspec_jsonl: str | Path | None = None,
                             out_jsonl: str | Path = GSM8K_WITH_ANSWERS) -> Path:
    from datasets import load_dataset  # lazy: only needed once

    deepspec_jsonl = Path(deepspec_jsonl or os.path.join(DEEPSPEC_DIR, "eval_datasets", "gsm8k.jsonl"))
    rows = [json.loads(l) for l in deepspec_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    hf = load_dataset("openai/gsm8k", "main", split="test")
    assert len(rows) == len(hf), f"row count mismatch: deepspec {len(rows)} vs HF {len(hf)}"

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for i, (row, ex) in enumerate(zip(rows, hf)):
            q = ex["question"].strip()
            assert row["turns"][0].startswith(q), f"row {i}: DeepSpec prompt does not start with HF question"
            gold = ex["answer"].split("####")[-1].strip()
            fh.write(json.dumps({"id": i, "turns": row["turns"], "answer": normalize_number(gold)}) + "\n")
    return out_jsonl


def load_gsm8k_with_answers(path: str | Path = GSM8K_WITH_ANSWERS) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m lossy_verify.metrics --build` once.")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["turns"] = r["turns"][:1]  # same as DeepSpec load_and_process_dataset
    return rows


# ---- extraction ---------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_boxed(text: str) -> str | None:
    """Content of the LAST \\boxed{...} with brace matching; None if absent/unbalanced."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def extract_last_number(text: str) -> str | None:
    m = _NUM_RE.findall(text)
    return m[-1] if m else None


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace(",", "").replace("\\%", "").replace("%", "").strip()
    s = s.rstrip(".").strip()
    # a boxed answer may still carry words ("18 dollars"); keep the last number inside it
    m = _NUM_RE.findall(s)
    return m[-1] if m else s


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) <= 1e-6
    except ValueError:
        return pred == gold


def grade(text: str, gold: str) -> dict:
    boxed = extract_boxed(text)
    pred_strict = normalize_number(boxed) if boxed is not None else None
    fallback = pred_strict is None
    pred_fallback = pred_strict if not fallback else normalize_number(extract_last_number(text))
    return {
        "pred_boxed": pred_strict,
        "pred_fallback": pred_fallback,
        "extract_fallback": fallback,
        "correct_strict": is_correct(pred_strict, gold),      # only \boxed{} counts
        "correct_fallback": is_correct(pred_fallback, gold),  # boxed, else last number in text
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build data/gsm8k_with_answers.jsonl from HF")
    a = ap.parse_args()
    if a.build:
        p = build_gsm8k_with_answers()
        print(f"wrote {p} ({sum(1 for _ in p.open())} rows)")
    else:
        ap.print_help()
