"""Speedup table: every run's tokens/s divided by the target-only run's tokens/s. No GPU.

    cd /workspace && python -m lossy_verify.speedup --ar runs/ar --runs runs/strict runs/a3_theta* runs/asd_g*

The first sample of each run is dropped from the timing (CUDA warm-up), for every run alike.
Accuracy is over all samples. Δacc is against the run named strict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(d: Path):
    rows = [json.loads(l) for l in (d / "samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    s = json.loads((d / "summary.json").read_text())
    timed = rows[1:]
    tps = sum(r["num_output_tokens"] for r in timed) / sum(r["wall_s"] for r in timed)
    acc = sum(r["correct_fallback"] for r in rows) / len(rows)
    return {"name": d.name, "rule": s.get("rule"), "params": s.get("rule_params") or {}, "tau": s.get("tau"),
            "acc": acc, "tps": tps, "ids": {r["id"]: r["output_ids"] for r in rows}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    ar = load(a.ar)
    runs = [load(d) for d in a.runs if (d / "summary.json").exists()]
    strict = next((r for r in runs if r["name"] == "strict"), None)

    print(f"target_only: {ar['tps']:.1f} tok/s  acc {100 * ar['acc']:.1f}")
    if strict is not None:
        same = sum(1 for i, x in strict["ids"].items() if ar["ids"].get(i) == x)
        print(f"strict vs target_only identical outputs: {same}/{len(strict['ids'])}")
    print()
    hdr = ["run", "params", "τ", "tok/s", "speedup vs target", "speedup vs strict", "acc", "Δacc vs strict"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    print(f"| target_only |  | 1 | {ar['tps']:.1f} | 1.00 | - | {100 * ar['acc']:.1f} | - |")
    for r in runs:
        p = ",".join(f"{k}={v}" for k, v in r["params"].items() if v is not None)
        vs_strict = f"{r['tps'] / strict['tps']:.3f}" if strict else "-"
        dacc = f"{100 * (r['acc'] - strict['acc']):+.1f}" if strict else "-"
        tau = f"{r['tau']:.3f}" if r["tau"] is not None else "-"
        print(f"| {r['name']} | {p} | {tau} | {r['tps']:.1f} | {r['tps'] / ar['tps']:.2f} | {vs_strict} | "
              f"{100 * r['acc']:.1f} | {dacc} |")


if __name__ == "__main__":
    main()
