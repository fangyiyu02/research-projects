"""Tables and text histograms over a directory of runs. No GPU, no matplotlib.

    python -m lossy_verify.analyze runs/                 # τ–accuracy table over every runs/*/summary.json
    python -m lossy_verify.analyze runs/ --hist strict   # c_k / r_i / p_D at match vs mismatch positions (from positions.jsonl)
    python -m lossy_verify.analyze runs/ --accepted-pos  # per run: where (pos_k) the accepted mismatches sit
    python -m lossy_verify.analyze runs/ --compare strict asd_g0.25   # two-run identity check on output_ids
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _runs(root: Path):
    for d in sorted(root.iterdir()):
        if (d / "summary.json").exists():
            yield d, json.loads((d / "summary.json").read_text())


def _fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table(root: Path):
    rows = list(_runs(root))
    if not rows:
        print("no runs found"); return
    base = next((s for d, s in rows if d.name == "strict"), None) or next((s for _, s in rows if s["rule"] == "strict"), None)
    hdr = ["run", "rule", "params", "n", "τ", "acc_fallback", "acc_strict", "Δacc(pp)", "hit_max", "fallback",
           "acc.mism/100tok", "fwd/tok", "mean_out_tok", "tok/s"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for d, s in rows:
        params = ",".join(f"{k}={v}" for k, v in (s.get("rule_params") or {}).items() if v is not None)
        dacc = None if base is None else 100 * (s["accuracy_fallback"] - base["accuracy_fallback"])
        print("| " + " | ".join([
            d.name, s["rule"], params, str(s["n"]), _fmt(s["tau"]), _fmt(s["accuracy_fallback"]),
            _fmt(s["accuracy_strict"]), _fmt(dacc, 1), str(s["hit_max_new_tokens_count"]),
            str(s["extract_fallback_count"]), _fmt(s["accepted_mismatches_per_100_tokens"], 2),
            _fmt(s["target_forwards_per_output_token"]), _fmt(s["mean_output_tokens"], 1), _fmt(s["tokens_per_s"], 1),
        ]) + " |")


def _quantiles(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {q: None for q in qs}
    return {q: xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs}


def _text_hist(xs, lo, hi, bins=10, width=40):
    xs = [x for x in xs if x is not None]
    if not xs:
        print("  (empty)"); return
    counts = [0] * bins
    for x in xs:
        b = min(bins - 1, max(0, int((x - lo) / (hi - lo) * bins)))
        counts[b] += 1
    mx = max(counts)
    for i, c in enumerate(counts):
        a, b = lo + (hi - lo) * i / bins, lo + (hi - lo) * (i + 1) / bins
        print(f"  [{a:5.2f},{b:5.2f}) {c:6d} {'#' * int(width * c / mx)}")


def hist(root: Path, run: str):
    p = root / run / "positions.jsonl"
    rows = _load_jsonl(p)
    match = [r for r in rows if r["strict_match"]]
    mism = [r for r in rows if not r["strict_match"]]
    print(f"{run}: {len(rows)} positions, {len(match)} match, {len(mism)} mismatch")
    for key, lo, hi in (("c_k", 0, 1), ("p_D_xk", 0, 1), ("r_i", 0, 10)):
        print(f"\n== {key} ==")
        for name, sub in (("match", match), ("mismatch", mism)):
            q = _quantiles([r[key] for r in sub])
            print(f" {name:9s} quantiles 10/25/50/75/90%: " + " ".join(_fmt(v) for v in q.values()))
            _text_hist([r[key] for r in sub], lo, hi)
    # c_k by position, mismatch only — tells where a θ cut would bite
    print("\n== mismatch c_k median by pos_k ==")
    K = 1 + max(r["pos_k"] for r in rows)
    for k in range(K):
        sub = [r["c_k"] for r in mism if r["pos_k"] == k]
        print(f"  pos {k}: n={len(sub):5d} median={_fmt(_quantiles(sub)[0.5])}")


def accepted_pos(root: Path):
    for d, s in _runs(root):
        if s["rule"] == "strict":
            continue
        rows = _load_jsonl(d / "mismatches.jsonl")
        acc = [r for r in rows if r["accepted"]]
        K = 1 + max((r["pos_k"] for r in rows), default=6)
        counts = [sum(1 for r in acc if r["pos_k"] == k) for k in range(K)]
        reg = [r["r_i"] for r in acc]
        print(f"{d.name:28s} accepted {len(acc):5d}/{len(rows):5d} examined | by pos " + " ".join(f"{c:4d}" for c in counts)
              + f" | r_i median {_fmt(_quantiles(reg)[0.5])}")


def compare(root: Path, a: str, b: str):
    ra = {r["id"]: r for r in _load_jsonl(root / a / "samples.jsonl")}
    rb = {r["id"]: r for r in _load_jsonl(root / b / "samples.jsonl")}
    ids = sorted(set(ra) & set(rb))
    same = sum(1 for i in ids if ra[i]["output_ids"] == rb[i]["output_ids"])
    print(f"{a} vs {b}: {same}/{len(ids)} samples identical output_ids; "
          f"only-in-{a}: {len(set(ra) - set(rb))}, only-in-{b}: {len(set(rb) - set(ra))}")
    for i in ids:
        if ra[i]["output_ids"] != rb[i]["output_ids"]:
            x, y = ra[i]["output_ids"], rb[i]["output_ids"]
            k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            print(f"  id {i}: first divergence at token {k} (len {len(x)} vs {len(y)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--hist", metavar="RUN", help="histograms from RUN/positions.jsonl")
    ap.add_argument("--accepted-pos", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    a = ap.parse_args()
    if a.hist:
        hist(a.root, a.hist)
    elif a.accepted_pos:
        accepted_pos(a.root)
    elif a.compare:
        compare(a.root, *a.compare)
    else:
        table(a.root)
