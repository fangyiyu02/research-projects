"""Distribution of tv_est over the mismatches of one Idea B run. No GPU.
    python -m idea_b.tv_quantiles runs_b/ideab_theta-1
"""
import json
import sys
from pathlib import Path

rows = [json.loads(l) for l in (Path(sys.argv[1]) / "mismatches.jsonl").read_text().splitlines() if l.strip()]
tv = sorted(r["tv_est"] for r in rows)
print(f"n = {len(tv)} mismatches")
for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
    print(f"  {int(q * 100):>2d}%  {tv[min(len(tv) - 1, int(q * len(tv)))]:.4f}")
by_pos = {}
for r in rows:
    by_pos.setdefault(r["pos_k"], []).append(r["tv_est"])
print("median tv_est by block position:")
for k in sorted(by_pos):
    v = sorted(by_pos[k]); print(f"  pos {k}: n={len(v):5d} median={v[len(v) // 2]:.4f}")
