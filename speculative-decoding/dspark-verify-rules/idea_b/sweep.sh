#!/usr/bin/env bash
# Loop idea_b.run over θ values. Output dirs: $RUNS/ideab_theta<θ>. Done runs (summary.json present) are skipped.
#   bash idea_b/sweep.sh 0.01 0.02 0.05 0.1
# Env: N (default 200), RUNS (default runs_b). Run from /workspace with venv-deepspec active.
set -euo pipefail
N="${N:-200}"; RUNS="${RUNS:-runs_b}"
mkdir -p "$RUNS"
for v in "$@"; do
  out="$RUNS/ideab_theta${v}"
  if [ -f "$out/summary.json" ]; then echo "skip $out (done)"; continue; fi
  echo "=== ideab --theta $v -> $out"
  python -m idea_b.run --theta "$v" --n "$N" --out "$out" 2>&1 | tee "$out.log"
done
