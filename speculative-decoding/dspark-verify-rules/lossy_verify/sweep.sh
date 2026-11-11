#!/usr/bin/env bash
# Loop run.py over a parameter list. Output dirs are named <rule>_<param><value>.
#
#   bash lossy_verify/sweep.sh asd 0 0.1 0.25 0.5 1.0          # sweeps --g
#   bash lossy_verify/sweep.sh a1  0.3 0.5 0.7 0.9             # sweeps --theta
#   bash lossy_verify/sweep.sh a3  0.3 0.5 0.7 0.9             # sweeps --theta, M/B off inside the rule
#   bash lossy_verify/sweep.sh p   0.3 0.5 0.7 0.9             # sweeps --theta (p_D threshold)
#   G=0.25 bash lossy_verify/sweep.sh a2 0.5 0.7 0.9           # sweeps --theta at fixed --g=$G
#
# Env: N (default 200), RUNS (default runs), G (a2 only, default 0.25), EXTRA (extra run.py flags).
# Run from /workspace with venv-deepspec active. A run whose summary.json exists is skipped.
set -euo pipefail
rule="$1"; shift
N="${N:-200}"; RUNS="${RUNS:-runs}"; G="${G:-0.25}"; EXTRA="${EXTRA:-}"

case "$rule" in
  asd) param=g ;;
  a1|a2|a3|p) param=theta ;;
  strict) echo "strict takes no parameter; run it directly: python -m lossy_verify.run --rule strict --n $N --out $RUNS/strict"; exit 1 ;;
  *) echo "unknown rule $rule"; exit 1 ;;
esac

for v in "$@"; do
  if [ "$rule" = "a2" ]; then out="$RUNS/a2_g${G}_theta${v}"; flags="--g $G --theta $v"
  else out="$RUNS/${rule}_${param}${v}"; flags="--$param $v"; fi
  if [ -f "$out/summary.json" ]; then echo "skip $out (done)"; continue; fi
  echo "=== $rule $flags -> $out"
  python -m lossy_verify.run --rule "$rule" $flags --n "$N" --out "$out" $EXTRA 2>&1 | tee "$out.log"
done
