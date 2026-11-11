set -e
for L in 0 1 2 3 4; do
  sed -i "s/^LAYERS = .*/LAYERS = ($L,)/" run_quant_exp.py
  echo "===== INT2 layer $L ====="
  python run_quant_exp.py
done
