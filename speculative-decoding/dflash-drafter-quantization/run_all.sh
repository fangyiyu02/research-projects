set -e
sed -i 's/^SCHEME = .*/SCHEME = None/' run_quant_exp.py
echo "===== BF16 ====="; python run_quant_exp.py
sed -i 's/^SCHEME = .*/SCHEME = "int8"/' run_quant_exp.py
echo "===== INT8 ====="; python run_quant_exp.py
sed -i 's/^SCHEME = .*/SCHEME = "int4"/' run_quant_exp.py
echo "===== INT4 ====="; python run_quant_exp.py
