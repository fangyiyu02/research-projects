#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# bootstrap.sh — 在新租的 GPU 实例上从零重建实验环境
#
# 用法:
#   chmod +x bootstrap.sh
#   ./bootstrap.sh
#
# 可用环境变量覆盖路径,例如:
#   PERSIST_ROOT=/data ./bootstrap.sh
# ============================================================

# ---------- Paths ----------
# WORK_ROOT   : 可随时删除重建的东西(venv / repo / runs)
# PERSIST_ROOT: 持久卷挂载点,放"重建代价高"的东西(权重 / pip 缓存)
#               按生命周期分目录,不按内容类型分。
WORK_ROOT="${WORK_ROOT:-/workspace}"
# 默认值仍位于实例磁盘；只有显式传入真实挂载卷（如 /data）才可在
# Destroy 实例后继续保留。Stop 状态下 /workspace 本身会保留。
PERSIST_ROOT="${PERSIST_ROOT:-${WORK_ROOT}/persist}"

VENV_DIR="${WORK_ROOT}/venvs/torch211-cu128"
REPO_DIR="${WORK_ROOT}/repos/LLaDA"
RUNS_DIR="${WORK_ROOT}/runs"

MODEL_DIR="${PERSIST_ROOT}/models"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PERSIST_ROOT}/pip-cache}"

# 只建"纯容器"目录。venv / repo 由各自的命令创建,
# 提前建空目录会让 venv 的存在性检查恒真、git clone 报错。
mkdir -p \
    "$MODEL_DIR" \
    "$RUNS_DIR" \
    "$PIP_CACHE_DIR" \
    "$(dirname "$VENV_DIR")" \
    "$(dirname "$REPO_DIR")"

cat <<EOF
============================================================
WORK_ROOT    : ${WORK_ROOT}
PERSIST_ROOT : ${PERSIST_ROOT}
VENV_DIR     : ${VENV_DIR}
REPO_DIR     : ${REPO_DIR}
RUNS_DIR     : ${RUNS_DIR}
MODEL_DIR    : ${MODEL_DIR}
PIP_CACHE    : ${PIP_CACHE_DIR}
============================================================
EOF

# ---------- Python venv ----------
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIRE_PYTHON="3.12.3"
ACTUAL_PYTHON="$($PYTHON_BIN -c 'import platform; print(platform.python_version())')"
if [ "$ACTUAL_PYTHON" != "$REQUIRE_PYTHON" ]; then
    echo "[FATAL] Python ${ACTUAL_PYTHON} != ${REQUIRE_PYTHON}" >&2
    echo "        请换用包含 Python ${REQUIRE_PYTHON} 的镜像，或通过 PYTHON_BIN 指定解释器。" >&2
    exit 1
fi

if [ -x "${VENV_DIR}/bin/python" ]; then
    echo "[venv] 已存在,跳过创建: $VENV_DIR"
elif [ -e "$VENV_DIR" ]; then
    echo "[FATAL] venv 目录存在但不完整: ${VENV_DIR}" >&2
    echo "        请人工检查后移走该目录，再重新运行。" >&2
    exit 1
else
    echo "[venv] 创建中: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# 激活的本质:把 venv/bin 插到 PATH 最前面。
# activate 脚本会引用 PS1 等可能未定义的变量,set -u 下会报错,故临时关闭。
set +u
source "${VENV_DIR}/bin/activate"
set -u

python -m pip install --upgrade pip setuptools wheel

echo "[venv] python: $(which python)"
echo "[venv] pip   : $(which pip)"

# ---------- Dependencies ----------
# 新 venv 默认不会继承镜像中的 torch，因此必须从官方 cu128 index
# 显式安装。driver 显示 CUDA 13.2、torch runtime 为 12.8 属正常。
python -m pip install \
    "torch==2.11.0" \
    --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
    "transformers==4.55.4" \
    "tokenizers==0.21.4" \
    "numpy==2.5.2" \
    "safetensors==0.8.0" \
    "huggingface-hub==0.36.2"

# 原机器没有 accelerate；它不是当前直接 .to(cuda) 加载路径的必需包。
if python -m pip show accelerate >/dev/null 2>&1; then
    echo "[FATAL] 当前 venv 中安装了 accelerate，但原机器未安装。" >&2
    echo "        请使用一个干净 venv，避免可选依赖改变加载路径。" >&2
    exit 1
fi

export REQUIRE_TORCH="2.11.0+cu128"
export REQUIRE_CUDA="12.8"
export REQUIRE_CUDNN="91900"
export REQUIRE_SM="sm_120"

python - <<'PY'
import os, platform, sys, torch
from importlib.metadata import PackageNotFoundError, version

want_torch = os.environ["REQUIRE_TORCH"]
want_cuda  = os.environ["REQUIRE_CUDA"]
want_cudnn = int(os.environ["REQUIRE_CUDNN"])
want_sm    = os.environ["REQUIRE_SM"]

expected = {
    "transformers": "4.55.4",
    "tokenizers": "0.21.4",
    "numpy": "2.5.2",
    "safetensors": "0.8.0",
    "huggingface-hub": "0.36.2",
}

print(f"[deps] python       : {platform.python_version()}")
print(f"[deps] torch        : {torch.__version__}")
print(f"[deps] torch cuda   : {torch.version.cuda}")
print(f"[deps] cuDNN        : {torch.backends.cudnn.version()}")
print(f"[deps] arch list    : {torch.cuda.get_arch_list()}")

if torch.__version__ != want_torch:
    sys.exit(f"[FATAL] torch {torch.__version__} != {want_torch}")

if torch.version.cuda != want_cuda:
    sys.exit(f"[FATAL] torch CUDA runtime {torch.version.cuda} != {want_cuda}")

if torch.backends.cudnn.version() != want_cudnn:
    sys.exit(f"[FATAL] cuDNN {torch.backends.cudnn.version()} != {want_cudnn}")

if not torch.cuda.is_available():
    sys.exit("[FATAL] CUDA not available")

if want_sm not in torch.cuda.get_arch_list():
    sys.exit(f"[FATAL] {want_sm} kernels missing — Blackwell 会 fallback,性能数据不可信")

for package, wanted in expected.items():
    try:
        actual = version(package)
    except PackageNotFoundError:
        sys.exit(f"[FATAL] missing package: {package}")
    print(f"[deps] {package:16s}: {actual}")
    if actual != wanted:
        sys.exit(f"[FATAL] {package} {actual} != {wanted}")

n = torch.cuda.device_count()
print(f"[deps] gpu count    : {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"[deps] gpu{i}: {p.name} | sm_{p.major}{p.minor} | {p.total_memory/1024**3:.1f} GB")

if not torch.backends.cuda.flash_sdp_enabled():
    sys.exit("[FATAL] PyTorch SDPA Flash backend is disabled")
PY

LOCK_FILE="${PERSIST_ROOT}/requirements.lock.txt"
python -m pip freeze | LC_ALL=C sort > "$LOCK_FILE"
echo "[deps] 完整依赖已冻结到 ${LOCK_FILE}"

# ---------- Repo ----------
# 锁死 commit:上游 main 会动。不锁的话不同机器 clone 到不同代码,
# 跨机器数据不可比,而且不会报任何错。
REPO_URL="${REPO_URL:-https://github.com/ML-GSAI/LLaDA.git}"
REPO_COMMIT="${REPO_COMMIT:-9182493720ed723ef8031210d85959364e51cbe0}"

if [ -d "${REPO_DIR}/.git" ]; then
    echo "[repo] 已存在,跳过 clone"
    if [ -n "$(git -C "$REPO_DIR" status --porcelain)" ]; then
        echo "[FATAL] checkout 前发现仓库有未提交改动，拒绝覆盖。" >&2
        git -C "$REPO_DIR" status --short >&2
        exit 1
    fi
elif [ -e "$REPO_DIR" ]; then
    echo "[FATAL] repo 路径存在但不是 Git 仓库: ${REPO_DIR}" >&2
    exit 1
else
    echo "[repo] clone: ${REPO_URL}"
    git clone "$REPO_URL" "$REPO_DIR"
fi

git -C "$REPO_DIR" fetch --all --quiet
git -C "$REPO_DIR" checkout --quiet "$REPO_COMMIT"

ACTUAL_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "$REPO_COMMIT" ]; then
    echo "[FATAL] commit 不匹配: ${ACTUAL_COMMIT} != ${REPO_COMMIT}" >&2
    exit 1
fi

# 工作区必须干净,否则实际跑的代码 ≠ 这个 commit 描述的代码
if [ -n "$(git -C "$REPO_DIR" status --porcelain)" ]; then
    echo "[FATAL] 工作区有未提交改动,数据不可复现" >&2
    git -C "$REPO_DIR" status --short >&2
    exit 1
fi

echo "[repo] commit: $(git -C "$REPO_DIR" log -1 --oneline)"

# ---------- Model weights ----------
# UltraLLaDA 用 trust_remote_code,建模代码(modeling_llada.py)随权重
# 一起从 HF 下载,既不在 LLaDA repo 里也不在 transformers 里。
# 因此必须锁 revision,否则真正跑的建模代码是浮动的。
HF_REPO="${HF_REPO:-relaxe-system-lab/UltraLLaDA}"
HF_REVISION="${HF_REVISION:-a02bb0490ddbe810bc3ac0deb667ed8fac2e8d47}"
MODEL_PATH="${MODEL_DIR}/UltraLLaDA"
COMPLETE_MARKER="${MODEL_PATH}/.download_complete.${HF_REVISION}"

MODEL_FILES=(
    config.json
    modeling_llada.py
    model.safetensors.index.json
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
    tokenizer.json
)

MODEL_IS_COMPLETE=1
for f in "${MODEL_FILES[@]}"; do
    if [ ! -f "${MODEL_PATH}/${f}" ]; then
        MODEL_IS_COMPLETE=0
        break
    fi
done

if [ -f "$COMPLETE_MARKER" ] && [ "$MODEL_IS_COMPLETE" -eq 1 ]; then
    echo "[model] 已完整下载,跳过: ${MODEL_PATH}"
else
    echo "[model] 下载 ${HF_REPO} @ ${HF_REVISION}"
    hf download "$HF_REPO" \
        --revision "$HF_REVISION" \
        --local-dir "$MODEL_PATH"
    # 只在下载成功返回后才打标记:
    # 中途断掉留下的残缺目录不会被误判为完整
    printf '%s@%s\n' "$HF_REPO" "$HF_REVISION" > "$COMPLETE_MARKER"
fi

for f in "${MODEL_FILES[@]}"; do
    [ -f "${MODEL_PATH}/${f}" ] || { echo "[FATAL] 缺失 ${f}" >&2; exit 1; }
done

echo "[model] path : ${MODEL_PATH}"
echo "[model] size : $(du -sh "$MODEL_PATH" | cut -f1)"
echo "[model] modeling_llada.py sha256:"
sha256sum "${MODEL_PATH}/modeling_llada.py"

# ---------- Activation helper ----------
# bootstrap.sh 是子进程，无法永久改变调用它的 shell；生成一个统一入口，
# 以后只需 source 这一个文件即可同时激活 venv 和 LLaDA PYTHONPATH。
ACTIVATE_FILE="${WORK_ROOT}/activate_experiment.sh"
cat > "$ACTIVATE_FILE" <<EOF
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
export ULTRALLADA_MODEL_PATH="${MODEL_PATH}"
EOF
chmod 600 "$ACTIVATE_FILE"

# ---------- Done ----------
cat <<EOF

============================================================
环境就绪。

当前 shell 请使用统一入口激活:
  source ${ACTIVATE_FILE}

模型路径:
  ${MODEL_PATH}

复现锁文件:
  ${LOCK_FILE}
============================================================
EOF
