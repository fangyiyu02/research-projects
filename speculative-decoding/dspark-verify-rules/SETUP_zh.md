# 新机器从零配置

所有命令在 /workspace 下执行  每一步看到期望输出再往下  出错就停下贴给我

---

## 0 先检查 GPU 是空的

```bash
nvidia-smi
```

| 看哪里 | 期望 |
|---|---|
| GPU 名称 | NVIDIA GeForce RTX 5090 |
| Driver Version | ≥ 570 |
| Memory-Usage | 接近 0  例如 1MiB / 32607MiB |

**Memory-Usage 不接近 0 就直接换机器 不要往下装**

---

## 1 环境变量和 git

```bash
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
echo 'export HF_HUB_OFFLINE=1' >> ~/.bashrc
source ~/.bashrc
```

---

## 2 上传并解压代码

在 Jupyter 文件浏览器里把 code_1102.tar.gz 上传到 /workspace

```bash
cd /workspace
tar xzf code_1102.tar.gz
ls /workspace
```

期望 /workspace 下直接看到 lossy_verify  idea_b  deepspec_hooks.patch

如果它们在一个叫 code_1102 的文件夹里 先移出来

```bash
mv code_1102/* /workspace/ && rmdir code_1102
ls /workspace
```

---

## 3 Python 环境

```bash
cd /workspace
python3 -m venv venv-deepspec
source venv-deepspec/bin/activate
pip install -U pip
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

---

## 4 DeepSpec 和源码改动

```bash
cd /workspace
git clone https://github.com/deepseek-ai/DeepSpec
cd DeepSpec
git checkout 005e03b
git checkout -b lossy-verify-hook
git apply ../deepspec_hooks.patch
git diff --stat
```

期望 2 files changed, 30 insertions(+), 1 deletion(-)

```bash
git commit -am "lossy-verify hook (Idea A) + full target logits (Idea B)"
git log --oneline -2
cd /workspace
```

期望第一行是刚才的提交  第二行是 005e03b
**如果第一行是 005e03b 说明 patch 没打上 停下**

```bash
pip install -r DeepSpec/requirements.txt
pip install "huggingface_hub[cli]>=1.5.0,<2.0"
python -c "import torch, transformers, huggingface_hub; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), transformers.__version__, huggingface_hub.__version__)"
```

期望 2.9.1+cu128 12.8 True NVIDIA GeForce RTX 5090 5.10.2 1.x.x

---

## 5 下载模型和 gsm8k 答案

.bashrc 里设了离线模式 下载时临时关掉

```bash
cd /workspace
export HF_HUB_OFFLINE=0
hf download Qwen/Qwen3-8B
hf download deepseek-ai/dspark_qwen3_8b_block7
python -m lossy_verify.metrics --build
export HF_HUB_OFFLINE=1
ls hf/hub
ls lossy_verify/data
```

| 期望 |
|---|
| metrics 最后一行显示 1319 rows |
| hf/hub 里有 models--Qwen--Qwen3-8B 和 models--deepseek-ai--dspark_qwen3_8b_block7 |
| lossy_verify/data 里有 gsm8k_with_answers.jsonl |

---

## 6 测试

CPU 测试 几秒

```bash
cd /workspace && source venv-deepspec/bin/activate
python lossy_verify/test_rules_cpu.py
python -m idea_b.test_ideab_cpu
```

期望 rules OK  logger OK  metrics OK  rule OK  missing-hook check OK

GPU smoke test 约 1 分钟

```bash
mkdir -p runs runs_b
python -m lossy_verify.run --rule strict --n 20 --out runs/step2_strict20
python -m idea_b.run --theta -1 --n 20 --out runs_b/smoke_ideab20
python -m lossy_verify.analyze runs/ --compare step2_strict20 ../runs_b/smoke_ideab20
```

| 检查 | 期望 |
|---|---|
| compare | 20/20 samples identical |
| 原始规则 20 题 τ | 6.552  与旧机器一致 |
| 原始规则 20 题正确率 | 0.900 |

---

## 7 重跑基准和实验

旧机器的逐条日志已丢 汇总数字在 README 和项目记录里
用 tmux 跑 vast 终端默认已在 tmux 里  离开按 Ctrl+B 再按 D
**每组跑完立刻打包 并在 Jupyter 里下载到本地**

### 7.1 基准 约 35 分钟

```bash
python -m lossy_verify.run --rule strict --n 200 --out runs/strict
python -m lossy_verify.ar_baseline --n 200 --out runs/ar
python -m idea_b.run --theta -1 --n 200 --out runs_b/ideab_theta-1
python -m lossy_verify.analyze runs/ --compare strict ../runs_b/ideab_theta-1
python -m idea_b.tv_quantiles runs_b/ideab_theta-1
tar czf backup_A.tar.gz runs runs_b
```

期望 原始规则 τ 6.475 正确率 0.920  compare 200/200

### 7.2 Idea B 扫 θ 约 35 分钟

```bash
bash idea_b/sweep.sh 0.001 0.01 0.05 0.1 0.3 0.75 0.99
tar czf backup_B.tar.gz runs runs_b
```

### 7.3 ASD 约 25 分钟

```bash
bash lossy_verify/sweep.sh asd 0 0.1 0.25 0.5 1.0
tar czf backup_C.tar.gz runs runs_b
```

### 7.4 出表

```bash
python -m lossy_verify.analyze runs/
python -m lossy_verify.analyze runs_b/
python -m lossy_verify.speedup --ar runs/ar --runs runs/strict runs/asd_g* runs_b/ideab_theta*
```

把这三张表贴出来
