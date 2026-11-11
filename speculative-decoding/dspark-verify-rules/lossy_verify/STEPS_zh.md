# 执行顺序（对应代码方案 §3）

全部在 5090 上、`/workspace` 下、`venv-deepspec` 激活后执行。目录布局：

```
/workspace/DeepSpec/        官方仓库，commit 005e03b，branch lossy-verify-hook 只含 patch
/workspace/lossy_verify/    本包
/workspace/runs/            所有输出
```

## 安装

```bash
cd /workspace/DeepSpec
git checkout -b lossy-verify-hook
git apply /workspace/lossy_verify/lossy-verify-hook.patch
git diff --stat                      # 应为 2 files changed, 29 insertions(+), 1 deletion(-)
git commit -am "lossy-verify hook: accept_rule in verify_draft_tokens; draft_logits in DSparkDraftProposal"

cd /workspace
python -m lossy_verify.metrics --build       # 生成 lossy_verify/data/gsm8k_with_answers.jsonl，带逐行题面断言
```

## 步 1 回归：patch 后 accept_rule=None 与原始 eval.py 逐 token 相同

```bash
# 1a 原始代码（main）跑 20 题
cd /workspace/DeepSpec && git stash -q 2>/dev/null; git checkout -q main
python - <<'EOF'
import sys, json, torch
sys.argv = ["eval.py","--target_name_or_path","Qwen/Qwen3-8B","--draft_name_or_path","deepseek-ai/dspark_qwen3_8b_block7","--temperature","0","--max-new-tokens","512"]
import eval as E
from deepspec.eval.dspark import Qwen3DSparkEvaluator
args = E.parse_args(); args.tasks = [("gsm8k", 20)]
ev = Qwen3DSparkEvaluator(0, args)
resps = ev.run_dataset(dataset_name="gsm8k", max_samples=20)
json.dump([r.output_ids[0].tolist() for r in resps], open("/workspace/runs/step1_master_ids.json","w"))
ev.clean_up()
EOF
# 1b patch 后同样 20 题
git checkout -q lossy-verify-hook
python - <<'EOF'
import sys, json
sys.argv = ["eval.py","--target_name_or_path","Qwen/Qwen3-8B","--draft_name_or_path","deepseek-ai/dspark_qwen3_8b_block7","--temperature","0","--max-new-tokens","512"]
import eval as E
from deepspec.eval.dspark import Qwen3DSparkEvaluator
args = E.parse_args(); args.tasks = [("gsm8k", 20)]
ev = Qwen3DSparkEvaluator(0, args)
resps = ev.run_dataset(dataset_name="gsm8k", max_samples=20)
json.dump([r.output_ids[0].tolist() for r in resps], open("/workspace/runs/step1_patched_ids.json","w"))
ev.clean_up()
EOF
python -c "import json;a=json.load(open('/workspace/runs/step1_master_ids.json'));b=json.load(open('/workspace/runs/step1_patched_ids.json'));print('identical' if a==b else 'DIFFER', sum(x==y for x,y in zip(a,b)),'/',len(a))"
```

过关：`identical 20 / 20`。

## 步 2 回归：strict 规则经 hook 与步 1 逐 token 相同

```bash
cd /workspace
python -m lossy_verify.run --rule strict --n 20 --out runs/step2_strict20
python - <<'EOF'
import json
a = json.load(open("/workspace/runs/step1_patched_ids.json"))
b = [json.loads(l) for l in open("/workspace/runs/step2_strict20/samples.jsonl")]
# step1 保存的是含 prompt 的完整 output_ids；samples.jsonl 只存生成部分，按尾部比对
ok = sum(1 for x, r in zip(a, b) if x[-len(r["output_ids"]):] == r["output_ids"] and len(x) - len(r["output_ids"]) > 0)
print("identical" if ok == len(b) else "DIFFER", ok, "/", len(b))
EOF
```

过关：`identical 20 / 20`。（`samples.jsonl` 顺序 = 步 1 的样本顺序，都是同一 seed 打乱后前 20。）

## 步 3 卡 0 arm ①：strict n=200 跑两遍

```bash
python -m lossy_verify.run --rule strict --n 200 --out runs/strict
python -m lossy_verify.run --rule strict --n 200 --out runs/strict_rerun
python -m lossy_verify.analyze runs/ --compare strict strict_rerun     # 200/200 identical
python -m lossy_verify.analyze runs/                                    # τ、accuracy
python -m lossy_verify.analyze runs/ --hist strict                      # c_k / r_i / p_D 直方图 → 定卡 A 的 θ
```

过关：两遍 200/200 一致；τ 与 DSpark 论文 gsm8k 值差 ≤ 0.2。不一致 → 加 `--deterministic` 重跑两遍。

## 步 4 卡 0 arm ②：ASD (B, g, M) = (8, 0.25, 2)

```bash
python -m lossy_verify.run --rule asd --g 0.25 --n 200 --out runs/asd_g0.25
python -m lossy_verify.analyze runs/
```

过关：τ 与 ASD 论文 DSpark-8B-block7 gsm8k 值差 ≤ 0.2。**跑之前先在 `lossy_verify/context.py` 确认 `Q_INCLUDES_SELF`。**

## 步 5–6 卡 A

```bash
bash lossy_verify/sweep.sh asd 0 0.1 0.25 0.5 1.0
bash lossy_verify/sweep.sh a1  <θ 由步 3 直方图定>
bash lossy_verify/sweep.sh a3  <同上>
bash lossy_verify/sweep.sh p   <由 p_D 直方图定>
G=0.25 bash lossy_verify/sweep.sh a2 <3 个 θ>        # 再换 G 跑两组
python -m lossy_verify.analyze runs/
python -m lossy_verify.analyze runs/ --accepted-pos
```

## 每个 run 目录的产物

| 文件 | 内容 |
|---|---|
| config.json | 全部参数、规则名、Q_INCLUDES_SELF、DeepSpec commit、torch 版本、GPU |
| summary.json | τ、accuracy_strict / accuracy_fallback、撞限数、抽取回退数、平均输出长度、target forward / token、tokens/s、examined / accepted mismatch 数、recorder 校准（仅 strict 有效） |
| samples.jsonl | 每题：id、生成文本、抽取答案、gold、对错、输出长度、轮数、acceptance_lengths、耗时、output_ids |
| mismatches.jsonl | 每个被 verify 看到的 mismatch 一行，字段 = proposal §4 |
| positions.jsonl | 每轮每个位置一行：strict_match、c_k、r_i、q_i、p_T、p_D |
