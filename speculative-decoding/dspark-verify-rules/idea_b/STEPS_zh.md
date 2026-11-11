# Idea B 执行步骤

## 目录

```
/workspace/DeepSpec        官方仓库 分支 lossy-verify-hook
/workspace/lossy_verify    Idea A  不做任何改动  idea_b 只读复用其中的 evaluator metrics logger analyze speedup
/workspace/idea_b          本包
/workspace/runs_b          Idea B 的输出  与 Idea A 的 runs 分开
```

## idea_b 里的文件

| 文件 | 内容 |
|---|---|
| rule.py | Idea B 规则 每个不一致都用 Markov head 判断  tv_est ≤ θ 放过 继续往后  否则截断  不用 r_i 没有 M 和 B |
| logger.py | 继承 Idea A 的日志类 只多记一列 tv_est |
| run.py | 单次运行入口 输出格式与 Idea A 相同 |
| sweep.sh | 扫 θ |
| tv_quantiles.py | 看一次运行里 tv_est 的分布 |
| test_ideab_cpu.py | CPU 测试 不需要 GPU |

## 第 0 步 DeepSpec 加一行

`deepspec/eval/base_evaluator.py` 的 hook 里 `SimpleNamespace(...)` 中
在 `target_logits=target_output.logits[:, :-1, :],` 下面加

```python
                    target_logits_all=target_output.logits,
```

原因 判断第 k 位的不一致要用第 k+1 位的打分 块内最后一位的下一位是 bonus 位置 原来没有传出来
不开规则时 hook 不执行 Idea A 的规则不读这个字段 所以对已有结果没有影响

```bash
cd /workspace/DeepSpec
git diff          # 只应看到这一行
git commit -am "hook: also pass full target logits incl. bonus position (Idea B)"
```

## 第 1 步 解压与测试

```bash
cd /workspace
tar xzf idea_b.tar.gz
source venv-deepspec/bin/activate
python -m idea_b.test_ideab_cpu      # rule OK  logger OK  missing-hook check OK
```

## 第 2 步 回归 θ = −1 永不放过 应与原始规则逐 token 一致

```bash
python -m idea_b.run --theta -1 --n 200 --out runs_b/ideab_theta-1
python -m lossy_verify.analyze runs/ --compare strict ../runs_b/ideab_theta-1
```

过关 200/200 identical
这一次同时把每个不一致的 tv_est 记下来 下一步用

## 第 3 步 看 tv_est 分布 定 θ 扫描点

```bash
python -m idea_b.tv_quantiles runs_b/ideab_theta-1
```

把输出贴出来 一起定 θ

## 第 4 步 扫 θ

```bash
bash idea_b/sweep.sh <θ 列表>
python -m lossy_verify.analyze runs_b/
python -m lossy_verify.speedup --ar runs/ar --runs runs/strict runs/asd_g* runs_b/ideab_theta*
```
