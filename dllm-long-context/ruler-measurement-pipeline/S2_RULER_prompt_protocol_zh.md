# S2 RULER Prompt生成与冻结协议 v1.1

## 1. 目标

本阶段只生成 UltraLLaDA 使用的 RULER prompt，不进行模型推理或计时。

核心要求：

1. 使用固定版本的 UltraLLaDA tokenizer。
2. prompt以最终 `input_ids` 形式落盘。
3. 实际输入token误差不超过0.2%，且最多±8 token。
4. 一次生成并冻结，后续实验禁止重新随机生成。
5. 效率测试与准确率测试使用不同的长度密度。
6. NIAH needle depth采用预先固定的分层安排。

本阶段不生成Llama版本，不处理LongBench、HELMET、NoLiMa或InfiniteBench。

---

## 2. 固定版本

### RULER

使用全新克隆的 NVIDIA 官方RULER仓库。

生成时记录：

```text
repository_url
git_commit
git_branch
git_dirty
```

要求：

```text
repository_url = https://github.com/NVIDIA/RULER.git
git_dirty = false
```

如果仓库存在未提交修改，生成器拒绝运行。

### Tokenizer

固定使用：

```text
tokenizer_id:
GSAI-ML/LLaDA-8B-Instruct

tokenizer_revision:
08b83a6feb34df1a6011b80c3c00c7563e963b07
```

同时记录：

```text
resolved_tokenizer_revision
tokenizer_config_sha256
chat_template_sha256
```

禁止在生成完成后更新tokenizer。

---

## 3. Prompt长度定义

文件名中的长度表示：

> 完整chat template、任务指令、上下文、问题及特殊token处理完成后，最终送入模型的 `input_ids` 长度。

不包含模型输出token。

生成流程：

```text
RULER任务实例
→ 调整task-native filler
→ 放置needle
→ apply_chat_template(add_generation_prompt=True)
→ tokenize(add_special_tokens=False)
→ 检查最终input_ids长度和needle位置
→ 保存
```

运行阶段直接加载保存的token IDs，禁止再次：

- 调用tokenizer；
- 添加BOS/EOS；
- 应用chat template；
- 拼接新的提示词；
- 截断或padding prompt。

---

## 4. 长度容差

采用：

```text
allowed_error = min(8, floor(target_tokens × 0.002))
```

| Target | 最大误差 |
|---:|---:|
| 1024 | ±2 |
| 2048 | ±4 |
| 4096及以上 | ±8 |

每条prompt必须满足：

```text
abs(exact_input_tokens - target_input_tokens) <= allowed_error
```

显存和时间拟合使用 `exact_input_tokens`，不能直接使用文件名中的nominal length。

生成器应优先精确命中目标，容差只是最大允许范围。

---

## 5. 长度调整方法

采用两阶段搜索：

1. 对filler单元数量执行二分搜索。
2. 在二分结果附近执行局部搜索，选择误差最小的候选。

不能只使用纯二分，因为BPE边界合并可能导致字符数和token数不严格单调。

只允许调整任务中的无关filler：

- NIAH：noise、essay或needle型干扰内容；
- VT：无关变量或噪声；
- CWE/FWE：非答案词；
- QA：非证据distractor documents。

禁止修改或截断：

- 问题；
- 正确答案；
- needle；
- 必要证据；
- VT关系链；
- CWE/FWE目标词；
- prompt template。

如果无法在容差内生成合法prompt，必须显式报错，不能静默降低任务难度。

---

## 6. 效率测试集

### 任务

```text
niah_single_1
```

### 长度

```text
1024
2048
4096
6144
8192
12288
16384
24576
32768
49152
65536
98304
131072
```

### 样本数

```text
N = 10 / length
```

同一组10个 `sample_id` 贯穿全部13个长度。

同一sample在不同长度下保持：

- 相同key；
- 相同value；
- 相同答案；
- 相同needle depth；
- 相同语义seed。

只改变filler长度。

### Needle depth分配

使用五个固定深度：

```text
10%, 30%, 50%, 70%, 90%
```

每个深度2条：

```text
10% × 2
30% × 2
50% × 2
70% × 2
90% × 2
```

因此效率集不是全部固定在50%，而是对上下文位置进行平衡覆盖。

---

## 7. Accuracy-under-compression测试集

### 任务

生成全部13个RULER v1任务：

```text
niah_single_1
niah_single_2
niah_single_3
niah_multikey_1
niah_multikey_2
niah_multikey_3
niah_multivalue
niah_multiquery
vt
cwe
fwe
qa_1
qa_2
```

### 长度

```text
4096
8192
16384
32768
65536
131072
```

不为准确率任务生成6K、12K、24K、48K或96K。

### 样本数

冻结池：

```text
N = 50 / task / length
```

初次实验只运行：

```text
N = 10 / task / length
```

如果10条不足以形成稳定结论，再继续运行已经冻结的第11–50条，禁止重新生成样本。

### NIAH深度分层

对于每个NIAH任务、每个长度，N=50按以下方式平衡：

```text
10% × 10
30% × 10
50% × 10
70% × 10
90% × 10
```

初次运行的10条也必须保持平衡：

```text
10% × 2
30% × 2
50% × 2
70% × 2
90% × 2
```

不能让初次运行的10条全部来自同一needle depth。

`niah_single_1` 标准六个长度下的平衡样本同时供效率与accuracy使用，不重复生成。

---

## 8. Needle depth定义

needle depth只相对于haystack计算，不相对于完整prompt计算。

定义：

```text
actual_needle_depth =
tokens_before_needle /
(tokens_before_needle + tokens_after_needle)
```

计算时不包括：

- system/user chat template；
- RULER任务说明；
- 最后的question；
- generation prompt；
- needle本身的token。

单needle任务记录：

```text
target_needle_depth
actual_needle_depth
needle_token_start
needle_token_end
```

多needle任务记录：

```text
target_needle_depths
actual_needle_depths
needle_token_spans
```

多needle任务禁止把全部needle放在同一位置。所有相关needle的位置必须确定性生成并完整记录。

实际needle depth与目标值的误差：

```text
abs(actual_needle_depth - target_needle_depth) <= 0.01
```

即最多偏差1个百分点。

长度搜索过程中必须同时维持needle depth。不能为了满足token长度而把needle从10%移动到20%等其它位置。

---

## 9. 随机种子

使用S1规定的全局seed。

每条语义实例的随机性由以下字段确定性派生：

```text
global_seed
task
sample_id
generator_version
```

语义seed不包含长度，因此相同 `task + sample_id` 在所有长度下保持相同任务身份。

needle depth通过固定分层表分配，分层内部顺序使用全局seed进行可复现随机化。

必须保证：

- 每10条样本包含五个深度各2条；
- 每50条样本包含五个深度各10条；
- 调整任务执行顺序不会改变分配结果；
- 增加新长度不会改变已有样本。

---

## 10. 文件结构

```text
prompts/
└── ruler_v1/
    ├── tokens/
    │   ├── niah_single_1/
    │   │   ├── 1024.pt
    │   │   ├── 2048.pt
    │   │   └── ...
    │   ├── vt/
    │   │   ├── 4096.pt
    │   │   └── ...
    │   └── ...
    ├── manifests/
    │   ├── samples.jsonl
    │   ├── efficiency.jsonl
    │   └── accuracy.jsonl
    ├── generation_config.json
    ├── checksums.sha256
    └── GENERATION_PROTOCOL.md
```

每个 `.pt` 文件保存：

```python
List[torch.LongTensor]
```

每个tensor必须满足：

```text
dtype = torch.int64
device = cpu
ndim = 1
```

文件名使用完整整数，不使用`4k`等缩写。

---

## 11. 元数据

每条样本至少记录：

```text
schema_version
generator_version
task
sample_id
global_seed
derived_seed
target_input_tokens
exact_input_tokens
length_error
allowed_error
answer
generation_budget
filler_type
filler_units
target_needle_depth
actual_needle_depth
needle_token_start
needle_token_end
tokenizer_id
tokenizer_revision
tokenizer_config_sha256
chat_template_sha256
ruler_repository_url
ruler_git_commit
ruler_task_config
ruler_task_config_sha256
tensor_sha256
created_at_utc
```

非NIAH任务允许将needle字段设为 `null`。

RULER输出预算不计入 `target_input_tokens`，但必须单独记录为 `generation_budget`。

---

## 12. 外部数据固定

生成essay NIAH和QA前，需要准备：

```text
PaulGrahamEssays.json
squad.json
hotpotqa.json
```

必须记录：

```text
source_url
download_date
file_size
sha256
```

开始正式生成后，禁止更换数据版本。

---

## 13. 原子写入和禁止覆盖

生成器先写入：

```text
<filename>.tmp
```

完成长度、needle depth、数量、dtype和校验和验证后，再原子改名为正式文件。

正式文件已存在时，生成器默认拒绝覆盖。

修改协议时必须创建新版本目录，例如：

```text
ruler_v2/
```

禁止覆盖 `ruler_v1/`。

---

## 14. 最终验收

冻结前必须自动验证：

- RULER仓库来源正确且工作区干净；
- tokenizer revision完全匹配；
- 所有预期文件存在；
- `.pt`样本数量正确；
- tensor均为CPU LongTensor；
- token长度误差在容差内；
- needle depth误差不超过1个百分点；
- 每10条NIAH样本包含五个深度各2条；
- 每50条NIAH样本包含五个深度各10条；
- 同一sample跨长度的答案和depth一致；
- 问题、答案、needle和必要证据均被保留；
- `sample_id`没有重复；
- manifest长度与tensor实际长度一致；
- 所有SHA-256校验通过；
- efficiency manifest只选择规定的10条；
- accuracy manifest只选择标准六个长度；
- 旧RULER数据没有混入新集合。

全部通过后生成：

```text
checksums.sha256
generation_complete.lock
```

存在 `generation_complete.lock` 后，生成器和实验代码都不得修改prompt集合。

---

## 15. 本阶段明确不做

S2不包含：

- Llama tokenizer版本；
- 模型推理；
- latency或显存计时；
- accuracy评分；
- SDPA backend检测；
- float64；
- `torch.use_deterministic_algorithms`；
- LongBench、HELMET、NoLiMa或InfiniteBench生成；
- 复用此前生成的JSONL。

S2的唯一输出是经过验证并冻结的RULER token-ID prompt集合及完整元数据。