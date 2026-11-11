# LLaDA 单数字价格检索任务

这套数据用于初步检验：从不同 Transformer 层开始阻断关键证据位置后，LLaDA-8B-Instruct 是否仍能恢复答案。当前只完成了数据构造和核验，尚未运行模型或施加干预。

每题包含四种商品的价格和普通商店背景描述，询问其中一种商品的价格。四种价格互不相同，答案在输入中只出现一次。商品名、干扰价格、背景和运行顺序按固定随机种子生成。

答案使用 1–9。已用本机缓存的官方 LLaDA tokenizer 核验，这些答案都是一个 token；例如 `278` 实际是三个 token。数字答案只是受控检索探针，不能代表复杂问答或长文本生成能力。

## 直接使用的数据

| 文件 | 内容 |
|---|---|
| `data/pilot.jsonl` | 10 条试跑样本，用于先检查未干预基线和输出格式 |
| `data/eval.jsonl` | 90 条正式样本，证据前、中、后各 30 条，每个价格各 10 条 |
| `data/example.txt` | 一条完整样本及单独标明的答案和位置信息 |
| `data/manifest.json` | tokenizer 指纹、随机种子、数据统计和核验记录 |

每个 JSONL 文件一行一题。**实际模型输入使用 `input_ids`，不要把整行 JSON 当成 prompt。** 标准答案只作为评分元数据，不能拼到模型输入中。

所有样本均已套用该 tokenizer 的官方聊天模板，末尾附加一个 `<|mdm_mask|>`。实际输入总长度为 501–512 tokens，包含模板和这个 mask。证据数字位于正文 token 长度约 20%、50%、80% 处，实际位置和目标位置的最大差距约为 1.37 个百分点。句子完整保留，没有强行补齐为相同长度。

## 关键字段

| 字段 | 用法 |
|---|---|
| `prompt` / `messages` | 尚未套模板的用户问题；需要查看或重新分词时使用 |
| `input_ids` | 可直接送给模型的完整输入，已经包含模板和一个 mask |
| `answer` / `answer_token_id` | 标准答案及单个 token ID，仅用于评分 |
| `answer_position` | 要读取 logits 的 mask 位置 |
| `critical_token_positions` | prompt 中答案数字的位置，后续干预此处 |
| `control_token_positions` | 一个非目标商品的价格数字，作为等量干预对照 |
| `evidence_text` / `control_text` | 对应证据句，方便人工核对 |
| `position_group` / `actual_context_depth` | 前中后分组及实际正文位置比例 |

token 位置均从 0 开始，相对于完整 `input_ids`，已经计入聊天模板。字符区间则相对于原始 `prompt`，采用 `[start, end)`。若修改模板、输入内容或在前面加 padding，必须重新定位，不能继续使用旧索引。

## 先试跑一条

在已经加载 `GSAI-ML/LLaDA-8B-Instruct` 的 Python 环境中运行以下代码。这里 `model` 是已有模型对象；建议先使用单条输入，避免 padding 改变位置。

```python
import json
from pathlib import Path
import torch

row = json.loads(Path("data/pilot.jsonl").read_text(encoding="utf-8").splitlines()[0])
x = torch.tensor([row["input_ids"]], dtype=torch.long, device=model.device)

model.eval()
with torch.inference_mode():
    logits = model(x, use_cache=False).logits
    z = logits[0, row["answer_position"]].float()
    predicted_id = z.argmax().item()
    correct_probability = z.softmax(-1)[row["answer_token_id"]].item()

print("gold:", row["answer"])
print("correct:", predicted_id == row["answer_token_id"])
print("P(gold):", correct_probability)
```

从 `retrieval_task` 目录执行，或自行修改数据文件路径。这里是单次前向探针，答案位置取同一位置的 logits，不做自回归的前移。不要调用生成函数再追加一段 mask，也不要再次给 `input_ids` 套聊天模板。若需要核对模板，可确认下面的结果与 `row['input_ids'][:-1]` 一致：

```python
assert tokenizer.apply_chat_template(
    row["messages"], tokenize=True, add_generation_prompt=True
) == row["input_ids"][:-1]
```

先看全部 10 条 pilot 的准确率。如果完整基线表现差，应先检查模板、位置和探针形式，不能直接把低准确率解释为信息扩散失败。正式比较使用固定的 90 条 eval；各干预条件使用同一批输入，保留全部样本，按题做配对比较。不同层、去噪步和对照结果是同一题的重复测量，不是额外的独立样本。

## 重新生成

生成器只读取分词文件，不下载或加载模型权重。在通常已经装有 transformers 的推理环境中，tokenizers 也已安装。本次构造使用 `tokenizers==0.22.1`。

```bash
python generate_task.py --tokenizer-dir /path/to/LLaDA-8B-Instruct --out data_new --seed 42
```

`--tokenizer-dir` 中需要 `tokenizer.json` 和 `tokenizer_config.json`。已保存的数据来自缓存版本 `08b83a6feb34df1a6011b80c3c00c7563e963b07`，精确文件指纹见 manifest。生成器核验官方聊天模板，发现不同模板会报错。可用 `--max-input-tokens 1024` 改变长度上限。生成器拒绝覆盖已存在的数据文件。

后续最小比较为：完整基线、从第 0/4/24 层开始持续屏蔽关键位置，以及在相同层屏蔽 `control_token_positions` 的对照。层编号从 0 开始；干预代码不属于当前的数据构造脚本。只屏蔽 attention 不能用于报告实际 token pruning 的速度收益。
