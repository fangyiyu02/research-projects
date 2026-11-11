# S3 UltraLLaDA 统一 Chunked Runtime 测量协议 v2.0-draft

## 0. 状态与目的

本文件是 Stage 3 的预注册式测量协议。正式代码只能在本协议、机器可读配置、方法清单和运行顺序冻结后使用；正式数据只能由冻结版本产生。

当前状态：`draft_pending_method_and_environment_freeze`。

本版本的核心决定是：**所有 no-cache 与 cache 方法必须使用同一个 chunked 执行底座**。Chunking 是共同的系统实现，不是某个方法独享的优化。主 baseline 定义为 `chunked_no_cache`，不再使用未分块的官方式实现作为长上下文主 baseline。

Stage 3 回答：

1. 在统一 runtime 下，三个方法的实际端到端生成 latency 如何随输入长度变化？
2. 三个方法的 PyTorch、allocator 与 NVML peak memory 分别是多少？
3. cache 相对 no-cache 的收益来自更少的有效计算、缓存复用还是其它方法逻辑，而不是不同的 dtype、attention backend、chunk size 或 placement？
4. 在单卡 GPU0 上，哪个长度开始 OOM、fragile-fit 或违反模型上下文上限？

本协议测量的是统一优化后的系统比较，不声称复现“官方未修改实现”的绝对 latency 或 peak memory。

---

## 1. 冻结对象

### 1.1 模型与基本运行模式

- Model ID：`relaxe-system-lab/UltraLLaDA`
- Model revision：`a02bb0490ddbe810bc3ac0deb667ed8fac2e8d47`
- dtype：BF16
- batch size：1
- `model.eval()`
- `torch.inference_mode()`
- 单进程、单卡，物理 GPU0
- 禁止量化、CPU offload、双卡 sharding、tensor parallel 或自动 device map

正式运行前必须冻结：model code revision、tokenizer revision、PyTorch/Transformers 版本、CUDA/driver、GPU UUID、allocator backend 和所有影响 CUDA/SDPA 的环境变量。

### 1.2 Prompt bank 与总长度

使用冻结的：

`prompts/ruler_v1/efficiency/niah_single_1/`

候选长度：

`1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072`

每个长度使用相同的 10 个冻结 `sample_id`。运行时直接加载 CPU `LongTensor`，禁止重新 tokenize、增加 BOS/EOS、修改 chat template、padding、截断或拼接文本。

必须分别记录：

- `exact_prompt_tokens`
- `gen_length`
- `max_sequence_tokens = exact_prompt_tokens + gen_length`

128K 点必须在正式冻结前通过上下文上限 preflight：

- 若验证后的硬上限是 131072 total tokens，则 128K-total condition 的 prompt 必须是 130944 tokens；
- 不能把 131072-token prompt 再加 128 token 后仍静默标为“128K total”；
- 若模型明确支持并实测通过 131200 total tokens，可建立具名的 `128K_input_plus_128_output` condition，并如实记录 exact total。

UltraLLaDA tokenizer 与生成 prompt 所用 tokenizer 的 JSON、chat template、vocabulary size、`mask/eos/pad` IDs 和 token-ID 合法范围必须一致，否则停机处理。

### 1.3 统一生成配置

主实验三个方法全部使用：

- `gen_length = 128`
- `block_length = 32`
- `steps_total = 128`
- `num_blocks = 4`
- `steps_per_block = 32`
- `temperature = 0`
- `cfg_scale = 0`
- `remasking = low_confidence`
- `mask_id = 126336`
- deterministic decoding

必须验证各方法 API 中 `steps` 的语义确实是全生成过程总步数，而不是每个 block 的步数。每次 run 记录实际 `model_forward_calls`；任何意外产生 512 次 forward 的实现均判为配置错误。

32/32/32 的官方论文复现只能作为单独的 `official_generation_reference`，不能与本协议的 128/32/128 主结果混合。

### 1.4 Attention backend

所有方法强制使用 PyTorch SDPA Flash backend：

- 只允许 `FLASH_ATTENTION`；
- 禁止 math、eager、efficient 或 cuDNN 静默替代；
- 使用一次真实模型 forward 的 profiler 事件验证实际 kernel；
- backend 不可用或 attention mask 不被 Flash 支持时直接失败，不自动降级；
- 每次 session 记录实际 backend、torch/CUDA 版本和 preflight 证据。

配置字段或“Flash 可用”探针不足以证明正式模型 forward 实际走了 Flash。

### 1.5 共同 Chunked Runtime

共同 runtime ID：`shared_chunked_bf16_flash_v1`。

| 项 | 冻结值 | 实现约束 |
|---|---:|---|
| Attention | SDPA Flash only | fallback 直接失败 |
| dtype | BF16 | 所有方法和所有模块相同 |
| FFN chunk | 8192 tokens | 只沿 sequence 维切分，保持 token 顺序 |
| lm_head scope | sampler-required positions | 先选择当前采样/置信度计算真正需要的位置 |
| lm_head chunk | 512 tokens | 逐块投影、消费、释放 |
| Placement | 单卡物理 GPU0 | 禁止自动迁移或 offload |

#### Chunked FFN

1. Attention 和 residual 得到完整 hidden state 后，只对 FFN/MLP 沿 sequence 维按最多 8192 token 切分。
2. 每个 chunk 使用完全相同的 FFN 权重、激活函数、dtype 和 eval 状态。
3. 输出按原 token 顺序写入预分配 tensor 或等价结构。
4. 禁止在 chunk 边界改变 cache key、token index、位置编码或 attention 逻辑。
5. 最后一个不足 8192 token 的 chunk 使用实际长度，禁止 padding 进入正式计算。

#### Chunked lm_head

1. 在执行 output projection 之前，从 hidden state 中选出当前 sampler、argmax、confidence 或 remasking 真正会读取的位置。
2. 选中位置按最多 512 token 分块执行 lm_head。
3. 每个 logits chunk 必须立即完成 argmax、logsumexp/softmax confidence、gather 和 remasking 所需计算，然后释放。
4. 禁止把 chunks `cat` 回完整 `[B, sequence, vocab]` logits。
5. 禁止物化 prompt 位置的 vocabulary logits，除非某个方法的已冻结算法确实需要；若需要，必须另立 condition 并记录投影位置数。
6. 每次 run 记录 `lm_head_projected_tokens` 与 `lm_head_chunk_calls`。

这一路径等价于对所有方法统一施加 exact prompt-logit pruning。它属于共同 runtime，不属于 cache 方法贡献。

### 1.6 方法与 baseline 边界

主比较固定三个 algorithm conditions：

1. `chunked_no_cache`：主 baseline；每个去噪 step 对算法要求的完整序列重新计算，不保留任何跨 step KV、feature、hidden-state 或 eviction cache。
2. `chunked_cache_method_a`：待冻结的 cache 方法 A。
3. `chunked_cache_method_b`：待冻结的 cache 方法 B。

冻结前必须把 A/B 替换为真实 method ID、论文/代码 revision 和全部参数。若方法数量或方法身份变化，协议必须升版。

三种方法只能在算法本身的 cache/reuse/eviction 行为上不同。以下必须完全一致：

- prompt token IDs 与 run order；
- model weights 与 revision；
- generation config；
- SDPA Flash backend；
- BF16；
- FFN/lm_head chunk policy；
- output-head position selection policy；
- 单卡 GPU0 placement；
- latency 和 memory instrumentation。

`matched_no_ffn_chunk_reference` 只在短长度、显存允许时使用。它与共同 runtime 完全相同，只关闭 FFN chunking，因此可以单独测量 FFN chunking overhead。另可保留一个 `official_full_logits_unchunked_reference` 做短长度输出等价性 sanity check，但其 latency 不得用于计算 chunking overhead。两者都不进入三个方法的主排名或长上下文主表。

---

## 2. 实验单位、重复与运行顺序

### 2.1 实验单位

- 独立实验单位：一个冻结 prompt（`sample_id`）。
- 技术重复：同一 prompt 的 5 次测量。
- 不能把 `10 prompts x 5 repeats = 50` 当作 50 个独立样本。

唯一 cell 定义为：

`method x exact_prompt_tokens x generation_config x runtime_id x placement`

### 2.2 Warmup 与重复

每个 cell：

1. fresh process 加载模型并完成 runtime/backend preflight；
2. 使用该长度固定 warmup prompt 做 3 次 warmup，全部丢弃；
3. 进入 5 个 repeat rounds；
4. 每个 round 将 10 个 prompt 各运行一次；
5. 共得到每个 prompt 5 次正式测量。

### 2.3 三方法配对与顺序平衡

- Stage 3 run-order seed：`2026081501`。
- 同一 `sample_id + repeat_id + exact_prompt_tokens` 是三方法之间的配对 block。
- 三方法在 block 内使用冻结的平衡顺序，例如 `ABC/BCA/CAB/CBA/ACB/BAC` 循环，而不是总让 baseline 先跑。
- 长度使用 seeded permutation，禁止始终由短到长。
- 每个 worker/cell 使用独立新进程；OOM 后结束进程，不能在同一 CUDA context 清缓存后继续正式测量。

### 2.4 环境控制

- GPU0 不得被其它计算进程共享；正式 run 前检查进程污染。
- 使用物理 GPU0 的 UUID 作为身份；若设置 `CUDA_VISIBLE_DEVICES=0`，同时记录物理与逻辑 index。
- 记录温度、graphics/memory clocks、功率、throttling reason、外来显存占用和 CPU/RAM 状态。
- 固定并记录 driver、CUDA、cuDNN、PyTorch、Transformers、Accelerate 和 allocator 配置。
- 主实验不得使用 GPU1，即使 GPU0 OOM 也不能自动迁移。

---

## 3. Latency 协议

### 3.1 主指标与边界

主指标：`generation_e2e_wall_ms`。

```text
冻结 input_ids 已驻留 GPU0
-> synchronize
-> 开始 wall timer 与 CUDA event
-> 完整 128-token generate
   包含 128 个配置步骤、chunk loop、cache 管理、logits->token、
   confidence、remasking、Python/CPU 调度
-> 结束 CUDA event
-> synchronize
-> 停止 wall timer
```

排除文件读取、tokenize、model load、首次 input H2D、输出 decode、评分与写盘。

同时记录：

- `generation_e2e_wall_ms`：主指标；
- `cuda_event_ms_device_0`：GPU span；
- `cpu_scheduling_gap_ms = wall - cuda_event`；
- `ffn_chunk_calls`；
- `lm_head_chunk_calls`；
- `lm_head_projected_tokens`；
- `model_forward_calls`；
- 方法特定 cache lookup/hit/eviction 计数。

禁止用 unchunked 的单 forward latency 乘以步数代表 chunked E2E。所有主 latency 必须真实运行完整 128-token generation。

### 3.2 Latency pass 的 allocator 规则

- 每个 cell 的 3 次 warmup 前执行一次 `synchronize -> empty_cache -> reset_peak_memory_stats`。
- 5 个正式 repeat rounds 之间不执行 `empty_cache()`。
- latency pass 不启用 NVML 高频采样、profiler、memory snapshot 或详细 tracing。
- backend profiler 只用于独立 preflight，不进入正式计时。

### 3.3 汇总

```text
prompt_latency = median(5 technical repeats)
length_latency = median(10 prompt_latency values)
```

主表报告 median，同时报告 P10/P90、MAD、prompt-level medians 和配对 speedup 分布。置信区间只能在 10 个 prompt-level medians 上 bootstrap。

所有 tokens/s 指标必须注明固定生成 128 tokens；同时报告 `wall_ms / model_forward_calls`，但不把它当作主 E2E。

---

## 4. Peak-memory 协议

### 4.1 必须记录的值

单卡 GPU0 每次 memory run 记录 raw bytes：

1. `torch_peak_allocated_bytes`
2. `torch_peak_reserved_bytes`
3. `nvml_process_peak_bytes`
4. `nvml_device_peak_used_bytes`
5. `model_load_nvml_peak_bytes`
6. `device_total_bytes`
7. `safe_headroom_bytes`

同时记录运行前 allocated、reserved、NVML process used 和 device used。主 fit 指标：

```text
max(model_load_nvml_peak_bytes, inference_nvml_process_peak_bytes)
```

### 4.2 独立 memory pass

```text
fresh process
-> 启动 NVML 20 ms 采样
-> 记录 model-load 峰值
-> backend/runtime preflight
-> 3 次 warmup
-> synchronize
-> empty_cache
-> reset_peak_memory_stats
-> 记录 pre-run baseline
-> 完整运行一次 128-token generate
-> synchronize
-> 读取 allocated/reserved peak
-> 停止并保存 NVML trace
```

不允许在 FFN 或 lm_head 的 chunk 边界 reset peak；峰值必须覆盖完整 generate。Memory 与 latency 使用相同 prompt/config，但分开运行。

容量规划主值使用所有成功 runs 中观测到的最大 peak，而不是 median。OOM 不是缺失值，必须记录阶段、异常、requested/free bytes 和最后一段 NVML trace。

### 4.3 Fit 定义

- `safe-fit`：model load、warmup、正式 run 和 correctness 全部通过，且 headroom 不低于 `max(2 GiB, 5% capacity)`。
- `fragile-fit`：成功但 headroom 低于阈值。
- `oom`：任何加载、warmup 或正式生成阶段 CUDA OOM。
- `context-limit`：违反验证后的模型 total-token 上限。
- `backend-invalid`：未实际使用强制 Flash backend。

若固定 chunk size 下失败，不得在同名 condition 自动缩小 chunk、切双卡或启用 offload。任何替代方案必须另立 condition 并升版。

---

## 5. Chunking 等价性与开销验收

### 5.1 正确性 gate

在 no-cache 路径选择 4K、16K 和 `largest_unchunked_safe` 三个锚点，对比：

- A：`matched_no_ffn_chunk_reference`，保持 sampler-required-position lm_head，只关闭 FFN chunking
- B：`chunked_no_cache`

固定输入、seed、BF16、Flash backend 和 128/32/128 生成配置。必须验证：

1. 128 个最终 output token IDs 完全一致；
2. 每一步被接受/重 mask 的 token positions 一致；
3. forward 次数一致；
4. A/B 均无完整 `[B, sequence, vocab]` logits 被物化；
5. FFN 与 lm_head chunk 计数符合解析预期。

另在 4K 做一次 `official_full_logits_unchunked_reference` 与 `chunked_no_cache` 的输出 token sanity check，用来证明共同 selective/streaming head 没有改变采样结果；该比较不用于归因 latency overhead。

若 BF16 GEMM 形状变化导致输出 token 不一致，不能用“误差很小”跳过。必须记录首次分叉 step、top-1/top-2 margin 和 logits 误差，并在修复或明确建立非等价 condition 后才能进入正式测量。

### 5.2 Chunking latency overhead

对上述锚点做 5 个配对 repeats：

```text
chunking_overhead_pct = 100 * (chunked_wall - unchunked_wall) / unchunked_wall
```

这里的 `unchunked_wall` 专指 matched reference，因此只归因 FFN chunking，不混入 prompt-logit pruning 的收益。Overhead 不设“必须小于某阈值”的通过条件，但必须公开报告。128K 若 matched reference 不可运行，只报告 chunked，不外推伪造 unchunked 数值。

### 5.3 Instrumentation overhead

在 4K、32K 和最大 native-safe 长度，对同一 prompt 做 5 个配对 repeats：

- A：wall + CUDA event；
- B：A + NVML 20 ms polling + 全部 memory counters。

若 paired median 绝对开销超过 1% 或 10 ms 中更严格者，继续保持 latency/memory 两个独立 pass；本协议默认始终分离。

---

## 6. 方法公平性与解释边界

### 6.1 共同优化不计入方法贡献

以下收益属于共同 runtime，不能计入 cache speedup：

- Flash SDPA；
- BF16；
- FFN chunking；
- sampler-required-position lm_head；
- lm_head streaming chunking；
- 单卡 GPU0 placement。

cache 方法的 speedup 定义为同一 paired block 内：

```text
speedup = chunked_no_cache_wall / chunked_cache_method_wall
```

memory saving 同样以 `chunked_no_cache` 为分母。

### 6.2 允许的方法差异

cache 方法可以改变：

- 跨 denoising step 保存与复用的状态；
- 实际重算的 token/layer/feature 数；
- cache lookup、update、eviction 策略；
- 因算法所需产生的额外元数据和控制流。

cache 方法不可以静默改变：

- attention backend、dtype、chunk size、lm_head scope；
- generation steps、block size、remasking；
- prompt、batch、设备；
- correctness criterion 或计时边界。

### 6.3 Correctness guard

每次 run 保存 output token IDs、output SHA-256、任务正确性和垃圾输出检测。主效率结果必须同时给出 accuracy/correctness；不能只报告更快但错误的 cache 输出。

若 cache 方法本身不承诺逐 token 等价，则：

- chunked 与该方法的 unchunked 版本仍须在可运行长度验证实现等价；
- cache 与 no-cache 的差异按方法 accuracy trade-off 报告，不把它误称为 chunking 误差。

---

## 7. 可直接派生的结果

无需增加模型调用即可得到：

- generation wall latency 与 CUDA span；
- wall-CUDA gap；
- 128-token effective tokens/s；
- latency per actual model forward；
- FFN/lm_head chunk counts 与 projected-token counts；
- cache lookup/hit/eviction/recompute 指标；
- allocated/reserved/NVML peaks 与 allocator/non-PyTorch gap；
- OOM/fragile/safe/context-limit 边界；
- latency/memory 对 exact prompt 和 exact total tokens 的 scaling curve；
- output hashes 与 accuracy；
- 温度、时钟、功率和 throttling 元数据。

需要单独 condition 的内容：

- 不同 FFN/lm_head chunk size；
- full-sequence lm_head；
- 32/32/32 官方生成配置；
- 双卡、CPU offload、量化；
- 非 Flash attention；
- 其它 output length 或 batch size。

---

## 8. 原始数据 schema

每次 run 至少记录：

```text
schema_version
experiment_id
protocol_version
protocol_sha256
config_sha256
run_id
session_id
runtime_id
condition_id
method
cache_enabled
sample_id
repeat_id
run_order_index
target_prompt_tokens
exact_prompt_tokens
gen_length
max_sequence_tokens
input_ids_sha256
model_id
model_revision
model_code_sha256
generation_code_sha256
attention_backend_requested
attention_backend_observed
dtype
batch_size
steps_total
num_blocks
block_length
steps_per_block
temperature
cfg_scale
remasking
physical_gpu_index
logical_gpu_index
gpu_uuid
ffn_chunk_size_tokens
lm_head_scope
lm_head_chunk_size_tokens
ffn_chunk_calls
lm_head_chunk_calls
lm_head_projected_tokens
model_forward_calls
cache_metrics
wall_time_ns
cuda_event_ms_device_0
torch_peak_allocated_bytes
torch_peak_reserved_bytes
nvml_process_peak_bytes
nvml_device_peak_used_bytes
model_load_nvml_peak_bytes
device_total_bytes
safe_headroom_bytes
fit_status
oom_stage
output_token_ids
output_sha256
correct
gpu_temperature_start_end
gpu_clocks_start_end
gpu_power_start_end
throttle_reasons
started_at_utc
ended_at_utc
```

缺少关键字段的 run 标记为 invalid，不能默默补零或从其它 run 推断。

---

## 9. 交付物

```text
S3_Measurement_Protocol.md
s3_measurement_config.json
s3_environment.json
s3_method_registry.json
s3_run_schedule.csv
s3_chunking_equivalence.jsonl
s3_chunking_overhead.csv
s3_raw_latency_runs.jsonl
s3_raw_memory_runs.jsonl
s3_nvml_traces/
s3_fit_matrix.csv
s3_latency_summary.csv
s3_memory_summary.csv
s3_validation_report.json
checksums.sha256
measurement_complete.lock
```

主图/表：

1. 三方法 wall latency vs exact prompt tokens；
2. paired speedup vs `chunked_no_cache`；
3. allocated/reserved/NVML peak vs length；
4. cache hit/recompute 与 speedup 的关系；
5. chunked vs unchunked reference overhead；
6. safe/fragile/OOM/context-limit fit matrix；
7. accuracy 与效率联合表。

---

## 10. 冻结前验收清单

- [ ] 两个 cache 方法的真实 ID、revision、参数和源码哈希已填写。
- [ ] 三个方法都调用同一个 `shared_chunked_bf16_flash_v1` 实现。
- [ ] 主 baseline 明确命名为 `chunked_no_cache`，没有使用“official unmodified”表述。
- [ ] 生成语义确认为 128 total steps、4 blocks、32 steps/block。
- [ ] 128K 点的 prompt/total-token 上限已经 preflight 并冻结。
- [ ] 物理 GPU0 UUID、软件环境和 allocator 已冻结。
- [ ] 正式 forward 实测为 SDPA Flash，无 silent fallback。
- [ ] FFN chunk=8192、lm_head chunk=512，且完整 logits 不会被重新拼接。
- [ ] 三种方法的 lm_head required-position policy 相同。
- [ ] chunked/unchunked 输出等价性 gate 已通过。
- [ ] latency 使用真实完整 128-token generation，不使用乘法推导主结果。
- [ ] latency/memory 使用两个 pass。
- [ ] 三方法运行顺序已经配对、平衡并冻结。
- [ ] raw schema、原子写入、resume、checksum 和 invalid-run 规则已实现。
- [ ] 旧协议产生的数据标记为 historical-only，不混入 v2 主表。

全部通过后将状态改为 `frozen`，生成协议、配置、方法注册表和源码 bundle 的 SHA-256。任何实质变更必须升版，不得覆盖已有正式数据。

---

## 11. 依据

- UltraLLaDA 模型与论文：<https://huggingface.co/relaxe-system-lab/UltraLLaDA>；<https://arxiv.org/abs/2510.10481>
- PyTorch SDPA 与 CUDA memory：<https://docs.pytorch.org/docs/main/notes/cuda.html>；<https://docs.pytorch.org/docs/stable/torch_cuda_memory.html>
- 本协议中的 chunked FFN 与 selective/streaming lm_head 是共同的 exact systems substrate；cache/no-cache 的方法贡献必须在该共同底座之上比较。
