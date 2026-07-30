# CacheForge

**中文 README** | [English README](README.en.md)

CacheForge Milestone 1 是一个面向学习的 PyTorch 小项目，用来理解 Llama 风格的 causal self-attention、Rotary Positional Embeddings (RoPE)、Grouped-Query Attention (GQA) 与 contiguous KV Cache。

这个仓库的重点不是“做一个完整推理框架”，而是把推理里的几个关键基础构件拆开、做小、做对，让学习者可以直接顺着源码和测试看懂：

- Attention 的 Tensor Shape 如何变化
- 为什么 GQA 会减少 KV Cache 内存
- Prefill 与 Decode 分别在做什么
- 为什么缓存路径与全量重算路径应该数值等价
- contiguous KV Cache 的布局、约束与代价是什么

## 文档导航

- [English README](README.en.md)
- [Milestone 1 实现导读](docs/milestone-1-walkthrough.zh-CN.md)
- [Tensor Shape 详细讲解](docs/tensor-shapes.zh-CN.md)
- [Benchmark 使用与解读](docs/benchmark-guide.zh-CN.md)
- [术语表 / Glossary](docs/glossary.zh-CN.md)

## 项目目标与 Milestone 1 范围

Milestone 1 聚焦一个**单层、教学用途**的 Llama 风格注意力模块，目标是先把推理阶段最核心的 Attention + KV Cache 路径讲清楚并验证正确性。

本里程碑**已包含**：

- 单个 Llama 风格 Attention 模块的 `AttentionConfig` 与参数校验
- 使用绝对位置（absolute positions）的 RoPE
- 完整上下文的 causal self-attention 计算
- `prefill()` + 单 token `decode()` 的缓存路径
- GQA：`num_attention_heads` 可以大于 `num_key_value_heads`
- 只存储 KV heads 的预分配 contiguous KV Cache
- CPU 优先、可重复的 correctness tests 与微基准（microbenchmark）

本里程碑**明确不包含**：

- PagedAttention / paged KV cache
- Continuous batching / scheduler
- KV-cache quantization
- Triton kernel 或自定义 CUDA kernel
- Hugging Face 模型加载
- vLLM 集成
- 完整 Transformer stack、采样器、服务端 API

换句话说，这个仓库当前关注的是**推理内部机制的正确性与可学习性**，而不是端到端吞吐最优实现。

## 安装

```bash
python -m pip install -e .[dev]
```

## 测试

```bash
pytest
```

## Lint / 格式检查

项目当前配置了 `ruff`：

```bash
ruff check .
```

## Benchmark

在 CPU 上运行 Attention 微基准：

```bash
PYTHONPATH=src python benchmarks/benchmark_decode.py --device cpu
```

如果本机有 CUDA：

```bash
PYTHONPATH=src python benchmarks/benchmark_decode.py --device cuda --dtype float16
```

常用参数：

- `--prompt-len`
- `--decode-steps`
- `--batch-size`
- `--hidden-size`
- `--num-heads`
- `--num-kv-heads`
- `--warmup`
- `--repeats`

这个 benchmark 比较两条路径：

1. **naive full-context recomputation**：每个 Decode step 都把从第 0 个 token 到当前 token 的全部上下文重新跑一遍 Attention。
2. **prefill + cached Decode**：先把 prompt 做一次 Prefill 并写入 KV Cache，之后每个 Decode step 只处理“当前新 token”，并读取历史缓存。

它会报告：

- 延迟统计
- `theoretical_kv_cache_bytes`
- `allocated_kv_cache_bytes`
- CUDA 下的 `cuda_peak_allocated_bytes`

请注意：这只是**单个 Attention 组件的 microbenchmark**，不是端到端 LLM benchmark，也不是 vLLM benchmark。

## Tensor Shape 快速总览

输入 `hidden_states` 的 Shape 是：

`[batch, seq_len, hidden_size]`

在当前实现中：

- Q projection：`[batch, seq_len, hidden_size] -> [batch, num_attention_heads, seq_len, head_dim]`
- K projection：`[batch, seq_len, hidden_size] -> [batch, num_kv_heads, seq_len, head_dim]`
- V projection：`[batch, seq_len, hidden_size] -> [batch, num_kv_heads, seq_len, head_dim]`
- Cache layout：`[batch, num_kv_heads, max_seq_len, head_dim]`
- GQA 计算时扩展：`[batch, num_kv_heads, seq_len, head_dim] -> [batch, num_attention_heads, seq_len, head_dim]`

关键点：**Cache 只存 KV heads，不存扩展后的 query-head 对齐版本。**

更详细的 Shape 推导见：[docs/tensor-shapes.zh-CN.md](docs/tensor-shapes.zh-CN.md)

## Prefill vs Decode

### Prefill

Prefill 处理的是一段 prompt：

- 对这段 token 一次性做 Q/K/V projection
- 对 Q 与 K 应用 RoPE
- 用 causal mask 让每个位置只能看到自己与历史
- 产出 prompt 上每个位置的输出
- 同时把这段 prompt 的 K/V 写入 cache

### Decode

Decode 处理的是**一个新 token**：

- 输入 Shape 固定是单 token（`seq_len == 1`）
- 当前 token 仍然需要做 Q/K/V projection
- 当前 token 的 Q/K 仍然需要按**绝对位置**应用 RoPE
- 当前 token 的 K/V 会 append 到 cache
- 当前 token 的 Q 会对“历史有效 cache + 当前 token 自己”做 attention

为什么要强调“绝对位置”？

因为 RoPE 的相位取决于 token 的真实位置。若 Decode 时把位置错误地重新从 0 开始，缓存路径就会与 full-context reference 的旋转相位不一致，结果也会偏离。

## KV Cache 内存公式

对于当前这个单 Attention 模块与 cache 布局：

`2 * batch * num_kv_heads * max_seq_len * head_dim * bytes_per_element`

解释：

- 前面的 `2` 表示同时存 K 和 V
- `batch * num_kv_heads * max_seq_len * head_dim` 是单个 K（或 V）Tensor 的元素数
- `bytes_per_element` 由 dtype 决定
- 如果扩展到完整 Transformer，还要再乘以 Attention 层数

## 为什么 GQA 能节省内存

在 Multi-Head Attention (MHA) 中，Q/K/V 的 head 数通常相同。

在 GQA 中：

- Q 仍然有 `num_attention_heads`
- K/V 只有 `num_key_value_heads`
- 两者之间的比值是 `num_key_value_groups = num_attention_heads / num_key_value_heads`

由于 KV Cache 的存储规模取决于 `num_kv_heads`，而不是 `num_attention_heads`，所以当 `num_kv_heads` 更少时，缓存内存会显著下降。

当前实现中，K/V 只在**计算 attention 分数时**按 group 扩展到 query heads 数量；cache 内部仍保留未扩展的紧凑表示。

## Benchmark 应该如何解读

这个 benchmark 的核心问题不是“这个项目有多快”，而是：

> 当上下文逐步增长时，缓存路径相对全量重算，节省了哪些工作，又还保留了哪些工作？

### 缓存路径避免了什么？

cached Decode 避免了：

- 每一步都对全部历史 token 重新做 K projection
- 每一步都对全部历史 token 重新做 V projection
- 每一步都对全部历史 token 重新重复应用 RoPE 到 K
- 每一步都重新构造完整历史的 Attention 输入

### 缓存路径仍然要做什么？

cached Decode 仍然需要：

- 当前 token 的 Q/K/V projection
- 当前 token 的 RoPE（Q 与 K）
- 当前 token 对所有有效历史位置的 attention score 计算
- softmax
- attention output 与 output projection

因此，KV Cache 不是“把 Decode 变成零成本”，而是**避免历史 K/V 的重复构造与重复写入**。

### 为什么 tiny CPU workload 可能看不出明显加速？

在非常小的 CPU 测试里，以下因素可能占主导：

- Python 调用开销
- 小矩阵下 BLAS / kernel 启动收益不明显
- 内存层级与缓存命中差异
- warmup 不充分

所以 CPU 小规模实验更适合帮助理解趋势，而不是得出生产级结论。

### 为什么 CUDA timing 需要 synchronization？

CUDA kernel 默认是异步发射的。如果在计时前后不做 `torch.cuda.synchronize()`，`time.perf_counter()` 很可能只量到“提交工作”的时间，而不是 GPU 真正完成计算的时间。

当前 benchmark 已在计时区间前后显式同步，因此它的 CUDA 延迟数据更可信。

### 理论 cache 字节数 vs 实际分配字节数

- `theoretical_kv_cache_bytes`：按公式计算的 K+V 理论大小
- `allocated_kv_cache_bytes`：PyTorch 实际为 `key` 与 `value` 两块 storage 分配的字节数

两者接近时，说明这个 cache 实现的存储开销很直接；但这仍然只是该模块的分配，不代表完整推理系统的总显存使用。

### Benchmark 的边界

这个 benchmark 是：

- Attention microbenchmark
- 教学用途的实现对比
- 用来理解 Prefill / Decode / KV Cache 行为

它不是：

- 端到端 LLM benchmark
- 含 tokenizer / sampler / networking 的服务 benchmark
- vLLM、SGLang 或 TensorRT-LLM 的系统级对比

## 当前限制

- 只实现了单个模块，不是完整 Transformer
- `decode()` 目前是 one-token-at-a-time API
- contiguous cache 不允许 unwritten gaps，更偏重正确性与易懂性
- 没有 fused kernels，也没有吞吐导向优化

## 下一步里程碑

1. Paged KV Cache
2. Scheduler / continuous batching
3. KV-cache quantization
4. 在 correctness 稳定后再做 profiling 与 Triton 风格优化

## 学习文档索引

如果你想把这个仓库当作 Milestone 1 的学习材料，建议按下面顺序阅读：

1. [Milestone 1 实现导读](docs/milestone-1-walkthrough.zh-CN.md)
2. [Tensor Shape 详细讲解](docs/tensor-shapes.zh-CN.md)
3. [Benchmark 使用与解读](docs/benchmark-guide.zh-CN.md)
4. [术语表 / Glossary](docs/glossary.zh-CN.md)

其中：

- 实现导读会按 `config.py -> rope.py -> contiguous cache -> attention -> tests -> benchmark` 的顺序讲源码
- Tensor Shape 文档会用一个具体 GQA 例子把每一步 Shape 展开
- Benchmark 指南聚焦如何运行、怎么看结果、该做什么实验
- Glossary 用中英双语统一术语，避免概念混淆

## 建议验证命令

```bash
ruff check .
pytest
PYTHONPATH=src python benchmarks/benchmark_decode.py --device cpu
```
