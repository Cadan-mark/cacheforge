# Benchmark 使用与解读

[返回中文 README](../README.md) | [English README](../README.en.md)

本文说明如何运行 CacheForge Milestone 1 的 CPU / CUDA microbenchmark，以及如何正确理解结果。

代码位置：[`benchmarks/benchmark_decode.py`](../benchmarks/benchmark_decode.py)

## 1. 如何运行

### CPU

```bash
PYTHONPATH=src python benchmarks/benchmark_decode.py --device cpu
```

### CUDA

```bash
PYTHONPATH=src python benchmarks/benchmark_decode.py --device cuda --dtype float16
```

如果希望自动选择设备：

```bash
PYTHONPATH=src python benchmarks/benchmark_decode.py --device auto
```

---

## 2. 重要 CLI 参数

- `--device`：`auto` / `cpu` / `cuda`
- `--dtype`：`float32` / `float16` / `bfloat16`
- `--batch-size`：批大小
- `--prompt-len`：Prefill 阶段 prompt 长度
- `--decode-steps`：后续 Decode token 数
- `--hidden-size`：隐藏维度
- `--num-heads`：Query heads 数
- `--num-kv-heads`：KV heads 数
- `--warmup`：预热次数
- `--repeats`：正式重复次数

常见实验示例：

```bash
PYTHONPATH=src python benchmarks/benchmark_decode.py \
  --device cpu \
  --prompt-len 128 \
  --decode-steps 32 \
  --batch-size 4 \
  --hidden-size 256 \
  --num-heads 8 \
  --num-kv-heads 2 \
  --warmup 2 \
  --repeats 5
```

---

## 3. benchmark 在比较什么？

脚本比较两种模式：

### `full_recompute`

每个 Decode step 都重新运行：

- 当前为止全部上下文的 Q/K/V projection
- 整段上下文的 RoPE
- 整段 causal attention

这是最直接的参考实现，但在自回归生成中会重复计算历史。

### `prefill_plus_cached`

先：

- 对 prompt 运行一次 `prefill()`
- 将 prompt 的 K/V 写入 contiguous KV Cache

然后每个 Decode step：

- 只处理当前 1 个新 token
- 把当前 token 的 K/V append 到 cache
- 读取有效历史 cache 做 attention

因此它更接近真实 LLM 推理中的核心优化思想。

---

## 4. 为什么 CUDA synchronization 很重要？

GPU 上多数运算是异步提交的。若不在计时前后调用 `torch.cuda.synchronize()`：

- Python 计时器可能只记录到 kernel 提交时间
- 而不是 GPU 真正完成运算的时间

本项目在 `sync_if_needed()` 中做了显式同步，并在计时区间前后调用，因此 CUDA 延迟读数更可靠。

这也是为什么“简单地在 CUDA 代码前后包 `time.perf_counter()`”通常并不可信。

---

## 5. 如何理解输出中的延迟

输出表里有：

- `mean_ms`
- `median_ms`
- `per_step_ms`

### `full_recompute`

如果这个值很高，说明随着上下文增长，重复处理历史 token 的代价明显。

### `prefill_plus_cached`

如果这个值更低，说明 KV Cache 成功避免了历史 K/V 的重复构造。

但要注意：cached Decode 并不意味着“只看当前 token，不再碰历史”。当前 token 仍要对所有有效历史 key 做注意力分数计算，所以历史长度增加时，它的成本也会增长，只是增长方式与 full recompute 不同。

---

## 6. theoretical vs allocated cache bytes

脚本会输出：

- `theoretical_kv_cache_bytes`
- `allocated_kv_cache_bytes`

### `theoretical_kv_cache_bytes`

这是按公式计算的理论值：

`2 * batch * num_kv_heads * max_seq_len * head_dim * bytes_per_element`

只表示 K 与 V 的理论存储大小。

### `allocated_kv_cache_bytes`

这是 `ContiguousKVCache.allocated_bytes()` 读取 PyTorch storage 得到的真实已分配字节数，即当前实现中 `key` 与 `value` 两块底层 storage 的总大小。

它有助于确认：

- 这个实现确实只按 KV heads 分配存储
- 预分配的大小是否与你设置的 `max_seq_len` 一致

---

## 7. CUDA peak memory

当设备是 CUDA 时，脚本还会输出：

- `cuda_peak_allocated_bytes`

这表示在该 benchmark 运行期间，PyTorch 记录到的峰值已分配显存。

它可以帮助你比较：

- 更长 prompt 是否拉高峰值显存
- 更大 batch 是否增加整体占用
- 不同 `num_kv_heads` / dtype 的影响

但不要把它误解为“生产推理服务的总显存需求”，因为这里没有：

- 多层模型参数
- tokenizer
- sampler
- 通信 / 服务框架
- 更复杂的缓存系统

---

## 8. warmup 与 repeats

### warmup

`--warmup` 用来做预热。预热的重要性在于：

- 首次运行可能包含额外初始化开销
- CUDA 场景下可能涉及 kernel/allocator 的准备成本
- CPU 下也可能出现缓存与分支预测尚未稳定的情况

### repeats

`--repeats` 用来正式多次测量，降低单次波动带来的误导。

一般来说：

- 小实验可用 `warmup=1, repeats=3`
- 稍认真一点的对比可提高到 `warmup=2, repeats=5` 或更多

---

## 9. 为什么 tiny CPU workload 可能看不出 speedup？

当 `prompt_len`、`decode_steps`、`hidden_size` 都很小时，性能结果可能不明显，原因包括：

- Python 函数调用与循环开销占比高
- 小矩阵乘法本身太快
- 内存与缓存效应掩盖了算法差异
- 测量噪声较大

因此，如果你想更清楚看到 cached Decode 的优势，可以尝试：

- 增大 `prompt_len`
- 增大 `decode_steps`
- 增大 `batch-size`
- 在 CUDA 上运行

---

## 10. 为什么这是 attention microbenchmark，而不是端到端 LLM benchmark？

这个脚本只测：

- 一个 Attention 模块
- 两种执行路径（全量重算 vs 缓存 Decode）
- 与 KV Cache 直接相关的时间和内存

它没有包含：

- 多层 Transformer
- FFN / RMSNorm / residual stack
- tokenizer
- 采样
- 网络传输
- continuous batching
- paged KV cache
- 服务端调度

因此它更适合回答“机制层面”的问题，而不是“系统级吞吐量”问题。

---

## 11. 建议做的实验

### 实验 1：增大 prompt length

观察：

- `full_recompute` 是否明显变慢
- `prefill_plus_cached` 与其差距是否扩大

### 实验 2：增大 decode steps

观察：

- 两种模式的累计成本如何增长
- `per_step_ms` 是否稳定

### 实验 3：增大 batch size

观察：

- batch 对缓存路径和全量重算路径的影响
- CUDA 峰值显存如何变化

### 实验 4：改变 KV heads 数

例如固定 `num_heads=8`，比较：

- `num_kv_heads=8`（更接近 MHA）
- `num_kv_heads=2`（GQA）
- `num_kv_heads=1`（更接近 MQA）

重点观察：

- theoretical / allocated cache bytes 是否按比例下降
- latency 是否也出现变化

### 实验 5：比较 dtype

在支持的设备上比较：

- `float32`
- `float16`
- `bfloat16`

重点观察：

- cache bytes 变化
- latency 变化
- 数值与设备支持差异

---

## 小结

解读这个 benchmark 时，最重要的是记住三件事：

1. **cached Decode 省掉的是历史 K/V 的重复构造，不是全部历史相关计算。**
2. **可信的 CUDA timing 需要同步。**
3. **这是教学型 Attention microbenchmark，不是完整推理系统的性能结论。**
