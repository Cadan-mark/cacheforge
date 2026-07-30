# Milestone 1 实现导读

[返回中文 README](../README.md) | [English README](../README.en.md)

本文按推荐阅读顺序讲解 CacheForge Milestone 1 的实现，并串起两条执行路径：

1. **full-context recomputation**
2. **prefill + one-token cached Decode**

目标不是逐行翻译代码，而是解释：每个模块在推理里扮演什么角色、Shape 如何变化、KV Cache 避免了什么重复工作、以及为什么缓存路径应与全量重算路径数值等价。

---

## 1. `config.py`

代码：[`src/cacheforge/config.py`](../src/cacheforge/config.py)

`AttentionConfig` 定义了这个教学实现最关键的结构参数：

- `hidden_size`
- `num_attention_heads`
- `num_key_value_heads`
- `max_seq_len`
- `rope_theta`

其中两个派生属性尤其重要：

- `head_dim = hidden_size / num_attention_heads`
- `num_key_value_groups = num_attention_heads / num_key_value_heads`

### 这里为什么要做严格校验？

因为后续所有 `view()`、`transpose()` 与 GQA 扩展都默认这些整除关系成立：

- `hidden_size % num_attention_heads == 0`
- `num_attention_heads % num_key_value_heads == 0`
- `head_dim % 2 == 0`

最后一个条件与 RoPE 有关：RoPE 会把最后一维两两配对做二维旋转，因此 `head_dim` 必须是偶数。

---

## 2. `rope.py`

代码：[`src/cacheforge/rope.py`](../src/cacheforge/rope.py)

`RotaryEmbedding` 的核心是先构造 `inv_freq`，再在 `apply_rotary_embedding()` 中根据 `positions` 生成每个位置对应的旋转角。

输入 Tensor 期望 Shape 为：

`[batch, heads, seq_len, head_dim]`

### 为什么这里使用 absolute positions？

因为 RoPE 不是“只和相对顺序有关”的抽象标签，而是直接把位置编码进向量旋转相位里。对于第 `p` 个 token，旋转角就是由 `p` 决定的。

这在 Decode 阶段尤其关键：

- 如果 prompt 长度是 128
- 当前 decode token 是第 129 个位置
- 那它必须以位置 128（0-based）去做旋转

如果错误地把 Decode token 当成位置 0，那么它的 Q/K 相位就与 full-context 路径不一致。

### RoPE 真正在做什么？

`head_dim` 的偶数/奇数索引维度被两两配对：

- `tensor[..., ::2]`
- `tensor[..., 1::2]`

随后每一对都进行二维旋转：

- `x' = x cos(theta) - y sin(theta)`
- `y' = x sin(theta) + y cos(theta)`

因此 RoPE 不是额外拼接一个位置向量，而是直接旋转原始 Q/K 特征。

---

## 3. `cache/contiguous.py`

代码：[`src/cacheforge/cache/contiguous.py`](../src/cacheforge/cache/contiguous.py)

这个文件实现了 Milestone 1 的 contiguous KV Cache。缓存布局是：

`[batch, num_kv_heads, max_seq_len, head_dim]`

它内部有三块关键状态：

- `self.key`
- `self.value`
- `self.lengths`

### 为什么要预分配（preallocate）？

因为 Decode 会持续追加 token。如果每来一个 token 就做 Tensor 拼接：

- 代码会更复杂
- 会反复重新分配内存
- 容易让“cache 的真实布局”变得不清晰

Milestone 1 选择一次性预分配 `max_seq_len`，再按位置写入，这样更容易把注意力放在缓存语义本身。

### `lengths` 的作用

`lengths[batch_slot]` 记录该样本当前已写入多少个 token。它决定：

- 下一次 `append()` 应该写到哪里
- `decode()` 时哪些历史位置是有效的
- 哪些尾部位置虽然已分配，但还没有写入，不能参加 attention

### 为什么不能留空洞（gaps）？

`write()` 不允许 `start > current_length`。例如长度当前是 4，却直接去写第 6 个位置，会留下未写入位置 4、5。这种空洞会让“哪些 key/value 有效”变得更难推理，也与 Milestone 1 的 contiguous 假设不一致。

### `append()` 为什么要求位置等于当前长度？

因为 `append()` 的语义不是“随便写一个位置”，而是**把新 token 接到当前序列尾部**。所以它要求：

`append_position == current_sequence_length`

这正对应了自回归生成里“序列增长 1”的过程。

### `allocated_bytes()` 表示什么？

它返回 `key` 与 `value` 两块 storage 总字节数，也就是该 cache 实现真实保留的 K+V 存储空间。

---

## 4. `attention.py::_project()`

代码：[`src/cacheforge/attention.py`](../src/cacheforge/attention.py)

`_project()` 做了三件事：

1. 线性投影得到 Q/K/V
2. 把投影结果 reshape/transpose 成 attention 布局
3. 对 Q 与 K 应用 RoPE

### 为什么 Q 用所有 attention heads，而 K/V 只用 KV heads？

因为这是 GQA 的核心设计：

- Query 仍保留较多 heads，维持表达能力
- Key/Value 使用更少 heads，降低缓存内存与带宽

所以：

- Q 的 Shape 变成 `[batch, num_attention_heads, seq_len, head_dim]`
- K/V 的 Shape 变成 `[batch, num_kv_heads, seq_len, head_dim]`

### 这里的 Shape 变化

投影后的原始 layout 还是 `[batch, seq_len, ...]`，需要通过 `view(...).transpose(1, 2)` 变成注意力更方便处理的 `[batch, heads, seq_len, head_dim]`。

这是后续矩阵乘法与 mask 广播最常用的 layout。

---

## 5. `attention.py::forward()`

代码：[`src/cacheforge/attention.py`](../src/cacheforge/attention.py)

`forward()` 是最朴素的完整上下文路径：

1. 规范化 `positions`
2. 调用 `_project()` 得到 Q/K/V
3. 让 query 和 key 使用同一段 `positions`
4. 进入 `_attention()`
5. 再通过 `_merge_heads()` 回到 `[batch, seq_len, hidden_size]`

这里没有使用 cache，因此每次调用都把当前输入序列的所有 token 从头完整计算一遍。

---

## 6. `attention.py::prefill()`

代码：[`src/cacheforge/attention.py`](../src/cacheforge/attention.py)

`prefill()` 处理一段连续 prompt。它与 `forward()` 的共同点是：

- 都要对整段输入做 Q/K/V projection
- 都要对整段输入做 causal attention
- 都要返回 prompt 上每个位置的输出

不同点是：

- `prefill()` 额外要求 `positions` 描述**连续区间**
- 它会把这段 prompt 的 K/V 写入 `ContiguousKVCache`

例如 prompt 长度为 4 时：

- 位置通常是 `[0, 1, 2, 3]`
- 写入 cache 后，`lengths` 会更新为 4

### 为什么 Prefill 与 full-context 前半段应当等价？

因为它们看到的是同一批 token、同一组绝对位置、同一条 causal mask 规则。唯一额外发生的事情是 `prefill()` 顺手把 K/V 存进 cache。这个写缓存动作不应改变输出数值。

测试 [`tests/test_attention.py`](../tests/test_attention.py) 中正是这样验证的：`prefill` 输出要与 `full[:, :prompt_len]` 对齐。

---

## 7. `attention.py::decode()`

代码：[`src/cacheforge/attention.py`](../src/cacheforge/attention.py)

`decode()` 明确只接受单 token：`seq_len == 1`。

执行顺序是：

1. 规范化当前 token 的 absolute position
2. 做当前 token 的 Q/K/V projection
3. 先把当前 token 的 K/V append 到 cache
4. 从 cache 读取当前有效长度范围内的历史 K/V
5. 构造 `key_positions = [0, 1, ..., current_length - 1]`
6. 用当前 query 对有效 cache 做 attention
7. 输出当前 token 的结果

### 为什么要“先 append，再做 attention”？

因为自回归 causal attention 允许一个 token 看到：

- 全部历史 token
- 它自己

如果先算 attention、后写入当前 token，那么当前 query 就看不到“自己位置”的 K/V；这会与 full-context 路径不一致。

### 为什么这条路径与 full-context recomputation 数值等价？

假设已经对 prompt `[0, 1, 2, 3]` 做了 prefill，接着解第 4 个 token：

- full-context 路径会对 `[0, 1, 2, 3, 4]` 全部投影并做因果注意力
- cached Decode 路径会重用 `[0, 1, 2, 3]` 的历史 K/V，只新算第 4 个 token 的 Q/K/V

两者最终给第 4 个 token 提供的注意力输入集合其实相同：

- 同样的历史 K/V
- 同样的当前位置 4
- 同样的 causal mask
- 同样的当前 token 自身 K/V

只要 cache 写入正确、RoPE 位置正确、mask 正确，两条路径的输出就应一致。

### KV Cache 到底避免了什么重复工作？

它避免了对历史 token 的：

- K projection 重算
- V projection 重算
- K 的 RoPE 重算
- 历史 K/V 的重复物化

### Decode 仍然剩下哪些工作？

它仍然必须做：

- 当前 token 的 Q/K/V projection
- 当前 token 的 Q/K RoPE
- 当前 query 对所有有效 key 的打分
- softmax
- 与 V 的加权求和
- output projection

因此 Decode 不是“完全不算”，而是“不再重复算历史 K/V”。

---

## 8. Correctness tests

代码：[`tests/test_attention.py`](../tests/test_attention.py)、[`tests/test_contiguous_cache.py`](../tests/test_contiguous_cache.py)

### `test_full_context_matches_prefill_then_cached_decode`

这条测试是里程碑里最关键的正确性证明：

- 先跑 `attention(hidden_states)` 得到 full-context reference
- 再对前半段做 `prefill()`
- 最后逐 token 调用 `decode()`
- 检查每一步结果都与 full-context 对应位置一致

它验证了：

- Prefill 前缀输出正确
- Cache 写入位置正确
- Decode 使用 absolute positions 正确
- “先 append 当前 token，再做 attention” 的逻辑正确

### `test_grouped_query_attention_shapes_and_equivalence`

这条测试确认：

- GQA 下 cache 的 K/V Shape 只与 `num_kv_heads` 有关
- `prefill()` / `decode()` 仍能与 full-context 对齐

### `test_contiguous_cache.py`

这些测试覆盖了：

- 多 token 写入后再单 token append
- 不同 batch slot 互不覆盖
- 越界、dtype 不匹配、shape 不匹配
- 禁止 unwritten gaps
- reset 后可复用
- `allocated_bytes()` 是否符合理论公式

---

## 9. Benchmark

代码：[`benchmarks/benchmark_decode.py`](../benchmarks/benchmark_decode.py)

这个脚本比较：

- `benchmark_full_recompute()`
- `benchmark_cached_decode()`

前者在每个 decode step 都调用完整 `attention(hidden_states[:, :prompt_len + step + 1])`。

后者则：

- 先建 `ContiguousKVCache`
- 运行一次 `prefill()`
- 再对每个 step 调用 `decode()`

### 为什么 benchmark 里需要 CUDA synchronize？

因为 GPU kernel 发射默认是异步的。若不在计时前后同步，测到的更像“Python 把任务提交给 GPU 的时间”，而不是 GPU 实际完成计算的时间。

### 这个 benchmark 可以回答什么问题？

它适合回答：

- prompt 越长时 cached Decode 与 full recompute 差距如何变化
- batch size、decode steps、KV head 数对该单 Attention 模块的影响如何变化

### 它不能回答什么问题？

它不能直接回答：

- 真实大模型服务的端到端吞吐是多少
- vLLM / PagedAttention / continuous batching 的系统级优势是多少

因为 Milestone 1 还没有实现这些机制。

---

## 两条执行路径的总览

### 路径 A：full-context recomputation

对长度为 `T` 的序列：

1. 全部 token 做 Q/K/V projection
2. 全部 token 的 Q/K 做 RoPE
3. 构造 `T x T` 的 causal attention 关系
4. 产出全部位置输出

优点：实现直接，适合作参考正确性。
缺点：Decode 时会重复计算历史。

### 路径 B：prefill + cached Decode

1. Prompt 阶段做一次 `prefill()`，写入历史 K/V
2. 每个新 token 调用 `decode()`
3. 只新算当前 token 的 Q/K/V
4. 历史 K/V 直接从 cache 读取

优点：避免历史 K/V 重算，更接近真实推理系统的核心优化。
缺点：需要额外维护 cache、位置语义与有效长度掩码。

---

## 推荐继续阅读

- [Tensor Shape 详细讲解](./tensor-shapes.zh-CN.md)
- [Benchmark 使用与解读](./benchmark-guide.zh-CN.md)
- [术语表 / Glossary](./glossary.zh-CN.md)
