# Tensor Shape 详细讲解

[返回中文 README](../README.md) | [Milestone 1 实现导读](./milestone-1-walkthrough.zh-CN.md)

本文用一个具体的 GQA 例子详细展开 CacheForge Milestone 1 的 Shape 变化。

## 示例配置

- `batch = 2`
- `seq_len = 4`
- `hidden_size = 128`
- `num_attention_heads = 8`
- `num_key_value_heads = 2`
- `head_dim = 16`

因为：

- `head_dim = hidden_size / num_attention_heads = 128 / 8 = 16`
- `num_key_value_groups = num_attention_heads / num_key_value_heads = 8 / 2 = 4`

这意味着：

- 每个 Query head 的维度是 16
- 一共有 8 个 Query heads
- 只有 2 个 KV heads
- 每个 KV head 会服务 4 个 Query heads

---

## 1. 输入 hidden states

输入 `hidden_states` Shape：

`[batch, seq_len, hidden_size] = [2, 4, 128]`

可以把它理解成：

- 2 个样本
- 每个样本 4 个 token
- 每个 token 是 128 维隐藏向量

---

## 2. Q / K / V projection 输出

### Query

`q_proj(hidden_states)` 先得到：

`[2, 4, 128]`

随后 reshape 为：

`[2, 4, 8, 16]`

再 `transpose(1, 2)` 变成：

`[2, 8, 4, 16]`

也就是：

- 2 个 batch
- 8 个 Query heads
- 每个 head 看到 4 个 token
- 每个 token 在该 head 上是 16 维

### Key

`k_proj(hidden_states)` 的输出宽度不是 128，而是：

`num_key_value_heads * head_dim = 2 * 16 = 32`

因此：

- 线性层输出先是 `[2, 4, 32]`
- reshape 后是 `[2, 4, 2, 16]`
- transpose 后是 `[2, 2, 4, 16]`

### Value

`v_proj(hidden_states)` 与 Key 相同：

`[2, 2, 4, 16]`

### 为什么 Q 和 K/V 的 head 数不同？

这正是 GQA：

- Query 保持更多 heads，用来保留较细的注意力分工
- Key / Value 使用更少 heads，主要为了节省 KV Cache 存储与带宽

---

## 3. RoPE 的输入与 absolute positions

在 `_project()` 中：

- Q 会进入 `self.rotary(query.transpose(1, 2), positions)`
- K 也会进入 `self.rotary(...)`
- V 不做 RoPE

RoPE 的输入 Shape 分别是：

- Q：`[2, 8, 4, 16]`
- K：`[2, 2, 4, 16]`
- `positions`：`[2, 4]`

若 positions 为默认连续范围，则通常是：

```text
[[0, 1, 2, 3],
 [0, 1, 2, 3]]
```

### 为什么必须是 absolute positions？

RoPE 的旋转角度依赖 token 的真实位置。Decode 时如果当前 token 是全序列第 4 个位置，它就必须用位置 4，而不是局部窗口里的 0。

---

## 4. GQA expansion：只为计算，不为存储

现在：

- Q 的 heads 数是 8
- K/V 的 heads 数是 2

为了做 attention，head 数必须对齐，所以 `_expand_kv()` 会把：

`[2, 2, 4, 16] -> [2, 8, 4, 16]`

具体来说，每个 KV head 被重复 `num_key_value_groups = 4` 次。

### 重要：为什么 expansion 只用于计算？

因为如果把扩展后的 K/V 也存进 cache：

- 内存会按 Query heads 数膨胀
- GQA 节省内存的意义就被抵消了

所以当前实现的策略是：

- **cache 中只保存紧凑的 `[batch, num_kv_heads, seq_len, head_dim]`**
- **进入 attention 计算时再临时扩展到 Query heads 数**

---

## 5. Attention score / probability / output shapes

扩展后：

- `query`: `[2, 8, 4, 16]`
- `expanded_key`: `[2, 8, 4, 16]`
- `expanded_value`: `[2, 8, 4, 16]`

### attention scores

`scores = query @ expanded_key.transpose(-1, -2)`

因此 Shape：

`[2, 8, 4, 4]`

含义：

- 对每个 batch
- 对每个 Query head
- 对每个 query token
- 都会得到它对 4 个 key positions 的分数

### causal mask

mask 的核心关系是：

`query_position >= key_position`

因此 Shape 可广播成：

`[2, 4, 4]`

再扩成 head 维度后应用到 `scores`。

这样第 0 个 token 只能看第 0 个位置，第 1 个 token 只能看第 0、1 个位置，以此类推。

### attention probabilities

`probs = softmax(scores, dim=-1)`

Shape 保持：

`[2, 8, 4, 4]`

### attention output

`probs @ expanded_value`

得到：

`[2, 8, 4, 16]`

---

## 6. merge heads 输出

`_merge_heads()` 会把：

`[2, 8, 4, 16]`

先转回：

`[2, 4, 8, 16]`

再 reshape 成：

`[2, 4, 128]`

最后通过 `o_proj`，输出 Shape 仍是：

`[2, 4, 128]`

这就回到了“每个 token 一个 `hidden_size` 向量”的标准布局。

---

## 7. Cache layout 与内存公式

Contiguous KV Cache 的布局是：

`[batch, num_kv_heads, max_seq_len, head_dim]`

若 `max_seq_len = 10`，则：

- `key.shape = [2, 2, 10, 16]`
- `value.shape = [2, 2, 10, 16]`

### 理论内存公式

单层 Attention 的 K+V 总字节数：

`2 * batch * num_kv_heads * max_seq_len * head_dim * bytes_per_element`

代入本例：

`2 * 2 * 2 * 10 * 16 * bytes_per_element = 1280 * bytes_per_element`

如果 `dtype=float32`，每个元素 4 字节，则总共是：

`5120 bytes`

### 为什么不是按 8 个 attention heads 算？

因为 cache 存的是 **KV heads**，不是 Query heads，也不是扩展后的计算视图。

这正是 GQA 的内存收益来源。

---

## 8. 单 token Decode 时的 Shape

假设已经 Prefill 了 4 个 token，现在 Decode 第 5 个 token（位置 4）。

当前输入 `hidden_states` Shape：

`[2, 1, 128]`

### 当前 token 的 Q/K/V

投影后：

- Q：`[2, 8, 1, 16]`
- K：`[2, 2, 1, 16]`
- V：`[2, 2, 1, 16]`

`positions` Shape：

`[2, 1]`

内容通常是：

```text
[[4],
 [4]]
```

### append 之后的 cache

若之前长度是 4，append 当前 token 后：

- `lengths = [5, 5]`
- 有效 `cached_key` / `cached_value` 读取范围是前 5 个位置

因此切片参与计算的 Key/Value Shape 是：

- `cached_key[:, :, :5, :] -> [2, 2, 5, 16]`
- `cached_value[:, :, :5, :] -> [2, 2, 5, 16]`

### 为 attention 计算做 GQA expansion

扩展后：

- Key：`[2, 8, 5, 16]`
- Value：`[2, 8, 5, 16]`

当前 Query 仍是：

- Query：`[2, 8, 1, 16]`

### Decode 的 attention scores

`scores` Shape：

`[2, 8, 1, 5]`

含义：

- 每个 batch
- 每个 Query head
- 当前这 1 个 query token
- 对 5 个可见 key positions 打分

### Decode 的输出

attention 输出：

`[2, 8, 1, 16]`

merge heads 后：

`[2, 1, 128]`

这正好对应“输出当前这个新 token 的 hidden state”。

---

## 9. 区分 Query heads、KV heads 与 `num_key_value_groups`

### Query heads

- 数量：`num_attention_heads = 8`
- 决定 Query 侧的注意力分工

### KV heads

- 数量：`num_key_value_heads = 2`
- 决定 K/V 投影宽度与 cache 存储规模

### `num_key_value_groups`

- 值：`8 / 2 = 4`
- 含义：每个 KV head 会被 4 个 Query heads 共享

所以在 GQA 中，并不是“每个 Query head 都有自己独立的 K/V head”。
而是多个 Query heads 共用同一个 KV head 组。

---

## 小结

本例中最值得记住的三点是：

1. Q 的 Shape 由 `num_attention_heads` 决定，K/V 的 Shape 由 `num_key_value_heads` 决定。
2. `_expand_kv()` 只是计算视图扩展，不应改变 cache 的紧凑存储布局。
3. Decode 时虽然只输入 1 个 token，但它仍会对“全部有效历史位置 + 当前 token 自己”做 attention。
