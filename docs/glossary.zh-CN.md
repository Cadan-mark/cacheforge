# 术语表 / Glossary

[返回中文 README](../README.md) | [English README](../README.en.md)

以下术语按本项目语境给出简洁的中英对照说明。Milestone 1 只实现其中一部分概念；未实现的概念会明确标注。

## Attention

**Attention（注意力）**：让当前 token 根据 Query / Key 的匹配关系，对不同位置的 Value 做加权汇总的机制。

## Causal Mask

**Causal Mask（因果掩码）**：限制第 `t` 个位置只能看到 `<= t` 的位置，防止模型在自回归生成时偷看未来 token。

## Query / Key / Value

**Query / Key / Value（查询 / 键 / 值）**：

- Query：当前要“发起查询”的表示
- Key：供 Query 匹配的表示
- Value：被加权汇总的内容表示

## Head / Head Dimension

**Head（注意力头）**：将注意力拆成多个并行子空间。

**Head Dimension（头维度）**：每个 head 的向量维度，在本项目中为：

`head_dim = hidden_size / num_attention_heads`

## MHA / GQA / MQA

- **MHA (Multi-Head Attention)**：Q/K/V 通常使用相同数量的 heads。
- **GQA (Grouped-Query Attention)**：Q heads 多于 KV heads；多个 Query heads 共享一组 K/V heads。
- **MQA (Multi-Query Attention)**：可视作 KV heads 极少（常见是 1）的特例。

Milestone 1 实现的是 **GQA-compatible** Attention。

## RoPE

**RoPE (Rotary Positional Embedding)**：通过对 Q/K 的维度对做旋转，把位置信息编码进向量相位中的方法。

## Prefill

**Prefill（提示阶段预填充）**：对整段 prompt 一次性计算 Attention，并把对应的 K/V 写入 KV Cache。

## Decode

**Decode（解码阶段）**：在已有 prompt / 历史缓存基础上，逐 token 生成新位置的输出。Milestone 1 当前实现的是 **one-token-at-a-time Decode**。

## Autoregressive Generation

**Autoregressive Generation（自回归生成）**：每一步都基于已有历史，生成下一个 token，再把它追加进序列。

## KV Cache

**KV Cache（键值缓存）**：把历史 token 的 Key / Value 保存起来，避免每次 Decode 都重算全部历史 K/V。

## Contiguous KV Cache

**Contiguous KV Cache（连续式 KV 缓存）**：按连续位置预分配并写入的缓存布局，通常形如：

`[batch, num_kv_heads, max_seq_len, head_dim]`

Milestone 1 实现的是这种最直接、最容易教学的形式。

## Paged KV Cache / PagedAttention

**Paged KV Cache / PagedAttention（分页式 KV 缓存 / 分页注意力）**：把 KV Cache 切成页（pages）来管理，便于更灵活地复用、回收与调度显存。

**注意：Milestone 1 没有实现 PagedAttention。** README 和文档中只把它当作后续里程碑概念。

## Continuous Batching

**Continuous Batching（连续批处理）**：在服务端动态把不同请求的 Prefill / Decode 步骤插入到共享执行批次中，以提升硬件利用率。

**注意：Milestone 1 没有实现 scheduler 或 continuous batching。**

## TTFT / TPOT

- **TTFT (Time To First Token)**：从请求开始到收到第一个输出 token 的时间。
- **TPOT (Time Per Output Token)**：生成阶段平均每个输出 token 的时间。

这两个术语在推理服务 benchmark 中很常见，但 Milestone 1 自带的脚本是 Attention microbenchmark，不直接产出 TTFT / TPOT。

## Memory-bound / Compute-bound

- **Memory-bound（受内存带宽限制）**：性能主要受数据搬运成本限制。
- **Compute-bound（受算力限制）**：性能主要受算术运算能力限制。

实际 Decode 路径经常同时受内存访问和计算影响，不能简单一概而论。

## Tensor shape / dtype / device

- **Tensor shape**：Tensor 各维度大小，例如 `[batch, heads, seq_len, head_dim]`
- **dtype**：数据类型，例如 `float32`、`float16`、`bfloat16`
- **device**：Tensor 所在设备，例如 CPU 或 CUDA GPU
