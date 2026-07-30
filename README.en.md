# CacheForge

[中文 README](README.md) | **English README**

CacheForge Milestone 1 is a small, educational PyTorch project for understanding Llama-style causal self-attention, rotary positional embeddings (RoPE), grouped-query attention (GQA), and a contiguous KV cache.

## Milestone 1 scope

Included in this milestone:

- A validated attention config for a single Llama-style attention module
- Rotary positional embeddings using absolute token positions
- Full-context causal self-attention and cached prefill + one-token decode
- Grouped-query attention where query heads can exceed KV heads
- A preallocated contiguous KV cache storing only KV heads
- Deterministic CPU-first tests and a small benchmark script

Explicitly not included yet:

- PagedAttention
- Continuous batching / schedulers
- KV-cache quantization
- Triton kernels or custom CUDA kernels
- Hugging Face model loading
- vLLM integration

## Installation

```bash
python -m pip install -e .[dev]
```

## Tests

```bash
pytest
```

## Benchmark

Run the attention microbenchmark on CPU:

```bash
PYTHONPATH=src python benchmarks/benchmark_decode.py --device cpu
```

If CUDA is available:

```bash
PYTHONPATH=src python benchmarks/benchmark_decode.py --device cuda --dtype float16
```

Useful options:

- `--prompt-len`
- `--decode-steps`
- `--batch-size`
- `--hidden-size`
- `--num-heads`
- `--num-kv-heads`
- `--warmup`
- `--repeats`

The benchmark compares naive full-context recomputation at every decode step against prefill + contiguous KV-cached decode. It reports latency, theoretical KV-cache bytes, actual allocated bytes, and CUDA peak allocated memory when running on CUDA. It is a microbenchmark of this single attention component, not an end-to-end LLM or vLLM benchmark.

## Tensor shape walkthrough

For input hidden states shaped `[batch, seq_len, hidden_size]`:

- Q projection: `[batch, seq_len, hidden_size] -> [batch, num_attention_heads, seq_len, head_dim]`
- K projection: `[batch, seq_len, hidden_size] -> [batch, num_kv_heads, seq_len, head_dim]`
- V projection: `[batch, seq_len, hidden_size] -> [batch, num_kv_heads, seq_len, head_dim]`
- Cached K/V layout: `[batch, num_kv_heads, max_seq_len, head_dim]`
- GQA expansion for attention only: `[batch, num_kv_heads, seq_len, head_dim] -> [batch, num_attention_heads, seq_len, head_dim]`

The cache stores only KV heads, not expanded query heads.

## Prefill vs decode

- **Prefill** processes a prompt range with causal masking, writes the prompt K/V tensors into the cache, and returns prompt outputs.
- **Decode** processes one new token at its absolute position, appends one KV entry into the cache, and attends to the valid historical cache plus the new token itself.

This is why decode positions must keep increasing. Resetting decode positions to zero would produce incorrect RoPE phases and cause the cached path to diverge from the full-context reference.

## KV-cache memory formula

For this single attention module and cache layout:

`2 * batch * num_kv_heads * max_seq_len * head_dim * bytes_per_element`

- The leading `2` is for K and V
- A full transformer multiplies this by the number of attention layers

GQA reduces KV-cache memory because the cache scales with `num_kv_heads`, not `num_attention_heads`.

## Benchmark interpretation and limitations

- Cached decode avoids recomputing historical K/V tensors for every step, but it still performs the new token's Q/K/V projection, RoPE on the current token, score computation against valid cache positions, softmax, and the output projection.
- CUDA timing should be measured with synchronization; the benchmark script already synchronizes before and after timed regions so asynchronous kernel launches do not under-report latency.
- `theoretical_kv_cache_bytes` is the formula-level size of K and V storage only. `allocated_kv_cache_bytes` comes from PyTorch storage allocation and helps show what this implementation really reserves.
- `cuda_peak_allocated_bytes` is helpful for comparing prompt length or batch size changes, but it is still the memory footprint of this microbenchmark, not the full memory profile of a production serving stack.
- Very small CPU workloads may show little or no speedup because Python overhead, cache effects, and tiny matrix sizes can dominate the runtime.

## Current limitations

- Single-module educational implementation rather than a full transformer stack
- Decode API is one-token-at-a-time
- The contiguous cache disallows unwritten gaps and prioritizes correctness/readability over advanced scheduling
- No fused kernels or throughput-oriented optimizations

## Next milestones

1. Paged KV cache
2. Scheduler / continuous batching
3. KV-cache quantization
4. Profiling and Triton-style optimization only after correctness is locked down

## Learning documents

Detailed learning notes are currently maintained in Chinese:

- [Milestone 1 walkthrough (Chinese)](docs/milestone-1-walkthrough.zh-CN.md)
- [Tensor shapes guide (Chinese)](docs/tensor-shapes.zh-CN.md)
- [Benchmark guide (Chinese)](docs/benchmark-guide.zh-CN.md)
- [Bilingual glossary (Chinese)](docs/glossary.zh-CN.md)

## Validation performed

The intended validation flow for this milestone is:

```bash
ruff check .
pytest
PYTHONPATH=src python benchmarks/benchmark_decode.py --device cpu
```
