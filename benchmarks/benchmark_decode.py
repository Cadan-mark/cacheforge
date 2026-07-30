from __future__ import annotations

import argparse
import statistics
import time

import torch

from cacheforge.attention import LlamaSelfAttention
from cacheforge.cache.contiguous import ContiguousKVCache
from cacheforge.config import AttentionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Microbenchmark for full-context recompute vs cached decode "
            "on a single attention module."
        )
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = getattr(torch, name)
    if device.type == "cpu" and dtype == torch.float16:
        raise SystemExit("float16 is not a useful CPU benchmark dtype; use float32 or bfloat16")
    return dtype


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_full_recompute(
    attention: LlamaSelfAttention,
    hidden_states: torch.Tensor,
    prompt_len: int,
    decode_steps: int,
    device: torch.device,
) -> float:
    sync_if_needed(device)
    start = time.perf_counter()
    for step in range(decode_steps):
        _ = attention(hidden_states[:, : prompt_len + step + 1])[:, -1:, :]
    sync_if_needed(device)
    return (time.perf_counter() - start) * 1000.0


def benchmark_cached_decode(
    attention: LlamaSelfAttention,
    hidden_states: torch.Tensor,
    config: AttentionConfig,
    prompt_len: int,
    decode_steps: int,
    device: torch.device,
) -> tuple[float, ContiguousKVCache]:
    cache = ContiguousKVCache(
        batch_size=hidden_states.shape[0],
        num_kv_heads=config.num_key_value_heads,
        max_seq_len=config.max_seq_len,
        head_dim=config.head_dim,
        dtype=hidden_states.dtype,
        device=device,
    )
    sync_if_needed(device)
    start = time.perf_counter()
    _ = attention.prefill(hidden_states[:, :prompt_len], cache)
    for step in range(decode_steps):
        position = prompt_len + step
        _ = attention.decode(
            hidden_states[:, position : position + 1],
            cache,
            positions=torch.full(
                (hidden_states.shape[0], 1),
                position,
                device=device,
                dtype=torch.long,
            ),
        )
    sync_if_needed(device)
    return (time.perf_counter() - start) * 1000.0, cache


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    total_len = args.prompt_len + args.decode_steps
    config = AttentionConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        max_seq_len=total_len,
    )

    torch.manual_seed(0)
    attention = LlamaSelfAttention(config).to(device=device, dtype=dtype).eval()
    hidden_states = torch.randn(
        args.batch_size,
        total_len,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.warmup):
        benchmark_full_recompute(
            attention,
            hidden_states,
            args.prompt_len,
            args.decode_steps,
            device,
        )
        benchmark_cached_decode(
            attention,
            hidden_states,
            config,
            args.prompt_len,
            args.decode_steps,
            device,
        )

    full_latencies = []
    cached_latencies = []
    cache = None
    for _ in range(args.repeats):
        full_latencies.append(
            benchmark_full_recompute(
                attention,
                hidden_states,
                args.prompt_len,
                args.decode_steps,
                device,
            )
        )
        cached_latency, cache = benchmark_cached_decode(
            attention,
            hidden_states,
            config,
            args.prompt_len,
            args.decode_steps,
            device,
        )
        cached_latencies.append(cached_latency)

    assert cache is not None
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    theoretical_cache_bytes = (
        2
        * args.batch_size
        * config.num_key_value_heads
        * config.max_seq_len
        * config.head_dim
        * bytes_per_element
    )

    print("CacheForge Milestone 1 attention microbenchmark")
    print("This compares full-context recomputation against prefill + contiguous KV-cached decode.")
    print("It is not an end-to-end LLM or vLLM benchmark.\n")
    print(f"device={device} dtype={dtype} batch_size={args.batch_size}")
    print(
        f"prompt_len={args.prompt_len} decode_steps={args.decode_steps} "
        f"hidden_size={args.hidden_size} heads={args.num_heads}/{args.num_kv_heads}"
    )
    print()
    print(f"{'mode':<24}{'mean_ms':>12}{'median_ms':>12}{'per_step_ms':>14}")
    print("-" * 62)
    print(
        f"{'full_recompute':<24}{statistics.mean(full_latencies):>12.3f}"
        f"{statistics.median(full_latencies):>12.3f}"
        f"{statistics.mean(full_latencies) / args.decode_steps:>14.3f}"
    )
    print(
        f"{'prefill_plus_cached':<24}{statistics.mean(cached_latencies):>12.3f}"
        f"{statistics.median(cached_latencies):>12.3f}"
        f"{statistics.mean(cached_latencies) / args.decode_steps:>14.3f}"
    )
    print()
    print(f"theoretical_kv_cache_bytes={theoretical_cache_bytes}")
    print(f"allocated_kv_cache_bytes={cache.allocated_bytes()}")
    if device.type == "cuda":
        print(f"cuda_peak_allocated_bytes={torch.cuda.max_memory_allocated(device)}")


if __name__ == "__main__":
    main()
