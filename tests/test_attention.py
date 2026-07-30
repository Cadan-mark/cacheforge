from __future__ import annotations

import torch

from cacheforge.attention import LlamaSelfAttention
from cacheforge.cache.contiguous import ContiguousKVCache
from cacheforge.config import AttentionConfig


def _make_inputs(config: AttentionConfig, seq_len: int) -> torch.Tensor:
    torch.manual_seed(0)
    values = torch.linspace(-1.5, 1.5, steps=seq_len * config.hidden_size, dtype=torch.float32)
    return values.view(1, seq_len, config.hidden_size)


def _make_attention(config: AttentionConfig) -> LlamaSelfAttention:
    torch.manual_seed(1234)
    attention = LlamaSelfAttention(config)
    attention.eval()
    return attention


def _make_cache(config: AttentionConfig, batch_size: int = 1) -> ContiguousKVCache:
    return ContiguousKVCache(
        batch_size=batch_size,
        num_kv_heads=config.num_key_value_heads,
        max_seq_len=config.max_seq_len,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device="cpu",
    )


def test_full_context_matches_prefill_then_cached_decode() -> None:
    config = AttentionConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_seq_len=8,
    )
    attention = _make_attention(config)
    hidden_states = _make_inputs(config, seq_len=6)

    full = attention(hidden_states)
    cache = _make_cache(config)
    prefill = attention.prefill(hidden_states[:, :4], cache)
    torch.testing.assert_close(prefill, full[:, :4], atol=1e-5, rtol=1e-5)

    for position in range(4, 6):
        decoded = attention.decode(
            hidden_states[:, position : position + 1],
            cache,
            positions=torch.tensor([[position]], dtype=torch.long),
        )
        torch.testing.assert_close(decoded, full[:, position : position + 1], atol=1e-5, rtol=1e-5)


def test_grouped_query_attention_shapes_and_equivalence() -> None:
    config = AttentionConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_seq_len=8,
    )
    attention = _make_attention(config)
    hidden_states = _make_inputs(config, seq_len=5)
    full = attention(hidden_states)

    cache = _make_cache(config)
    prefill = attention.prefill(hidden_states[:, :3], cache)
    assert prefill.shape == (1, 3, 128)
    assert cache.key.shape == (1, 2, 8, 16)
    assert cache.value.shape == (1, 2, 8, 16)
    torch.testing.assert_close(prefill, full[:, :3], atol=1e-5, rtol=1e-5)

    for position in range(3, 5):
        decoded = attention.decode(
            hidden_states[:, position : position + 1],
            cache,
            positions=torch.tensor([[position]], dtype=torch.long),
        )
        torch.testing.assert_close(decoded, full[:, position : position + 1], atol=1e-5, rtol=1e-5)


def test_config_validation_rejects_invalid_gqa_ratio() -> None:
    try:
        AttentionConfig(
            hidden_size=96,
            num_attention_heads=6,
            num_key_value_heads=4,
            max_seq_len=8,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected invalid GQA ratio to raise ValueError")
