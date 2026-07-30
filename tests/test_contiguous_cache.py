from __future__ import annotations

import pytest
import torch

from cacheforge.cache.contiguous import ContiguousKVCache


@pytest.fixture()
def cache() -> ContiguousKVCache:
    return ContiguousKVCache(
        batch_size=2,
        num_kv_heads=2,
        max_seq_len=8,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )


def test_multi_token_write_then_single_token_append(cache: ContiguousKVCache) -> None:
    prefill_key = torch.arange(24, dtype=torch.float32).view(1, 2, 3, 4)
    prefill_value = prefill_key + 100
    cache.write(batch_indices=[0], start_positions=[0], key=prefill_key, value=prefill_value)

    append_key = torch.full((1, 2, 1, 4), 7.0)
    append_value = torch.full((1, 2, 1, 4), 9.0)
    cache.append(batch_indices=[0], positions=[3], key=append_key, value=append_value)

    key, value, lengths = cache.view([0])
    assert lengths.tolist() == [4]
    torch.testing.assert_close(key[0, :, :3], prefill_key[0])
    torch.testing.assert_close(value[0, :, :3], prefill_value[0])
    torch.testing.assert_close(key[0, :, 3:4], append_key[0])
    torch.testing.assert_close(value[0, :, 3:4], append_value[0])


def test_batch_slots_do_not_overwrite_each_other(cache: ContiguousKVCache) -> None:
    key = torch.tensor(
        [
            [[[-1.0, -1.0, -1.0, -1.0]], [[-2.0, -2.0, -2.0, -2.0]]],
            [[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]]],
        ]
    )
    value = key + 10
    cache.write(batch_indices=[0, 1], start_positions=[0, 0], key=key, value=value)

    key_view, value_view, lengths = cache.view()
    assert lengths.tolist() == [1, 1]
    assert key_view[0, 0, 0, 0].item() == -1.0
    assert key_view[1, 0, 0, 0].item() == 1.0
    assert value_view[0, 1, 0, 0].item() == 8.0
    assert value_view[1, 1, 0, 0].item() == 12.0


def test_bounds_and_invalid_shape_errors(cache: ContiguousKVCache) -> None:
    good = torch.zeros(1, 2, 1, 4)
    with pytest.raises(ValueError, match="same shape"):
        cache.write(batch_indices=[0], start_positions=[0], key=good, value=torch.zeros(1, 2, 2, 4))
    with pytest.raises(IndexError, match="out of bounds"):
        cache.view([2])
    with pytest.raises(IndexError, match="exceeds max_seq_len"):
        cache.write(
            batch_indices=[0],
            start_positions=[8],
            key=good,
            value=good.clone(),
        )
    with pytest.raises(ValueError, match="dtype"):
        cache.write(
            batch_indices=[0],
            start_positions=[0],
            key=good.to(torch.float64),
            value=good.to(torch.float64),
        )
    with pytest.raises(ValueError, match="unwritten gaps"):
        cache.write(
            batch_indices=[0],
            start_positions=[2],
            key=torch.zeros(1, 2, 1, 4),
            value=torch.zeros(1, 2, 1, 4),
        )


def test_reset_clear_and_reuse(cache: ContiguousKVCache) -> None:
    tensor = torch.ones(1, 2, 2, 4)
    cache.write(batch_indices=[0], start_positions=[0], key=tensor, value=tensor)
    cache.reset([0])

    key, value, lengths = cache.view([0])
    assert lengths.tolist() == [0]
    assert torch.count_nonzero(key) == 0
    assert torch.count_nonzero(value) == 0

    cache.write(batch_indices=[0], start_positions=[0], key=tensor * 3, value=tensor * 4)
    key, value, lengths = cache.view([0])
    assert lengths.tolist() == [2]
    assert key[0, 0, 0, 0].item() == 3.0
    assert value[0, 0, 0, 0].item() == 4.0


def test_allocated_bytes_matches_theoretical_formula(cache: ContiguousKVCache) -> None:
    expected = 2 * 2 * 2 * 8 * 4 * torch.tensor([], dtype=torch.float32).element_size()
    assert cache.allocated_bytes() == expected
