from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


class ContiguousKVCache:
    """Preallocated K/V cache with layout [batch, num_kv_heads, max_seq_len, head_dim]."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.key = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device
        )
        self.value = torch.zeros_like(self.key)
        self.lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    def allocated_bytes(self) -> int:
        return (self.key.untyped_storage().nbytes() + self.value.untyped_storage().nbytes())

    def reset(self, batch_indices: int | Sequence[int] | Tensor | None = None) -> None:
        indices = self._normalize_batch_indices(batch_indices)
        self.key[indices] = 0
        self.value[indices] = 0
        self.lengths[indices] = 0

    def write(
        self,
        *,
        batch_indices: int | Sequence[int] | Tensor,
        start_positions: int | Sequence[int] | Tensor,
        key: Tensor,
        value: Tensor,
    ) -> None:
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        if key.ndim != 4:
            raise ValueError("key/value must have shape [batch, num_kv_heads, seq_len, head_dim]")
        if key.shape[1] != self.num_kv_heads:
            raise ValueError("key/value num_kv_heads does not match cache")
        if key.shape[3] != self.head_dim:
            raise ValueError("key/value head_dim does not match cache")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("key/value dtype must match cache dtype")
        if key.device != self.device or value.device != self.device:
            raise ValueError("key/value device must match cache device")

        indices = self._normalize_batch_indices(batch_indices)
        starts = self._normalize_start_positions(start_positions, indices.numel())
        if key.shape[0] != indices.numel():
            raise ValueError("batch dimension of key/value must match batch_indices")

        seq_len = key.shape[2]
        for row, slot in enumerate(indices.tolist()):
            start = int(starts[row].item())
            end = start + seq_len
            if start < 0:
                raise ValueError("start_positions must be non-negative")
            if end > self.max_seq_len:
                raise IndexError("write exceeds max_seq_len")
            current_length = int(self.lengths[slot].item())
            if start > current_length:
                raise ValueError("writes may not leave unwritten gaps in the cache")
            self.key[slot, :, start:end, :] = key[row]
            self.value[slot, :, start:end, :] = value[row]
            self.lengths[slot] = max(current_length, end)

    def append(
        self,
        *,
        batch_indices: int | Sequence[int] | Tensor,
        positions: int | Sequence[int] | Tensor,
        key: Tensor,
        value: Tensor,
    ) -> None:
        if key.ndim != 4 or key.shape[2] != 1:
            raise ValueError("append expects key/value with seq_len == 1")
        indices = self._normalize_batch_indices(batch_indices)
        starts = self._normalize_start_positions(positions, indices.numel())
        for row, slot in enumerate(indices.tolist()):
            if int(starts[row].item()) != int(self.lengths[slot].item()):
                raise ValueError("append position must equal the current sequence length")
        self.write(batch_indices=indices, start_positions=starts, key=key, value=value)

    def view(
        self, batch_indices: int | Sequence[int] | Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        indices = self._normalize_batch_indices(batch_indices)
        return self.key[indices], self.value[indices], self.lengths[indices]

    def _normalize_batch_indices(
        self, batch_indices: int | Sequence[int] | Tensor | None
    ) -> Tensor:
        if batch_indices is None:
            result = torch.arange(self.batch_size, device=self.device)
        elif isinstance(batch_indices, int):
            result = torch.tensor([batch_indices], device=self.device)
        elif isinstance(batch_indices, Tensor):
            result = batch_indices.to(device=self.device, dtype=torch.long).flatten()
        else:
            result = torch.tensor(list(batch_indices), device=self.device, dtype=torch.long)
        if result.numel() == 0:
            raise ValueError("batch_indices cannot be empty")
        if (result < 0).any() or (result >= self.batch_size).any():
            raise IndexError("batch_indices out of bounds")
        return result

    def _normalize_start_positions(
        self, start_positions: int | Sequence[int] | Tensor, expected: int
    ) -> Tensor:
        if isinstance(start_positions, int):
            result = torch.full((expected,), start_positions, device=self.device, dtype=torch.long)
        elif isinstance(start_positions, Tensor):
            result = start_positions.to(device=self.device, dtype=torch.long).flatten()
        else:
            result = torch.tensor(list(start_positions), device=self.device, dtype=torch.long)
        if result.numel() != expected:
            raise ValueError("start_positions must match the number of batch indices")
        return result
