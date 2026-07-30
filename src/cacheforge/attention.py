from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from cacheforge.cache.contiguous import ContiguousKVCache
from cacheforge.config import AttentionConfig
from cacheforge.rope import RotaryEmbedding


class LlamaSelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, base=config.rope_theta)

    def forward(self, hidden_states: Tensor, positions: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        positions = self._normalize_positions(positions, batch_size, seq_len, hidden_states.device)
        query, key, value = self._project(hidden_states, positions)
        attended = self._attention(
            query=query,
            key=key,
            value=value,
            query_positions=positions,
            key_positions=positions,
        )
        return self._merge_heads(attended)

    def prefill(
        self,
        hidden_states: Tensor,
        cache: ContiguousKVCache,
        positions: Tensor | None = None,
        batch_indices: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        positions = self._normalize_positions(positions, batch_size, seq_len, hidden_states.device)
        self._validate_contiguous_positions(positions)
        slots = self._normalize_cache_batch_indices(cache, batch_indices, batch_size)
        query, key, value = self._project(hidden_states, positions)
        attended = self._attention(
            query=query,
            key=key,
            value=value,
            query_positions=positions,
            key_positions=positions,
        )
        cache.write(
            batch_indices=slots,
            start_positions=positions[:, 0],
            key=key,
            value=value,
        )
        return self._merge_heads(attended)

    def decode(
        self,
        hidden_states: Tensor,
        cache: ContiguousKVCache,
        positions: Tensor,
        batch_indices: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len != 1:
            raise ValueError("decode expects a single token per batch")
        positions = self._normalize_positions(positions, batch_size, seq_len, hidden_states.device)
        slots = self._normalize_cache_batch_indices(cache, batch_indices, batch_size)
        query, key, value = self._project(hidden_states, positions)
        cache.append(batch_indices=slots, positions=positions[:, 0], key=key, value=value)
        cached_key, cached_value, lengths = cache.view(slots)
        max_length = int(lengths.max().item())
        key_positions = torch.arange(max_length, device=hidden_states.device).unsqueeze(0).expand(
            batch_size, -1
        )
        attended = self._attention(
            query=query,
            key=cached_key[:, :, :max_length, :],
            value=cached_value[:, :, :max_length, :],
            query_positions=positions,
            key_positions=key_positions,
            key_lengths=lengths,
        )
        return self._merge_heads(attended)

    def _project(self, hidden_states: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim
        )
        query = self.rotary(query.transpose(1, 2), positions)
        key = self.rotary(key.transpose(1, 2), positions)
        value = value.transpose(1, 2)
        return query, key, value

    def _attention(
        self,
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_positions: Tensor,
        key_positions: Tensor,
        key_lengths: Tensor | None = None,
    ) -> Tensor:
        expanded_key = self._expand_kv(key)
        expanded_value = self._expand_kv(value)
        scale = 1.0 / math.sqrt(self.config.head_dim)
        scores = torch.matmul(query, expanded_key.transpose(-1, -2)) * scale
        mask = self._build_attention_mask(query_positions, key_positions, key_lengths, query.device)
        scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, expanded_value)

    def _expand_kv(self, tensor: Tensor) -> Tensor:
        if self.config.num_key_value_groups == 1:
            return tensor
        return tensor.repeat_interleave(self.config.num_key_value_groups, dim=1)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        merged = tensor.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.config.hidden_size,
        )
        return self.o_proj(merged)

    def _build_attention_mask(
        self,
        query_positions: Tensor,
        key_positions: Tensor,
        key_lengths: Tensor | None,
        device: torch.device,
    ) -> Tensor:
        if key_positions.ndim == 1:
            key_positions = key_positions.unsqueeze(0).expand(query_positions.shape[0], -1)
        causal_mask = query_positions.unsqueeze(-1) >= key_positions.unsqueeze(-2)
        if key_lengths is None:
            return causal_mask
        valid_keys = key_positions < key_lengths.to(device=device).unsqueeze(-1)
        return causal_mask & valid_keys.unsqueeze(1)

    def _normalize_positions(
        self,
        positions: Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        if positions is None:
            return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(
                batch_size, -1
            )
        if positions.ndim == 1:
            if positions.numel() != seq_len:
                raise ValueError("1D positions must match seq_len")
            return positions.to(device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        if positions.shape != (batch_size, seq_len):
            raise ValueError("positions must have shape [batch, seq_len]")
        return positions.to(device=device, dtype=torch.long)

    def _normalize_cache_batch_indices(
        self, cache: ContiguousKVCache, batch_indices: Tensor | None, expected_batch: int
    ) -> Tensor:
        if batch_indices is None:
            batch_indices = torch.arange(expected_batch, device=cache.device)
        if batch_indices.numel() != expected_batch:
            raise ValueError("batch_indices must match the batch size of hidden_states")
        return batch_indices.to(device=cache.device, dtype=torch.long)

    def _validate_contiguous_positions(self, positions: Tensor) -> None:
        expected = positions[:, :1] + torch.arange(positions.shape[1], device=positions.device)
        if not torch.equal(positions, expected):
            raise ValueError("positions for prefill must describe a contiguous range")
