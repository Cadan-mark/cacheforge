from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, tensor: Tensor, positions: Tensor) -> Tensor:
        return apply_rotary_embedding(tensor, positions, self.inv_freq)


def apply_rotary_embedding(tensor: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    # 输入保持 [batch, heads, seq_len, head_dim]，这样 positions 能直接按 token 位置广播。
    if tensor.ndim != 4:
        raise ValueError("tensor must have shape [batch, heads, seq_len, head_dim]")
    batch, _, seq_len, head_dim = tensor.shape
    if head_dim != inv_freq.numel() * 2:
        raise ValueError("head_dim does not match rotary frequencies")
    if positions.ndim == 1:
        if positions.numel() != seq_len:
            raise ValueError("1D positions must match seq_len")
        positions = positions.unsqueeze(0).expand(batch, -1)
    if positions.shape != (batch, seq_len):
        raise ValueError("positions must have shape [batch, seq_len]")

    # Decode 阶段必须传入 token 的绝对位置；否则历史缓存路径与 full-context 的 RoPE 相位会失配。
    angles = positions.to(device=tensor.device, dtype=inv_freq.dtype).unsqueeze(-1) * inv_freq.to(
        tensor.device
    )
    cos = angles.cos().unsqueeze(1).to(dtype=tensor.dtype)
    sin = angles.sin().unsqueeze(1).to(dtype=tensor.dtype)

    # RoPE 把偶/奇维度两两配对，视作二维向量后做旋转。
    first_half = tensor[..., ::2]
    second_half = tensor[..., 1::2]
    rotated_first = first_half * cos - second_half * sin
    rotated_second = first_half * sin + second_half * cos
    return torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)
