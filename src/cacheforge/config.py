from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for a single Llama-style attention module.

    保持 `hidden_size -> num_attention_heads -> head_dim` 的整除关系，
    这样后续的 Q/K/V reshape 与 RoPE 配对旋转才能严格成立。
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_seq_len: int
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        # `head_dim` 会被用于 [heads, head_dim] 的拆分；这里不允许出现不能整除的配置。
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        # GQA 依赖“多个 query heads 共享一个 KV head 组”，因此这里必须能整除。
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads for GQA"
            )
        # RoPE 会把最后一维两两配对做旋转，所以 head_dim 必须是偶数。
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even so rotary embeddings can rotate pairs")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        # 每个 KV head 需要被多少个 query heads 共享。
        return self.num_attention_heads // self.num_key_value_heads
