from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionConfig:
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
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads for GQA"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even so rotary embeddings can rotate pairs")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
