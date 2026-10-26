# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   H = attention heads,
#   K = query/key head dimension, V = value head dimension.

from dataclasses import dataclass
import torch


class MultiHeadAttention():

    @dataclass(kw_only=True, slots=True)
    class Config():
        dim: int
        n_heads: int | None = None
        head_dim: int | None = None

        def __post_init__(self) -> None:
            if self.head_dim is None and self.dim % self.n_heads != 0:
                raise ValueError(
                    f"dim ({self.dim}) must be divisible by n_heads "
                    f"({self.n_heads}) when head_dim is not specified"
                )


    def __init__(self, config: Config):
        self.n_heads = config.n_heads
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else config.dim // config.n_heads
        )


    def forward(
        self,
        x_D: torch.Tensor,
    ) -> torch.Tensor:

        return x_D