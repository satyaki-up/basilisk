# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = MLP hidden dimension,
#   H = attention heads,
#   K = query/key head dimension, V = value head dimension.

import torch

from basilisk.rope import Rotary
from basilisk.transformer_block import TransformerBlock


class Transformer(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_blocks: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_blocks = n_blocks
        self.rotary = Rotary(head_dim)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, head_dim, dropout)
                for _ in range(n_blocks)
            ]
        )


    def forward(
        self,
        x_BLD: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            x_BLD = block(x_BLD, self.rotary)  # (B, L, D)
        return x_BLD
