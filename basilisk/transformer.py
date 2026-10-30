# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = MLP hidden dimension,
#   H = attention heads,
#   K = query/key head dimension, V = value head dimension.

import torch

from basilisk.embedding import Embedding
from basilisk.norm import RMSNorm
from basilisk.rope import Rotary
from basilisk.transformer_block import TransformerBlock


class Transformer(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_blocks: int,
        vocab_size: int,
        dropout: float = 0.0,
        tensor_parallel: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, dim)
        self.rotary = Rotary(head_dim)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, head_dim, dropout, tensor_parallel)
                for _ in range(n_blocks)
            ]
        )
        self.norm = RMSNorm(dim)
        self.w_D_vocab = torch.nn.Parameter(torch.empty(dim, vocab_size))  # (D, vocab)
        torch.nn.init.xavier_uniform_(self.w_D_vocab)


    def forward(
        self,
        tokens_BL: torch.Tensor,
    ) -> torch.Tensor:
        x_BLD = self.embedding(tokens_BL)  # (B, L, D)
        for block in self.blocks:
            x_BLD = block(x_BLD, self.rotary)  # (B, L, D)
        x_BLD = self.norm(x_BLD)  # (B, L, D)
        return x_BLD @ self.w_D_vocab  # (B, L, vocab)
