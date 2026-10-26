# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   L = sequence length,
#   D = model dimension,
#   H = attention heads,
#   K = query/key head dimension, V = value head dimension.

from dataclasses import dataclass
import torch


class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w_q_DD = torch.nn.Parameter(torch.empty(self.dim, self.dim))
        self.w_k_DD = torch.nn.Parameter(torch.empty(self.dim, self.dim))
        self.w_v_DD = torch.nn.Parameter(torch.empty(self.dim, self.dim))
        torch.nn.init.xavier_uniform_(self.w_q_DD)
        torch.nn.init.xavier_uniform_(self.w_k_DD)
        torch.nn.init.xavier_uniform_(self.w_v_DD)


    def forward(
        self,
        q_LD: torch.Tensor,
        k_LD: torch.Tensor,
        v_LD: torch.Tensor,
    ) -> torch.Tensor:
        q_LD = q_LD @ self.w_q_DD
        k_LD = k_LD @ self.w_k_DD
        v_LD = v_LD @ self.w_v_DD

        attn_LL = torch.matmul(q_LD, k_LD.transpose(-2, -1))
        attn_LL = attn_LL / (self.dim ** 0.5)
        attn_LL = torch.softmax(attn_LL, dim=-1)
        return torch.matmul(attn_LL, v_LD)
