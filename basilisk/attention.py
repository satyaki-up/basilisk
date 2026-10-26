# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   L = sequence length,
#   D = model dimension,
#   H = attention heads,
#   K = query/key head dimension, V = value head dimension.

from dataclasses import dataclass
from einops import rearrange
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


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.w_q_DHK = torch.nn.Parameter(torch.empty(dim, n_heads, head_dim))
        self.w_k_DHK = torch.nn.Parameter(torch.empty(dim, n_heads, head_dim))
        self.w_v_DHK = torch.nn.Parameter(torch.empty(dim, n_heads, head_dim))
        self.w_o_HKD = torch.nn.Parameter(torch.empty(n_heads, head_dim, dim))
        torch.nn.init.xavier_uniform_(rearrange(self.w_q_DHK, "d h k -> d (h k)"))
        torch.nn.init.xavier_uniform_(rearrange(self.w_k_DHK, "d h k -> d (h k)"))
        torch.nn.init.xavier_uniform_(rearrange(self.w_v_DHK, "d h k -> d (h k)"))
        torch.nn.init.xavier_uniform_(rearrange(self.w_o_HKD, "h k d -> (h k) d"))


    def forward(
        self,
        q_LD: torch.Tensor,
        k_LD: torch.Tensor,
        v_LD: torch.Tensor,
    ) -> torch.Tensor:
        q_HLK = self._split_heads(q_LD, self.w_q_DHK)
        k_HLK = self._split_heads(k_LD, self.w_k_DHK)
        v_HLK = self._split_heads(v_LD, self.w_v_DHK)

        attn_HLL = torch.matmul(q_HLK, rearrange(k_HLK, "h l k -> h k l"))  # (H, L, L)
        attn_HLL = attn_HLL / (self.head_dim ** 0.5)
        attn_HLL = torch.softmax(attn_HLL, dim=-1)
        out_HLK = torch.einsum("h l s, h s k -> h l k", attn_HLL, v_HLK)  # (H, L, K)

        out_LHK = rearrange(out_HLK, "h l k -> l h k")  # (L, H, K)
        joined_LH = rearrange(out_LHK, "l h k -> l (h k)")  # (L, H * K)
        return joined_LH @ rearrange(self.w_o_HKD, "h k d -> (h k) d")  # (L, D)


    def _split_heads(self, x_LD: torch.Tensor, w_DHK: torch.Tensor) -> torch.Tensor:
        projected_LH = x_LD @ rearrange(w_DHK, "d h k -> d (h k)")  # (L, H * K)
        return rearrange(projected_LH, "l (h k) -> h l k", h=self.n_heads)  # (H, L, K)
