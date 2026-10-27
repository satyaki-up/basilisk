# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
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
        q_BLD: torch.Tensor,
        k_BLD: torch.Tensor,
        v_BLD: torch.Tensor,
    ) -> torch.Tensor:
        q_BLD = q_BLD @ self.w_q_DD
        k_BLD = k_BLD @ self.w_k_DD
        v_BLD = v_BLD @ self.w_v_DD

        attn_BLL = torch.matmul(q_BLD, k_BLD.transpose(-2, -1))
        attn_BLL = attn_BLL / (self.dim ** 0.5)
        attn_BLL = torch.softmax(attn_BLL, dim=-1)
        return torch.matmul(attn_BLL, v_BLD)


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
        x_BLD: torch.Tensor,
    ) -> torch.Tensor:
        q_BHLK = self._split_heads(x_BLD, self.w_q_DHK)
        k_BHLK = self._split_heads(x_BLD, self.w_k_DHK)
        v_BHLK = self._split_heads(x_BLD, self.w_v_DHK)

        attn_BHLL = torch.einsum("b h l k, b h s k -> b h l s", q_BHLK, k_BHLK)  # (B, H, L, L)
        attn_BHLL = attn_BHLL / (self.head_dim ** 0.5)
        attn_BHLL = torch.softmax(attn_BHLL, dim=-1)
        out_BHLK = torch.einsum("b h l s, b h s k -> b h l k", attn_BHLL, v_BHLK)  # (B, H, L, K)

        out_BLHK = rearrange(out_BHLK, "b h l k -> b l h k")  # (B, L, H, K)
        joined_BLH = rearrange(out_BLHK, "b l h k -> b l (h k)")  # (B, L, H * K)
        return joined_BLH @ rearrange(self.w_o_HKD, "h k d -> (h k) d")  # (B, L, D)


    def _split_heads(self, x_BLD: torch.Tensor, w_DHK: torch.Tensor) -> torch.Tensor:
        projected_BLH = x_BLD @ rearrange(w_DHK, "d h k -> d (h k)")  # (B, L, H * K)
        return rearrange(projected_BLH, "b l (h k) -> b h l k", h=self.n_heads)  # (B, H, L, K)
