# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   H = attention heads,
#   K = query/key head dimension.

import torch


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (K / 2,)
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=2):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)  # (L,)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (L, K / 2)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)  # (L, K)
            self.cos_cached = emb.cos()[None, None, :, :]  # (1, 1, L, K)
            self.sin_cached = emb.sin()[None, None, :, :]  # (1, 1, L, K)
        return self.cos_cached, self.sin_cached  # (1, 1, L, K)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]  # (B, H, L, K / 2)
    return torch.cat(
        (-x2, x1), dim=-1
    )  # (B, H, L, K)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)  # (B, H, L, K)