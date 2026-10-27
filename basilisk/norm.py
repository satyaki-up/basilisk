# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension.

import torch


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w_gamma_D = torch.nn.Parameter(torch.ones(self.dim))


    def forward(
        self,
        x_BLD: torch.Tensor,
    ) -> torch.Tensor:
        rms_BL = torch.sqrt(x_BLD.square().mean(dim=-1, keepdim=True) + 1e-5)  # (B, L, 1)
        return self.w_gamma_D * x_BLD / rms_BL  # (B, L, D)
