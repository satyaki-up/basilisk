# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = MLP hidden dimension.

from einops import rearrange
import torch

class MLP(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w_1_DD = torch.nn.Parameter(torch.empty(self.dim, self.dim))
        self.w_2_DD = torch.nn.Parameter(torch.empty(self.dim, self.dim))
        torch.nn.init.xavier_uniform_(self.w_1_DD)
        torch.nn.init.xavier_uniform_(self.w_2_DD)

    def forward(
        self,
        x_BLD: torch.Tensor,
    ) -> torch.Tensor:
        x_BLD = x_BLD @ self.w_1_DD
        x_BLD = torch.nn.functional.silu(x_BLD)
        x_BLD = x_BLD @ self.w_2_DD
        return x_BLD