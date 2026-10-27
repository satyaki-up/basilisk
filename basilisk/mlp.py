# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension,
#   M = MLP hidden dimension.

import torch


class MLP(torch.nn.Module):

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim if hidden_dim is None else hidden_dim
        self.w_gate_DM = torch.nn.Parameter(torch.empty(self.dim, self.hidden_dim))
        self.w_up_DM = torch.nn.Parameter(torch.empty(self.dim, self.hidden_dim))
        self.w_down_MD = torch.nn.Parameter(torch.empty(self.hidden_dim, self.dim))
        torch.nn.init.xavier_uniform_(self.w_gate_DM)
        torch.nn.init.xavier_uniform_(self.w_up_DM)
        torch.nn.init.xavier_uniform_(self.w_down_MD)


    def forward(
        self,
        x_BLD: torch.Tensor,
    ) -> torch.Tensor:
        gate_BLM = torch.nn.functional.silu(x_BLD @ self.w_gate_DM)  # (B, L, M)
        up_BLM = x_BLD @ self.w_up_DM  # (B, L, M)
        return (gate_BLM * up_BLM) @ self.w_down_MD  # (B, L, D)
