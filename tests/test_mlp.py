import torch

from basilisk.mlp import MLP


def test_mlp_matches_torch():
    torch.manual_seed(0)
    dim = 4
    x_BLD = torch.randn(2, 3, dim)

    mlp = MLP(dim)
    reference = torch.nn.Sequential(
        torch.nn.Linear(dim, dim, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(dim, dim, bias=False),
    )
    with torch.no_grad():
        reference[0].weight.copy_(mlp.w_1_DD.T)
        reference[2].weight.copy_(mlp.w_2_DD.T)

    torch.testing.assert_close(mlp(x_BLD), reference(x_BLD))
