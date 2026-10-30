import torch
import torch.distributed as dist


def _rank_and_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class _SumOutputs(torch.autograd.Function):

    @staticmethod
    def forward(ctx, partial_BLD: torch.Tensor) -> torch.Tensor:
        out_BLD = partial_BLD.detach().clone()
        dist.all_reduce(out_BLD, op=dist.ReduceOp.SUM)
        return out_BLD

    @staticmethod
    def backward(ctx, grad_BLD: torch.Tensor) -> torch.Tensor:
        return grad_BLD


class _SumInputGrads(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_BLD: torch.Tensor) -> torch.Tensor:
        return x_BLD

    @staticmethod
    def backward(ctx, grad_BLD: torch.Tensor) -> torch.Tensor:
        out_BLD = grad_BLD.detach().clone()
        dist.all_reduce(out_BLD, op=dist.ReduceOp.SUM)
        return out_BLD
