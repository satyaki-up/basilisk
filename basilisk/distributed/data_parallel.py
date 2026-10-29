import torch
import torch.distributed as dist


class DataParallel:
    """Replica of one model per rank. Gradients are the mean over every rank's tokens."""

    def __init__(self, model: torch.nn.Module):
        if not dist.is_initialized():
            raise RuntimeError("Call init_process_group before DataParallel")
        self.model = model


    @torch.no_grad()
    def broadcast_parameters(self) -> None:
        # performed once at the start to distribute the model weights to all ranks
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        for buf in self.model.buffers():
            dist.broadcast(buf, src=0)


    @torch.no_grad()
    def synchronize(self, loss: torch.Tensor, token_count: int) -> torch.Tensor:
        grads = [param.grad for param in self.model.parameters() if param.grad is not None]
        if not grads:
            raise RuntimeError("backward produced no gradients")
        # One vector, so the reduction is a single collective instead of one per parameter.
        flat = torch.cat([grad.reshape(-1) for grad in grads])
        loss_sum = loss.detach().reshape(1) * token_count
        count = torch.tensor([token_count], dtype=flat.dtype, device=flat.device)
        # Gradient sums, then the summed loss, then the token count
        payload = torch.cat([flat * token_count, loss_sum, count])
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        total_tokens = payload[-1]
        if total_tokens.item() == 0:
            raise RuntimeError("no tokens contributed to the loss")
        reduced = payload[:-1] / total_tokens
        offset = 0
        for grad in grads:
            numel = grad.numel()
            grad.copy_(reduced[offset : offset + numel].view_as(grad))
            offset += numel
        return reduced[-1]
