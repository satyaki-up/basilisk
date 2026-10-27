from collections.abc import Iterable

import torch


class AdamW:

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        self.params = list(params)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        # Keyed by id(param), because a Parameter cannot be a dict key.
        # Each entry holds step (int) plus the moment buffers exp_avg and exp_avg_sq.
        self.state: dict[int, dict[str, torch.Tensor | int]] = {}


    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None


    @torch.no_grad()
    def step(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")

            state = self.state.get(id(param))
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(param),
                    "exp_avg_sq": torch.zeros_like(param),
                }
                self.state[id(param)] = state

            grad = param.grad
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if self.weight_decay != 0:
                param.mul_(1 - self.lr * self.weight_decay)

            exp_avg.lerp_(grad, 1 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            bias_correction1 = 1 - self.beta1 ** state["step"]
            bias_correction2 = 1 - self.beta2 ** state["step"]
            step_size = self.lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(self.eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)
