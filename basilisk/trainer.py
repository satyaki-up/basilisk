# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension.

import torch

from basilisk.distributed.data_parallel import DataParallel
from basilisk.optimizers.adamw import AdamW


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Trainer:

    def __init__(self, model: torch.nn.Module, parallel: DataParallel | None = None):
        self.device = get_device()
        self.model = model.to(self.device)
        self.parallel = parallel
        if parallel is not None:
            parallel.broadcast_parameters()
        self.optimizer = AdamW(self.model)


    def step(self, tokens_BL: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.train()
        tokens_BL = tokens_BL.to(self.device)
        # -100 is padding. The embedding table has no row for it
        inputs_BL = tokens_BL.clamp(min=0)
        logits = self.model(inputs_BL)  # (B, L, vocab)
        targets = tokens_BL[:, 1:]  # (B, L - 1)
        token_count = int((targets != -100).sum().item())
        if token_count < 1:
            raise ValueError("batch has no target tokens")
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),  # (B * (L - 1), vocab)
            targets.reshape(-1),  # (B * (L - 1),)
            ignore_index=-100,
        )
        self.optimizer.zero_grad()
        loss.backward()
        local_loss = loss.detach()
        if self.parallel is not None:
            loss = self.parallel.synchronize(loss, token_count)
        self.optimizer.step()
        return loss.detach(), local_loss


    def train(self, batches) -> None:
        for tokens_BL in batches:
            self.step(tokens_BL)
