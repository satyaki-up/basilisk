# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length,
#   D = model dimension.

import torch

from basilisk.optimizers.adamw import AdamW


class Trainer:

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.optimizer = AdamW(model)


    def step(self, tokens_BL: torch.Tensor) -> torch.Tensor:
        self.model.train()
        logits = self.model(tokens_BL)  # (B, L, vocab)
        # Position t predicts token t + 1. Pad ids are -100 and drop out of the mean.
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),  # (B * (L - 1), vocab)
            tokens_BL[:, 1:].reshape(-1),  # (B * (L - 1),)
            ignore_index=-100,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()


    def train(self, batches) -> None:
        for tokens_BL in batches:
            self.step(tokens_BL)
