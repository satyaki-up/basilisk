# Trains a Transformer on the documents written by scripts/prep_data.py.
# Each document is one batch of shape (1, L). Lengths differ, so documents
# are not stacked.

# Example usage: uv run python examples/train_gpt.py --limit 100
import argparse
from pathlib import Path

import torch

from basilisk.trainer import Trainer, get_device
from basilisk.transformer import Transformer


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pile-10k.pt"
DIM = 64
N_HEADS = 4
HEAD_DIM = 16
N_BLOCKS = 2


def load_documents(path: Path) -> tuple[list[torch.Tensor], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run: uv run python scripts/prep_data.py")
    saved = torch.load(path, weights_only=True)
    documents = saved["tokens"]
    if not documents:
        raise ValueError(f"{path} contains no documents")
    return documents, int(saved["vocab_size"])


def document_batches(documents: list[torch.Tensor], limit: int | None):
    if limit is not None:
        documents = documents[:limit]
    for document in documents:
        if document.ndim != 1:
            raise ValueError(f"Expected a document of shape (L,), got {tuple(document.shape)}")
        if document.numel() < 2:
            continue
        yield document.unsqueeze(0)  # (1, L)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer on tokenized pile-10k documents.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be at least 1, got {args.limit}")

    documents, vocab_size = load_documents(args.data)
    device = get_device()
    torch.manual_seed(0)
    model = Transformer(
        dim=DIM,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        n_blocks=N_BLOCKS,
        vocab_size=vocab_size,
    ).to(device)
    trainer = Trainer(model)
    for step, tokens_BL in enumerate(document_batches(documents, args.limit), start=1):
        loss = trainer.step(tokens_BL.to(device))
        print(f"step {step} loss {loss.item():.4f} length {tokens_BL.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
