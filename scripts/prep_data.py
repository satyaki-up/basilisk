# Downloads NeelNanda/pile-10k, tokenizes it with the Llama 2 tokenizer, and
# writes one tensor per document. Each document is <bos> tokens <eos>, truncated
# to MAX_SEQ_LEN.

# Example usage: uv run python scripts/prep_data.py --output data/pile-10k.pt
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import LlamaTokenizer


DATASET_NAME = "NeelNanda/pile-10k"
# Public Llama 2 tokenizer. Same 32k vocabulary as meta-llama/Llama-2-7b-hf.
TOKENIZER_NAME = "NousResearch/Llama-2-7b-hf"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "pile-10k.pt"
MAX_SEQ_LEN = 2048
TOKENIZE_BATCH = 32


def load_llama2_tokenizer() -> LlamaTokenizer:
    return LlamaTokenizer.from_pretrained(TOKENIZER_NAME)


def tokenize_texts(texts: list[str], tokenizer: LlamaTokenizer) -> list[torch.Tensor]:
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("Llama 2 tokenizer is missing a bos or eos id")

    documents: list[torch.Tensor] = []
    for start in range(0, len(texts), TOKENIZE_BATCH):
        encoded = tokenizer(
            texts[start : start + TOKENIZE_BATCH],
            add_special_tokens=False,
            truncation=True,
            # bos and eos are added below, so the stored row is at most MAX_SEQ_LEN.
            max_length=MAX_SEQ_LEN - 2,
            return_attention_mask=False,
        )
        for ids in encoded["input_ids"]:
            documents.append(torch.tensor([bos_id, *ids, eos_id], dtype=torch.long))  # (L,)
    if not documents:
        raise ValueError("Dataset produced no tokens")
    return documents


def prepare(output: Path = DEFAULT_OUTPUT) -> list[torch.Tensor]:
    dataset = load_dataset(DATASET_NAME, split="train")
    tokenizer = load_llama2_tokenizer()
    documents = tokenize_texts(list(dataset["text"]), tokenizer)
    vocab_size = tokenizer.vocab_size
    for document in documents:
        if int(document.min()) < 0 or int(document.max()) >= vocab_size:
            raise ValueError(
                f"Token ids {(int(document.min()), int(document.max()))} "
                f"fall outside vocab_size={vocab_size}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": documents, "vocab_size": vocab_size}, output)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize NeelNanda/pile-10k with the Llama 2 tokenizer.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    documents = prepare(args.output)
    lengths = [document.numel() for document in documents]
    print(
        f"saved {len(documents)} documents "
        f"(length {min(lengths)}-{max(lengths)}) to {args.output}"
    )


if __name__ == "__main__":
    main()
