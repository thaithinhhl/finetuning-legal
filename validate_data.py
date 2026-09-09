from __future__ import annotations

import argparse
from collections import Counter

from src.legal_ft.config import load_config
from src.legal_ft.data import load_splits, tokenize_splits
from src.legal_ft.model import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize dataset splits")
    parser.add_argument("--config", default="configs/qwen2_5_7b_bf16_lora.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(config["model"])
    raw = load_splits(config["data"])
    tokenized = tokenize_splits(raw, tokenizer, config["data"])

    for name, split in tokenized.items():
        lengths = [len(row["input_ids"]) for row in split]
        trainable = [sum(label != -100 for label in row["labels"]) for row in split]
        print(
            f"{name}: n={len(split)}, tokens(mean/max)={sum(lengths)/len(lengths):.1f}/{max(lengths)}, "
            f"answer_tokens(mean/min)={sum(trainable)/len(trainable):.1f}/{min(trainable)}"
        )

    # Fast exact-duplicate leakage check on raw rows.
    signatures = {}
    for name, split in raw.items():
        signatures[name] = Counter(str(sorted(row.items())) for row in split)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(signatures[left]) & set(signatures[right])
        print(f"exact overlap {left}<->{right}: {len(overlap)} unique rows")


if __name__ == "__main__":
    main()
