from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, load_dataset


def _loader_name(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(f"Unsupported data extension: {suffix}")


def load_splits(data_config: dict[str, Any]) -> DatasetDict:
    files = {
        "train": data_config["train_file"],
        "validation": data_config["validation_file"],
        "test": data_config["test_file"],
    }
    missing = [path for path in files.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")
    loaders = {_loader_name(path) for path in files.values()}
    if len(loaders) != 1:
        raise ValueError("train/dev/test must use the same file type")
    return load_dataset(loaders.pop(), data_files=files)


def _schema(example: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    if isinstance(example.get("messages"), list):
        return "messages"
    if "instruction" in example and "output" in example:
        return "instruction"
    if "text" in example:
        return "text"
    raise ValueError(
        "Cannot infer schema. Expected `messages`, `instruction` + `output`, or `text`."
    )


def _conversation(
    example: dict[str, Any], schema: str, system_prompt: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]] | tuple[None, str]:
    if schema == "text":
        return None, str(example["text"])

    if schema == "messages":
        messages = example["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Each messages example must end with an assistant message")
        return messages[:-1], messages

    if schema == "instruction":
        prompt = str(example["instruction"])
        extra_input = str(example.get("input") or "").strip()
        if extra_input:
            prompt = f"{prompt}\n\n{extra_input}"
        prefix = []
        if system_prompt:
            prefix.append({"role": "system", "content": system_prompt})
        prefix.append({"role": "user", "content": prompt})
        full = prefix + [{"role": "assistant", "content": str(example["output"])}]
        return prefix, full

    raise ValueError(f"Unknown schema: {schema}")


def tokenize_splits(dataset: DatasetDict, tokenizer, data_config: dict[str, Any]) -> DatasetDict:
    max_length = int(data_config.get("max_length", 2048))
    requested_schema = data_config.get("schema", "auto")
    system_prompt = data_config.get("system_prompt", "")
    first = dataset["train"][0]
    schema = _schema(first, requested_schema)

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        prompt_messages, full = _conversation(example, schema, system_prompt)
        if schema == "text":
            encoded = tokenizer(
                full,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            encoded["labels"] = list(encoded["input_ids"])
            return encoded

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            full,
            tokenize=True,
            add_generation_prompt=False,
        )
        full_ids = full_ids[:max_length]
        prompt_length = min(len(prompt_ids), len(full_ids))
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": [-100] * prompt_length + full_ids[prompt_length:],
        }

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names,
        num_proc=int(data_config.get("num_proc", 1)),
        desc=f"Tokenizing ({schema} schema)",
    )
    before = {name: len(split) for name, split in tokenized.items()}
    tokenized = tokenized.filter(
        lambda row: any(label != -100 for label in row["labels"]),
        num_proc=int(data_config.get("num_proc", 1)),
        desc="Removing samples whose answer was fully truncated",
    )
    after = {name: len(split) for name, split in tokenized.items()}
    print(f"Dataset schema={schema}; examples before={before}; after={after}")
    if after["train"] == 0 or after["validation"] == 0 or after["test"] == 0:
        raise ValueError("At least one split is empty after tokenization")
    return tokenized


class CompletionCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Do not mutate dataset rows; Trainer may reuse Python objects across workers.
        labels = [feature["labels"] for feature in features]
        model_features = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        length = batch["input_ids"].shape[1]
        padded_labels = [label + [-100] * (length - len(label)) for label in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch
