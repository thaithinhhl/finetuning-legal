from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed

from src.legal_ft.config import load_config
from src.legal_ft.data import CompletionCollator, load_splits, tokenize_splits
from src.legal_ft.model import load_qlora_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with 4-bit QLoRA")
    parser.add_argument("--config", default="configs/qwen2_5_7b_qlora.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["training"]
    set_seed(int(train_cfg.get("seed", 42)))

    tokenizer = load_tokenizer(config["model"])
    raw_data = load_splits(config["data"])
    data = tokenize_splits(raw_data, tokenizer, config["data"])
    model, dtype = load_qlora_model(config)
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bf16 = dtype == torch.bfloat16
    eval_strategy = train_cfg.get("eval_strategy", "epoch")
    save_strategy = train_cfg.get("save_strategy", eval_strategy)
    if eval_strategy != save_strategy:
        raise ValueError("eval_strategy and save_strategy must match when loading the best model")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy=eval_strategy,
        eval_steps=int(train_cfg.get("eval_steps", 100)),
        save_strategy=save_strategy,
        save_steps=int(train_cfg.get("save_steps", 100)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        optim="paged_adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        report_to=train_cfg.get("report_to", "none"),
        seed=int(train_cfg.get("seed", 42)),
        data_seed=int(train_cfg.get("seed", 42)),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=CompletionCollator(tokenizer),
    )
    train_result = trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(output_dir / "best_adapter"))
    tokenizer.save_pretrained(str(output_dir / "best_adapter"))

    validation_metrics = trainer.evaluate(data["validation"], metric_key_prefix="validation")
    test_metrics = trainer.evaluate(data["test"], metric_key_prefix="test")
    for result, prefix in ((validation_metrics, "validation"), (test_metrics, "test")):
        loss = result.get(f"{prefix}_loss")
        if loss is not None:
            result[f"{prefix}_perplexity"] = math.exp(min(loss, 20))
    metrics = {**train_result.metrics, **validation_metrics, **test_metrics}
    trainer.log_metrics("all", metrics)
    trainer.save_metrics("all", metrics)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
