from __future__ import annotations

import os
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_dtype(name: str, *, require_cuda: bool = True) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"Unsupported compute dtype: {name}")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("BF16 LoRA training requires a CUDA GPU")
    if name == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; set model.dtype to float16")
    return values[name]


def load_tokenizer(model_config: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_lora_model(config: dict[str, Any]):
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 LoRA training requires a CUDA GPU")

    model_config = config["model"]
    lora = config["lora"]
    training = config["training"]
    dtype = compute_dtype(model_config.get("dtype", "bfloat16"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    attn_implementation = model_config.get("attn_implementation", "sdpa")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        torch_dtype=dtype,
        device_map={"": local_rank},
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if bool(training.get("gradient_checkpointing", True)):
        # Required because the frozen embedding output otherwise has no gradient.
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, dtype
