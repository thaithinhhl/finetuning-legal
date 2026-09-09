from __future__ import annotations

import os
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def compute_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"Unsupported compute dtype: {name}")
    if name == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("GPU does not support bf16; falling back to float16")
        return torch.float16
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


def load_qlora_model(config: dict[str, Any]):
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA 4-bit training requires a supported accelerator; CUDA is recommended")

    model_config = config["model"]
    qlora = config["qlora"]
    training = config["training"]
    dtype = compute_dtype(qlora.get("compute_dtype", "bfloat16"))
    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(qlora.get("load_in_4bit", True)),
        bnb_4bit_quant_type=qlora.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(qlora.get("double_quant", True)),
        bnb_4bit_compute_dtype=dtype,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    kwargs = {}
    if model_config.get("use_flash_attention_2", False):
        kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map={"": local_rank},
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        **kwargs,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
    )
    lora_config = LoraConfig(
        r=int(qlora["r"]),
        lora_alpha=int(qlora["alpha"]),
        lora_dropout=float(qlora.get("dropout", 0.05)),
        target_modules=qlora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, dtype
