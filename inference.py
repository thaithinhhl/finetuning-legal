from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained LoRA adapter")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default="Bạn là trợ lý pháp luật hữu ích, chính xác và thận trọng.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("BF16 inference requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer_tokens = output[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(answer_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
