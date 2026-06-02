from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from scripts.benchmark.eval_judges import decode_escaped_for_prompt, sanitize_prediction
except ModuleNotFoundError:
    from eval_judges import decode_escaped_for_prompt, sanitize_prediction


# FIX Bug3: 去掉 apply_chat_template，强制用与训练时 binarize 完全一致的手动 ChatML 拼接
def build_messages_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        if role not in {"system", "user"}:
            continue
        content = decode_escaped_for_prompt(str(msg.get("content", "")))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


@torch.inference_mode()
def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    out = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_id,
    )
    decoded = tokenizer.decode(out[0][input_len:], skip_special_tokens=False)
    return sanitize_prediction(decoded)


@torch.inference_mode()
def generate_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    try:
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
            remove_invalid_values=True,
            renormalize_logits=True,
        )
    except RuntimeError:
        out = model.generate(
            **inputs,
            do_sample=False,
            num_beams=max(2, num_samples),
            num_return_sequences=num_samples,
            early_stopping=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )

    results: list[str] = []
    for i in range(num_samples):
        decoded = tokenizer.decode(out[i][input_len:], skip_special_tokens=False)
        results.append(sanitize_prediction(decoded))
    return results


def _chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _resolve_generation_context_limit(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> int:
    model_limit = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(model_limit, int) or model_limit <= 0:
        model_limit = getattr(tokenizer, "model_max_length", 0) or 0
    if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 100000:
        model_limit = 4096
    return int(model_limit)


@torch.inference_mode()
def generate_batch_greedy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    max_input_tokens: int,
) -> list[str]:
    if not prompts:
        return []

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    out = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_id,
    )

    results: list[str] = []
    for i in range(len(prompts)):
        decoded = tokenizer.decode(out[i][prompt_len:], skip_special_tokens=False)
        results.append(sanitize_prediction(decoded))
    return results


@torch.inference_mode()
def generate_batch_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    max_input_tokens: int,
    num_samples: int,
    temperature: float,
    top_p: float,
) -> list[list[str]]:
    if not prompts:
        return []

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    try:
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
            remove_invalid_values=True,
            renormalize_logits=True,
        )
    except RuntimeError:
        out = model.generate(
            **inputs,
            do_sample=False,
            num_beams=max(2, num_samples),
            num_return_sequences=num_samples,
            early_stopping=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )

    batch_size = len(prompts)
    results: list[list[str]] = []
    for i in range(batch_size):
        preds_i: list[str] = []
        for j in range(num_samples):
            row_idx = i * num_samples + j
            decoded = tokenizer.decode(out[row_idx][prompt_len:], skip_special_tokens=False)
            preds_i.append(sanitize_prediction(decoded))
        results.append(preds_i)
    return results

