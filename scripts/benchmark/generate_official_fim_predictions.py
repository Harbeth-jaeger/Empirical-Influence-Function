#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.benchmark_official_common import read_jsonl, sanitize_completion, write_json, write_jsonl  # noqa: E402


def load_requests(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if limit > 0:
        rows = rows[:limit]
    return rows


def resolve_context_limit(model: Any, tokenizer: Any, fallback: int) -> int:
    if fallback > 0:
        return fallback
    model_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not isinstance(model_limit, int) or model_limit <= 0:
        model_limit = getattr(tokenizer, "model_max_length", 0) or 0
    if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 100000:
        model_limit = 4096
    return int(model_limit)


def generated_suffixes(tokenizer: Any, outputs: Any, prompt_width: int) -> list[str]:
    suffixes: list[str] = []
    for row in outputs:
        decoded = tokenizer.decode(row[prompt_width:], skip_special_tokens=False)
        suffixes.append(sanitize_completion(decoded))
    return suffixes


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_input_tokens: int,
    max_new_tokens: int,
    num_return_sequences: int,
    temperature: float,
    top_p: float,
) -> list[list[str]]:
    import torch

    if not prompts:
        return []

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    prompt_width = int(inputs["input_ids"].shape[1])
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_id == tokenizer.unk_token_id:
        eos_id = tokenizer.eos_token_id

    do_sample = num_return_sequences > 1 or temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_id,
        "num_return_sequences": num_return_sequences,
        "remove_invalid_values": True,
        "renormalize_logits": True,
    }
    if do_sample:
        generation_kwargs.update({"do_sample": True, "temperature": max(temperature, 1e-5), "top_p": top_p})
    else:
        generation_kwargs.update({"do_sample": False, "num_beams": 1})

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    flat = generated_suffixes(tokenizer, outputs, prompt_width)
    grouped: list[list[str]] = []
    for i in range(0, len(flat), num_return_sequences):
        grouped.append(flat[i : i + num_return_sequences])
    return grouped


def iter_chunks(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    batch_size = max(1, batch_size)
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def command_generate(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_requests(args.input_path, args.limit)
    if not rows:
        raise SystemExit(f"No inference requests found: {args.input_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    max_input_tokens = resolve_context_limit(model, tokenizer, args.max_input_tokens)
    out_rows: list[dict[str, Any]] = []
    for batch in tqdm(iter_chunks(rows, args.batch_size), desc="generate"):
        prompts = [str(row["prompt"]) for row in batch]
        max_new_tokens = max(int(row.get("max_new_tokens") or args.max_new_tokens) for row in batch)
        predictions = generate_batch(
            model,
            tokenizer,
            prompts,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for row, preds in zip(batch, predictions):
            out_rows.append(
                {
                    "uid": row["uid"],
                    "prediction": preds[0] if preds else "",
                    "predictions": preds,
                    "model_name_or_path": args.model_name_or_path,
                }
            )

    n = write_jsonl(args.output_path, out_rows)
    report = {
        "input": str(args.input_path),
        "output": str(args.output_path),
        "model_name_or_path": args.model_name_or_path,
        "rows": n,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_return_sequences": args.num_return_sequences,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    write_json(args.output_path.with_suffix(args.output_path.suffix + ".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate predictions for prepared official FIM benchmark requests.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=0, help="0 means infer from model/tokenizer.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.set_defaults(func=command_generate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
