#!/usr/bin/env python3
"""Compute clickable saliency for two model prediction dumps.

This script is intentionally inference-only. It consumes a manifest built from
``eval_dump.py`` outputs and runs each model once per selected sample under
teacher forcing on that model's displayed completion text.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.visual_prediction.compute_prediction_saliency import (  # noqa: E402
    alti_model_view,
    build_messages_prompt,
    display_tokens,
    encode_completion_tokens,
    encode_prompt_tokens,
    load_model_and_tokenizer,
    slugify,
    topk_from_saliency,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--model_a_path", required=True)
    p.add_argument("--model_b_path", required=True)
    p.add_argument("--model_a_name", default="Model A")
    p.add_argument("--model_b_name", default="Model B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_completion_tokens", type=int, default=120)
    p.add_argument("--max_saliency_tokens", type=int, default=80)
    p.add_argument("--max_seq_len", type=int, default=3072)
    p.add_argument("--alti_p", type=int, default=1)
    p.add_argument("--alti_chunk_size", type=int, default=8)
    p.add_argument("--cache_dir", default="outputs/visual_failure/dual_model_saliency_cache")
    p.add_argument("--overwrite_cache", action="store_true")
    p.add_argument("--partial_write_every", type=int, default=1)
    p.add_argument("--progress_every", type=int, default=20)
    p.add_argument("--local_files_only", type=int, default=1)
    p.add_argument(
        "--completion_source",
        choices=["prediction", "ground_truth"],
        default="prediction",
        help="Use each model prediction or the shared ground-truth completion as the saliency target text.",
    )
    return p.parse_args()


def cache_path(cache_dir: Path, model_key: str, sample_id: str) -> Path:
    return cache_dir / model_key / f"{slugify(sample_id)}.json"


def model_prediction(sample: dict[str, Any], model_key: str, completion_source: str) -> str:
    if completion_source == "ground_truth":
        return str(sample.get("ground_truth", ""))
    return str((sample.get(model_key) or {}).get("prediction", ""))


def compute_one_model_sample(
    *,
    model: Any,
    alti_model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    model_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch
    from src.attribution.saliency import compute_alti_saliency_vector

    prompt_text = build_messages_prompt(sample["row"].get("messages", []))
    prompt_ids, prompt_tokens = encode_prompt_tokens(prompt_text, tokenizer)
    if len(prompt_ids) >= args.max_seq_len:
        return {"skipped": True, "reason": f"prompt_len {len(prompt_ids)} >= max_seq_len {args.max_seq_len}"}

    completion_text = model_prediction(sample, model_key, args.completion_source)
    completion_ids, completion_tokens = encode_completion_tokens(
        completion_text,
        tokenizer,
        len(prompt_ids),
        args.max_completion_tokens,
    )
    if not completion_ids:
        return {"skipped": True, "reason": "empty completion"}

    full_ids = prompt_ids + completion_ids
    if len(full_ids) > args.max_seq_len:
        keep_completion = max(0, args.max_seq_len - len(prompt_ids))
        completion_ids = completion_ids[:keep_completion]
        completion_tokens = completion_tokens[:keep_completion]
        full_ids = prompt_ids + completion_ids
    if not completion_ids:
        return {"skipped": True, "reason": "completion truncated to zero"}

    prompt_display_indices = [t["idx"] for t in prompt_tokens if t["region"] in {"prefix", "suffix"}]
    all_tokens = prompt_tokens + completion_tokens
    tokens_by_idx = {int(t["idx"]): t for t in all_tokens}
    input_t = torch.tensor([full_ids], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_t, device=args.device)

    targets: dict[str, Any] = {}
    targets_by_step: dict[str, int] = {}
    max_steps = min(len(completion_tokens), args.max_saliency_tokens)
    for step, tok in enumerate(completion_tokens[:max_steps]):
        if step == 0 or (step + 1) % max(1, args.progress_every) == 0:
            print(
                f"[saliency] {model_key} {sample.get('sample_id')} token {step + 1}/{max_steps}",
                flush=True,
            )
        target_idx = int(tok["idx"])
        saliency = compute_alti_saliency_vector(
            alti_model,
            {"input_ids": input_t, "attention_mask": attention_mask},
            target_idx_in_seq=target_idx,
            p=args.alti_p,
            chunk_size=args.alti_chunk_size,
        )
        candidate_indices = [
            i
            for i in range(target_idx)
            if i in tokens_by_idx and (i in prompt_display_indices or i >= len(prompt_ids))
        ]
        targets[str(target_idx)] = {
            "target_idx": target_idx,
            "step": int(step),
            "token": tok["text"],
            "display": tok["display"],
            "sources": topk_from_saliency(saliency, tokens_by_idx, candidate_indices, args.top_k),
        }
        targets_by_step[str(step)] = target_idx

    return {
        "skipped": False,
        "prompt_len": len(prompt_ids),
        "completion_source": args.completion_source,
        "prediction": str((sample.get(model_key) or {}).get("prediction", "")),
        "displayed_completion": completion_text,
        "tokens": display_tokens(prompt_tokens, completion_tokens),
        "targets": targets,
        "targets_by_step": targets_by_step,
        "generated_token_indices": [int(t["idx"]) for t in completion_tokens[:max_steps]],
    }


def load_existing_or_compute(
    *,
    model: Any,
    alti_model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    model_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cp = cache_path(Path(args.cache_dir), model_key, str(sample.get("sample_id") or sample.get("uid")))
    if cp.exists() and not args.overwrite_cache:
        print(f"[cache hit] {model_key} {sample.get('sample_id')}", flush=True)
        return json.loads(cp.read_text(encoding="utf-8"))
    result = compute_one_model_sample(
        model=model,
        alti_model=alti_model,
        tokenizer=tokenizer,
        sample=sample,
        model_key=model_key,
        args=args,
    )
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    import torch

    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = list(manifest.get("samples", []))
    if args.limit > 0:
        samples = samples[: args.limit]

    payload: dict[str, Any] = {
        "version": 1,
        "title": "双模型预测与 saliency 对比",
        "subtitle": "查看两个模型在正确/失败样本上的预测差异，并点击 completion token 对比 saliency top-k source tokens",
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "model_a_path": args.model_a_path,
        "model_b_path": args.model_b_path,
        "completion_source": args.completion_source,
        "top_k": args.top_k,
        "categories": manifest.get("categories", {}),
        "category_counts": manifest.get("category_counts", {}),
        "samples": [],
    }

    for model_key, model_path in [("model_a", args.model_a_path), ("model_b", args.model_b_path)]:
        print(f"loading {model_key}: {model_path}", flush=True)
        model, tokenizer, resolved = load_model_and_tokenizer(
            model_path,
            args.device,
            dtype,
            bool(args.local_files_only),
        )
        alti_model = alti_model_view(model)
        if model_key == "model_a":
            payload["resolved_model_a_path"] = resolved
        else:
            payload["resolved_model_b_path"] = resolved

        for idx, sample in enumerate(samples):
            if model_key == "model_a":
                payload["samples"].append({
                    "sample_id": sample.get("sample_id"),
                    "filtered_index": sample.get("filtered_index"),
                    "uid": sample.get("uid"),
                    "source_dataset": sample.get("source_dataset"),
                    "language": sample.get("language"),
                    "raw_id": sample.get("raw_id"),
                    "entry_point": sample.get("entry_point"),
                    "categories": sample.get("categories", []),
                    "prefix": sample.get("prefix", ""),
                    "suffix": sample.get("suffix", ""),
                    "ground_truth": sample.get("ground_truth", ""),
                    "models": {},
                })
            print(
                f"[sample] {model_key} {idx + 1}/{len(samples)} #{sample.get('filtered_index')} {sample.get('uid')}",
                flush=True,
            )
            result = load_existing_or_compute(
                model=model,
                alti_model=alti_model,
                tokenizer=tokenizer,
                sample=sample,
                model_key=model_key,
                args=args,
            )
            meta = dict(sample.get(model_key) or {})
            meta.update(result)
            payload["samples"][idx]["models"][model_key] = meta
            if args.partial_write_every > 0 and (idx + 1) % args.partial_write_every == 0:
                write_json_atomic(Path(args.output_path), payload, indent=2)
                print(f"[partial write] {args.output_path}", flush=True)

        del model, alti_model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json_atomic(Path(args.output_path), payload, indent=2)
    print(f"wrote {args.output_path} ({len(payload['samples'])} samples)")


if __name__ == "__main__":
    main()
