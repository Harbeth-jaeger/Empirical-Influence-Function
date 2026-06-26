#!/usr/bin/env python
"""Run benchmark generation/judging and dump every prediction for failure analysis."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.benchmark_eval import load_model_for_eval, read_jsonl, resolve_model_path
from scripts.benchmark.eval_generation import (
    _resolve_generation_context_limit,
    build_messages_prompt,
    generate_batch_greedy,
    generate_batch_samples,
)
from scripts.benchmark.eval_judges import decode_escaped_for_prompt, judge_candidate, parse_fim_prompt
from scripts.benchmark.eval_reporting import RowMetric, aggregate_pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--baseline_name", required=True)
    p.add_argument("--eval_path", default="/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/eval_data/rendered_chatml_fim_eval.jsonl")
    p.add_argument("--output_path", required=True, help="JSONL dump path.")
    p.add_argument("--summary_path", default="", help="Optional summary JSON path.")
    p.add_argument("--source_datasets", default="humaneval")
    p.add_argument("--languages", default="")
    p.add_argument("--max_rows", type=int, default=1000)
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens_cap", type=int, default=512)
    p.add_argument("--infer_batch_size", type=int, default=4)
    p.add_argument("--sample_infer_batch_size", type=int, default=2)
    p.add_argument("--judge_workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    p.add_argument("--judge_timeout_sec", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _split_filter(value: str) -> set[str]:
    return {x.strip().lower() for x in value.split(",") if x.strip()}


def _chunk(items: list[Any], n: int) -> list[list[Any]]:
    n = max(1, int(n))
    return [items[i:i + n] for i in range(0, len(items), n)]


def _row_identity(row: dict[str, Any], filtered_index: int) -> dict[str, Any]:
    prefix, suffix = parse_fim_prompt(str(row.get("fim_prompt", "")))
    return {
        "filtered_index": filtered_index,
        "uid": row.get("uid", ""),
        "source_dataset": str(row.get("source_dataset", "unknown")),
        "language": str(row.get("language", "unknown")),
        "raw_id": row.get("raw_id", ""),
        "task_type": row.get("task_type", ""),
        "prefix": decode_escaped_for_prompt(prefix),
        "suffix": decode_escaped_for_prompt(suffix),
        "ground_truth": decode_escaped_for_prompt(str(row.get("fim_completion", ""))),
        "entry_point": row.get("metadata", {}).get("entry_point", ""),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path) if args.summary_path else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    source_datasets = _split_filter(args.source_datasets)
    languages = _split_filter(args.languages)
    rows = read_jsonl(
        args.eval_path,
        max_rows=args.max_rows,
        source_datasets=source_datasets or None,
        languages=languages or None,
    )
    if not rows:
        raise ValueError("No eval rows found.")

    resolved_model = resolve_model_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_for_eval(resolved_model)
    generation_context_limit = _resolve_generation_context_limit(model, tokenizer)

    prepared: list[dict[str, Any]] = []
    for filtered_index, row in enumerate(rows):
        prompt = build_messages_prompt(tokenizer, row.get("messages", []))
        ref = decode_escaped_for_prompt(str(row.get("fim_completion", "")))
        ref_tokens = tokenizer(ref, add_special_tokens=False).input_ids
        max_new_tokens = min(args.max_new_tokens_cap, max(32, len(ref_tokens) * 2 + 32))
        prepared.append({
            "filtered_index": filtered_index,
            "row": row,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "identity": _row_identity(row, filtered_index),
        })

    metrics: list[RowMetric] = []
    judge_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.judge_workers))
    written = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        pbar = tqdm(total=len(prepared), desc=f"eval-dump {args.baseline_name}")
        try:
            for chunk in _chunk(prepared, args.infer_batch_size):
                prompts = [it["prompt"] for it in chunk]
                max_new_tokens = max(it["max_new_tokens"] for it in chunk)
                max_input_tokens = max(1, generation_context_limit - max_new_tokens)
                greedy_preds = generate_batch_greedy(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    max_input_tokens=max_input_tokens,
                )

                sampled_by_row: list[list[str]] = [[] for _ in chunk]
                for start in range(0, len(chunk), max(1, args.sample_infer_batch_size)):
                    sub = chunk[start:start + max(1, args.sample_infer_batch_size)]
                    sub_prompts = [it["prompt"] for it in sub]
                    sub_max_new = max(it["max_new_tokens"] for it in sub)
                    sub_max_input = max(1, generation_context_limit - sub_max_new)
                    sampled = generate_batch_samples(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=sub_prompts,
                        max_new_tokens=sub_max_new,
                        max_input_tokens=sub_max_input,
                        num_samples=args.num_samples,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    for local_idx, preds in enumerate(sampled):
                        sampled_by_row[start + local_idx] = preds

                greedy_futures = [
                    judge_pool.submit(judge_candidate, item["row"], greedy_preds[i], args.judge_timeout_sec)
                    for i, item in enumerate(chunk)
                ]
                sampled_futures = [
                    [judge_pool.submit(judge_candidate, item["row"], pred, args.judge_timeout_sec) for pred in sampled_by_row[i]]
                    for i, item in enumerate(chunk)
                ]

                for i, item in enumerate(chunk):
                    row = item["row"]
                    pass1_raw, status1, detail1 = greedy_futures[i].result()
                    sample_results = [f.result() for f in sampled_futures[i]]
                    any_judged = pass1_raw is not None or any(p is not None for p, _, _ in sample_results)
                    sampled_passes = [None if p is None else bool(p) for p, _, _ in sample_results]
                    pass10_raw = any(p is True for p in sampled_passes)
                    pass1 = 1.0 if pass1_raw else 0.0
                    pass10 = 1.0 if pass10_raw else 0.0
                    metrics.append(RowMetric(
                        source_dataset=str(row.get("source_dataset", "unknown")),
                        language=str(row.get("language", "unknown")),
                        judged=bool(any_judged),
                        pass1=pass1 if any_judged else 0.0,
                        pass10=pass10 if any_judged else 0.0,
                    ))
                    record = {
                        "baseline": args.baseline_name,
                        "model_path": resolved_model,
                        **item["identity"],
                        "max_new_tokens": item["max_new_tokens"],
                        "greedy": {
                            "prediction": greedy_preds[i],
                            "pass": None if pass1_raw is None else bool(pass1_raw),
                            "status": status1,
                            "detail": detail1,
                        },
                        "samples": [
                            {
                                "index": j + 1,
                                "prediction": pred,
                                "pass": sampled_passes[j],
                                "status": sample_results[j][1],
                                "detail": sample_results[j][2],
                            }
                            for j, pred in enumerate(sampled_by_row[i])
                        ],
                        "pass1": pass1,
                        "pass10": pass10,
                        "judged": bool(any_judged),
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    written += 1
                pbar.update(len(chunk))
        finally:
            pbar.close()
            judge_pool.shutdown(wait=True)

    agg = aggregate_pass(metrics)
    summary = {
        "baseline": args.baseline_name,
        "model_path": resolved_model,
        "eval_path": args.eval_path,
        "output_path": str(output_path),
        "num_rows": len(rows),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "overall": agg["overall"],
        "detail": agg["detail"],
        "written": written,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"wrote dump -> {output_path}")
    print(f"wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
