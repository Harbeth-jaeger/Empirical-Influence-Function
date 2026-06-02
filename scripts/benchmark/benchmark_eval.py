from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
try:
    from scripts.benchmark.eval_generation import (
        _resolve_generation_context_limit,
        build_messages_prompt,
        generate_batch_greedy,
        generate_batch_samples,
    )
    from scripts.benchmark.eval_judges import decode_escaped_for_prompt, judge_candidate
    from scripts.benchmark.eval_reporting import RowMetric, aggregate_pass, write_benchmark_tables_markdown
except ModuleNotFoundError:
    from eval_generation import (
        _resolve_generation_context_limit,
        build_messages_prompt,
        generate_batch_greedy,
        generate_batch_samples,
    )
    from eval_judges import decode_escaped_for_prompt, judge_candidate
    from eval_reporting import RowMetric, aggregate_pass, write_benchmark_tables_markdown


SUPPORTED_LANGUAGES = {"python", "cpp", "java", "csharp", "c", "go"}


def _import_codebleu():
    try:
        from codebleu import calc_codebleu  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CodeBLEU support requires the `codebleu` package. Install it in the eif-bench env, "
            "for example: `.micromamba/envs/eif-bench/bin/python -m pip install codebleu`."
        ) from exc
    return calc_codebleu


def _codebleu_lang(language: str) -> str | None:
    lang = language.lower()
    aliases = {
        "python": "python",
        "java": "java",
        "go": "go",
        "cpp": "cpp",
        "c++": "cpp",
        "c": "c",
        "csharp": "c_sharp",
        "c#": "c_sharp",
    }
    return aliases.get(lang)


def _empty_codebleu_bucket() -> dict[str, float]:
    return {
        "n_total": 0,
        "n_supported": 0,
        "n_failed": 0,
        "codebleu_sum": 0.0,
        "ngram_match_score_sum": 0.0,
        "weighted_ngram_match_score_sum": 0.0,
        "syntax_match_score_sum": 0.0,
        "dataflow_match_score_sum": 0.0,
    }


def _add_codebleu_score(bucket: dict[str, float], score: dict[str, Any]) -> None:
    bucket["n_supported"] += 1
    for key in (
        "codebleu",
        "ngram_match_score",
        "weighted_ngram_match_score",
        "syntax_match_score",
        "dataflow_match_score",
    ):
        bucket[f"{key}_sum"] += float(score.get(key, 0.0))


def _finalize_codebleu_bucket(bucket: dict[str, float]) -> dict[str, Any]:
    n_supported = int(bucket["n_supported"])
    out = {
        "n_total": int(bucket["n_total"]),
        "n_supported": n_supported,
        "n_failed": int(bucket["n_failed"]),
    }
    for key in (
        "codebleu",
        "ngram_match_score",
        "weighted_ngram_match_score",
        "syntax_match_score",
        "dataflow_match_score",
    ):
        out[key] = bucket[f"{key}_sum"] / n_supported if n_supported else 0.0
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Judge-based benchmark evaluator: use messages as prompt, generate assistant completion, "
            "stitch prefix+prediction+suffix, then execute judge_payload per language."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Finetuned model directory (output dir or checkpoint dir).",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="data/benchmarks/eval_data/rendered_chatml_fim_eval.jsonl",
        help="Unified benchmark eval JSONL path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/benchmark",
        help="Directory to write evaluation outputs.",
    )
    parser.add_argument(
        "--baseline_name",
        type=str,
        default="Ours-GraphSignal",
        help="Baseline/model display name in result tables.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of sampled generations per prompt for pass@10.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for pass@10 candidates.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for sampled generations.",
    )
    parser.add_argument(
        "--max_new_tokens_cap",
        type=int,
        default=512,   # FIX Bug2: 原来是256，改为512，避免截断
        help="Upper cap for generation length.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="If >0 only evaluate first N rows.",
    )
    parser.add_argument(
        "--per_group_limit",
        type=int,
        default=0,
        help="If >0, keep at most N rows for each (source_dataset, language) group.",
    )
    parser.add_argument(
        "--source_datasets",
        type=str,
        default="",
        help="Comma-separated source_dataset filter, e.g. humaneval,safim,mceval.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="",
        help="Comma-separated language filter, e.g. python,cpp,java.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split eval rows into this many deterministic shards.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Evaluate only rows where filtered_row_index %% num_shards == shard_index.",
    )
    parser.add_argument(
        "--judge_timeout_sec",
        type=int,
        default=10,
        help="Per-test process timeout in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=4,
        help="Batch size for greedy generation.",
    )
    parser.add_argument(
        "--sample_infer_batch_size",
        type=int,
        default=2,
        help="Batch size for sampled generation (pass@10).",
    )
    parser.add_argument(
        "--judge_workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 2),
        help="Number of parallel workers for judge execution.",
    )
    parser.add_argument(
        "--compute_codebleu",
        action="store_true",
        help="Also compute greedy-prediction CodeBLEU against fim_completion.",
    )
    return parser.parse_args()


def resolve_model_path(model_path: str) -> str:
    p = Path(model_path)
    if not p.exists():
        # Allow HuggingFace repo ids such as "Qwen/Qwen2.5-Coder-1.5B-Instruct".
        # from_pretrained(..., local_files_only=True) will still enforce offline cache use.
        if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", model_path):
            return model_path
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if (p / "adapter_config.json").exists() and (p / "adapter_model.safetensors").exists():
        return str(p)

    checkpoints: list[tuple[int, Path]] = []
    for child in p.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            step = child.name.split("-")[-1]
            if step.isdigit():
                checkpoints.append((int(step), child))

    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        return str(checkpoints[-1][1])
    return str(p)


def load_model_for_eval(model_path: str):
    if (Path(model_path) / "adapter_config.json").exists():
        peft_config = PeftConfig.from_pretrained(model_path, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        return PeftModel.from_pretrained(base_model, model_path, is_trainable=False).eval()

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()


def read_jsonl(
    path: str,
    max_rows: int = 0,
    per_group_limit: int = 0,
    source_datasets: set[str] | None = None,
    languages: set[str] | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("--num_shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard_index must be in [0, num_shards).")

    rows: list[dict[str, Any]] = []
    group_counts: dict[tuple[str, str], int] = {}
    filtered_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if source_datasets and str(row.get("source_dataset", "")).lower() not in source_datasets:
                continue
            if languages and str(row.get("language", "")).lower() not in languages:
                continue
            current_idx = filtered_idx
            filtered_idx += 1
            if current_idx % num_shards != shard_index:
                continue
            if per_group_limit > 0:
                key = (row.get("source_dataset", "unknown"), row.get("language", "unknown"))
                ct = group_counts.get(key, 0)
                if ct >= per_group_limit:
                    continue
                group_counts[key] = ct + 1
            rows.append(row)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def write_csv(path: str, rows: list[dict[str, Any]], headers: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    chunk_size = max(1, int(chunk_size))
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    resolved_model = resolve_model_path(args.model_path)
    source_datasets = {x.strip().lower() for x in args.source_datasets.split(",") if x.strip()}
    languages = {x.strip().lower() for x in args.languages.split(",") if x.strip()}
    rows = read_jsonl(
        args.eval_path,
        max_rows=args.max_rows,
        per_group_limit=args.per_group_limit,
        source_datasets=source_datasets or None,
        languages=languages or None,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if len(rows) == 0:
        raise ValueError("No eval rows found.")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_for_eval(resolved_model)

    generation_context_limit = _resolve_generation_context_limit(model, tokenizer)

    prepared_items: list[dict[str, Any]] = []
    for row in rows:
        prompt = build_messages_prompt(tokenizer, row.get("messages", []))
        ref = decode_escaped_for_prompt(str(row.get("fim_completion", "")))
        ref_tokens = tokenizer(ref, add_special_tokens=False).input_ids
        max_new_tokens = min(args.max_new_tokens_cap, max(32, len(ref_tokens) * 2 + 32))
        prepared_items.append(
            {
                "row": row,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "source_dataset": str(row.get("source_dataset", "unknown")),
                "language": str(row.get("language", "unknown")),
            }
        )

    metrics: list[RowMetric] = []
    sample_dump: list[dict[str, Any]] = []
    calc_codebleu = _import_codebleu() if args.compute_codebleu else None
    codebleu_overall = _empty_codebleu_bucket()
    codebleu_by_dataset_language: dict[tuple[str, str], dict[str, float]] = {}
    judge_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.judge_workers))
    pbar = tqdm(total=len(prepared_items), desc="Evaluating (batched)")
    try:
        for greedy_chunk in _chunk_list(prepared_items, args.infer_batch_size):
            prompts = [it["prompt"] for it in greedy_chunk]
            max_new_tokens_chunk = max(it["max_new_tokens"] for it in greedy_chunk)
            max_input_tokens_chunk = max(1, generation_context_limit - max_new_tokens_chunk)
            greedy_preds = generate_batch_greedy(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens_chunk,
                max_input_tokens=max_input_tokens_chunk,
            )

            sampled_by_row: list[list[str]] = [[] for _ in range(len(greedy_chunk))]
            for sub_chunk_start in range(0, len(greedy_chunk), max(1, args.sample_infer_batch_size)):
                sub_chunk = greedy_chunk[sub_chunk_start: sub_chunk_start + max(1, args.sample_infer_batch_size)]
                sub_prompts = [it["prompt"] for it in sub_chunk]
                sub_max_new_tokens = max(it["max_new_tokens"] for it in sub_chunk)
                sub_max_input_tokens = max(1, generation_context_limit - sub_max_new_tokens)
                sampled_sub = generate_batch_samples(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=sub_prompts,
                    max_new_tokens=sub_max_new_tokens,
                    max_input_tokens=sub_max_input_tokens,
                    num_samples=args.num_samples,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                for local_idx, preds in enumerate(sampled_sub):
                    sampled_by_row[sub_chunk_start + local_idx] = preds

            greedy_futures: list[concurrent.futures.Future[tuple[bool | None, str, str]]] = []
            sampled_futures_by_row: list[list[concurrent.futures.Future[tuple[bool | None, str, str]]]] = []

            for idx, item in enumerate(greedy_chunk):
                row = item["row"]
                greedy_futures.append(
                    judge_pool.submit(
                        judge_candidate,
                        row,
                        greedy_preds[idx],
                        args.judge_timeout_sec,
                    )
                )
                sampled_futures_by_row.append(
                    [
                        judge_pool.submit(judge_candidate, row, pred, args.judge_timeout_sec)
                        for pred in sampled_by_row[idx]
                    ]
                )

            for idx, item in enumerate(greedy_chunk):
                row = item["row"]
                source_dataset = item["source_dataset"]
                language = item["language"]

                pass1_raw, status1, detail1 = greedy_futures[idx].result()
                sampled_results = [f.result() for f in sampled_futures_by_row[idx]]

                pass10_raw = False
                pass10_status = "unsupported"
                pass10_detail = ""
                any_judged = pass1_raw is not None

                for p, st, dt in sampled_results:
                    if p is None:
                        pass10_status = st
                        pass10_detail = dt
                        continue
                    any_judged = True
                    if p:
                        pass10_raw = True
                        pass10_status = "ok"
                        pass10_detail = ""
                        break
                    pass10_status = st
                    pass10_detail = dt

                pass1 = 1.0 if pass1_raw else 0.0
                pass10 = 1.0 if pass10_raw else 0.0

                greedy_codebleu: dict[str, Any] | None = None
                if calc_codebleu is not None:
                    cb_key = (source_dataset.lower(), language.lower())
                    cb_bucket = codebleu_by_dataset_language.setdefault(cb_key, _empty_codebleu_bucket())
                    codebleu_overall["n_total"] += 1
                    cb_bucket["n_total"] += 1
                    cb_lang = _codebleu_lang(language)
                    if cb_lang is None:
                        codebleu_overall["n_failed"] += 1
                        cb_bucket["n_failed"] += 1
                    else:
                        ref = decode_escaped_for_prompt(str(row.get("fim_completion", "")))
                        try:
                            greedy_codebleu = calc_codebleu(
                                references=[ref],
                                predictions=[greedy_preds[idx]],
                                lang=cb_lang,
                            )
                            _add_codebleu_score(codebleu_overall, greedy_codebleu)
                            _add_codebleu_score(cb_bucket, greedy_codebleu)
                        except Exception as exc:  # CodeBLEU parser can fail on unsupported grammar/input.
                            greedy_codebleu = {"error": f"{type(exc).__name__}: {exc}"}
                            codebleu_overall["n_failed"] += 1
                            cb_bucket["n_failed"] += 1

                metrics.append(
                    RowMetric(
                        source_dataset=source_dataset,
                        language=language,
                        judged=any_judged,
                        pass1=pass1 if any_judged else 0.0,
                        pass10=pass10 if any_judged else 0.0,
                    )
                )

                if len(sample_dump) < 30:
                    sample_dump.append(
                        {
                            "uid": row.get("uid", ""),
                            "source_dataset": source_dataset,
                            "language": language,
                            "greedy_prediction": greedy_preds[idx],
                            "greedy_judge_status": status1,
                            "greedy_judge_detail": detail1,
                            "sampled_predictions": sampled_by_row[idx][:3],
                            "pass1": pass1,
                            "pass10": pass10,
                            "judged": any_judged,
                            "pass10_judge_status": pass10_status,
                            "pass10_judge_detail": pass10_detail,
                            "greedy_codebleu": greedy_codebleu,
                        }
                    )

            pbar.update(len(greedy_chunk))
    finally:
        pbar.close()
        judge_pool.shutdown(wait=True)

    agg = aggregate_pass(metrics)

    json_out = {
        "baseline": args.baseline_name,
        "model_path": resolved_model,
        "eval_path": args.eval_path,
        "num_rows": len(rows),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_samples_for_pass10": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "judge_timeout_sec": args.judge_timeout_sec,
        "overall": agg["overall"],
        "by_dataset_language": agg["detail"],
        "sample_predictions": sample_dump,
    }
    if args.compute_codebleu:
        json_out["codebleu"] = {
            "scope": "greedy_prediction_vs_fim_completion",
            "overall": _finalize_codebleu_bucket(codebleu_overall),
            "by_dataset_language": [
                {
                    "source_dataset": ds,
                    "language": lang,
                    **_finalize_codebleu_bucket(bucket),
                }
                for (ds, lang), bucket in sorted(codebleu_by_dataset_language.items())
            ],
        }

    json_path = os.path.join(args.output_dir, f"{args.baseline_name}_benchmark_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    overall_csv_path = os.path.join(args.output_dir, f"{args.baseline_name}_benchmark_overall.csv")
    write_csv(
        overall_csv_path,
        [
            {
                "Baseline": args.baseline_name,
                "Pass@1": f"{agg['overall']['pass@1']:.6f}",
                "Pass@10": f"{agg['overall']['pass@10']:.6f}",
                "CodeBLEU": f"{json_out.get('codebleu', {}).get('overall', {}).get('codebleu', 0.0):.6f}" if args.compute_codebleu else "",
                "N": agg["overall"]["n_judged"],
                "N_total": agg["overall"]["n_total"],
                "N_unsupported": agg["overall"]["n_unsupported"],
            }
        ],
        headers=["Baseline", "Pass@1", "Pass@10", "CodeBLEU", "N", "N_total", "N_unsupported"],
    )

    detail_rows: list[dict[str, Any]] = []
    codebleu_lookup = {
        (row["source_dataset"], row["language"]): row
        for row in json_out.get("codebleu", {}).get("by_dataset_language", [])
    }
    for row in agg["detail"]:
        cb = codebleu_lookup.get((row["source_dataset"], row["language"]), {})
        detail_rows.append(
            {
                "Baseline": args.baseline_name,
                "Dataset": row["source_dataset"],
                "Language": row["language"],
                "Pass@1": f"{row['pass@1']:.6f}",
                "Pass@10": f"{row['pass@10']:.6f}",
                "CodeBLEU": f"{float(cb.get('codebleu', 0.0)):.6f}" if args.compute_codebleu else "",
                "N": row["n_judged"],
                "N_total": row["n_total"],
                "N_unsupported": row["n_unsupported"],
            }
        )

    detail_csv_path = os.path.join(args.output_dir, f"{args.baseline_name}_benchmark_by_dataset_language.csv")
    write_csv(
        detail_csv_path,
        detail_rows,
        headers=["Baseline", "Dataset", "Language", "Pass@1", "Pass@10", "CodeBLEU", "N", "N_total", "N_unsupported"],
    )

    for dataset_name in ("mceval", "safim", "humaneval"):
        ds_rows = [r for r in detail_rows if r["Dataset"] == dataset_name]
        ds_rows.sort(key=lambda x: x["Language"])
        ds_path = os.path.join(args.output_dir, f"{args.baseline_name}_{dataset_name}_by_language.csv")
        write_csv(
            ds_path,
            ds_rows,
            headers=["Baseline", "Language", "Pass@1", "Pass@10", "CodeBLEU", "N", "N_total", "N_unsupported"],
        )

    tables_md_path = os.path.join(args.output_dir, f"{args.baseline_name}_benchmark_tables.md")
    write_benchmark_tables_markdown(tables_md_path, args.baseline_name, detail_rows)

    print(f"\n{'='*60}")
    print(f"  Overall Pass@1 : {agg['overall']['pass@1']:.4f}")
    print(f"  Overall Pass@10: {agg['overall']['pass@10']:.4f}")
    print(f"  Judged / Total : {agg['overall']['n_judged']} / {agg['overall']['n_total']}")
    print(f"{'='*60}")
    print(f"Saved overall: {overall_csv_path}")
    print(f"Saved detail:  {detail_csv_path}")
    print(f"Saved json:    {json_path}")
    print(f"Saved tables:  {tables_md_path}")


if __name__ == "__main__":
    main()
