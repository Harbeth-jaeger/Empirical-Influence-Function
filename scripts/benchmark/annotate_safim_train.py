#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

try:
    import tqdm
except ImportError:  # pragma: no cover - convenience for minimal envs
    class _TqdmFallback:
        @staticmethod
        def tqdm(iterable: Any, **_: Any) -> Any:
            return iterable

    tqdm = _TqdmFallback()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotate.target_evidence_annot import annotate_safim_row, require_llm_credentials  # noqa: E402


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {path}")
    return rows


def write_jsonl_atomic(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"Saved {len(rows)} rows to {path}")


def row_key(row: dict[str, Any]) -> str:
    uid = row.get("uid")
    if uid:
        return f"uid:{uid}"
    payload = json.dumps(
        {
            "messages": row.get("messages"),
            "prefix": row.get("prefix", ""),
            "target": row.get("target", ""),
            "suffix": row.get("suffix", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "sha:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("key")
            if key:
                cache[str(key)] = obj
    print(f"Loaded annotation cache: {len(cache)} entries from {p}")
    return cache


def save_cache(cache: dict[str, dict[str, Any]], path: str | Path | None) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for key in sorted(cache):
            f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
    tmp.replace(p)
    print(f"Saved annotation cache: {len(cache)} entries to {p}")


def load_seed_compact(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = row.get("uid") or row.get("raw_id")
            if uid and row.get("attention_edges"):
                out[f"uid:{uid}"] = row
    print(f"Loaded seed compact annotations: {len(out)} entries from {p}")
    return out


def normalize_language(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {"python": "python", "java": "java", "cpp": "cpp", "c++": "cpp", "csharp": "csharp", "c#": "csharp"}
    return aliases.get(text, text)


def select_rows(
    rows: list[dict[str, Any]],
    *,
    languages: list[str],
    task_types: list[str],
    samples_per_bucket: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    lang_set = {normalize_language(lang) for lang in languages}
    task_set = set(task_types)
    selected: list[dict[str, Any]] = []
    if samples_per_bucket > 0:
        for lang in languages:
            lang_norm = normalize_language(lang)
            for task_type in task_types:
                bucket = [
                    row for row in rows
                    if normalize_language(row.get("language")) == lang_norm and row.get("task_type") == task_type
                ]
                selected.extend(bucket[:samples_per_bucket])
    else:
        selected = [
            row for row in rows
            if normalize_language(row.get("language")) in lang_set and row.get("task_type") in task_set
        ]
    if max_rows > 0:
        selected = selected[:max_rows]
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate SAFIM file-level ChatML-FIM train rows with completion-only attention edges."
    )
    p.add_argument("--input-path", type=Path, default=Path("data/benchmark/train_data/safim/safim_train_chatml.jsonl"))
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/benchmark/train_data/safim/safim_train_compact_annotated.jsonl"),
    )
    p.add_argument(
        "--annotation-cache-path",
        type=Path,
        default=Path("data/benchmark/train_data/safim/safim_train_annotation_cache.jsonl"),
    )
    p.add_argument("--seed-compact-path", type=Path, default=None)
    p.add_argument("--model-name-or-path", default="models/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--model-max-length", type=int, default=16384)
    p.add_argument("--languages", nargs="+", default=["python", "java", "cpp", "csharp"])
    p.add_argument(
        "--task-types",
        nargs="+",
        default=["algorithmic_block", "control_flow_expression", "api_function_call"],
    )
    p.add_argument("--samples-per-bucket", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--selected-indices", nargs="+", type=int, default=None)
    p.add_argument("--update-existing-output", action="store_true")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--annotation-mode", choices=["target_evidence", "rules"], default="target_evidence")
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--max-edges", type=int, default=128)
    p.add_argument("--max-structural-edges", type=int, default=48)
    p.add_argument("--max-rule-edges", type=int, default=96)
    p.add_argument("--max-teacher-edges", type=int, default=48)
    p.add_argument("--completion-internal-edges", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--flush-every", type=int, default=20)
    p.add_argument("--overwrite-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.annotation_mode in {"hybrid", "agent"}:
        require_llm_credentials()

    rows = load_jsonl(args.input_path)
    rows = select_rows(
        rows,
        languages=args.languages,
        task_types=args.task_types,
        samples_per_bucket=args.samples_per_bucket,
        max_rows=args.max_rows,
    )
    if args.selected_indices is not None:
        selected_rows = []
        for idx in args.selected_indices:
            if idx < 0 or idx >= len(rows):
                raise SystemExit(f"selected index {idx} out of range for {len(rows)} selected rows")
            selected_rows.append(rows[idx])
        rows = selected_rows
    if not rows:
        raise SystemExit("no rows selected for annotation")

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (normalize_language(row.get("language")), str(row.get("task_type", "")))
        counts[key] = counts.get(key, 0) + 1
    print(f"Selected {len(rows)} rows from {args.input_path}: {counts}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    cache = load_cache(args.annotation_cache_path)
    seed = load_seed_compact(args.seed_compact_path)
    cache_lock = threading.Lock()

    output_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        if not args.overwrite_cache and key in seed:
            output_by_key[key] = copy.deepcopy(seed[key])
        elif not args.overwrite_cache and key in cache and cache[key].get("record"):
            output_by_key[key] = copy.deepcopy(cache[key]["record"])

    todo = [row for row in rows if row_key(row) not in output_by_key]
    print(
        "Annotation plan: "
        f"todo={len(todo)}, cached={len(output_by_key)}, workers={args.num_workers}, "
        f"mode={args.annotation_mode}, completion_internal={args.completion_internal_edges}"
    )

    def work(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        key = row_key(row)
        try:
            record = annotate_safim_row(
                row,
                tokenizer,
                model_max_length=args.model_max_length,
                annotation_mode=args.annotation_mode,
                max_edges=args.max_edges,
                max_structural_edges=args.max_structural_edges,
                max_rule_edges=args.max_rule_edges,
                max_teacher_edges=args.max_teacher_edges,
                max_rounds=args.max_rounds,
                include_completion_internal=args.completion_internal_edges,
            )
            if record is None:
                return key, None, "annotate_safim_row returned None"
            return key, record, None
        except Exception as exc:
            return key, None, repr(exc)

    done_since_flush = 0
    failures: dict[str, str] = {}
    if args.num_workers <= 1:
        iterator: Any = map(work, todo)
        pool = None
    else:
        pool = ThreadPoolExecutor(max_workers=args.num_workers)
        futures = [pool.submit(work, row) for row in todo]
        iterator = (future.result() for future in as_completed(futures))

    try:
        for key, record, err in tqdm.tqdm(iterator, total=len(todo), desc="annotate safim"):
            if record is None:
                failures[key] = err or "unknown"
            else:
                output_by_key[key] = record
                with cache_lock:
                    cache[key] = {"key": key, "record": record}
                done_since_flush += 1
            if done_since_flush >= max(1, args.flush_every):
                save_cache(cache, args.annotation_cache_path)
                done_since_flush = 0
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    save_cache(cache, args.annotation_cache_path)
    output = [output_by_key[row_key(row)] for row in rows if row_key(row) in output_by_key]
    if args.update_existing_output and args.output_path.exists():
        existing_rows = load_jsonl(args.output_path)
        replace_keys = set(output_by_key)
        replaced = 0
        merged = []
        for old_row in existing_rows:
            key = row_key(old_row)
            if key in replace_keys:
                merged.append(output_by_key[key])
                replaced += 1
            else:
                merged.append(old_row)
        existing_keys = {row_key(old_row) for old_row in existing_rows}
        for key in replace_keys - existing_keys:
            merged.append(output_by_key[key])
        output = merged
        print(f"Updated existing output rows: replaced={replaced}, total={len(output)}")
    write_jsonl_atomic(output, args.output_path)

    fail_path = args.output_path.with_suffix(args.output_path.suffix + ".failures.json")
    if failures:
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Failures: {len(failures)} -> {fail_path}")
    elif fail_path.exists():
        fail_path.unlink()
    print(f"Done: {len(output)}/{len(rows)} annotated -> {args.output_path}")


if __name__ == "__main__":
    main()
