from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable



from .adapters import adapt_row


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    model_path: str
    annotate_model: str
    language: str = "auto"
    source_dataset: str = "auto"
    run_name: str = "fim_annotation"
    max_rows: int = 0
    max_accepted_rows: int = 0
    num_workers: int = 4
    model_max_length: int = 4096
    max_teacher_edges: int = 64
    use_llm: bool = True
    skip_annotation: bool = False
    force: bool = False
    strip_cjk_comments: bool = True
    max_target_nonempty_lines: int = 10
    max_target_rough_tokens: int = 192
    max_target_chars: int = 1024
    flush_every: int = 20


@dataclass
class PipelineResult:
    canonical_path: Path
    chatml_path: Path
    compact_path: Path
    report_path: Path
    accepted: int
    annotated: int
    rejected: int


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield idx, obj


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def prepare_rows(config: PipelineConfig) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    rejects: list[dict[str, Any]] = []
    seen_uid: set[str] = set()
    for index, row in iter_jsonl(config.input_path):
        if config.max_rows > 0 and stats["seen"] >= config.max_rows:
            break
        if config.max_accepted_rows > 0 and stats["accepted"] >= config.max_accepted_rows:
            break
        stats["seen"] += 1
        sample, reason = adapt_row(
            row,
            index,
            language=config.language,
            source_dataset=config.source_dataset,
            strip_cjk=config.strip_cjk_comments,
            max_target_nonempty_lines=config.max_target_nonempty_lines,
            max_target_rough_tokens=config.max_target_rough_tokens,
            max_target_chars=config.max_target_chars,
        )
        if sample is None:
            stats["rejected"] += 1
            stats[f"reject_{reason or 'unknown'}"] += 1
            rejects.append({"index": index, "reason": reason or "unknown", "raw_id": row.get("task_id") or row.get("uid")})
            continue
        uid = str(sample.get("uid"))
        if uid in seen_uid:
            stats["rejected"] += 1
            stats["reject_duplicate_uid"] += 1
            rejects.append({"index": index, "reason": "duplicate_uid", "uid": uid})
            continue
        seen_uid.add(uid)
        rows.append(sample)
        stats["accepted"] += 1
        stats[f"language_{sample.get('language', 'unknown')}"] += 1
    return rows, stats, rejects


def annotate_rows(rows: list[dict[str, Any]], config: PipelineConfig, cache_path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    from src.annotate.run import annotate_row, append_cache, load_cache, row_key, save_cache

    if not rows:
        return [], {}
    if config.annotate_model:
        os.environ["ANNOTATE_MODEL"] = config.annotate_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    cache = {} if config.force else load_cache(cache_path)
    output_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        if not config.force and key in cache and cache[key].get("record"):
            output_by_key[key] = copy.deepcopy(cache[key]["record"])

    todo = [row for row in rows if row_key(row) not in output_by_key]
    failures: dict[str, str] = {}
    buffer: list[dict[str, Any]] = []

    def work(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        key = row_key(row)
        try:
            record = annotate_row(
                row,
                tokenizer,
                config.model_max_length,
                config.max_teacher_edges,
                config.use_llm,
            )
            return key, record, None
        except Exception as exc:
            return key, None, repr(exc)

    if config.num_workers <= 1:
        iterator = map(work, todo)
    else:
        pool = ThreadPoolExecutor(max_workers=config.num_workers)
        futures = [pool.submit(work, row) for row in todo]
        iterator = (future.result() for future in as_completed(futures))

    try:
        for key, record, err in iterator:
            if record is None:
                failures[key] = err or "annotation returned None"
                continue
            output_by_key[key] = record
            cache[key] = {"key": key, "record": record}
            buffer.append({"key": key, "record": record})
            if len(buffer) >= max(1, config.flush_every):
                append_cache(buffer, cache_path)
                buffer = []
    finally:
        if config.num_workers > 1:
            pool.shutdown(wait=False, cancel_futures=True)
        if buffer:
            append_cache(buffer, cache_path)
    save_cache(cache, cache_path)
    return [output_by_key[key] for key in (row_key(row) for row in rows) if key in output_by_key], failures


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = config.output_dir / f"{config.run_name}_canonical.jsonl"
    chatml_path = config.output_dir / f"{config.run_name}_chatml.jsonl"
    compact_path = config.output_dir / f"{config.run_name}_compact.jsonl"
    report_path = config.output_dir / f"{config.run_name}_report.json"
    cache_path = config.output_dir / f"{config.run_name}_annotation_cache.jsonl"
    rejects_path = config.output_dir / f"{config.run_name}_rejects.jsonl"
    failures_path = config.output_dir / f"{config.run_name}_failures.json"

    rows, stats, rejects = prepare_rows(config)
    write_jsonl(canonical_path, rows)
    write_jsonl(chatml_path, rows)
    if rejects:
        write_jsonl(rejects_path, rejects)

    compact: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    if not config.skip_annotation:
        compact, failures = annotate_rows(rows, config, cache_path)
        write_jsonl(compact_path, compact)
        if failures:
            failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        write_jsonl(compact_path, [])

    report = {
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "run_name": config.run_name,
        "seen": int(stats.get("seen", 0)),
        "accepted": int(stats.get("accepted", 0)),
        "rejected": int(stats.get("rejected", 0)),
        "annotated": len(compact),
        "skip_annotation": config.skip_annotation,
        "use_llm": config.use_llm,
        "num_workers": config.num_workers,
        "reject_reasons": {k[len("reject_"):]: int(v) for k, v in stats.items() if k.startswith("reject_")},
        "language_counts": {k[len("language_"):]: int(v) for k, v in stats.items() if k.startswith("language_")},
        "paths": {
            "canonical": str(canonical_path),
            "chatml": str(chatml_path),
            "compact": str(compact_path),
            "cache": str(cache_path),
            "rejects": str(rejects_path) if rejects else "",
            "failures": str(failures_path) if failures else "",
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return PipelineResult(
        canonical_path=canonical_path,
        chatml_path=chatml_path,
        compact_path=compact_path,
        report_path=report_path,
        accepted=len(rows),
        annotated=len(compact),
        rejected=int(stats.get("rejected", 0)),
    )
