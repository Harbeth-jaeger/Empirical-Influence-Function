#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Iterable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.go_single.go_single_pipeline import (
    BuildStats,
    should_reject_codesearchnet_row,
    find_function_slice,
    extract_statement_candidates,
    filter_by_function_lines,
    render_chatml,
    write_jsonl,
    write_report,
    write_samples_report,
    make_canonical_sample,
    stable_hash,
    normalize_code,
)


def iter_gz_jsonl(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def choose_single_candidate(candidates):
    """Pick one deterministic target per raw function for evaluation.

    Full eval should measure one completion task per CodeSearchNet function, not
    inflate a single function into many masked-line tasks.  Prefer a target near
    the middle of the candidate list so both prefix and suffix contain context.
    """
    kind_priority = {"assignment": 0, "return": 1, "call": 2}
    center = (len(candidates) - 1) / 2.0
    return min(
        enumerate(candidates),
        key=lambda item: (abs(item[0] - center), kind_priority.get(item[1].target_kind, 99), item[1].line_no),
    )[1]


def build_split(split: str, raw_root: Path, output_dir: Path, report_dir: Path) -> None:
    split_root = raw_root / split
    paths = sorted(split_root.glob("*.jsonl.gz"))
    if not paths:
        raise SystemExit(f"No files matched: {split_root}/*.jsonl.gz")

    stats = BuildStats()
    samples: list[dict] = []
    seen_function_norm: set[str] = set()

    for path in paths:
        stats.files_seen += 1
        print(f"[{split}] reading {path}")
        for row in iter_gz_jsonl(path):
            stats.rows_seen += 1
            reason = should_reject_codesearchnet_row(row)
            if reason:
                stats.reject(reason)
                continue

            code = str(row.get("code") or row.get("original_string") or "")
            func_slice = find_function_slice(code, row.get("func_name")) or (0, len(code), code)
            _, _, function_code = func_slice
            function_norm_key = stable_hash(normalize_code(function_code))
            if function_norm_key in seen_function_norm:
                stats.reject("duplicate_function_normalized")
                continue
            seen_function_norm.add(function_norm_key)

            candidates, reason = extract_statement_candidates(function_code)
            if reason:
                stats.reject(reason)
                continue
            if not candidates:
                stats.reject("no_statement_candidate")
                continue

            stats.candidates_seen += len(candidates)
            cand = choose_single_candidate(candidates)
            if len(candidates) > 1:
                stats.reject("extra_candidates_skipped_one_per_function")

            uid_seed = "".join([
                str(row.get("repo", "")),
                str(row.get("path", "")),
                str(row.get("func_name", "")),
                str(row.get("sha", "")),
                function_norm_key,
                str(cand.line_no),
                cand.target,
            ])
            sample = make_canonical_sample(
                uid=f"codesearchnet_go_{split}_{stable_hash(uid_seed)}",
                source_dataset="codesearchnet_go",
                split=split,
                function_code=function_code,
                cand=cand,
                metadata={
                    "repo": row.get("repo"),
                    "source_path": row.get("path"),
                    "function_name": row.get("func_name"),
                    "sha": row.get("sha"),
                    "url": row.get("url"),
                    "target_line": cand.line_no,
                    "target_span": [cand.start, cand.end],
                    "candidate_count_before_one_per_function": len(candidates),
                },
            )
            samples.append(sample)
            stats.accept(cand.target_kind)

    samples, line_filter = filter_by_function_lines(samples, max_lines=0, quantile=1.0)
    chatml = [render_chatml(sample) for sample in samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    canonical_out = output_dir / f"codesearchnet_go_{split}_full_canonical.jsonl"
    chatml_out = output_dir / f"codesearchnet_go_{split}_full_chatml.jsonl"
    report_out = report_dir / f"codesearchnet_go_{split}_full_build_report.md"
    preview_out = report_dir / f"codesearchnet_go_{split}_full_samples.md"

    write_jsonl(canonical_out, samples)
    write_jsonl(chatml_out, chatml)
    write_report(
        report_out,
        title=f"CodeSearchNet-Go Full {split} Build Report",
        train_stats=stats,
        eval_stats=BuildStats(),
        train_samples=samples,
        eval_samples=[],
        train_line_filter=line_filter,
    )
    write_samples_report(preview_out, samples, limit=80, seed=42)

    print(f"[done] {split}: rows={len(samples)}")
    print(f"[done] {split}: canonical={canonical_out}")
    print(f"[done] {split}: chatml={chatml_out}")
    print(f"[done] {split}: report={report_out}")
    print(f"[done] {split}: preview={preview_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full CodeSearchNet-Go valid/test single-statement eval data.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/go_single/raw_data/codesearchnet/go/final/jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/go_single/eval_data"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/go_single/reports"))
    parser.add_argument("--splits", default="valid,test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
        build_split(split, args.raw_root, args.output_dir, args.report_dir)


if __name__ == "__main__":
    main()
