from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded benchmark_eval.py JSON outputs.")
    parser.add_argument("--input_glob", required=True, help="Glob for shard *_benchmark_eval.json files.")
    parser.add_argument(
        "--override_glob",
        default="",
        help="Optional glob for corrected shard JSON files. Groups present here replace groups from input_glob.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to write merged outputs.")
    parser.add_argument("--baseline_name", default="Ours Graphsignal", help="Baseline display name.")
    return parser.parse_args()


def write_csv(path: str, rows: list[dict[str, Any]], headers: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.6f}"


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


def _add_codebleu_row(bucket: dict[str, float], row: dict[str, Any]) -> None:
    n_supported = int(row.get("n_supported", 0))
    bucket["n_total"] += int(row.get("n_total", 0))
    bucket["n_supported"] += n_supported
    bucket["n_failed"] += int(row.get("n_failed", 0))
    for key in (
        "codebleu",
        "ngram_match_score",
        "weighted_ngram_match_score",
        "syntax_match_score",
        "dataflow_match_score",
    ):
        bucket[f"{key}_sum"] += float(row.get(key, 0.0)) * n_supported


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


def score_for(detail: dict[tuple[str, str], dict[str, Any]], dataset: str, language: str) -> tuple[str, str]:
    row = detail.get((dataset, language))
    if not row or row["n_judged"] <= 0:
        return "", ""
    return f"{row['pass@1']:.4f}", f"{row['pass@10']:.4f}"


def write_tables(path: str, baseline_name: str, detail: dict[tuple[str, str], dict[str, Any]]) -> None:
    def is_current(display_name: str) -> bool:
        return display_name.lower().replace("-", "").replace("_", "").replace(" ", "") == baseline_name.lower().replace("-", "").replace("_", "").replace(" ", "")

    lines: list[str] = []
    lines.extend([
        "# Benchmark Results",
        "",
        "## HumanEval",
        "",
        "| **Competitors** | **Pass@1** | **Pass@10** |",
        "| --- | --- | --- |",
    ])
    p1, p10 = score_for(detail, "humaneval", "python")
    for name in ["Ours Graphsignal", "TokenCleaning", "XTF", "LLM-CleanCode", "CLEAR"]:
        vals = (p1, p10) if is_current(name) else ("", "")
        lines.append(f"| **{name}** | {vals[0]} | {vals[1]} |")

    lines.extend([
        "",
        "## SAFIM",
        "",
        "| **Baseline** | **Language** | **Pass@1** | **Pass@10** |",
        "| --- | --- | --- | --- |",
    ])
    for name in ["Ours Graphsignal", "XTF", "CLEAR", "LLM-CleanCode", "TokenCleaning"]:
        for idx, lang in enumerate(["Python", "Java", "C++", "C#"]):
            key = {"Python": "python", "Java": "java", "C++": "cpp", "C#": "csharp"}[lang]
            vals = score_for(detail, "safim", key) if is_current(name) else ("", "")
            label = f"**{name}**" if idx == 0 else ""
            lines.append(f"| {label} | {lang} | {vals[0]} | {vals[1]} |")

    lines.extend([
        "",
        "## McEval",
        "",
        "| **Baseline** | **Language** | **Pass@1** | **Pass@10** |",
        "| --- | --- | --- | --- |",
    ])
    for name in ["Ours Graphsignal", "TokenCleaning", "XTF", "CLEAR", "LLM-CleanCode"]:
        for idx, lang in enumerate(["C", "C++", "C#", "Go", "Java", "Python"]):
            key = {"C": "c", "C++": "cpp", "C#": "csharp", "Go": "go", "Java": "java", "Python": "python"}[lang]
            vals = score_for(detail, "mceval", key) if is_current(name) else ("", "")
            label = f"**{name}**" if idx == 0 else ""
            lines.append(f"| {label} | {lang} | {vals[0]} | {vals[1]} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")
    override_paths = sorted(glob.glob(args.override_glob)) if args.override_glob else []

    buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"n_total": 0, "n_judged": 0, "n_unsupported": 0, "pass1_sum": 0.0, "pass10_sum": 0.0})
    codebleu_buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(_empty_codebleu_bucket)
    meta: dict[str, Any] = {"merged_files": paths, "override_files": override_paths}

    def add_paths(paths_to_add: list[str]) -> set[tuple[str, str]]:
        seen_groups: set[tuple[str, str]] = set()
        for path in paths_to_add:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            meta.setdefault("model_path", obj.get("model_path", ""))
            meta.setdefault("eval_path", obj.get("eval_path", ""))
            for row in obj.get("by_dataset_language", []):
                key = (str(row["source_dataset"]).lower(), str(row["language"]).lower())
                seen_groups.add(key)
                n_judged = int(row.get("n_judged", 0))
                bucket = buckets[key]
                bucket["n_total"] += int(row.get("n_total", 0))
                bucket["n_judged"] += n_judged
                bucket["n_unsupported"] += int(row.get("n_unsupported", 0))
                bucket["pass1_sum"] += float(row.get("pass@1", 0.0)) * n_judged
                bucket["pass10_sum"] += float(row.get("pass@10", 0.0)) * n_judged
            for row in obj.get("codebleu", {}).get("by_dataset_language", []):
                key = (str(row["source_dataset"]).lower(), str(row["language"]).lower())
                _add_codebleu_row(codebleu_buckets[key], row)
        return seen_groups

    add_paths(paths)

    if override_paths:
        override_groups: set[tuple[str, str]] = set()
        for path in override_paths:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            for row in obj.get("by_dataset_language", []):
                override_groups.add((str(row["source_dataset"]).lower(), str(row["language"]).lower()))
        for key in override_groups:
            buckets.pop(key, None)
            codebleu_buckets.pop(key, None)
        add_paths(override_paths)

    detail: dict[tuple[str, str], dict[str, Any]] = {}
    for key, bucket in sorted(buckets.items()):
        n_judged = int(bucket["n_judged"])
        detail[key] = {
            "source_dataset": key[0],
            "language": key[1],
            "n_total": int(bucket["n_total"]),
            "n_judged": n_judged,
            "n_unsupported": int(bucket["n_unsupported"]),
            "pass@1": bucket["pass1_sum"] / n_judged if n_judged else 0.0,
            "pass@10": bucket["pass10_sum"] / n_judged if n_judged else 0.0,
        }

    total_judged = sum(row["n_judged"] for row in detail.values())
    total_all = sum(row["n_total"] for row in detail.values())
    total_unsupported = sum(row["n_unsupported"] for row in detail.values())
    overall = {
        "n_total": total_all,
        "n_judged": total_judged,
        "n_unsupported": total_unsupported,
        "pass@1": sum(row["pass@1"] * row["n_judged"] for row in detail.values()) / total_judged if total_judged else 0.0,
        "pass@10": sum(row["pass@10"] * row["n_judged"] for row in detail.values()) / total_judged if total_judged else 0.0,
    }
    codebleu_detail = {
        key: {
            "source_dataset": key[0],
            "language": key[1],
            **_finalize_codebleu_bucket(bucket),
        }
        for key, bucket in sorted(codebleu_buckets.items())
    }
    codebleu_overall_bucket = _empty_codebleu_bucket()
    for bucket in codebleu_buckets.values():
        finalized = _finalize_codebleu_bucket(bucket)
        _add_codebleu_row(codebleu_overall_bucket, finalized)
    codebleu_overall = _finalize_codebleu_bucket(codebleu_overall_bucket)

    os.makedirs(args.output_dir, exist_ok=True)
    merged = {
        "baseline": args.baseline_name,
        **meta,
        "overall": overall,
        "by_dataset_language": list(detail.values()),
    }
    if codebleu_buckets:
        merged["codebleu"] = {
            "scope": "greedy_prediction_vs_fim_completion",
            "overall": codebleu_overall,
            "by_dataset_language": list(codebleu_detail.values()),
        }
    json_path = os.path.join(args.output_dir, f"{args.baseline_name}_benchmark_eval_merged.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    overall_csv = os.path.join(args.output_dir, f"{args.baseline_name}_benchmark_overall_merged.csv")
    write_csv(overall_csv, [{
        "Baseline": args.baseline_name,
        "Pass@1": fmt(overall["pass@1"]),
        "Pass@10": fmt(overall["pass@10"]),
        "CodeBLEU": fmt(codebleu_overall.get("codebleu", 0.0)) if codebleu_buckets else "",
        "N": overall["n_judged"],
        "N_total": overall["n_total"],
        "N_unsupported": overall["n_unsupported"],
    }], ["Baseline", "Pass@1", "Pass@10", "CodeBLEU", "N", "N_total", "N_unsupported"])

    detail_rows = [{
        "Baseline": args.baseline_name,
        "Dataset": row["source_dataset"],
        "Language": row["language"],
        "Pass@1": fmt(row["pass@1"]),
        "Pass@10": fmt(row["pass@10"]),
        "CodeBLEU": fmt(codebleu_detail.get((row["source_dataset"], row["language"]), {}).get("codebleu", 0.0)) if codebleu_buckets else "",
        "N": row["n_judged"],
        "N_total": row["n_total"],
        "N_unsupported": row["n_unsupported"],
    } for row in detail.values()]
    detail_csv = os.path.join(args.output_dir, f"{args.baseline_name}_benchmark_by_dataset_language_merged.csv")
    write_csv(detail_csv, detail_rows, ["Baseline", "Dataset", "Language", "Pass@1", "Pass@10", "CodeBLEU", "N", "N_total", "N_unsupported"])

    tables_path = os.path.join(args.output_dir, f"{args.baseline_name}_benchmark_tables_merged.md")
    write_tables(tables_path, args.baseline_name, detail)

    print(f"merged {len(paths)} files")
    print(f"saved json: {json_path}")
    print(f"saved tables: {tables_path}")


if __name__ == "__main__":
    main()
