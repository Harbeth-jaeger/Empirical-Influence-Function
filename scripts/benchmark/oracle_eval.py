from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.benchmark_eval import (
    RowMetric,
    aggregate_pass,
    judge_candidate,
    read_jsonl,
    write_benchmark_tables_markdown,
    write_csv,
)
from scripts.benchmark.eval_judges import decode_escaped_for_prompt


def write_oracle_tables_markdown(path: str, baseline_name: str, detail_rows: list[dict[str, Any]]) -> None:
    lines = ["# Gold Oracle Results", ""]
    lines.extend(["| **Dataset** | **Language** | **Pass@1** | **Pass@10** | **N** | **N_total** |", "| --- | --- | --- | --- | --- | --- |"])
    for row in sorted(detail_rows, key=lambda r: (str(r["Dataset"]), str(r["Language"]))):
        lines.append(
            "| {dataset} | {language} | {p1:.4f} | {p10:.4f} | {n} | {n_total} |".format(
                dataset=row["Dataset"],
                language=row["Language"],
                p1=float(row["Pass@1"]),
                p10=float(row["Pass@10"]),
                n=row["N"],
                n_total=row["N_total"],
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle benchmark evaluator: use each row's gold fim_completion as the predicted "
            "completion, then run the normal judge. This checks data stitching and judge wrappers."
        )
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
        default="outputs/benchmark/oracle_eval",
        help="Directory to write oracle evaluation outputs.",
    )
    parser.add_argument(
        "--baseline_name",
        type=str,
        default="Gold Oracle",
        help="Display name in result files.",
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
        "--judge_timeout_sec",
        type=int,
        default=10,
        help="Per-test process timeout in seconds.",
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
        help="Evaluate only rows where filtered_row_index % num_shards == shard_index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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
    if not rows:
        raise ValueError("No eval rows found.")

    metrics: list[RowMetric] = []
    sample_dump: list[dict[str, Any]] = []
    status_counts: dict[tuple[str, str, str], int] = {}

    for row in tqdm(rows, desc="Oracle judging"):
        source_dataset = str(row.get("source_dataset", "unknown"))
        language = str(row.get("language", "unknown"))
        gold_completion = decode_escaped_for_prompt(str(row.get("fim_completion", "")))
        passed, status, detail = judge_candidate(row, gold_completion, args.judge_timeout_sec)
        judged = passed is not None
        pass_value = 1.0 if passed else 0.0
        metrics.append(
            RowMetric(
                source_dataset=source_dataset,
                language=language,
                judged=judged,
                pass1=pass_value if judged else 0.0,
                pass10=pass_value if judged else 0.0,
            )
        )
        status_counts[(source_dataset, language, status)] = status_counts.get((source_dataset, language, status), 0) + 1
        if len(sample_dump) < 80 or not passed:
            sample_dump.append(
                {
                    "uid": row.get("uid", ""),
                    "source_dataset": source_dataset,
                    "language": language,
                    "judge_status": status,
                    "judge_detail": detail,
                    "passed": passed,
                    "fim_completion_head": gold_completion[:400],
                }
            )

    agg = aggregate_pass(metrics)
    status_rows = [
        {
            "Dataset": ds,
            "Language": lang,
            "Status": status,
            "Count": count,
        }
        for (ds, lang, status), count in sorted(status_counts.items())
    ]

    json_out = {
        "baseline": args.baseline_name,
        "eval_path": args.eval_path,
        "num_rows": len(rows),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "overall": agg["overall"],
        "by_dataset_language": agg["detail"],
        "status_counts": status_rows,
        "sample_judgments": sample_dump,
    }

    json_path = os.path.join(args.output_dir, f"{args.baseline_name}_oracle_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    overall_csv_path = os.path.join(args.output_dir, f"{args.baseline_name}_oracle_overall.csv")
    write_csv(
        overall_csv_path,
        [
            {
                "Baseline": args.baseline_name,
                "Pass@1": f"{agg['overall']['pass@1']:.6f}",
                "Pass@10": f"{agg['overall']['pass@10']:.6f}",
                "N": agg["overall"]["n_judged"],
                "N_total": agg["overall"]["n_total"],
                "N_unsupported": agg["overall"]["n_unsupported"],
            }
        ],
        headers=["Baseline", "Pass@1", "Pass@10", "N", "N_total", "N_unsupported"],
    )

    detail_rows: list[dict[str, Any]] = []
    for row in agg["detail"]:
        detail_rows.append(
            {
                "Baseline": args.baseline_name,
                "Dataset": row["source_dataset"],
                "Language": row["language"],
                "Pass@1": f"{row['pass@1']:.6f}",
                "Pass@10": f"{row['pass@10']:.6f}",
                "N": row["n_judged"],
                "N_total": row["n_total"],
                "N_unsupported": row["n_unsupported"],
            }
        )

    detail_csv_path = os.path.join(args.output_dir, f"{args.baseline_name}_oracle_by_dataset_language.csv")
    write_csv(
        detail_csv_path,
        detail_rows,
        headers=["Baseline", "Dataset", "Language", "Pass@1", "Pass@10", "N", "N_total", "N_unsupported"],
    )

    status_csv_path = os.path.join(args.output_dir, f"{args.baseline_name}_oracle_status_counts.csv")
    write_csv(status_csv_path, status_rows, headers=["Dataset", "Language", "Status", "Count"])

    tables_md_path = os.path.join(args.output_dir, f"{args.baseline_name}_oracle_tables.md")
    write_oracle_tables_markdown(tables_md_path, args.baseline_name, detail_rows)

    print(f"\n{'='*60}")
    print(f"  Oracle Pass@1 : {agg['overall']['pass@1']:.4f}")
    print(f"  Judged / Total: {agg['overall']['n_judged']} / {agg['overall']['n_total']}")
    print(f"{'='*60}")
    print(f"Saved json:    {json_path}")
    print(f"Saved detail:  {detail_csv_path}")
    print(f"Saved status:  {status_csv_path}")
    print(f"Saved tables:  {tables_md_path}")


if __name__ == "__main__":
    main()
