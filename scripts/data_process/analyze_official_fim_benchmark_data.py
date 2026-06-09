#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rough_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def nonempty_lines(text: str) -> int:
    return sum(1 for line in (text or "").splitlines() if line.strip())


def quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def stats(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "p25": 0, "mean": 0, "p50": 0, "p75": 0, "p90": 0, "p95": 0, "max": 0}
    return {
        "min": min(values),
        "p25": round(quantile(values, 0.25), 2),
        "mean": round(mean(values), 2),
        "p50": round(quantile(values, 0.50), 2),
        "p75": round(quantile(values, 0.75), 2),
        "p90": round(quantile(values, 0.90), 2),
        "p95": round(quantile(values, 0.95), 2),
        "max": max(values),
    }


def collect_files(test_dir: Path) -> list[Path]:
    names = [
        "humaneval_infilling_python.jsonl",
        "safim_python.jsonl",
        "safim_java.jsonl",
        "safim_cpp.jsonl",
        "safim_csharp.jsonl",
    ]
    return [test_dir / name for name in names]


def analyze_file(path: Path) -> dict:
    rows = list(read_jsonl(path))
    field_sets = [set(row.keys()) for row in rows]
    fields_union = sorted(set().union(*field_sets)) if field_sets else []
    fields_intersection = sorted(set.intersection(*field_sets)) if field_sets else []
    task_counts = Counter(row.get("task_type", "unknown") for row in rows)

    length_values = defaultdict(list)
    for row in rows:
        for field in ("prefix", "suffix", "target"):
            text = row.get(field) or ""
            length_values[f"{field}_chars"].append(len(text))
            length_values[f"{field}_rough_tokens"].append(rough_tokens(text))
            length_values[f"{field}_nonempty_lines"].append(nonempty_lines(text))
        context = (row.get("prefix") or "") + (row.get("suffix") or "")
        length_values["context_chars"].append(len(context))
        length_values["context_rough_tokens"].append(rough_tokens(context))
        length_values["context_nonempty_lines"].append(nonempty_lines(context))

    total = len(rows)
    first = rows[0] if rows else {}
    return {
        "path": str(path),
        "file": path.name,
        "benchmark": first.get("benchmark"),
        "language": first.get("language"),
        "num_samples": total,
        "fields_union": fields_union,
        "fields_intersection": fields_intersection,
        "task_type_counts": dict(task_counts),
        "task_type_ratios": {k: round(v / total, 6) for k, v in task_counts.items()} if total else {},
        "length_stats": {name: stats(values) for name, values in sorted(length_values.items())},
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Official FIM Benchmark Data Analysis",
        "",
        "## Files",
        "",
        "| file | benchmark | language | samples |",
        "| --- | --- | --- | ---: |",
    ]
    for item in summary["files"]:
        lines.append(f"| `{item['file']}` | {item['benchmark']} | {item['language']} | {item['num_samples']} |")

    lines += [
        "",
        "## Common Fields",
        "",
        "Fields shared by all five normalized files:",
        "",
        ", ".join(f"`{field}`" for field in summary["common_fields_all_files"]),
        "",
        "## Task Type Distribution",
        "",
        "| file | task_type | count | ratio |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in summary["task_type_rows"]:
        lines.append(f"| `{row['file']}` | {row['task_type']} | {row['count']} | {row['ratio']:.4f} |")

    lines += [
        "",
        "## Target Length",
        "",
        "| file | target token mean | target token p50 | target token p90 | target lines mean | target lines p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["files"]:
        tok = item["length_stats"]["target_rough_tokens"]
        lines_ = item["length_stats"]["target_nonempty_lines"]
        lines.append(
            f"| `{item['file']}` | {tok['mean']} | {tok['p50']} | {tok['p90']} | {lines_['mean']} | {lines_['p90']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze normalized official FIM benchmark test data.")
    parser.add_argument("--test-dir", type=Path, default=Path("data/benchmark/test_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark/data_analysis"))
    args = parser.parse_args()

    files = collect_files(args.test_dir)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing normalized benchmark files: {missing}")

    file_summaries = [analyze_file(path) for path in files]
    common_fields = sorted(set.intersection(*(set(item["fields_intersection"]) for item in file_summaries)))

    task_type_rows = []
    for item in file_summaries:
        for task_type, count in item["task_type_counts"].items():
            task_type_rows.append(
                {
                    "file": item["file"],
                    "benchmark": item["benchmark"],
                    "language": item["language"],
                    "task_type": task_type,
                    "count": count,
                    "ratio": item["task_type_ratios"][task_type],
                }
            )

    length_rows = []
    for item in file_summaries:
        base = {"file": item["file"], "benchmark": item["benchmark"], "language": item["language"]}
        for metric, values in item["length_stats"].items():
            length_rows.append({**base, "metric": metric, **values})

    summary = {
        "test_dir": str(args.test_dir),
        "files": file_summaries,
        "common_fields_all_files": common_fields,
        "task_type_rows": task_type_rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "benchmark_data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        args.out_dir / "benchmark_task_type_counts.csv",
        task_type_rows,
        ["file", "benchmark", "language", "task_type", "count", "ratio"],
    )
    write_csv(
        args.out_dir / "benchmark_length_stats.csv",
        length_rows,
        ["file", "benchmark", "language", "metric", "min", "p25", "mean", "p50", "p75", "p90", "p95", "max"],
    )
    write_markdown(args.out_dir / "benchmark_data_analysis.md", summary)

    print(json.dumps({"summary": str(args.out_dir / "benchmark_data_summary.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
