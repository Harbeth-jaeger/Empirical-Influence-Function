#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_csv_filter(text: str) -> set[str]:
    return {part.strip().lower() for part in text.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small benchmark eval subset JSONL.")
    parser.add_argument("--input_path", default="data/benchmarks/eval_data/rendered_chatml_fim_eval.jsonl")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--source_datasets", default="", help="Comma-separated source_dataset filter, e.g. safim.")
    parser.add_argument("--languages", default="", help="Comma-separated language filter, e.g. python,java,cpp,csharp.")
    parser.add_argument("--max_rows", type=int, default=0, help="Global max rows after filters. 0 means unlimited.")
    parser.add_argument(
        "--per_group_limit",
        type=int,
        default=0,
        help="Max rows per (source_dataset, language) group after filters. 0 means unlimited.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_filter = parse_csv_filter(args.source_datasets)
    lang_filter = parse_csv_filter(args.languages)

    rows: list[dict[str, Any]] = []
    group_counts: Counter[tuple[str, str]] = Counter()

    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source_dataset", "")).lower()
            lang = str(row.get("language", "")).lower()
            if source_filter and source not in source_filter:
                continue
            if lang_filter and lang not in lang_filter:
                continue

            key = (source or "unknown", lang or "unknown")
            if args.per_group_limit > 0 and group_counts[key] >= args.per_group_limit:
                continue

            rows.append(row)
            group_counts[key] += 1
            if args.max_rows > 0 and len(rows) >= args.max_rows:
                break

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} rows to {output_path}")
    for (source, lang), count in sorted(group_counts.items()):
        print(f"  {source}/{lang}: {count}")


if __name__ == "__main__":
    main()
