#!/usr/bin/env python
"""Join two visual-failure eval dumps and classify per-sample outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours_dump", required=True)
    p.add_argument("--clear_dump", required=True)
    p.add_argument("--output_path", default="outputs/visual_failure/ours_vs_clear_humaneval1000.json")
    p.add_argument("--max_prediction_chars", type=int, default=20000)
    return p.parse_args()


def read_dump(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("uid") or row.get("filtered_index"))
            rows[key] = row
    return rows


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def trim_candidate(candidate: dict[str, Any], max_chars: int) -> dict[str, Any]:
    out = dict(candidate)
    out["prediction"] = trim_text(str(out.get("prediction", "")), max_chars)
    return out


def model_view(row: dict[str, Any], max_chars: int) -> dict[str, Any]:
    samples = [trim_candidate(c, max_chars) for c in row.get("samples", [])]
    return {
        "baseline": row.get("baseline", ""),
        "model_path": row.get("model_path", ""),
        "pass1": bool(row.get("pass1", 0.0)),
        "pass10": bool(row.get("pass10", 0.0)),
        "greedy": trim_candidate(row.get("greedy", {}), max_chars),
        "samples": samples,
    }


def classify(ours: dict[str, Any], clear: dict[str, Any]) -> list[str]:
    op1, op10 = bool(ours.get("pass1")), bool(ours.get("pass10"))
    cp1, cp10 = bool(clear.get("pass1")), bool(clear.get("pass10"))
    cats: list[str] = []
    if (not op10) and cp10:
        cats.append("clear_pass10_ours_fail10")
    if (not op1) and cp1:
        cats.append("clear_pass1_ours_fail1")
    if (not op1) and op10:
        cats.append("ours_unstable_pass10_fail1")
    if (not op10) and (not cp10):
        cats.append("both_fail10")
    if op10 and (not cp10):
        cats.append("ours_pass10_clear_fail10")
    if op1 and (not cp1):
        cats.append("ours_pass1_clear_fail1")
    if not cats:
        cats.append("other")
    return cats


def main() -> None:
    args = parse_args()
    ours_rows = read_dump(Path(args.ours_dump))
    clear_rows = read_dump(Path(args.clear_dump))
    common_keys = sorted(set(ours_rows) & set(clear_rows), key=lambda k: int(ours_rows[k].get("filtered_index", 10**12)))
    if not common_keys:
        raise ValueError("No common rows between dumps.")

    samples: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for key in common_keys:
        ours = ours_rows[key]
        clear = clear_rows[key]
        categories = classify(ours, clear)
        category_counts.update(categories)
        sample = {
            "key": key,
            "categories": categories,
            "filtered_index": ours.get("filtered_index"),
            "uid": ours.get("uid", ""),
            "source_dataset": ours.get("source_dataset", ""),
            "language": ours.get("language", ""),
            "raw_id": ours.get("raw_id", ""),
            "task_type": ours.get("task_type", ""),
            "entry_point": ours.get("entry_point", ""),
            "prefix": trim_text(str(ours.get("prefix", "")), args.max_prediction_chars),
            "suffix": trim_text(str(ours.get("suffix", "")), args.max_prediction_chars),
            "ground_truth": trim_text(str(ours.get("ground_truth", "")), args.max_prediction_chars),
            "ours": model_view(ours, args.max_prediction_chars),
            "clear": model_view(clear, args.max_prediction_chars),
        }
        samples.append(sample)

    data = {
        "config": {
            "ours_dump": args.ours_dump,
            "clear_dump": args.clear_dump,
            "max_prediction_chars": args.max_prediction_chars,
        },
        "categories": {
            "clear_pass10_ours_fail10": "CLEAR pass@10, Ours fail@10",
            "clear_pass1_ours_fail1": "CLEAR pass@1, Ours fail@1",
            "ours_unstable_pass10_fail1": "Ours pass@10 but fail@1",
            "both_fail10": "Both fail@10",
            "ours_pass10_clear_fail10": "Ours pass@10, CLEAR fail@10",
            "ours_pass1_clear_fail1": "Ours pass@1, CLEAR fail@1",
            "other": "Other",
        },
        "category_counts": dict(category_counts),
        "num_common": len(common_keys),
        "samples": samples,
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"common rows: {len(common_keys)}")
    print(dict(category_counts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
