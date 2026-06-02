#!/usr/bin/env python
"""Build a saliency manifest from Ours-vs-CLEAR failure viewer data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a precise code completion assistant. Complete exactly one missing code span. "
    "Return code only, no explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure_data", default="outputs/visual_failure/ours_vs_clear_humaneval1000.json")
    parser.add_argument("--output_path", default="outputs/visual_failure/ours_vs_clear_humaneval1000_saliency_manifest.json")
    parser.add_argument("--category", default="", help="Optional failure category filter.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of samples after filtering.")
    parser.add_argument("--include_row", action="store_true", default=True)
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def render_fim(prefix: str, suffix: str) -> str:
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"


def render_chatml_fim_user(language: str, prefix: str, suffix: str) -> str:
    return (
        f"Complete the missing {language} code span using FIM format.\\n"
        "Return only the missing span.\\n\\n"
        f"{render_fim(prefix, suffix)}"
    )


def make_row(sample: dict[str, Any]) -> dict[str, Any]:
    prefix = str(sample.get("prefix", ""))
    suffix = str(sample.get("suffix", ""))
    target = str(sample.get("ground_truth", ""))
    language = str(sample.get("language", ""))
    return {
        "uid": sample.get("uid", ""),
        "source_dataset": sample.get("source_dataset", ""),
        "split": "eval",
        "language": language,
        "task_type": sample.get("task_type", ""),
        "raw_id": sample.get("raw_id", ""),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_chatml_fim_user(language, prefix, suffix)},
            {"role": "assistant", "content": target},
        ],
        "fim_prompt": render_fim(prefix, suffix),
        "fim_completion": target,
        "metadata": {"entry_point": sample.get("entry_point", "")},
        "judge_payload": {},
    }


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.failure_data).read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if args.category:
        samples = [s for s in samples if args.category in s.get("categories", [])]
    if args.limit > 0:
        samples = samples[: args.limit]

    out_samples: list[dict[str, Any]] = []
    for sample in samples:
        uid = str(sample.get("uid") or sample.get("key") or sample.get("raw_id"))
        item = {
            "sample_id": f"failure_{sample.get('filtered_index', len(out_samples))}_{slugify(uid)}",
            "row_index": sample.get("filtered_index"),
            "uid": sample.get("uid"),
            "source_dataset": sample.get("source_dataset"),
            "language": sample.get("language"),
            "raw_id": sample.get("raw_id"),
            "selection": {
                "categories": sample.get("categories", []),
                "ours_pass1": sample.get("ours", {}).get("pass1"),
                "ours_pass10": sample.get("ours", {}).get("pass10"),
                "clear_pass1": sample.get("clear", {}).get("pass1"),
                "clear_pass10": sample.get("clear", {}).get("pass10"),
            },
        }
        if args.include_row:
            item["row"] = make_row(sample)
        out_samples.append(item)

    payload = {
        "version": 1,
        "source": args.failure_data,
        "usage": "Pass this file to tools/visual_saliency/compute_model_saliency.py --sample_manifest.",
        "samples": out_samples,
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(out_samples)} samples)")


if __name__ == "__main__":
    main()
