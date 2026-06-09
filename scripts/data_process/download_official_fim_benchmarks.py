#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HUMANEVAL_FILES = {
    "single_line": "HumanEval-SingleLineInfilling.jsonl.gz",
    "multi_line": "HumanEval-MultiLineInfilling.jsonl.gz",
    "random_span": "HumanEval-RandomSpanInfilling.jsonl.gz",
    "random_span_light": "HumanEval-RandomSpanInfillingLight.jsonl.gz",
}
HUMANEVAL_BASE = "https://raw.githubusercontent.com/openai/human-eval-infilling/master/data"

SAFIM_FILES = {
    "algorithmic_block": "block_completion.jsonl.gz",
    "algorithmic_block_v2": "block_completion_v2.jsonl.gz",
    "control_flow_expression": "control_completion_fixed.jsonl.gz",
    "api_function_call": "api_completion.jsonl.gz",
}
SAFIM_BASE = "https://huggingface.co/datasets/gonglinyuan/safim/resolve/main"


def download(url: str, path: Path, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        print(f"skip existing {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"download {url} -> {path}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        f.write(resp.read())
    tmp.replace(path)


def write_manifest(root: Path, entries: list[dict]) -> None:
    manifest = {
        "humaneval_infilling_source": "https://github.com/openai/human-eval-infilling/tree/master/data",
        "safim_source": "https://huggingface.co/datasets/gonglinyuan/safim",
        "entries": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official HumanEval-Infilling and SAFIM raw benchmark data.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_data"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-humaneval", action="store_true")
    parser.add_argument("--skip-safim", action="store_true")
    args = parser.parse_args()

    entries: list[dict] = []
    if not args.skip_humaneval:
        out_dir = args.raw_root / "humaneval_infilling"
        for task_type, filename in HUMANEVAL_FILES.items():
            url = f"{HUMANEVAL_BASE}/{filename}"
            path = out_dir / filename
            download(url, path, overwrite=args.overwrite)
            entries.append({"benchmark": "humaneval_infilling", "task_type": task_type, "path": str(path), "url": url})
    if not args.skip_safim:
        out_dir = args.raw_root / "safim"
        for task_type, filename in SAFIM_FILES.items():
            url = f"{SAFIM_BASE}/{filename}"
            path = out_dir / filename
            download(url, path, overwrite=args.overwrite)
            entries.append({"benchmark": "safim", "task_type": task_type, "path": str(path), "url": url})
    write_manifest(args.raw_root, entries)


if __name__ == "__main__":
    main()
