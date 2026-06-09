#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

HUMANEVAL_FILES = {
    "single_line": "HumanEval-SingleLineInfilling.jsonl.gz",
    "multi_line": "HumanEval-MultiLineInfilling.jsonl.gz",
    "random_span": "HumanEval-RandomSpanInfilling.jsonl.gz",
    "random_span_light": "HumanEval-RandomSpanInfillingLight.jsonl.gz",
}

SAFIM_FILES = {
    "algorithmic_block": "block_completion.jsonl.gz",
    "control_flow_expression": "control_completion_fixed.jsonl.gz",
    "api_function_call": "api_completion.jsonl.gz",
}

LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "java": "java",
    "cpp": "cpp",
    "c++": "cpp",
    "cs": "csharp",
    "c#": "csharp",
    "csharp": "csharp",
}

TARGET_SAFIM_LANGUAGES = ("python", "java", "cpp", "csharp")
SAFIM_MARKER = "{{completion}}"


def read_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def normalize_language(lang: str) -> str:
    key = (lang or "").strip().lower()
    if key not in LANGUAGE_ALIASES:
        raise ValueError(f"unsupported SAFIM language: {lang!r}")
    return LANGUAGE_ALIASES[key]


def build_humaneval(raw_root: Path, out_dir: Path) -> dict:
    raw_dir = raw_root / "humaneval_infilling"

    def rows() -> Iterable[dict]:
        for task_type, filename in HUMANEVAL_FILES.items():
            path = raw_dir / filename
            for idx, obj in enumerate(read_jsonl_gz(path)):
                uid = f"humaneval_infilling:python:{task_type}:{idx:06d}"
                yield {
                    "uid": uid,
                    "benchmark": "humaneval_infilling",
                    "source": "official",
                    "language": "python",
                    "task_type": task_type,
                    "official_task_id": obj["task_id"],
                    "entry_point": obj.get("entry_point"),
                    "prefix": obj.get("prompt", ""),
                    "suffix": obj.get("suffix", ""),
                    "target": obj.get("canonical_solution", ""),
                    "test": obj.get("test", ""),
                    "unit_tests": None,
                    "official_prompt": obj.get("prompt", ""),
                    "official_eval_prompt": None,
                    "raw_fields": sorted(obj.keys()),
                }

    out_path = out_dir / "humaneval_infilling_python.jsonl"
    return {str(out_path): write_jsonl(out_path, rows())}


def split_safim_eval_prompt(eval_prompt: str) -> tuple[str, str]:
    if SAFIM_MARKER not in eval_prompt:
        raise ValueError(f"SAFIM eval_prompt missing {SAFIM_MARKER!r}")
    prefix, suffix = eval_prompt.split(SAFIM_MARKER, 1)
    return prefix, suffix


def build_safim(raw_root: Path, out_dir: Path) -> dict:
    raw_dir = raw_root / "safim"
    buckets: dict[str, list[dict]] = {lang: [] for lang in TARGET_SAFIM_LANGUAGES}

    for task_type, filename in SAFIM_FILES.items():
        path = raw_dir / filename
        for idx, obj in enumerate(read_jsonl_gz(path)):
            language = normalize_language(obj.get("lang", ""))
            if language not in buckets:
                continue
            prefix, suffix = split_safim_eval_prompt(obj.get("eval_prompt", ""))
            official_task_id = obj.get("task_id") or f"{task_type}_{idx:06d}"
            uid = f"safim:{language}:{task_type}:{official_task_id}"
            buckets[language].append(
                {
                    "uid": uid,
                    "benchmark": "safim",
                    "source": "official",
                    "language": language,
                    "task_type": task_type,
                    "official_task_id": official_task_id,
                    "entry_point": None,
                    "prefix": prefix,
                    "suffix": suffix,
                    "target": obj.get("ground_truth", ""),
                    "test": None,
                    "unit_tests": obj.get("unit_tests", ""),
                    "official_prompt": obj.get("prompt", ""),
                    "official_eval_prompt": obj.get("eval_prompt", ""),
                    "raw_fields": sorted(obj.keys()),
                }
            )

    counts = {}
    for language, rows in buckets.items():
        out_path = out_dir / f"safim_{language}.jsonl"
        counts[str(out_path)] = write_jsonl(out_path, rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize official HumanEval-Infilling and SAFIM raw data into five benchmark test JSONL files."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/benchmark/test_data"))
    args = parser.parse_args()

    counts = {}
    counts.update(build_humaneval(args.raw_root, args.out_dir))
    counts.update(build_safim(args.raw_root, args.out_dir))

    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps({"outputs": counts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": counts, "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
