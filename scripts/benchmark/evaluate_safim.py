#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.benchmark_official_common import (  # noqa: E402
    SAFIM_TASK_TO_OFFICIAL,
    load_prediction_map,
    read_jsonl,
    render_chatml_messages,
    render_chatml_text,
    sanitize_completion,
    write_json,
    write_jsonl,
)

OFFICIAL_ROOT = ROOT / "scripts" / "benchmark" / "official_evaluators" / "safim"
SAFIM_FILES = {
    "python": "safim_python.jsonl",
    "java": "safim_java.jsonl",
    "cpp": "safim_cpp.jsonl",
    "csharp": "safim_csharp.jsonl",
}


def load_rows(test_dir: Path, languages: set[str] | None = None, task_type: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for language, filename in SAFIM_FILES.items():
        if languages and language not in languages:
            continue
        path = test_dir / filename
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if task_type and row.get("task_type") != task_type:
                continue
            rows.append(row)
    return rows


def parse_languages(value: str) -> set[str] | None:
    if not value.strip():
        return None
    out = {x.strip().lower() for x in value.split(",") if x.strip()}
    unknown = out - set(SAFIM_FILES)
    if unknown:
        raise ValueError(f"unsupported SAFIM languages: {sorted(unknown)}")
    return out


def command_prepare(args: argparse.Namespace) -> None:
    rows = load_rows(args.test_dir, parse_languages(args.languages), args.task_type or None)
    chatml_rows = []
    request_rows = []
    for row in rows:
        messages = render_chatml_messages(row["language"], row["prefix"], row["suffix"])
        official_task = SAFIM_TASK_TO_OFFICIAL[row["task_type"]]
        common = {
            "uid": row["uid"],
            "benchmark": row["benchmark"],
            "completion_type": official_task,
            "language": row["language"],
            "task_type": row["task_type"],
            "official_task_id": row["official_task_id"],
            "prefix": row["prefix"],
            "suffix": row["suffix"],
            "target": row["target"],
        }
        chatml_rows.append({**common, "messages": messages, "only_last_turn_loss": False})
        request_rows.append({**common, "prompt": render_chatml_text(messages), "max_new_tokens": args.max_new_tokens})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chatml_path = args.out_dir / "safim_chatml.jsonl"
    requests_path = args.out_dir / "safim_infer_requests.jsonl"
    report = {
        "input": str(args.test_dir),
        "rows": len(rows),
        "chatml": str(chatml_path),
        "infer_requests": str(requests_path),
    }
    write_jsonl(chatml_path, chatml_rows)
    write_jsonl(requests_path, request_rows)
    write_json(args.out_dir / "safim_prepare_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_make_oracle(args: argparse.Namespace) -> None:
    rows = load_rows(args.test_dir, parse_languages(args.languages), args.task_type or None)
    out_rows = [
        {"uid": row["uid"], "prediction": row["target"], "predictions": [row["target"]] * args.num_samples}
        for row in rows
    ]
    n = write_jsonl(args.output_path, out_rows)
    print(json.dumps({"output": str(args.output_path), "rows": n}, ensure_ascii=False, indent=2))


def postprocess_rows(rows: list[dict[str, Any]], pred_map: dict[str, list[str]]) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in SAFIM_TASK_TO_OFFICIAL.values()}
    missing: list[str] = []
    for row in rows:
        preds = pred_map.get(row["uid"])
        if not preds:
            missing.append(row["uid"])
            continue
        completion_type = SAFIM_TASK_TO_OFFICIAL[row["task_type"]]
        for pred in preds:
            grouped[completion_type].append({"task_id": row["official_task_id"], "completion": sanitize_completion(pred)})
    return grouped, missing


def command_postprocess(args: argparse.Namespace) -> None:
    rows = load_rows(args.test_dir, parse_languages(args.languages), args.task_type or None)
    pred_map = load_prediction_map(args.predictions_path)
    grouped, missing = postprocess_rows(rows, pred_map)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, int] = {}
    for completion_type, out_rows in grouped.items():
        if not out_rows:
            continue
        path = args.out_dir / f"safim_{completion_type}_official_samples.jsonl"
        outputs[str(path)] = write_jsonl(path, out_rows)
    if missing:
        write_jsonl(args.out_dir / "safim_missing_predictions.jsonl", [{"uid": uid} for uid in missing])
    report = {"outputs": outputs, "missing_predictions": len(missing)}
    write_json(args.out_dir / "safim_postprocess_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_eval_exact(args: argparse.Namespace) -> None:
    rows = load_rows(args.test_dir, parse_languages(args.languages), args.task_type or None)
    pred_map = load_prediction_map(args.predictions_path)
    totals: dict[str, dict[str, int]] = {}
    missing = 0
    for row in rows:
        key = f"{row['task_type']}::{row['language']}"
        totals.setdefault(key, {"passed": 0, "total": 0})
        preds = pred_map.get(row["uid"], [])
        if not preds:
            missing += 1
            passed = False
        else:
            passed = any(sanitize_completion(pred) == row["target"] for pred in preds[: args.k])
        totals[key]["passed"] += int(passed)
        totals[key]["total"] += 1

    summary: dict[str, Any] = {"missing_predictions": missing, "groups": {}}
    all_passed = all_total = 0
    for key, item in sorted(totals.items()):
        passed = item["passed"]
        total = item["total"]
        all_passed += passed
        all_total += total
        summary["groups"][key] = {"passed": passed, "total": total, "pass_rate": passed / total if total else 0.0}
    summary["all"] = {"passed": all_passed, "total": all_total, "pass_rate": all_passed / all_total if all_total else 0.0}
    write_json(args.out_dir / "safim_exact_eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_run_official(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_files = {
        "block": args.samples_dir / "safim_block_official_samples.jsonl",
        "control": args.samples_dir / "safim_control_official_samples.jsonl",
        "api": args.samples_dir / "safim_api_official_samples.jsonl",
    }
    outputs: dict[str, str] = {}
    for completion_type, sample_path in sample_files.items():
        if not sample_path.exists():
            continue
        out_path = args.out_dir / f"safim_{completion_type}_official_results.jsonl"
        cmd = [sys.executable, str(OFFICIAL_ROOT / "evaluate.py"), completion_type, str(sample_path), str(out_path)]
        subprocess.run(cmd, cwd=str(OFFICIAL_ROOT), check=True)
        outputs[completion_type] = str(out_path)
    write_json(args.out_dir / "safim_official_eval_outputs.json", outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAFIM official-format benchmark wrapper.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--test-dir", type=Path, default=Path("data/benchmark/test_data"))
    common.add_argument("--languages", default="", help="Comma-separated subset: python,java,cpp,csharp")
    common.add_argument("--task-type", default="", choices=["", *SAFIM_TASK_TO_OFFICIAL.keys()])

    p = sub.add_parser("prepare", parents=[common])
    p.add_argument("--out-dir", type=Path, default=Path("data/benchmark/test_data/safim_prepared"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.set_defaults(func=command_prepare)

    p = sub.add_parser("make-oracle", parents=[common])
    p.add_argument("--output-path", type=Path, default=Path("data/benchmark/test_data/safim_oracle_predictions.jsonl"))
    p.add_argument("--num-samples", type=int, default=10)
    p.set_defaults(func=command_make_oracle)

    p = sub.add_parser("postprocess", parents=[common])
    p.add_argument("--predictions-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/benchmark/test_data/safim_official_samples"))
    p.set_defaults(func=command_postprocess)

    p = sub.add_parser("eval-exact", parents=[common])
    p.add_argument("--predictions-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark/eval_results/safim"))
    p.add_argument("--k", type=int, default=1)
    p.set_defaults(func=command_eval_exact)

    p = sub.add_parser("run-official")
    p.add_argument("--samples-dir", type=Path, default=Path("data/benchmark/test_data/safim_official_samples"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark/eval_results/safim_official"))
    p.set_defaults(func=command_run_official)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
