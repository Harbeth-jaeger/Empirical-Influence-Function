#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.benchmark_official_common import (  # noqa: E402
    HUMANEVAL_TASK_TO_OFFICIAL,
    load_prediction_map,
    read_jsonl,
    render_chatml_messages,
    render_chatml_text,
    sanitize_completion,
    write_json,
    write_jsonl,
)

OFFICIAL_ROOT = ROOT / "scripts" / "benchmark" / "official_evaluators" / "human_eval_infilling"


def add_official_to_path() -> None:
    official = str(OFFICIAL_ROOT)
    if official not in sys.path:
        sys.path.insert(0, official)


def load_rows(test_path: Path, task_type: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(test_path)
    if task_type:
        rows = [row for row in rows if row.get("task_type") == task_type]
    return rows


def command_prepare(args: argparse.Namespace) -> None:
    rows = load_rows(args.test_path, args.task_type or None)
    chatml_rows = []
    request_rows = []
    for row in rows:
        messages = render_chatml_messages(row["language"], row["prefix"], row["suffix"])
        official_benchmark = HUMANEVAL_TASK_TO_OFFICIAL[row["task_type"]]
        common = {
            "uid": row["uid"],
            "benchmark": row["benchmark"],
            "benchmark_name": official_benchmark,
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
    chatml_path = args.out_dir / "humaneval_infilling_chatml.jsonl"
    requests_path = args.out_dir / "humaneval_infilling_infer_requests.jsonl"
    report = {
        "input": str(args.test_path),
        "rows": len(rows),
        "chatml": str(chatml_path),
        "infer_requests": str(requests_path),
    }
    write_jsonl(chatml_path, chatml_rows)
    write_jsonl(requests_path, request_rows)
    write_json(args.out_dir / "humaneval_infilling_prepare_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_make_oracle(args: argparse.Namespace) -> None:
    rows = load_rows(args.test_path, args.task_type or None)
    out_rows = [
        {"uid": row["uid"], "prediction": row["target"], "predictions": [row["target"]] * args.num_samples}
        for row in rows
    ]
    n = write_jsonl(args.output_path, out_rows)
    print(json.dumps({"output": str(args.output_path), "rows": n}, ensure_ascii=False, indent=2))


def postprocess_rows(rows: list[dict[str, Any]], pred_map: dict[str, list[str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in HUMANEVAL_TASK_TO_OFFICIAL.values()}
    missing: list[str] = []
    for row in rows:
        uid = row["uid"]
        preds = pred_map.get(uid)
        if not preds:
            missing.append(uid)
            continue
        benchmark_name = HUMANEVAL_TASK_TO_OFFICIAL[row["task_type"]]
        for pred in preds:
            grouped[benchmark_name].append({"task_id": row["official_task_id"], "completion": sanitize_completion(pred)})
    return grouped | {"__missing__": [{"uid": uid} for uid in missing]}  # type: ignore[return-value]


def command_postprocess(args: argparse.Namespace) -> None:
    rows = load_rows(args.test_path, args.task_type or None)
    pred_map = load_prediction_map(args.predictions_path)
    grouped = postprocess_rows(rows, pred_map)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, int] = {}
    for benchmark_name, out_rows in grouped.items():
        if benchmark_name == "__missing__":
            continue
        if not out_rows:
            continue
        path = args.out_dir / f"humaneval_{benchmark_name}_official_samples.jsonl"
        outputs[str(path)] = write_jsonl(path, out_rows)
    missing = grouped.get("__missing__", [])
    if missing:
        write_jsonl(args.out_dir / "humaneval_missing_predictions.jsonl", missing)
    report = {"outputs": outputs, "missing_predictions": len(missing)}
    write_json(args.out_dir / "humaneval_postprocess_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def summarize_existing_results(results_path: Path, k_values: list[int]) -> dict[str, float]:
    add_official_to_path()
    from human_eval_infilling.evaluation import estimate_pass_at_k  # type: ignore

    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in read_jsonl(results_path):
        grouped[row["task_id"]].append(bool(row["passed"]))
    if not grouped:
        raise SystemExit(f"Empty official result file: {results_path}")

    total = []
    correct = []
    for task_results in grouped.values():
        total.append(len(task_results))
        correct.append(sum(task_results))

    import numpy as np

    total_arr = np.array(total)
    correct_arr = np.array(correct)
    return {
        f"pass@{k}": float(estimate_pass_at_k(total_arr, correct_arr, k).mean())
        for k in k_values
        if (total_arr >= k).all()
    }


def command_eval(args: argparse.Namespace) -> None:
    add_official_to_path()
    from human_eval_infilling.evaluation import evaluate_functional_correctness  # type: ignore

    sample_files = sorted(args.samples_dir.glob("humaneval_*_official_samples.jsonl"))
    if args.benchmark_names:
        allowed = set(args.benchmark_names)
        sample_files = [
            path
            for path in sample_files
            if path.name.removeprefix("humaneval_").removesuffix("_official_samples.jsonl") in allowed
        ]
    if not sample_files:
        raise SystemExit(f"No official sample files found in {args.samples_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for sample_file in sample_files:
        name = sample_file.name
        benchmark_name = name.removeprefix("humaneval_").removesuffix("_official_samples.jsonl")
        results_path = Path(str(sample_file) + "_results.jsonl")
        if args.reuse_existing and results_path.exists():
            result = summarize_existing_results(results_path, args.k)
            print(f"Reused existing HumanEval-Infilling results: {results_path}")
        else:
            result = evaluate_functional_correctness(
                benchmark_name=benchmark_name,
                sample_file=str(sample_file),
                k=args.k,
                n_workers=args.n_workers,
                timeout=args.timeout,
            )
        summary[benchmark_name] = result
    write_json(args.out_dir / "humaneval_eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_eval_subset(args: argparse.Namespace) -> None:
    add_official_to_path()
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np
    import tqdm
    from human_eval_infilling.data import read_problems  # type: ignore
    from human_eval_infilling.evaluation import estimate_pass_at_k  # type: ignore
    from human_eval_infilling.execution import check_correctness  # type: ignore

    sample_rows = read_jsonl(args.sample_file)
    if not sample_rows:
        raise SystemExit(f"Empty sample file: {args.sample_file}")
    problems = read_problems(args.benchmark_name)
    completion_id: Counter[str] = Counter()
    results: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = []
        for sample in tqdm.tqdm(sample_rows, desc="submit"):
            task_id = sample["task_id"]
            if task_id not in problems:
                raise KeyError(f"Unknown task_id for {args.benchmark_name}: {task_id}")
            args_tuple = (problems[task_id], sample["completion"], args.timeout, completion_id[task_id])
            futures.append(executor.submit(check_correctness, *args_tuple))
            completion_id[task_id] += 1
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="check"):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    total = []
    correct = []
    for task_results in results.values():
        task_results.sort()
        passed = [bool(row[1]["passed"]) for row in task_results]
        total.append(len(passed))
        correct.append(sum(passed))
    total_arr = np.array(total)
    correct_arr = np.array(correct)
    summary = {
        f"pass@{k}": float(estimate_pass_at_k(total_arr, correct_arr, k).mean())
        for k in args.k
        if (total_arr >= k).all()
    }
    summary["attempted_problems"] = len(results)
    summary["samples"] = len(sample_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "humaneval_subset_eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HumanEval-Infilling official-format benchmark wrapper.")
    sub = parser.add_subparsers(dest="command", required=True)

    common_test = argparse.ArgumentParser(add_help=False)
    common_test.add_argument("--test-path", type=Path, default=Path("data/benchmark/test_data/humaneval_infilling_python.jsonl"))
    common_test.add_argument("--task-type", default="", choices=["", *HUMANEVAL_TASK_TO_OFFICIAL.keys()])

    p = sub.add_parser("prepare", parents=[common_test])
    p.add_argument("--out-dir", type=Path, default=Path("data/benchmark/test_data/humaneval_infilling_prepared"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.set_defaults(func=command_prepare)

    p = sub.add_parser("make-oracle", parents=[common_test])
    p.add_argument("--output-path", type=Path, default=Path("data/benchmark/test_data/humaneval_infilling_oracle_predictions.jsonl"))
    p.add_argument("--num-samples", type=int, default=10)
    p.set_defaults(func=command_make_oracle)

    p = sub.add_parser("postprocess", parents=[common_test])
    p.add_argument("--predictions-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/benchmark/test_data/humaneval_infilling_official_samples"))
    p.set_defaults(func=command_postprocess)

    p = sub.add_parser("eval")
    p.add_argument("--samples-dir", type=Path, default=Path("data/benchmark/test_data/humaneval_infilling_official_samples"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark/eval_results/humaneval_infilling"))
    p.add_argument("--k", type=int, nargs="+", default=[1])
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--benchmark-names", nargs="*", default=[], choices=list(HUMANEVAL_TASK_TO_OFFICIAL.values()))
    p.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=command_eval)

    p = sub.add_parser("eval-subset")
    p.add_argument("--sample-file", type=Path, required=True)
    p.add_argument("--benchmark-name", required=True, choices=list(HUMANEVAL_TASK_TO_OFFICIAL.values()))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark/eval_results/humaneval_infilling_subset"))
    p.add_argument("--k", type=int, nargs="+", default=[1])
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=3.0)
    p.set_defaults(func=command_eval_subset)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
