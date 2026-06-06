#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_project_go_bin = ROOT / ".micromamba" / "envs" / "eif-bench" / "go" / "bin"
if _project_go_bin.exists():
    os.environ["PATH"] = f"{_project_go_bin}:{os.environ.get('PATH', '')}"
else:
    _go_bin = Path("/usr/local/go/bin")
    if _go_bin.exists() and not any((Path(part) / "go").exists() for part in os.environ.get("PATH", "").split(os.pathsep) if part):
        os.environ["PATH"] = f"{_go_bin}:{os.environ.get('PATH', '')}"

from scripts.benchmark.eval_judges import (  # noqa: E402
    judge_mceval,
    normalize_go_mceval_source,
    prepare_go_mceval_test,
    run_cmd,
    sanitize_prediction,
    toolchain_available,
)
from scripts.go_singleline_fim_exp.go_single_pipeline import write_jsonl  # noqa: E402


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_predictions(path: Path) -> dict[str, list[str]]:
    pred_map: dict[str, list[str]] = {}
    for row in iter_jsonl(path):
        uid = str(row.get("uid") or "")
        if not uid:
            continue
        values = row.get("predictions")
        if values is None:
            values = row.get("raw_generation")
        if values is None and "prediction" in row:
            values = [row.get("prediction")]
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        out: list[str] = []
        for item in values:
            if isinstance(item, dict):
                text = item.get("text") or item.get("prediction") or item.get("content") or ""
            else:
                text = item
            if text is not None:
                out.append(str(text))
        pred_map[uid] = out
    return pred_map


def first_go_statement(text: str) -> str:
    text = sanitize_prediction(text).strip("\n")
    if not text.strip():
        return text
    lines = text.splitlines()
    kept: list[str] = []
    brace_depth = 0
    for line in lines:
        kept.append(line)
        brace_depth += line.count("{") - line.count("}")
        stripped = line.strip()
        if brace_depth <= 0 and stripped:
            if stripped.startswith("return") or re.search(r"(?<![=!<>:])(?::=|=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=)(?![=>])", stripped):
                break
            if re.match(r"^(?:[A-Za-z_]\w*|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*\(.*\)$", stripped):
                break
    return "\n".join(kept).strip("\n")


def clean_prediction(text: str, row: dict[str, Any]) -> str:
    text = sanitize_prediction(text).strip("\n")
    if str(row.get("task_type", "")).endswith("single_statement_completion_derived") or row.get("target_kind") in {"assignment", "return", "call"}:
        return first_go_statement(text)
    return text


def full_code_for(row: dict[str, Any], prediction: str) -> str:
    judge_payload = row.get("judge_payload", {}) or {}
    prefix = str(judge_payload.get("judge_prefix", row.get("prefix", "")))
    suffix = str(judge_payload.get("judge_suffix", row.get("suffix", "")))
    return prefix + prediction + suffix


def strip_leading_go_imports(source: str) -> str:
    pattern = re.compile(r"(?ms)^\s*import\s*(?:\([^)]*\)|\"[^\"]+\")\s*")
    previous = None
    while previous != source:
        previous = source
        source = pattern.sub("", source, count=1)
    return source.lstrip()


def wrap_humaneval_x_go_code(full_code: str) -> str:
    full_code = strip_leading_go_imports(full_code)
    return "\n".join([
        "package main",
        "",
        "import (",
        '    "bytes"',
        '    "fmt"',
        '    "math"',
        '    "math/rand"',
        '    "regexp"',
        '    "sort"',
        '    "strconv"',
        '    "strings"',
        '    "time"',
        '    "unicode"',
        ")",
        "",
        "var _ = bytes.Compare",
        "var _ = fmt.Sprintf",
        "var _ = math.Abs",
        "var _ = rand.New",
        "var _ = regexp.MustCompile",
        "var _ = sort.Ints",
        "var _ = strconv.Itoa",
        "var _ = strings.Builder{}",
        "var _ = time.Now",
        "var _ = unicode.IsLetter",
        "",
        full_code.strip(),
        "",
    ])


def judge_humaneval_x_go(full_code: str, row: dict[str, Any], timeout_sec: int) -> tuple[bool | None, str, str]:
    available, reason = toolchain_available("go")
    if not available:
        return None, "unsupported", reason
    payload = row.get("judge_payload", {}) or {}
    test_src = str(payload.get("test") or "")
    if not test_src.strip():
        return None, "unsupported", "missing humaneval-x test"

    full_code = normalize_go_mceval_source(full_code)
    program = wrap_humaneval_x_go_code(full_code)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        go_path = work / "main.go"
        test_path = work / "main_test.go"
        go_path.write_text(program, encoding="utf-8")
        test_path.write_text(prepare_go_mceval_test(test_src), encoding="utf-8")
        rc, stdout, stderr, timed_out = run_cmd(
            ["go", "test", str(go_path), str(test_path)],
            cwd=str(work),
            timeout_sec=timeout_sec,
        )
        if timed_out:
            return False, "timeout", (stderr or stdout)[:300]
        if rc != 0:
            err = stderr or stdout
            if "syntax error" in err or "undefined" in err or "cannot" in err:
                return False, "compile_error", err[:300]
            return False, "runtime_error", err[:300]
        return True, "ok", ""


def judge_row(row: dict[str, Any], full_code: str, timeout_sec: int) -> tuple[bool | None, str, str]:
    kind = str((row.get("judge_payload", {}) or {}).get("kind") or "")
    if kind == "mceval_go_test":
        return judge_mceval("go", full_code, row, timeout_sec=timeout_sec)
    if kind == "derived_humaneval_x_go_test":
        return judge_humaneval_x_go(full_code, row, timeout_sec=timeout_sec)
    return None, "unsupported", f"unsupported judge kind: {kind}"


def perfect_codebleu() -> dict[str, float]:
    return {
        "codebleu": 1.0,
        "ngram_match_score": 1.0,
        "weighted_ngram_match_score": 1.0,
        "syntax_match_score": 1.0,
        "dataflow_match_score": 1.0,
    }


def import_codebleu():
    try:
        from codebleu import calc_codebleu  # type: ignore
    except ModuleNotFoundError:
        return None
    return calc_codebleu


def compute_codebleu(reference: str, prediction: str, calc_codebleu: Any) -> dict[str, Any]:
    if reference == prediction:
        return perfect_codebleu()
    if calc_codebleu is None:
        return {"error": "codebleu package not installed"}
    return calc_codebleu(references=[reference], predictions=[prediction], lang="go")


def empty_bucket() -> dict[str, float]:
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


def add_codebleu(bucket: dict[str, float], score: dict[str, Any]) -> None:
    bucket["n_total"] += 1
    if "error" in score:
        bucket["n_failed"] += 1
        return
    bucket["n_supported"] += 1
    for key in ("codebleu", "ngram_match_score", "weighted_ngram_match_score", "syntax_match_score", "dataflow_match_score"):
        bucket[f"{key}_sum"] += float(score.get(key, 0.0))


def finalize_bucket(bucket: dict[str, float]) -> dict[str, Any]:
    n = int(bucket["n_supported"])
    out: dict[str, Any] = {
        "n_total": int(bucket["n_total"]),
        "n_supported": n,
        "n_failed": int(bucket["n_failed"]),
    }
    for key in ("codebleu", "ngram_match_score", "weighted_ngram_match_score", "syntax_match_score", "dataflow_match_score"):
        out[key] = bucket[f"{key}_sum"] / n if n else 0.0
    return out


def predictions_for(row: dict[str, Any], pred_map: dict[str, list[str]], oracle: bool, pass_k: int) -> list[str]:
    if oracle:
        target = row.get("target")
        if target is None:
            return []
        return [str(target)] * max(1, pass_k)
    return pred_map.get(str(row.get("uid") or ""), [])[: max(1, pass_k)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Go external FIM benchmark predictions with pass@k and CodeBLEU.")
    p.add_argument("--eval-data", type=Path, nargs="+", default=[
        Path("data/go_single_fim/test_data/mceval/mceval_go_single_v2_canonical.jsonl"),
        Path("data/go_single_fim/test_data/humaneval_x/humaneval_x_go_derived_canonical.jsonl"),
    ])
    p.add_argument("--predictions", type=Path, help="JSONL with uid + prediction/predictions/raw_generation. Not needed with --oracle.")
    p.add_argument("--oracle", action="store_true", help="Use gold target repeated k times as predictions; validates evaluator plumbing.")
    p.add_argument("--pass-k", type=int, default=10)
    p.add_argument("--timeout-sec", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("outputs/go_singleline_fim_exp/eval_results/go_external_fim_eval_results.jsonl"))
    p.add_argument("--summary", type=Path, default=Path("outputs/go_singleline_fim_exp/eval_results/go_external_fim_eval_summary.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.oracle and args.predictions is None:
        raise SystemExit("--predictions is required unless --oracle is set")
    pred_map = {} if args.oracle else load_predictions(args.predictions)
    calc_codebleu = import_codebleu()

    rows: list[dict[str, Any]] = []
    for path in args.eval_data:
        rows.extend(iter_jsonl(path))

    results: list[dict[str, Any]] = []
    totals = Counter()
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    cb_overall = empty_bucket()
    cb_by_dataset: dict[str, dict[str, float]] = defaultdict(empty_bucket)

    for row in rows:
        uid = str(row.get("uid") or "")
        dataset = str(row.get("source_dataset") or "unknown")
        totals["n_total"] += 1
        by_dataset[dataset]["n_total"] += 1
        preds = predictions_for(row, pred_map, args.oracle, args.pass_k)
        if not preds:
            totals["n_missing_prediction"] += 1
            by_dataset[dataset]["n_missing_prediction"] += 1
            results.append({"uid": uid, "source_dataset": dataset, "status": "missing_prediction", "pass1": False, f"pass@{args.pass_k}": False})
            continue

        target = str(row.get("target") or "")
        reference_full = full_code_for(row, target)
        candidate_results: list[dict[str, Any]] = []
        sample_pass1 = False
        sample_passk = False
        greedy_codebleu: dict[str, Any] | None = None

        for idx, raw_pred in enumerate(preds):
            pred = clean_prediction(str(raw_pred), row)
            candidate_full = full_code_for(row, pred)
            ok, status, err = judge_row(row, candidate_full, timeout_sec=args.timeout_sec)
            if idx == 0:
                greedy_codebleu = compute_codebleu(reference_full, candidate_full, calc_codebleu)
                add_codebleu(cb_overall, greedy_codebleu)
                add_codebleu(cb_by_dataset[dataset], greedy_codebleu)
            if ok is True:
                if idx == 0:
                    sample_pass1 = True
                sample_passk = True
            candidate_results.append({
                "index": idx,
                "prediction": pred,
                "pass": ok,
                "status": status,
                "error_summary": err,
            })
            if sample_passk:
                break

        totals["n_supported"] += 1
        by_dataset[dataset]["n_supported"] += 1
        if sample_pass1:
            totals["n_pass1"] += 1
            by_dataset[dataset]["n_pass1"] += 1
        if sample_passk:
            totals["n_passk"] += 1
            by_dataset[dataset]["n_passk"] += 1

        results.append({
            "uid": uid,
            "source_dataset": dataset,
            "target_kind": row.get("target_kind"),
            "pass1": sample_pass1,
            f"pass@{args.pass_k}": sample_passk,
            "greedy_codebleu": greedy_codebleu,
            "candidates": candidate_results,
        })

    n_supported = totals["n_supported"]
    summary: dict[str, Any] = {
        "eval_data": [str(p) for p in args.eval_data],
        "mode": "oracle" if args.oracle else "predictions",
        "n_total": totals["n_total"],
        "n_supported": n_supported,
        "n_missing_prediction": totals["n_missing_prediction"],
        "pass1": totals["n_pass1"] / n_supported if n_supported else 0.0,
        f"pass@{args.pass_k}": totals["n_passk"] / n_supported if n_supported else 0.0,
        "codebleu": finalize_bucket(cb_overall),
        "by_dataset": {},
    }
    for dataset, c in sorted(by_dataset.items()):
        ds_supported = c["n_supported"]
        summary["by_dataset"][dataset] = {
            "n_total": c["n_total"],
            "n_supported": ds_supported,
            "n_missing_prediction": c["n_missing_prediction"],
            "pass1": c["n_pass1"] / ds_supported if ds_supported else 0.0,
            f"pass@{args.pass_k}": c["n_passk"] / ds_supported if ds_supported else 0.0,
            "codebleu": finalize_bucket(cb_by_dataset[dataset]),
        }

    write_jsonl(args.output, results)
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] results={args.output}")
    print(f"[done] summary={args.summary}")


if __name__ == "__main__":
    main()
