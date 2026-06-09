#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.go_singleline_fim_exp.go_single_pipeline import (  # noqa: E402
    BuildStats,
    CandidateSpan,
    build_mceval_candidates,
    count_nonempty_lines,
    extract_statement_candidates,
    find_function_slice,
    make_canonical_sample,
    render_chatml,
    stable_hash,
    write_jsonl,
)


FUNC_NAME_RE = re.compile(r"\bfunc\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(")
LINE_COMMENT_RE = re.compile(r"(?m)^\s*//.*(?:\n|$)")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


@dataclass
class ExternalBuildStats:
    benchmark: str
    rows_seen: int = 0
    rows_accepted: int = 0
    candidates_seen: int = 0
    reject_reasons: Counter[str] = field(default_factory=Counter)
    target_kinds: Counter[str] = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.reject_reasons[reason] += 1

    def accept(self, kind: str, *, candidates: int = 1) -> None:
        self.rows_accepted += 1
        self.candidates_seen += candidates
        self.target_kinds[kind] += 1


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading MultiPL-E parquet files requires pandas/pyarrow in the active env."
        ) from exc
    return pd.read_parquet(path).to_dict(orient="records")


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def strip_leading_go_comments(text: str) -> str:
    text = BLOCK_COMMENT_RE.sub("", text)
    text = LINE_COMMENT_RE.sub("", text)
    return text


def extract_go_func_name(source: str) -> str:
    m = FUNC_NAME_RE.search(source)
    return m.group(1) if m else ""


def choose_single_candidate(candidates: list[CandidateSpan]) -> CandidateSpan:
    kind_priority = {"assignment": 0, "return": 1, "call": 2}
    center = (len(candidates) - 1) / 2.0
    return min(
        enumerate(candidates),
        key=lambda item: (abs(item[0] - center), kind_priority.get(item[1].target_kind, 99), item[1].line_no),
    )[1]


def render_mask_chatml(sample: dict[str, Any]) -> dict[str, Any]:
    """Render the project-specific ChatML-MASK format.

    Most samples reuse go_single_pipeline.render_chatml.  MultiPL-E body-mask
    samples do not have a reference target, so this helper keeps an empty
    assistant turn only for schema compatibility.
    """
    if sample.get("target") is not None:
        return render_chatml(sample)
    user = (
        "Fill the [MASK] in the Go function. Return only the missing Go code, "
        "without Markdown fences or explanation.\n\n"
        "* Incomplete Code:\n"
        f"{sample['prefix']}[MASK]{sample['suffix']}"
    )
    out = dict(sample)
    out["messages"] = [
        {"role": "system", "content": "You are a Go code completion assistant."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ""},
    ]
    out["only_last_turn_loss"] = True
    return out


def build_humaneval_x_go(
    raw_path: Path,
    *,
    per_task: int,
    limit: int,
) -> tuple[list[dict[str, Any]], ExternalBuildStats]:
    stats = ExternalBuildStats("humaneval_x_go")
    samples: list[dict[str, Any]] = []
    seen_uid: set[str] = set()

    if not raw_path.exists():
        stats.reject("missing_raw_file")
        return samples, stats

    for row in iter_jsonl(raw_path):
        if limit > 0 and stats.rows_seen >= limit:
            break
        stats.rows_seen += 1
        task_id = str(row.get("task_id") or "")
        imports = str(row.get("import") or "")
        declaration = str(row.get("declaration") or "")
        solution = str(row.get("canonical_solution") or "")
        test = str(row.get("test") or "")
        test_setup = str(row.get("test_setup") or "")
        entry = extract_go_func_name(declaration)
        if not task_id or not declaration or not solution or not test or not entry:
            stats.reject("missing_required_field")
            continue

        prompt = str(row.get("prompt") or "")
        prompt_code = strip_leading_go_comments(prompt)
        if prompt_code.strip():
            reference_code = prompt_code + solution
        else:
            reference_code = declaration + solution
        func_slice = find_function_slice(reference_code, entry)
        if func_slice is None:
            stats.reject("entry_function_not_found")
            continue
        func_start, func_end, function_code = func_slice
        leading_code = reference_code[:func_start]
        trailing_code = reference_code[func_end:]

        candidates, reason = extract_statement_candidates(function_code)
        if reason:
            stats.reject(reason)
            continue
        if not candidates:
            stats.reject("no_statement_candidate")
            continue

        kept = 0
        ordered = [choose_single_candidate(candidates)] if per_task == 1 else candidates
        for cand in ordered:
            if kept >= per_task:
                break
            uid_seed = "\n".join([task_id, entry, str(cand.line_no), cand.target])
            uid = f"humaneval_x_go_{stable_hash(uid_seed)}"
            if uid in seen_uid:
                stats.reject("duplicate_uid")
                continue
            seen_uid.add(uid)
            sample = make_canonical_sample(
                uid=uid,
                source_dataset="humaneval_x_go",
                split="test",
                function_code=function_code,
                cand=cand,
                metadata={
                    "raw_task_id": task_id,
                    "entry_point": entry,
                    "source_benchmark": "HumanEval-X",
                    "derivation": "comment_free_single_statement_mask",
                    "target_line": cand.line_no,
                    "target_span": [cand.start, cand.end],
                    "function_line_count": count_nonempty_lines(function_code),
                    "candidate_count_before_per_task": len(candidates),
                    "filters": ["no_comments", "single_line", cand.target_kind],
                },
                judge_payload={
                    "kind": "derived_humaneval_x_go_test",
                    "task_id": task_id,
                    "entry_point": entry,
                    "test": test,
                    "test_setup": test_setup,
                    "example_test": row.get("example_test"),
                    "imports": imports,
                    "judge_prefix": leading_code + function_code[:cand.start],
                    "judge_suffix": function_code[cand.end:] + trailing_code,
                    "raw_humaneval_x": row,
                },
            )
            sample["task_type"] = "go_single_statement_completion_derived"
            samples.append(sample)
            stats.accept(cand.target_kind, candidates=len(candidates) if kept == 0 else 0)
            kept += 1
    return samples, stats


def build_multipl_e_go_bodymask(
    parquet_paths: list[Path],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], ExternalBuildStats]:
    """Build function-body mask samples for MultiPL-E Go.

    The public MultiPL-E HF parquet files contain prompt/tests/stop_tokens, but
    not Go canonical solutions.  Without a reference solution we cannot derive
    single-statement gold targets.  These rows are still useful for pass@k with
    a body-level [MASK], and metadata marks target_available=false.
    """
    stats = ExternalBuildStats("multipl_e_go_bodymask")
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()

    for path in parquet_paths:
        if not path.exists():
            stats.reject(f"missing_raw_file:{path}")
            continue
        for row in read_parquet_rows(path):
            if limit > 0 and stats.rows_seen >= limit:
                return samples, stats
            stats.rows_seen += 1
            name = str(row.get("name") or "")
            prompt = str(row.get("prompt") or "")
            tests = str(row.get("tests") or "")
            if not name or not prompt or not tests:
                stats.reject("missing_required_field")
                continue
            func_start = prompt.find("func ")
            if func_start < 0:
                stats.reject("function_signature_not_found")
                continue
            func_prefix = strip_leading_go_comments(prompt[func_start:])
            entry = extract_go_func_name(func_prefix)
            if not entry:
                stats.reject("entry_function_not_found")
                continue
            if "{" not in func_prefix:
                stats.reject("function_open_brace_not_found")
                continue
            raw_row = jsonable(row)
            stop_tokens = raw_row.get("stop_tokens") if isinstance(raw_row, dict) else []
            if not isinstance(stop_tokens, list):
                stop_tokens = []
            uid_seed = "\n".join([path.parent.name, name, entry])
            uid = f"multipl_e_{path.parent.name.replace('-', '_')}_{stable_hash(uid_seed)}"
            if uid in seen:
                stats.reject("duplicate_uid")
                continue
            seen.add(uid)
            source_dataset = f"multipl_e_{path.parent.name.replace('-', '_')}"
            sample = {
                "uid": uid,
                "source_dataset": source_dataset,
                "split": "test",
                "language": "go",
                "task_type": "go_function_body_completion_derived",
                "prefix": func_prefix,
                "target": None,
                "suffix": "\n}",
                "full_code": None,
                "target_kind": "function_body",
                "metadata": {
                    "raw_task_id": name,
                    "entry_point": entry,
                    "source_benchmark": "MultiPL-E",
                    "source_file": str(path),
                    "derivation": "body_mask_no_reference_solution",
                    "target_available": False,
                    "codebleu_supported": False,
                    "filters": ["go", "body_mask", "no_reference_solution"],
                },
                "judge_payload": {
                    "kind": "derived_multipl_e_go_test",
                    "task_id": name,
                    "entry_point": entry,
                    "test": tests,
                    "stop_tokens": stop_tokens,
                    "raw_multipl_e": raw_row,
                },
            }
            samples.append(sample)
            stats.accept("function_body")
    return samples, stats


def write_report(path: Path, sections: list[tuple[str, list[dict[str, Any]], ExternalBuildStats]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Go External Derived FIM Build Report", ""]
    for title, samples, stats in sections:
        lines.extend([
            f"## {title}",
            "",
            f"- rows_seen: {stats.rows_seen}",
            f"- rows_accepted: {stats.rows_accepted}",
            f"- candidates_seen: {stats.candidates_seen}",
            f"- final_samples: {len(samples)}",
            "",
            "### target_kind",
            "",
        ])
        for kind, count in stats.target_kinds.most_common():
            lines.append(f"- {kind}: {count}")
        lines.extend(["", "### reject_reasons", ""])
        for reason, count in stats.reject_reasons.most_common(30):
            lines.append(f"- {reason}: {count}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def external_stats_from_build_stats(name: str, stats: BuildStats, sample_count: int) -> ExternalBuildStats:
    out = ExternalBuildStats(name)
    out.rows_seen = stats.rows_seen
    out.rows_accepted = sample_count
    out.candidates_seen = stats.candidates_seen
    out.reject_reasons = Counter(stats.reject_reasons)
    out.target_kinds = Counter(stats.target_kinds)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Go-only derived FIM test data from MultiPL-E and HumanEval-X.")
    parser.add_argument("--multipl-e-root", type=Path, default=Path("data/raw_data/multipl_e"))
    parser.add_argument("--humaneval-x-root", type=Path, default=Path("data/raw_data/humaneval_x"))
    parser.add_argument("--mceval-root", type=Path, default=Path("data/raw_data/mceval"))
    parser.add_argument("--mceval-prebuilt-dir", type=Path, default=Path("data/go_singleline_fim_exp/eval_data"))
    parser.add_argument("--mceval-mode", choices=["prebuilt", "raw"], default="prebuilt")
    parser.add_argument("--output-root", type=Path, default=Path("data/go_singleline_fim_exp/test_data"))
    parser.add_argument("--report", type=Path, default=Path("outputs/go_singleline_fim_exp/reports/go_external_derived_fim_build_report.md"))
    parser.add_argument("--benchmarks", default="humaneval_x,mceval", help="Comma list: humaneval_x,mceval,multipl_e")
    parser.add_argument("--per-task", type=int, default=1, help="HumanEval-X statement masks per raw task.")
    parser.add_argument("--limit", type=int, default=0, help="Preview limit per benchmark; 0 means all rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {x.strip() for x in args.benchmarks.split(",") if x.strip()}
    sections: list[tuple[str, list[dict[str, Any]], ExternalBuildStats]] = []

    if "humaneval_x" in selected:
        raw_path = args.humaneval_x_root / "data" / "go" / "data" / "humaneval.jsonl"
        samples, stats = build_humaneval_x_go(raw_path, per_task=max(1, args.per_task), limit=args.limit)
        out_dir = args.output_root / "humaneval_x"
        write_jsonl(out_dir / "humaneval_x_go_derived_canonical.jsonl", samples)
        write_jsonl(out_dir / "humaneval_x_go_derived_chatml.jsonl", [render_mask_chatml(s) for s in samples])
        sections.append(("HumanEval-X Go single-statement derived FIM", samples, stats))
        print(f"[humaneval_x] samples={len(samples)} -> {out_dir}")

    if "mceval" in selected:
        out_dir = args.output_root / "mceval"
        if args.mceval_mode == "prebuilt":
            canonical_in = args.mceval_prebuilt_dir / "mceval_go_single_v2_canonical.jsonl"
            chatml_in = args.mceval_prebuilt_dir / "mceval_go_single_v2_chatml.jsonl"
            if not canonical_in.exists() or not chatml_in.exists():
                raise SystemExit(f"Missing prebuilt MCEval files under {args.mceval_prebuilt_dir}; use --mceval-mode raw to rebuild from raw data.")
            samples = list(iter_jsonl(canonical_in))
            chatml_rows = list(iter_jsonl(chatml_in))
            if args.limit > 0:
                samples = samples[: args.limit]
                chatml_rows = chatml_rows[: args.limit]
            for sample in samples:
                sample["split"] = "test"
            for sample in chatml_rows:
                sample["split"] = "test"
            write_jsonl(out_dir / "mceval_go_single_v2_canonical.jsonl", samples)
            write_jsonl(out_dir / "mceval_go_single_v2_chatml.jsonl", chatml_rows)
            ext_stats = ExternalBuildStats("mceval_go_prebuilt")
            ext_stats.rows_seen = len(samples)
            ext_stats.rows_accepted = len(samples)
            ext_stats.candidates_seen = len(samples)
            ext_stats.target_kinds = Counter(str(s.get("target_kind", "unknown")) for s in samples)
        else:
            stats = BuildStats()
            samples = build_mceval_candidates(args.mceval_root, per_task=max(1, args.per_task), stats=stats)
            if args.limit > 0:
                samples = samples[: args.limit]
            for sample in samples:
                sample["split"] = "test"
            write_jsonl(out_dir / "mceval_go_single_v2_canonical.jsonl", samples)
            write_jsonl(out_dir / "mceval_go_single_v2_chatml.jsonl", [render_chatml(s) for s in samples])
            ext_stats = external_stats_from_build_stats("mceval_go_raw", stats, len(samples))
        sections.append(("MCEval Go single-statement FIM", samples, ext_stats))
        print(f"[mceval] samples={len(samples)} -> {out_dir}")

    if "multipl_e" in selected:
        parquet_paths = [
            args.multipl_e_root / "humaneval-go" / "test-00000-of-00001.parquet",
            args.multipl_e_root / "mbpp-go" / "test-00000-of-00001.parquet",
        ]
        samples, stats = build_multipl_e_go_bodymask(parquet_paths, limit=args.limit)
        out_dir = args.output_root / "multipl_e"
        write_jsonl(out_dir / "multipl_e_go_bodymask_canonical.jsonl", samples)
        write_jsonl(out_dir / "multipl_e_go_bodymask_chatml.jsonl", [render_mask_chatml(s) for s in samples])
        sections.append(("MultiPL-E Go body-mask derived eval", samples, stats))
        print(f"[multipl_e] samples={len(samples)} -> {out_dir}")

    write_report(args.report, sections)
    print(f"[done] report={args.report}")


if __name__ == "__main__":
    main()
