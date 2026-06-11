#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
warnings.filterwarnings("ignore", category=SyntaxWarning)


SIMPLE_NODE_TYPES = (
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Return,
    ast.Expr,
)

COMPOUND_NODE_TYPES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.Try,
    ast.With,
    ast.AsyncWith,
)


@dataclass(frozen=True)
class Candidate:
    task_type: str
    start_line: int
    end_line: int
    node_type: str
    score: tuple[int, int, int]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_docstring_text(text: str, max_blank_run: int = 1) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip():
            cleaned.append(line.strip() if not cleaned else line)
            blank_run = 0
        else:
            blank_run += 1
            if cleaned and blank_run <= max_blank_run:
                cleaned.append("")
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned).strip()


def truncate_docstring(text: str, max_chars: int) -> str:
    text = clean_docstring_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    paragraph_cut = cut.rfind("\n\n")
    sentence_cut = max(cut.rfind(". "), cut.rfind("。"), cut.rfind("! "), cut.rfind("? "))
    if paragraph_cut >= max_chars * 0.55:
        cut = cut[:paragraph_cut].rstrip()
    elif sentence_cut >= max_chars * 0.55:
        cut = cut[: sentence_cut + 1].rstrip()
    return cut + "\n..."


def make_docstring(prompt: str, tests: Any, source_dataset: str, mode: str, max_chars: int, max_tests: int = 3) -> str:
    if mode in {"none", "preserve"}:
        return ""
    prompt = clean_docstring_text(prompt)
    if mode == "compact":
        prompt = truncate_docstring(prompt, max_chars)
    parts = [prompt]
    if source_dataset == "mbpp" and mode != "none":
        test_lines = []
        for item in to_plain_jsonable(tests) or []:
            text = clean_docstring_text(str(item))
            if text:
                test_lines.append(text)
        if test_lines:
            examples = "Examples:\n" + "\n".join(test_lines[:max_tests])
            parts.append(examples)
    doc = "\n\n".join(part for part in parts if part)
    if mode == "compact":
        doc = truncate_docstring(doc, max_chars)
    return doc


def to_plain_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return [to_plain_jsonable(x) for x in value]
    if isinstance(value, list):
        return [to_plain_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): to_plain_jsonable(v) for k, v in value.items()}
    return value


def load_mbpp_rows(raw_dir: Path, subset: str, splits: list[str]) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading MBPP parquet requires pandas in the active environment.") from exc

    rows: list[dict[str, Any]] = []
    for split in splits:
        path = raw_dir / subset / f"{split}-00000-of-00001.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing MBPP parquet: {path}")
        df = pd.read_parquet(path)
        for obj in df.to_dict(orient="records"):
            tests = to_plain_jsonable(obj.get("test_list")) or []
            rows.append(
                {
                    "source_dataset": "mbpp",
                    "source_split": split,
                    "raw_task_id": str(obj.get("task_id", "")),
                    "prompt": clean_docstring_text(str(obj.get("text", "") or "")),
                    "code": str(obj.get("code", "") or ""),
                    "tests": tests,
                    "raw_meta": {
                        "mbpp_subset": subset,
                        "test_setup_code": to_plain_jsonable(obj.get("test_setup_code")) or "",
                        "challenge_test_list": to_plain_jsonable(obj.get("challenge_test_list")) or [],
                    },
                }
            )
    return rows


def load_apps_rows(raw_path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, obj in enumerate(read_jsonl(raw_path)):
        if max_rows is not None and idx >= max_rows:
            break
        try:
            solutions = json.loads(obj.get("solutions") or "[]")
        except json.JSONDecodeError:
            solutions = []
        try:
            input_output = json.loads(obj.get("input_output") or "{}", parse_int=str)
        except json.JSONDecodeError:
            input_output = {}
        rows.append(
            {
                "source_dataset": "apps",
                "source_split": "train",
                "raw_task_id": str(obj.get("problem_id", obj.get("id", ""))),
                "prompt": clean_docstring_text(str(obj.get("question", "") or "")),
                "solutions": [str(x) for x in solutions if isinstance(x, str)],
                "tests": input_output,
                "entry_point": input_output.get("fn_name"),
                "raw_meta": {
                    "difficulty": obj.get("difficulty"),
                    "url": obj.get("url"),
                    "starter_code": obj.get("starter_code", ""),
                },
            }
        )
    return rows


def load_selfcodealign_rows(raw_dir: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Reading SelfCodeAlign parquet requires pandas in the active environment.") from exc

    parquet_files = sorted(path for path in raw_dir.rglob("*.parquet") if ".cache" not in path.parts)
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found under {raw_dir}")

    rows: list[dict[str, Any]] = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        for obj in df.to_dict(orient="records"):
            if max_rows is not None and len(rows) >= max_rows:
                return rows
            content = str(obj.get("content", "") or "")
            if not content.strip():
                continue
            raw_id = obj.get("id") or obj.get("sha1") or f"{path.name}:{len(rows)}"
            rows.append(
                {
                    "source_dataset": "selfcodealign_sc2",
                    "source_split": path.stem,
                    "raw_task_id": str(raw_id),
                    "code": content,
                    "tests": None,
                    "entry_point": None,
                    "raw_meta": {
                        "dataset": "bigcode/python-stack-v1-functions-filtered-sc2",
                        "local_file": str(path),
                        "sha1": obj.get("sha1"),
                    },
                }
            )
    return rows


def infer_entry_point_from_tests(tests: Any) -> str | None:
    if not isinstance(tests, list):
        return None
    names: list[str] = []
    for item in tests:
        text = str(item)
        match = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", text)
        if match:
            names.append(match.group(1))
    if not names:
        return None
    return Counter(names).most_common(1)[0][0]


def iter_function_defs(tree: ast.AST) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def find_function(tree: ast.AST, entry_point: str | None) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    funcs = list(iter_function_defs(tree))
    if not funcs:
        return None
    if entry_point:
        for fn in funcs:
            if fn.name == entry_point:
                return fn
    return funcs[0]


def add_or_replace_docstring(tree: ast.Module, entry_point: str | None, docstring: str) -> tuple[ast.Module, str | None]:
    fn = find_function(tree, entry_point)
    if fn is None:
        return tree, None
    clean_doc = clean_docstring_text(docstring)
    has_doc = bool(
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    )
    if not clean_doc:
        if has_doc:
            fn.body.pop(0)
    else:
        doc_node = ast.Expr(value=ast.Constant(value=clean_doc))
        if has_doc:
            fn.body[0] = doc_node
        else:
            fn.body.insert(0, doc_node)
    ast.fix_missing_locations(tree)
    return tree, fn.name


def normalize_module(
    code: str,
    prompt: str,
    entry_point: str | None,
    *,
    docstring_mode: str,
    max_docstring_chars: int,
    tests: Any = None,
    source_dataset: str = "",
) -> tuple[str, str | None] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if docstring_mode == "preserve":
        resolved = find_function(tree, entry_point)
        resolved_entry = resolved.name if resolved is not None else None
    else:
        docstring = make_docstring(prompt, tests, source_dataset, docstring_mode, max_docstring_chars)
        tree, resolved_entry = add_or_replace_docstring(tree, entry_point, docstring)
    if resolved_entry is None:
        return None
    try:
        normalized = ast.unparse(tree).strip() + "\n"
        ast.parse(normalized)
    except Exception:
        return None
    return normalized, resolved_entry


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def slice_by_lines(text: str, start_line: int, end_line: int) -> tuple[str, str, str]:
    offsets = line_offsets(text)
    start = offsets[start_line - 1]
    end = offsets[end_line] if end_line < len(offsets) else len(text)
    return text[:start], text[start:end], text[end:]


def is_call_expr(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)


def is_single_candidate(node: ast.AST) -> bool:
    if not isinstance(node, SIMPLE_NODE_TYPES):
        return False
    if isinstance(node, ast.Expr) and not is_call_expr(node):
        return False
    return hasattr(node, "lineno") and hasattr(node, "end_lineno") and node.lineno == node.end_lineno


def is_trivial_target(target: str) -> bool:
    stripped = target.strip()
    if not stripped:
        return True
    if stripped in {"pass", "return None", "return True", "return False"}:
        return True
    if re.fullmatch(r"return\s+[-+]?\d+(\.\d+)?", stripped):
        return True
    return False


def function_docstring_line_count(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    if not fn.body:
        return 0
    first = fn.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
        if isinstance(first.value.value, str):
            return max(0, first.end_lineno - first.lineno + 1)
    return 0


def collect_candidates(full_code: str, entry_point: str) -> list[Candidate]:
    tree = ast.parse(full_code)
    fn = find_function(tree, entry_point)
    if fn is None:
        return []
    doc_end = fn.body[0].end_lineno if function_docstring_line_count(fn) else fn.lineno
    candidates: list[Candidate] = []

    for node in ast.walk(fn):
        if node is fn or not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        if node.lineno <= doc_end:
            continue
        prefix, target, suffix = slice_by_lines(full_code, node.lineno, node.end_lineno)
        if prefix + target + suffix != full_code or is_trivial_target(target):
            continue
        line_count = node.end_lineno - node.lineno + 1
        target_len = len(target.strip())
        if is_single_candidate(node) and 8 <= target_len <= 220:
            priority = 3 if isinstance(node, (ast.Assign, ast.AugAssign, ast.Return)) else 2
            candidates.append(
                Candidate("single_line", node.lineno, node.end_lineno, type(node).__name__, (priority, target_len, -node.lineno))
            )
        if isinstance(node, COMPOUND_NODE_TYPES) and 2 <= line_count <= 12 and target_len <= 1200:
            priority = 4 if isinstance(node, (ast.For, ast.While, ast.If)) else 2
            candidates.append(
                Candidate("multi_line", node.lineno, node.end_lineno, type(node).__name__, (priority, line_count, target_len))
            )

    # Fallback: contiguous top-level function-body statements.
    body = [stmt for stmt in fn.body if getattr(stmt, "lineno", 0) > doc_end]
    for width in range(2, min(8, len(body)) + 1):
        for start_idx in range(0, len(body) - width + 1):
            group = body[start_idx : start_idx + width]
            start_line = group[0].lineno
            end_line = group[-1].end_lineno
            prefix, target, suffix = slice_by_lines(full_code, start_line, end_line)
            line_count = end_line - start_line + 1
            if prefix + target + suffix != full_code or is_trivial_target(target):
                continue
            if 2 <= line_count <= 12 and len(target.strip()) <= 1200:
                candidates.append(
                    Candidate("multi_line", start_line, end_line, "statement_group", (1, line_count, len(target.strip())))
                )
        if candidates:
            break

    return candidates


def choose_candidates(candidates: list[Candidate], task_type: str, limit: int, rng: random.Random) -> list[Candidate]:
    filtered = [c for c in candidates if c.task_type == task_type]
    if not filtered or limit <= 0:
        return []
    filtered.sort(key=lambda c: c.score, reverse=True)
    top = filtered[: min(max(limit * 3, limit), len(filtered))]
    rng.shuffle(top)
    selected = top[:limit]
    selected.sort(key=lambda c: (c.start_line, c.end_line, c.node_type))
    return selected


def render_chatml_messages(language: str, prefix: str, suffix: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": f"You are a {language} code infilling assistant."},
        {
            "role": "user",
            "content": (
                f"Fill the [MASK] in the {language} code. Return only the missing {language} code, "
                "without Markdown fences or explanation.\n\n"
                f"* Incomplete Code:\n{prefix}[MASK]{suffix}"
            ),
        },
    ]


def make_sample(
    *,
    source_row: dict[str, Any],
    full_code: str,
    entry_point: str,
    candidate: Candidate,
    source_index: int,
) -> dict[str, Any]:
    prefix, target, suffix = slice_by_lines(full_code, candidate.start_line, candidate.end_line)
    if prefix + target + suffix != full_code:
        raise ValueError("internal slicing error")
    source_dataset = source_row["source_dataset"]
    raw_task_id = source_row["raw_task_id"]
    uid_base = f"humaneval:python:{candidate.task_type}:{source_dataset}:{raw_task_id}:{candidate.start_line}:{candidate.end_line}:{source_index}"
    uid_base += f":{stable_hash(full_code)[:12]}"
    uid = stable_hash(uid_base)[:16]
    return {
        "uid": f"humaneval_train_python_{uid}",
        "benchmark": "humaneval",
        "source": "derived_train",
        "source_dataset": source_dataset,
        "source_split": source_row.get("source_split"),
        "language": "python",
        "task_type": candidate.task_type,
        "entry_point": entry_point,
        "prefix": prefix,
        "target": target,
        "suffix": suffix,
        "full_code": full_code,
        "target_kind": candidate.node_type,
        "metadata": {
            "raw_task_id": raw_task_id,
            "source_dataset": source_dataset,
            "source_split": source_row.get("source_split"),
            "target_line_start": candidate.start_line,
            "target_line_end": candidate.end_line,
            "tests": source_row.get("tests"),
            "raw_meta": source_row.get("raw_meta", {}),
        },
    }


def load_decontam_hashes(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    hashes: set[str] = set()
    for row in read_jsonl(path):
        if row.get("task_type") not in {"single_line", "multi_line"}:
            continue
        full_code = str(row.get("prefix", "")) + str(row.get("target", "")) + str(row.get("suffix", ""))
        if full_code.strip():
            hashes.add(stable_hash(normalize_text(full_code)))
    return hashes


def build_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    report: dict[str, Any] = {"skipped": Counter(), "candidate_counts": Counter()}
    source_rows: list[dict[str, Any]] = []
    if args.source == "selfcodealign":
        source_rows.extend(load_selfcodealign_rows(args.selfcodealign_raw_dir, args.max_source_rows))
    elif args.source == "mbpp_apps":
        source_rows.extend(load_mbpp_rows(args.mbpp_raw_dir, args.mbpp_subset, args.mbpp_splits))
        source_rows.extend(load_apps_rows(args.apps_raw_path, args.max_apps_rows))
    else:
        raise ValueError(f"unsupported source: {args.source}")

    samples_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_code_hashes: set[str] = set()
    decontam_hashes = load_decontam_hashes(args.decontaminate_test_path)

    for source_index, row in enumerate(source_rows):
        codes: list[str]
        entry_point = row.get("entry_point") or infer_entry_point_from_tests(row.get("tests"))
        if row["source_dataset"] == "apps":
            codes = row.get("solutions") or []
        else:
            codes = [row.get("code", "")]

        for code in codes[: args.max_solutions_per_problem]:
            normalized = normalize_module(
                code,
                row.get("prompt", ""),
                entry_point,
                docstring_mode=args.docstring_mode,
                max_docstring_chars=args.max_docstring_chars,
                tests=row.get("tests"),
                source_dataset=row.get("source_dataset", ""),
            )
            if normalized is None:
                report["skipped"][f"{row['source_dataset']}:parse_or_no_function"] += 1
                continue
            full_code, resolved_entry = normalized
            if args.max_full_chars and len(full_code) > args.max_full_chars:
                report["skipped"][f"{row['source_dataset']}:too_long_full_code"] += 1
                continue
            code_hash = stable_hash(normalize_text(full_code))
            if code_hash in decontam_hashes:
                report["skipped"][f"{row['source_dataset']}:decontam_exact_full_code"] += 1
                continue
            if code_hash in seen_code_hashes:
                report["skipped"][f"{row['source_dataset']}:duplicate_code"] += 1
                continue
            if args.max_prompt_chars:
                # Check the shortest possible prompt before candidate-specific masking.
                if len(full_code) > args.max_prompt_chars + 1500:
                    report["skipped"][f"{row['source_dataset']}:too_long_prompt_context"] += 1
                    continue
            candidates = collect_candidates(full_code, resolved_entry)
            if not candidates:
                report["skipped"][f"{row['source_dataset']}:no_mask_candidate"] += 1
                continue
            made_for_code = 0
            for task_type in ("single_line", "multi_line"):
                selected_candidates = choose_candidates(candidates, task_type, args.max_masks_per_code_per_type, rng)
                for candidate in selected_candidates:
                    sample = make_sample(
                        source_row=row,
                        full_code=full_code,
                        entry_point=resolved_entry,
                        candidate=candidate,
                        source_index=source_index,
                    )
                    if args.max_prompt_chars and len(sample["prefix"] + "[MASK]" + sample["suffix"]) > args.max_prompt_chars:
                        report["skipped"][f"{row['source_dataset']}:too_long_prompt"] += 1
                        continue
                    samples_by_type[task_type].append(sample)
                    made_for_code += 1
                    report["candidate_counts"][f"{row['source_dataset']}:{task_type}"] += 1
            if made_for_code:
                seen_code_hashes.add(code_hash)

    single = samples_by_type["single_line"]
    multi = samples_by_type["multi_line"]
    rng.shuffle(single)
    rng.shuffle(multi)

    if args.max_samples:
        total = min(args.max_samples, len(single) + len(multi))
    else:
        total = len(single) + len(multi)

    target_single = round(total * args.single_ratio)
    target_multi = total - target_single
    if target_single > len(single):
        target_single = len(single)
        target_multi = min(len(multi), round(target_single * (1 - args.single_ratio) / args.single_ratio))
    if target_multi > len(multi):
        target_multi = len(multi)
        target_single = min(len(single), round(target_multi * args.single_ratio / (1 - args.single_ratio)))

    selected = single[:target_single] + multi[:target_multi]
    rng.shuffle(selected)
    for idx, sample in enumerate(selected):
        sample["sample_index"] = idx

    report["source_rows"] = len(source_rows)
    report["available_by_type"] = {"single_line": len(single), "multi_line": len(multi)}
    report["selected_by_type"] = dict(Counter(row["task_type"] for row in selected))
    report["selected_total"] = len(selected)
    report["skipped"] = dict(report["skipped"])
    report["candidate_counts"] = dict(report["candidate_counts"])
    return selected, report


def make_chatml_rows(canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in canonical_rows:
        messages = render_chatml_messages("python", row["prefix"], row["suffix"])
        rows.append(
            {
                "uid": row["uid"],
                "benchmark": row["benchmark"],
                "source": row["source"],
                "source_dataset": row["source_dataset"],
                "language": row["language"],
                "task_type": row["task_type"],
                "entry_point": row["entry_point"],
                "prefix": row["prefix"],
                "target": row["target"],
                "suffix": row["suffix"],
                "messages": messages,
                "only_last_turn_loss": True,
                "metadata": row["metadata"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build humaneval-style Python FIM train data from documented Python functions.")
    parser.add_argument("--source", choices=("selfcodealign", "mbpp_apps"), default="selfcodealign")
    parser.add_argument("--selfcodealign-raw-dir", type=Path, default=Path("data/raw_data/selfcodealign_python_functions"))
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--mbpp-raw-dir", type=Path, default=Path("data/raw_data/mbpp"))
    parser.add_argument("--mbpp-subset", choices=("full", "sanitized"), default="full")
    parser.add_argument("--mbpp-splits", nargs="+", default=["train", "validation", "prompt"])
    parser.add_argument("--apps-raw-path", type=Path, default=Path("data/raw_data/apps/train.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/benchmark/train_data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-ratio", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--max-apps-rows", type=int, default=None)
    parser.add_argument("--max-solutions-per-problem", type=int, default=5)
    parser.add_argument("--max-masks-per-code-per-type", type=int, default=3)
    parser.add_argument("--docstring-mode", choices=("preserve", "compact", "full", "none"), default="preserve")
    parser.add_argument("--max-docstring-chars", type=int, default=1200)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--max-full-chars", type=int, default=8000)
    parser.add_argument("--decontaminate-test-path", type=Path, default=Path("data/benchmark/test_data/humaneval_infilling_python.jsonl"))
    args = parser.parse_args()

    if not 0.0 < args.single_ratio < 1.0:
        raise ValueError("--single-ratio must be between 0 and 1")

    canonical_rows, report = build_candidates(args)
    chatml_rows = make_chatml_rows(canonical_rows)

    canonical_path = args.out_dir / "humaneval_python_canonical.jsonl"
    chatml_path = args.out_dir / "humaneval_python_chatml.jsonl"
    report_path = args.out_dir / "humaneval_python_build_report.json"

    n_canonical = write_jsonl(canonical_path, canonical_rows)
    n_chatml = write_jsonl(chatml_path, chatml_rows)
    report.update(
        {
            "outputs": {
                "canonical": str(canonical_path),
                "chatml": str(chatml_path),
                "report": str(report_path),
            },
            "written": {"canonical": n_canonical, "chatml": n_chatml},
            "config": {
                "source": args.source,
                "selfcodealign_raw_dir": str(args.selfcodealign_raw_dir),
                "max_source_rows": args.max_source_rows,
                "mbpp_raw_dir": str(args.mbpp_raw_dir),
                "mbpp_subset": args.mbpp_subset,
                "mbpp_splits": args.mbpp_splits,
                "apps_raw_path": str(args.apps_raw_path),
                "single_ratio": args.single_ratio,
                "multi_ratio": 1 - args.single_ratio,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "max_apps_rows": args.max_apps_rows,
                "max_solutions_per_problem": args.max_solutions_per_problem,
                "max_masks_per_code_per_type": args.max_masks_per_code_per_type,
                "docstring_mode": args.docstring_mode,
                "max_docstring_chars": args.max_docstring_chars,
                "max_prompt_chars": args.max_prompt_chars,
                "max_full_chars": args.max_full_chars,
                "decontaminate_test_path": str(args.decontaminate_test_path) if args.decontaminate_test_path else None,
            },
        }
    )
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
