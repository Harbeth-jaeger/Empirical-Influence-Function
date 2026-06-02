#!/usr/bin/env python3
"""Build canonical FIM data and render it to ChatML-FIM for Qwen-Instruct SFT/eval."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a precise code completion assistant. Complete exactly one missing code span. "
    "Return code only, no explanation."
)


@dataclass
class UnifiedSample:
    uid: str
    source_dataset: str
    split: str
    language: str
    task_type: str
    prefix: str
    suffix: str
    target: str
    raw_id: str
    metadata: dict[str, Any]
    judge_payload: dict[str, Any]


FENCED_CODE_BLOCK_RE = re.compile(r"`{3}[^\n`]*\n(.*?)`{3}", re.DOTALL)
TARGET_LANGUAGES = {"c", "cpp", "csharp", "go", "java", "python"}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_lang(s: str | None) -> str:
    if not s:
        return "unknown"
    s = s.strip().lower()
    mapping = {
        "c#": "csharp",
        "cs": "csharp",
        "c++": "cpp",
    }
    return mapping.get(s, s)


def split_by_placeholder(text: str, placeholders: list[str]) -> tuple[str, str, str] | None:
    for marker in placeholders:
        if marker in text:
            parts = text.split(marker)
            if len(parts) == 2:
                return parts[0], parts[1], marker
    return None


def render_fim(prefix: str, suffix: str) -> str:
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"


def render_chatml_user(language: str, prefix: str, suffix: str) -> str:
    return (
        f"Complete the missing {language} code span between prefix and suffix.\\n"
        "Return only the missing code span.\\n\\n"
        "[PREFIX]\\n"
        f"{prefix}\\n"
        "[/PREFIX]\\n\\n"
        "[SUFFIX]\\n"
        f"{suffix}\\n"
        "[/SUFFIX]"
    )


def render_chatml_fim_user(language: str, prefix: str, suffix: str) -> str:
    return (
        f"Complete the missing {language} code span using FIM format.\\n"
        "Return only the missing span.\\n\\n"
        f"{render_fim(prefix, suffix)}"
    )


def extract_fenced_code_blocks(text: Any) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    return FENCED_CODE_BLOCK_RE.findall(text)


def strip_outer_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def synthetic_fim_from_code(code: str) -> tuple[str, str, str, dict[str, Any]] | None:
    """Create one deterministic single-hole FIM sample from a complete code block.

    McEval-Instruct train rows are instruction-to-complete-code examples, not native
    infilling examples. For benchmark SFT we turn each complete code answer into a
    function-level/code-span completion example by masking a contiguous middle span.
    """
    lines = strip_outer_blank_lines(code.splitlines(keepends=True))
    nonempty_indexes = [i for i, line in enumerate(lines) if line.strip()]
    if len(nonempty_indexes) < 6:
        return None

    # Keep prefix and suffix non-empty. The central window avoids trivial imports,
    # package headers, and trailing test-only lines more often than edge masking.
    window = max(2, min(24, len(nonempty_indexes) // 4))
    center = len(nonempty_indexes) // 2
    start_pos = max(1, center - window // 2)
    end_pos = min(len(nonempty_indexes) - 1, start_pos + window)
    if end_pos <= start_pos:
        return None

    start_line = nonempty_indexes[start_pos]
    end_line = nonempty_indexes[end_pos - 1] + 1

    prefix = "".join(lines[:start_line])
    target = "".join(lines[start_line:end_line])
    suffix = "".join(lines[end_line:])
    if not prefix.strip() or not target.strip() or not suffix.strip():
        return None

    metadata = {
        "synthetic_fim": True,
        "fim_construction": "single_middle_nonempty_line_window",
        "total_lines": len(lines),
        "nonempty_lines": len(nonempty_indexes),
        "target_start_line": start_line + 1,
        "target_end_line": end_line,
        "target_nonempty_lines": sum(1 for line in lines[start_line:end_line] if line.strip()),
    }
    return prefix, suffix, target, metadata


def validate_sample(sample: UnifiedSample) -> list[str]:
    issues: list[str] = []
    if not sample.prefix:
        issues.append("empty_prefix")
    if not sample.target.strip():
        issues.append("empty_target")
    if sample.task_type != "single_hole_completion":
        issues.append("invalid_task_type")
    return issues


def adapt_legacy_sft(path: Path) -> tuple[list[UnifiedSample], list[dict[str, Any]]]:
    out: list[UnifiedSample] = []
    rejects: list[dict[str, Any]] = []
    for line_no, obj in iter_jsonl(path):
        text = str(obj.get("input", ""))
        parsed = split_by_placeholder(text, ["<mask>", "[MASK]", "<MASK>", "{{completion}}"])
        if parsed is None:
            rejects.append({"dataset": "sft", "line": line_no, "reason": "missing_single_placeholder"})
            continue
        prefix, suffix, marker = parsed
        raw_id = str(obj.get("id", f"line_{line_no}"))
        sample = UnifiedSample(
            uid=f"sft:{raw_id}",
            source_dataset="legacy_sft",
            split="train",
            language=normalize_lang(str(obj.get("language", "unknown"))),
            task_type="single_hole_completion",
            prefix=prefix,
            suffix=suffix,
            target=str(obj.get("output", "")),
            raw_id=raw_id,
            metadata={
                "instruction": obj.get("instruction", ""),
                "mask_count": obj.get("mask_count"),
                "mask_type": obj.get("mask_type"),
                "granularity": obj.get("granularity"),
                "hole_marker": marker,
            },
            judge_payload={},
        )
        out.append(sample)
    return out, rejects


def adapt_mceval_instruct(path: Path) -> tuple[list[UnifiedSample], list[dict[str, Any]]]:
    out: list[UnifiedSample] = []
    rejects: list[dict[str, Any]] = []
    for line_no, obj in iter_jsonl(path):
        lang = normalize_lang(str(obj.get("language", "unknown")))
        if lang not in TARGET_LANGUAGES:
            rejects.append(
                {
                    "dataset": "mceval_instruct",
                    "line": line_no,
                    "reason": "unsupported_language",
                    "language": lang,
                }
            )
            continue

        code_blocks = extract_fenced_code_blocks(obj.get("output", ""))
        if len(code_blocks) != 1:
            rejects.append(
                {
                    "dataset": "mceval_instruct",
                    "line": line_no,
                    "reason": "not_one_fenced_code_block",
                    "num_code_blocks": len(code_blocks),
                }
            )
            continue

        synthetic = synthetic_fim_from_code(code_blocks[0])
        if synthetic is None:
            rejects.append(
                {
                    "dataset": "mceval_instruct",
                    "line": line_no,
                    "reason": "failed_synthetic_fim_split",
                }
            )
            continue
        prefix, suffix, target, fim_metadata = synthetic

        raw_id = str(obj.get("id") or obj.get("task_id") or f"line_{line_no}")
        sample = UnifiedSample(
            uid=f"mceval_instruct:{raw_id}",
            source_dataset="mceval_instruct",
            split="train",
            language=lang,
            task_type="single_hole_completion",
            prefix=prefix,
            suffix=suffix,
            target=target,
            raw_id=raw_id,
            metadata={
                "instruction": obj.get("instruction", ""),
                "source": obj.get("source", ""),
                "original_language": obj.get("language", ""),
                **fim_metadata,
            },
            judge_payload={},
        )
        out.append(sample)
    return out, rejects


def adapt_humaneval(path: Path) -> tuple[list[UnifiedSample], list[dict[str, Any]]]:
    out: list[UnifiedSample] = []
    rejects: list[dict[str, Any]] = []
    for line_no, obj in iter_jsonl(path):
        raw_id = str(obj.get("task_id", f"line_{line_no}"))
        prefix = str(obj.get("prompt", ""))
        suffix = str(obj.get("suffix", ""))
        target = str(obj.get("canonical_solution", ""))
        sample = UnifiedSample(
            uid=f"humaneval:{raw_id}",
            source_dataset="humaneval",
            split="eval",
            language="python",
            task_type="single_hole_completion",
            prefix=prefix,
            suffix=suffix,
            target=target,
            raw_id=raw_id,
            metadata={"entry_point": obj.get("entry_point")},
            judge_payload={"test": obj.get("test", "")},
        )
        out.append(sample)
    return out, rejects


def adapt_mceval(path: Path) -> tuple[list[UnifiedSample], list[dict[str, Any]]]:
    out: list[UnifiedSample] = []
    rejects: list[dict[str, Any]] = []
    for line_no, obj in iter_jsonl(path):
        raw_id = str(obj.get("task_id", f"line_{line_no}"))
        lang = normalize_lang(raw_id.split("/")[0] if "/" in raw_id else obj.get("language", "unknown"))

        prefix = str(obj.get("prefix_code", ""))
        suffix = str(obj.get("suffix_code", ""))
        target = str(obj.get("masked_spans", ""))

        if not (prefix and target):
            parsed = split_by_placeholder(str(obj.get("mask_code", "")), ["[MASK]", "<mask>", "<MASK>"])
            if parsed is None:
                rejects.append({"dataset": "mceval", "line": line_no, "reason": "missing_prefix_target"})
                continue
            prefix, suffix, _ = parsed
            target = str(obj.get("canonical_solution", ""))

        sample = UnifiedSample(
            uid=f"mceval:{raw_id}",
            source_dataset="mceval",
            split="eval",
            language=lang,
            task_type="single_hole_completion",
            prefix=prefix,
            suffix=suffix,
            target=target,
            raw_id=raw_id,
            metadata={
                "entry_point": obj.get("entry_point"),
                "signature": obj.get("signature"),
                "docstring": obj.get("docstring"),
            },
            judge_payload={"test": obj.get("test", "")},
        )
        out.append(sample)
    return out, rejects


def adapt_safim(path: Path) -> tuple[list[UnifiedSample], list[dict[str, Any]]]:
    out: list[UnifiedSample] = []
    rejects: list[dict[str, Any]] = []
    for line_no, obj in iter_jsonl(path):
        raw_id = str(obj.get("task_id", f"line_{line_no}"))
        eval_prompt = str(obj.get("eval_prompt", ""))
        parsed = split_by_placeholder(eval_prompt, ["{{completion}}", "<mask>", "[MASK]"])
        if parsed is None:
            rejects.append({"dataset": "safim", "line": line_no, "reason": "missing_eval_placeholder"})
            continue
        prefix, suffix, marker = parsed
        sample = UnifiedSample(
            uid=f"safim:{raw_id}",
            source_dataset="safim",
            split="eval",
            language=normalize_lang(str(obj.get("lang", "unknown"))),
            task_type="single_hole_completion",
            prefix=prefix,
            suffix=suffix,
            target=str(obj.get("ground_truth", "")),
            raw_id=raw_id,
            metadata={
                "annotations": obj.get("annotations", []),
                "tokens": obj.get("tokens", []),
                "hole_marker": marker,
            },
            judge_payload={"unit_tests": obj.get("unit_tests", "")},
        )
        out.append(sample)
    return out, rejects


def to_unified_row(sample: UnifiedSample) -> dict[str, Any]:
    return asdict(sample)


def to_chatml_fim_row(sample: UnifiedSample) -> dict[str, Any]:
    user_text = render_chatml_fim_user(sample.language, sample.prefix, sample.suffix)
    return {
        "uid": sample.uid,
        "source_dataset": sample.source_dataset,
        "split": sample.split,
        "language": sample.language,
        "task_type": sample.task_type,
        "raw_id": sample.raw_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": sample.target},
        ],
        "fim_prompt": render_fim(sample.prefix, sample.suffix),
        "fim_completion": sample.target,
        "metadata": sample.metadata,
        "judge_payload": sample.judge_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified benchmark schema + ChatML+FIM rendered views.")
    parser.add_argument(
        "--train",
        default="data/benchmarks/sft_data/mceval_instruct/mceval_instruct_train_filtered.jsonl",
        help="Filtered McEval-Instruct train JSONL.",
    )
    parser.add_argument(
        "--legacy-sft",
        default="",
        help="Optional legacy placeholder-style SFT JSONL to include as train data.",
    )
    parser.add_argument("--humaneval", default="data/benchmarks/eval_data/humaneval.jsonl")
    parser.add_argument("--mceval", default="data/benchmarks/eval_data/mceval.jsonl")
    parser.add_argument("--safim", default="data/benchmarks/eval_data/safim.jsonl")
    parser.add_argument("--canonical-dir", default="data/benchmarks")
    parser.add_argument("--rendered-train-dir", default="data/benchmarks/sft_data")
    parser.add_argument("--rendered-eval-dir", default="data/benchmarks/eval_data")
    parser.add_argument("--report-dir", default="data/benchmarks/unified_chatml_fim")
    args = parser.parse_args()

    canonical_dir = Path(args.canonical_dir)
    rendered_train_dir = Path(args.rendered_train_dir)
    rendered_eval_dir = Path(args.rendered_eval_dir)
    report_dir = Path(args.report_dir)
    for path in (canonical_dir, rendered_train_dir, rendered_eval_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    all_samples: list[UnifiedSample] = []
    rejects: list[dict[str, Any]] = []

    adapters: list[tuple[str, Any, Path]] = [
        ("mceval_instruct", adapt_mceval_instruct, Path(args.train)),
        ("humaneval", adapt_humaneval, Path(args.humaneval)),
        ("mceval", adapt_mceval, Path(args.mceval)),
        ("safim", adapt_safim, Path(args.safim)),
    ]
    if args.legacy_sft:
        adapters.insert(1, ("legacy_sft", adapt_legacy_sft, Path(args.legacy_sft)))

    for adapter in adapters:
        name, fn, path = adapter
        samples, rejected = fn(path)
        all_samples.extend(samples)
        rejects.extend(rejected)
        print(f"[{name}] accepted={len(samples)} rejected={len(rejected)}")

    valid_samples: list[UnifiedSample] = []
    validation_issues: list[dict[str, Any]] = []
    for sample in all_samples:
        issues = validate_sample(sample)
        if issues:
            validation_issues.append(
                {
                    "uid": sample.uid,
                    "source_dataset": sample.source_dataset,
                    "split": sample.split,
                    "language": sample.language,
                    "raw_id": sample.raw_id,
                    "issues": issues,
                }
            )
            continue
        valid_samples.append(sample)

    unified_rows = [to_unified_row(s) for s in valid_samples]
    rendered_rows = [to_chatml_fim_row(s) for s in valid_samples]

    train_rows = [r for r in rendered_rows if r["split"] == "train"]
    eval_rows = [r for r in rendered_rows if r["split"] == "eval"]
    unified_train_rows = [r for r in unified_rows if r["split"] == "train"]
    unified_eval_rows = [r for r in unified_rows if r["split"] == "eval"]

    write_jsonl(canonical_dir / "canonical_fim_all.jsonl", unified_rows)
    write_jsonl(canonical_dir / "canonical_fim_train.jsonl", unified_train_rows)
    write_jsonl(canonical_dir / "canonical_fim_eval.jsonl", unified_eval_rows)
    write_jsonl(rendered_train_dir / "rendered_chatml_fim_train.jsonl", train_rows)
    write_jsonl(rendered_eval_dir / "rendered_chatml_fim_eval.jsonl", eval_rows)

    # Keep the previous report directory as a compatibility/debug view.
    write_jsonl(report_dir / "unified_all.jsonl", unified_rows)
    write_jsonl(report_dir / "rendered_chatml_fim_all.jsonl", rendered_rows)
    write_jsonl(report_dir / "rendered_chatml_fim_train.jsonl", train_rows)
    write_jsonl(report_dir / "rendered_chatml_fim_eval.jsonl", eval_rows)
    write_jsonl(report_dir / "rejected_rows.jsonl", rejects)
    write_jsonl(report_dir / "validation_issues.jsonl", validation_issues)

    summary = {
        "canonical_dir": str(canonical_dir),
        "rendered_train_dir": str(rendered_train_dir),
        "rendered_eval_dir": str(rendered_eval_dir),
        "report_dir": str(report_dir),
        "num_unified_rows": len(unified_rows),
        "num_rendered_rows": len(rendered_rows),
        "num_train_rows": len(train_rows),
        "num_eval_rows": len(eval_rows),
        "num_rejected_rows": len(rejects),
        "num_validation_issues": len(validation_issues),
        "num_dropped_validation_rows": len(validation_issues),
        "sources": dict(Counter(s.source_dataset for s in valid_samples)),
        "splits": dict(Counter(s.split for s in valid_samples)),
        "languages": dict(Counter(s.language for s in valid_samples)),
        "rejected_reasons": dict(Counter(r["reason"] for r in rejects if "reason" in r)),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write tiny schema note for quick reference.
    schema_note = {
        "unified_fields": [
            "uid",
            "source_dataset",
            "split",
            "language",
            "task_type",
            "prefix",
            "suffix",
            "target",
            "raw_id",
            "metadata",
            "judge_payload",
        ],
        "rendered_fields": [
            "uid",
            "source_dataset",
            "split",
            "language",
            "messages",
            "fim_prompt",
            "fim_completion",
            "metadata",
            "judge_payload",
        ],
    }
    (report_dir / "schema_note.json").write_text(json.dumps(schema_note, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
