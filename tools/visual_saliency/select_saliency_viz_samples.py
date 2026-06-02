#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = ROOT / "data/benchmarks/sft_data/rendered_chatml_fim_train.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "outputs/visual_saliency/saliency_viz_samples.json"

TARGET_LANGUAGES = ["python", "cpp", "c", "csharp", "go", "java"]
KEYWORDS = {
    "if", "else", "elif", "return", "for", "while", "switch", "case", "try", "catch",
    "finally", "throw", "raise", "class", "struct", "def", "func", "function", "public",
    "private", "static", "new", "null", "None", "true", "false", "break", "continue",
    "template", "typename", "interface", "extends", "implements", "map", "range", "defer",
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                yield idx, json.loads(line)


def decode_escaped(text: str) -> str:
    if "\n" in text or "\t" in text:
        return text
    return text.replace("\\n", "\n").replace("\\t", "\t")


def split_fim_prompt(fim_prompt: str) -> tuple[str, str] | None:
    fp = "<|fim_prefix|>"
    fs = "<|fim_suffix|>"
    fm = "<|fim_middle|>"
    a = fim_prompt.find(fp)
    b = fim_prompt.find(fs, a + len(fp)) if a >= 0 else -1
    c = fim_prompt.find(fm, b + len(fs)) if b >= 0 else -1
    if a < 0 or b < 0 or c < 0:
        return None
    prefix = fim_prompt[a + len(fp):b]
    suffix = fim_prompt[b + len(fs):c]
    return prefix, suffix


def remove_c_like_comments(code: str) -> str:
    out: list[str] = []
    i = 0
    n = len(code)
    state = "normal"
    quote = ""
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if state == "normal":
            if ch in {'\"', "'"}:
                quote = ch
                state = "string"
                out.append(ch)
                i += 1
            elif ch == "/" and nxt == "/":
                i += 2
                while i < n and code[i] != "\n":
                    i += 1
                if i < n:
                    out.append("\n")
                    i += 1
            elif ch == "/" and nxt == "*":
                i += 2
                while i + 1 < n and not (code[i] == "*" and code[i + 1] == "/"):
                    if code[i] == "\n":
                        out.append("\n")
                    i += 1
                i += 2 if i + 1 < n else 0
            else:
                out.append(ch)
                i += 1
        elif state == "string":
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 2
            elif ch == quote:
                state = "normal"
                i += 1
            else:
                i += 1
    return "".join(out)


def remove_python_comments(code: str) -> str:
    out: list[str] = []
    i = 0
    n = len(code)
    state = "normal"
    quote = ""
    triple = ""
    while i < n:
        ch = code[i]
        tri = code[i:i + 3]
        if state == "normal":
            if tri == ('\"' * 3) or tri == ("'" * 3):
                triple = tri
                state = "triple"
                i += 3
            elif ch in {'\"', "'"}:
                quote = ch
                state = "string"
                out.append(ch)
                i += 1
            elif ch == "#":
                while i < n and code[i] != "\n":
                    i += 1
                if i < n:
                    out.append("\n")
                    i += 1
            else:
                out.append(ch)
                i += 1
        elif state == "string":
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 2
            elif ch == quote:
                state = "normal"
                i += 1
            else:
                i += 1
        elif state == "triple":
            if code.startswith(triple, i):
                i += 3
                state = "normal"
            else:
                if code[i] == "\n":
                    out.append("\n")
                i += 1
    return "".join(out)


def normalize_blank_lines(code: str) -> str:
    lines = [ln.rstrip() for ln in code.splitlines()]
    compact: list[str] = []
    blank = 0
    for ln in lines:
        if ln.strip():
            compact.append(ln)
            blank = 0
        else:
            blank += 1
            if blank <= 1:
                compact.append("")
    return "\n".join(compact).strip("\n") + ("\n" if code.endswith("\n") else "")


def strip_comments(code: str, language: str) -> str:
    lang = language.lower()
    if lang == "python":
        cleaned = remove_python_comments(code)
    else:
        cleaned = remove_c_like_comments(code)
    return normalize_blank_lines(cleaned)


def rebuild_row_with_clean_code(row: dict[str, Any], language: str) -> dict[str, Any] | None:
    fim_prompt = decode_escaped(str(row.get("fim_prompt", "")))
    parts = split_fim_prompt(fim_prompt)
    if parts is None:
        return None
    prefix, suffix = parts
    completion = decode_escaped(str(row.get("fim_completion", "")))
    clean_prefix = strip_comments(prefix, language)
    clean_suffix = strip_comments(suffix, language)
    clean_completion = strip_comments(completion, language)
    clean_fim_prompt = f"<|fim_prefix|>{clean_prefix}<|fim_suffix|>{clean_suffix}<|fim_middle|>"

    clean = dict(row)
    clean["fim_prompt"] = clean_fim_prompt
    clean["fim_completion"] = clean_completion
    clean["visual_saliency_comments_stripped"] = True

    messages = []
    replaced_fim = False
    for msg in row.get("messages", []):
        new_msg = dict(msg)
        content = decode_escaped(str(new_msg.get("content", "")))
        role = str(new_msg.get("role", ""))
        if role == "user":
            if fim_prompt in content:
                content = content.replace(fim_prompt, clean_fim_prompt, 1)
                replaced_fim = True
            elif "<|fim_prefix|>" in content and "<|fim_middle|>" in content:
                content = re.sub(r"<\|fim_prefix\|>.*<\|fim_suffix\|>.*<\|fim_middle\|>", clean_fim_prompt, content, count=1, flags=re.S)
                replaced_fim = True
            new_msg["content"] = content
        elif role == "assistant":
            new_msg["content"] = clean_completion
        messages.append(new_msg)
    if not replaced_fim:
        # Fall back to a minimal two-message ChatML payload if the rendered user message is unusual.
        messages = [
            {"role": "system", "content": "You are a precise code completion assistant. Complete exactly one missing code span. Return code only, no explanation."},
            {"role": "user", "content": "Complete the missing code span using FIM format.\n" + clean_fim_prompt},
            {"role": "assistant", "content": clean_completion},
        ]
    clean["messages"] = messages
    return clean


def tokenize_like_code(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*|==|!=|<=|>=|->|::|&&|\|\||\S", text)


def score_row(row: dict[str, Any], language: str, min_code_chars: int, max_code_chars: int) -> tuple[float, dict[str, Any]]:
    fim_prompt = decode_escaped(str(row.get("fim_prompt", "")))
    completion = decode_escaped(str(row.get("fim_completion", "")))
    parts = split_fim_prompt(fim_prompt)
    if parts is None:
        return -1e9, {"reason": "missing fim markers"}
    prefix, suffix = parts
    code_blob = prefix + "\n" + completion + "\n" + suffix
    code_chars = len(code_blob)
    lines = [ln for ln in code_blob.splitlines() if ln.strip()]
    tokens = tokenize_like_code(code_blob)
    keyword_hits = [tok for tok in tokens if tok in KEYWORDS]
    unique_keywords = sorted(set(keyword_hits))
    call_like = len(re.findall(r"\b[A-Za-z_]\w*\s*\(", code_blob))
    branches = len(re.findall(r"\b(if|else|elif|switch|case|catch)\b", code_blob))
    loops = len(re.findall(r"\b(for|while|range)\b", code_blob))
    assignments = len(re.findall(r"(?<![=!<>])=(?!=)", code_blob))
    if code_chars < min_code_chars or code_chars > max_code_chars:
        return -1e9, {"reason": "length_filter", "code_chars": code_chars}
    complexity = 0.010 * min(code_chars, 7000) + 1.8 * branches + 1.6 * loops + 0.6 * call_like + 0.4 * assignments + 1.2 * len(unique_keywords)
    # Prefer longer/complex examples, but avoid pathological full files.
    length_bonus = -abs(code_chars - 4200) / 1200.0
    score = complexity + length_bonus
    return score, {
        "code_chars_after_comment_strip": code_chars,
        "nonempty_lines_after_comment_strip": len(lines),
        "code_tokens_after_comment_strip": len(tokens),
        "branches": branches,
        "loops": loops,
        "call_like": call_like,
        "assignments": assignments,
        "unique_keywords": unique_keywords,
        "score": round(score, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select complex training-set samples for free-run ALTI saliency visualization.")
    parser.add_argument("--train_path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--languages", default=",".join(TARGET_LANGUAGES))
    parser.add_argument("--samples_per_language", type=int, default=2)
    parser.add_argument("--min_code_chars", type=int, default=1200)
    parser.add_argument("--max_code_chars", type=int, default=7000)
    parser.add_argument("--source_datasets", default="", help="Optional comma-separated dataset filter.")
    parser.add_argument("--include_row", action="store_true", default=True)
    args = parser.parse_args()

    train_path = Path(args.train_path)
    output_path = Path(args.output_path)
    languages = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    source_datasets = {x.strip().lower() for x in args.source_datasets.split(",") if x.strip()}

    buckets: dict[str, list[tuple[float, int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row_index, row in iter_jsonl(train_path):
        lang = str(row.get("language", "")).lower()
        ds = str(row.get("source_dataset", "")).lower()
        if lang not in languages:
            continue
        if source_datasets and ds not in source_datasets:
            continue
        clean = rebuild_row_with_clean_code(row, lang)
        if clean is None:
            continue
        score, info = score_row(clean, lang, args.min_code_chars, args.max_code_chars)
        if score <= -1e8:
            continue
        buckets[lang].append((score, row_index, clean, info))

    selected: list[dict[str, Any]] = []
    for lang in languages:
        candidates = sorted(buckets.get(lang, []), key=lambda x: (x[0], -x[1]), reverse=True)
        picked = candidates[:args.samples_per_language]
        for rank, (score, row_index, row, info) in enumerate(picked):
            entry = {
                "sample_id": f"{lang}_{rank:02d}_{row.get('source_dataset', 'unknown')}_{row_index}",
                "row_index": row_index,
                "uid": row.get("uid", ""),
                "source_dataset": row.get("source_dataset", "unknown"),
                "language": lang,
                "raw_id": row.get("raw_id", ""),
                "selection": info,
            }
            if args.include_row:
                entry["row"] = row
            selected.append(entry)

    payload = {
        "version": 2,
        "input_split": "train",
        "train_path": str(train_path),
        "selection_policy": {
            "languages": languages,
            "samples_per_language": args.samples_per_language,
            "min_code_chars": args.min_code_chars,
            "max_code_chars": args.max_code_chars,
            "comment_policy": "Strip comments from FIM prefix/suffix/completion before model input.",
            "score": "longer and structurally complex code: branches, loops, calls, assignments, keywords",
        },
        "samples": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected {len(selected)} samples -> {output_path}")
    for s in selected:
        sel = s["selection"]
        print(
            f"{s['language']:7s} {s['source_dataset']:9s} row={s['row_index']:5d} "
            f"chars={sel['code_chars_after_comment_strip']:5d} lines={sel['nonempty_lines_after_comment_strip']:4d} "
            f"score={sel['score']:7.2f} uid={s['uid']}"
        )


if __name__ == "__main__":
    main()
