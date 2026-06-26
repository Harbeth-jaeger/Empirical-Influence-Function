#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

SUBTYPE_COLORS: dict[str, str] = {
    "bracket": "#64748b",
    "defuse": "#0ea5e9",
    "call": "#f97316",
    "return": "#ef4444",
    "type": "#22c55e",
    "dataflow": "#fb923c",
    "semantic": "#a855f7",
    "api": "#ec4899",
    "unknown": "#94a3b8",
    "": "#94a3b8",
}

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
TARGET_LANGUAGES = ["python", "cpp", "c", "csharp", "go", "java"]


def decode_text(text: str) -> str:
    if "\n" in text or "\t" in text:
        return text
    return text.replace("\\n", "\n").replace("\\t", "\t")


def decode_token_for_display(token: str) -> str:
    return token.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


def compact_token_text(text: str) -> str:
    clean = text.replace("\n", "\\n").replace("\t", "\\t")
    if clean.strip():
        return clean.strip()
    if "\n" in text:
        return "\\n"
    if "\t" in text:
        return "\\t"
    if text:
        return "space"
    return ""


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                yield idx, json.loads(line)


def build_chatml_sequence(messages: list[dict[str, Any]]) -> tuple[str, dict[str, tuple[int, int]], dict[str, str]]:
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    contents: dict[str, str] = {}
    for msg in messages:
        role = str(msg.get("role", ""))
        content = decode_text(str(msg.get("content", "")))
        parts.append(f"<|im_start|>{role}\n")
        start = sum(len(x) for x in parts)
        parts.append(content)
        end = sum(len(x) for x in parts)
        parts.append("<|im_end|>\n")
        spans[role] = (start, end)
        contents[role] = content
    return "".join(parts), spans, contents


def comment_ranges_c_like(code: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
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
                i += 1
            elif ch == "/" and nxt == "/":
                start = i
                i += 2
                while i < n and code[i] != "\n":
                    i += 1
                ranges.append((start, i))
            elif ch == "/" and nxt == "*":
                start = i
                i += 2
                while i + 1 < n and not (code[i] == "*" and code[i + 1] == "/"):
                    i += 1
                i = min(n, i + 2)
                ranges.append((start, i))
            else:
                i += 1
        else:
            if ch == "\\" and i + 1 < n:
                i += 2
            elif ch == quote:
                state = "normal"
                i += 1
            else:
                i += 1
    return ranges


def comment_ranges_python(code: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
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
                start = i
                triple = tri
                i += 3
                while i < n and not code.startswith(triple, i):
                    i += 1
                i = min(n, i + 3)
                ranges.append((start, i))
            elif ch in {'\"', "'"}:
                quote = ch
                state = "string"
                i += 1
            elif ch == "#":
                start = i
                while i < n and code[i] != "\n":
                    i += 1
                ranges.append((start, i))
            else:
                i += 1
        else:
            if ch == "\\" and i + 1 < n:
                i += 2
            elif ch == quote:
                state = "normal"
                i += 1
            else:
                i += 1
    return ranges


def comment_ranges(code: str, language: str) -> list[tuple[int, int]]:
    return comment_ranges_python(code) if language.lower() == "python" else comment_ranges_c_like(code)


def fim_code_spans_in_user(user_text: str) -> list[tuple[int, int]]:
    p0 = user_text.find(FIM_PREFIX)
    p1 = user_text.find(FIM_SUFFIX, p0 + len(FIM_PREFIX)) if p0 >= 0 else -1
    p2 = user_text.find(FIM_MIDDLE, p1 + len(FIM_SUFFIX)) if p1 >= 0 else -1
    if p0 < 0 or p1 < 0 or p2 < 0:
        return []
    return [(p0 + len(FIM_PREFIX), p1), (p1 + len(FIM_SUFFIX), p2)]


def global_comment_ranges(sequence: str, spans: dict[str, tuple[int, int]], contents: dict[str, str], language: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    user_text = contents.get("user", "")
    if "user" in spans:
        user_start, _ = spans["user"]
        for start, end in fim_code_spans_in_user(user_text):
            code = user_text[start:end]
            for a, b in comment_ranges(code, language):
                out.append((user_start + start + a, user_start + start + b))
    if "assistant" in spans:
        assistant_start, assistant_end = spans["assistant"]
        code = contents.get("assistant", "")
        for a, b in comment_ranges(code, language):
            out.append((assistant_start + a, assistant_start + b))
    return [(a, b) for a, b in out if b > a]


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def tokenize_sequence(tokenizer: Any, sequence: str) -> tuple[list[int], list[tuple[int, int]], list[str]]:
    enc = tokenizer(sequence, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)
    display_texts = [decode_token_for_display(t) for t in token_strings]
    return input_ids, offsets, display_texts


def prompt_len_from_offsets(offsets: list[tuple[int, int]], assistant_start: int) -> int:
    for i, (_s, e) in enumerate(offsets):
        if e > assistant_start:
            return i
    return len(offsets)


def normalize_edge(edge: dict[str, Any]) -> tuple[int, int, str] | None:
    try:
        src = int(edge.get("src", edge.get("token_i_idx", -1)))
        dst = int(edge.get("dst", edge.get("token_j_idx", -1)))
    except (TypeError, ValueError):
        return None
    subtype = str(edge.get("subtype", edge.get("reason", "")) or "unknown")
    return src, dst, subtype


def edge_stats(edge_row: dict[str, Any], seq_len: int, prompt_len: int, hidden: set[int]) -> dict[str, Any]:
    total = p2c = c2c = p2p = 0
    p2c_dsts: set[int] = set()
    by_type: dict[str, int] = {}
    visible_edges: list[tuple[int, int, str]] = []
    for raw_edge in edge_row.get("attention_edges") or []:
        edge = normalize_edge(raw_edge)
        if edge is None:
            continue
        src, dst, subtype = edge
        if not (0 <= src < seq_len and 0 <= dst < seq_len and src < dst):
            continue
        if src in hidden or dst in hidden:
            continue
        total += 1
        by_type[subtype] = by_type.get(subtype, 0) + 1
        visible_edges.append(edge)
        if src < prompt_len <= dst:
            p2c += 1
            p2c_dsts.add(dst)
        elif prompt_len <= src < dst:
            c2c += 1
        elif src < prompt_len and dst < prompt_len:
            p2p += 1
    return {
        "total": total,
        "p2c": p2c,
        "p2c_dst_count": len(p2c_dsts),
        "c2c": c2c,
        "p2p": p2p,
        "type_count": len(by_type),
        "types": by_type,
        "visible_edges": visible_edges,
    }


def sample_score(stats: dict[str, Any], visible_tokens: int, prompt_len: int, completion_len: int) -> float:
    if visible_tokens < 350 or completion_len < 35:
        return -1e9
    density = stats["total"] / max(visible_tokens, 1)
    p2c_density = stats["p2c"] / max(completion_len, 1)
    length_penalty = abs(visible_tokens - 850) / 400.0
    return (
        5.0 * stats["p2c_dst_count"]
        + 1.5 * stats["p2c"]
        + 35.0 * density
        + 25.0 * p2c_density
        + 8.0 * stats["type_count"]
        + 0.15 * stats["c2c"]
        - length_penalty
    )


def build_viewer_sample(
    *,
    row_index: int,
    raw: dict[str, Any],
    edge_row: dict[str, Any],
    tokenizer: Any,
    sequence: str,
    offsets: list[tuple[int, int]],
    display_texts: list[str],
    prompt_len: int,
    hidden: set[int],
    stats: dict[str, Any],
    score: float,
) -> dict[str, Any]:
    old_to_new: dict[int, int] = {}
    tokens: list[dict[str, Any]] = []
    for old_idx, text in enumerate(display_texts):
        if old_idx in hidden:
            continue
        new_idx = len(tokens)
        old_to_new[old_idx] = new_idx
        tokens.append({
            "idx": new_idx,
            "original_idx": old_idx,
            "text": text,
            "display": compact_token_text(text),
            "is_completion": old_idx >= prompt_len,
        })

    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for src, dst, subtype in stats["visible_edges"]:
        if src not in old_to_new or dst not in old_to_new:
            continue
        ns, nd = old_to_new[src], old_to_new[dst]
        key = (ns, nd, subtype)
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "source": ns,
            "target": nd,
            "subtype": subtype or "unknown",
            "color": SUBTYPE_COLORS.get(subtype, SUBTYPE_COLORS["unknown"]),
        })

    new_prompt_len = sum(1 for i in range(prompt_len) if i not in hidden)
    return {
        "sample_index": row_index,
        "source_dataset": raw.get("source_dataset", ""),
        "language": raw.get("language", ""),
        "raw_id": raw.get("raw_id", ""),
        "uid": raw.get("uid", ""),
        "prompt_len": new_prompt_len,
        "tokens": tokens,
        "edges": edges,
        "attention_topk": {},
        "selection": {
            "score": round(score, 4),
            "visible_tokens": len(tokens),
            "original_tokens": len(display_texts),
            "hidden_comment_tokens": len(hidden),
            "completion_tokens": sum(1 for t in tokens if t["is_completion"]),
            "total_edges": len(edges),
            "p2c": stats["p2c"],
            "p2c_dst_count": stats["p2c_dst_count"],
            "c2c": stats["c2c"],
            "p2p": stats["p2p"],
            "type_count": stats["type_count"],
            "types": stats["types"],
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank", "idx", "language", "source_dataset", "uid", "score", "visible_tokens", "original_tokens",
        "hidden_comment_tokens", "completion_tokens", "total_edges", "p2c", "p2c_dst_count", "c2c", "p2p", "type_count", "types",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            sel = row["selection"]
            writer.writerow({
                "rank": rank,
                "idx": row["sample_index"],
                "language": row["language"],
                "source_dataset": row["source_dataset"],
                "uid": row.get("uid", ""),
                "score": sel["score"],
                "visible_tokens": sel["visible_tokens"],
                "original_tokens": sel["original_tokens"],
                "hidden_comment_tokens": sel["hidden_comment_tokens"],
                "completion_tokens": sel["completion_tokens"],
                "total_edges": sel["total_edges"],
                "p2c": sel["p2c"],
                "p2c_dst_count": sel["p2c_dst_count"],
                "c2c": sel["c2c"],
                "p2p": sel["p2p"],
                "type_count": sel["type_count"],
                "types": ";".join(f"{k}:{v}" for k, v in sorted(sel["types"].items(), key=lambda item: item[1], reverse=True)),
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select annotation-rich training samples and build a comment-hidden dynamic annotation viewer.")
    parser.add_argument("--raw_data_path", default="/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/sft_data/rendered_chatml_fim_train.jsonl")
    parser.add_argument("--edge_data_path", default="/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/sft_data/ours_graphsignal_train.json")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--languages", default=",".join(TARGET_LANGUAGES))
    parser.add_argument("--samples_per_language", type=int, default=3)
    parser.add_argument("--max_rows", type=int, default=0, help="0 means scan all rows available in edge_data_path.")
    parser.add_argument("--template_path", default="src/tools/viz_annotation/dynamic_annotation_viewer.html")
    parser.add_argument("--output_html", default="outputs/viz_annotation/visualization/annotation_showcase_dynamic_annotation_viewer.html")
    parser.add_argument("--output_json", default="outputs/viz_annotation/visualization/annotation_showcase_samples.json")
    parser.add_argument("--output_csv", default="outputs/viz_annotation/visualization/annotation_showcase_candidates.csv")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    languages = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=args.local_files_only)

    raw_iter = iter_jsonl(Path(args.raw_data_path))
    edge_iter = iter_jsonl(Path(args.edge_data_path))
    buckets: dict[str, list[tuple[float, dict[str, Any]]]] = {lang: [] for lang in languages}
    scanned = aligned = 0

    for (raw_idx, raw), (edge_idx, edge_row) in zip(raw_iter, edge_iter):
        if args.max_rows > 0 and scanned >= args.max_rows:
            break
        scanned += 1
        if raw_idx != edge_idx:
            raise RuntimeError(f"raw/edge row mismatch: raw={raw_idx}, edge={edge_idx}")
        language = str(raw.get("language", "")).lower()
        if language not in buckets:
            continue
        messages = raw.get("messages") or []
        if not messages:
            continue
        sequence, spans, contents = build_chatml_sequence(messages)
        input_ids, offsets, display_texts = tokenize_sequence(tokenizer, sequence)
        edge_ids = list(edge_row.get("input_ids") or edge_row.get("input_id") or [])
        seq_len = min(len(input_ids), len(edge_ids))
        if seq_len <= 0 or input_ids[:seq_len] != edge_ids[:seq_len]:
            continue
        aligned += 1
        input_ids = input_ids[:seq_len]
        offsets = offsets[:seq_len]
        display_texts = display_texts[:seq_len]
        assistant_start = spans.get("assistant", (len(sequence), len(sequence)))[0]
        prompt_len = min(prompt_len_from_offsets(offsets, assistant_start), seq_len)
        comment_spans = global_comment_ranges(sequence, spans, contents, language)
        hidden = {i for i, (s, e) in enumerate(offsets) if any(overlaps(s, e, a, b) for a, b in comment_spans)}
        stats = edge_stats(edge_row, seq_len, prompt_len, hidden)
        visible_tokens = seq_len - len(hidden)
        completion_len = max(0, visible_tokens - sum(1 for i in range(prompt_len) if i not in hidden))
        score = sample_score(stats, visible_tokens, prompt_len, completion_len)
        if score <= -1e8:
            continue
        sample = build_viewer_sample(
            row_index=raw_idx,
            raw=raw,
            edge_row=edge_row,
            tokenizer=tokenizer,
            sequence=sequence,
            offsets=offsets,
            display_texts=display_texts,
            prompt_len=prompt_len,
            hidden=hidden,
            stats=stats,
            score=score,
        )
        buckets[language].append((score, sample))

    selected: list[dict[str, Any]] = []
    for lang in languages:
        ranked = sorted(buckets.get(lang, []), key=lambda x: x[0], reverse=True)
        selected.extend(sample for _score, sample in ranked[: args.samples_per_language])

    selected = sorted(selected, key=lambda s: (languages.index(str(s.get("language", "")).lower()), -float(s["selection"]["score"])))
    viewer_data = {
        "config": {
            "raw_data_path": args.raw_data_path,
            "edge_data_path": args.edge_data_path,
            "model_path": args.model_path,
            "selection_policy": "Top samples per language by annotation richness: high edge density, many prompt-to-completion targets, edge-type diversity, moderate token length. Comment tokens are hidden and edges are reindexed for display only.",
            "scanned_rows": scanned,
            "aligned_rows": aligned,
            "samples_per_language": args.samples_per_language,
        },
        "subtype_colors": SUBTYPE_COLORS,
        "samples": selected,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(viewer_data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), selected)

    template = Path(args.template_path).read_text(encoding="utf-8")
    data_json = json.dumps(viewer_data, ensure_ascii=False)
    html = template.replace("__VIEWER_DATA__", data_json.replace("</script", "<\\/script"))
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")

    print(f"scanned={scanned} aligned={aligned} selected={len(selected)}")
    for s in selected:
        sel = s["selection"]
        print(
            f"{s['language']:7s} idx={s['sample_index']:4d} score={sel['score']:8.2f} "
            f"tok={sel['visible_tokens']:4d}/{sel['original_tokens']:4d} hidden={sel['hidden_comment_tokens']:3d} "
            f"edges={sel['total_edges']:4d} p2c={sel['p2c']:4d} dst={sel['p2c_dst_count']:3d} "
            f"types={sel['type_count']} uid={s.get('uid','')}"
        )
    print(f"wrote json -> {output_json}")
    print(f"wrote csv  -> {args.output_csv}")
    print(f"wrote html -> {output_html}")


if __name__ == "__main__":
    main()
