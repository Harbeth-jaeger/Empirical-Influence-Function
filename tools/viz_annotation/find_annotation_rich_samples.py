#!/usr/bin/env python
"""Find training samples with rich tokenizer-aligned annotation edges.

This script is intended for debugging GraphSignal annotation quality.  It
reconstructs the same ChatML sequence used by SFT/attention visualization,
checks whether the edge dataset is token-aligned, and ranks samples by edge
counts such as prompt-to-completion edges.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from transformers import AutoTokenizer


def decode_for_training_chatml(text: str) -> str:
    return text.replace("\\n", "\n").replace("\\t", "\t")


def build_chatml_text(row: dict[str, Any]) -> tuple[str, str, str]:
    messages = row.get("messages") or []
    prompt_parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = decode_for_training_chatml(str(msg.get("content", "")))
        if role in {"system", "user"}:
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    prompt = "".join(prompt_parts) + "<|im_start|>assistant\n"

    assistant_content = ""
    for msg in messages:
        if str(msg.get("role", "")).strip() == "assistant":
            assistant_content = decode_for_training_chatml(str(msg.get("content", "")))
            break
    if not assistant_content:
        assistant_content = decode_for_training_chatml(str(row.get("fim_completion", row.get("target", ""))))

    completion = assistant_content + "<|im_end|>\n"
    return prompt, completion, prompt + completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_data_path",
        default="data/benchmarks/sft_data/rendered_chatml_fim_train.jsonl",
        help="Original benchmark SFT JSONL with messages.",
    )
    parser.add_argument(
        "--edge_data_path",
        default="data/benchmarks/sft_data/ours_graphsignal_train.json",
        help="Binarized SFT JSONL with tokenizer-aligned attention_edges.",
    )
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Tokenizer path used for SFT.",
    )
    parser.add_argument("--top_n", type=int, default=40)
    parser.add_argument(
        "--sort_by",
        choices=["p2c", "p2c_dst_count", "total", "c2c"],
        default="p2c",
        help="Ranking metric.",
    )
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    rows: list[dict[str, Any]] = []
    with open(args.raw_data_path, encoding="utf-8") as raw_f, open(args.edge_data_path, encoding="utf-8") as edge_f:
        for idx, (raw_line, edge_line) in enumerate(zip(raw_f, edge_f)):
            raw = json.loads(raw_line)
            edge_row = json.loads(edge_line)

            prompt, _completion, full = build_chatml_text(raw)
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            full_ids = tokenizer(full, add_special_tokens=False).input_ids
            prompt_len = len(prompt_ids)
            seq_len = len(full_ids)

            edge_ids = edge_row.get("input_ids") or edge_row.get("input_id") or []
            aligned = edge_ids[:seq_len] == full_ids

            total = len(edge_row.get("attention_edges") or [])
            p2c = c2c = p2p = 0
            p2c_dsts: set[int] = set()
            by_type: dict[str, int] = {}

            if aligned:
                for edge in edge_row.get("attention_edges") or []:
                    try:
                        src = int(edge.get("src", edge.get("token_i_idx", -1)))
                        dst = int(edge.get("dst", edge.get("token_j_idx", -1)))
                    except (TypeError, ValueError):
                        continue
                    if not (0 <= src < seq_len and 0 <= dst < seq_len):
                        continue

                    subtype = edge.get("subtype", "") or "unknown"
                    by_type[subtype] = by_type.get(subtype, 0) + 1

                    if src < prompt_len <= dst and src < dst:
                        p2c += 1
                        p2c_dsts.add(dst)
                    elif prompt_len <= src < dst:
                        c2c += 1
                    elif src < prompt_len and dst < prompt_len:
                        p2p += 1

            rows.append(
                {
                    "idx": idx,
                    "source_dataset": raw.get("source_dataset", ""),
                    "language": raw.get("language", ""),
                    "raw_id": raw.get("raw_id", ""),
                    "aligned": aligned,
                    "seq_len": seq_len,
                    "prompt_len": prompt_len,
                    "completion_len": max(0, seq_len - prompt_len),
                    "total": total,
                    "p2c": p2c,
                    "p2c_dst_count": len(p2c_dsts),
                    "c2c": c2c,
                    "p2p": p2p,
                    "types": ";".join(f"{k}:{v}" for k, v in sorted(by_type.items(), key=lambda item: item[1], reverse=True)),
                }
            )

    print(f"samples: {len(rows)}")
    print(f"aligned: {sum(r['aligned'] for r in rows)} / {len(rows)}")
    if not rows:
        raise SystemExit(
            "No aligned rows to rank. Check that --edge_data_path exists, is non-empty, "
            "and has the same sample order as --raw_data_path."
        )
    for key in ["total", "p2c", "p2c_dst_count", "c2c", "p2p"]:
        vals = [int(r[key]) for r in rows]
        print(
            f"{key}: nonzero={sum(v > 0 for v in vals)}/{len(vals)} "
            f"mean={mean(vals):.2f} median={median(vals):.1f} max={max(vals)}"
        )

    rows_sorted = sorted(rows, key=lambda row: (row[args.sort_by], row["p2c_dst_count"], row["total"]), reverse=True)
    print(f"\nTop {args.top_n} by {args.sort_by}:")
    for row in rows_sorted[: args.top_n]:
        print(
            f"idx={row['idx']:4d} {row['source_dataset']}/{row['language']} "
            f"aligned={row['aligned']} total={row['total']:4d} p2c={row['p2c']:4d} "
            f"p2c_dst={row['p2c_dst_count']:3d} c2c={row['c2c']:4d} p2p={row['p2p']:4d} "
            f"prompt={row['prompt_len']:4d} comp={row['completion_len']:4d} "
            f"types={row['types']}"
        )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows_sorted)
        print(f"\nWrote CSV -> {output_path}")


if __name__ == "__main__":
    main()
