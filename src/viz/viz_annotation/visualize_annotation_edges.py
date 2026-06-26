#!/usr/bin/env python
"""Render tokenizer-aligned annotation edges as standalone HTML files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.annotate.viz_utils import visualize_correlations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edge_data_path",
        default="/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/sft_data/ours_graphsignal_train.json",
        help="JSONL with input_ids and attention_edges.",
    )
    parser.add_argument(
        "--sample_indices",
        nargs="+",
        type=int,
        required=True,
        help="One or more 0-based sample indices to visualize.",
    )
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Tokenizer path used to decode input_ids.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/viz_annotation/visualization/annotation_edges",
    )
    parser.add_argument("--open_browser", action="store_true")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    return parser.parse_args()


def load_selected_rows(path: Path, sample_indices: set[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in sample_indices:
                rows[idx] = json.loads(line)
            if len(rows) == len(sample_indices):
                break
    missing = sorted(sample_indices - rows.keys())
    if missing:
        raise ValueError(f"sample indices not found in {path}: {missing}")
    return rows


def decode_token_for_display(token: str) -> str:
    return token.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


def build_code_and_subwords(token_texts: list[str]) -> tuple[str, list[SimpleNamespace]]:
    parts: list[str] = []
    subwords: list[SimpleNamespace] = []
    cursor = 0
    for idx, token in enumerate(token_texts):
        text = decode_token_for_display(token)
        start = cursor
        end = start + len(text)
        parts.append(text)
        subwords.append(
            SimpleNamespace(
                surface=text,
                clean=text.strip() or text,
                token_id=idx,
                char_start=start,
                char_end=end,
            )
        )
        cursor = end
    return "".join(parts), subwords


def build_correlations(row: dict[str, Any], token_texts: list[str]) -> list[SimpleNamespace]:
    correlations: list[SimpleNamespace] = []
    seq_len = len(token_texts)
    for edge in list(row.get("attention_edges") or []) + list(row.get("prompt_attention_edges") or []):
        try:
            src = int(edge.get("src", edge.get("token_i_idx", -1)))
            dst = int(edge.get("dst", edge.get("token_j_idx", -1)))
        except (TypeError, ValueError):
            continue
        if not (0 <= src < seq_len and 0 <= dst < seq_len):
            continue
        correlations.append(
            SimpleNamespace(
                token_i=token_texts[src],
                token_j=token_texts[dst],
                source=edge.get("source", "GraphSignal"),
                subtype=edge.get("subtype", ""),
                token_i_idx=src,
                token_j_idx=dst,
            )
        )
    return correlations


def main() -> None:
    args = parse_args()
    edge_path = Path(args.edge_data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    rows = load_selected_rows(edge_path, set(args.sample_indices))
    for sample_idx in sorted(rows):
        row = rows[sample_idx]
        input_ids = row.get("input_ids") or row.get("input_id")
        if not input_ids:
            raise ValueError(f"sample {sample_idx} has no input_ids")

        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        code, subwords = build_code_and_subwords(token_texts)
        correlations = build_correlations(row, token_texts)

        output_path = output_dir / f"annotate_sample{sample_idx}_ours_graphsignal.html"
        visualize_correlations(
            correlations=correlations,
            title=f"Ours GraphSignal annotation edges · sample {sample_idx}",
            code=code,
            subwords=subwords,
            output_path=str(output_path),
            open_browser=args.open_browser,
        )
        print(
            f"sample={sample_idx} tokens={len(input_ids)} "
            f"edges={len(correlations)} output={output_path}"
        )


if __name__ == "__main__":
    main()
