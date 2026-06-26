#!/usr/bin/env python3
"""Teacher-forcing Base-vs-SFT saliency aligned with annotation edges.

This is for implementation/debug validation, not free-run evaluation.  Each
sample is fed as the fixed tokenized prompt + ground-truth completion once, and
saliency is computed for annotated completion target rows using the same
contribution definition used by src/train/loss.py.

Only causal annotation edges with source < target are eligible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IGNORE_INDEX = -100
DEFAULT_DATA = ROOT / "data/subsets/ours_graphsignal_train_first64.json"
DEFAULT_BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_SFT = ROOT / "outputs/benchmark/sft_qwen_ours_graphsignal_500_newloss_full_small64"
DEFAULT_OUTPUT = ROOT / "outputs/viz_saliency/base_vs_sft_teacher_forcing_annotation_saliency_data.json"
DEFAULT_TOP_K = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA))
    parser.add_argument("--base_model", default=str(DEFAULT_BASE))
    parser.add_argument("--sft_model", default=str(DEFAULT_SFT))
    parser.add_argument("--base_name", default="Base Qwen")
    parser.add_argument("--sft_name", default="Ours GraphSignal")
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--row_indices", default="", help="Comma-separated raw JSONL row indices to keep, e.g. 479.")
    parser.add_argument("--max_targets_per_sample", type=int, default=24)
    parser.add_argument("--min_sources_per_target", type=int, default=2)
    parser.add_argument(
        "--target_scope",
        choices=["completion", "prompt", "all"],
        default="completion",
        help="Which annotated query tokens to include. Training uses all; the viewer originally used completion.",
    )
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--local_files_only", type=int, default=1)
    parser.add_argument("--source_chunk_size", type=int, default=32)
    parser.add_argument("--language", default="python")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_completion_idx(labels: list[int]) -> int | None:
    return next((i for i, value in enumerate(labels) if int(value) != IGNORE_INDEX), None)


def token_display(text: str) -> str:
    if text == "\n":
        return "\\n"
    if text == "\t":
        return "\\t"
    return text.replace("\n", "\\n").replace("\t", "\\t")


def decode_tokens(tokenizer: Any, input_ids: list[int], completion_start: int) -> list[dict[str, Any]]:
    tokens = []
    for idx, tid in enumerate(input_ids):
        text = tokenizer.decode([int(tid)], skip_special_tokens=False)
        is_special = text.startswith("<|") and text.endswith("|>")
        tokens.append({
            "idx": idx,
            "id": int(tid),
            "text": text,
            "display": token_display(text),
            "region": "completion" if idx >= completion_start else "prompt_code",
            "is_special": bool(is_special),
            "is_whitespace": bool(text.strip() == ""),
            "step": int(idx - completion_start) if idx >= completion_start else None,
        })
    return tokens


def normalize_edge(edge: dict[str, Any]) -> tuple[int, int, str]:
    src = int(edge.get("src", edge.get("source", edge.get("token_i_idx", -1))))
    dst = int(edge.get("dst", edge.get("target", edge.get("token_j_idx", -1))))
    subtype = str(edge.get("subtype", edge.get("type", "")) or "")
    return src, dst, subtype


def causal_edges_by_target(
    row: dict[str, Any],
    seq_len: int,
    completion_start: int,
    target_scope: str,
) -> dict[int, list[dict[str, Any]]]:
    edges_by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in row.get("attention_edges", []):
        src, dst, subtype = normalize_edge(edge)
        # Hard requirement for this diagnostic: only source < query is valid.
        if not (0 <= src < dst < seq_len):
            continue
        if target_scope == "completion" and dst < completion_start:
            continue
        if target_scope == "prompt" and dst >= completion_start:
            continue
        edges_by_q[dst].append({"src": src, "dst": dst, "subtype": subtype})
    return edges_by_q


def select_rows(rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = []
    keep_rows = {int(x) for x in str(args.row_indices).split(",") if x.strip()}
    for row_index, row in enumerate(rows):
        if keep_rows and row_index not in keep_rows:
            continue
        input_ids = [int(x) for x in row.get("input_ids", [])]
        labels = [int(x) for x in row.get("label", row.get("labels", []))]
        if not input_ids or len(input_ids) != len(labels) or len(input_ids) > args.max_len:
            continue
        completion_start = first_completion_idx(labels)
        if completion_start is None:
            continue
        tokens = decode_tokens(tokenizer, input_ids, completion_start)
        edges_by_q = causal_edges_by_target(row, len(input_ids), completion_start, args.target_scope)
        eligible_targets = []
        for q, edges in edges_by_q.items():
            tgt = tokens[q]
            if tgt["is_special"] or tgt["is_whitespace"]:
                continue
            srcs = sorted({e["src"] for e in edges})
            if len(srcs) < args.min_sources_per_target:
                continue
            eligible_targets.append({"q": q, "num_edges": len(edges), "num_sources": len(srcs)})
        if not eligible_targets:
            continue
        eligible_targets.sort(key=lambda x: (x["num_sources"], x["num_edges"], x["q"]), reverse=True)
        score = sum(t["num_sources"] for t in eligible_targets[: args.max_targets_per_sample])
        candidates.append({
            "row_index": row_index,
            "row": row,
            "input_ids": input_ids,
            "labels": labels,
            "completion_start": completion_start,
            "tokens": tokens,
            "edges_by_q": edges_by_q,
            "eligible_targets": eligible_targets,
            "score": score,
            "num_edges": sum(len(v) for v in edges_by_q.values()),
        })
    candidates.sort(key=lambda x: (x["score"], len(x["eligible_targets"]), x["num_edges"]), reverse=True)
    return candidates[: args.max_samples]


def setup_tokenizer(path: str, local_files_only: bool):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        padding_side="right",
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(path_or_repo: str, device: str, dtype: torch.dtype, local_files_only: bool):
    from transformers import AutoModelForCausalLM
    p = Path(path_or_repo)
    if p.exists() and (p / "adapter_config.json").exists():
        from peft import PeftConfig, PeftModel
        peft_config = PeftConfig.from_pretrained(path_or_repo, local_files_only=local_files_only)
        base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model = PeftModel.from_pretrained(base, path_or_repo, is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path_or_repo,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    model.config.use_cache = False
    model.to(device)
    model.eval()
    return model


def make_sources(row_scores: torch.Tensor, tokens: list[dict[str, Any]], q: int, top_k: int) -> list[dict[str, Any]]:
    scores = row_scores[:q].detach().float().cpu()
    order = torch.argsort(scores, descending=True).tolist()[:top_k]
    out = []
    for rank, idx in enumerate(order, start=1):
        tok = tokens[idx]
        out.append({
            "rank": rank,
            "idx": int(idx),
            "token": tok["text"],
            "display": tok["display"],
            "region": tok["region"],
            "value": float(scores[idx]),
        })
    return out


def ranking_metrics(sources: list[dict[str, Any]], annotation_sources: set[int], top_k: int) -> dict[str, Any]:
    k_eff = min(max(1, int(top_k)), len(sources))
    ranked = [int(item["idx"]) for item in sources[:k_eff]]
    hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(ranked, start=1):
        if idx in annotation_sources:
            hits += 1
            precision_sum += hits / rank
    denom = max(1, len(annotation_sources))
    return {
        "top_k": int(top_k),
        "num_annotation_sources": int(len(annotation_sources)),
        "num_hits": int(hits),
        "recall_at_k": float(hits / denom),
        "precision_at_k": float(hits / max(1, k_eff)),
        "map_at_k": float(precision_sum / denom),
    }


def compact_model_result(
    *,
    name: str,
    tokens: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    c_rows: torch.Tensor,
    top_k: int,
    completion_start: int,
    edges_by_q: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    targets = {}
    targets_by_step = {}
    generated_token_indices = []
    for row_id, target in enumerate(target_rows):
        q = int(target["q"])
        tok = tokens[q]
        step = int(q - completion_start)
        targets_by_step[str(step)] = q
        generated_token_indices.append(q)
        sources = make_sources(c_rows[row_id], tokens, q, top_k)
        annotation_sources = {int(edge["src"]) for edge in edges_by_q.get(q, []) if int(edge["src"]) < q}
        targets[str(q)] = {
            "target_idx": q,
            "query_idx": q,
            "step": step,
            "token": tok["text"],
            "display": tok["display"],
            "sources": sources,
            "metrics": ranking_metrics(sources, annotation_sources, top_k),
            "saliency_definition": "Teacher-forcing last-layer contribution rows from src/train/loss.py::build_contribution_rows; only causal sources s<q are ranked.",
        }
    return {
        "name": name,
        "tokens": tokens,
        "generated_token_indices": generated_token_indices,
        "targets_by_step": targets_by_step,
        "targets": targets,
        "generated_text": "".join(tokens[q]["text"] for q in generated_token_indices),
        "skipped": False,
        "reason": "",
    }


def annotation_payload(
    tokens: list[dict[str, Any]],
    edges_by_q: dict[int, list[dict[str, Any]]],
    target_rows: list[dict[str, Any]],
    target_scope: str,
) -> dict[str, Any]:
    target_set = {int(t["q"]) for t in target_rows}
    edges_by_dst: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for q in sorted(target_set):
        rows = []
        for edge in edges_by_q.get(q, []):
            src = int(edge["src"])
            dst = int(edge["dst"])
            if not src < dst:
                continue
            rows.append({
                "src": src,
                "dst": dst,
                "subtype": edge.get("subtype", ""),
                "src_display": tokens[src]["display"],
                "src_text": tokens[src]["text"],
                "dst_display": tokens[dst]["display"],
            })
        if rows:
            edges_by_dst[str(q)] = rows
            total += len(rows)
    return {
        "num_edges": total,
        "edges_by_dst": edges_by_dst,
        "selection": {
            "causal_only": True,
            "target_scope": target_scope,
            "condition": f"source < target and target_scope={target_scope}",
        },
        "note": "Teacher-forcing annotation edges filtered with strict causal condition s < q.",
    }


def compute_model_rows(model: Any, input_ids: list[int], target_rows: list[dict[str, Any]], args: argparse.Namespace) -> torch.Tensor:
    from src.train.loss import build_contribution_rows

    device = next(model.parameters()).device
    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_t, device=device)
    row_batch = torch.zeros(len(target_rows), dtype=torch.long, device=device)
    row_qry = torch.tensor([int(t["q"]) for t in target_rows], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_t,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        c_rows = build_contribution_rows(
            model,
            outputs.hidden_states[-2],
            outputs.attentions[-1],
            row_batch,
            row_qry,
            source_chunk_size=args.source_chunk_size,
        )
    return c_rows.detach().cpu()


def main() -> None:
    args = parse_args()
    local_files_only = bool(args.local_files_only)
    dtype = dtype_from_name(args.dtype)

    tokenizer = setup_tokenizer(args.base_model, local_files_only)
    rows = load_rows(Path(args.data_path))
    selected = select_rows(rows, tokenizer, args)
    if not selected:
        raise RuntimeError("No eligible samples found. Try lowering --min_sources_per_target or increasing --max_len.")
    print(f"selected {len(selected)} samples from {args.data_path}")
    for s in selected:
        print(
            f"  row {s['row_index']} len={len(s['input_ids'])} "
            f"targets={len(s['eligible_targets'])} causal_edges={s['num_edges']}"
        )

    print(f"loading base: {args.base_model}")
    base_model = load_model(args.base_model, args.device, dtype, local_files_only)
    base_rows_by_sample = []
    for i, sample in enumerate(selected):
        target_rows = sample["eligible_targets"][: args.max_targets_per_sample]
        print(f"[base] sample {i+1}/{len(selected)} row {sample['row_index']} targets={len(target_rows)}", flush=True)
        base_rows_by_sample.append(compute_model_rows(base_model, sample["input_ids"], target_rows, args))
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"loading sft: {args.sft_model}")
    sft_model = load_model(args.sft_model, args.device, dtype, local_files_only)
    sft_rows_by_sample = []
    for i, sample in enumerate(selected):
        target_rows = sample["eligible_targets"][: args.max_targets_per_sample]
        print(f"[sft] sample {i+1}/{len(selected)} row {sample['row_index']} targets={len(target_rows)}", flush=True)
        sft_rows_by_sample.append(compute_model_rows(sft_model, sample["input_ids"], target_rows, args))
    del sft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_samples = []
    for sample, base_c, sft_c in zip(selected, base_rows_by_sample, sft_rows_by_sample):
        target_rows = sample["eligible_targets"][: args.max_targets_per_sample]
        row_index = int(sample["row_index"])
        sample_id = f"tf_train_{row_index}"
        out_samples.append({
            "sample_id": sample_id,
            "row_index": row_index,
            "uid": f"train_row_{row_index}",
            "source_dataset": Path(args.data_path).stem,
            "language": args.language,
            "raw_id": f"row_{row_index}",
            "selection": {
                "num_eligible_targets": len(sample["eligible_targets"]),
                "num_causal_annotation_edges": sample["num_edges"],
                "min_sources_per_target": args.min_sources_per_target,
                "teacher_forcing": True,
                "strict_causal": "s < q",
                "target_scope": args.target_scope,
            },
            "base": compact_model_result(
                name=args.base_name,
                tokens=sample["tokens"],
                target_rows=target_rows,
                c_rows=base_c,
                top_k=args.top_k,
                completion_start=sample["completion_start"],
                edges_by_q=sample["edges_by_q"],
            ),
            "ours": compact_model_result(
                name=args.sft_name,
                tokens=sample["tokens"],
                target_rows=target_rows,
                c_rows=sft_c,
                top_k=args.top_k,
                completion_start=sample["completion_start"],
                edges_by_q=sample["edges_by_q"],
            ),
            "annotation": annotation_payload(sample["tokens"], sample["edges_by_q"], target_rows, args.target_scope),
        })

    payload = {
        "version": 3,
        "title": "Teacher-forcing Base/SFT saliency with annotation edges",
        "mode": "teacher_forcing_gt_completion",
        "target_scope": args.target_scope,
        "strict_causal": "Only annotation edges with source < target are used.",
        "saliency_definition": "Last-layer contribution row saliency from src/train/loss.py::build_contribution_rows; same contribution definition as training loss, without free-run generation.",
        "top_k": args.top_k,
        "metric_names": {
            "recall_at_k": f"recall@{args.top_k}",
            "precision_at_k": f"precision@{args.top_k}",
            "map_at_k": f"mAP@{args.top_k}",
        },
        "metric_definition": "For each target token, rank causal source tokens by saliency. recall@k = hits among annotation sources / annotation source count; precision@k = hits / min(k, ranked source count); mAP@k = mean precision at annotated hits up to k, normalized by annotation source count.",
        "data_path": args.data_path,
        "base_model": args.base_model,
        "sft_model": args.sft_model,
        "samples": out_samples,
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {out} ({len(out_samples)} samples)")


if __name__ == "__main__":
    main()
