#!/usr/bin/env python3
"""Overfit one GoSingle row with saliency loss and report alignment dynamics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json")
    p.add_argument("--row_index", type=int, default=479)
    p.add_argument("--base_model", default="models/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--target_scope", choices=["all", "prompt", "completion"], default="completion")
    p.add_argument("--loss_type", choices=["softmax_margin", "infonce_floor", "softmax"], default="softmax_margin")
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--floor_eps", type=float, default=2.4)
    p.add_argument("--min_sources_per_target", type=int, default=2)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--local_files_only", type=int, default=1)
    p.add_argument("--source_chunk_size", type=int, default=16)
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def read_jsonl_row(path: Path, row_index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == row_index:
                return json.loads(line)
    raise IndexError(f"row_index={row_index} not found in {path}")


def first_completion_idx(labels: list[int]) -> int:
    for idx, value in enumerate(labels):
        if int(value) != IGNORE_INDEX:
            return idx
    raise ValueError("no completion label found")


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
        tokens.append({
            "idx": idx,
            "id": int(tid),
            "text": text,
            "display": token_display(text),
            "region": "completion" if idx >= completion_start else "prompt_code",
            "is_special": bool(text.startswith("<|") and text.endswith("|>")),
            "is_whitespace": bool(text.strip() == ""),
        })
    return tokens


def normalize_edge(edge: dict[str, Any]) -> tuple[int, int, str]:
    src = int(edge.get("src", edge.get("source", edge.get("token_i_idx", -1))))
    dst = int(edge.get("dst", edge.get("target", edge.get("token_j_idx", -1))))
    subtype = str(edge.get("subtype", edge.get("type", "")) or "")
    return src, dst, subtype


def build_edges_by_q(
    row: dict[str, Any],
    tokens: list[dict[str, Any]],
    completion_start: int,
    target_scope: str,
    min_sources_per_target: int,
) -> dict[int, list[dict[str, Any]]]:
    seq_len = len(tokens)
    raw: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in row.get("attention_edges", []):
        src, dst, subtype = normalize_edge(edge)
        if not (0 <= src < dst < seq_len):
            continue
        if target_scope == "completion" and dst < completion_start:
            continue
        if target_scope == "prompt" and dst >= completion_start:
            continue
        raw[dst].append({"src": src, "dst": dst, "subtype": subtype})

    out: dict[int, list[dict[str, Any]]] = {}
    for q, edges in raw.items():
        tok = tokens[q]
        if tok["is_special"] or tok["is_whitespace"]:
            continue
        if len({int(e["src"]) for e in edges}) < min_sources_per_target:
            continue
        out[q] = edges
    return dict(sorted(out.items()))


def pairs_from_edges(edges_by_q: dict[int, list[dict[str, Any]]], device: torch.device) -> torch.Tensor:
    pairs = []
    seen = set()
    for q, edges in edges_by_q.items():
        for edge in edges:
            key = (int(edge["src"]), int(q))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device)


def setup_model_and_tokenizer(args: argparse.Namespace):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        padding_side="right",
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype_from_name(args.dtype),
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )
    model.config.use_cache = False
    model.to(args.device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def canonical_loss_type(value: str) -> str:
    return "infonce_floor" if value == "softmax_margin" else value


def rank_metrics(ranked: list[int], annot: set[int], k: int) -> dict[str, float | int]:
    top = ranked[:k]
    hits = 0
    ap = 0.0
    for rank, src in enumerate(top, start=1):
        if src in annot:
            hits += 1
            ap += hits / rank
    denom = max(1, len(annot))
    return {"hits": hits, "recall": hits / denom, "precision": hits / max(1, len(top)), "ap": ap / denom}


def summarize_rows(
    c_rows: torch.Tensor,
    target_qs: list[int],
    edges_by_q: dict[int, list[dict[str, Any]]],
    tokens: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_rows = []
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_id, q in enumerate(target_qs):
        scores = c_rows[row_id, :q].detach().float().cpu()
        ranked = torch.argsort(scores, descending=True).tolist()
        annot = {int(e["src"]) for e in edges_by_q[q]}
        rank_by_src = {src: rank for rank, src in enumerate(ranked, start=1)}
        ann_ranks = [rank_by_src[src] for src in annot if src in rank_by_src]
        m10 = rank_metrics(ranked, annot, 10)
        m20 = rank_metrics(ranked, annot, 20)
        region = str(tokens[q]["region"])
        row = {
            "query": q,
            "display": tokens[q]["display"],
            "region": region,
            "num_annotation_sources": len(annot),
            "recall@10": m10["recall"],
            "recall@20": m20["recall"],
            "precision@10": m10["precision"],
            "mAP@10": m10["ap"],
            "mean_annotation_rank": mean(ann_ranks) if ann_ranks else None,
            "min_annotation_rank": min(ann_ranks) if ann_ranks else None,
            "max_annotation_rank": max(ann_ranks) if ann_ranks else None,
            "top10": [
                {
                    "rank": rank + 1,
                    "src": int(src),
                    "display": tokens[int(src)]["display"],
                    "region": tokens[int(src)]["region"],
                    "is_annotation": int(src) in annot,
                    "value": float(scores[int(src)]),
                }
                for rank, src in enumerate(ranked[:10])
            ],
            "missed@10": [
                {
                    "src": int(src),
                    "display": tokens[int(src)]["display"],
                    "region": tokens[int(src)]["region"],
                    "rank": rank_by_src.get(int(src)),
                    "value": float(scores[int(src)]),
                }
                for src in sorted(annot, key=lambda s: rank_by_src.get(s, 10**9))
                if src not in set(ranked[:10])
            ],
        }
        query_rows.append(row)
        by_region["all"].append(row)
        by_region[region].append(row)

    summary = {}
    for region, rows in sorted(by_region.items()):
        summary[region] = {
            "n_queries": len(rows),
            "recall@10": mean([float(r["recall@10"]) for r in rows]) if rows else 0.0,
            "recall@20": mean([float(r["recall@20"]) for r in rows]) if rows else 0.0,
            "precision@10": mean([float(r["precision@10"]) for r in rows]) if rows else 0.0,
            "mAP@10": mean([float(r["mAP@10"]) for r in rows]) if rows else 0.0,
            "mean_annotation_rank": mean([float(r["mean_annotation_rank"]) for r in rows if r["mean_annotation_rank"] is not None]) if rows else None,
        }
    return summary, query_rows


@torch.no_grad()
def evaluate(model, input_ids, attention_mask, annot_pairs, edges_by_q, tokens, args, step: int, train_loss: float | None = None) -> dict[str, Any]:
    from src.train.loss import build_contribution_rows, saliency_loss_from_outputs

    model.eval()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    eval_diag = saliency_loss_from_outputs(
        model,
        outputs,
        [annot_pairs],
        alpha=args.alpha,
        eps=args.eps,
        floor_eps=args.floor_eps,
        loss_type=canonical_loss_type(args.loss_type),
    )
    target_qs = sorted(edges_by_q)
    row_batch = torch.zeros(len(target_qs), dtype=torch.long, device=input_ids.device)
    row_qry = torch.tensor(target_qs, dtype=torch.long, device=input_ids.device)
    c_rows = build_contribution_rows(
        model,
        outputs.hidden_states[-2],
        outputs.attentions[-1],
        row_batch,
        row_qry,
        source_chunk_size=args.source_chunk_size,
    )
    summary, query_rows = summarize_rows(c_rows, target_qs, edges_by_q, tokens)
    worst = sorted(query_rows, key=lambda r: (float(r["recall@10"]), -(r["num_annotation_sources"])))[:8]
    return {
        "step": int(step),
        "train_loss": train_loss,
        "eval_loss": float(eval_diag.loss.detach().cpu()),
        "Cbar": eval_diag.avg_C,
        "Nbar": eval_diag.avg_N,
        "ratio": eval_diag.avg_ratio,
        "n_queries": eval_diag.n_queries,
        "summary": summary,
        "worst_queries": worst,
    }


def main() -> None:
    args = parse_args()
    model, tokenizer = setup_model_and_tokenizer(args)
    row = read_jsonl_row(Path(args.data_path), args.row_index)
    input_ids_list = [int(x) for x in row["input_ids"]]
    labels = [int(x) for x in row.get("label", row.get("labels", []))]
    completion_start = first_completion_idx(labels)
    tokens = decode_tokens(tokenizer, input_ids_list, completion_start)
    edges_by_q = build_edges_by_q(row, tokens, completion_start, args.target_scope, args.min_sources_per_target)
    if not edges_by_q:
        raise RuntimeError("No eligible annotation targets for requested scope")

    device = torch.device(args.device)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    annot_pairs = pairs_from_edges(edges_by_q, device=device)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    history = []
    history.append(evaluate(model, input_ids, attention_mask, annot_pairs, edges_by_q, tokens, args, step=0))
    print(json.dumps({"step": 0, "summary": history[-1]["summary"], "eval_loss": history[-1]["eval_loss"]}, ensure_ascii=False), flush=True)

    from src.train.loss import saliency_loss_from_outputs

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        diag = saliency_loss_from_outputs(
            model,
            outputs,
            [annot_pairs],
            alpha=args.alpha,
            eps=args.eps,
            floor_eps=args.floor_eps,
            loss_type=canonical_loss_type(args.loss_type),
        )
        loss = diag.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.max_grad_norm)
        optimizer.step()
        train_loss = float(loss.detach().cpu())
        del outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if step % args.eval_every == 0 or step == args.steps:
            rec = evaluate(model, input_ids, attention_mask, annot_pairs, edges_by_q, tokens, args, step=step, train_loss=train_loss)
            history.append(rec)
            print(json.dumps({"step": step, "summary": rec["summary"], "eval_loss": rec["eval_loss"]}, ensure_ascii=False), flush=True)

    report = {
        "row_index": args.row_index,
        "data_path": args.data_path,
        "base_model": args.base_model,
        "target_scope": args.target_scope,
        "loss_type": args.loss_type,
        "alpha": args.alpha,
        "floor_eps": args.floor_eps,
        "steps": args.steps,
        "lr": args.lr,
        "completion_start": completion_start,
        "num_annotation_pairs": int(annot_pairs.shape[0]),
        "target_queries": sorted(edges_by_q),
        "history": history,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
