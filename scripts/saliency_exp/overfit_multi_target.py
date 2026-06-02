#!/usr/bin/env python3
"""Single-sample, multi-completion-token pure saliency overfit experiment.

This script fixes one annotated training sample, collects completion target
positions q with at least one strict-causal labeled source s < q, and optimizes
only the current saliency InfoNCE loss over all selected q.

Each logged step records only the compact diagnostics needed for this stage:
per-query |Aq|, hit@|Aq|, Lq; plus global mean_hit@|Aq| and Lsal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "data/benchmarks/sft_data/ours_graphsignal_train_first500.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_RUN_DIR = ROOT / "runs/saliency_exp"
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA))
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--sample_index", type=int, default=110)
    parser.add_argument("--mode", choices=["ce_only", "saliency_only", "saliency_ce"], default="saliency_only")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ce_lambda", type=float, default=1.0, help="Deprecated compatibility arg; saliency_ce now uses CE + saliency_lambda * Lsal.")
    parser.add_argument("--saliency_lambda", type=float, default=0.1, help="Weight for Lsal in saliency_ce: total = CE + saliency_lambda * Lsal")
    parser.add_argument("--alpha", type=float, default=1.5, help="InfoNCE temperature used by current loss.py")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min_sources_per_target", type=int, default=1)
    parser.add_argument("--max_targets", type=int, default=0, help="0 means use all eligible completion targets")
    parser.add_argument("--top_k", type=int, default=10, help="K for recall@K, precision@K, and AP@K diagnostics")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use_peft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--jsonl_path", default="")
    parser.add_argument("--text_log_path", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local_files_only", type=int, default=1)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_jsonl_row(path: Path, index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"sample_index={index} not found in {path}")


def token_display(text: str) -> str:
    if text == "\n":
        return "\\n"
    if text == "\t":
        return "\\t"
    return text.replace("\n", "\\n").replace("\t", "\\t")


def decode_tokens(tokenizer: Any, input_ids: list[int]) -> list[dict[str, Any]]:
    tokens = []
    for idx, tid in enumerate(input_ids):
        text = tokenizer.decode([int(tid)], skip_special_tokens=False)
        tokens.append({
            "idx": idx,
            "id": int(tid),
            "text": text,
            "display": token_display(text),
        })
    return tokens


def first_label_index(labels: list[int]) -> int | None:
    return next((i for i, value in enumerate(labels) if int(value) != IGNORE_INDEX), None)


def normalize_edge(edge: dict[str, Any]) -> tuple[int, int]:
    src = int(edge.get("src", edge.get("source", edge.get("token_i_idx", -1))))
    dst = int(edge.get("dst", edge.get("target", edge.get("token_j_idx", -1))))
    return src, dst


def collect_query_sources(
    row: dict[str, Any],
    labels: list[int],
    completion_start: int,
    seq_len: int,
    min_sources_per_target: int,
    max_targets: int,
) -> dict[int, list[int]]:
    by_q: dict[int, set[int]] = {}
    for edge in row.get("attention_edges", []):
        src, dst = normalize_edge(edge)
        if not (0 <= src < dst < seq_len):
            continue
        if dst < completion_start or int(labels[dst]) == IGNORE_INDEX:
            continue
        by_q.setdefault(dst, set()).add(src)

    filtered = {
        q: sorted(srcs)
        for q, srcs in by_q.items()
        if len(srcs) >= min_sources_per_target and q > 0
    }
    if max_targets and max_targets > 0:
        ranked = sorted(filtered.items(), key=lambda item: (-len(item[1]), item[0]))[:max_targets]
        filtered = dict(sorted(ranked, key=lambda item: item[0]))
    else:
        filtered = dict(sorted(filtered.items()))
    return filtered


def load_model_and_tokenizer(args: argparse.Namespace):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        padding_side="right",
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype_from_name(args.dtype),
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )
    model.config.use_cache = False
    model.to(args.device)

    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def make_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"multi_target_sample{args.sample_index}_{args.mode}"
    jsonl = Path(args.jsonl_path) if args.jsonl_path else run_dir / f"{stem}.jsonl"
    text = Path(args.text_log_path) if args.text_log_path else run_dir / f"{stem}.log"
    for path in (jsonl, text):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite and path.exists():
            path.unlink()
    return jsonl, text


def build_annot_pairs(query_sources: dict[int, list[int]], device: str) -> torch.Tensor:
    pairs = []
    for q, srcs in query_sources.items():
        for src in srcs:
            pairs.append([src, q])
    return torch.tensor(pairs, dtype=torch.long, device=device)


def selected_query_ce_loss(outputs: Any, input_ids: torch.Tensor, query_sources: dict[int, list[int]]) -> torch.Tensor:
    queries = torch.tensor(sorted(query_sources), dtype=torch.long, device=input_ids.device)
    logits = outputs.logits[0, queries - 1, :].float()
    labels = input_ids[0, queries]
    return F.cross_entropy(logits, labels)


def compute_query_metrics(
    c_matrix: torch.Tensor,
    query_sources: dict[int, list[int]],
    tokens: list[dict[str, Any]],
    *,
    alpha: float,
    eps: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    losses = []
    query_rows: list[dict[str, Any]] = []
    for q, srcs in query_sources.items():
        row = c_matrix[0, q, :q].float()
        labeled = torch.tensor(srcs, dtype=torch.long, device=row.device)
        scores = torch.log(row.clamp_min(eps))
        logits = scores / alpha
        log_probs = F.log_softmax(logits, dim=-1)
        Lq = -log_probs[labeled].mean()
        losses.append(Lq)

        k = len(srcs)
        k_eff = min(max(1, int(top_k)), int(row.numel()))
        order = torch.argsort(row.detach(), descending=True).cpu().tolist()
        top = order[:k_eff]
        labeled_set = set(srcs)
        hit_count = sum(1 for idx in top if idx in labeled_set)
        hit_rate = hit_count / max(1, k)
        precision_at_k = hit_count / max(1, k_eff)
        recall_at_k = hit_count / max(1, k)
        hits_so_far = 0
        precision_sum = 0.0
        for rank, idx in enumerate(top, start=1):
            if idx in labeled_set:
                hits_so_far += 1
                precision_sum += hits_so_far / rank
        ap_at_k = precision_sum / max(1, k)
        query_rows.append({
            "q": int(q),
            "target_token": tokens[q]["text"],
            "target_display": tokens[q]["display"],
            "num_labeled_sources": int(k),
            "top_k": int(top_k),
            "hit_at_Aq_count": int(hit_count),
            "hit_at_Aq": float(hit_rate),
            "recall_at_k": float(recall_at_k),
            "precision_at_k": float(precision_at_k),
            "ap_at_k": float(ap_at_k),
            "Lq": float(Lq.detach().cpu()),
        })
    if not losses:
        raise RuntimeError("No selected queries to optimize")
    return query_rows, torch.stack(losses).mean()


def format_record(record: dict[str, Any]) -> str:
    head = (
        f"step={record['step']:04d} Lsal={record['Lsal']:.6f} "
        f"mean_hit@Aq={record['mean_hit_at_Aq']:.4f} "
        f"R@{record['top_k']}={record['mean_recall_at_k']:.4f} "
        f"P@{record['top_k']}={record['mean_precision_at_k']:.4f} "
        f"mAP@{record['top_k']}={record['mean_map_at_k']:.4f} "
        f"num_q={record['num_queries']}"
    )
    if record.get("CE") is not None:
        head += f" CE={record['CE']:.6f} sal_lambda={record.get('saliency_lambda', 0.0):.4f} total={record['total_loss']:.6f}"
    lines = [head]
    for qrow in record["queries"]:
        lines.append(
            f"  q=#{qrow['q']:>4} {qrow['target_display']!r} "
            f"|Aq|={qrow['num_labeled_sources']:<2} "
            f"hit@|Aq|={qrow['hit_at_Aq_count']}/{qrow['num_labeled_sources']} "
            f"({qrow['hit_at_Aq']:.3f}) "
            f"R@{qrow['top_k']}={qrow['recall_at_k']:.3f} "
            f"P@{qrow['top_k']}={qrow['precision_at_k']:.3f} "
            f"AP@{qrow['top_k']}={qrow['ap_at_k']:.3f} Lq={qrow['Lq']:.6f}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    jsonl_path, text_log_path = make_output_paths(args)

    row = load_jsonl_row(Path(args.data_path), args.sample_index)
    input_ids_full = [int(x) for x in row.get("input_ids", [])]
    labels_full = [int(x) for x in row.get("label", row.get("labels", []))]
    if not input_ids_full or len(input_ids_full) != len(labels_full):
        raise ValueError("row must contain same-length input_ids and label/labels")

    completion_start = first_label_index(labels_full)
    if completion_start is None:
        raise ValueError("row has no completion labels")

    model, tokenizer = load_model_and_tokenizer(args)
    tokens = decode_tokens(tokenizer, input_ids_full)
    query_sources = collect_query_sources(
        row,
        labels_full,
        completion_start,
        len(input_ids_full),
        args.min_sources_per_target,
        args.max_targets,
    )
    if not query_sources:
        raise ValueError("No eligible completion target with strict-causal annotation sources")

    input_ids = torch.tensor([input_ids_full], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids, device=args.device)
    annot_pairs = build_annot_pairs(query_sources, args.device)

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Use --use_peft or unfreeze the model.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    header = {
        "type": "header",
        "config": vars(args),
        "sample_index": args.sample_index,
        "completion_start": completion_start,
        "sequence_len": len(input_ids_full),
        "num_queries": len(query_sources),
        "num_edges": int(annot_pairs.size(0)),
        "queries": [
            {
                "q": int(q),
                "target_token": tokens[q]["text"],
                "target_display": tokens[q]["display"],
                "num_labeled_sources": len(srcs),
                "labeled_sources": srcs,
            }
            for q, srcs in query_sources.items()
        ],
        "note": (
            "Single-sample multi-target experiment. Lsal is mean_q Lq. "
            "hit@|Aq| uses top-|Aq| saliency sources for each q. "
            "recall@K, precision@K, and AP@K use strict-causal top-K sources. "
            "In saliency_ce/ce_only mode, CE is averaged over the same selected q positions."
        ),
    }
    text_log_path.write_text(json.dumps(header, ensure_ascii=False, indent=2) + "\n\n", encoding="utf-8")
    with jsonl_path.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps(header, ensure_ascii=False) + "\n")

    print(f"jsonl -> {jsonl_path}")
    print(f"text  -> {text_log_path}")
    print(f"sample={args.sample_index} num_queries={len(query_sources)} num_edges={annot_pairs.size(0)}")

    from src.train.loss import build_contribution_matrix, compute_saliency_loss

    for step in range(args.steps + 1):
        should_log = step == 0 or step == args.steps or (step % max(1, args.log_every) == 0)
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
        c_matrix = build_contribution_matrix(model, outputs.hidden_states[-2], outputs.attentions[-1])
        diag = compute_saliency_loss(c_matrix, [annot_pairs], alpha=args.alpha, eps=args.eps)
        query_rows, Lsal_recomputed = compute_query_metrics(
            c_matrix,
            query_sources,
            tokens,
            alpha=args.alpha,
            eps=args.eps,
            top_k=args.top_k,
        )
        Lsal = diag.loss
        if args.mode in {"saliency_ce", "ce_only"}:
            ce_loss = selected_query_ce_loss(outputs, input_ids, query_sources)
        else:
            ce_loss = None
        if args.mode == "saliency_ce":
            total_loss = ce_loss + args.saliency_lambda * Lsal
        elif args.mode == "ce_only":
            total_loss = ce_loss
        else:
            total_loss = Lsal

        if should_log:
            mean_hit = sum(row["hit_at_Aq"] for row in query_rows) / len(query_rows)
            mean_recall = sum(row["recall_at_k"] for row in query_rows) / len(query_rows)
            mean_precision = sum(row["precision_at_k"] for row in query_rows) / len(query_rows)
            mean_map = sum(row["ap_at_k"] for row in query_rows) / len(query_rows)
            record = {
                "type": "step",
                "step": int(step),
                "mode": args.mode,
                "Lsal": float(Lsal.detach().cpu()),
                "Lsal_recomputed": float(Lsal_recomputed.detach().cpu()),
                "CE": None if ce_loss is None else float(ce_loss.detach().cpu()),
                "ce_lambda": args.ce_lambda,
                "saliency_lambda": args.saliency_lambda,
                "total_loss": float(total_loss.detach().cpu()),
                "top_k": int(args.top_k),
                "mean_hit_at_Aq": float(mean_hit),
                "mean_recall_at_k": float(mean_recall),
                "mean_precision_at_k": float(mean_precision),
                "mean_map_at_k": float(mean_map),
                "num_queries": len(query_rows),
                "queries": query_rows,
            }
            formatted = format_record(record)
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            with text_log_path.open("a", encoding="utf-8") as tf:
                tf.write(formatted + "\n\n")
            print(formatted + "\n", flush=True)

        if step >= args.steps:
            break
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        del outputs, c_matrix, diag, query_rows, Lsal_recomputed, Lsal, total_loss
        if ce_loss is not None:
            del ce_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("done")


if __name__ == "__main__":
    main()
