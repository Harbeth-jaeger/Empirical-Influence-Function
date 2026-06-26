#!/usr/bin/env python3
"""Fit one annotation-rich sample with saliency loss and append it to the viewer.

This is a presentation-oriented utility:

* choose one tokenizer-aligned GraphSignal sample,
* start from Base Qwen with a small LoRA adapter,
* optimize only the saliency loss on that single sample,
* compute before/after saliency on the ground-truth completion tokens, and
* append the resulting sample to the existing annotation-saliency viewer data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_RICH = ROOT / "outputs/viz_annotation/rich_top_edges/viewer_data.json"
DEFAULT_TRAIN = ROOT / "/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/sft_data/ours_graphsignal_train.json"
DEFAULT_DATA = ROOT / "outputs/viz_saliency/base_vs_ours_annotation_saliency_data_v2.json"
DEFAULT_MODEL_OUT = ROOT / "outputs/viz_saliency/rich_line319_saliency_only_lora"
DEFAULT_REPORT = ROOT / "outputs/viz_saliency/rich_line319_saliency_only_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default=DEFAULT_BASE)
    parser.add_argument("--rich_data", default=str(DEFAULT_RICH))
    parser.add_argument("--train_data", default=str(DEFAULT_TRAIN))
    parser.add_argument("--viewer_data", default=str(DEFAULT_DATA))
    parser.add_argument("--output_model", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument("--report_path", default=str(DEFAULT_REPORT))
    parser.add_argument("--sample_index", type=int, default=318)
    parser.add_argument("--sample_uid", default="mceval_instruct:line_319")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_display_completion_tokens", type=int, default=120)
    parser.add_argument("--local_files_only", type=int, default=1)
    parser.add_argument("--ranking", choices=["saliency", "delta", "both"], default="saliency")
    parser.add_argument("--overwrite_sample", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_jsonl_row(path: Path, index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"row {index} not found in {path}")


def token_display(text: str) -> str:
    if text == "\n":
        return "\\n"
    if text == "\t":
        return "\\t"
    return text.replace("\n", "\\n").replace("\t", "\\t")


def normalize_rich_token(tok: dict[str, Any], input_ids: list[int]) -> dict[str, Any]:
    idx = int(tok["idx"])
    original_idx = int(tok.get("original_idx", idx))
    text = str(tok.get("text") or tok.get("display") or "")
    display = str(tok.get("display") or token_display(text))
    region = "completion" if tok.get("is_completion") else "prompt_code"
    out = {
        "idx": idx,
        "original_idx": original_idx,
        "id": int(input_ids[original_idx]) if 0 <= original_idx < len(input_ids) else None,
        "text": text,
        "display": display,
        "region": region,
        "is_special": bool(text.startswith("<|") and text.endswith("|>")),
        "is_whitespace": bool(text.strip() == ""),
    }
    return out


def annotation_payload(rich_sample: dict[str, Any], token_by_idx: dict[int, dict[str, Any]]) -> dict[str, Any]:
    edges_by_dst: dict[str, list[dict[str, Any]]] = {}
    for edge in rich_sample.get("edges", []):
        src = int(edge["source"])
        dst = int(edge["target"])
        src_tok = token_by_idx.get(src, {})
        dst_tok = token_by_idx.get(dst, {})
        edges_by_dst.setdefault(str(dst), []).append({
            "src": src,
            "dst": dst,
            "subtype": edge.get("subtype", ""),
            "src_display": src_tok.get("display") or src_tok.get("text") or str(src),
            "src_text": src_tok.get("text") or src_tok.get("display") or str(src),
            "dst_display": dst_tok.get("display") or dst_tok.get("text") or str(dst),
        })
    return {
        "num_edges": len(rich_sample.get("edges", [])),
        "edges_by_dst": edges_by_dst,
        "selection": rich_sample.get("selection", {}),
        "note": "Ground-truth completion; annotation-rich showcase from outputs/viz_annotation/rich_top_edges.",
    }


def make_source_rows(
    c_matrix,
    target_tok: dict[str, Any],
    visible_tokens: list[dict[str, Any]],
    *,
    top_k: int,
    rank_by_abs: bool = True,
) -> list[dict[str, Any]]:
    target_orig = int(target_tok["original_idx"])
    pairs: list[tuple[float, dict[str, Any]]] = []
    for src_tok in visible_tokens:
        src_orig = int(src_tok["original_idx"])
        if src_orig >= target_orig:
            continue
        raw_val = float(c_matrix[target_orig, src_orig])
        val = abs(raw_val) if rank_by_abs else raw_val
        pairs.append((val, src_tok))
    pairs.sort(key=lambda item: item[0], reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, (val, src_tok) in enumerate(pairs[:top_k], start=1):
        rows.append({
            "rank": rank,
            "idx": int(src_tok["idx"]),
            "token": src_tok["text"],
            "display": src_tok["display"],
            "region": src_tok["region"],
            "value": float(val),
        })
    return rows


def compact_model_payload(
    name: str,
    c_matrix,
    visible_tokens: list[dict[str, Any]],
    completion_tokens: list[dict[str, Any]],
    *,
    top_k: int,
    rank_by_abs: bool = True,
    saliency_definition: str = "Last-layer ALTI contribution matrix, computed on ground-truth sequence.",
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    targets_by_step: dict[str, int] = {}
    for step, tok in enumerate(completion_tokens):
        tok = dict(tok)
        tok["step"] = step
        visible_tokens[tok["idx"]]["step"] = step
        targets[str(tok["idx"])] = {
            "target_idx": int(tok["idx"]),
            "query_idx": int(tok["idx"]),
            "step": int(step),
            "token": tok["text"],
            "display": tok["display"],
            "sources": make_source_rows(c_matrix, tok, visible_tokens, top_k=top_k, rank_by_abs=rank_by_abs),
            "saliency_definition": saliency_definition,
        }
        targets_by_step[str(step)] = int(tok["idx"])
    return {
        "name": name,
        "tokens": visible_tokens,
        "generated_token_indices": [int(t["idx"]) for t in completion_tokens],
        "targets_by_step": targets_by_step,
        "targets": targets,
        "generated_text": "".join(t["text"] for t in completion_tokens),
        "skipped": False,
        "reason": "",
    }


def build_showcase_sample(
    *,
    rich_sample: dict[str, Any],
    visible_tokens: list[dict[str, Any]],
    completion_tokens: list[dict[str, Any]],
    annotation: dict[str, Any],
    base_c,
    fit_c,
    top_k: int,
    ranking: str,
) -> dict[str, Any]:
    base_tokens = [dict(t) for t in visible_tokens]
    fit_tokens = [dict(t) for t in visible_tokens]
    base_completion = [dict(t) for t in completion_tokens]
    fit_completion = [dict(t) for t in completion_tokens]
    base_payload = compact_model_payload(
        "Base Qwen",
        base_c,
        base_tokens,
        base_completion,
        top_k=top_k,
        rank_by_abs=True,
    )

    sample_id = f"rich_showcase_{rich_sample['sample_index']}_python"
    uid = str(rich_sample["uid"])
    if ranking == "delta":
        sample_id += "_delta_saliency"
        uid += " · Δsaliency"
        right_matrix = fit_c - base_c
        right_definition = "Positive delta of last-layer ALTI contribution after training minus Base, computed on ground-truth sequence."
        rank_by_abs = False
    else:
        right_matrix = fit_c
        right_definition = "Last-layer ALTI contribution matrix, computed on ground-truth sequence."
        rank_by_abs = True

    fit_payload = compact_model_payload(
        "Ours GraphSignal",
        right_matrix,
        fit_tokens,
        fit_completion,
        top_k=top_k,
        rank_by_abs=rank_by_abs,
        saliency_definition=right_definition,
    )

    return {
        "sample_id": sample_id,
        "row_index": int(rich_sample["sample_index"]),
        "uid": uid,
        "source_dataset": rich_sample.get("source_dataset", "mceval_instruct"),
        "language": rich_sample.get("language", "python"),
        "raw_id": rich_sample.get("raw_id", ""),
        "selection": rich_sample.get("selection", {}),
        "base": base_payload,
        "ours": fit_payload,
        "annotation": annotation,
        "ranking": ranking,
    }


def saliency_overlap(model_payload: dict[str, Any], annotation: dict[str, Any], top_k: int) -> dict[str, Any]:
    total = 0
    hit = 0
    best = {"step": None, "target_idx": None, "hits": 0, "ann": 0, "display": ""}
    for target in model_payload.get("targets", {}).values():
        ann_rows = annotation["edges_by_dst"].get(str(target["target_idx"]), [])
        if not ann_rows:
            continue
        ann_src = {int(e["src"]) for e in ann_rows}
        sal_src = {int(r["idx"]) for r in target.get("sources", [])[:top_k]}
        h = len(ann_src & sal_src)
        total += 1
        hit += int(h > 0)
        if h > best["hits"]:
            best = {
                "step": target["step"],
                "target_idx": target["target_idx"],
                "hits": h,
                "ann": len(ann_src),
                "display": target.get("display", ""),
            }
    return {"targets_with_annotation": total, "targets_with_topk_hit": hit, "best": best}


def load_model_and_tokenizer(args):
    import torch
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
    model.print_trainable_parameters()
    return model, tokenizer


def compute_c_matrix(model, input_ids, attention_mask):
    import torch
    from src.train.loss import build_contribution_matrix

    model.eval()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        c_matrix = build_contribution_matrix(model, outputs.hidden_states[-2], outputs.attentions[-1])[0].float().cpu()
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return c_matrix


def train_saliency_only(model, input_ids, attention_mask, annot_pairs, args) -> list[dict[str, Any]]:
    import torch
    from src.train.loss import saliency_loss_from_outputs

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    history: list[dict[str, Any]] = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        diag = saliency_loss_from_outputs(model, outputs, [annot_pairs], alpha=args.alpha)
        loss = diag.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        row = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "avg_C": diag.avg_C,
            "avg_N": diag.avg_N,
            "avg_ratio": diag.avg_ratio,
            "n_queries": diag.n_queries,
        }
        history.append(row)
        print(
            f"[fit] step {row['step']:03d}/{args.steps} "
            f"loss={row['loss']:.5f} ratio={row['avg_ratio']:.3f} queries={row['n_queries']}",
            flush=True,
        )
        del outputs, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


def main() -> None:
    args = parse_args()
    import torch

    rich_data = json.loads(Path(args.rich_data).read_text(encoding="utf-8"))
    rich_sample = next(
        (
            s for s in rich_data.get("samples", [])
            if int(s.get("sample_index", -1)) == args.sample_index or s.get("uid") == args.sample_uid
        ),
        None,
    )
    if rich_sample is None:
        raise SystemExit(f"rich sample not found: index={args.sample_index} uid={args.sample_uid}")

    train_row = load_jsonl_row(Path(args.train_data), int(rich_sample["sample_index"]))
    input_ids_list = [int(x) for x in train_row["input_ids"]]
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids, device=args.device)
    annot_pairs = torch.tensor(
        [[int(e["src"]), int(e["dst"])] for e in train_row.get("attention_edges", [])],
        dtype=torch.long,
        device=args.device,
    )

    model, _tokenizer = load_model_and_tokenizer(args)
    print(f"[sample] uid={rich_sample['uid']} row={rich_sample['sample_index']} edges={len(train_row.get('attention_edges', []))}")
    print("[base] computing before-fit contribution matrix", flush=True)
    base_c = compute_c_matrix(model, input_ids, attention_mask)

    history = train_saliency_only(model, input_ids, attention_mask, annot_pairs, args)
    out_model = Path(args.output_model)
    out_model.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_model)
    print(f"[save] adapter -> {out_model}", flush=True)

    print("[fit] computing after-fit contribution matrix", flush=True)
    fit_c = compute_c_matrix(model, input_ids, attention_mask)

    visible_tokens = [normalize_rich_token(t, input_ids_list) for t in rich_sample.get("tokens", [])]
    visible_tokens.sort(key=lambda t: int(t["idx"]))
    token_by_idx = {int(t["idx"]): t for t in visible_tokens}
    max_completion = max(1, int(args.max_display_completion_tokens))
    completion_tokens = [t for t in visible_tokens if t["region"] == "completion"][:max_completion]

    annotation = annotation_payload(rich_sample, token_by_idx)

    rankings = ["saliency", "delta"] if args.ranking == "both" else [args.ranking]
    showcase_samples = [
        build_showcase_sample(
            rich_sample=rich_sample,
            visible_tokens=visible_tokens,
            completion_tokens=completion_tokens,
            annotation=annotation,
            base_c=base_c,
            fit_c=fit_c,
            top_k=args.top_k,
            ranking=ranking,
        )
        for ranking in rankings
    ]

    viewer_path = Path(args.viewer_data)
    viewer = json.loads(viewer_path.read_text(encoding="utf-8")) if viewer_path.exists() else {"version": 2, "samples": []}
    samples = list(viewer.get("samples", []))
    new_ids = {s["sample_id"] for s in showcase_samples}
    legacy_ids = {f"rich_showcase_{rich_sample['sample_index']}_python_saliency_only"}
    if args.overwrite_sample:
        samples = [s for s in samples if s.get("sample_id") not in new_ids and s.get("sample_id") not in legacy_ids]
    elif any(s.get("sample_id") in new_ids for s in samples):
        raise SystemExit(f"sample already exists in {viewer_path}; pass --overwrite_sample to replace it")
    for sample in reversed(showcase_samples):
        samples.insert(0, sample)
    viewer["samples"] = samples
    viewer["top_k"] = args.top_k
    viewer["saliency_source"] = "ground-truth completion saliency comparison"
    viewer["annotation_source"] = str(args.rich_data)
    viewer_path.parent.mkdir(parents=True, exist_ok=True)
    viewer_path.write_text(json.dumps(viewer, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "sample_ids": [s["sample_id"] for s in showcase_samples],
        "uids": [s["uid"] for s in showcase_samples],
        "sample_index": rich_sample["sample_index"],
        "num_visible_tokens": len(visible_tokens),
        "num_completion_tokens_displayed": len(completion_tokens),
        "num_original_tokens": len(input_ids_list),
        "num_annotation_edges": len(train_row.get("attention_edges", [])),
        "history": history,
        "overlap": {
            s["sample_id"]: {
                "base": saliency_overlap(s["base"], annotation, args.top_k),
                "ours": saliency_overlap(s["ours"], annotation, args.top_k),
            }
            for s in showcase_samples
        },
        "output_model": str(out_model),
        "viewer_data": str(viewer_path),
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] viewer data -> {viewer_path}", flush=True)
    print(f"[write] report -> {report_path}", flush=True)
    print("[overlap]", report["overlap"], flush=True)


if __name__ == "__main__":
    main()
