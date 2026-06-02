#!/usr/bin/env python3
"""Compute Ours/SFT saliency on selected prediction examples."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "outputs/visual_prediction/prediction_saliency_manifest_50.json"
DEFAULT_OUTPUT = ROOT / "outputs/visual_prediction/prediction_saliency_50.json"
DEFAULT_MODEL = ROOT / "outputs/benchmark/sft_qwen_ours_graphsignal_500"
DEFAULT_CACHE = ROOT / "outputs/visual_prediction/cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--output_path", default=str(DEFAULT_OUTPUT))
    p.add_argument("--model_path", default=str(DEFAULT_MODEL))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_completion_tokens", type=int, default=120)
    p.add_argument("--max_saliency_tokens", type=int, default=80)
    p.add_argument("--max_seq_len", type=int, default=3072)
    p.add_argument("--alti_p", type=int, default=1)
    p.add_argument("--alti_chunk_size", type=int, default=8)
    p.add_argument("--cache_dir", default=str(DEFAULT_CACHE))
    p.add_argument("--overwrite_cache", action="store_true")
    p.add_argument("--partial_write_every", type=int, default=1)
    p.add_argument("--progress_every", type=int, default=10)
    p.add_argument("--local_files_only", type=int, default=1)
    return p.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def write_json_atomic(path: Path, payload: Any, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)


def decode_escaped_for_prompt(text: str) -> str:
    if "\n" in text or "\t" in text:
        return text
    return text.replace("\\n", "\n").replace("\\t", "\t")


def build_messages_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        if role not in {"system", "user"}:
            continue
        content = decode_escaped_for_prompt(str(msg.get("content", "")))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def token_display(text: str) -> str:
    if text == "\n":
        return "\\n"
    if text == "\t":
        return "\\t"
    return text.replace("\n", "\\n").replace("\t", "\\t")


def fim_ranges(prompt_text: str) -> dict[str, tuple[int, int]]:
    fp = "<|fim_prefix|>"
    fs = "<|fim_suffix|>"
    fm = "<|fim_middle|>"
    a = prompt_text.find(fp)
    b = prompt_text.find(fs, a + len(fp)) if a >= 0 else -1
    c = prompt_text.find(fm, b + len(fs)) if b >= 0 else -1
    out: dict[str, tuple[int, int]] = {}
    if a >= 0 and b >= 0 and b > a + len(fp):
        out["prefix"] = (a + len(fp), b)
    if b >= 0 and c >= 0 and c > b + len(fs):
        out["suffix"] = (b + len(fs), c)
    return out


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def encode_prompt_tokens(prompt_text: str, tokenizer) -> tuple[list[int], list[dict[str, Any]]]:
    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    ranges = fim_ranges(prompt_text)
    tokens: list[dict[str, Any]] = []
    for idx, (tid, (start, end)) in enumerate(zip(input_ids, offsets)):
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        start_i, end_i = int(start), int(end)
        region = "prompt_other"
        for name, (a, b) in ranges.items():
            if overlaps(start_i, end_i, a, b):
                region = name
                break
        tokens.append({
            "idx": idx,
            "id": int(tid),
            "text": piece,
            "display": token_display(piece),
            "region": region,
            "is_special": bool(piece.startswith("<|") and piece.endswith("|>")),
            "is_whitespace": bool(piece.strip() == ""),
        })
    return input_ids, tokens


def encode_completion_tokens(text: str, tokenizer, start_idx: int, max_tokens: int) -> tuple[list[int], list[dict[str, Any]]]:
    ids = tokenizer(text, add_special_tokens=False).input_ids[:max_tokens]
    tokens = []
    for step, tid in enumerate(ids):
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        idx = start_idx + step
        tokens.append({
            "idx": idx,
            "id": int(tid),
            "text": piece,
            "display": token_display(piece),
            "region": "completion",
            "step": step,
            "is_special": bool(piece.startswith("<|") and piece.endswith("|>")),
            "is_whitespace": bool(piece.strip() == ""),
        })
    return ids, tokens


def display_tokens(prompt_tokens: list[dict[str, Any]], completion_tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix = [t for t in prompt_tokens if t["region"] == "prefix"]
    suffix = [t for t in prompt_tokens if t["region"] == "suffix"]
    return prefix + completion_tokens + suffix


def topk_from_saliency(saliency: list[float], tokens_by_idx: dict[int, dict[str, Any]], candidate_indices: list[int], top_k: int) -> list[dict[str, Any]]:
    pairs: list[tuple[float, int]] = []
    n = len(saliency)
    for idx in candidate_indices:
        if 0 <= idx < n and idx in tokens_by_idx:
            pairs.append((abs(float(saliency[idx])), idx))
    pairs.sort(key=lambda x: x[0], reverse=True)
    out = []
    for rank, (val, idx) in enumerate(pairs[:top_k], start=1):
        tok = tokens_by_idx[idx]
        out.append({
            "rank": rank,
            "idx": idx,
            "token": tok["text"],
            "display": tok["display"],
            "region": tok["region"],
            "value": val,
        })
    return out


def resolve_checkpoint(path_or_repo: str) -> str:
    p = Path(path_or_repo)
    if not p.exists():
        return path_or_repo
    if (p / "adapter_config.json").exists() or (p / "config.json").exists():
        return str(p)
    checkpoints = []
    for child in p.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            step = child.name.rsplit("-", 1)[-1]
            if step.isdigit():
                checkpoints.append((int(step), child))
    return str(sorted(checkpoints)[-1][1]) if checkpoints else str(p)


def load_model_and_tokenizer(path_or_repo: str, device: str, dtype: Any, local_files_only: bool):
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_checkpoint(path_or_repo)
    if Path(resolved).exists() and (Path(resolved) / "adapter_config.json").exists():
        peft_config = PeftConfig.from_pretrained(resolved, local_files_only=local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=True,
            local_files_only=local_files_only,
            pad_token="<|endoftext|>",
            eos_token="<|im_end|>",
            padding_side="right",
        )
        base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model = PeftModel.from_pretrained(base, resolved, is_trainable=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=True,
            local_files_only=local_files_only,
            pad_token="<|endoftext|>",
            eos_token="<|im_end|>",
            padding_side="right",
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device).eval()
    return model, tokenizer, resolved


def alti_model_view(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        if base is not None:
            base.eval()
            return base
    return model


def cache_path(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / f"{slugify(sample_id)}.json"


def compute_sample(model, alti_model, tokenizer, sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from src.attribution.saliency import compute_alti_saliency_vector

    prompt_text = build_messages_prompt(sample["row"].get("messages", []))
    prompt_ids, prompt_tokens = encode_prompt_tokens(prompt_text, tokenizer)
    if len(prompt_ids) >= args.max_seq_len:
        return {"skipped": True, "reason": f"prompt_len {len(prompt_ids)} >= max_seq_len {args.max_seq_len}"}

    base_ids, base_completion = encode_completion_tokens(
        sample.get("base_prediction", ""),
        tokenizer,
        len(prompt_ids),
        args.max_completion_tokens,
    )
    ours_ids, ours_completion = encode_completion_tokens(
        sample.get("ours_prediction", ""),
        tokenizer,
        len(prompt_ids),
        args.max_completion_tokens,
    )
    full_ids = prompt_ids + ours_ids
    if not ours_ids:
        return {"skipped": True, "reason": "empty Ours prediction"}

    prompt_display_indices = [t["idx"] for t in prompt_tokens if t["region"] in {"prefix", "suffix"}]
    targets: dict[str, Any] = {}
    targets_by_step: dict[str, int] = {}
    all_tokens = prompt_tokens + ours_completion
    tokens_by_idx = {int(t["idx"]): t for t in all_tokens}
    input_t = torch.tensor([full_ids], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_t, device=args.device)

    max_steps = min(len(ours_completion), args.max_saliency_tokens)
    for step, tok in enumerate(ours_completion[:max_steps]):
        if step == 0 or (step + 1) % max(1, args.progress_every) == 0:
            print(f"[saliency] {sample['sample_id']} token {step + 1}/{max_steps}", flush=True)
        target_idx = int(tok["idx"])
        saliency = compute_alti_saliency_vector(
            alti_model,
            {"input_ids": input_t, "attention_mask": attention_mask},
            target_idx_in_seq=target_idx,
            p=args.alti_p,
            chunk_size=args.alti_chunk_size,
        )
        candidate_indices = [i for i in range(target_idx) if i in tokens_by_idx and (i in prompt_display_indices or i >= len(prompt_ids))]
        targets[str(target_idx)] = {
            "target_idx": target_idx,
            "step": int(step),
            "token": tok["text"],
            "display": tok["display"],
            "sources": topk_from_saliency(saliency, tokens_by_idx, candidate_indices, args.top_k),
        }
        targets_by_step[str(step)] = target_idx

    return {
        "skipped": False,
        "prompt_len": len(prompt_ids),
        "base": {
            "tokens": display_tokens(prompt_tokens, base_completion),
            "prediction": sample.get("base_prediction", ""),
        },
        "ours": {
            "tokens": display_tokens(prompt_tokens, ours_completion),
            "prediction": sample.get("ours_prediction", ""),
            "targets": targets,
            "targets_by_step": targets_by_step,
            "generated_token_indices": [int(t["idx"]) for t in ours_completion],
        },
        "ground_truth": sample.get("ground_truth", ""),
    }


def main() -> None:
    args = parse_args()
    global torch
    import torch

    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"loading model: {args.model_path}", flush=True)
    model, tokenizer, resolved = load_model_and_tokenizer(args.model_path, args.device, dtype, bool(args.local_files_only))
    alti_model = alti_model_view(model)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    out_samples: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        sid = str(sample.get("sample_id") or sample.get("uid") or idx)
        result = None
        cp = cache_path(cache_dir, sid) if cache_dir else None
        if cp and cp.exists() and not args.overwrite_cache:
            result = json.loads(cp.read_text(encoding="utf-8"))
            print(f"[cache hit] {sid}", flush=True)
        if result is None:
            print(f"[sample] {idx + 1}/{len(samples)} #{sample.get('filtered_index')} {sample.get('uid')}", flush=True)
            result = compute_sample(model, alti_model, tokenizer, sample, args)
            if cp:
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {k: sample.get(k) for k in [
            "sample_id", "filtered_index", "uid", "source_dataset", "language", "raw_id",
            "entry_point", "prefix", "suffix", "ground_truth", "metrics",
        ]}
        row.update(result)
        out_samples.append(row)
        if args.partial_write_every > 0 and (idx + 1) % args.partial_write_every == 0:
            write_json_atomic(Path(args.output_path), {
                "version": 1,
                "model_path": args.model_path,
                "resolved_model_path": resolved,
                "top_k": args.top_k,
                "samples": out_samples,
            }, indent=2)
            print(f"[partial write] {args.output_path}", flush=True)

    payload = {
        "version": 1,
        "title": "训练前后test样本预测效果对比",
        "model_path": args.model_path,
        "resolved_model_path": resolved,
        "top_k": args.top_k,
        "samples": out_samples,
    }
    write_json_atomic(Path(args.output_path), payload, indent=2)
    print(f"wrote {args.output_path} ({len(out_samples)} samples)")


if __name__ == "__main__":
    main()
