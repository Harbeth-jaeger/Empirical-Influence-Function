#!/usr/bin/env python3
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

DEFAULT_MANIFEST = ROOT / "outputs/visual_saliency/saliency_viz_samples.json"
DEFAULT_OUTPUT = ROOT / "outputs/visual_saliency/saliency_comparison_data.json"
DEFAULT_CACHE_DIR = ROOT / "outputs/visual_saliency/alti_freerun_cache"
DEFAULT_OURS = ROOT / "outputs/benchmark/sft_qwen_ours_graphsignal_500"
DEFAULT_BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

DEFAULT_MODEL_SPECS = [
    f"Ours GraphSignal={DEFAULT_OURS}",
    f"CLEAR={ROOT / 'outputs/benchmark/sft_qwen_clear'}",
    f"XTF={ROOT / 'outputs/benchmark/sft_qwen_xtf'}",
    f"IB-FT={ROOT / 'outputs/benchmark/sft_qwen_ibft'}",
    f"TokenCleaning={ROOT / 'outputs/benchmark/sft_qwen_tokencleaning'}",
    f"Base Qwen={DEFAULT_BASE}",
    f"LLM-CleanCode={ROOT / 'outputs/benchmark/sft_qwen_llm_cleaning'}",
]


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def write_json_atomic(path: Path, payload: Any, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)


def cache_path_for(cache_dir: Path, model_name: str, sample_id: str) -> Path:
    return cache_dir / slugify(model_name) / f"{slugify(sample_id)}.json"


def read_cached_result(cache_dir: Path | None, model_name: str, sample_id: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = cache_path_for(cache_dir, model_name, sample_id)
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("result", obj)


def write_cached_result(cache_dir: Path | None, model_name: str, sample_id: str, result: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    path = cache_path_for(cache_dir, model_name, sample_id)
    write_json_atomic(path, {"model": model_name, "sample_id": sample_id, "result": result})


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


def parse_model_specs(values: list[str]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"model spec must be NAME=PATH_OR_REPO, got: {value}")
        name, path = value.split("=", 1)
        specs.append({"name": name.strip(), "path": path.strip()})
    return specs


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
    if checkpoints:
        return str(sorted(checkpoints)[-1][1])
    return str(p)


def setup_tokenizer(path: str, local_files_only: bool):
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


def load_model_and_tokenizer(path_or_repo: str, device: str, dtype: Any, local_files_only: bool):
    resolved = resolve_checkpoint(path_or_repo)
    if Path(resolved).exists() and (Path(resolved) / "adapter_config.json").exists():
        peft_config = PeftConfig.from_pretrained(resolved, local_files_only=local_files_only)
        tokenizer = setup_tokenizer(resolved, local_files_only=local_files_only)
        base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model = PeftModel.from_pretrained(base, resolved, is_trainable=False)
    else:
        tokenizer = setup_tokenizer(resolved, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    model.to(device)
    model.eval()
    return model, tokenizer, resolved


def alti_model_view(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        if base is not None:
            base.eval()
            return base
    return model


def token_display(text: str) -> str:
    if text == "\n":
        return "\\n"
    if text == "\t":
        return "\\t"
    return text.replace("\n", "\\n").replace("\t", "\\t")


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def find_fim_code_ranges(prompt_text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    fp = "<|fim_prefix|>"
    fs = "<|fim_suffix|>"
    fm = "<|fim_middle|>"
    a = prompt_text.find(fp)
    b = prompt_text.find(fs, a + len(fp)) if a >= 0 else -1
    c = prompt_text.find(fm, b + len(fs)) if b >= 0 else -1
    if a >= 0 and b >= 0:
        ranges.append((a + len(fp), b))
    if b >= 0 and c >= 0:
        ranges.append((b + len(fs), c))
    return [(x, y) for x, y in ranges if y > x]


def classify_prompt_region(start: int, end: int, code_ranges: list[tuple[int, int]]) -> str:
    if any(overlaps(start, end, a, b) for a, b in code_ranges):
        return "prompt_code"
    return "prompt_other"


def build_prompt_tokens(prompt_text: str, tokenizer) -> tuple[list[int], list[dict[str, Any]], list[int], list[int]]:
    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("Tokenizer must support return_offsets_mapping for the saliency viewer.")
    code_ranges = find_fim_code_ranges(prompt_text)
    tokens: list[dict[str, Any]] = []
    prompt_code_indices: list[int] = []
    prompt_all_indices: list[int] = []
    for idx, (tid, (start, end)) in enumerate(zip(input_ids, offsets)):
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        start_i, end_i = int(start), int(end)
        if end_i < start_i:
            start_i, end_i = 0, 0
        region = classify_prompt_region(start_i, end_i, code_ranges)
        is_special = piece.startswith("<|") and piece.endswith("|>")
        tok = {
            "idx": idx,
            "id": int(tid),
            "text": piece,
            "display": token_display(piece),
            "char_start": start_i,
            "char_end": end_i,
            "region": region,
            "is_special": bool(is_special),
            "is_whitespace": bool(piece.strip() == ""),
        }
        tokens.append(tok)
        prompt_all_indices.append(idx)
        if region == "prompt_code":
            prompt_code_indices.append(idx)
    return input_ids, tokens, prompt_code_indices, prompt_all_indices


def make_generated_token(idx: int, token_id: int, tokenizer) -> dict[str, Any]:
    piece = tokenizer.decode([token_id], skip_special_tokens=False)
    is_special = piece.startswith("<|") and piece.endswith("|>")
    return {
        "idx": int(idx),
        "id": int(token_id),
        "text": piece,
        "display": token_display(piece),
        "char_start": None,
        "char_end": None,
        "region": "completion",
        "is_special": bool(is_special),
        "is_whitespace": bool(piece.strip() == ""),
    }


def topk_from_saliency(saliency: list[float], tokens: list[dict[str, Any]], candidate_indices: list[int], top_k: int) -> list[dict[str, Any]]:
    pairs = []
    n = len(saliency)
    for idx in candidate_indices:
        if 0 <= idx < n and idx < len(tokens):
            pairs.append((abs(float(saliency[idx])), idx))
    pairs.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for rank, (val, idx) in enumerate(pairs[:top_k], start=1):
        tok = tokens[idx]
        out.append({
            "rank": rank,
            "idx": int(idx),
            "token": tok["text"],
            "display": tok["display"],
            "region": tok["region"],
            "value": float(val),
        })
    return out


def should_stop_token(token_id: int, text: str, tokenizer) -> bool:
    eos_ids = {x for x in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")] if x is not None and x != -1}
    if int(token_id) in eos_ids:
        return True
    return "<|im_end|>" in text


def compute_for_sample(
    model: Any,
    alti_model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    top_k: int,
    max_new_tokens: int,
    max_seq_len: int,
    alti_p: int,
    alti_chunk_size: int,
    progress_label: str = "",
    progress_every: int = 10,
) -> dict[str, Any]:
    prompt_text = build_messages_prompt(row.get("messages", []))
    prompt_ids, tokens, prompt_code_indices, prompt_all_indices = build_prompt_tokens(prompt_text, tokenizer)
    if not prompt_ids:
        return {"skipped": True, "reason": "empty prompt"}
    if len(prompt_ids) >= max_seq_len:
        return {"skipped": True, "reason": f"prompt_len {len(prompt_ids)} >= max_seq_len {max_seq_len}"}

    device = next(model.parameters()).device
    input_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_token_indices: list[int] = []
    targets: dict[str, Any] = {}
    targets_by_step: dict[str, int] = {}

    for step in range(max_new_tokens):
        if progress_label and (step == 0 or (step + 1) % max(1, progress_every) == 0):
            print(f"{progress_label} token {step + 1}/{max_new_tokens}", flush=True)
        target_idx = int(input_t.size(1))
        if target_idx >= max_seq_len:
            break

        attention_mask = torch.ones_like(input_t, device=device)
        with torch.inference_mode():
            outputs = model(input_ids=input_t, attention_mask=attention_mask, use_cache=False, return_dict=True)
            next_id = int(torch.argmax(outputs.logits[0, -1, :]).item())
            del outputs

        batch = {"input_ids": input_t, "attention_mask": attention_mask}
        saliency = compute_alti_saliency_vector(
            alti_model,
            batch,
            target_idx_in_seq=target_idx,
            p=alti_p,
            chunk_size=alti_chunk_size,
        )

        next_tok = make_generated_token(target_idx, next_id, tokenizer)
        all_source_indices = list(range(target_idx))
        targets[str(target_idx)] = {
            "target_idx": target_idx,
            "query_idx": target_idx - 1,
            "step": int(step),
            "token": next_tok["text"],
            "display": next_tok["display"],
            "saliency_definition": "ALTI rollout contribution from src/attribution/saliency.py::compute_alti_saliency_vector",
            "scopes": {
                "prompt_code": topk_from_saliency(saliency, tokens, prompt_code_indices, top_k),
                "prompt_all": topk_from_saliency(saliency, tokens, prompt_all_indices, top_k),
                "all_causal": topk_from_saliency(saliency, tokens, all_source_indices, top_k),
            },
        }
        targets_by_step[str(step)] = target_idx
        generated_token_indices.append(target_idx)
        tokens.append(next_tok)

        input_t = torch.cat([input_t, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if should_stop_token(next_id, next_tok["text"], tokenizer):
            break

    generated_text = tokenizer.decode([tokens[i]["id"] for i in generated_token_indices], skip_special_tokens=False)
    return {
        "skipped": False,
        "prompt_len": len(prompt_ids),
        "seq_len": len(tokens),
        "num_generated_tokens_computed": len(generated_token_indices),
        "generated_text": generated_text,
        "tokens": tokens,
        "generated_token_indices": generated_token_indices,
        "targets_by_step": targets_by_step,
        "targets": targets,
    }


def build_output_payload(
    *,
    samples_in: list[dict[str, Any]],
    model_meta: list[dict[str, Any]],
    top_k: int,
    max_new_tokens: int,
    alti_p: int,
    alti_chunk_size: int,
    cache_dir: str | None,
) -> dict[str, Any]:
    samples_out: list[dict[str, Any]] = []
    for sample in samples_in:
        row = sample.get("row", {})
        samples_out.append({
            "sample_id": sample.get("sample_id"),
            "row_index": sample.get("row_index"),
            "uid": sample.get("uid"),
            "source_dataset": sample.get("source_dataset"),
            "language": sample.get("language"),
            "raw_id": sample.get("raw_id"),
            "selection": sample.get("selection", {}),
            "fim_prompt_head": str(row.get("fim_prompt", ""))[:400],
            "models": sample.get("_model_results", {}),
        })
    return {
        "version": 3,
        "generation_mode": "free-run greedy token-by-token",
        "saliency_definition": "Full ALTI rollout from src/attribution/saliency.py::compute_alti_saliency_vector. For each generated token, the model is run on the current prefix only; saliency is computed before appending that token.",
        "target_semantics": "target_idx is the generated token position; query_idx=target_idx-1 is the prefix state that predicts it.",
        "source_scopes": {
            "prompt_code": "Only stripped FIM prefix/suffix code tokens in the prompt.",
            "prompt_all": "All tokens in the ChatML prompt before assistant generation.",
            "all_causal": "All causal source tokens before the target, including prior generated tokens.",
        },
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "alti_p": alti_p,
        "alti_chunk_size": alti_chunk_size,
        "cache_dir": cache_dir,
        "models": model_meta,
        "samples": samples_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute free-run ALTI saliency top-k for the dynamic viewer.")
    parser.add_argument("--sample_manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model_spec", action="append", default=[], help="NAME=PATH_OR_REPO. Can be repeated.")
    parser.add_argument("--use_default_model_specs", action="store_true", help="Use benchmark baseline adapter dirs in HumanEval score order.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--max_seq_len", type=int, default=3072)
    parser.add_argument("--alti_p", type=int, default=1)
    parser.add_argument("--alti_chunk_size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--local_files_only", type=int, default=1)
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR), help="Per model/sample cache directory. Empty string disables cache.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Recompute even if a model/sample cache file exists.")
    parser.add_argument("--partial_write_every", type=int, default=1, help="Write output JSON after this many completed model/sample results. 0 writes only at the end.")
    parser.add_argument("--progress_every", type=int, default=10, help="Print token-level progress every N generated tokens.")
    args = parser.parse_args()

    global torch, AutoModelForCausalLM, AutoTokenizer, PeftConfig, PeftModel, compute_alti_saliency_vector
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftConfig, PeftModel
    from src.attribution.saliency import compute_alti_saliency_vector

    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    raw_specs = args.model_spec or [f"Ours GraphSignal={DEFAULT_OURS}"]
    if args.use_default_model_specs:
        raw_specs = DEFAULT_MODEL_SPECS
    specs = parse_model_specs(raw_specs)

    manifest = json.loads(Path(args.sample_manifest).read_text(encoding="utf-8"))
    samples_in = manifest.get("samples", [])
    model_meta: list[dict[str, Any]] = []
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    completed_since_write = 0
    for sample in samples_in:
        sample["_model_results"] = {}

    for spec in specs:
        print(f"\n=== Loading {spec['name']} from {spec['path']} ===", flush=True)
        try:
            model, tokenizer, resolved = load_model_and_tokenizer(spec["path"], args.device, dtype, bool(args.local_files_only))
            alti_model = alti_model_view(model)
        except Exception as exc:
            model_meta.append({"name": spec["name"], "path": spec["path"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] failed to load {spec['name']}: {exc}", flush=True)
            continue
        model_meta.append({"name": spec["name"], "path": spec["path"], "resolved_path": resolved})

        for idx, sample in enumerate(samples_in):
            row = sample.get("row")
            if row is None:
                raise ValueError("sample manifest must include row; rerun select script with --include_row")
            sid = sample["sample_id"]
            print(f"[{spec['name']}] sample {idx+1}/{len(samples_in)} {sid}", flush=True)
            result = None
            if not args.overwrite_cache:
                result = read_cached_result(cache_dir, spec["name"], sid)
                if result is not None:
                    print(f"[cache hit] {spec['name']} {sid}", flush=True)
            if result is None:
                try:
                    result = compute_for_sample(
                        model=model,
                        alti_model=alti_model,
                        tokenizer=tokenizer,
                        row=row,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        max_seq_len=args.max_seq_len,
                        alti_p=args.alti_p,
                        alti_chunk_size=args.alti_chunk_size,
                        progress_label=f"[{spec['name']}] {sid}",
                        progress_every=args.progress_every,
                    )
                except Exception as exc:
                    result = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}
                    print(f"[error] {spec['name']} {sid}: {exc}", flush=True)
                write_cached_result(cache_dir, spec["name"], sid, result)
                print(f"[cache write] {spec['name']} {sid}", flush=True)
            sample["_model_results"][spec["name"]] = result
            completed_since_write += 1
            if args.partial_write_every > 0 and completed_since_write >= args.partial_write_every:
                payload = build_output_payload(
                    samples_in=samples_in,
                    model_meta=model_meta,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    alti_p=args.alti_p,
                    alti_chunk_size=args.alti_chunk_size,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
                write_json_atomic(Path(args.output_path), payload, indent=2)
                print(f"[partial write] {args.output_path}", flush=True)
                completed_since_write = 0

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = build_output_payload(
        samples_in=samples_in,
        model_meta=model_meta,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        alti_p=args.alti_p,
        alti_chunk_size=args.alti_chunk_size,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    out = Path(args.output_path)
    write_json_atomic(out, payload, indent=2)
    print(f"\nsaved saliency data -> {out}")


if __name__ == "__main__":
    main()
