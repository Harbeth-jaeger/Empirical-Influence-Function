from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover - PEFT is available in the benchmark env.
    PeftConfig = None
    PeftModel = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TRAIN_PATH = "data/benchmarks/sft_data/rendered_chatml_fim_train.jsonl"
DEFAULT_EDGE_DATA_PATH = "data/benchmarks/sft_data/ours_graphsignal_train.json"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a correlation-report-compatible JSON file that visualizes "
            "prompt-token attention for every selected completion token."
        )
    )
    parser.add_argument("--data_path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument(
        "--edge_data_path",
        default=DEFAULT_EDGE_DATA_PATH,
        help=(
            "Optional JSONL/JSON-lines file that contains tokenizer-aligned "
            "attention_edges for the same sample order. Use an empty string to disable."
        ),
    )
    parser.add_argument("--sample_index", type=int, required=True)
    parser.add_argument("--model_path", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--report_index",
        type=int,
        default=None,
        help=(
            "Index used in correlation_matching_results_test{N}_all_tokens.json. "
            "Use different values for base vs SFT to avoid overwriting."
        ),
    )
    parser.add_argument("--model_label", default=None)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_output_tokens", type=int, default=80)
    parser.add_argument("--top_k_prompt", type=int, default=8)
    parser.add_argument("--last_n_layers", type=int, default=4)
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl_row(path: str | Path, index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"sample_index={index} is out of range for {path}")


def read_optional_jsonl_row(path: str | Path | None, index: int) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[warn] edge_data_path does not exist; skip annotation edges: {p}")
        return None
    try:
        return read_jsonl_row(p, index)
    except Exception as exc:
        print(f"[warn] failed to read edge_data_path={p}: {exc}")
        return None


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(args: argparse.Namespace):
    model_path = Path(args.model_path)
    dtype = resolve_dtype(args.torch_dtype)

    if model_path.exists() and (model_path / "adapter_config.json").exists():
        if PeftConfig is None or PeftModel is None:
            raise ImportError("peft is required to load LoRA adapter checkpoints.")
        peft_config = PeftConfig.from_pretrained(args.model_path, local_files_only=args.local_files_only)
        base_model_name = peft_config.base_model_name_or_path
        config = AutoConfig.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
            attn_implementation="eager",
            output_attentions=True,
            use_cache=False,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            config=config,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        model = PeftModel.from_pretrained(base_model, args.model_path, is_trainable=False).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        return model, tokenizer

    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
        output_attentions=True,
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    return model, tokenizer


def decode_for_training_chatml(text: str) -> str:
    """Mirror src.sft.binarize_data.chatml_format_preprocess escaping.

    The benchmark JSON can contain a mix of real newlines and escaped ``\\n``.
    Training data conversion unconditionally replaces escaped newlines/tabs, so
    the visualization report must do the same to stay token-index aligned with
    ``attention_edges``.
    """
    return text.replace("\\n", "\n").replace("\\t", "\t")


def build_chatml_text(row: dict[str, Any]) -> tuple[str, str, str]:
    messages = row.get("messages") or []
    if len(messages) < 3:
        raise ValueError("Expected messages with system/user/assistant turns.")

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
        assistant_content = decode_for_training_chatml(str(row.get("fim_completion", "")))

    completion = assistant_content + "<|im_end|>\n"
    return prompt, completion, prompt + completion


def token_strings(tokenizer: AutoTokenizer, input_ids: torch.Tensor) -> list[str]:
    return tokenizer.convert_ids_to_tokens(input_ids.detach().cpu().tolist())


def load_annotation_edges(
    *,
    edge_row: dict[str, Any] | None,
    full_ids: torch.Tensor,
    prompt_len: int,
    tokens: list[str],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Group tokenizer-aligned annotation edges by completion target token."""
    meta = {
        "edge_data_loaded": bool(edge_row),
        "edge_data_aligned": False,
        "edge_count": 0,
        "prompt_to_completion_edge_count": 0,
    }
    if not edge_row:
        return {}, meta

    edge_input_ids = edge_row.get("input_ids") or edge_row.get("input_id") or []
    current_ids = full_ids.detach().cpu().tolist()
    if edge_input_ids and edge_input_ids[: len(current_ids)] != current_ids:
        print("[warn] edge_data_path input_ids are not aligned with the tokenized report sample; skip annotation edges.")
        return {}, meta

    meta["edge_data_aligned"] = True
    raw_edges = edge_row.get("attention_edges") or []
    meta["edge_count"] = len(raw_edges)

    parents_by_dst: dict[int, list[dict[str, Any]]] = {}
    for edge in raw_edges:
        try:
            src = int(edge.get("src", edge.get("token_i_idx", -1)))
            dst = int(edge.get("dst", edge.get("token_j_idx", -1)))
        except (TypeError, ValueError):
            continue
        if not (0 <= src < prompt_len <= dst < len(tokens)):
            continue
        if src >= dst:
            continue
        item = {
            "source_token_index": src,
            "source_token": tokens[src],
            "target_token_index": dst,
            "target_token": tokens[dst],
            "subtype": str(edge.get("subtype", edge.get("reason", "")) or ""),
        }
        parents_by_dst.setdefault(dst, []).append(item)

    meta["prompt_to_completion_edge_count"] = sum(len(v) for v in parents_by_dst.values())
    return parents_by_dst, meta


def is_trivial_token(tokenizer: AutoTokenizer, token_id: int) -> bool:
    text = tokenizer.decode([token_id])
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in {"{", "}", "(", ")", "[", "]", ",", ";"}:
        return True
    return len(stripped) == 1 and not (stripped.isalnum() or stripped == "_")


def model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


@torch.inference_mode()
def build_report(args: argparse.Namespace) -> dict[str, Any]:
    row = read_jsonl_row(args.data_path, args.sample_index)
    model, tokenizer = load_model_and_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_text, _completion_text, full_text = build_chatml_text(row)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    if full_ids.numel() > args.max_length:
        full_ids = full_ids[: args.max_length]
    prompt_len = min(int(prompt_ids.numel()), int(full_ids.numel()))
    if prompt_len >= int(full_ids.numel()):
        raise ValueError(
            "The selected sample has no completion tokens after truncation. "
            "Increase --max_length or choose a shorter sample."
        )

    input_ids = full_ids.unsqueeze(0).to(model_input_device(model))
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        use_cache=False,
    )
    if not outputs.attentions:
        raise RuntimeError(
            "Model did not return attentions. Make sure attn_implementation='eager' is supported."
        )

    selected_attn = outputs.attentions[-args.last_n_layers :]
    attn = torch.stack([x.detach().float().cpu() for x in selected_attn], dim=0).mean(dim=(0, 2))[0]
    tokens = token_strings(tokenizer, full_ids)
    edge_row = read_optional_jsonl_row(args.edge_data_path, args.sample_index)
    annotation_parents_by_dst, edge_meta = load_annotation_edges(
        edge_row=edge_row,
        full_ids=full_ids,
        prompt_len=prompt_len,
        tokens=tokens,
    )

    max_end = min(int(full_ids.numel()), prompt_len + args.max_output_tokens)
    per_token_results: list[dict[str, Any]] = []
    for target_idx in range(prompt_len, max_end):
        token_id = int(full_ids[target_idx].item())
        if is_trivial_token(tokenizer, token_id):
            continue

        prompt_scores = attn[target_idx, :prompt_len]
        k = min(args.top_k_prompt, prompt_scores.numel())
        if k <= 0:
            continue
        top_values, top_indices = torch.topk(prompt_scores, k=k)
        prompt_attention_mass = float(prompt_scores.sum().item())

        target_token = tokens[target_idx]
        annotation_parents = annotation_parents_by_dst.get(target_idx, [])
        annotation_source_set = {int(parent["source_token_index"]) for parent in annotation_parents}
        top_correlations = []
        for value, source_idx in zip(top_values.tolist(), top_indices.tolist(), strict=True):
            top_correlations.append(
                {
                    "source_token_index": int(source_idx),
                    "source_token": tokens[int(source_idx)],
                    "target_token_index": int(target_idx),
                    "target_token": target_token,
                    "saliency_score": float(value),
                    "raw_attention": float(value),
                    "has_annotation_edge": int(source_idx) in annotation_source_set,
                }
            )

        per_token_results.append(
            {
                "target_token_index": int(target_idx),
                "target_token": target_token,
                "prompt_attention_mass": prompt_attention_mass,
                "annotation_parents": annotation_parents,
                "top_correlations": top_correlations,
                "correlation_pairs": [],
            }
        )

    report_index = args.report_index if args.report_index is not None else args.sample_index
    model_label = args.model_label or args.model_path
    report = {
        "experiment_meta": {
            "test_sample_index": int(report_index),
            "source_sample_index": int(args.sample_index),
            "mode": "all_tokens",
            "report_type": "prompt_attention",
            "model_label": model_label,
            "tokens_analyzed": len(per_token_results),
            "config": {
                "data_path": args.data_path,
                "edge_data_path": args.edge_data_path,
                "model_path": args.model_path,
                "max_length": args.max_length,
                "max_output_tokens": args.max_output_tokens,
                "top_k_prompt": args.top_k_prompt,
                "last_n_layers": args.last_n_layers,
                "edge_meta": edge_meta,
            },
        },
        "test_sample_baseline": {
            "full_tokens": tokens,
            "correct_full_tokens": tokens,
            "prompt_len": int(prompt_len),
        },
        "per_token_results": per_token_results,
        "train_sample_details": {},
    }
    return round_floats(report, ndigits=6)


def round_floats(obj: Any, ndigits: int) -> Any:
    if isinstance(obj, float):
        if math.isfinite(obj):
            return round(obj, ndigits)
        return obj
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_index = args.report_index if args.report_index is not None else args.sample_index
    output_path = output_dir / f"correlation_matching_results_test{report_index}_all_tokens.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(
        "Put this file under the repository root if you want tools/correlation-report "
        "to discover it automatically."
    )


if __name__ == "__main__":
    main()
