#!/usr/bin/env python
"""Build a standalone dynamic HTML viewer for annotation edges and attention top-k."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover
    PeftConfig = None
    PeftModel = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SUBTYPE_COLORS: dict[str, str] = {
    "bracket": "#64748b",
    "defuse": "#0ea5e9",
    "call": "#f97316",
    "return": "#ef4444",
    "type": "#22c55e",
    "dataflow": "#fb923c",
    "semantic": "#a855f7",
    "api": "#ec4899",
    "": "#94a3b8",
}

DEFAULT_EDGE_DATA_PATH = "data/benchmarks/sft_data/ours_graphsignal_train.json"
DEFAULT_SAMPLES_CSV = "outputs/viz_annotation/visualization/annotation_rich_samples.csv"
DEFAULT_TEMPLATE_PATH = "tools/viz_annotation/dynamic_annotation_viewer.html"
DEFAULT_OUTPUT_PATH = "outputs/viz_annotation/visualization/dynamic_annotation_viewer.html"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge_data_path", default=DEFAULT_EDGE_DATA_PATH)
    parser.add_argument("--samples_csv", default=DEFAULT_SAMPLES_CSV)
    parser.add_argument("--sample_indices", nargs="+", type=int, default=None)
    parser.add_argument("--top_n_from_csv", type=int, default=8)
    parser.add_argument("--model_path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--template_path", default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--with_attention", action="store_true")
    parser.add_argument("--top_k_prompt", type=int, default=5)
    parser.add_argument("--last_n_layers", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


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


def add_visual_mask_newlines(tokens: list[dict[str, Any]]) -> None:
    """Make code [MASK] visually occupy the hidden line it represents.

    The actual prompt stores the suffix immediately after [MASK] when the hidden
    span contains the newline.  For inspection, show a line break after the code
    mask if the next token is indentation-only whitespace.  This changes only
    the HTML display, not token ids or annotation indices.
    """
    for i, tok in enumerate(tokens):
        if "ASK" not in str(tok.get("text", "")):
            continue
        close = None
        for j in range(i, min(i + 4, len(tokens))):
            if "]" in str(tokens[j].get("text", "")):
                close = j
                break
        if close is None or close + 1 >= len(tokens):
            continue
        next_text = str(tokens[close + 1].get("text", ""))
        if next_text and next_text.strip() == "" and "\n" not in next_text:
            tokens[close]["visual_text"] = str(tokens[close].get("text", "")) + "\n"
            tokens[close]["visual_display"] = compact_token_text(tokens[close]["visual_text"])


def read_jsonl_rows(path: Path, indices: list[int]) -> dict[int, dict[str, Any]]:
    wanted = set(indices)
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in wanted:
                rows[idx] = json.loads(line)
                if len(rows) == len(wanted):
                    break
    missing = sorted(wanted - rows.keys())
    if missing:
        raise ValueError(f"Sample indices not found in {path}: {missing}")
    return rows


def load_csv_metadata(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                idx = int(row["idx"])
            except (KeyError, TypeError, ValueError):
                continue
            out[idx] = row
    return out


def choose_sample_indices(args: argparse.Namespace, csv_meta: dict[int, dict[str, Any]]) -> list[int]:
    if args.sample_indices:
        return list(dict.fromkeys(args.sample_indices))
    if csv_meta:
        return list(csv_meta.keys())[: max(1, args.top_n_from_csv)]
    raise ValueError("Provide --sample_indices or a readable --samples_csv.")


def infer_prompt_len(row: dict[str, Any], meta: dict[str, Any] | None) -> int:
    if meta and meta.get("prompt_len"):
        try:
            return int(meta["prompt_len"])
        except (TypeError, ValueError):
            pass

    labels = row.get("label", row.get("labels")) or []
    segments: list[tuple[int, int]] = []
    start = prev = None
    for i, value in enumerate(labels):
        if value == -100:
            continue
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = prev = i
    if start is not None:
        segments.append((start, prev))

    long_segments = [(s, e) for s, e in segments if e - s + 1 >= 5]
    if long_segments:
        return int(long_segments[-1][0])

    edges = row.get("attention_edges") or []
    dsts: list[int] = []
    for edge in edges:
        try:
            dst = int(edge.get("dst", edge.get("token_j_idx", -1)))
        except (TypeError, ValueError):
            continue
        if 0 <= dst < len(labels) and (not labels or labels[dst] != -100):
            dsts.append(dst)
    if dsts:
        return min(dsts)

    return max(0, len(row.get("input_ids", [])) - 1)


def normalize_edges(row: dict[str, Any], seq_len: int) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for edge in row.get("attention_edges") or []:
        if not isinstance(edge, dict):
            continue
        try:
            src = int(edge.get("src", edge.get("token_i_idx", -1)))
            dst = int(edge.get("dst", edge.get("token_j_idx", -1)))
        except (TypeError, ValueError):
            continue
        subtype = str(edge.get("subtype", edge.get("reason", "")) or "")
        if not (0 <= src < seq_len and 0 <= dst < seq_len and src < dst):
            continue
        key = (src, dst, subtype)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "source": src,
                "target": dst,
                "subtype": subtype or "unknown",
                "color": SUBTYPE_COLORS.get(subtype, SUBTYPE_COLORS[""]),
            }
        )
    return edges


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


def model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


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


def load_tokenizer_for_model(args: argparse.Namespace):
    tokenizer_path = args.model_path
    model_path = Path(args.model_path)
    if model_path.exists() and (model_path / "adapter_config.json").exists():
        if PeftConfig is not None:
            peft_config = PeftConfig.from_pretrained(args.model_path, local_files_only=args.local_files_only)
            tokenizer_path = peft_config.base_model_name_or_path
        else:
            adapter_config = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
            tokenizer_path = adapter_config.get("base_model_name_or_path", args.model_path)

    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )


@torch.inference_mode()
def compute_attention_topk(
    *,
    model: torch.nn.Module,
    input_ids: list[int],
    prompt_len: int,
    completion_indices: list[int],
    top_k: int,
    last_n_layers: int,
) -> dict[str, list[dict[str, Any]]]:
    device = model_input_device(model)
    ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    outputs = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        output_attentions=True,
        use_cache=False,
    )
    if not outputs.attentions:
        raise RuntimeError("Model did not return attentions.")
    selected = outputs.attentions[-last_n_layers:] if 0 < last_n_layers <= len(outputs.attentions) else outputs.attentions
    attn = torch.stack([x.detach().float().cpu() for x in selected], dim=0).mean(dim=(0, 2))[0]
    prompt_len = min(prompt_len, len(input_ids))
    result: dict[str, list[dict[str, Any]]] = {}
    for dst in completion_indices:
        if dst <= 0 or dst >= attn.size(0):
            continue
        source_limit = min(prompt_len, dst)
        if source_limit <= 0:
            continue
        scores = attn[dst, :source_limit]
        k = min(top_k, scores.numel())
        if k <= 0:
            continue
        values, indices = torch.topk(scores, k=k)
        result[str(dst)] = [
            {"source": int(src), "score": round(float(score), 6)}
            for score, src in zip(values.tolist(), indices.tolist(), strict=True)
        ]
    return result


def build_sample_payload(
    *,
    sample_idx: int,
    row: dict[str, Any],
    meta: dict[str, Any] | None,
    tokenizer: Any,
    attention_topk: dict[str, list[dict[str, Any]]] | None,
    max_length: int,
) -> dict[str, Any]:
    input_ids = list(row.get("input_ids") or row.get("input_id") or [])
    labels = list(row.get("label", row.get("labels")) or [-100] * len(input_ids))
    if not input_ids:
        raise ValueError(f"sample {sample_idx} has no input_ids")

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    prompt_len = min(infer_prompt_len(row, meta), len(input_ids))
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)
    display_texts = [decode_token_for_display(t) for t in token_strings]
    tokens = [
        {
            "idx": i,
            "text": display_texts[i],
            "display": compact_token_text(display_texts[i]),
            "is_completion": i >= prompt_len,
        }
        for i in range(len(input_ids))
    ]
    add_visual_mask_newlines(tokens)
    edges = normalize_edges({"attention_edges": row.get("attention_edges") or []}, len(input_ids))

    if attention_topk:
        for rows in attention_topk.values():
            for item in rows:
                src = int(item["source"])
                item["token"] = tokens[src]["display"] if 0 <= src < len(tokens) else str(src)

    payload = {
        "sample_index": sample_idx,
        "source_dataset": (meta or {}).get("source_dataset", ""),
        "language": (meta or {}).get("language", ""),
        "raw_id": (meta or {}).get("raw_id", ""),
        "prompt_len": prompt_len,
        "tokens": tokens,
        "edges": edges,
        "attention_topk": attention_topk or {},
    }
    return round_floats(payload, ndigits=6)


def round_floats(obj: Any, ndigits: int) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits) if math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def main() -> None:
    args = parse_args()
    edge_path = Path(args.edge_data_path)
    csv_meta = load_csv_metadata(Path(args.samples_csv))
    sample_indices = choose_sample_indices(args, csv_meta)
    rows = read_jsonl_rows(edge_path, sample_indices)

    model = None
    if args.with_attention:
        model, tokenizer = load_model_and_tokenizer(args)
    else:
        tokenizer = load_tokenizer_for_model(args)

    samples: list[dict[str, Any]] = []
    for sample_idx in sample_indices:
        row = rows[sample_idx]
        meta = csv_meta.get(sample_idx)
        input_ids = list(row.get("input_ids") or row.get("input_id") or [])[: args.max_length]
        prompt_len = min(infer_prompt_len(row, meta), len(input_ids))
        completion_indices = list(range(prompt_len, len(input_ids)))
        attention_topk = None
        if model is not None:
            attention_topk = compute_attention_topk(
                model=model,
                input_ids=input_ids,
                prompt_len=prompt_len,
                completion_indices=completion_indices,
                top_k=args.top_k_prompt,
                last_n_layers=args.last_n_layers,
            )
        sample = build_sample_payload(
            sample_idx=sample_idx,
            row=row,
            meta=meta,
            tokenizer=tokenizer,
            attention_topk=attention_topk,
            max_length=args.max_length,
        )
        samples.append(sample)
        print(
            f"sample={sample_idx} tokens={len(sample['tokens'])} edges={len(sample['edges'])} "
            f"attention_targets={len(sample['attention_topk'])}"
        )

    viewer_data = {
        "config": {
            "edge_data_path": args.edge_data_path,
            "samples_csv": args.samples_csv,
            "model_path": args.model_path,
            "with_attention": bool(args.with_attention),
            "top_k_prompt": args.top_k_prompt,
            "last_n_layers": args.last_n_layers,
            "max_length": args.max_length,
        },
        "subtype_colors": SUBTYPE_COLORS,
        "samples": samples,
    }

    template = Path(args.template_path).read_text(encoding="utf-8")
    data_json = json.dumps(viewer_data, ensure_ascii=False)
    html = template.replace("__VIEWER_DATA__", data_json.replace("</script", "<\\/script"))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
