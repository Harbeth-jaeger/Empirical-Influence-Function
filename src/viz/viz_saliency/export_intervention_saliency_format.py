#!/usr/bin/env python3
"""Export visual-saliency data to the legacy intervention_experiment saliency format.

The reference schema is the latest_saliency.json style produced on the
alti-correlation-matching branch:

{
  "related_train_samples": [
    {
      "target_idx": int,
      "before_original": {
        "full_tokens": [str, ...],
        "start_index": int,
        "saliency_list": [
          {"index": int, "saliency": [float, ...]}, ...
        ]
      }
    }
  ],
  "target_test_sample": {...}
}

Our current saliency_comparison_data.json stores only top-k source tokens for
front-end use, not full dense ALTI vectors. This exporter therefore emits a
legacy-compatible sparse vector: source positions outside the selected top-k
scope are filled with 0.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/viz_saliency/saliency_comparison_data.json"
DEFAULT_OUTPUT = ROOT / "outputs/viz_saliency/saliency_intervention_format.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--scope",
        default="all_causal",
        choices=["prompt_code", "prompt_all", "all_causal"],
        help="Which stored top-k scope to project into the legacy saliency vector.",
    )
    parser.add_argument(
        "--model_name",
        default="",
        help="Optional single model name to export. Empty exports every available model.",
    )
    parser.add_argument(
        "--max_targets",
        type=int,
        default=0,
        help="Optional max generated targets per sample/model. 0 exports all stored targets.",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def token_text(token: dict[str, Any]) -> str:
    return str(token.get("text", token.get("token", token.get("display", ""))))


def make_sparse_saliency_vector(
    *,
    target: dict[str, Any],
    seq_len: int,
    scope: str,
) -> list[float]:
    vec = [0.0] * seq_len
    for item in target.get("scopes", {}).get(scope, []) or []:
        idx = item.get("idx")
        if idx is None:
            continue
        idx = int(idx)
        if 0 <= idx < seq_len:
            vec[idx] = float(item.get("value", item.get("saliency", 0.0)))
    return vec


def sorted_targets(model_result: dict[str, Any]) -> list[dict[str, Any]]:
    targets = model_result.get("targets", {}) or {}
    rows = list(targets.values())
    rows.sort(key=lambda x: (int(x.get("step", 10**9)), int(x.get("target_idx", x.get("index", 10**9)))))
    return rows


def convert_model_sample(
    *,
    sample: dict[str, Any],
    model_name: str,
    model_result: dict[str, Any],
    scope: str,
    max_targets: int,
) -> dict[str, Any] | None:
    if model_result.get("skipped"):
        return {
            "sample_id": sample.get("sample_id"),
            "row_index": sample.get("row_index"),
            "uid": sample.get("uid"),
            "language": sample.get("language"),
            "model_name": model_name,
            "skipped": True,
            "reason": model_result.get("reason", "skipped"),
        }

    tokens = model_result.get("tokens", []) or sample.get("tokens", []) or []
    full_tokens = [token_text(t) for t in tokens]
    seq_len = len(full_tokens)
    targets = sorted_targets(model_result)
    if max_targets > 0:
        targets = targets[:max_targets]
    if not targets or seq_len == 0:
        return None

    saliency_list = []
    for target in targets:
        target_idx = int(target.get("target_idx", target.get("index")))
        saliency_list.append({
            "index": target_idx,
            "target_token": target.get("token", ""),
            "target_display": target.get("display", target.get("token", "")),
            "query_idx": target.get("query_idx"),
            "step": target.get("step"),
            "saliency": make_sparse_saliency_vector(target=target, seq_len=seq_len, scope=scope),
            "top_sources": target.get("scopes", {}).get(scope, []),
        })

    start_index = int(saliency_list[0]["index"])
    block = {
        "full_tokens": full_tokens,
        "start_index": start_index,
        "saliency_list": saliency_list,
    }
    return {
        "sample_id": sample.get("sample_id"),
        "row_index": sample.get("row_index"),
        "uid": sample.get("uid"),
        "source_dataset": sample.get("source_dataset"),
        "language": sample.get("language"),
        "raw_id": sample.get("raw_id"),
        "model_name": model_name,
        "target_idx": start_index,
        "before_original": block,
        "before_generation": block,
    }


def main() -> None:
    args = parse_args()
    src = Path(args.input_path)
    data = json.loads(src.read_text(encoding="utf-8"))

    related = []
    skipped = []
    for sample in data.get("samples", []) or []:
        for model_name, model_result in (sample.get("models", {}) or {}).items():
            if args.model_name and model_name != args.model_name:
                continue
            converted = convert_model_sample(
                sample=sample,
                model_name=model_name,
                model_result=model_result,
                scope=args.scope,
                max_targets=args.max_targets,
            )
            if converted is None:
                skipped.append({
                    "sample_id": sample.get("sample_id"),
                    "model_name": model_name,
                    "reason": "no targets or no tokens",
                })
            elif converted.get("skipped"):
                skipped.append(converted)
            else:
                related.append(converted)

    target_test_sample = {}
    if related:
        first = related[0]["before_original"]
        target_test_sample = {"before": first, "after": first}

    out = {
        "format": "intervention_experiment.latest_saliency.compat",
        "format_reference": "alti-correlation-matching:src/NIF.py latest_saliency.json",
        "source_data_path": str(src),
        "source_data_version": data.get("version"),
        "source_generation_mode": data.get("generation_mode"),
        "saliency_definition": data.get("saliency_definition"),
        "scope": args.scope,
        "sparse_topk_export": True,
        "note": "Current viz_saliency data stores top-k sources only; saliency vectors are zero-filled except stored top-k source positions.",
        "target_test_sample": target_test_sample,
        "related_train_samples": related,
        "skipped": skipped,
    }

    dst = Path(args.output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=args.indent), encoding="utf-8")
    print(f"converted {len(related)} sample/model entries -> {dst}")
    if skipped:
        print(f"skipped {len(skipped)} entries")


if __name__ == "__main__":
    main()
