#!/usr/bin/env python3
"""Export visual-saliency data as per-sample legacy latest_saliency.json files.

This is stricter than export_intervention_saliency_format.py.  It writes one
sample directory at a time and keeps the JSON shape close to the historical
alti-correlation-matching results/<sample>/latest_saliency.json schema:

{
  "related_train_samples": [
    {
      "target_idx": int,
      "before_original": {
        "full_tokens": [str, ...],
        "start_index": int,
        "saliency_list": [{"index": int, "saliency": [float, ...]}, ...]
      },
      "before_generation": {...}
    }
  ],
  "overfit_test_results": [],
  "target_test_sample": {"before": {...}, "after": {...}}
}

The current visual-saliency data stores top-k sources only, so the exported
saliency arrays are sparse: non-top-k positions are filled with 0.0.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/visual_saliency/saliency_comparison_data.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/visual_saliency/legacy_by_sample"


def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
    return s or "sample"


def token_text(token: dict[str, Any]) -> str:
    return str(token.get("text", token.get("token", token.get("display", ""))))


def sorted_targets(model_result: dict[str, Any]) -> list[dict[str, Any]]:
    targets = list((model_result.get("targets", {}) or {}).values())
    targets.sort(key=lambda x: (int(x.get("step", 10**9)), int(x.get("target_idx", x.get("index", 10**9)))))
    return targets


def sparse_vector(target: dict[str, Any], seq_len: int, scope: str) -> list[float]:
    vec = [0.0] * seq_len
    for item in target.get("scopes", {}).get(scope, []) or []:
        idx = int(item.get("idx", -1))
        if 0 <= idx < seq_len:
            vec[idx] = float(item.get("value", item.get("saliency", 0.0)))
    return vec


def legacy_block(tokens: list[dict[str, Any]], targets: list[dict[str, Any]], scope: str, max_targets: int) -> dict[str, Any]:
    full_tokens = [token_text(t) for t in tokens]
    if max_targets > 0:
        targets = targets[:max_targets]
    saliency_list = []
    for target in targets:
        saliency_list.append({
            "index": int(target.get("target_idx", target.get("index"))),
            "saliency": sparse_vector(target, len(full_tokens), scope),
        })
    start_index = int(saliency_list[0]["index"]) if saliency_list else 0
    return {
        "full_tokens": full_tokens,
        "start_index": start_index,
        "saliency_list": saliency_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_name", default="Ours GraphSignal", help="Model to export per sample.")
    parser.add_argument("--scope", default="all_causal", choices=["prompt_code", "prompt_all", "all_causal"])
    parser.add_argument("--max_targets", type=int, default=0, help="0 exports all stored generated targets.")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    data = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "legacy_by_sample.latest_saliency",
        "source_data_path": str(args.input_path),
        "model_name": args.model_name,
        "scope": args.scope,
        "sparse_topk_export": True,
        "samples": [],
        "skipped": [],
    }

    for sample in data.get("samples", []) or []:
        sid = str(sample.get("sample_id") or sample.get("uid") or sample.get("row_index"))
        model_result = (sample.get("models", {}) or {}).get(args.model_name)
        if not model_result:
            manifest["skipped"].append({"sample_id": sid, "reason": f"missing model {args.model_name}"})
            continue
        if model_result.get("skipped"):
            manifest["skipped"].append({"sample_id": sid, "reason": model_result.get("reason", "skipped")})
            continue
        tokens = model_result.get("tokens", []) or sample.get("tokens", []) or []
        targets = sorted_targets(model_result)
        if not tokens or not targets:
            manifest["skipped"].append({"sample_id": sid, "reason": "missing tokens or targets"})
            continue

        block = legacy_block(tokens, targets, args.scope, args.max_targets)
        payload = {
            "related_train_samples": [
                {
                    "target_idx": block["start_index"],
                    "before_original": block,
                    "before_generation": block,
                }
            ],
            "overfit_test_results": [],
            "target_test_sample": {
                "before": block,
                "after": block,
            },
        }

        sample_dir = out_dir / slugify(sid)
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = sample_dir / "latest_saliency.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=args.indent), encoding="utf-8")
        manifest["samples"].append({
            "sample_id": sid,
            "row_index": sample.get("row_index"),
            "language": sample.get("language"),
            "uid": sample.get("uid"),
            "path": str(path),
            "num_tokens": len(block["full_tokens"]),
            "num_saliency_items": len(block["saliency_list"]),
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=args.indent), encoding="utf-8")
    print(f"wrote {len(manifest['samples'])} per-sample files -> {out_dir}")
    if manifest["skipped"]:
        print(f"skipped {len(manifest['skipped'])} samples")


if __name__ == "__main__":
    main()
