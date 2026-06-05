#!/usr/bin/env python3
"""Export visual saliency data as per-model/per-sample legacy files.

The output layout is:

  output_dir/
    manifest.json
    ours_graphsignal/
      manifest.json
      python_00_mceval_instruct_999/latest_saliency.json
      ...
    clear/
      ...

Each ``latest_saliency.json`` keeps the single-model legacy schema used by the
historical alti-correlation-matching viewer, while the directory layout carries
the seven-model comparison.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/viz_saliency/saliency_comparison_data.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/viz_saliency/legacy_by_model_sample"


def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
    return s or "sample"


def model_slug(model_name: str) -> str:
    s = model_name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "model"


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
        target_idx = int(target.get("target_idx", target.get("index")))
        saliency_list.append(
            {
                "index": target_idx,
                "target_token": target.get("token", ""),
                "target_display": target.get("display", target.get("token", "")),
                "query_idx": target.get("query_idx"),
                "step": target.get("step"),
                "saliency": sparse_vector(target, len(full_tokens), scope),
                "top_sources": target.get("scopes", {}).get(scope, []),
            }
        )
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
    parser.add_argument("--scope", default="all_causal", choices=["prompt_code", "prompt_all", "all_causal"])
    parser.add_argument("--max_targets", type=int, default=0, help="0 exports all stored generated targets.")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    data = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_order = [m.get("name") for m in data.get("models", []) if m.get("name")]
    manifest = {
        "format": "legacy_by_model_sample.latest_saliency",
        "source_data_path": str(args.input_path),
        "scope": args.scope,
        "sparse_topk_export": True,
        "model_order": model_order,
        "models": [],
        "skipped": [],
    }

    for model_name in model_order:
        mslug = model_slug(model_name)
        model_dir = out_dir / mslug
        model_dir.mkdir(parents=True, exist_ok=True)
        model_manifest = {
            "model_name": model_name,
            "model_slug": mslug,
            "scope": args.scope,
            "sparse_topk_export": True,
            "samples": [],
            "skipped": [],
        }

        for sample in data.get("samples", []) or []:
            sid = str(sample.get("sample_id") or sample.get("uid") or sample.get("row_index"))
            sample_models = sample.get("models", {}) or {}
            model_result = sample_models.get(model_name)
            if not model_result:
                item = {"sample_id": sid, "model_name": model_name, "reason": "missing model"}
                manifest["skipped"].append(item)
                model_manifest["skipped"].append(item)
                continue
            if model_result.get("skipped"):
                item = {"sample_id": sid, "model_name": model_name, "reason": model_result.get("reason", "skipped")}
                manifest["skipped"].append(item)
                model_manifest["skipped"].append(item)
                continue

            tokens = model_result.get("tokens", []) or sample.get("tokens", []) or []
            targets = sorted_targets(model_result)
            if not tokens or not targets:
                item = {"sample_id": sid, "model_name": model_name, "reason": "missing tokens or targets"}
                manifest["skipped"].append(item)
                model_manifest["skipped"].append(item)
                continue

            block = legacy_block(tokens, targets, args.scope, args.max_targets)
            payload = {
                "format": "legacy_by_sample.latest_saliency",
                "model_name": model_name,
                "model_slug": mslug,
                "sample_id": sid,
                "row_index": sample.get("row_index"),
                "uid": sample.get("uid"),
                "source_dataset": sample.get("source_dataset"),
                "language": sample.get("language"),
                "raw_id": sample.get("raw_id"),
                "related_train_samples": [
                    {
                        "model_name": model_name,
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

            sample_dir = model_dir / slugify(sid)
            sample_dir.mkdir(parents=True, exist_ok=True)
            path = sample_dir / "latest_saliency.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=args.indent), encoding="utf-8")
            model_manifest["samples"].append(
                {
                    "sample_id": sid,
                    "row_index": sample.get("row_index"),
                    "language": sample.get("language"),
                    "uid": sample.get("uid"),
                    "path": str(path),
                    "num_tokens": len(block["full_tokens"]),
                    "num_saliency_items": len(block["saliency_list"]),
                }
            )

        (model_dir / "manifest.json").write_text(
            json.dumps(model_manifest, ensure_ascii=False, indent=args.indent),
            encoding="utf-8",
        )
        manifest["models"].append(
            {
                "model_name": model_name,
                "model_slug": mslug,
                "path": str(model_dir),
                "num_samples": len(model_manifest["samples"]),
                "num_skipped": len(model_manifest["skipped"]),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=args.indent), encoding="utf-8")
    print(f"wrote {len(manifest['models'])} model folders -> {out_dir}")
    for item in manifest["models"]:
        print(f"- {item['model_name']}: {item['num_samples']} samples -> {item['model_slug']}")
    if manifest["skipped"]:
        print(f"skipped {len(manifest['skipped'])} sample/model entries")


if __name__ == "__main__":
    main()
