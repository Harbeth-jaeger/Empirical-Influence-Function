#!/usr/bin/env python3
"""Merge rich annotation samples with Base/Ours saliency for the annotation viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich_data", default="outputs/visual_annotation/rich_top_edges/viewer_data.json")
    parser.add_argument("--saliency_data", default="outputs/visual_saliency/rich_top_edges_saliency_data.json")
    parser.add_argument("--output_path", default="outputs/visual_saliency/base_vs_ours_annotation_saliency_data_v2.json")
    parser.add_argument("--scope", default="all_causal", choices=["all_causal", "prompt_code", "prompt_all"])
    return parser.parse_args()


def model_result(sample: dict[str, Any], preferred: str) -> dict[str, Any]:
    models = sample.get("models", {})
    for key in models:
        if preferred.lower() in key.lower():
            return models[key]
    return {}


def compact_model(result: dict[str, Any], scope: str) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for key, target in (result.get("targets") or {}).items():
        out = dict(target)
        out["sources"] = list((target.get("scopes") or {}).get(scope) or (target.get("scopes") or {}).get("all_causal") or [])
        targets[str(key)] = out
    return {
        "tokens": result.get("tokens", []),
        "generated_token_indices": result.get("generated_token_indices", []),
        "targets_by_step": result.get("targets_by_step", {}),
        "targets": targets,
        "generated_text": result.get("generated_text", ""),
        "skipped": bool(result.get("skipped", False)),
        "reason": result.get("reason", ""),
    }


def annotation_payload(rich_sample: dict[str, Any]) -> dict[str, Any]:
    token_by_idx = {int(t.get("idx", -1)): t for t in rich_sample.get("tokens", [])}
    edges_by_dst: dict[str, list[dict[str, Any]]] = {}
    for edge in rich_sample.get("edges", []):
        src = int(edge.get("source"))
        dst = int(edge.get("target"))
        src_tok = token_by_idx.get(src, {})
        dst_tok = token_by_idx.get(dst, {})
        item = {
            "src": src,
            "dst": dst,
            "subtype": edge.get("subtype", ""),
            "src_display": src_tok.get("display") or src_tok.get("text") or str(src),
            "src_text": src_tok.get("text") or src_tok.get("display") or str(src),
            "dst_display": dst_tok.get("display") or dst_tok.get("text") or str(dst),
        }
        edges_by_dst.setdefault(str(dst), []).append(item)
    return {
        "num_edges": len(rich_sample.get("edges", [])),
        "edges_by_dst": edges_by_dst,
        "selection": rich_sample.get("selection", {}),
        "note": "Annotation edges come from outputs/visual_annotation/rich_top_edges.",
    }


def main() -> None:
    args = parse_args()
    rich = json.loads((ROOT / args.rich_data).read_text(encoding="utf-8"))
    saliency = json.loads((ROOT / args.saliency_data).read_text(encoding="utf-8"))
    rich_by_uid = {s.get("uid"): s for s in rich.get("samples", [])}

    samples: list[dict[str, Any]] = []
    for sample in saliency.get("samples", []):
        uid = sample.get("uid")
        rich_sample = rich_by_uid.get(uid)
        if not rich_sample:
            continue
        base = model_result(sample, "base")
        ours = model_result(sample, "ours")
        samples.append({
            "sample_id": sample.get("sample_id"),
            "row_index": sample.get("row_index"),
            "uid": uid,
            "source_dataset": sample.get("source_dataset"),
            "language": sample.get("language"),
            "raw_id": sample.get("raw_id"),
            "selection": sample.get("selection") or rich_sample.get("selection", {}),
            "base": compact_model(base, args.scope),
            "ours": compact_model(ours, args.scope),
            "annotation": annotation_payload(rich_sample),
        })

    payload = {
        "version": 2,
        "title": "Base/Ours saliency with rich annotation edges",
        "saliency_scope": args.scope,
        "saliency_source": args.saliency_data,
        "annotation_source": args.rich_data,
        "top_k": saliency.get("top_k", 5),
        "samples": samples,
    }
    out = ROOT / args.output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
