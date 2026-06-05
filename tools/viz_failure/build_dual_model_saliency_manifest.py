#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_uid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("uid") or r.get("filtered_index")): r for r in rows}


def load_eval_rows(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        uid = str(row.get("uid") or "")
        if uid:
            out[uid] = row
    return out


def pass_bool(row: dict[str, Any], k: int) -> bool:
    return bool(row.get(f"pass{k}", False))


def categories(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    ap1, bp1 = pass_bool(a, 1), pass_bool(b, 1)
    ap10, bp10 = pass_bool(a, 10), pass_bool(b, 10)
    cats = []
    if ap1 and not bp1:
        cats.append("a_pass1_b_fail1")
    if bp1 and not ap1:
        cats.append("b_pass1_a_fail1")
    if ap1 and bp1:
        cats.append("both_pass1")
    if (not ap1) and (not bp1):
        cats.append("both_fail1")
    if ap10 and not bp10:
        cats.append("a_pass10_b_fail10")
    if bp10 and not ap10:
        cats.append("b_pass10_a_fail10")
    if ap10 and bp10:
        cats.append("both_pass10")
    if (not ap10) and (not bp10):
        cats.append("both_fail10")
    return cats or ["other"]


def prediction(row: dict[str, Any]) -> str:
    return str((row.get("greedy") or {}).get("prediction", ""))


def model_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": row.get("baseline", ""),
        "model_path": row.get("model_path", ""),
        "pass1": pass_bool(row, 1),
        "pass10": pass_bool(row, 10),
        "prediction": prediction(row),
        "greedy": row.get("greedy", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build sample manifest for two-model prediction/saliency viewer.")
    ap.add_argument("--model_a_dump", required=True)
    ap.add_argument("--model_b_dump", required=True)
    ap.add_argument("--model_a_name", default="Model A")
    ap.add_argument("--model_b_name", default="Model B")
    ap.add_argument("--eval_path", default="data/benchmarks/eval_data/rendered_chatml_fim_eval.jsonl")
    ap.add_argument("--output_path", default="outputs/viz_failure/dual_model_saliency_manifest.json")
    ap.add_argument("--limit_per_category", type=int, default=12)
    ap.add_argument("--languages", default="python")
    args = ap.parse_args()

    langs = {x.strip().lower() for x in args.languages.split(",") if x.strip()}
    a_rows = by_uid(read_jsonl(Path(args.model_a_dump)))
    b_rows = by_uid(read_jsonl(Path(args.model_b_dump)))
    eval_by_uid = load_eval_rows(Path(args.eval_path))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    common = sorted(set(a_rows) & set(b_rows), key=lambda k: int(a_rows[k].get("filtered_index", 10**12)))
    for key in common:
        a = a_rows[key]
        b = b_rows[key]
        uid = str(a.get("uid") or key)
        row = eval_by_uid.get(uid)
        if not row:
            continue
        lang = str(a.get("language") or row.get("language", "")).lower()
        if langs and lang not in langs:
            continue
        cats = categories(a, b)
        counts.update(cats)
        sample = {
            "sample_id": f"dual_{int(a.get('filtered_index', len(buckets))):04d}",
            "key": key,
            "categories": cats,
            "filtered_index": a.get("filtered_index"),
            "uid": uid,
            "source_dataset": a.get("source_dataset", row.get("source_dataset", "")),
            "language": lang,
            "raw_id": a.get("raw_id", row.get("raw_id", "")),
            "entry_point": a.get("entry_point", ""),
            "prefix": a.get("prefix", ""),
            "suffix": a.get("suffix", ""),
            "ground_truth": a.get("ground_truth", ""),
            "row": row,
            "model_a": model_view(a),
            "model_b": model_view(b),
        }
        for c in cats:
            if len(buckets[c]) < args.limit_per_category:
                buckets[c].append(sample)

    seen = set()
    samples = []
    preferred = ["a_pass1_b_fail1", "b_pass1_a_fail1", "both_pass1", "both_fail1", "a_pass10_b_fail10", "b_pass10_a_fail10"]
    for cat in preferred + sorted(set(buckets) - set(preferred)):
        for s in buckets.get(cat, []):
            if s["sample_id"] not in seen:
                seen.add(s["sample_id"])
                samples.append(s)

    payload = {
        "version": 1,
        "title": "双模型预测与 saliency 对比",
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "model_a_dump": args.model_a_dump,
        "model_b_dump": args.model_b_dump,
        "eval_path": args.eval_path,
        "categories": {
            "a_pass1_b_fail1": f"{args.model_a_name} P@1 pass, {args.model_b_name} fail",
            "b_pass1_a_fail1": f"{args.model_b_name} P@1 pass, {args.model_a_name} fail",
            "both_pass1": "Both P@1 pass",
            "both_fail1": "Both P@1 fail",
            "a_pass10_b_fail10": f"{args.model_a_name} P@10 pass, {args.model_b_name} fail",
            "b_pass10_a_fail10": f"{args.model_b_name} P@10 pass, {args.model_a_name} fail",
            "both_pass10": "Both P@10 pass",
            "both_fail10": "Both P@10 fail",
        },
        "category_counts": dict(counts),
        "num_common": len(common),
        "samples": samples,
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"common={len(common)} selected={len(samples)}")
    print(dict(counts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
