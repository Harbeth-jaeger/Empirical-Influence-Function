#!/usr/bin/env python3
"""Select medium-difficulty Base-fail/Ours-pass test examples for prediction saliency."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_RESULT = ROOT / "outputs/benchmark/eval_results/base_qwen/Base Qwen_benchmark_eval_merged.json"
DEFAULT_OURS_DUMP = ROOT / "outputs/viz_failure/dumps/ours_graphsignal_500_humaneval1000.jsonl"
DEFAULT_EVAL = ROOT / "/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/eval_data/rendered_chatml_fim_eval.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/viz_prediction/prediction_saliency_manifest_50.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_result", default=str(DEFAULT_BASE_RESULT))
    p.add_argument("--ours_dump", default=str(DEFAULT_OURS_DUMP))
    p.add_argument("--eval_path", default=str(DEFAULT_EVAL))
    p.add_argument("--output_path", default=str(DEFAULT_OUTPUT))
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--min_completion_chars", type=int, default=12)
    p.add_argument("--max_completion_chars", type=int, default=900)
    p.add_argument("--min_ours_gt_similarity", type=float, default=0.25)
    p.add_argument("--languages", default="python")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_base_predictions(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("merged_files") or [str(path)]
    out: dict[str, dict[str, Any]] = {}
    for rel in files:
        p = (ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
        obj = json.loads(p.read_text(encoding="utf-8"))
        for row in obj.get("sample_predictions", []):
            uid = str(row.get("uid", ""))
            if uid:
                out[uid] = row
    return out


def load_ours_dump(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid = str(row.get("uid", ""))
        if uid:
            out[uid] = row
    return out


def load_eval_rows(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid = str(row.get("uid", ""))
        if uid:
            out[uid] = row
    return out


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()


def length_score(n: int) -> float:
    if 80 <= n <= 420:
        return 1.0
    if 30 <= n < 80:
        return 0.55 + (n - 30) / 50 * 0.35
    if 420 < n <= 900:
        return max(0.2, 1.0 - (n - 420) / 480 * 0.8)
    return 0.0


def main() -> None:
    args = parse_args()
    languages = {x.strip().lower() for x in args.languages.split(",") if x.strip()}
    base_by_uid = load_base_predictions(Path(args.base_result))
    ours_by_uid = load_ours_dump(Path(args.ours_dump))
    eval_by_uid = load_eval_rows(Path(args.eval_path))

    candidates: list[dict[str, Any]] = []
    for uid, ours in ours_by_uid.items():
        base = base_by_uid.get(uid)
        row = eval_by_uid.get(uid)
        if not base or not row:
            continue
        lang = str(ours.get("language", row.get("language", ""))).lower()
        if languages and lang not in languages:
            continue
        if bool(base.get("pass1")) or not bool(ours.get("pass1")):
            continue
        gt = str(ours.get("ground_truth", ""))
        ours_pred = str((ours.get("greedy") or {}).get("prediction", ""))
        base_pred = str(base.get("greedy_prediction", ""))
        if not (args.min_completion_chars <= len(ours_pred) <= args.max_completion_chars):
            continue
        if base_pred.strip() == ours_pred.strip():
            continue
        ours_gt = sim(ours_pred, gt)
        base_gt = sim(base_pred, gt)
        if ours_gt < args.min_ours_gt_similarity:
            continue
        score = (ours_gt - base_gt) * 2.0 + length_score(len(ours_pred))
        candidates.append({
            "sample_id": f"pred_{int(ours.get('filtered_index', len(candidates))):04d}",
            "filtered_index": ours.get("filtered_index"),
            "uid": uid,
            "source_dataset": ours.get("source_dataset", row.get("source_dataset", "")),
            "language": lang,
            "raw_id": ours.get("raw_id", row.get("raw_id", "")),
            "entry_point": ours.get("entry_point", ""),
            "prefix": ours.get("prefix", ""),
            "suffix": ours.get("suffix", ""),
            "ground_truth": gt,
            "base_prediction": base_pred,
            "ours_prediction": ours_pred,
            "metrics": {
                "base_pass1": bool(base.get("pass1")),
                "ours_pass1": bool(ours.get("pass1")),
                "base_gt_similarity": base_gt,
                "ours_gt_similarity": ours_gt,
                "prediction_chars": len(ours_pred),
                "score": score,
            },
            "row": row,
        })

    candidates.sort(key=lambda x: x["metrics"]["score"], reverse=True)
    samples = candidates[: max(1, args.limit)]
    payload = {
        "version": 1,
        "selection": "Base greedy fail, Ours/SFT greedy pass; medium-length heuristic; judge results are used only for offline screening.",
        "base_result": str(args.base_result),
        "ours_dump": str(args.ours_dump),
        "eval_path": str(args.eval_path),
        "num_candidates": len(candidates),
        "samples": samples,
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"candidates: {len(candidates)}")
    print(f"wrote {out} ({len(samples)} samples)")
    for s in samples[:10]:
        m = s["metrics"]
        print(f"#{s['filtered_index']} {s['uid']} score={m['score']:.3f} ours_gt={m['ours_gt_similarity']:.3f} base_gt={m['base_gt_similarity']:.3f} chars={m['prediction_chars']}")


if __name__ == "__main__":
    main()
