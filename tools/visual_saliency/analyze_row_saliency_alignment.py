#!/usr/bin/env python3
"""Analyze teacher-forcing saliency/annotation alignment for one row.

The input is produced by tools/visual_saliency/compute_teacher_forcing_annotation_saliency.py
with a sufficiently large --top_k so the rank list is effectively full for each query.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True, help="Items like ce=path.json softmax_margin=path.json")
    p.add_argument("--row_index", type=int, default=479)
    p.add_argument("--top_k", type=int, nargs="+", default=[10, 20])
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--floor_eps", type=float, default=0.0, help="Proxy softmax-margin floor for reporting loss only.")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument("--max_detail_queries", type=int, default=12)
    return p.parse_args()


def read_labeled_input(spec: str) -> tuple[str, dict[str, Any]]:
    if "=" not in spec:
        raise ValueError(f"--inputs item must be label=path, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    return label, json.loads(path.read_text(encoding="utf-8"))


def find_sample(payload: dict[str, Any], row_index: int) -> dict[str, Any]:
    for sample in payload.get("samples", []):
        if int(sample.get("row_index", -1)) == row_index:
            return sample
    raise KeyError(f"row_index={row_index} not found")


def token_region(tokens: dict[int, dict[str, Any]], idx: int) -> str:
    return str(tokens.get(idx, {}).get("region", "unknown"))


def token_display(tokens: dict[int, dict[str, Any]], idx: int) -> str:
    tok = tokens.get(idx, {})
    return str(tok.get("display") or tok.get("text") or idx)


def annotation_sources(edges: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for edge in edges:
        src = int(edge["src"])
        item = grouped.setdefault(
            src,
            {
                "src": src,
                "display": edge.get("src_display") or edge.get("src_text") or str(src),
                "text": edge.get("src_text") or edge.get("src_display") or str(src),
                "subtypes": [],
            },
        )
        subtype = str(edge.get("subtype", ""))
        if subtype and subtype not in item["subtypes"]:
            item["subtypes"].append(subtype)
    return grouped


def rank_metrics(ranked_sources: list[int], annot_set: set[int], k: int) -> dict[str, float | int]:
    ranked = ranked_sources[: max(1, k)]
    hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(ranked, start=1):
        if idx in annot_set:
            hits += 1
            precision_sum += hits / rank
    denom = max(1, len(annot_set))
    return {
        "hits": hits,
        "recall": hits / denom,
        "precision": hits / max(1, len(ranked)),
        "ap": precision_sum / denom,
    }


def proxy_softmax_margin_loss(target: dict[str, Any], annot_set: set[int], alpha: float, eps: float, floor_eps: float) -> float | None:
    sources = target.get("sources", [])
    if not sources or not annot_set:
        return None
    values = {int(src["idx"]): max(float(src.get("value", 0.0)), 0.0) for src in sources}
    ranked_set = set(values)
    if not annot_set.issubset(ranked_set):
        return None
    logits = {idx: math.log(val + eps) / max(alpha, eps) for idx, val in values.items()}
    floor_logit = math.log(max(floor_eps, 0.0) + eps) / max(alpha, eps)
    negs = [idx for idx in values if idx not in annot_set]
    losses = []
    for pos in annot_set:
        pos_logit = logits[pos]
        terms = [max(logits[n], floor_logit) for n in negs]
        terms.append(pos_logit)
        m = max(terms)
        log_denom = m + math.log(sum(math.exp(x - m) for x in terms))
        losses.append(log_denom - pos_logit)
    return mean(losses) if losses else None


def summarize_query(
    *,
    model_label: str,
    model_name: str,
    target: dict[str, Any],
    ann_by_src: dict[int, dict[str, Any]],
    tokens: dict[int, dict[str, Any]],
    top_ks: list[int],
    alpha: float,
    eps: float,
    floor_eps: float,
) -> dict[str, Any]:
    q = int(target["target_idx"])
    region = token_region(tokens, q)
    sources = target.get("sources", [])
    ranked = [int(src["idx"]) for src in sources]
    value_by_src = {int(src["idx"]): float(src.get("value", 0.0)) for src in sources}
    rank_by_src = {idx: rank for rank, idx in enumerate(ranked, start=1)}
    annot_set = set(ann_by_src)

    top_metrics = {str(k): rank_metrics(ranked, annot_set, k) for k in top_ks}
    k0 = top_ks[0]
    top0 = set(ranked[:k0])
    missed = []
    for src, ann in sorted(ann_by_src.items(), key=lambda item: (rank_by_src.get(item[0], 10**9), item[0])):
        if src in top0:
            continue
        missed.append({
            "src": src,
            "display": ann["display"],
            "subtypes": ann["subtypes"],
            "rank": rank_by_src.get(src),
            "value": value_by_src.get(src),
        })
    false_positive_top = []
    for rank, src in enumerate(ranked[:k0], start=1):
        if src in annot_set:
            continue
        false_positive_top.append({
            "rank": rank,
            "src": src,
            "display": token_display(tokens, src),
            "region": token_region(tokens, src),
            "value": value_by_src.get(src),
        })

    ann_ranks = [rank_by_src[src] for src in annot_set if src in rank_by_src]
    return {
        "model_label": model_label,
        "model_name": model_name,
        "query": q,
        "query_display": target.get("display") or token_display(tokens, q),
        "query_region": region,
        "num_annotation_sources": len(annot_set),
        "metrics": top_metrics,
        "proxy_softmax_margin_loss": proxy_softmax_margin_loss(target, annot_set, alpha, eps, floor_eps),
        "mean_annotation_rank": mean(ann_ranks) if ann_ranks else None,
        "min_annotation_rank": min(ann_ranks) if ann_ranks else None,
        "max_annotation_rank": max(ann_ranks) if ann_ranks else None,
        "missed_at_top_k": missed,
        "false_positive_top_k": false_positive_top,
    }


def aggregate(rows: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[str(row["query_region"])].append(row)
    for region, items in sorted(groups.items()):
        obj: dict[str, Any] = {"n_queries": len(items)}
        for k in top_ks:
            recalls = [float(x["metrics"][str(k)]["recall"]) for x in items]
            precisions = [float(x["metrics"][str(k)]["precision"]) for x in items]
            aps = [float(x["metrics"][str(k)]["ap"]) for x in items]
            obj[f"recall@{k}"] = mean(recalls) if recalls else 0.0
            obj[f"precision@{k}"] = mean(precisions) if precisions else 0.0
            obj[f"mAP@{k}"] = mean(aps) if aps else 0.0
        losses = [x["proxy_softmax_margin_loss"] for x in items if x.get("proxy_softmax_margin_loss") is not None]
        ranks = [x["mean_annotation_rank"] for x in items if x.get("mean_annotation_rank") is not None]
        obj["proxy_softmax_margin_loss"] = mean(losses) if losses else None
        obj["mean_annotation_rank"] = mean(ranks) if ranks else None
        out[region] = obj
    return out


def markdown_table(summary: dict[str, Any], top_ks: list[int]) -> list[str]:
    k = top_ks[0]
    lines = [
        f"| model | region | q | recall@{k} | precision@{k} | mAP@{k} | proxy loss | mean ann rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_label, model_summary in summary.items():
        for region in ["all", "prompt_code", "completion", "unknown"]:
            row = model_summary.get(region)
            if not row:
                continue
            loss = row.get("proxy_softmax_margin_loss")
            rank = row.get("mean_annotation_rank")
            lines.append(
                f"| {model_label} | {region} | {row['n_queries']} | "
                f"{row.get(f'recall@{k}', 0.0):.4f} | {row.get(f'precision@{k}', 0.0):.4f} | "
                f"{row.get(f'mAP@{k}', 0.0):.4f} | "
                f"{loss:.4f} | " if loss is not None else
                f"| {model_label} | {region} | {row['n_queries']} | {row.get(f'recall@{k}', 0.0):.4f} | {row.get(f'precision@{k}', 0.0):.4f} | {row.get(f'mAP@{k}', 0.0):.4f} |  | "
            )
            if loss is not None:
                lines[-1] += f"{rank:.2f} |" if rank is not None else " |"
            else:
                lines[-1] += f"{rank:.2f} |" if rank is not None else " |"
    return lines


def main() -> None:
    args = parse_args()
    top_ks = sorted({int(k) for k in args.top_k})
    model_queries: dict[str, list[dict[str, Any]]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    base_added = False

    for label, payload in map(read_labeled_input, args.inputs):
        sample = find_sample(payload, args.row_index)
        for model_key, model_label in [("base", "base"), ("ours", label)]:
            if model_key == "base" and base_added:
                continue
            if model_key == "base":
                base_added = True
            model = sample[model_key]
            tokens = {int(tok["idx"]): tok for tok in model.get("tokens", [])}
            ann_edges = sample["annotation"]["edges_by_dst"]
            rows = []
            for key, target in sorted(model.get("targets", {}).items(), key=lambda item: int(item[0])):
                q = int(target["target_idx"])
                ann = annotation_sources(ann_edges.get(str(q), []))
                if not ann:
                    continue
                rows.append(summarize_query(
                    model_label=model_label,
                    model_name=str(model.get("name", model_label)),
                    target=target,
                    ann_by_src=ann,
                    tokens=tokens,
                    top_ks=top_ks,
                    alpha=float(args.alpha),
                    eps=float(args.eps),
                    floor_eps=float(args.floor_eps),
                ))
            model_queries[model_label] = rows
            details[model_label] = sorted(
                rows,
                key=lambda row: (float(row["metrics"][str(top_ks[0])]["recall"]), -(row.get("num_annotation_sources") or 0)),
            )[: args.max_detail_queries]

    summary = {label: aggregate(rows, top_ks) for label, rows in model_queries.items()}
    report = {
        "row_index": args.row_index,
        "top_k": top_ks,
        "alpha": args.alpha,
        "floor_eps": args.floor_eps,
        "summary": summary,
        "worst_queries": details,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [f"# Row {args.row_index} Saliency Alignment", ""]
    md += markdown_table(summary, top_ks)
    md += ["", f"## Worst Queries by recall@{top_ks[0]}", ""]
    for label, rows in details.items():
        md += [f"### {label}", ""]
        for row in rows:
            m = row["metrics"][str(top_ks[0])]
            missed = ", ".join(
                f"{x['src']}:{x['display']}@{x.get('rank')}" for x in row["missed_at_top_k"][:8]
            )
            false_pos = ", ".join(
                f"#{x['rank']} {x['src']}:{x['display']}" for x in row["false_positive_top_k"][:8]
            )
            md.append(
                f"- q={row['query']} `{row['query_display']}` region={row['query_region']} "
                f"ann={row['num_annotation_sources']} recall={m['recall']:.4f} "
                f"mean_rank={row.get('mean_annotation_rank')} loss={row.get('proxy_softmax_margin_loss')}"
            )
            md.append(f"  - missed: {missed}")
            md.append(f"  - false-positive top: {false_pos}")
        md.append("")
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
