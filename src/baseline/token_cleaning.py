from __future__ import annotations

import copy
import math
from typing import Any

from .common import (
    IGNORE_INDEX,
    copy_with_masked_labels,
    finite_percentile,
    labels_of,
    sample_uid,
    supervised_indices,
)


def token_quality_scores(base_losses: list[float], ref_losses: list[float]) -> list[float]:
    """Token Cleaning fixed-model score: loss_base - loss_ref."""
    if len(base_losses) != len(ref_losses):
        raise ValueError("base_losses and ref_losses must have the same length")
    return [float(b) - float(r) for b, r in zip(base_losses, ref_losses)]


def apply_token_cleaning_from_scores(
    samples: list[dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    keep_ratio: float = 0.6,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mask low-quality supervised tokens using precomputed per-position scores."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")

    staged: list[tuple[dict[str, Any], list[float], list[int]]] = []
    all_supervised_scores: list[float] = []
    missing_scores = 0

    for idx, sample in enumerate(samples):
        uid = sample_uid(sample, fallback=str(idx))
        score_row = score_rows.get(uid)
        labels = labels_of(sample)
        supervised = supervised_indices(labels, ignore_index=ignore_index)
        if not score_row or "scores" not in score_row:
            missing_scores += 1
            staged.append((sample, [], supervised))
            continue

        scores = [float("nan") if x is None else float(x) for x in score_row["scores"]]
        if len(scores) != len(labels):
            raise ValueError(f"score length mismatch for {uid}: {len(scores)} vs {len(labels)}")
        for pos in supervised:
            if math.isfinite(scores[pos]):
                all_supervised_scores.append(scores[pos])
        staged.append((sample, scores, supervised))

    if not all_supervised_scores:
        return [copy.deepcopy(s) for s in samples], {
            "method": "token_cleaning",
            "keep_ratio": keep_ratio,
            "threshold": None,
            "missing_score_rows": missing_scores,
            "kept_tokens": 0,
            "total_supervised_tokens": 0,
        }

    threshold = finite_percentile(all_supervised_scores, (1.0 - keep_ratio) * 100.0)
    cleaned: list[dict[str, Any]] = []
    kept = 0
    total = 0
    for sample, scores, supervised in staged:
        if not scores:
            cleaned.append(copy.deepcopy(sample))
            continue
        keep_positions = {pos for pos in supervised if math.isfinite(scores[pos]) and scores[pos] >= threshold}
        kept += len(keep_positions)
        total += len(supervised)
        cleaned.append(copy_with_masked_labels(sample, keep_positions, ignore_index=ignore_index))

    return cleaned, {
        "method": "token_cleaning",
        "keep_ratio": keep_ratio,
        "threshold": threshold,
        "missing_score_rows": missing_scores,
        "kept_tokens": kept,
        "total_supervised_tokens": total,
    }


def compute_token_cleaning_score_rows(
    samples: list[dict[str, Any]],
    base_model: Any,
    ref_model: Any,
    tokenizer: Any | None = None,
    device: str | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> list[dict[str, Any]]:
    """Compute Token Cleaning scores aligned with each sample's label sequence."""
    import torch
    import torch.nn.functional as F

    from .common import resolve_model_device, to_2d_long_tensor

    runtime_device = resolve_model_device(base_model, device)
    base_model.to(runtime_device).eval()
    ref_model.to(runtime_device).eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            uid = sample_uid(sample, fallback=str(idx))
            input_ids = to_2d_long_tensor(sample["input_ids"], runtime_device)
            labels = to_2d_long_tensor(labels_of(sample), runtime_device)
            if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
                attention_mask = input_ids.ne(int(tokenizer.pad_token_id)).long()
            else:
                attention_mask = torch.ones_like(input_ids)

            logits_base = base_model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits_ref = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_base = logits_base[..., :-1, :].contiguous()
            shift_ref = logits_ref[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_base = F.cross_entropy(
                shift_base.view(-1, shift_base.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=ignore_index,
            ).view(shift_labels.size())
            loss_ref = F.cross_entropy(
                shift_ref.view(-1, shift_ref.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=ignore_index,
            ).view(shift_labels.size())

            scores: list[float | None] = [None] * labels.size(1)
            diff = (loss_base - loss_ref)[0].detach().cpu().tolist()
            valid = shift_labels[0].ne(ignore_index).detach().cpu().tolist()
            for j, is_valid in enumerate(valid, start=1):
                if is_valid:
                    scores[j] = float(diff[j - 1])
            rows.append({"uid": uid, "scores": scores})
    return rows
