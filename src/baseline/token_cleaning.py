from __future__ import annotations

import copy
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
    """Token Cleaning fixed-model score: loss_base - loss_ref.

    Larger means the reference model improves more on this token, so the token is
    treated as more task-informative.
    """
    if len(base_losses) != len(ref_losses):
        raise ValueError("base_losses and ref_losses must have the same length")
    return [float(b) - float(r) for b, r in zip(base_losses, ref_losses)]


def apply_token_cleaning_from_scores(
    samples: list[dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    keep_ratio: float = 0.6,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mask low-quality supervised tokens using precomputed per-position scores.

    score_rows rows should contain a `scores` list aligned with `labels`.
    """
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

        scores = [float(x) for x in score_row["scores"]]
        if len(scores) != len(labels):
            raise ValueError(f"score length mismatch for {uid}: {len(scores)} vs {len(labels)}")
        for pos in supervised:
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
        keep_positions = {pos for pos in supervised if scores[pos] >= threshold}
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

