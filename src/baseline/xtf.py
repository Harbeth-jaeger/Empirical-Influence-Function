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


def iqr_low_threshold(values: list[float]) -> float | None:
    finite = [float(v) for v in values if v == v]
    if len(finite) < 2:
        return None
    q1 = finite_percentile(finite, 25.0)
    q3 = finite_percentile(finite, 75.0)
    return q1 - (q3 - q1)


def xtf_noisy_positions(
    labels: list[int],
    *,
    ri_scores: list[float] | None = None,
    pcp_probs: list[float] | None = None,
    tr_scores: list[float] | None = None,
    pcp_threshold: float = 0.95,
    tr_percentile: float = 10.0,
    ignore_index: int = IGNORE_INDEX,
) -> set[int]:
    """Return supervised token positions considered noisy by XTF-style rules."""
    supervised = supervised_indices(labels, ignore_index=ignore_index)
    noisy: set[int] = set()

    if ri_scores:
        ri_values = [float(ri_scores[pos]) for pos in supervised]
        threshold = iqr_low_threshold(ri_values)
        if threshold is not None:
            noisy.update(pos for pos in supervised if float(ri_scores[pos]) < threshold)

    if pcp_probs:
        noisy.update(pos for pos in supervised if float(pcp_probs[pos]) > pcp_threshold)

    if tr_scores:
        tr_values = [float(tr_scores[pos]) for pos in supervised]
        threshold = finite_percentile(tr_values, tr_percentile)
        noisy.update(pos for pos in supervised if float(tr_scores[pos]) < threshold)

    return noisy


def apply_xtf_from_scores(
    samples: list[dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    pcp_threshold: float = 0.95,
    tr_percentile: float = 10.0,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    missing_scores = 0
    total_supervised = 0
    total_masked = 0

    for idx, sample in enumerate(samples):
        uid = sample_uid(sample, fallback=str(idx))
        score_row = score_rows.get(uid)
        if not score_row:
            missing_scores += 1
            cleaned.append(copy.deepcopy(sample))
            continue

        labels = labels_of(sample)
        supervised = supervised_indices(labels, ignore_index=ignore_index)
        noisy = xtf_noisy_positions(
            labels,
            ri_scores=score_row.get("ri_scores"),
            pcp_probs=score_row.get("pcp_probs") or score_row.get("pcp_scores"),
            tr_scores=score_row.get("tr_scores"),
            pcp_threshold=pcp_threshold,
            tr_percentile=tr_percentile,
            ignore_index=ignore_index,
        )
        keep_positions = set(supervised) - noisy
        total_supervised += len(supervised)
        total_masked += len(noisy)
        cleaned.append(copy_with_masked_labels(sample, keep_positions, ignore_index=ignore_index))

    return cleaned, {
        "method": "xtf",
        "pcp_threshold": pcp_threshold,
        "tr_percentile": tr_percentile,
        "missing_score_rows": missing_scores,
        "masked_tokens": total_masked,
        "total_supervised_tokens": total_supervised,
    }

