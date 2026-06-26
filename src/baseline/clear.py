from __future__ import annotations

import copy
from difflib import SequenceMatcher
from typing import Any

from .common import replace_response_fields, sample_uid


def target_consistency(target: str, sampled_outputs: list[str]) -> float:
    target = target.strip()
    outputs = [out.strip() for out in sampled_outputs if out and out.strip()]
    if not target or not outputs:
        return 0.0
    return sum(SequenceMatcher(None, target, out).ratio() for out in outputs) / len(outputs)


def clear_confidence(observed_consistency: float, self_reflection_certainty: float, alpha: float = 0.5) -> float:
    value = alpha * float(observed_consistency) + (1.0 - alpha) * float(self_reflection_certainty)
    return max(0.0, min(1.0, value))


def apply_clear_scores(
    samples: list[dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    gamma: float = 0.5,
    eta: float = 0.8,
    alpha: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """CLEAR-style filtering and optional correction from precomputed scores.

    score row fields:
      - confidence, or observed_consistency + self_reflection_certainty
      - optional candidate_response + candidate_confidence + candidate_better_score
    """
    out: list[dict[str, Any]] = []
    missing = 0
    filtered = 0
    replaced = 0

    for idx, sample in enumerate(samples):
        uid = sample_uid(sample, fallback=str(idx))
        row = score_rows.get(uid)
        if not row:
            missing += 1
            out.append(copy.deepcopy(sample))
            continue

        confidence = row.get("confidence")
        if confidence is None:
            confidence = clear_confidence(
                float(row.get("observed_consistency", 0.0)),
                float(row.get("self_reflection_certainty", 0.5)),
                alpha=alpha,
            )
        confidence = float(confidence)

        candidate = str(row.get("candidate_response", "")).strip()
        candidate_confidence = float(row.get("candidate_confidence", 0.0))
        better_score = float(row.get("candidate_better_score", 0.0))
        if candidate and better_score > eta and candidate_confidence >= confidence:
            out.append(replace_response_fields(sample, candidate))
            replaced += 1
            continue

        if confidence <= gamma:
            filtered += 1
            continue
        out.append(copy.deepcopy(sample))

    return out, {
        "method": "clear",
        "gamma": gamma,
        "eta": eta,
        "alpha": alpha,
        "missing_score_rows": missing,
        "filtered_samples": filtered,
        "replaced_samples": replaced,
        "kept_samples": len(out),
    }

