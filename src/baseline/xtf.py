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


def _valid_at(values: list[float | None], pos: int) -> float | None:
    if pos >= len(values) or values[pos] is None:
        return None
    value = float(values[pos])
    return value if math.isfinite(value) else None


def iqr_low_threshold(values: list[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return None
    q1 = finite_percentile(finite, 25.0)
    q3 = finite_percentile(finite, 75.0)
    return q1 - (q3 - q1)


def xtf_noisy_positions(
    labels: list[int],
    *,
    ri_scores: list[float | None] | None = None,
    pcp_probs: list[float | None] | None = None,
    tr_scores: list[float | None] | None = None,
    pcp_threshold: float = 0.95,
    tr_percentile: float = 10.0,
    ignore_index: int = IGNORE_INDEX,
) -> set[int]:
    """Return supervised token positions considered noisy by XTF-style rules."""
    supervised = supervised_indices(labels, ignore_index=ignore_index)
    noisy: set[int] = set()

    if ri_scores:
        pairs = [(pos, _valid_at(ri_scores, pos)) for pos in supervised]
        ri_values = [value for _, value in pairs if value is not None]
        threshold = iqr_low_threshold(ri_values)
        if threshold is not None:
            noisy.update(pos for pos, value in pairs if value is not None and value < threshold)

    if pcp_probs:
        for pos in supervised:
            value = _valid_at(pcp_probs, pos)
            if value is not None and value > pcp_threshold:
                noisy.add(pos)

    if tr_scores:
        pairs = [(pos, _valid_at(tr_scores, pos)) for pos in supervised]
        tr_values = [value for _, value in pairs if value is not None]
        if tr_values:
            threshold = finite_percentile(tr_values, tr_percentile)
            noisy.update(pos for pos, value in pairs if value is not None and value < threshold)

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


def compute_xtf_score_rows(
    samples: list[dict[str, Any]],
    model: Any,
    tokenizer: Any | None = None,
    device: str | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> list[dict[str, Any]]:
    """Compute XTF RI/KN/TR scores aligned with each sample's label sequence."""
    import torch
    import torch.nn.functional as F

    from .common import resolve_model_device, to_2d_long_tensor

    runtime_device = resolve_model_device(model, device)
    model.to(runtime_device).eval()
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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            length = labels.size(1)
            label_list = labels[0].detach().cpu().tolist()
            supervised = supervised_indices(label_list, ignore_index=ignore_index)
            ri_scores: list[float | None] = [None] * length
            pcp_probs: list[float | None] = [None] * length
            tr_scores: list[float | None] = [None] * length

            if supervised:
                final_attn = outputs.attentions[-1].mean(dim=1)[0].float()
                received = final_attn.sum(dim=0)
                hidden = outputs.hidden_states[-1][0].float()
                domain_vector = hidden[supervised].mean(dim=0, keepdim=True)
                cosine = F.cosine_similarity(hidden, domain_vector, dim=-1)

                probs = F.softmax(outputs.logits[..., :-1, :], dim=-1)
                shift_labels = labels[..., 1:].contiguous()
                valid = shift_labels.ne(ignore_index)
                safe = shift_labels.masked_fill(~valid, 0)
                gathered = probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)

                for pos in supervised:
                    ri_scores[pos] = float(received[pos].detach().cpu())
                    tr_scores[pos] = float(cosine[pos].detach().cpu())
                    if pos > 0:
                        pcp_probs[pos] = float(gathered[0, pos - 1].detach().cpu())

            rows.append({
                "uid": uid,
                "ri_scores": ri_scores,
                "pcp_probs": pcp_probs,
                "tr_scores": tr_scores,
            })
    return rows
