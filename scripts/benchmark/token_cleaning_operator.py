from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.benchmark.governance_common import IGNORE_INDEX, _resolve_device, _to_2d_long_tensor


def token_cleaning_operator(
    samples: list[dict[str, Any]],
    base_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer | None,
    keep_ratio: float = 0.6,
    device: str | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> list[dict[str, Any]]:
    """
    Token Cleaning (fixed-model variant):
    score = loss_base - loss_ref, then keep top keep_ratio globally.
    """
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1].")

    base_model.eval()
    ref_model.eval()
    runtime_device = _resolve_device(base_model, device)

    all_scores: list[float] = []
    staged: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample in tqdm.tqdm(samples, desc="  token_cleaning.score", mininterval=5):
            local = copy.deepcopy(sample)

            input_ids = _to_2d_long_tensor(local["input_ids"], runtime_device)
            labels = _to_2d_long_tensor(local["labels"], runtime_device)

            if "attention_mask" in local:
                attention_mask = _to_2d_long_tensor(local["attention_mask"], runtime_device)
            else:
                if tokenizer is not None and tokenizer.pad_token_id is not None:
                    attention_mask = (input_ids != tokenizer.pad_token_id).long()
                else:
                    attention_mask = torch.ones_like(input_ids)

            logits_base = base_model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits_ref = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

            shift_logits_base = logits_base[..., :-1, :].contiguous()
            shift_logits_ref = logits_ref[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_base = F.cross_entropy(
                shift_logits_base.view(-1, shift_logits_base.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=ignore_index,
            ).view(shift_labels.size())
            loss_ref = F.cross_entropy(
                shift_logits_ref.view(-1, shift_logits_ref.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=ignore_index,
            ).view(shift_labels.size())

            score = loss_base - loss_ref
            valid_mask = shift_labels != ignore_index

            if valid_mask.any():
                all_scores.extend(score[valid_mask].detach().cpu().tolist())

            local["_raw_scores"] = score.detach().cpu()
            local["_valid_mask"] = valid_mask.detach().cpu()
            staged.append(local)

    if len(all_scores) == 0:
        for local in staged:
            local.pop("_raw_scores", None)
            local.pop("_valid_mask", None)
        return staged

    threshold = float(np.percentile(np.array(all_scores, dtype=np.float32), (1.0 - keep_ratio) * 100.0))

    cleaned_samples: list[dict[str, Any]] = []
    for local in tqdm.tqdm(staged, desc="  token_cleaning.apply", mininterval=5):
        score = local.pop("_raw_scores")
        valid_mask = local.pop("_valid_mask")

        new_labels = torch.as_tensor(local["labels"], dtype=torch.long)
        if new_labels.dim() == 1:
            # score/valid_mask are aligned with new_labels[1:]
            to_filter = (score[0] < threshold) & valid_mask[0]
            if to_filter.any():
                indices = torch.where(to_filter)[0] + 1
                new_labels[indices] = ignore_index
        else:
            # Keep behavior consistent if labels are already batched.
            to_filter = (score < threshold) & valid_mask
            if to_filter.any():
                rows, cols = torch.where(to_filter)
                local_indices = cols + 1
                for r, c in zip(rows.tolist(), local_indices.tolist()):
                    new_labels[r, c] = ignore_index

        local["labels"] = new_labels.tolist()
        cleaned_samples.append(local)

    return cleaned_samples


