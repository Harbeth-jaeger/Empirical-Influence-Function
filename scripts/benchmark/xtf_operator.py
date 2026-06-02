from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.benchmark.governance_common import IGNORE_INDEX, _resolve_device, _to_2d_long_tensor


def xtf_token_filter_operator(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    pcp_threshold: float = 0.95,
    tr_percentile: float = 10.0,
    attention_mask: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    XTF-inspired token filtering for a single sample [1, L].
    NOTE: TR stage uses percentile approximation (not full Multi-Otsu clustering).
    """
    model.eval()

    if input_ids.dim() != 2 or labels.dim() != 2 or input_ids.size(0) != 1 or labels.size(0) != 1:
        raise ValueError("xtf_token_filter_operator expects input_ids and labels with shape [1, L].")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    logits = outputs.logits
    attentions = outputs.attentions[-1]
    embeddings = outputs.hidden_states[-1]

    label_mask = labels[0] != ignore_index
    if not label_mask.any():
        return labels.clone()

    # RI: attention-based importance
    avg_attn = attentions.mean(dim=1)[0]
    ri_scores = avg_attn.sum(dim=0).float()
    ri_relevant = ri_scores[label_mask].float()
    if ri_relevant.numel() >= 2:
        q1 = torch.quantile(ri_relevant, 0.25)
        q3 = torch.quantile(ri_relevant, 0.75)
        ri_threshold = q1 - (q3 - q1)
        ri_mask = ri_scores < ri_threshold
    else:
        ri_mask = torch.zeros_like(label_mask, dtype=torch.bool)

    # KN: PCP-based novelty
    probs = F.softmax(logits, dim=-1)
    shift_probs = probs[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_pos = shift_labels != ignore_index
    safe_labels = shift_labels.masked_fill(~valid_pos, 0)
    gathered = shift_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    pcp_scores = torch.zeros_like(labels, dtype=torch.float)
    pcp_scores[..., 1:] = torch.where(valid_pos, gathered, torch.zeros_like(gathered))
    kn_mask = pcp_scores[0] > pcp_threshold

    # TR: embedding-domain relevance (percentile fallback)
    emb = embeddings[0]
    domain_vector = emb[label_mask].mean(dim=0, keepdim=True)
    tr_scores = F.cosine_similarity(emb, domain_vector, dim=-1).float()

    tr_relevant = tr_scores[label_mask].float()
    if tr_relevant.numel() >= 2:
        tr_threshold = torch.quantile(tr_relevant, tr_percentile / 100.0)
        tr_mask = tr_scores < tr_threshold
    else:
        tr_mask = torch.zeros_like(label_mask, dtype=torch.bool)

    final_noise_mask = (ri_mask | kn_mask | tr_mask) & label_mask

    new_labels = labels.clone()
    new_labels[0, final_noise_mask] = ignore_index
    return new_labels


def xtf_token_filter_dataset(
    samples: list[dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer | None = None,
    pcp_threshold: float = 0.95,
    tr_percentile: float = 10.0,
    device: str | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> list[dict[str, Any]]:
    """Apply XTF-style filtering to a dataset list with input_ids/labels."""
    runtime_device = _resolve_device(model, device)
    model.eval()

    out: list[dict[str, Any]] = []
    for sample in tqdm.tqdm(samples, desc="  xtf.filter", mininterval=5):
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

        filtered = xtf_token_filter_operator(
            model=model,
            input_ids=input_ids,
            labels=labels,
            pcp_threshold=pcp_threshold,
            tr_percentile=tr_percentile,
            attention_mask=attention_mask,
            ignore_index=ignore_index,
        )

        if isinstance(local["labels"], list) and local["labels"] and isinstance(local["labels"][0], list):
            local["labels"] = filtered.detach().cpu().tolist()
        else:
            local["labels"] = filtered[0].detach().cpu().tolist()
        out.append(local)

    return out


