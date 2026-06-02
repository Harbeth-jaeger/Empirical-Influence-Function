from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


from scripts.benchmark.governance_common import (
    IGNORE_INDEX,
    _extract_code_block_or_raw,
    _resolve_device,
    _safe_float,
    _to_2d_long_tensor,
)


from scripts.benchmark.graphsignal_annotations import (
    DEFAULT_GRAPH_REASON_WEIGHTS,
    _as_annotation_dicts,
    _extract_chat_message_text,
    _extract_fim_parts,
    _normalize_language,
    _parse_teacher_json,
    _summarize_chat_response,
    add_graphsignal_teacher_annotations,
)


def _get_annotations(sample: dict[str, Any], annotation_field: Literal["auto", "qwen_annotations", "annotations"]) -> list[dict[str, Any]]:
    if annotation_field == "qwen_annotations":
        return list(sample.get("qwen_annotations", []))
    if annotation_field == "annotations":
        return list(sample.get("annotations", []))
    # auto: prefer qwen_annotations because they are tokenizer-aligned.
    if "qwen_annotations" in sample:
        return list(sample.get("qwen_annotations", []))
    return list(sample.get("annotations", []))


def _infer_token_count(sample: dict[str, Any]) -> int:
    # Priority aligns with training tensors.
    for key in ("labels", "input_ids", "attention_mask"):
        if key in sample:
            arr = sample[key]
            if isinstance(arr, list):
                if len(arr) == 0:
                    return 0
                if isinstance(arr[0], list):
                    return len(arr[0])
                return len(arr)
            t = torch.as_tensor(arr)
            if t.dim() == 0:
                return 0
            if t.dim() == 1:
                return int(t.size(0))
            return int(t.size(1))

    # Fallback to token tables if tensors are not present.
    if "qwen_tokens" in sample and isinstance(sample["qwen_tokens"], list):
        return len(sample["qwen_tokens"])
    if "tokens" in sample and isinstance(sample["tokens"], list):
        return len(sample["tokens"])
    return 0


def _build_graph_importance(
    sample: dict[str, Any],
    annotation_field: Literal["auto", "qwen_annotations", "annotations"] = "auto",
    reason_weights: dict[str, float] | None = None,
    alpha_in: float = 1.0,
    alpha_out: float = 0.7,
    self_bias: float = 1e-6,
) -> np.ndarray:
    """
    Build token importance scores from directed annotation edges.
    Score_j = self_bias + alpha_in * in_weight_j + alpha_out * out_weight_j
    Normalized to [0, 1] per sample.
    """
    L = _infer_token_count(sample)
    if L <= 0:
        return np.zeros((0,), dtype=np.float32)

    ann = _get_annotations(sample, annotation_field)
    if len(ann) == 0:
        return np.ones((L,), dtype=np.float32)

    w = dict(DEFAULT_GRAPH_REASON_WEIGHTS)
    if reason_weights:
        w.update(reason_weights)

    in_w = np.zeros((L,), dtype=np.float32)
    out_w = np.zeros((L,), dtype=np.float32)

    for e in ann:
        i = e.get("token_i_idx", -1)
        j = e.get("token_j_idx", -1)
        if not isinstance(i, int) or not isinstance(j, int):
            continue
        if i < 0 or j < 0 or i >= L or j >= L or i == j:
            continue

        reason = str(e.get("subtype", "") or e.get("reason", "") or "semantic").lower()
        edge_w = float(w.get(reason, w["semantic"]))

        out_w[i] += edge_w
        in_w[j] += edge_w

    score = self_bias + alpha_in * in_w + alpha_out * out_w

    s_min = float(score.min())
    s_max = float(score.max())
    if s_max - s_min < 1e-12:
        return np.ones((L,), dtype=np.float32)
    return ((score - s_min) / (s_max - s_min)).astype(np.float32)


def graph_signal_operator(
    samples: list[dict[str, Any]],
    mode: Literal["hard_mask", "soft_weight"] = "hard_mask",
    keep_ratio: float = 0.6,
    min_weight: float = 0.1,
    annotation_field: Literal["auto", "qwen_annotations", "annotations"] = "auto",
    reason_weights: dict[str, float] | None = None,
    alpha_in: float = 1.0,
    alpha_out: float = 0.7,
    ignore_index: int = IGNORE_INDEX,
    output_weight_key: str = "token_weights",
) -> list[dict[str, Any]]:
    """
    Annotation-graph-guided governance.

    Modes:
    - hard_mask: keep top keep_ratio tokens by graph score (on valid label positions),
      set other valid labels to ignore_index.
    - soft_weight: keep labels unchanged; export per-token weights in [min_weight, 1].
    """
    if mode not in {"hard_mask", "soft_weight"}:
        raise ValueError("mode must be 'hard_mask' or 'soft_weight'.")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1].")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("min_weight must be in [0, 1].")

    staged: list[dict[str, Any]] = []
    all_valid_scores: list[float] = []

    for sample in tqdm.tqdm(samples, desc="  graph_signal.score", mininterval=5):
        local = copy.deepcopy(sample)
        score = _build_graph_importance(
            local,
            annotation_field=annotation_field,
            reason_weights=reason_weights,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
        )
        local["_graph_score"] = score

        if "labels" in local:
            labels_t = torch.as_tensor(local["labels"], dtype=torch.long)
            if labels_t.dim() == 1:
                valid = labels_t != ignore_index
                if valid.any() and score.size > 0:
                    n = min(score.size, int(labels_t.size(0)))
                    valid_np = valid[:n].cpu().numpy()
                    all_valid_scores.extend(score[:n][valid_np].tolist())
                local["_valid_mask"] = valid
            elif labels_t.dim() == 2:
                valid = labels_t != ignore_index
                n = min(score.size, int(labels_t.size(1)))
                if valid.any() and score.size > 0 and n > 0:
                    valid_np = valid[:, :n].cpu().numpy()
                    expanded = np.tile(score[:n], (valid.size(0), 1))
                    all_valid_scores.extend(expanded[valid_np].tolist())
                local["_valid_mask"] = valid
            else:
                local["_valid_mask"] = None
        else:
            local["_valid_mask"] = None

        staged.append(local)

    threshold = None
    if mode == "hard_mask" and len(all_valid_scores) > 0:
        threshold = float(np.percentile(np.array(all_valid_scores, dtype=np.float32), (1.0 - keep_ratio) * 100.0))

    out: list[dict[str, Any]] = []
    for local in tqdm.tqdm(staged, desc="  graph_signal.apply", mininterval=5):
        score = local.pop("_graph_score")
        valid = local.pop("_valid_mask", None)

        if mode == "soft_weight":
            if "labels" in local:
                labels_t = torch.as_tensor(local["labels"], dtype=torch.long)
                if labels_t.dim() == 1:
                    n = min(score.size, int(labels_t.size(0)))
                    w = np.ones((int(labels_t.size(0)),), dtype=np.float32)
                    if n > 0:
                        w[:n] = min_weight + (1.0 - min_weight) * score[:n]
                    w[labels_t.cpu().numpy() == ignore_index] = 0.0
                    local[output_weight_key] = w.tolist()
                elif labels_t.dim() == 2:
                    B, T = int(labels_t.size(0)), int(labels_t.size(1))
                    n = min(score.size, T)
                    row = np.ones((T,), dtype=np.float32)
                    if n > 0:
                        row[:n] = min_weight + (1.0 - min_weight) * score[:n]
                    w = np.tile(row, (B, 1))
                    mask = labels_t.cpu().numpy() == ignore_index
                    w[mask] = 0.0
                    local[output_weight_key] = w.tolist()
            else:
                local[output_weight_key] = (min_weight + (1.0 - min_weight) * score).tolist()

            out.append(local)
            continue

        # hard_mask path
        if "labels" not in local or threshold is None:
            out.append(local)
            continue

        labels_t = torch.as_tensor(local["labels"], dtype=torch.long)
        if labels_t.dim() == 1:
            n = min(score.size, int(labels_t.size(0)))
            if n > 0:
                filt = torch.as_tensor(score[:n] < threshold)
                if valid is not None:
                    filt = filt & valid[:n]
                labels_t[:n][filt] = ignore_index
            local["labels"] = labels_t.tolist()
        elif labels_t.dim() == 2:
            B, T = int(labels_t.size(0)), int(labels_t.size(1))
            n = min(score.size, T)
            if n > 0:
                row_filt = torch.as_tensor(score[:n] < threshold)
                for b in range(B):
                    filt = row_filt.clone()
                    if valid is not None:
                        filt = filt & valid[b, :n]
                    labels_t[b, :n][filt] = ignore_index
            local["labels"] = labels_t.tolist()

        out.append(local)

    return out


from scripts.benchmark.token_cleaning_operator import token_cleaning_operator
from scripts.benchmark.xtf_operator import xtf_token_filter_dataset, xtf_token_filter_operator
from scripts.benchmark.llm_cleaning_operator import llm_code_cleaning_dataset, llm_code_cleaning_operator
from scripts.benchmark.clear_operator import clear_curation_dataset, clear_curation_operator


def apply_governance_operator(
    samples: list[dict[str, Any]],
    operator: Literal["token_cleaning", "xtf", "llm_code_cleaning", "clear", "graph_signal"],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Unified entrypoint for governance operators.

    Required kwargs by operator:
    - token_cleaning: base_model, ref_model, tokenizer (optional), keep_ratio, device
    - xtf: model, tokenizer (optional), pcp_threshold, tr_percentile, device
    - llm_code_cleaning: api_key, model_name (optional), prompt_key/code_key/test_cases_key/output_key
    - clear: client, model_name (optional), stage, gamma, eta
    - graph_signal: mode(hard_mask|soft_weight), annotation_field, keep_ratio, min_weight, reason_weights
    """
    if operator == "token_cleaning":
        return token_cleaning_operator(samples=samples, **kwargs)
    if operator == "xtf":
        return xtf_token_filter_dataset(samples=samples, **kwargs)
    if operator == "llm_code_cleaning":
        return llm_code_cleaning_dataset(samples=samples, **kwargs)
    if operator == "clear":
        return clear_curation_dataset(samples=samples, **kwargs)
    if operator == "graph_signal":
        return graph_signal_operator(samples=samples, **kwargs)
    raise ValueError(f"Unknown operator: {operator}")
