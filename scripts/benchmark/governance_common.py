from __future__ import annotations

import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM


IGNORE_INDEX = -100


def _resolve_device(model: AutoModelForCausalLM, device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_2d_long_tensor(x: Any, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.long, device=device)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"Expected 2D tensor-like input, got shape={tuple(t.shape)}")
    return t


def _extract_code_block_or_raw(text: str) -> str:
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    return text.strip()


def _safe_float(text: str, default: float = 0.5) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return default
    val = float(m.group(0))
    return max(0.0, min(1.0, val))


