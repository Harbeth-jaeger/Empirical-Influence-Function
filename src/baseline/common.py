from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

IGNORE_INDEX = -100


def read_jsonl(path: str | Path, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def sample_uid(sample: dict[str, Any], fallback: str = "") -> str:
    for key in ("uid", "task_id", "id", "sample_id"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return fallback


def label_field(sample: dict[str, Any]) -> str:
    if "labels" in sample:
        return "labels"
    if "label" in sample:
        return "label"
    raise KeyError("sample must contain either 'labels' or 'label'")


def labels_of(sample: dict[str, Any]) -> list[int]:
    return [int(x) for x in sample[label_field(sample)]]


def set_labels(sample: dict[str, Any], labels: list[int]) -> None:
    sample[label_field(sample)] = [int(x) for x in labels]


def supervised_indices(labels: list[int], ignore_index: int = IGNORE_INDEX) -> list[int]:
    return [i for i, value in enumerate(labels) if int(value) != ignore_index]


def copy_with_masked_labels(
    sample: dict[str, Any],
    keep_positions: set[int],
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
    local = copy.deepcopy(sample)
    labels = labels_of(local)
    for pos in supervised_indices(labels, ignore_index=ignore_index):
        if pos not in keep_positions:
            labels[pos] = ignore_index
    set_labels(local, labels)
    return local


def load_score_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load score JSONL keyed by uid/task_id/id/sample_id."""
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(read_jsonl(path)):
        uid = sample_uid(row, fallback=str(idx))
        out[uid] = row
    return out


def numeric_list(values: Any) -> list[float]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise TypeError("expected a list of numeric scores")
    out: list[float] = []
    for value in values:
        if value is None:
            out.append(float("nan"))
        else:
            out.append(float(value))
    return out


def finite_percentile(values: list[float], percentile: float) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return float("nan")
    percentile = max(0.0, min(100.0, float(percentile)))
    if len(finite) == 1:
        return finite[0]
    rank = percentile / 100.0 * (len(finite) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return finite[lo]
    frac = rank - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def to_2d_long_tensor(values: Any, device: Any = None):
    import torch

    tensor = torch.as_tensor(values, dtype=torch.long, device=device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"expected 1D/2D token tensor, got shape={tuple(tensor.shape)}")
    return tensor


def resolve_model_device(model: Any, explicit_device: str | None = None):
    import torch

    if explicit_device:
        return torch.device(explicit_device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_code_block_or_raw(text: str) -> str:
    blocks = re.findall(r"```(?:[A-Za-z0-9_+#.-]+)?\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return text.strip()


def replace_response_fields(sample: dict[str, Any], response: str) -> dict[str, Any]:
    local = copy.deepcopy(sample)
    for key in ("response", "fim_completion", "target", "completion"):
        if key in local:
            local[key] = response
    for msg in local.get("messages", []) or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            msg["content"] = response
    return local
