from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


HUMANEVAL_TASK_TO_OFFICIAL = {
    "single_line": "single-line",
    "multi_line": "multi-line",
    "random_span": "random-span",
    "random_span_light": "random-span-light",
}

SAFIM_TASK_TO_OFFICIAL = {
    "algorithmic_block": "block",
    "control_flow_expression": "control",
    "api_function_call": "api",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_chatml_messages(language: str, prefix: str, suffix: str) -> list[dict[str, str]]:
    lang = language.strip() or "code"
    return [
        {"role": "system", "content": f"You are a {lang} code infilling assistant."},
        {
            "role": "user",
            "content": (
                f"Fill the [MASK] in the {lang} code. Return only the missing {lang} code, "
                "without Markdown fences or explanation.\n\n"
                f"* Incomplete Code:\n{prefix}[MASK]{suffix}"
            ),
        },
    ]


def render_chatml_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```[A-Za-z0-9_+#.-]*\s*\n(.*)\n```$", stripped, re.S)
    if match:
        return match.group(1).strip("\n")
    return text


def sanitize_completion(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return strip_markdown_fence(text)


def load_prediction_map(path: Path) -> dict[str, list[str]]:
    pred_map: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        uid = str(row.get("uid") or row.get("id") or "")
        if not uid:
            continue
        values = None
        for key in ("predictions", "completions", "raw_generations"):
            if key in row:
                values = row[key]
                break
        if values is None:
            for key in ("prediction", "completion", "text", "raw_generation"):
                if key in row:
                    values = [row[key]]
                    break
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        preds: list[str] = []
        for item in values:
            if isinstance(item, dict):
                item = item.get("text") or item.get("prediction") or item.get("completion") or item.get("content") or ""
            preds.append(sanitize_completion(str(item)))
        pred_map[uid] = preds
    return pred_map


def pass_at_1_from_results(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1 for row in results if row.get("passed")) / len(results)
