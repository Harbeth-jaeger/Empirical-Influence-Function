from __future__ import annotations

import copy
from typing import Any

from .common import extract_code_block_or_raw, replace_response_fields, sample_uid


STAGE_INSTRUCTIONS = {
    "rename": "Rename variables to be descriptive while preserving behavior.",
    "modularize": "Refactor the code into clearer helper functions while preserving behavior.",
    "planning": "Add concise natural-language planning comments while preserving behavior.",
}


def build_cleaning_messages(prompt: str, code: str, language: str = "", stage: str = "rename") -> list[dict[str, str]]:
    instruction = STAGE_INSTRUCTIONS.get(stage, stage)
    lang = f" {language}" if language else ""
    return [
        {"role": "system", "content": f"You are an expert{lang} code refactoring assistant. Return only code."},
        {"role": "user", "content": f"[Prompt]\n{prompt}\n\n[Code]\n{code}\n\n[Task]\n{instruction}"},
    ]


def apply_llm_cleaning_rewrites(
    samples: list[dict[str, Any]],
    rewrite_rows: dict[str, dict[str, Any]],
    rewrite_key: str = "cleaned_response",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply externally generated LLM-cleaned targets to compact/canonical rows."""
    out: list[dict[str, Any]] = []
    replaced = 0
    missing = 0
    for idx, sample in enumerate(samples):
        uid = sample_uid(sample, fallback=str(idx))
        row = rewrite_rows.get(uid)
        if not row or not row.get(rewrite_key):
            missing += 1
            out.append(copy.deepcopy(sample))
            continue
        cleaned = extract_code_block_or_raw(str(row[rewrite_key]))
        out.append(replace_response_fields(sample, cleaned))
        replaced += 1
    return out, {
        "method": "llm_code_cleaning",
        "rewrite_key": rewrite_key,
        "replaced_samples": replaced,
        "missing_rewrite_rows": missing,
    }

