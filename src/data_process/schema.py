from __future__ import annotations

import hashlib
from typing import Any

MASK_TOKEN = "[MASK]"

LANG_ALIASES = {
    "go": "Go",
    "golang": "Go",
    "java": "Java",
    "python": "Python",
    "py": "Python",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "c": "C",
    "cpp": "CPP",
    "c++": "CPP",
    "csharp": "C#",
    "c#": "C#",
    "cs": "C#",
}


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def normalize_language(value: Any, default: str = "Go") -> str:
    text = str(value or default).strip()
    if not text:
        text = default
    return LANG_ALIASES.get(text.lower(), text)


def system_prompt(language: str) -> str:
    lang = normalize_language(language)
    return f"You are a {lang} code completion assistant."


def user_prompt(prefix: str, suffix: str, language: str) -> str:
    lang = normalize_language(language)
    return (
        f"Fill the missing part in the {lang} code. Return only the missing {lang} code, "
        "without Markdown fences or explanation.\n\n"
        "* Incomplete Code:\n"
        f"{prefix}{MASK_TOKEN}{suffix}"
    )


def render_messages(prefix: str, target: str, suffix: str, language: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(language)},
        {"role": "user", "content": user_prompt(prefix, suffix, language)},
        {"role": "assistant", "content": target},
    ]


def make_canonical_sample(
    *,
    uid: str,
    language: str,
    prefix: str,
    target: str,
    suffix: str,
    source_dataset: str,
    split: str = "train",
    task_type: str | None = None,
    raw_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lang = normalize_language(language)
    sample = {
        "uid": uid,
        "raw_id": raw_id or uid,
        "source_dataset": source_dataset,
        "split": split,
        "language": lang,
        "task_type": task_type or f"{lang.lower()}_fim_completion",
        "prefix": prefix,
        "target": target,
        "suffix": suffix,
        "full_code": prefix + target + suffix,
        "messages": render_messages(prefix, target, suffix, lang),
        "only_last_turn_loss": True,
        "metadata": dict(metadata or {}),
    }
    return sample
