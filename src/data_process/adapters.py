from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schema import MASK_TOKEN, make_canonical_sample, normalize_language, render_messages, stable_hash

PRE_RE = re.compile(r"<PRE>(.*?)<SUF>(.*?)<MID>", re.DOTALL)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|:=|&&|\|\||[-+*/%&|^!~<>=.:;,{}()\[\]]|\S")


def infer_language(row: dict[str, Any], override: str = "auto") -> str:
    if override and override.lower() != "auto":
        return normalize_language(override)
    for key in ("language", "lang"):
        if row.get(key):
            return normalize_language(row[key])
    text = " ".join(str(row.get(k, ""))[:500].lower() for k in ("prompt", "task_type", "source_dataset"))
    if "java" in text:
        return "Java"
    if "python" in text or " py " in text:
        return "Python"
    if "javascript" in text or " js " in text:
        return "JavaScript"
    if "golang" in text or " go " in f" {text} ":
        return "Go"
    return "Go"


def normalize_marker_payload(text: str) -> str:
    # Keep trailing newlines: they are part of the FIM boundary and prevent
    # target code from being glued to the previous prefix line.
    if text.startswith("\n"):
        text = text[1:]
    if text.startswith(" "):
        text = text[1:]
    return text


def extract_pre_suf_mid(prompt: str) -> tuple[str, str] | None:
    match = PRE_RE.search(prompt)
    if not match:
        return None
    return normalize_marker_payload(match.group(1)), normalize_marker_payload(match.group(2))


def rough_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def nonempty_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def strip_cjk_comments(text: str, language: str) -> tuple[str, int]:
    """Remove comments containing CJK chars while preserving code and newlines."""
    lang = normalize_language(language).lower()
    slash_comment = lang in {"go", "java", "javascript", "typescript", "c", "cpp", "c#", "csharp"}
    hash_comment = lang in {"python", "ruby", "shell"}
    out: list[str] = []
    i = 0
    removed = 0
    quote: str | None = None
    escaped = False
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if quote:
            out.append(ch)
            if quote != "`" and escaped:
                escaped = False
            elif quote != "`" and ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in {'"', "'", "`"}:
            quote = ch
            out.append(ch)
            i += 1
            continue
        if slash_comment and ch == "/" and nxt == "/":
            end = text.find("\n", i)
            if end < 0:
                comment, newline, i = text[i:], "", n
            else:
                comment, newline, i = text[i:end], text[end], end + 1
            if CJK_RE.search(comment):
                removed += 1
                out.append(newline)
            else:
                out.append(comment + newline)
            continue
        if slash_comment and ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            if end < 0:
                comment, i = text[i:], n
            else:
                comment, i = text[i:end + 2], end + 2
            if CJK_RE.search(comment):
                removed += 1
                out.append("\n" * comment.count("\n"))
            else:
                out.append(comment)
            continue
        if hash_comment and ch == "#":
            end = text.find("\n", i)
            if end < 0:
                comment, newline, i = text[i:], "", n
            else:
                comment, newline, i = text[i:end], text[end], end + 1
            if CJK_RE.search(comment):
                removed += 1
                out.append(newline)
            else:
                out.append(comment + newline)
            continue
        out.append(ch)
        i += 1
    return "".join(out), removed


def _target_from_row(row: dict[str, Any]) -> str:
    return str(row.get("target", row.get("response", row.get("fim_completion", ""))) or "")


def _adapt_chatml(row: dict[str, Any], index: int, language: str) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    target = _target_from_row(row)
    if not target:
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                target = str(msg.get("content", ""))
    user = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user = str(msg.get("content", ""))
            break
    if MASK_TOKEN not in user or not target:
        return None
    out = dict(row)
    out.setdefault("uid", f"chatml_{stable_hash(json.dumps(row, ensure_ascii=False, sort_keys=True))}")
    out.setdefault("raw_id", out["uid"])
    out["language"] = language
    out["target"] = target
    out.setdefault("only_last_turn_loss", True)
    return out


def adapt_row(
    row: dict[str, Any],
    index: int,
    *,
    language: str = "auto",
    source_dataset: str = "auto",
    strip_cjk: bool = True,
    max_target_nonempty_lines: int = 10,
    max_target_rough_tokens: int = 192,
    max_target_chars: int = 1024,
) -> tuple[dict[str, Any] | None, str | None]:
    lang = infer_language(row, language)
    chatml = _adapt_chatml(row, index, lang)
    if chatml is not None:
        return chatml, None

    target = _target_from_row(row)
    source_format = "unknown"
    if "prefix" in row and "suffix" in row and target:
        prefix = str(row.get("prefix") or "")
        suffix = str(row.get("suffix") or "")
        source_format = "canonical_prefix_target_suffix"
    else:
        prompt = str(row.get("prompt") or "")
        parts = extract_pre_suf_mid(prompt)
        if parts is None or not target:
            return None, "missing_prefix_suffix_or_target"
        prefix, suffix = parts
        source_format = "huawei_pre_suf_mid_prompt"

    removed_comments = {"prefix": 0, "target": 0, "suffix": 0}
    if strip_cjk:
        prefix, removed_comments["prefix"] = strip_cjk_comments(prefix, lang)
        target, removed_comments["target"] = strip_cjk_comments(target, lang)
        suffix, removed_comments["suffix"] = strip_cjk_comments(suffix, lang)
        if not target.strip():
            return None, "empty_target_after_comment_strip"

    target_lines = nonempty_line_count(target)
    target_tokens = rough_token_count(target)
    target_chars = len(target)
    if max_target_nonempty_lines > 0 and target_lines > max_target_nonempty_lines:
        return None, "target_too_many_nonempty_lines"
    if max_target_rough_tokens > 0 and target_tokens > max_target_rough_tokens:
        return None, "target_too_many_rough_tokens"
    if max_target_chars > 0 and target_chars > max_target_chars:
        return None, "target_too_many_chars"

    raw_id = str(row.get("raw_id") or row.get("task_id") or row.get("uid") or f"row_{index}")
    uid = str(row.get("uid") or f"fim_{normalize_language(lang).lower()}_{stable_hash(raw_id + prefix + target + suffix)}")
    dataset = source_dataset if source_dataset != "auto" else str(row.get("source_dataset") or "raw_fim")
    return make_canonical_sample(
        uid=uid,
        raw_id=raw_id,
        language=lang,
        prefix=prefix,
        target=target,
        suffix=suffix,
        source_dataset=dataset,
        task_type=str(row.get("task_type") or f"{normalize_language(lang).lower()}_fim_completion"),
        metadata={
            "raw_index": index,
            "source_format": source_format,
            "target_nonempty_lines": target_lines,
            "target_rough_tokens": target_tokens,
            "target_chars": target_chars,
            "strip_cjk_comments": strip_cjk,
            "removed_cjk_comments": removed_comments,
        },
    ), None
