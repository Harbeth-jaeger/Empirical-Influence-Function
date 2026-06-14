#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.go_singleline_fim_exp.annotate_chatml_with_src_annotate import (  # noqa: E402
    annotate_row,
    build_messages,
    chatml_text,
    get_chatml_offsets,
    load_cache,
    load_jsonl,
    load_seed_compact,
    row_key,
    save_cache,
    write_jsonl_atomic,
)
from src.annotate.neural_annot import (  # noqa: E402
    _rate_limited_chat_completion,
    get_thread_local_openai_client,
)


WORD_RE = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|\"[^\"\n]*\"|'[^'\n]*'|“[^”\n]*”|‘[^’\n]*’")
TRIPLE_STRING_RE = re.compile("\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?'''")


def _span_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _token_indices_for_span(qwen_tokens: list[dict[str, Any]], start: int, end: int) -> list[int]:
    out = []
    for idx, tok in enumerate(qwen_tokens):
        if _span_overlaps(int(tok.get("char_start", 0)), int(tok.get("char_end", 0)), start, end):
            out.append(idx)
    return out


def _word_items(text: str, base_offset: int, *, max_items: int = 160) -> list[dict[str, Any]]:
    items = []
    quote_pairs = {'"': '"', "'": "'", '“': '”', '‘': '’'}
    for match in WORD_RE.finditer(text):
        surface = match.group(0)
        start = base_offset + match.start()
        end = base_offset + match.end()
        if len(surface) >= 2 and surface[0] in quote_pairs and surface[-1] == quote_pairs[surface[0]]:
            surface = surface[1:-1]
            start += 1
            end -= 1
        items.append({"idx": len(items), "text": surface, "start": start, "end": end})
        if len(items) >= max_items:
            break
    return items


def _tag_items(items: list[dict[str, Any]], section: str) -> list[dict[str, Any]]:
    for item in items:
        item["section"] = section
    return items


def _norm_doc_code_text(text: str) -> str:
    text = str(text).strip().strip('"\'“”‘’`')
    text = text.lstrip('.')
    return re.sub(r"[^A-Za-z0-9_]+", "", text).lower()


def _is_high_value_literal(text: str) -> bool:
    norm = _norm_doc_code_text(text)
    raw = str(text).strip()
    if raw[:1] in {'"', "'", '“', '‘'} or raw[-1:] in {'"', "'", '”', '’'}:
        return True
    return norm in {"true", "false", "none", "null", "yes", "no", "high", "low", "normal", "ok", "error"}


DOCSTRING_MATCH_STOPWORDS = {
    "a", "an", "the", "and", "or", "if", "in", "of", "to", "for", "with",
    "is", "are", "be", "this", "that", "whether", "value", "values",
    "string", "returns", "return", "current", "limit", "threshold", "file",
}


DOCSTRING_TYPE_STOPWORDS = {
    "default", "defaults", "none", "true", "false", "the", "a", "an", "of", "or", "and", "class",
}
DOCSTRING_TYPE_ALIASES = {
    "array", "ndarray", "tensor", "list", "tuple", "dict", "dictionary", "set",
    "bool", "boolean", "int", "integer", "float", "double", "str", "string",
    "callable", "iterable", "sequence", "dataframe", "series", "path", "file",
    "any", "optional", "pandas",
}
DOCSTRING_TYPE_GENERIC_REJECTS = {
    "name", "number", "value", "values", "unique", "output", "outputs", "result", "results",
}


def _split_type_terms(type_text: str) -> list[str]:
    cleaned = str(type_text)
    cleaned = re.sub(r":[A-Za-z_][\w-]*:`([^`]+)`", r" \1 ", cleaned)
    cleaned = cleaned.replace("`", " ")
    cleaned = cleaned.replace("|", " ").replace("/", " ").replace(" or ", " ")
    return [part.strip(" ,.;:()[]{}") for part in re.split(r"\s+|,", cleaned) if part.strip(" ,.;:()[]{}")]


def _looks_like_type_text(type_text: str) -> bool:
    terms = [t for t in _split_type_terms(type_text) if _norm_doc_code_text(t) not in DOCSTRING_TYPE_STOPWORDS]
    if not terms or len(terms) > 5:
        return False
    if len(terms) == 1 and _norm_doc_code_text(terms[0]) in DOCSTRING_TYPE_GENERIC_REJECTS:
        return False
    joined = str(type_text).strip()
    for term in terms:
        norm = _norm_doc_code_text(term)
        if not norm:
            continue
        if norm in DOCSTRING_TYPE_GENERIC_REJECTS:
            continue
        if norm in DOCSTRING_TYPE_ALIASES:
            return True
        if "." in term or "[" in term or "]" in term:
            return True
        if term[:1].isupper():
            return True
    return False


def _looks_like_bare_type_line(type_text: str) -> bool:
    stripped = str(type_text).strip()
    lower = stripped.lower().rstrip(":")
    if lower in {"return", "returns", "yield", "yields", "parameter", "parameters", "args", "arguments"}:
        return False
    raw_terms = _split_type_terms(stripped)
    if not raw_terms or len(raw_terms) > 3:
        return False
    for term in raw_terms:
        norm = _norm_doc_code_text(term)
        if not norm or norm in DOCSTRING_TYPE_STOPWORDS or norm in DOCSTRING_MATCH_STOPWORDS:
            return False
        if norm in DOCSTRING_TYPE_ALIASES:
            continue
        if "." in term or "[" in term or "]" in term:
            continue
        if term[:1].isupper():
            continue
        return False
    return True



DOCSTRING_TYPE_PREFIX_STOP_WORDS = {
    "to", "for", "with", "from", "of", "if", "when", "used", "using", "containing", "contains",
    "whose", "which", "that", "is", "are", "be", "in", "on", "by", "as", "returns", "return",
}


def _type_prefix_span(text: str, *, name: str | None = None) -> tuple[int, int] | None:
    """Return the likely type prefix span inside a docstring fact payload."""
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return None
    name_norm = _norm_doc_code_text(name or "").strip("_")
    start_i = 0
    while start_i < len(matches) and _norm_doc_code_text(matches[start_i].group(0)) in {"the", "a", "an"}:
        start_i += 1
    if start_i >= len(matches):
        return None
    if name_norm and _norm_doc_code_text(matches[start_i].group(0)).strip("_") == name_norm and start_i + 1 < len(matches):
        next_segment = text[matches[start_i + 1].start() : matches[start_i + 1].end()]
        if _looks_like_type_text(next_segment):
            start_i += 1

    best: tuple[int, int] | None = None
    saw_type_token = False
    for end_i in range(start_i, min(len(matches), start_i + 8)):
        token_text = matches[end_i].group(0)
        token_norm = _norm_doc_code_text(token_text).strip("_")
        if end_i > start_i and token_norm in DOCSTRING_TYPE_PREFIX_STOP_WORDS:
            break
        if end_i > start_i and _is_high_value_literal(token_text):
            break
        segment = text[matches[start_i].start() : matches[end_i].end()]
        if _looks_like_type_text(segment):
            saw_type_token = True
            best = (matches[start_i].start(), matches[end_i].end())
            if end_i == start_i and token_norm in DOCSTRING_TYPE_ALIASES:
                tail = text[matches[end_i].end() :].lstrip()
                if tail.startswith("["):
                    break
        elif saw_type_token and token_norm not in {"or", "and", "optional", "any"}:
            break
    return best


def _type_and_desc_items_from_span(
    line_items: list[dict[str, Any]],
    start: int,
    end: int,
    *,
    name: str | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    line = _line_text(line_items)
    payload = line[start:end]
    prefix = _type_prefix_span(payload, name=name)
    if prefix is None:
        return "", [], _line_items_in_span(line_items, start, end)
    type_start = start + prefix[0]
    type_end = start + prefix[1]
    type_text = line[type_start:type_end]
    type_items = _line_items_in_span(line_items, type_start, type_end, drop_type_stopwords=True)
    desc_items = _line_items_in_span(line_items, type_end, end)
    return type_text, type_items, desc_items


def _doc_items_by_line(doc_items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for item in doc_items:
        key = (int(item.get("line_start", -1)), str(item.get("line_text", "")))
        groups.setdefault(key, []).append(item)
    return [groups[key] for key in sorted(groups)]


def _items_in_line_span(line_items: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    out = []
    if not line_items:
        return out
    line_start = int(line_items[0].get("line_start", 0))
    abs_start = line_start + start
    abs_end = line_start + end
    for item in line_items:
        if _span_overlaps(int(item["start"]), int(item["end"]), abs_start, abs_end):
            if _norm_doc_code_text(item.get("text", "")) not in DOCSTRING_TYPE_STOPWORDS:
                out.append(item)
    return out


def _doc_type_facts(doc_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for line_items in _doc_items_by_line(doc_items):
        if not line_items:
            continue
        line = str(line_items[0].get("line_text", ""))
        stripped = line.strip()
        section = str(line_items[0].get("doc_section", "other"))
        matches: list[tuple[str, str, int, int, str | None]] = []

        m = re.match(r"^\s*:param\s+(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>.+?)\s*$", line)
        if m and _looks_like_type_text(m.group("type")):
            matches.append(("param", m.group("type"), m.start("type"), m.end("type"), m.group("name")))
        m = re.match(r"^\s*:type\s+(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>.+?)\s*$", line)
        if m and _looks_like_type_text(m.group("type")):
            matches.append(("param", m.group("type"), m.start("type"), m.end("type"), m.group("name")))
        m = re.match(r"^\s*:returns?\s*:?\s*(?P<type>.+?)\s*$", line)
        if m and _looks_like_type_text(m.group("type")):
            matches.append(("return", m.group("type"), m.start("type"), m.end("type"), None))

        m = re.match(r"^\s*(?P<name>[A-Za-z_]\w*)\s*\((?P<type>[^)]+)\)\s*:", line)
        if m and _looks_like_type_text(m.group("type")):
            kind = "return" if section == "returns" or m.group("name").lower() in {"return", "returns"} else "param"
            name = None if kind == "return" and m.group("name").lower() in {"return", "returns"} else m.group("name")
            matches.append((kind, m.group("type"), m.start("type"), m.end("type"), name))

        m = re.match(r"^\s*(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>[^.\n]+)", line)
        if m and not stripped.startswith(":") and _looks_like_type_text(m.group("type")):
            kind = "return" if section == "returns" else "param"
            matches.append((kind, m.group("type"), m.start("type"), m.end("type"), m.group("name")))

        if section == "returns" and not matches and _looks_like_bare_type_line(stripped):
            # Numpy-style return type line without a named output.
            first = line.find(stripped)
            matches.append(("return", stripped, first, first + len(stripped), None))

        for kind, type_text, start, end, name in matches:
            type_items = _items_in_line_span(line_items, start, end)
            if type_items:
                facts.append({"kind": kind, "name": name, "type_text": type_text, "type_items": type_items})
    return facts


def _returned_code_items(code_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, item in enumerate(code_items[:-1]):
        if _norm_doc_code_text(item.get("text", "")) != "return":
            continue
        for cand in code_items[i + 1 : i + 5]:
            norm = _norm_doc_code_text(cand.get("text", ""))
            if norm and norm not in {"if", "else", "elif"}:
                out.append(cand)
                break
    return out


def _extract_doc_and_code_items(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages = build_messages(row)
    text = chatml_text(messages)
    user_start, assistant_start, user_content, assistant_content = get_chatml_offsets(messages)
    marker = "* Incomplete Code:\n"
    marker_pos = user_content.find(marker)
    code_rel_start = marker_pos + len(marker) if marker_pos >= 0 else 0
    code_text = user_content[code_rel_start:]
    code_abs_start = user_start + code_rel_start

    doc_spans: list[tuple[int, int]] = []
    for match in TRIPLE_STRING_RE.finditer(code_text):
        inner_start = code_abs_start + match.start() + 3
        inner_end = code_abs_start + match.end() - 3
        if inner_start < inner_end:
            doc_spans.append((inner_start, inner_end))

    doc_items: list[dict[str, Any]] = []
    for start, end in doc_spans:
        span_text = text[start:end]
        section = "other"
        line_abs = start
        for line_no, line in enumerate(span_text.splitlines(keepends=True)):
            stripped = line.strip()
            lower = stripped.lower().rstrip(":")
            if lower in {"args", "arguments", "parameters", "params"}:
                section = "args"
            elif lower in {"returns", "return", "yields", "yield"}:
                section = "returns"
            elif lower in {"examples", "example"} or stripped.startswith((">>>", "...")) or stripped.startswith("Traceback"):
                section = "examples"
            elif re.fullmatch(r"[-=~`]{3,}", stripped):
                line_abs += len(line)
                continue
            line_items = _word_items(line, line_abs, max_items=80 - len(doc_items))
            for item in line_items:
                item["doc_section"] = section
                item["line_no"] = line_no
                item["line_text"] = line
                item["line_start"] = line_abs
            doc_items.extend(line_items)
            if len(doc_items) >= 80:
                break
            line_abs += len(line)
        if len(doc_items) >= 80:
            break

    mask_rel = code_text.find("[MASK]")
    mask_abs_start = code_abs_start + mask_rel if mask_rel >= 0 else -1
    mask_abs_end = mask_abs_start + len("[MASK]") if mask_abs_start >= 0 else -1

    code_items: list[dict[str, Any]] = []
    cursor = 0
    for start, end in doc_spans:
        rel_start = max(0, start - code_abs_start)
        rel_end = max(0, end - code_abs_start)
        if cursor < rel_start:
            code_items.extend(_tag_items(_word_items(code_text[cursor:rel_start], code_abs_start + cursor, max_items=200 - len(code_items)), "prompt_code"))
        cursor = max(cursor, rel_end)
    if cursor < len(code_text):
        code_items.extend(_tag_items(_word_items(code_text[cursor:], code_abs_start + cursor, max_items=200 - len(code_items)), "prompt_code"))
    if mask_abs_start >= 0:
        code_items = [item for item in code_items if not _span_overlaps(item["start"], item["end"], mask_abs_start, mask_abs_end)]
    code_items.extend(_tag_items(_word_items(assistant_content, assistant_start, max_items=max(0, 240 - len(code_items))), "completion"))

    if doc_items:
        first_doc = min(item["start"] for item in doc_items)
        code_items = [item for item in code_items if item["start"] > first_doc]
    for idx, item in enumerate(doc_items):
        item["idx"] = idx
    for idx, item in enumerate(code_items):
        item["idx"] = idx
    return doc_items, code_items


def _parse_docstring_pairs(text: str) -> list[dict[str, int]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    pairs = parsed.get("pairs", []) if isinstance(parsed, dict) else parsed
    out = []
    if not isinstance(pairs, list):
        return out
    for pair in pairs:
        try:
            if isinstance(pair, dict):
                d = int(pair.get("doc", pair.get("doc_idx", pair.get("i"))))
                c = int(pair.get("code", pair.get("code_idx", pair.get("j"))))
            else:
                d = int(pair[0])
                c = int(pair[1])
            out.append({"doc": d, "code": c})
        except Exception:
            continue
    return out


DOCFACT_SEMANTIC_STOPWORDS = DOCSTRING_MATCH_STOPWORDS | DOCSTRING_TYPE_STOPWORDS | {
    "description", "param", "params", "parameter", "parameters", "argument", "arguments",
    "optional", "object", "name", "number", "used", "using", "use", "set", "check",
}


def _line_groups(doc_items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return _doc_items_by_line(doc_items)


def _line_text(line_items: list[dict[str, Any]]) -> str:
    return str(line_items[0].get("line_text", "")) if line_items else ""


def _line_section(line_items: list[dict[str, Any]]) -> str:
    return str(line_items[0].get("doc_section", "other")) if line_items else "other"


def _line_items_in_span(line_items: list[dict[str, Any]], start: int, end: int, *, drop_type_stopwords: bool = False) -> list[dict[str, Any]]:
    if not line_items:
        return []
    line_start = int(line_items[0].get("line_start", 0))
    abs_start = line_start + max(0, start)
    abs_end = line_start + max(start, end)
    out = []
    for item in line_items:
        if not _span_overlaps(int(item["start"]), int(item["end"]), abs_start, abs_end):
            continue
        if drop_type_stopwords and _norm_doc_code_text(item.get("text", "")) in DOCSTRING_TYPE_STOPWORDS:
            continue
        out.append(item)
    return out


def _is_fact_line(line: str, section: str) -> bool:
    stripped = line.strip()
    lower = stripped.lower().rstrip(":")
    if not stripped or re.fullmatch(r"[-=~`]{3,}", stripped):
        return True
    if lower in {"args", "arguments", "parameters", "params", "returns", "return", "yields", "yield", "examples", "example"}:
        return True
    if stripped.startswith((">>>", "...")) or stripped.startswith("Traceback"):
        return True
    patterns = [
        r"^\s*:(?:param|type)\s+[A-Za-z_]\w*\s*:",
        r"^\s*:returns?\s*:",
        r"^\s*@(?:param|return|returns)\b",
        r"^\s*[A-Za-z_]\w*\s*\([^)]+\)\s*:",
        r"^\s*[A-Za-z_]\w*\s*:\s*[^\n]+",
    ]
    if any(re.match(pat, line) for pat in patterns):
        return True
    if section == "returns" and _looks_like_bare_type_line(stripped):
        return True
    return False


def _continuation_items(lines: list[list[dict[str, Any]]], start_idx: int, section: str, *, max_lines: int = 3) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_items in lines[start_idx + 1 : start_idx + 1 + max_lines]:
        line = _line_text(line_items)
        if _line_section(line_items) != section or _is_fact_line(line, section):
            break
        items.extend(line_items)
    return items


def _make_doc_fact(
    *,
    section: str,
    name: str | None,
    type_text: str,
    desc_text: str,
    type_items: list[dict[str, Any]],
    desc_items: list[dict[str, Any]],
    source: str = "rule",
) -> dict[str, Any] | None:
    if section not in {"args", "returns"}:
        return None
    type_items = [item for item in type_items if _norm_doc_code_text(item.get("text", "")) not in DOCSTRING_TYPE_STOPWORDS]
    desc_items = [item for item in desc_items if _norm_doc_code_text(item.get("text", ""))]
    if not type_items and not desc_items:
        return None
    return {
        "section": section,
        "name": name,
        "type_text": type_text.strip(),
        "desc_text": desc_text.strip(),
        "type_items": type_items,
        "desc_items": desc_items,
        "source": source,
    }


def _doc_facts_from_rules(doc_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    lines = _line_groups(doc_items)
    for line_idx, line_items in enumerate(lines):
        if not line_items:
            continue
        line = _line_text(line_items)
        stripped = line.strip()
        lower_heading = stripped.lower().rstrip(":")
        section = _line_section(line_items)
        if (
            not stripped
            or lower_heading in {"args", "arguments", "parameters", "params", "returns", "return", "yields", "yield", "examples", "example"}
            or re.fullmatch(r"[-=~`]{3,}", stripped)
            or stripped.startswith((">>>", "..."))
            or stripped.startswith("Traceback")
        ):
            continue
        line_facts: list[dict[str, Any] | None] = []

        m = re.match(r"^\s*:type\s+(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>.+?)\s*$", line)
        if m:
            type_text = m.group("type")
            type_items = _line_items_in_span(line_items, m.start("type"), m.end("type"), drop_type_stopwords=True)
            line_facts.append(_make_doc_fact(section="args", name=m.group("name"), type_text=type_text, desc_text="", type_items=type_items, desc_items=[], source="rule"))

        m = re.match(r"^\s*:param\s+(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<desc>.+?)\s*$", line)
        if m:
            desc = m.group("desc")
            type_text, type_items, desc_items = _type_and_desc_items_from_span(line_items, m.start("desc"), m.end("desc"), name=m.group("name"))
            if not type_items:
                desc_items += _continuation_items(lines, line_idx, section)
            line_facts.append(_make_doc_fact(section="args", name=m.group("name"), type_text=type_text, desc_text=desc, type_items=type_items, desc_items=desc_items, source="rule"))

        m = re.match(r"^\s*@param\s+(?P<name>[A-Za-z_]\w*)\s+(?P<desc>.+?)\s*$", line)
        if m:
            desc = m.group("desc")
            type_text, type_items, desc_items = _type_and_desc_items_from_span(line_items, m.start("desc"), m.end("desc"), name=m.group("name"))
            if not type_items:
                desc_items += _continuation_items(lines, line_idx, "args")
            line_facts.append(_make_doc_fact(section="args", name=m.group("name"), type_text=type_text, desc_text=desc, type_items=type_items, desc_items=desc_items, source="rule"))

        m = re.match(r"^\s*:returns?\s*:?\s*(?P<desc>.+?)\s*$", line)
        if m:
            desc = m.group("desc")
            type_text, type_items, desc_items = _type_and_desc_items_from_span(line_items, m.start("desc"), m.end("desc"))
            if not type_items:
                desc_items += _continuation_items(lines, line_idx, "returns")
            line_facts.append(_make_doc_fact(section="returns", name=None, type_text=type_text, desc_text=desc, type_items=type_items, desc_items=desc_items, source="rule"))

        m = re.match(r"^\s*@returns?\s+(?P<desc>.+?)\s*$", line)
        if m:
            desc = m.group("desc")
            type_text, type_items, desc_items = _type_and_desc_items_from_span(line_items, m.start("desc"), m.end("desc"))
            if not type_items:
                desc_items += _continuation_items(lines, line_idx, "returns")
            line_facts.append(_make_doc_fact(section="returns", name=None, type_text=type_text, desc_text=desc, type_items=type_items, desc_items=desc_items, source="rule"))

        m = re.match(r"^\s*Returns?\s*:\s*(?P<desc>.+?)\s*$", line, flags=re.IGNORECASE)
        if m:
            desc = m.group("desc")
            type_text, type_items, desc_items = _type_and_desc_items_from_span(line_items, m.start("desc"), m.end("desc"))
            if not type_items:
                desc_items += _continuation_items(lines, line_idx, "returns")
            line_facts.append(_make_doc_fact(section="returns", name=None, type_text=type_text, desc_text=desc, type_items=type_items, desc_items=desc_items, source="rule"))

        m = re.match(r"^\s*\*{1,2}`?(?P<name>[A-Za-z_]\w*)`?\*{1,2}\s*:\s*(?P<rest>[^\n]*)$", line)
        if m and section in {"args", "returns"}:
            rest = m.group("rest").strip()
            kind = section
            name = m.group("name")
            type_text, type_items, desc_items = _type_and_desc_items_from_span(line_items, m.start("rest"), m.end("rest"), name=name if kind == "args" else None)
            if kind == "returns" and _looks_like_type_text(name) and not type_items:
                type_text = name
                type_items = _line_items_in_span(line_items, m.start("name"), m.end("name"), drop_type_stopwords=True)
            if not type_items:
                desc_items += _continuation_items(lines, line_idx, section)
            line_facts.append(_make_doc_fact(section=kind, name=name if kind == "args" else None, type_text=type_text, desc_text=rest, type_items=type_items, desc_items=desc_items, source="rule"))

        m = re.match(r"^\s*\((?P<type>[^)]+)\)\s*(?P<desc>.*)$", line)
        if m and section == "returns":
            type_items = _line_items_in_span(line_items, m.start("type"), m.end("type"), drop_type_stopwords=True)
            desc_items = _line_items_in_span(line_items, m.start("desc"), m.end("desc")) + _continuation_items(lines, line_idx, section)
            line_facts.append(_make_doc_fact(section="returns", name=None, type_text=m.group("type"), desc_text=m.group("desc"), type_items=type_items, desc_items=desc_items, source="rule"))

        m = re.match(r"^\s*(?P<name>[A-Za-z_]\w*)\s*\((?P<type>[^)]+)\)\s*:\s*(?P<desc>.*)$", line)
        if m and section in {"args", "returns"}:
            kind = "returns" if section == "returns" or m.group("name").lower() in {"return", "returns"} else "args"
            name = None if kind == "returns" and m.group("name").lower() in {"return", "returns"} else m.group("name")
            type_items = _line_items_in_span(line_items, m.start("type"), m.end("type"), drop_type_stopwords=True)
            desc_items = _line_items_in_span(line_items, m.start("desc"), m.end("desc"))
            if not type_items:
                desc_items += _continuation_items(lines, line_idx, section)
            line_facts.append(_make_doc_fact(section=kind, name=name, type_text=m.group("type"), desc_text=m.group("desc"), type_items=type_items, desc_items=desc_items, source="rule"))

        m = re.match(r"^\s*(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<rest>[^\n]*)$", line)
        if m and not line_facts and not stripped.startswith(":") and section in {"args", "returns"}:
            rest = m.group("rest").strip()
            type_text, type_items, desc_items = _type_and_desc_items_from_span(line_items, m.start("rest"), m.end("rest"), name=m.group("name"))
            if not type_items:
                desc_items += _continuation_items(lines, line_idx, section)
            line_facts.append(_make_doc_fact(section=section, name=m.group("name"), type_text=type_text, desc_text=rest, type_items=type_items, desc_items=desc_items, source="rule"))

        if section == "returns" and not line_facts:
            type_text, type_items, desc_items = _type_and_desc_items_from_span(line_items, 0, len(line))
            if type_items:
                desc_items += _continuation_items(lines, line_idx, section)
                line_facts.append(_make_doc_fact(section="returns", name=None, type_text=type_text, desc_text=line.strip(), type_items=type_items, desc_items=desc_items, source="rule"))
        if section == "returns" and not line_facts and _looks_like_bare_type_line(stripped):
            start = line.find(stripped)
            type_items = _line_items_in_span(line_items, start, start + len(stripped), drop_type_stopwords=True)
            desc_items = _continuation_items(lines, line_idx, section)
            line_facts.append(_make_doc_fact(section="returns", name=None, type_text=stripped, desc_text="", type_items=type_items, desc_items=desc_items, source="rule"))

        for fact in line_facts:
            if fact is not None:
                facts.append(fact)
    return facts

def _semantic_terms_from_fact(fact: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in list(fact.get("desc_items") or []):
        norm = _norm_doc_code_text(item.get("text", "")).strip("_")
        if not norm or norm in DOCFACT_SEMANTIC_STOPWORDS:
            continue
        out.append(item)
    # Return names like `weight: array` are semantically useful cues for returned variables.
    if fact.get("section") == "returns" and fact.get("name"):
        for item in list(fact.get("desc_items") or []) + list(fact.get("type_items") or []):
            pass
    return out


def _code_item_norms(item: dict[str, Any]) -> set[str]:
    text = str(item.get("text", ""))
    norm = _norm_doc_code_text(text).strip("_")
    out = {norm} if norm else set()
    for part in re.split(r"_+", norm):
        if part:
            out.add(part)
    # `.storage` and `=color` should also match `storage` / `color`.
    trimmed = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text)
    tnorm = _norm_doc_code_text(trimmed).strip("_")
    if tnorm:
        out.add(tnorm)
    return out


def _target_items_for_name(code_items: list[dict[str, Any]], name: str | None) -> list[dict[str, Any]]:
    if not name:
        return []
    name_norm = _norm_doc_code_text(name).strip("_")
    if not name_norm:
        return []
    return [item for item in code_items if name_norm in _code_item_norms(item)]


def _doc_facts_to_pairs(doc_items: list[dict[str, Any]], code_items: list[dict[str, Any]], *, max_edges: int) -> list[dict[str, Any]]:
    facts = _doc_facts_from_rules(doc_items)
    returned_items = _returned_code_items(code_items)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()

    def add(doc_item: dict[str, Any], code_item: dict[str, Any], *, subtype: str, source: str) -> None:
        if len(pairs) >= max_edges:
            return
        key = (int(doc_item["idx"]), int(code_item["idx"]), subtype)
        if key in seen:
            return
        seen.add(key)
        pairs.append({"doc": key[0], "code": key[1], "source": source, "subtype": subtype})

    for fact in facts:
        source_prefix = "DocFactLLM" if fact.get("source") == "llm" else "DocFact"
        name_targets = _target_items_for_name(code_items, fact.get("name"))
        semantic_sources = _semantic_terms_from_fact(fact)
        return_literal_targets: list[dict[str, Any]] = []
        if fact.get("section") == "returns":
            literal_norms = {
                _norm_doc_code_text(item.get("text", "")).strip("_")
                for item in semantic_sources
                if _is_high_value_literal(item.get("text", ""))
            }
            return_literal_targets = [
                item for item in code_items
                if item.get("section") == "completion" and _norm_doc_code_text(item.get("text", "")).strip("_") in literal_norms
            ]
        return_targets = return_literal_targets or (returned_items if fact.get("section") == "returns" else [])
        type_targets = name_targets or return_targets
        if fact.get("section") == "returns" and name_targets:
            type_targets = name_targets + [item for item in return_targets if item not in name_targets]

        for type_item in fact.get("type_items") or []:
            for target in type_targets:
                add(type_item, target, subtype="type", source=f"{source_prefix}Type")

        for doc_item in semantic_sources:
            dnorm = _norm_doc_code_text(doc_item.get("text", "")).strip("_")
            if not dnorm:
                continue
            for code_item in code_items:
                if fact.get("section") == "returns" and _is_high_value_literal(doc_item.get("text", "")):
                    if code_item.get("section") == "completion" and _norm_doc_code_text(code_item.get("text", "")).strip("_") == dnorm:
                        add(doc_item, code_item, subtype="semantic", source=f"{source_prefix}Semantic")
                elif dnorm in _code_item_norms(code_item):
                    add(doc_item, code_item, subtype="semantic", source=f"{source_prefix}Semantic")

    return pairs[:max_edges]


def _candidate_docstring_type_pairs(
    doc_items: list[dict[str, Any]],
    code_items: list[dict[str, Any]],
    *,
    max_edges: int,
) -> list[dict[str, Any]]:
    facts = _doc_type_facts(doc_items)
    if not facts:
        return []
    returned_items = _returned_code_items(code_items)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for fact in facts:
        targets: list[dict[str, Any]] = []
        name_norm = _norm_doc_code_text(fact.get("name", "") or "")
        if fact.get("kind") == "param" and name_norm:
            targets = [item for item in code_items if _norm_doc_code_text(item.get("text", "")) == name_norm]
        elif fact.get("kind") == "return":
            if name_norm:
                targets.extend(item for item in code_items if _norm_doc_code_text(item.get("text", "")) == name_norm)
            targets.extend(returned_items)
        for type_item in fact.get("type_items", []):
            for target in targets:
                key = (int(type_item["idx"]), int(target["idx"]))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"doc": key[0], "code": key[1], "source": "DocstringType", "subtype": "type"})
                if len(pairs) >= max_edges:
                    return pairs
    return pairs


def _candidate_docstring_pairs(
    doc_items: list[dict[str, Any]],
    code_items: list[dict[str, Any]],
    *,
    max_edges: int,
) -> list[dict[str, Any]]:
    """High-precision docstring return-answer edges.

    Only connect candidate answers in the docstring return description to the
    same answer appearing in the hidden completion, and only when that answer is
    absent from non-docstring prompt code.  Args/parameter descriptions are
    intentionally ignored because signature def-use already covers them.
    """
    prompt_norms = {
        _norm_doc_code_text(item.get("text", ""))
        for item in code_items
        if item.get("section") == "prompt_code"
    }
    candidates: list[tuple[tuple[int, int], dict[str, Any]]] = []
    seen: set[tuple[int, int]] = set()
    for doc in doc_items:
        if doc.get("doc_section") != "returns":
            continue
        dnorm = _norm_doc_code_text(doc.get("text", ""))
        if len(dnorm) <= 1 or dnorm in DOCSTRING_MATCH_STOPWORDS:
            continue
        for code in code_items:
            if code.get("section") != "completion":
                continue
            cnorm = _norm_doc_code_text(code.get("text", ""))
            if dnorm != cnorm or cnorm in prompt_norms:
                continue
            if not (_is_high_value_literal(doc.get("text", "")) or _is_high_value_literal(code.get("text", ""))):
                continue
            key = (int(doc["idx"]), int(code["idx"]))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(((int(code["idx"]), int(doc["idx"])), {"doc": key[0], "code": key[1], "source": "DocstringReturnCandidate"}))
    candidates.sort(key=lambda item: item[0])
    return [pair for _, pair in candidates[:max_edges]]


def add_docstring_to_code_edges(
    *,
    row: dict[str, Any],
    record: dict[str, Any],
    tokenizer: Any,
    max_edges: int,
    use_llm: bool = False,
    use_type_edges: bool = True,
) -> dict[str, Any]:
    if max_edges <= 0:
        return record
    doc_items, code_items = _extract_doc_and_code_items(row)
    if not doc_items or not code_items:
        record.setdefault("annotation_meta", {})["docstring_edges_added"] = 0
        return record

    pairs = _doc_facts_to_pairs(doc_items, code_items, max_edges=max_edges)
    if not use_type_edges:
        pairs = [pair for pair in pairs if pair.get("subtype") != "type"]
    # `use_llm` is reserved for a DocFact extractor fallback.  Direct LLM edge
    # generation is intentionally disabled so every docstring edge remains
    # traceable to a parsed Args/Returns fact and a deterministic mapper.

    qwen_tokens = list(record.get("qwen_tokens") or [])
    labels = list(record.get("label") or [])
    existing = {(int(e.get("src", -1)), int(e.get("dst", -1)), str(e.get("subtype", ""))) for e in record.get("attention_edges") or []}
    qwen_annotations = list(record.get("qwen_annotations") or [])
    attention_edges = list(record.get("attention_edges") or [])
    added = 0
    added_by_source: dict[str, int] = {}
    for pair in pairs:
        if added >= max_edges:
            break
        if not (0 <= pair["doc"] < len(doc_items) and 0 <= pair["code"] < len(code_items)):
            continue
        doc = doc_items[pair["doc"]]
        code = code_items[pair["code"]]
        for src in _token_indices_for_span(qwen_tokens, doc["start"], doc["end"]):
            for dst in _token_indices_for_span(qwen_tokens, code["start"], code["end"]):
                if not (0 <= src < dst < len(labels)):
                    continue
                source = str(pair.get("source", "DocFactSemantic"))
                subtype = str(pair.get("subtype", "semantic"))
                key = (src, dst, subtype)
                if key in existing:
                    continue
                existing.add(key)
                qwen_annotations.append(
                    {
                        "token_i": qwen_tokens[src].get("surface", ""),
                        "token_j": qwen_tokens[dst].get("surface", ""),
                        "source": source,
                        "subtype": subtype,
                        "token_i_idx": src,
                        "token_j_idx": dst,
                    }
                )
                attention_edges.append({"src": src, "dst": dst, "subtype": subtype})
                added += 1
                added_by_source[source] = added_by_source.get(source, 0) + 1
                if added >= max_edges:
                    break
            if added >= max_edges:
                break
    record["qwen_annotations"] = qwen_annotations
    record["attention_edges"] = attention_edges
    record.setdefault("annotation_meta", {})["docstring_edges_added"] = added
    record["annotation_meta"]["docstring_edges_by_source"] = added_by_source
    record["annotation_meta"]["docstring_fact_count"] = len(_doc_facts_from_rules(doc_items))
    record["annotation_meta"]["num_causal_attention_edges"] = len(attention_edges)
    return record


def require_llm_credentials() -> None:
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY"):
        return
    raise SystemExit(
        "Missing OpenAI credentials for annotation. "
        "Please export OPENAI_API_KEY (and OPENAI_BASE_URL if using a proxy/local gateway) before running."
    )


def select_balanced_rows(
    rows: list[dict[str, Any]],
    *,
    samples_per_task: int,
    task_types: list[str],
    seed: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for task_type in task_types:
        candidates = [row for row in rows if row.get("task_type") == task_type]
        rng.shuffle(candidates)
        take = samples_per_task if samples_per_task > 0 else len(candidates)
        selected.extend(candidates[:take])
    if max_rows > 0:
        selected = selected[:max_rows]
    rng.shuffle(selected)
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate humaneval Python ChatML-FIM train rows with src.annotate and export compact edges."
    )
    p.add_argument("--input-path", type=Path, default=Path("data/benchmark/train_data/humaneval/humaneval_python_chatml.jsonl"))
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/benchmark/train_data/humaneval/humaneval_python_compact_annotated.jsonl"),
    )
    p.add_argument(
        "--annotation-cache-path",
        type=Path,
        default=Path("data/benchmark/train_data/humaneval/humaneval_python_annotation_cache.jsonl"),
    )
    p.add_argument("--seed-compact-path", type=Path, default=None)
    p.add_argument("--model-name-or-path", default="models/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--model-max-length", type=int, default=4096)
    p.add_argument("--task-types", nargs="+", default=["single_line", "multi_line"])
    p.add_argument("--samples-per-task", type=int, default=15)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--selected-indices", nargs="+", type=int, default=None, help="Annotate only these indices after balanced selection/shuffle, e.g. viewer sample #4.")
    p.add_argument("--update-existing-output", action="store_true", help="Replace selected records inside an existing output jsonl instead of writing only selected rows.")
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--annotation-mode", choices=["oneshot", "agent"], default="oneshot")
    p.add_argument("--max-teacher-edges", type=int, default=128)
    p.add_argument("--flush-every", type=int, default=5)
    p.add_argument("--docstring-edges", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--docstring-llm-edges", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--docstring-type-edges", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-docstring-edges", type=int, default=32)
    p.add_argument("--overwrite-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_llm_credentials()
    rows = load_jsonl(args.input_path)
    rows = select_balanced_rows(
        rows,
        samples_per_task=args.samples_per_task,
        task_types=args.task_types,
        seed=args.selection_seed,
        max_rows=args.max_rows,
    )
    if args.selected_indices is not None:
        selected_rows = []
        for idx in args.selected_indices:
            if idx < 0 or idx >= len(rows):
                raise SystemExit(f"selected index {idx} out of range for {len(rows)} selected rows")
            selected_rows.append(rows[idx])
        rows = selected_rows
    if not rows:
        raise SystemExit("no rows selected for annotation")

    counts: dict[str, int] = {}
    for row in rows:
        task_type = str(row.get("task_type", ""))
        counts[task_type] = counts.get(task_type, 0) + 1
    print(f"Selected {len(rows)} rows from {args.input_path}: {counts}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    cache = load_cache(args.annotation_cache_path)
    seed = load_seed_compact(args.seed_compact_path)
    cache_lock = threading.Lock()

    output_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        if not args.overwrite_cache and key in seed:
            output_by_key[key] = copy.deepcopy(seed[key])
        elif not args.overwrite_cache and key in cache and cache[key].get("record"):
            output_by_key[key] = copy.deepcopy(cache[key]["record"])

    todo = [row for row in rows if row_key(row) not in output_by_key]
    print(
        "Annotation plan: "
        f"todo={len(todo)}, cached={len(output_by_key)}, workers={args.num_workers}, "
        f"mode={args.annotation_mode}, max_rounds={args.max_rounds}, docstring_edges={args.docstring_edges}"
    )

    def work(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        key = row_key(row)
        try:
            record = annotate_row(
                row,
                tokenizer,
                args.model_max_length,
                args.max_rounds,
                args.annotation_mode,
                args.max_teacher_edges,
            )
            if record is not None and args.docstring_edges:
                record = add_docstring_to_code_edges(
                    row=row,
                    record=record,
                    tokenizer=tokenizer,
                    max_edges=args.max_docstring_edges,
                    use_llm=args.docstring_llm_edges,
                    use_type_edges=args.docstring_type_edges,
                )
            return key, record, None
        except Exception as exc:
            return key, None, repr(exc)

    done_since_flush = 0
    failures: dict[str, str] = {}
    if args.num_workers <= 1:
        iterator: Any = map(work, todo)
        pool = None
    else:
        pool = ThreadPoolExecutor(max_workers=args.num_workers)
        futures = [pool.submit(work, row) for row in todo]
        iterator = (future.result() for future in as_completed(futures))

    try:
        for key, record, err in tqdm.tqdm(iterator, total=len(todo), desc="annotate humaneval"):
            if record is None:
                failures[key] = err or "unknown"
            else:
                output_by_key[key] = record
                with cache_lock:
                    cache[key] = {"key": key, "record": record}
                done_since_flush += 1
            if done_since_flush >= max(1, args.flush_every):
                save_cache(cache, args.annotation_cache_path)
                done_since_flush = 0
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    save_cache(cache, args.annotation_cache_path)
    output = [output_by_key[row_key(row)] for row in rows if row_key(row) in output_by_key]
    if args.update_existing_output and args.output_path.exists():
        existing_rows = load_jsonl(args.output_path)
        replace_keys = set(output_by_key)
        replaced = 0
        merged = []
        for old_row in existing_rows:
            key = row_key(old_row)
            if key in replace_keys:
                merged.append(output_by_key[key])
                replaced += 1
            else:
                merged.append(old_row)
        existing_keys = {row_key(old_row) for old_row in existing_rows}
        for key in replace_keys - existing_keys:
            merged.append(output_by_key[key])
        output = merged
        print(f"Updated existing output rows: replaced={replaced}, total={len(output)}")
    write_jsonl_atomic(output, args.output_path)

    if failures:
        fail_path = args.output_path.with_suffix(args.output_path.suffix + ".failures.json")
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Failures: {len(failures)} -> {fail_path}")
    print(f"Done: {len(output)}/{len(rows)} annotated -> {args.output_path}")


if __name__ == "__main__":
    main()
