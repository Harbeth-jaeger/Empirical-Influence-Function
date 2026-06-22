from __future__ import annotations

import bisect
import dataclasses
import json
import re
from pathlib import Path
from typing import Any

from src.annotate.utils import (
    SubwordToken,
    TokenCorrelation,
    get_token_indices_in_span,
    tokenize_code_for_annotation,
)


IGNORE_INDEX = -100
MASK_TOKEN = "[MASK]"
INCOMPLETE_CODE_MARKER = "* Incomplete Code:\n"

EDGE_SUBTYPES = {"declaration", "function", "class", "semantics", "dataflow"}
IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")
LITERAL_RE = re.compile(r"^(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+(?:\.\d+)?|\".*\"|'.*')$")

FALLBACK_TOKEN_RE = re.compile(
    r"==|!=|<=|>=|&&|\|\||::|->|\+=|-=|\*=|/=|"
    r"[A-Za-z_]\w*|"
    r"0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+(?:\.\d+)?|"
    r"\S"
)

LANGUAGE_ALIASES = {
    "py": "Python",
    "python": "Python",
    "java": "Java",
    "cpp": "CPP",
    "c++": "CPP",
    "cc": "CPP",
    "cxx": "CPP",
    "cs": "C#",
    "c#": "C#",
    "csharp": "C#",
}

CONTROL_TOKENS = {
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "try", "catch", "finally", "throw", "break", "continue", "return",
}

DECLARATION_QUALIFIERS = {
    "const", "constexpr", "static", "final", "readonly", "volatile", "mutable",
    "unsigned", "signed", "long", "short", "public", "private", "protected",
    "internal", "extern", "ref", "out", "in", "params", "typename", "auto",
    "var", "struct", "class", "enum", "virtual", "override", "inline",
}

BUILTIN_TYPES = {
    "int", "char", "bool", "float", "double", "void", "string", "size_t",
    "long", "short", "byte", "decimal", "object", "var", "auto", "str",
    "bytes", "list", "dict", "set", "tuple", "boolean", "String",
}

TYPE_PUNCT = {"*", "&", "&&", "?", "[", "]", "<", ">", "::", ".", ","}


@dataclasses.dataclass(frozen=True)
class FimSpan:
    prefix: str
    target: str
    suffix: str
    full_code: str
    target_start: int
    target_end: int


def normalize_language(language: Any) -> str:
    text = str(language or "Python").strip()
    return LANGUAGE_ALIASES.get(text.lower(), text)


def tokenize_code_with_comment_words(code: str) -> list[SubwordToken]:
    return [
        SubwordToken(m.group(), -1, m.start(), m.end())
        for m in FALLBACK_TOKEN_RE.finditer(code)
    ]


def get_user_content(row: dict[str, Any]) -> str:
    for msg in row.get("messages") or []:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(row.get("instruction", ""))


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in row.get("messages") or []
    ]
    if not any(m["role"] == "assistant" for m in messages):
        messages.append({"role": "assistant", "content": str(row.get("target", ""))})
    return messages


def chatml_text(messages: list[dict[str, str]]) -> str:
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def get_chatml_offsets(messages: list[dict[str, str]]) -> tuple[int, int, str, str]:
    parts: list[str] = []
    user_start = -1
    assistant_start = -1
    user_content = ""
    assistant_content = ""
    for msg in messages:
        prefix = f"<|im_start|>{msg['role']}\n"
        if msg["role"] == "user" and user_start < 0:
            user_start = sum(len(p) for p in parts) + len(prefix)
            user_content = msg["content"]
        if msg["role"] == "assistant":
            assistant_start = sum(len(p) for p in parts) + len(prefix)
            assistant_content = msg["content"]
        parts.append(f"{prefix}{msg['content']}<|im_end|>\n")
    if user_start < 0 or assistant_start < 0:
        raise ValueError("messages must contain user and assistant content")
    return user_start, assistant_start, user_content, assistant_content


def code_rel_start_in_user(user_content: str) -> int:
    marker_pos = user_content.find(INCOMPLETE_CODE_MARKER)
    if marker_pos >= 0:
        return marker_pos + len(INCOMPLETE_CODE_MARKER)
    mask_pos = user_content.find(MASK_TOKEN)
    if mask_pos < 0:
        raise ValueError("user content does not contain [MASK]")
    return max(0, mask_pos)


def binarize_messages(messages: list[dict[str, str]], tokenizer: Any, max_len: int) -> dict[str, Any] | None:
    text = chatml_text(messages)
    enc = tokenizer(text, add_special_tokens=False)
    input_ids = list(enc.input_ids)
    if not input_ids or len(input_ids) > max_len:
        return None

    labels = [IGNORE_INDEX] * len(input_ids)
    cursor = 0
    assistant_indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]
    if not assistant_indices:
        return None
    target_idx = assistant_indices[-1]
    for idx, msg in enumerate(messages):
        prefix = f"<|im_start|>{msg['role']}\n"
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        content_ids = tokenizer(msg["content"], add_special_tokens=False).input_ids
        content_start = cursor + len(prefix_ids)
        content_end = content_start + len(content_ids)
        if idx == target_idx:
            labels[content_start:content_end] = input_ids[content_start:content_end]
        cursor += len(tokenizer(f"{prefix}{msg['content']}<|im_end|>\n", add_special_tokens=False).input_ids)

    if not any(v != IGNORE_INDEX for v in labels):
        return None
    return {"input_ids": input_ids, "label": labels, "length": len(input_ids), "chatml_text": text}


def extract_fim_span(row: dict[str, Any]) -> FimSpan:
    prefix = str(row.get("prefix", ""))
    target = str(row.get("target", row.get("fim_completion", "")))
    suffix = str(row.get("suffix", ""))
    full_code = str(row.get("full_code") or (prefix + target + suffix))
    target_start = len(prefix)
    target_end = target_start + len(target)

    meta = row.get("metadata") or {}
    try:
        meta_start = int(meta.get("target_start"))
        meta_end = int(meta.get("target_end"))
    except Exception:
        meta_start = target_start
        meta_end = target_end

    if 0 <= meta_start <= meta_end <= len(full_code) and full_code[meta_start:meta_end] == target:
        target_start, target_end = meta_start, meta_end
    elif full_code[target_start:target_end] != target:
        found = full_code.find(target)
        if found >= 0:
            target_start, target_end = found, found + len(target)
        else:
            raise ValueError("cannot locate target in full_code")

    return FimSpan(
        prefix=full_code[:target_start],
        target=target,
        suffix=full_code[target_end:],
        full_code=full_code,
        target_start=target_start,
        target_end=target_end,
    )


def _norm(surface: Any) -> str:
    text = str(surface).strip()
    if text.startswith(".") and len(text) > 1:
        text = text[1:]
    return re.sub(r"[^A-Za-z0-9_]+", "", text).lower()


def _is_identifier(surface: Any) -> bool:
    return bool(IDENT_RE.match(str(surface).strip()))


def _is_literal(surface: Any) -> bool:
    return bool(LITERAL_RE.match(str(surface).strip()))


def _is_meaningful(surface: Any) -> bool:
    text = str(surface).strip()
    norm = _norm(text)
    if not norm:
        return False
    if _is_identifier(text) or _is_literal(text):
        return True
    return text in {
        "=", "+", "-", "*", "/", "%", "+=", "-=", "*=", "/=", "==", "!=", "<=", ">=", "<", ">",
        "&&", "||", ".", "::", "->",
    }


def _is_type_like(surface: Any) -> bool:
    text = str(surface).strip()
    norm = _norm(text)
    if not norm:
        return False
    if text in BUILTIN_TYPES or norm in {t.lower() for t in BUILTIN_TYPES}:
        return True
    if _is_identifier(text) and text[:1].isupper():
        return True
    if norm.endswith("_t") or norm.startswith(("uint", "int")):
        return True
    return False


def _is_attribute_owner(tokens: list[SubwordToken], idx: int) -> bool:
    return idx + 1 < len(tokens) and tokens[idx + 1].surface in {".", "::", "->"}


def _line_end_for_token(full_code: str, token: SubwordToken) -> int:
    end = full_code.find("\n", token.char_end)
    return len(full_code) if end < 0 else end


def _import_binding_indices(tokens: list[SubwordToken], full_code: str, language: str) -> dict[str, list[int]]:
    if normalize_language(language) != "Python":
        return {}
    out: dict[str, list[int]] = {}
    for idx, tok in enumerate(tokens):
        if tok.surface != "import":
            continue
        line_end = _line_end_for_token(full_code, tok)
        line_indices = [
            j for j in range(idx + 1, len(tokens))
            if tokens[j].char_start < line_end and _is_identifier(tokens[j].surface)
        ]
        if not line_indices:
            continue
        for pos, j in enumerate(line_indices):
            if tokens[j - 1].surface == "as":
                out.setdefault(_norm(tokens[j].surface), []).append(j)
                continue
            if pos == 0 or tokens[j - 1].surface == ",":
                out.setdefault(_norm(tokens[j].surface), []).append(j)
    return out


def _add_edge(
    edges: list[TokenCorrelation],
    tokens: list[SubwordToken],
    src: int,
    dst: int,
    subtype: str,
) -> None:
    if src == dst or subtype not in EDGE_SUBTYPES:
        return
    if not (0 <= src < len(tokens) and 0 <= dst < len(tokens)):
        return
    edges.append(
        TokenCorrelation(
            token_i=tokens[src].surface,
            token_j=tokens[dst].surface,
            source="TargetEvidence",
            subtype=subtype,
            token_i_idx=src,
            token_j_idx=dst,
        )
    )


def _dedupe_edges(edges: list[TokenCorrelation], max_edges: int) -> list[TokenCorrelation]:
    out: list[TokenCorrelation] = []
    seen: set[tuple[int, int, str]] = set()
    pair_subtypes: dict[tuple[int, int], set[str]] = {}
    for edge in edges:
        pair = (int(edge.token_i_idx), int(edge.token_j_idx))
        subtype = str(edge.subtype)
        key = (pair[0], pair[1], subtype)
        if key in seen or pair[0] == pair[1]:
            continue
        existing = pair_subtypes.setdefault(pair, set())
        if subtype == "dataflow" and "declaration" in existing:
            continue
        if subtype == "declaration" and "dataflow" in existing:
            out = [e for e in out if not (int(e.token_i_idx), int(e.token_j_idx)) == pair or e.subtype != "dataflow"]
            seen = {k for k in seen if not (k[0], k[1]) == pair or k[2] != "dataflow"}
            existing.discard("dataflow")
        seen.add(key)
        existing.add(subtype)
        out.append(edge)
        if len(out) >= max_edges:
            break
    return out


def _matching_left(tokens: list[SubwordToken], right_idx: int, left: str, right: str, lower: int = 0) -> int:
    depth = 0
    for idx in range(right_idx, lower - 1, -1):
        surface = tokens[idx].surface
        if surface == right:
            depth += 1
        elif surface == left:
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _matching_right(tokens: list[SubwordToken], left_idx: int, left: str, right: str, upper: int | None = None) -> int:
    depth = 0
    limit = len(tokens) if upper is None else min(upper, len(tokens))
    for idx in range(left_idx, limit):
        surface = tokens[idx].surface
        if surface == left:
            depth += 1
        elif surface == right:
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _nearby_context_indices(completion_set: set[int], token_count: int, per_side: int = 180) -> set[int]:
    if not completion_set:
        return set()
    first = min(completion_set)
    last = max(completion_set)
    return {
        idx
        for idx in range(max(0, first - per_side), min(token_count, last + 1 + per_side))
        if idx not in completion_set
    }


def _line_start_offsets(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer(r"\n", text):
        starts.append(match.end())
    return starts


def _line_no_for_offset(starts: list[int], offset: int) -> int:
    lo, hi = 0, len(starts)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if starts[mid] <= offset:
            lo = mid
        else:
            hi = mid
    return lo


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" 	"))


def _python_scope_char_range(full_code: str, target_start: int) -> tuple[int, int] | None:
    starts = _line_start_offsets(full_code)
    lines = full_code.splitlines(keepends=True)
    if not lines:
        return None
    target_line = _line_no_for_offset(starts, target_start)
    header_line = -1
    header_indent = 0
    for line_no in range(target_line, -1, -1):
        stripped = lines[line_no].lstrip()
        if stripped.startswith(("def ", "async def ", "class ")):
            header_line = line_no
            header_indent = _indent_of(lines[line_no])
            break
    if header_line < 0:
        return None
    end_line = len(lines)
    for line_no in range(header_line + 1, len(lines)):
        stripped = lines[line_no].strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        if _indent_of(lines[line_no]) <= header_indent:
            end_line = line_no
            break
    return starts[header_line], starts[end_line] if end_line < len(starts) else len(full_code)


def _brace_scope_token_range(tokens: list[SubwordToken], target_first: int) -> tuple[int, int] | None:
    open_idx = -1
    depth = 0
    for idx in range(target_first - 1, -1, -1):
        surface = tokens[idx].surface
        if surface == "}":
            depth += 1
        elif surface == "{":
            if depth > 0:
                depth -= 1
            else:
                open_idx = idx
                break
    if open_idx < 0:
        return None
    close_idx = _matching_right(tokens, open_idx, "{", "}")
    if close_idx < 0:
        close_idx = len(tokens) - 1
    return open_idx, close_idx


def _scoped_context_indices(
    *,
    full_code: str,
    tokens: list[SubwordToken],
    language: str,
    completion_set: set[int],
    target_start: int,
) -> set[int]:
    fallback = _nearby_context_indices(completion_set, len(tokens), per_side=96)
    if not completion_set:
        return fallback
    if normalize_language(language) == "Python":
        char_range = _python_scope_char_range(full_code, target_start)
        if char_range is None:
            return fallback
        start, end = char_range
        scoped = {
            idx for idx, tok in enumerate(tokens)
            if idx not in completion_set and tok.char_start < end and tok.char_end > start
        }
        return scoped or fallback
    token_range = _brace_scope_token_range(tokens, min(completion_set))
    if token_range is None:
        return fallback
    start_idx, end_idx = token_range
    scoped = {idx for idx in range(start_idx, end_idx + 1) if idx not in completion_set}
    return scoped or fallback


def _class_name_indices(tokens: list[SubwordToken]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, tok in enumerate(tokens[:-1]):
        if tok.surface in {"class", "struct", "enum"} and _is_identifier(tokens[idx + 1].surface):
            out.setdefault(_norm(tokens[idx + 1].surface), []).append(idx + 1)
    return out


def _enclosing_class_indices(tokens: list[SubwordToken], target_first: int) -> list[int]:
    out: list[int] = []
    for idx in range(0, target_first):
        if tokens[idx].surface not in {"class", "struct"}:
            continue
        if idx + 1 >= len(tokens) or not _is_identifier(tokens[idx + 1].surface):
            continue
        brace = -1
        for j in range(idx + 2, min(len(tokens), idx + 80)):
            if tokens[j].surface == "{":
                brace = j
                break
            if tokens[j].surface == ";":
                break
        if brace < 0:
            continue
        close = _matching_right(tokens, brace, "{", "}")
        if close < 0 or brace < target_first < close:
            out.append(idx + 1)
    return out[-3:]


def _enclosing_function_signature(tokens: list[SubwordToken], target_first: int) -> dict[str, Any] | None:
    open_brace = -1
    depth = 0
    for idx in range(target_first - 1, -1, -1):
        surface = tokens[idx].surface
        if surface == "}":
            depth += 1
        elif surface == "{":
            if depth > 0:
                depth -= 1
            else:
                open_brace = idx
                break
    if open_brace < 0:
        return None

    close_paren = -1
    for idx in range(open_brace - 1, max(-1, open_brace - 120), -1):
        if tokens[idx].surface == ")":
            close_paren = idx
            break
        if tokens[idx].surface in {";", "}"}:
            return None
    if close_paren < 0:
        return None
    open_paren = _matching_left(tokens, close_paren, "(", ")", max(0, close_paren - 180))
    if open_paren <= 0:
        return None

    name_idx = open_paren - 1
    while name_idx >= 0 and tokens[name_idx].surface in {"~", "*", "&", "::"}:
        name_idx -= 1
    if name_idx < 0 or not _is_identifier(tokens[name_idx].surface):
        return None

    return_type_indices: list[int] = []
    idx = name_idx - 1
    scanned = 0
    while idx >= 0 and scanned < 16:
        surface = tokens[idx].surface
        if surface in {";", "{", "}", ":", "public", "private", "protected"}:
            break
        if _is_type_like(surface) or surface in TYPE_PUNCT or _norm(surface) in DECLARATION_QUALIFIERS:
            if _is_meaningful(surface) and surface not in TYPE_PUNCT and _norm(surface) not in DECLARATION_QUALIFIERS:
                return_type_indices.append(idx)
        elif return_type_indices:
            break
        idx -= 1
        scanned += 1

    param_indices = [
        idx
        for idx in range(open_paren + 1, close_paren)
        if _is_identifier(tokens[idx].surface) and _norm(tokens[idx].surface) not in DECLARATION_QUALIFIERS
    ]
    return {
        "name": name_idx,
        "return_types": list(reversed(return_type_indices[-4:])),
        "params": param_indices,
        "open_brace": open_brace,
    }


def _find_same_name_context(tokens: list[SubwordToken], completion_set: set[int], dst: int) -> list[int]:
    norm = _norm(tokens[dst].surface)
    if not norm:
        return []
    context = [idx for idx, tok in enumerate(tokens) if idx not in completion_set and _norm(tok.surface) == norm]
    return sorted(context, key=lambda i: (abs(i - dst), i))[:6]


def _looks_like_declaration_before(tokens: list[SubwordToken], name_idx: int, *, max_scan: int = 10) -> list[int]:
    sources: list[int] = []
    idx = name_idx - 1
    scanned = 0
    while idx >= 0 and scanned < max_scan:
        surface = tokens[idx].surface
        norm = _norm(surface)
        if surface in {"=", ";", "{", "}", ")"}:
            break
        if surface in {",", "("}:
            if not sources:
                break
            idx -= 1
            scanned += 1
            continue
        if _is_type_like(surface):
            sources.append(idx)
        elif norm in DECLARATION_QUALIFIERS or surface in TYPE_PUNCT:
            pass
        elif sources:
            break
        else:
            break
        idx -= 1
        scanned += 1
    return list(reversed(sources[-4:]))


def _assignment_groups(tokens: list[SubwordToken], completion_indices: list[int]) -> list[tuple[int, int, list[int]]]:
    groups: list[tuple[int, int, list[int]]] = []
    completion_set = set(completion_indices)
    for eq_idx in completion_indices:
        if tokens[eq_idx].surface not in {"=", "+=", "-=", "*=", "/=", "%="}:
            continue
        lhs = eq_idx - 1
        while lhs >= 0 and tokens[lhs].surface in {"*", "&", "]"}:
            lhs -= 1
        if lhs < 0 or lhs not in completion_set or not _is_identifier(tokens[lhs].surface):
            continue
        rhs: list[int] = []
        idx = eq_idx + 1
        while idx < len(tokens) and tokens[idx].surface not in {";", "{", "}"}:
            if idx in completion_set and _is_meaningful(tokens[idx].surface):
                rhs.append(idx)
            idx += 1
        groups.append((lhs, eq_idx, rhs))
    return groups


def _target_local_declaration_groups(tokens: list[SubwordToken], completion_indices: list[int]) -> list[tuple[list[int], int]]:
    groups: list[tuple[list[int], int]] = []
    completion_set = set(completion_indices)
    for idx in completion_indices:
        if not _is_identifier(tokens[idx].surface):
            continue
        type_sources = [src for src in _looks_like_declaration_before(tokens, idx) if src in completion_set]
        if type_sources:
            groups.append((type_sources, idx))
    return groups


def _tokens_for_target(full_code: str, target_start: int, target_end: int) -> tuple[list[SubwordToken], list[int]]:
    tokens = tokenize_code_for_annotation(full_code)
    completion_indices = sorted(get_token_indices_in_span(tokens, target_start, target_end))
    coarse_completion_token = False
    if len(completion_indices) == 1:
        tok = tokens[completion_indices[0]]
        coarse_completion_token = tok.char_start < target_start or tok.char_end > target_end
    if not completion_indices or coarse_completion_token:
        tokens = tokenize_code_with_comment_words(full_code)
        completion_indices = sorted(get_token_indices_in_span(tokens, target_start, target_end))
    return tokens, completion_indices


def _add_declaration_edges(
    edges: list[TokenCorrelation],
    tokens: list[SubwordToken],
    completion_indices: list[int],
    completion_set: set[int],
) -> None:
    for dst in completion_indices:
        if not _is_identifier(tokens[dst].surface):
            continue
        # Attribute/module owners such as ``os`` in ``os.path.join`` are not
        # variable uses. Treat their binding as class/API evidence instead of
        # fanning same-name declaration edges across the file.
        if not _is_attribute_owner(tokens, dst):
            for src in _find_same_name_context(tokens, completion_set, dst):
                _add_edge(edges, tokens, src, dst, "declaration")
        for src in _looks_like_declaration_before(tokens, dst):
            if src in completion_set:
                _add_edge(edges, tokens, src, dst, "declaration")


def _add_class_edges(
    edges: list[TokenCorrelation],
    tokens: list[SubwordToken],
    completion_indices: list[int],
    completion_set: set[int],
    *,
    full_code: str,
    language: str,
) -> None:
    class_names = _class_name_indices(tokens)
    import_bindings = _import_binding_indices(tokens, full_code, language)
    target_first = min(completion_indices)
    enclosing = _enclosing_class_indices(tokens, target_first)
    for dst in completion_indices:
        if not _is_identifier(tokens[dst].surface):
            continue
        norm = _norm(tokens[dst].surface)
        if _is_attribute_owner(tokens, dst):
            for src in import_bindings.get(norm, [])[:2]:
                if src not in completion_set:
                    _add_edge(edges, tokens, src, dst, "class")
        for src in class_names.get(norm, [])[:4]:
            if src not in completion_set:
                _add_edge(edges, tokens, src, dst, "class")
        if _is_meaningful(tokens[dst].surface):
            for src in enclosing:
                _add_edge(edges, tokens, src, dst, "class")


def _completion_anchor_indices(tokens: list[SubwordToken], completion_indices: list[int]) -> list[int]:
    completion_set = set(completion_indices)
    anchors: list[int] = []
    for lhs, _, _ in _assignment_groups(tokens, completion_indices):
        anchors.append(lhs)
    for idx in completion_indices:
        if idx + 1 < len(tokens) and tokens[idx + 1].surface == "(" and _is_identifier(tokens[idx].surface):
            anchors.append(idx)
    if not anchors:
        anchors = [
            idx for idx in completion_indices
            if (_is_identifier(tokens[idx].surface) or _is_literal(tokens[idx].surface))
            and not _is_attribute_owner(tokens, idx)
        ][:3]
    out: list[int] = []
    seen: set[int] = set()
    for idx in anchors:
        if idx in completion_set and idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out[:6]


def _add_function_edges(
    edges: list[TokenCorrelation],
    tokens: list[SubwordToken],
    completion_indices: list[int],
) -> None:
    completion_set = set(completion_indices)
    sig = _enclosing_function_signature(tokens, min(completion_indices))
    meaningful_targets = _completion_anchor_indices(tokens, completion_indices)
    if sig is not None:
        sources = [sig["name"]] + list(sig["return_types"][:2]) + list(sig["params"][:4])
        for src in sources:
            for dst in meaningful_targets:
                if src not in completion_set:
                    _add_edge(edges, tokens, src, dst, "function")

    for idx in completion_indices:
        if not _is_identifier(tokens[idx].surface):
            continue
        if idx + 1 >= len(tokens) or tokens[idx + 1].surface != "(":
            continue
        close = _matching_right(tokens, idx + 1, "(", ")")
        if close < 0:
            continue
        for dst in range(idx + 2, close):
            if dst in completion_set and _is_meaningful(tokens[dst].surface):
                _add_edge(edges, tokens, idx, dst, "function")


def _add_semantics_edges(
    edges: list[TokenCorrelation],
    tokens: list[SubwordToken],
    completion_indices: list[int],
    completion_set: set[int],
    *,
    full_code: str,
    target_start: int,
    language: str,
) -> None:
    nearby = _scoped_context_indices(
        full_code=full_code,
        tokens=tokens,
        language=language,
        completion_set=completion_set,
        target_start=target_start,
    )
    target_first = min(completion_indices)
    target_last = max(completion_indices)
    control_sources = [
        idx
        for idx in sorted(nearby)
        if tokens[idx].surface.lower() in CONTROL_TOKENS
    ]
    for src in control_sources[-10:]:
        for dst in completion_indices[:18]:
            if _is_meaningful(tokens[dst].surface):
                _add_edge(edges, tokens, src, dst, "semantics")

    for dst in completion_indices:
        if not _is_identifier(tokens[dst].surface):
            continue
        for src in _find_same_name_context(tokens, completion_set, dst):
            if src > target_last:
                prev = src - 1
                if prev >= 0 and tokens[prev].surface.lower() in CONTROL_TOKENS:
                    _add_edge(edges, tokens, prev, dst, "semantics")
                    _add_edge(edges, tokens, src, dst, "semantics")

    bracket_pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    close_to_open = {v: k for k, v in bracket_pairs.items()}
    stack: list[tuple[str, int]] = []
    for idx in completion_indices:
        surface = tokens[idx].surface
        if surface in bracket_pairs:
            stack.append((surface, idx))
        elif surface in close_to_open:
            for pos in range(len(stack) - 1, -1, -1):
                if stack[pos][0] == close_to_open[surface]:
                    _, src = stack.pop(pos)
                    _add_edge(edges, tokens, src, idx, "semantics")
                    break

    for idx in completion_indices:
        if tokens[idx].surface.lower() in CONTROL_TOKENS:
            for dst in range(idx + 1, min(len(tokens), idx + 18)):
                if dst in completion_set and _is_meaningful(tokens[dst].surface):
                    _add_edge(edges, tokens, idx, dst, "semantics")


def _add_dataflow_edges(
    edges: list[TokenCorrelation],
    tokens: list[SubwordToken],
    completion_indices: list[int],
    completion_set: set[int],
) -> None:
    target_first = min(completion_indices)
    lhs_before_target = -1
    eq_before_target = -1
    for idx in range(target_first - 1, max(-1, target_first - 40), -1):
        if tokens[idx].surface == "=":
            eq_before_target = idx
            break
        if tokens[idx].surface in {";", "{", "}"}:
            break
    if eq_before_target >= 0:
        lhs = eq_before_target - 1
        while lhs >= 0 and tokens[lhs].surface in {"*", "&", "]"}:
            lhs -= 1
        if lhs >= 0 and _is_identifier(tokens[lhs].surface):
            lhs_before_target = lhs

    if lhs_before_target >= 0:
        added = 0
        for dst in _completion_anchor_indices(tokens, completion_indices):
            if _is_attribute_owner(tokens, dst):
                continue
            if _is_identifier(tokens[dst].surface) or _is_literal(tokens[dst].surface):
                _add_edge(edges, tokens, lhs_before_target, dst, "dataflow")
                added += 1
            if added >= 4:
                break

    for lhs, _, rhs in _assignment_groups(tokens, completion_indices):
        added = 0
        for src in rhs:
            if _is_attribute_owner(tokens, src):
                continue
            if _is_identifier(tokens[src].surface) or _is_literal(tokens[src].surface):
                _add_edge(edges, tokens, src, lhs, "dataflow")
                added += 1
            if added >= 6:
                break

    for type_sources, name_idx in _target_local_declaration_groups(tokens, completion_indices):
        eq_idx = next((idx for idx in range(name_idx + 1, min(len(tokens), name_idx + 8)) if tokens[idx].surface == "="), -1)
        if eq_idx < 0:
            continue
        rhs = [
            idx for idx in range(eq_idx + 1, min(len(tokens), eq_idx + 20))
            if idx in completion_set
            and (_is_identifier(tokens[idx].surface) or _is_literal(tokens[idx].surface))
            and not _is_attribute_owner(tokens, idx)
        ][:6]
        for src in rhs:
            _add_edge(edges, tokens, src, name_idx, "dataflow")
        for src in rhs:
            if _is_identifier(tokens[src].surface):
                for ctx in _find_same_name_context(tokens, completion_set, src)[:2]:
                    _add_edge(edges, tokens, ctx, name_idx, "dataflow")


def annotate_completion_simple(
    *,
    full_code: str,
    target_start: int,
    target_end: int,
    language: str,
    task_type: str = "",
    max_edges: int = 160,
) -> tuple[list[SubwordToken], list[TokenCorrelation]]:
    del task_type  # The target-centric evidence rules are intentionally task-agnostic.
    language = normalize_language(language)
    tokens, completion_indices = _tokens_for_target(full_code, target_start, target_end)
    completion_set = set(completion_indices)
    if not completion_set:
        return tokens, []

    edges: list[TokenCorrelation] = []
    _add_declaration_edges(edges, tokens, completion_indices, completion_set)
    _add_function_edges(edges, tokens, completion_indices)
    _add_class_edges(
        edges,
        tokens,
        completion_indices,
        completion_set,
        full_code=full_code,
        language=language,
    )
    _add_semantics_edges(
        edges,
        tokens,
        completion_indices,
        completion_set,
        full_code=full_code,
        target_start=target_start,
        language=language,
    )
    _add_dataflow_edges(edges, tokens, completion_indices, completion_set)

    return tokens, _dedupe_edges(edges, max_edges=max_edges)


def annotate_prompt_simple(
    *,
    full_code: str,
    target_start: int,
    target_end: int,
    language: str,
    max_edges: int = 160,
) -> tuple[list[SubwordToken], list[TokenCorrelation]]:
    language = normalize_language(language)
    tokens, completion_indices = _tokens_for_target(full_code, target_start, target_end)
    completion_set = set(completion_indices)
    if not completion_set:
        return tokens, []

    scope = _scoped_context_indices(
        full_code=full_code,
        tokens=tokens,
        language=language,
        completion_set=completion_set,
        target_start=target_start,
    )
    context_indices = sorted(scope)
    context_set = set(context_indices)
    edges: list[TokenCorrelation] = []

    for src in context_indices:
        if tokens[src].surface.lower() not in CONTROL_TOKENS:
            continue
        added = 0
        for dst in context_indices:
            if dst <= src:
                continue
            if tokens[dst].char_start - tokens[src].char_start > 500:
                break
            if _is_meaningful(tokens[dst].surface):
                _add_edge(edges, tokens, src, dst, "semantics")
                added += 1
            if added >= 12:
                break

    last_by_norm: dict[str, int] = {}
    for dst in context_indices:
        surface = tokens[dst].surface
        norm = _norm(surface)
        if _is_identifier(surface) and norm:
            src = last_by_norm.get(norm)
            if src is not None and not _is_attribute_owner(tokens, dst):
                _add_edge(edges, tokens, src, dst, "declaration")
            last_by_norm[norm] = dst

    sig = _enclosing_function_signature(tokens, min(completion_set))
    if sig is not None:
        sources = [sig["name"]] + list(sig["return_types"][:2]) + list(sig["params"][:4])
        targets = [
            idx for idx in context_indices
            if idx > sig["name"] and (_is_identifier(tokens[idx].surface) or _is_literal(tokens[idx].surface))
            and not _is_attribute_owner(tokens, idx)
        ][:24]
        for src in sources:
            if src not in context_set:
                continue
            for dst in targets:
                if dst > src:
                    _add_edge(edges, tokens, src, dst, "function")

    for src in _enclosing_class_indices(tokens, min(completion_set)):
        if src not in context_set:
            continue
        added = 0
        for dst in context_indices:
            if dst > src and _is_identifier(tokens[dst].surface) and not _is_attribute_owner(tokens, dst):
                _add_edge(edges, tokens, src, dst, "class")
                added += 1
            if added >= 24:
                break

    for eq_idx in context_indices:
        if tokens[eq_idx].surface not in {"=", "+=", "-=", "*=", "/=", "%="}:
            continue
        lhs = eq_idx - 1
        while lhs >= 0 and tokens[lhs].surface in {"*", "&", "]"}:
            lhs -= 1
        if lhs not in context_set or not _is_identifier(tokens[lhs].surface):
            continue
        line_end = full_code.find("\n", tokens[eq_idx].char_end)
        if line_end < 0:
            line_end = len(full_code)
        added = 0
        for src in context_indices:
            if src <= eq_idx or tokens[src].char_start >= line_end:
                continue
            if _is_attribute_owner(tokens, src):
                continue
            if _is_identifier(tokens[src].surface) or _is_literal(tokens[src].surface):
                _add_edge(edges, tokens, src, lhs, "dataflow")
                added += 1
            if added >= 6:
                break

    return tokens, _dedupe_edges(edges, max_edges=max_edges)


def make_qwen_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    enc = tokenizer(chatml_text(messages), add_special_tokens=False, return_offsets_mapping=True)
    return [
        {
            "surface": tokenizer.decode([int(token_id)]),
            "token_id": int(token_id),
            "char_start": int(start),
            "char_end": int(end),
        }
        for token_id, (start, end) in zip(enc["input_ids"], enc["offset_mapping"])
    ]


def _span_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _simple_to_qwen_map(
    *,
    simple_tokens: list[SubwordToken],
    qwen_tokens: list[dict[str, Any]],
    span: FimSpan,
    user_start: int,
    assistant_start: int,
    code_rel_start: int,
) -> dict[int, list[int]]:
    shrink = len(span.target) - len(MASK_TOKEN)
    mapped_spans: list[tuple[int, int]] = []
    for tok in simple_tokens:
        tok_start = int(tok.char_start)
        tok_end = int(tok.char_end)
        if tok_end <= span.target_start:
            start = user_start + code_rel_start + tok_start
            end = user_start + code_rel_start + tok_end
        elif tok_start >= span.target_end:
            start = user_start + code_rel_start + tok_start - shrink
            end = user_start + code_rel_start + tok_end - shrink
        else:
            overlap_start = max(tok_start, span.target_start)
            overlap_end = min(tok_end, span.target_end)
            start = assistant_start + (overlap_start - span.target_start)
            end = assistant_start + (overlap_end - span.target_start)
        mapped_spans.append((start, end))

    out: dict[int, list[int]] = {idx: [] for idx in range(len(simple_tokens))}
    qwen_ends = [int(tok["char_end"]) for tok in qwen_tokens]
    for si, (start, end) in enumerate(mapped_spans):
        # FIM ChatML order is prefix + mask + suffix in the user message, then
        # completion in the assistant message. Source-code order is therefore
        # not monotonic in Qwen-token coordinates around the masked span.
        qpos = bisect.bisect_right(qwen_ends, start)
        qi = qpos
        while qi < len(qwen_tokens) and int(qwen_tokens[qi]["char_start"]) < end:
            if _span_overlaps(start, end, int(qwen_tokens[qi]["char_start"]), int(qwen_tokens[qi]["char_end"])):
                out[si].append(qi)
            qi += 1
    return out


def map_annotations_to_qwen(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    simple_tokens: list[SubwordToken],
    annotations: list[TokenCorrelation],
    span: FimSpan,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qwen_tokens = make_qwen_tokens(tokenizer, messages)
    user_start, assistant_start, user_content, _ = get_chatml_offsets(messages)
    code_rel_start = code_rel_start_in_user(user_content)
    s2q = _simple_to_qwen_map(
        simple_tokens=simple_tokens,
        qwen_tokens=qwen_tokens,
        span=span,
        user_start=user_start,
        assistant_start=assistant_start,
        code_rel_start=code_rel_start,
    )

    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for ann in annotations:
        subtype = ann.subtype if ann.subtype in EDGE_SUBTYPES else "semantics"
        for qi in s2q.get(int(ann.token_i_idx), []):
            for qj in s2q.get(int(ann.token_j_idx), []):
                if qi == qj:
                    continue
                src, dst = (qi, qj) if qi < qj else (qj, qi)
                key = (src, dst, subtype)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "token_i": qwen_tokens[src]["surface"],
                        "token_j": qwen_tokens[dst]["surface"],
                        "source": ann.source,
                        "subtype": subtype,
                        "token_i_idx": int(src),
                        "token_j_idx": int(dst),
                    }
                )
    return qwen_tokens, out


def qwen_to_attention_edges(
    qwen_annotations: list[dict[str, Any]],
    labels: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completion_edges: list[dict[str, Any]] = []
    prompt_edges: list[dict[str, Any]] = []
    seen_completion: set[tuple[int, int, str]] = set()
    seen_prompt: set[tuple[int, int, str]] = set()
    for ann in qwen_annotations:
        try:
            src = int(ann["token_i_idx"])
            dst = int(ann["token_j_idx"])
        except Exception:
            continue
        subtype = str(ann.get("subtype", "semantics") or "semantics")
        if not (0 <= src < dst < len(labels)):
            continue
        key = (src, dst, subtype)
        edge = {"src": src, "dst": dst, "subtype": subtype}
        if labels[dst] != IGNORE_INDEX:
            if key in seen_completion:
                continue
            seen_completion.add(key)
            completion_edges.append(edge)
        elif labels[src] == IGNORE_INDEX:
            if key in seen_prompt:
                continue
            seen_prompt.add(key)
            prompt_edges.append(edge)
    return completion_edges, prompt_edges


def annotate_safim_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    model_max_length: int = 16384,
    annotation_mode: str = "target_evidence",
    max_edges: int = 160,
    **_: Any,
) -> dict[str, Any] | None:
    if annotation_mode not in {"target_evidence", "rules"}:
        raise ValueError(f"unsupported annotation_mode for target evidence annotator: {annotation_mode}")

    span = extract_fim_span(row)
    if not span.target or MASK_TOKEN not in get_user_content(row):
        return None
    messages = build_messages(row)
    packed = binarize_messages(messages, tokenizer, model_max_length)
    if packed is None:
        return None

    language = normalize_language(row.get("language", "Python"))
    task_type = str(row.get("task_type", "algorithmic_block"))
    simple_tokens, completion_annotations = annotate_completion_simple(
        full_code=span.full_code,
        target_start=span.target_start,
        target_end=span.target_end,
        language=language,
        task_type=task_type,
        max_edges=max_edges,
    )
    _prompt_tokens, prompt_annotations = annotate_prompt_simple(
        full_code=span.full_code,
        target_start=span.target_start,
        target_end=span.target_end,
        language=language,
        max_edges=max_edges,
    )
    annotations = _dedupe_edges(completion_annotations + prompt_annotations, max_edges=max_edges * 2)
    qwen_tokens, qwen_annotations = map_annotations_to_qwen(
        tokenizer=tokenizer,
        messages=messages,
        simple_tokens=simple_tokens,
        annotations=annotations,
        span=span,
    )
    attention_edges, prompt_attention_edges = qwen_to_attention_edges(qwen_annotations, packed["label"])

    by_subtype: dict[str, int] = {}
    for edge in attention_edges:
        subtype = str(edge.get("subtype", ""))
        by_subtype[subtype] = by_subtype.get(subtype, 0) + 1

    return {
        "input_ids": packed["input_ids"],
        "label": packed["label"],
        "length": packed["length"],
        "attention_edges": attention_edges,
        "prompt_attention_edges": prompt_attention_edges,
        "uid": row.get("uid"),
        "language": row.get("language", language),
        "task_type": task_type,
        "raw_id": row.get("raw_id", row.get("uid")),
        "annotation_meta": {
            "annotator": "src.annotate.target_evidence_annot.target_evidence",
            "annotation_scope": "target_centric_file_level_fim",
            "edge_scheme": ["declaration", "function", "class", "semantics", "dataflow"],
            "valid_edge_classes": ["prompt->completion", "completion->completion"],
            "auxiliary_edge_classes": ["prompt->prompt"],
            "target_start": span.target_start,
            "target_end": span.target_end,
            "num_completion_simple_tokens": len(get_token_indices_in_span(simple_tokens, span.target_start, span.target_end)),
            "num_simple_annotations": len(annotations),
            "num_qwen_annotations": len(qwen_annotations),
            "num_causal_attention_edges": len(attention_edges),
            "num_prompt_attention_edges": len(prompt_attention_edges),
            "attention_edges_by_subtype": by_subtype,
        },
        "tokens": [dataclasses.asdict(tok) for tok in simple_tokens],
        "annotations": [dataclasses.asdict(ann) for ann in annotations],
        "qwen_tokens": qwen_tokens,
        "qwen_annotations": qwen_annotations,
    }


def require_llm_credentials() -> None:
    """Compatibility hook for old callers; this annotator is rule-only."""
    return None
