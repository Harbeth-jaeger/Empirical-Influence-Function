from __future__ import annotations

import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import tqdm
from openai import OpenAI


DEFAULT_GRAPH_REASON_WEIGHTS: dict[str, float] = {
    "dataflow": 1.0,
    "defuse": 0.9,
    "call": 0.7,
    "return": 0.7,
    "type": 0.6,
    "api": 0.5,
    "semantic": 0.4,
    "bracket": 0.25,
}

_LANGUAGE_ALIASES: dict[str, str] = {
    "py": "Python",
    "python": "Python",
    "java": "Java",
    "cpp": "C++",
    "c++": "C++",
    "cc": "C++",
    "cxx": "C++",
    "c": "C",
    "cs": "C#",
    "c#": "C#",
    "csharp": "C#",
    "go": "Go",
    "golang": "Go",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "rb": "Ruby",
    "ruby": "Ruby",
    "php": "PHP",
}


def _normalize_language(language: str | None) -> str:
    if not language:
        return "Python"
    return _LANGUAGE_ALIASES.get(str(language).strip().lower(), str(language).strip())


def _annotation_edges_present(sample: dict[str, Any]) -> bool:
    return bool(sample.get("qwen_annotations") or sample.get("annotations"))


def _as_annotation_dicts(edges: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for edge in edges:
        if isinstance(edge, dict):
            i = edge.get("token_i_idx", edge.get("i", -1))
            j = edge.get("token_j_idx", edge.get("j", -1))
            subtype = edge.get("subtype", edge.get("reason", "semantic"))
            token_i = edge.get("token_i", "")
            token_j = edge.get("token_j", "")
            source = edge.get("source", "Teacher")
        else:
            i = getattr(edge, "token_i_idx", -1)
            j = getattr(edge, "token_j_idx", -1)
            subtype = getattr(edge, "subtype", "semantic")
            token_i = getattr(edge, "token_i", "")
            token_j = getattr(edge, "token_j", "")
            source = getattr(edge, "source", "Teacher")
        if isinstance(i, int) and isinstance(j, int) and i >= 0 and j >= 0 and i != j:
            out.append({
                "token_i": token_i,
                "token_j": token_j,
                "source": source,
                "subtype": subtype,
                "token_i_idx": i,
                "token_j_idx": j,
            })
    return out


def _extract_fim_parts(sample: dict[str, Any]) -> tuple[str, str, str, str]:
    prompt = str(sample.get("fim_prompt") or "")
    completion = str(sample.get("fim_completion") or sample.get("response") or "")
    if not completion:
        for msg in sample.get("messages", []):
            if msg.get("role") == "assistant":
                completion = str(msg.get("content", ""))

    prefix_tag = "<|fim_prefix|>"
    suffix_tag = "<|fim_suffix|>"
    middle_tag = "<|fim_middle|>"
    prefix = ""
    suffix = ""
    if prefix_tag in prompt and suffix_tag in prompt:
        after_prefix = prompt.split(prefix_tag, 1)[1]
        prefix, rest = after_prefix.split(suffix_tag, 1)
        suffix = rest.split(middle_tag, 1)[0] if middle_tag in rest else rest

    full_code = prefix + completion + suffix
    if not full_code:
        full_code = completion
    return full_code, completion, prefix, suffix


def _find_sublist(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    n = len(needle)
    for idx in range(len(haystack) - n, -1, -1):
        if haystack[idx:idx + n] == needle:
            return idx
    return -1


def _map_simple_edges_to_completion_bpe(
    *,
    edges: list[dict[str, Any]],
    simple_tokens: list[Any],
    completion: str,
    completion_start_char: int,
    tokenizer: Any,
    input_ids: list[int],
) -> list[dict[str, Any]]:
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    content_start = _find_sublist(input_ids, completion_ids)
    if content_start < 0:
        return []

    enc = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    offsets = list(enc.get("offset_mapping", []))

    def simple_to_bpe(simple_idx: int) -> int:
        if simple_idx < 0 or simple_idx >= len(simple_tokens):
            return -1
        tok = simple_tokens[simple_idx]
        start = int(tok.char_start) - completion_start_char
        end = int(tok.char_end) - completion_start_char
        if end <= 0 or start >= len(completion):
            return -1
        for bpe_idx, (b_start, b_end) in enumerate(offsets):
            if start < b_end and b_start < end:
                return content_start + bpe_idx
        return -1

    mapped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for edge in edges:
        qi = simple_to_bpe(int(edge["token_i_idx"]))
        qj = simple_to_bpe(int(edge["token_j_idx"]))
        subtype = str(edge.get("subtype", "semantic"))
        key = (qi, qj, subtype)
        if qi < 0 or qj < 0 or qi == qj or key in seen:
            continue
        seen.add(key)
        mapped.append({
            "token_i": edge.get("token_i", ""),
            "token_j": edge.get("token_j", ""),
            "source": edge.get("source", "Teacher"),
            "subtype": subtype,
            "token_i_idx": qi,
            "token_j_idx": qj,
        })
    return mapped


def _teacher_complete_edges(
    *,
    full_code: str,
    simple_tokens: list[Any],
    target_indices: set[int],
    structural_edges: list[dict[str, Any]],
    language: str,
    api_key: str,
    api_base_url: str,
    model_name: str,
    max_tokens: int = 768,
    context_chars: int = 6000,
) -> list[dict[str, Any]]:
    if not api_key:
        return []

    indexed = {
        i: tok.surface
        for i, tok in enumerate(simple_tokens)
        if i in target_indices
    }
    if not indexed:
        return []

    client = OpenAI(api_key=api_key, base_url=api_base_url)
    system = (
        f"You are a {language} code dependency annotator. Return JSON only. "
        "No markdown. No prose. No comments. "
        'Schema: {"pairs":[{"i":0,"j":1,"reason":"dataflow"}]}. '
        "Allowed reasons: dataflow, semantic, api, defuse, call, return, type, bracket. "
        "Use only token indices listed in target_tokens. Do not repeat seeded_structural_edges."
    )
    if context_chars > 0 and len(full_code) > context_chars:
        target_starts = [int(simple_tokens[i].char_start) for i in target_indices if i < len(simple_tokens)]
        target_ends = [int(simple_tokens[i].char_end) for i in target_indices if i < len(simple_tokens)]
        if target_starts and target_ends:
            target_start = min(target_starts)
            target_end = max(target_ends)
            side = max(0, (context_chars - (target_end - target_start)) // 2)
            lo = max(0, target_start - side)
            hi = min(len(full_code), target_end + side)
            code_for_teacher = full_code[lo:hi]
        else:
            code_for_teacher = full_code[:context_chars]
    else:
        code_for_teacher = full_code

    compact_tokens = [[int(i), str(tok)] for i, tok in indexed.items()]
    compact_structural = [
        [int(e["token_i_idx"]), int(e["token_j_idx"]), str(e.get("subtype", "semantic"))]
        for e in structural_edges
        if isinstance(e.get("token_i_idx"), int) and isinstance(e.get("token_j_idx"), int)
    ]
    request_payload = {
        "language": language,
        "code_context": code_for_teacher,
        "target_tokens": compact_tokens,
        "seeded_structural_edges": compact_structural,
        "task": (
            "Find only the most important high-confidence missing dataflow, semantic, and API dependency edges among target_tokens. "
            'Return exactly {"pairs":[...]} with at most 16 edges. If none, return {"pairs":[]}.'
        ),
    }
    user = json.dumps(request_payload, ensure_ascii=False, separators=(",", ":"))

    raw = ""
    debug = ""
    for payload_text in (user, _minimal_teacher_payload(language, indexed, structural_edges)):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": payload_text},
                ],
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": payload_text},
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                )
            except Exception as exc2:
                print(f"    [graph_signal.teacher] API error: {exc2}")
                return []

        raw = _extract_chat_message_text(response)
        debug = _summarize_chat_response(response)
        if raw.strip():
            break

    payload = _parse_teacher_json(raw)
    if payload is None:
        preview = raw.replace("\n", "\\n")[:300]
        print(
            "    [graph_signal.teacher] failed to parse teacher JSON; "
            f"using structural edges only; raw={preview!r}; response={debug}"
        )
        return []

    valid_reasons = set(DEFAULT_GRAPH_REASON_WEIGHTS)
    out: list[dict[str, Any]] = []
    pairs = payload if isinstance(payload, list) else payload.get("pairs", [])
    for pair in pairs:
        try:
            if isinstance(pair, dict):
                i = int(pair["i"])
                j = int(pair["j"])
                reason = str(pair.get("reason", "semantic")).lower()
            else:
                i = int(pair[0])
                j = int(pair[1])
                reason = str(pair[2] if len(pair) > 2 else "semantic").lower()
        except Exception:
            continue
        if i not in target_indices or j not in target_indices or i == j:
            continue
        if reason not in valid_reasons:
            reason = "semantic"
        out.append({
            "token_i": indexed.get(i, ""),
            "token_j": indexed.get(j, ""),
            "source": "DeepSeekTeacher",
            "subtype": reason,
            "token_i_idx": i,
            "token_j_idx": j,
        })
    return out


def _minimal_teacher_payload(
    language: str,
    indexed: dict[int, str],
    structural_edges: list[dict[str, Any]],
) -> str:
    compact_tokens = [[int(i), str(tok)] for i, tok in indexed.items()]
    compact_structural = [
        [int(e["token_i_idx"]), int(e["token_j_idx"]), str(e.get("subtype", "semantic"))]
        for e in structural_edges
        if isinstance(e.get("token_i_idx"), int) and isinstance(e.get("token_j_idx"), int)
    ]
    return json.dumps({
        "language": language,
        "target_tokens": compact_tokens,
        "seeded_structural_edges": compact_structural,
        "task": (
            'Return JSON only: {"pairs":[{"i":int,"j":int,"reason":"dataflow|semantic|api"}]}. '
            "Use only target token indices. At most 20 edges."
        ),
    }, ensure_ascii=False, separators=(",", ":"))


def _extract_chat_message_text(response: Any) -> str:
    try:
        message = response.choices[0].message
    except Exception:
        return ""

    chunks: list[str] = []
    for attr in ("content", "reasoning_content", "reasoning", "refusal"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value:
            chunks.append(value)
        elif isinstance(value, list):
            for part in value:
                if isinstance(part, str):
                    chunks.append(part)
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str):
                        chunks.append(text)
                    elif isinstance(text, dict) and isinstance(text.get("value"), str):
                        chunks.append(text["value"])

    if chunks:
        return "\n".join(chunks)

    if hasattr(message, "model_dump"):
        dumped = message.model_dump()
        for key in ("content", "reasoning_content", "reasoning", "refusal"):
            value = dumped.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list):
                texts = [str(part) for part in value if part]
                if texts:
                    return "\n".join(texts)
    return ""


def _summarize_chat_response(response: Any) -> str:
    try:
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        msg = message.model_dump() if hasattr(message, "model_dump") else {}
        keys = sorted(k for k, v in msg.items() if v not in (None, "", [], {}))
        preview = str({k: msg.get(k) for k in keys[:6]})[:300]
        return f"finish_reason={finish_reason!r}, message_keys={keys}, message_preview={preview}"
    except Exception as exc:
        return f"unavailable:{exc}"


def _parse_teacher_json(raw: str) -> Any | None:
    raw = (raw or "").strip()
    if not raw:
        return None

    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(part.strip() for part in fenced if part.strip())
    candidates.append(raw)

    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        candidates.append(raw[brace_start:brace_end + 1])

    bracket_start = raw.find("[")
    bracket_end = raw.rfind("]")
    if bracket_start >= 0 and bracket_end > bracket_start:
        candidates.append(raw[bracket_start:bracket_end + 1])

    for text in candidates:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
            return payload
        if isinstance(payload, list):
            return payload

    # Some API responses hit max_tokens and end with finish_reason="length".
    # The prefix often still contains many complete pair objects.  Salvage
    # those instead of discarding the teacher signal entirely.
    dict_pair_re = re.compile(
        r'\{\s*"i"\s*:\s*(-?\d+)\s*,\s*"j"\s*:\s*(-?\d+)\s*,\s*"reason"\s*:\s*"([^"]+)"\s*\}'
    )
    pairs = [
        {"i": int(i), "j": int(j), "reason": reason}
        for i, j, reason in dict_pair_re.findall(raw)
    ]
    list_pair_re = re.compile(r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*"([^"]+)"\s*\]')
    pairs.extend(
        {"i": int(i), "j": int(j), "reason": reason}
        for i, j, reason in list_pair_re.findall(raw)
    )
    if pairs:
        return {"pairs": pairs}
    return None


def add_graphsignal_teacher_annotations(
    samples: list[dict[str, Any]],
    tokenizer: Any,
    api_key: str | None = None,
    api_base_url: str = "https://api.deepseek.com",
    model_name: str = "deepseek-v4",
    overwrite: bool = False,
    num_workers: int = 1,
    teacher_max_tokens: int = 768,
    teacher_context_chars: int = 6000,
) -> list[dict[str, Any]]:
    """
    Fill qwen_annotations for GraphSignal when the input data lacks them.
    The teacher annotates the completed FIM code and edges are remapped to the
    assistant completion token positions in the ChatML training sequence.
    """
    from src.annotate.neural_annot import SyntacticCheckerTool
    from src.annotate.utils import get_token_indices_in_span, tokenize_code_for_annotation

    api_key = api_key or ""
    def annotate_one(sample: dict[str, Any]) -> dict[str, Any]:
        syntactic_tool = SyntacticCheckerTool()
        local = copy.deepcopy(sample)
        if _annotation_edges_present(local) and not overwrite:
            return local

        full_code, completion, prefix, _suffix = _extract_fim_parts(local)
        if not completion or "input_ids" not in local:
            return local

        language = _normalize_language(local.get("language"))
        simple_tokens = tokenize_code_for_annotation(full_code)
        completion_start = len(prefix)
        target_indices = get_token_indices_in_span(
            simple_tokens,
            completion_start,
            completion_start + len(completion),
        )
        if not target_indices:
            return local

        structural_raw = syntactic_tool.get_edges(full_code, simple_tokens, language)
        structural = [
            {
                "token_i": simple_tokens[e.token_i_idx].surface,
                "token_j": simple_tokens[e.token_j_idx].surface,
                "source": "Structural",
                "subtype": e.reason,
                "token_i_idx": e.token_i_idx,
                "token_j_idx": e.token_j_idx,
            }
            for e in structural_raw
            if e.token_i_idx in target_indices and e.token_j_idx in target_indices
        ]
        neural = _teacher_complete_edges(
            full_code=full_code,
            simple_tokens=simple_tokens,
            target_indices=target_indices,
            structural_edges=structural,
            language=language,
            api_key=api_key,
            api_base_url=api_base_url,
            model_name=model_name,
            max_tokens=teacher_max_tokens,
            context_chars=teacher_context_chars,
        )
        simple_edges = _as_annotation_dicts(structural + neural)
        qwen_edges = _map_simple_edges_to_completion_bpe(
            edges=simple_edges,
            simple_tokens=simple_tokens,
            completion=completion,
            completion_start_char=completion_start,
            tokenizer=tokenizer,
            input_ids=list(local["input_ids"]),
        )
        if qwen_edges:
            local["annotations"] = simple_edges
            local["qwen_annotations"] = qwen_edges
        return local

    if num_workers <= 1:
        iterator = map(annotate_one, samples)
    else:
        executor = ThreadPoolExecutor(max_workers=num_workers)
        iterator = executor.map(annotate_one, samples)

    out: list[dict[str, Any]] = []
    try:
        for local in tqdm.tqdm(iterator, total=len(samples), desc="  graphsignal.teacher", mininterval=5):
            out.append(local)
    finally:
        if num_workers > 1:
            executor.shutdown(wait=True)

    return out


