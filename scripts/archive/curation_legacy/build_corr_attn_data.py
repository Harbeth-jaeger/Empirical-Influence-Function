#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import tqdm
import transformers
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.apply_governance_operator import (
    DEFAULT_GRAPH_REASON_WEIGHTS,
    _as_annotation_dicts,
    _extract_chat_message_text,
    _extract_fim_parts,
    _normalize_language,
    _parse_teacher_json,
    _summarize_chat_response,
    add_graphsignal_teacher_annotations,
)
from src.annotate.utils import get_token_indices_in_span, tokenize_code_for_annotation

try:
    from src.sft.binarize_data import chatml_format_preprocess, setup_tokenizer
except ModuleNotFoundError:
    def setup_tokenizer(tokenizer: transformers.PreTrainedTokenizer) -> transformers.PreTrainedTokenizer:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
        return tokenizer

    def _chatml_text(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", ""))
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(parts)

    def chatml_format_preprocess(
        sources: list[dict[str, Any]],
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        only_last_turn_loss: bool = True,
    ) -> dict[str, list[int]] | None:
        text = _chatml_text(sources)
        enc = tokenizer(text, add_special_tokens=False)
        input_ids = list(enc.input_ids)
        if not input_ids or len(input_ids) > max_len:
            return None

        labels = [IGNORE_INDEX] * len(input_ids)
        assistant_indices = [i for i, msg in enumerate(sources) if msg.get("role") == "assistant"]
        if not assistant_indices:
            return None
        target_assistants = [assistant_indices[-1]] if only_last_turn_loss else assistant_indices

        cursor = 0
        for idx, msg in enumerate(sources):
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", ""))
            prefix = f"<|im_start|>{role}\n"
            prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
            content_ids = tokenizer(content, add_special_tokens=False).input_ids
            content_start = cursor + len(prefix_ids)
            content_end = content_start + len(content_ids)
            if idx in target_assistants:
                labels[content_start:content_end] = input_ids[content_start:content_end]
            msg_text = f"{prefix}{content}<|im_end|>\n"
            cursor += len(tokenizer(msg_text, add_special_tokens=False).input_ids)

        if not any(v != IGNORE_INDEX for v in labels):
            return None
        return {"input_ids": input_ids, "label": labels, "length": len(input_ids)}


IGNORE_INDEX = -100
DEFAULT_INPUT_PATH = ROOT / "data/benchmarks/sft_data/rendered_chatml_fim_train.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "data/benchmarks/sft_data/ours_corr_attn_train.json"
DEFAULT_CACHE_PATH = ROOT / "runs/benchmark/curation_data/ours_corr_attn_context.annotation_cache.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
CONTEXT_CACHE_VERSION = "corr_attn_context_v2"

_TEACHER_RATE_LIMIT_LOCK = threading.Lock()
_TEACHER_LAST_CALL_AT = 0.0


def wait_teacher_rate_limit(min_interval_seconds: float) -> None:
    global _TEACHER_LAST_CALL_AT
    if min_interval_seconds <= 0:
        return
    with _TEACHER_RATE_LIMIT_LOCK:
        now = time.monotonic()
        wait = float(min_interval_seconds) - (now - _TEACHER_LAST_CALL_AT)
        if wait > 0:
            time.sleep(wait)
        _TEACHER_LAST_CALL_AT = time.monotonic()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data


def write_jsonl(data: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {path}")


def sample_cache_key(sample: dict[str, Any]) -> str:
    uid = sample.get("uid")
    if uid:
        return f"uid:{uid}"
    source = sample.get("source_dataset", "")
    raw_id = sample.get("raw_id", "")
    if source or raw_id:
        return f"raw:{source}:{raw_id}"
    return f"idx:{sample.get('_source_index', '')}"


def load_annotation_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}

    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("key")
            if key:
                cache[str(key)] = obj
    print(f"Loaded annotation cache: {len(cache)} entries from {cache_path}")
    return cache


def save_annotation_cache(cache: dict[str, dict[str, Any]], path: str | Path | None) -> None:
    if not path:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for key in sorted(cache):
            f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
    print(f"Saved annotation cache: {len(cache)} entries to {cache_path}")


def apply_annotation_cache(
    samples: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    cache_version: str,
) -> int:
    applied = 0
    for sample in samples:
        key = sample_cache_key(sample)
        cached = cache.get(key)
        if not cached:
            continue
        if cached.get("cache_version") != cache_version:
            continue
        if cached.get("annotations"):
            sample["annotations"] = cached["annotations"]
        if cached.get("qwen_annotations"):
            sample["qwen_annotations"] = cached["qwen_annotations"]
        if sample.get("annotations") or sample.get("qwen_annotations"):
            applied += 1
    return applied


def update_annotation_cache(
    cache: dict[str, dict[str, Any]],
    samples: list[dict[str, Any]],
    cache_version: str,
) -> int:
    updated = 0
    for sample in samples:
        annotations = sample.get("annotations") or []
        qwen_annotations = sample.get("qwen_annotations") or []
        if not annotations and not qwen_annotations:
            continue
        key = sample_cache_key(sample)
        cache[key] = {
            "key": key,
            "cache_version": cache_version,
            "uid": sample.get("uid"),
            "source_dataset": sample.get("source_dataset"),
            "language": sample.get("language"),
            "raw_id": sample.get("raw_id"),
            "annotations": annotations,
            "qwen_annotations": qwen_annotations,
        }
        updated += 1
    return updated


def copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(msg) for msg in messages]


def setup_qwen_tokenizer(model_path: str, model_max_length: int) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=False,
        add_bos_token=False,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        cache_dir=None,
        model_max_length=model_max_length,
        truncation=True,
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
    )
    return setup_tokenizer(tokenizer)


def binarize_with_indices(
    samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    out: list[dict[str, Any]] = []
    source_indices: list[int] = []
    skipped = 0

    for idx, sample in enumerate(tqdm.tqdm(samples, desc="binarize", mininterval=5)):
        messages = copy_messages(sample.get("messages", []))
        result = chatml_format_preprocess(
            messages,
            tokenizer,
            max_len=max_len,
            only_last_turn_loss=sample.get("only_last_turn_loss", True),
        )
        if result is None:
            skipped += 1
            continue
        out.append(result)
        source_indices.append(idx)

    print(f"binarize: {len(samples)} -> {len(out)} (skipped {skipped})")
    return out, source_indices


def find_sublist(
    haystack: list[int],
    needle: list[int],
    *,
    start: int = 0,
    end: int | None = None,
    reverse: bool = False,
) -> int:
    if not needle:
        return -1
    end = len(haystack) if end is None else min(end, len(haystack))
    start = max(0, start)
    last = end - len(needle)
    if last < start:
        return -1
    indices = range(last, start - 1, -1) if reverse else range(start, last + 1)
    for idx in indices:
        if haystack[idx:idx + len(needle)] == needle:
            return idx
    return -1


def locate_segment(
    input_ids: list[int],
    tokenizer: transformers.PreTrainedTokenizer,
    text: str,
    *,
    start: int = 0,
    end: int | None = None,
    reverse: bool = False,
) -> tuple[int, list[tuple[int, int]]] | None:
    if not text:
        return None
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = list(enc.get("input_ids", []))
    offsets = list(enc.get("offset_mapping", []))
    if not token_ids:
        return None
    seq_start = find_sublist(input_ids, token_ids, start=start, end=end, reverse=reverse)
    if seq_start < 0:
        return None
    return seq_start, offsets


def map_simple_edges_to_chatml_bpe(
    *,
    edges: list[dict[str, Any]],
    simple_tokens: list[Any],
    prefix: str,
    completion: str,
    suffix: str,
    tokenizer: transformers.PreTrainedTokenizer,
    input_ids: list[int],
) -> list[dict[str, Any]]:
    """Map full-code simple-token edges to ChatML/Qwen token positions.

    FIM code order is ``prefix + completion + suffix`` while ChatML order is
    roughly ``<fim_prefix> prefix <fim_suffix> suffix <fim_middle>`` followed
    by the assistant completion.  Therefore prefix/suffix source tokens are
    prompt tokens in the sequence and may validly attend into completion dsts.
    """
    completion_start_char = len(prefix)
    suffix_start_char = completion_start_char + len(completion)

    completion_loc = locate_segment(input_ids, tokenizer, completion, reverse=True)
    completion_seq_start = completion_loc[0] if completion_loc else len(input_ids)
    prefix_loc = locate_segment(input_ids, tokenizer, prefix, end=completion_seq_start)
    suffix_loc = locate_segment(input_ids, tokenizer, suffix, end=completion_seq_start)

    segments: list[dict[str, Any]] = []
    if prefix_loc:
        segments.append({
            "name": "prefix",
            "char_start": 0,
            "char_end": len(prefix),
            "seq_start": prefix_loc[0],
            "offsets": prefix_loc[1],
        })
    if completion_loc:
        segments.append({
            "name": "completion",
            "char_start": completion_start_char,
            "char_end": suffix_start_char,
            "seq_start": completion_loc[0],
            "offsets": completion_loc[1],
        })
    if suffix_loc:
        segments.append({
            "name": "suffix",
            "char_start": suffix_start_char,
            "char_end": suffix_start_char + len(suffix),
            "seq_start": suffix_loc[0],
            "offsets": suffix_loc[1],
        })

    def simple_to_bpe(simple_idx: int) -> list[int]:
        if simple_idx < 0 or simple_idx >= len(simple_tokens):
            return []
        tok = simple_tokens[simple_idx]
        tok_start = int(tok.char_start)
        tok_end = int(tok.char_end)
        out: list[int] = []
        for seg in segments:
            if tok_start >= seg["char_end"] or tok_end <= seg["char_start"]:
                continue
            local_start = max(tok_start, seg["char_start"]) - seg["char_start"]
            local_end = min(tok_end, seg["char_end"]) - seg["char_start"]
            for bpe_idx, (b_start, b_end) in enumerate(seg["offsets"]):
                if local_start < b_end and b_start < local_end:
                    out.append(int(seg["seq_start"]) + bpe_idx)
        return out

    mapped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for edge in edges:
        qis = simple_to_bpe(int(edge["token_i_idx"]))
        qjs = simple_to_bpe(int(edge["token_j_idx"]))
        subtype = str(edge.get("subtype", "semantic") or "semantic")
        for qi in qis:
            for qj in qjs:
                key = (qi, qj, subtype)
                if qi == qj or qi >= qj or key in seen:
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


def select_source_indices(
    simple_tokens: list[Any],
    target_indices: set[int],
    *,
    completion_start: int,
    completion_end: int,
    max_context_tokens: int,
) -> set[int]:
    if max_context_tokens <= 0:
        return set(range(len(simple_tokens)))

    prefix = [
        i for i, tok in enumerate(simple_tokens)
        if int(tok.char_end) <= completion_start
    ]
    suffix = [
        i for i, tok in enumerate(simple_tokens)
        if int(tok.char_start) >= completion_end
    ]
    half = max_context_tokens // 2
    selected = set(prefix[-half:]) | set(suffix[:max_context_tokens - half])
    selected |= set(target_indices)
    return selected


def teacher_context_to_completion_edges(
    *,
    full_code: str,
    simple_tokens: list[Any],
    source_indices: set[int],
    target_indices: set[int],
    structural_edges: list[dict[str, Any]],
    language: str,
    api_key: str,
    api_base_url: str,
    model_name: str,
    max_tokens: int = 768,
    context_chars: int = 6000,
    max_edges: int = 96,
    min_interval_seconds: float = 0.0,
    max_retries: int = 0,
    retry_backoff_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    if not api_key or not source_indices or not target_indices:
        return []

    source_tokens = {
        i: simple_tokens[i].surface
        for i in sorted(source_indices)
        if 0 <= i < len(simple_tokens)
    }
    target_tokens = {
        i: simple_tokens[i].surface
        for i in sorted(target_indices)
        if 0 <= i < len(simple_tokens)
    }
    if not source_tokens or not target_tokens:
        return []

    client = OpenAI(api_key=api_key, base_url=api_base_url)
    system = (
        f"You are a {language} code dependency annotator. Return JSON only. "
        "No markdown. No prose. No comments. "
        'Schema: {"pairs":[{"i":0,"j":1,"reason":"dataflow"}]}. '
        "Allowed reasons: dataflow, semantic, api, defuse, call, return, type, bracket. "
        "Each pair must mean source token i helps determine target/completion token j. "
        "Use i only from source_tokens and j only from target_tokens. "
        "Do not repeat seeded_structural_edges."
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
    compact_structural = [
        [int(e["token_i_idx"]), int(e["token_j_idx"]), str(e.get("subtype", "semantic"))]
        for e in structural_edges
        if isinstance(e.get("token_i_idx"), int) and isinstance(e.get("token_j_idx"), int)
    ]
    request_payload = {
        "language": language,
        "code_context": code_for_teacher,
        "source_tokens": [[int(i), str(tok)] for i, tok in source_tokens.items()],
        "target_tokens": [[int(i), str(tok)] for i, tok in target_tokens.items()],
        "seeded_structural_edges": compact_structural,
        "task": (
            "Find only high-confidence missing dependency edges where source token i is useful for predicting "
            f'target/completion token j. Return exactly {{"pairs":[...]}} with at most {max_edges} edges. '
            'If none, return {"pairs":[]}.'
        ),
    }
    user = json.dumps(request_payload, ensure_ascii=False, separators=(",", ":"))

    response = None
    last_exc: Exception | None = None
    for attempt in range(max(0, int(max_retries)) + 1):
        wait_teacher_rate_limit(min_interval_seconds)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            break
        except Exception as exc:
            last_exc = exc
            try:
                wait_teacher_rate_limit(min_interval_seconds)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                break
            except Exception as exc2:
                last_exc = exc2
                if attempt < max(0, int(max_retries)):
                    sleep_s = float(retry_backoff_seconds) * (2 ** attempt)
                    print(f"    [corr_attn.teacher] API error: {exc2}; retry {attempt + 1}/{max_retries} after {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                else:
                    print(f"    [corr_attn.teacher] API error after retries: {exc2}")
    if response is None:
        return []

    raw = _extract_chat_message_text(response)
    payload = _parse_teacher_json(raw)
    if payload is None:
        preview = raw.replace("\n", "\\n")[:300]
        print(
            "    [corr_attn.teacher] failed to parse teacher JSON; "
            f"using structural edges only; raw={preview!r}; response={_summarize_chat_response(response)}"
        )
        return []

    valid_reasons = set(DEFAULT_GRAPH_REASON_WEIGHTS)
    pairs = payload if isinstance(payload, list) else payload.get("pairs", [])
    out: list[dict[str, Any]] = []
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
        if i not in source_indices or j not in target_indices or i == j:
            continue
        if reason not in valid_reasons:
            reason = "semantic"
        out.append({
            "token_i": source_tokens.get(i, ""),
            "token_j": target_tokens.get(j, ""),
            "source": "DeepSeekTeacher",
            "subtype": reason,
            "token_i_idx": i,
            "token_j_idx": j,
        })
        if len(out) >= max_edges:
            break
    return out


def add_context_teacher_annotations(
    samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    api_key: str | None,
    api_base_url: str,
    model_name: str,
    overwrite: bool,
    num_workers: int,
    teacher_max_tokens: int,
    teacher_context_chars: int,
    source_context_tokens: int,
    teacher_max_edges: int,
    teacher_min_interval_seconds: float,
    teacher_max_retries: int,
    teacher_retry_backoff_seconds: float,
) -> list[dict[str, Any]]:
    from src.annotate.neural_annot import SyntacticCheckerTool

    api_key = api_key or ""

    def annotate_one(sample: dict[str, Any]) -> dict[str, Any]:
        local = copy.deepcopy(sample)
        if (local.get("qwen_annotations") or local.get("annotations")) and not overwrite:
            return local

        full_code, completion, prefix, suffix = _extract_fim_parts(local)
        if not completion or "input_ids" not in local:
            return local

        language = _normalize_language(local.get("language"))
        simple_tokens = tokenize_code_for_annotation(full_code)
        completion_start = len(prefix)
        completion_end = completion_start + len(completion)
        target_indices = get_token_indices_in_span(simple_tokens, completion_start, completion_end)
        if not target_indices:
            return local

        source_indices = select_source_indices(
            simple_tokens,
            target_indices,
            completion_start=completion_start,
            completion_end=completion_end,
            max_context_tokens=source_context_tokens,
        )

        syntactic_tool = SyntacticCheckerTool()
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
            if e.token_i_idx in source_indices and e.token_j_idx in target_indices and e.token_i_idx != e.token_j_idx
        ]
        neural = teacher_context_to_completion_edges(
            full_code=full_code,
            simple_tokens=simple_tokens,
            source_indices=source_indices,
            target_indices=target_indices,
            structural_edges=structural,
            language=language,
            api_key=api_key,
            api_base_url=api_base_url,
            model_name=model_name,
            max_tokens=teacher_max_tokens,
            context_chars=teacher_context_chars,
            max_edges=teacher_max_edges,
            min_interval_seconds=teacher_min_interval_seconds,
            max_retries=teacher_max_retries,
            retry_backoff_seconds=teacher_retry_backoff_seconds,
        )
        simple_edges = _as_annotation_dicts(structural + neural)
        qwen_edges = map_simple_edges_to_chatml_bpe(
            edges=simple_edges,
            simple_tokens=simple_tokens,
            prefix=prefix,
            completion=completion,
            suffix=suffix,
            tokenizer=tokenizer,
            input_ids=list(local["input_ids"]),
        )
        if qwen_edges:
            local["annotations"] = simple_edges
            local["qwen_annotations"] = qwen_edges
        return local

    if num_workers <= 1:
        return list(tqdm.tqdm(map(annotate_one, samples), total=len(samples), desc="corr_attn.teacher"))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(tqdm.tqdm(executor.map(annotate_one, samples), total=len(samples), desc="corr_attn.teacher"))


def get_annotations(sample: dict[str, Any], field: str = "auto") -> list[dict[str, Any]]:
    if field == "qwen_annotations":
        return list(sample.get("qwen_annotations", []))
    if field == "annotations":
        return list(sample.get("annotations", []))
    if sample.get("qwen_annotations"):
        return list(sample.get("qwen_annotations", []))
    return list(sample.get("annotations", []))


def edge_to_attention_edge(edge: dict[str, Any], seq_len: int, labels: list[int]) -> dict[str, Any] | None:
    src = edge.get("src", edge.get("token_i_idx", edge.get("i", -1)))
    dst = edge.get("dst", edge.get("token_j_idx", edge.get("j", -1)))
    try:
        src = int(src)
        dst = int(dst)
    except (TypeError, ValueError):
        return None

    if src < 0 or dst < 0 or src >= seq_len or dst >= seq_len:
        return None
    if src >= dst:
        return None
    if labels[dst] == IGNORE_INDEX:
        return None

    return {
        "src": src,
        "dst": dst,
        "subtype": str(edge.get("subtype", edge.get("reason", "")) or ""),
    }


def build_attention_edges(
    sample: dict[str, Any],
    seq_len: int,
    labels: list[int],
    annotation_field: str,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()

    for edge in get_annotations(sample, annotation_field):
        if not isinstance(edge, dict):
            continue
        local = edge_to_attention_edge(edge, seq_len=seq_len, labels=labels)
        if local is None:
            continue
        key = (local["src"], local["dst"], local["subtype"])
        if key in seen:
            continue
        seen.add(key)
        edges.append(local)

    return edges


def normalize_api_model_name(name: str | None) -> str:
    aliases = {
        "deepseek-v4": DEFAULT_DEEPSEEK_MODEL,
        "deepseek-v4-fast": DEFAULT_DEEPSEEK_MODEL,
    }
    return aliases.get((name or DEFAULT_DEEPSEEK_MODEL).strip(), (name or DEFAULT_DEEPSEEK_MODEL).strip())


def parse_csv_filter(text: str) -> set[str]:
    return {part.strip().lower() for part in text.split(",") if part.strip()}


def filter_samples(
    samples: list[dict[str, Any]],
    *,
    source_datasets: set[str],
    languages: set[str],
    max_rows: int,
    per_group_limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    group_counts: dict[tuple[str, str], int] = {}

    for sample in samples:
        source = str(sample.get("source_dataset", "")).lower()
        language = str(sample.get("language", "")).lower()
        if source_datasets and source not in source_datasets:
            continue
        if languages and language not in languages:
            continue

        key = (source or "unknown", language or "unknown")
        if per_group_limit > 0:
            count = group_counts.get(key, 0)
            if count >= per_group_limit:
                continue
            group_counts[key] = count + 1

        out.append(sample)
        if max_rows > 0 and len(out) >= max_rows:
            break

    print(f"Filtered samples: {len(samples)} -> {len(out)}")
    if group_counts:
        for (source, language), count in sorted(group_counts.items()):
            print(f"  {source}/{language}: {count}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GraphSignal CE + correlation-attention SFT data.")
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--source_datasets", default="", help="Comma-separated source_dataset filter for train data.")
    parser.add_argument("--languages", default="", help="Comma-separated language filter for train data.")
    parser.add_argument("--max_rows", type=int, default=0, help="Global max train rows after filters. 0 means all.")
    parser.add_argument(
        "--per_group_limit",
        type=int,
        default=0,
        help="Max rows per (source_dataset, language) group after filters. 0 means all.",
    )
    parser.add_argument(
        "--edge_scope",
        choices=["context", "completion"],
        default="context",
        help=(
            "context: allow prompt/FIM-context source tokens to point into completion dst tokens. "
            "completion: legacy completion-only GraphSignal annotation."
        ),
    )
    parser.add_argument("--annotation_field", choices=["auto", "qwen_annotations", "annotations"], default="auto")
    parser.add_argument("--annotate_missing", action="store_true", help="Run src.annotate/teacher flow when edges are missing.")
    parser.add_argument("--api_base_url", default="https://api.deepseek.com")
    parser.add_argument("--api_model_name", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--graph_teacher_workers", type=int, default=1)
    parser.add_argument("--graph_teacher_max_tokens", type=int, default=2048)
    parser.add_argument("--graph_teacher_max_edges", type=int, default=96)
    parser.add_argument("--graph_teacher_context_chars", type=int, default=6000)
    parser.add_argument("--graph_teacher_chunk_size", type=int, default=64)
    parser.add_argument(
        "--graph_teacher_source_context_tokens",
        type=int,
        default=160,
        help="Max non-completion source-token candidates around the FIM hole for context-aware teacher annotation.",
    )
    parser.add_argument("--graph_teacher_min_interval_seconds", type=float, default=0.0)
    parser.add_argument("--graph_teacher_max_retries", type=int, default=0)
    parser.add_argument("--graph_teacher_retry_backoff_seconds", type=float, default=20.0)
    parser.add_argument("--annotation_cache_path", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--no_annotation_cache", action="store_true")
    parser.add_argument("--teacher_overwrite_annotations", action="store_true")
    parser.add_argument("--keep_metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.input_path)
    samples = filter_samples(
        samples,
        source_datasets=parse_csv_filter(args.source_datasets),
        languages=parse_csv_filter(args.languages),
        max_rows=args.max_rows,
        per_group_limit=args.per_group_limit,
    )
    tokenizer = setup_qwen_tokenizer(args.model_name_or_path, args.model_max_length)
    binarized, source_indices = binarize_with_indices(samples, tokenizer, args.model_max_length)

    staged: list[dict[str, Any]] = []
    for src_idx, b in zip(source_indices, binarized):
        sample = samples[src_idx]
        local = dict(sample)
        local["messages"] = copy_messages(sample.get("messages", []))
        local["input_ids"] = b["input_ids"]
        local["labels"] = b["label"]
        local["_source_index"] = src_idx
        if args.edge_scope == "context":
            # The legacy annotations are completion-only.  Context-aware CE +
            # attention training needs a fresh edge set where src may come
            # from prompt/FIM context and dst remains in completion.
            local.pop("annotations", None)
            local.pop("qwen_annotations", None)
        staged.append(local)

    cache_path = None if args.no_annotation_cache else args.annotation_cache_path
    cache_version = CONTEXT_CACHE_VERSION if args.edge_scope == "context" else "corr_attn_completion_v1"
    annotation_cache = load_annotation_cache(cache_path)
    if annotation_cache and not args.teacher_overwrite_annotations:
        applied = apply_annotation_cache(staged, annotation_cache, cache_version)
        print(f"Applied cached annotations: {applied}/{len(staged)}")

    missing = sum(1 for sample in staged if not (sample.get("qwen_annotations") or sample.get("annotations")))
    print(f"Annotation missing before teacher: {missing}/{len(staged)}")

    if args.annotate_missing and missing:
        api_key = (
            args.api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "--annotate_missing was set, but no API key was found. "
                "Set DEEPSEEK_API_KEY or pass --api_key."
            )
        to_annotate = [
            sample for sample in staged
            if args.teacher_overwrite_annotations
            or not (sample.get("qwen_annotations") or sample.get("annotations"))
        ]
        chunk_size = max(1, int(args.graph_teacher_chunk_size))
        print(
            "Teacher annotation plan: "
            f"{len(to_annotate)} samples, workers={args.graph_teacher_workers}, "
            f"chunk_size={chunk_size}, max_tokens={args.graph_teacher_max_tokens}, "
            f"context_chars={args.graph_teacher_context_chars}"
        )

        by_key = {sample_cache_key(sample): sample for sample in staged}
        for start in range(0, len(to_annotate), chunk_size):
            chunk = to_annotate[start:start + chunk_size]
            print(f"Teacher chunk {start // chunk_size + 1}: samples {start}..{start + len(chunk) - 1}")
            if args.edge_scope == "context":
                annotated_chunk = add_context_teacher_annotations(
                    samples=chunk,
                    tokenizer=tokenizer,
                    api_key=api_key,
                    api_base_url=args.api_base_url,
                    model_name=normalize_api_model_name(args.api_model_name),
                    overwrite=args.teacher_overwrite_annotations,
                    num_workers=args.graph_teacher_workers,
                    teacher_max_tokens=args.graph_teacher_max_tokens,
                    teacher_context_chars=args.graph_teacher_context_chars,
                    source_context_tokens=args.graph_teacher_source_context_tokens,
                    teacher_max_edges=args.graph_teacher_max_edges,
                    teacher_min_interval_seconds=args.graph_teacher_min_interval_seconds,
                    teacher_max_retries=args.graph_teacher_max_retries,
                    teacher_retry_backoff_seconds=args.graph_teacher_retry_backoff_seconds,
                )
            else:
                annotated_chunk = add_graphsignal_teacher_annotations(
                    samples=chunk,
                    tokenizer=tokenizer,
                    api_key=api_key,
                    api_base_url=args.api_base_url,
                    model_name=normalize_api_model_name(args.api_model_name),
                    overwrite=args.teacher_overwrite_annotations,
                    num_workers=args.graph_teacher_workers,
                    teacher_max_tokens=args.graph_teacher_max_tokens,
                    teacher_context_chars=args.graph_teacher_context_chars,
                )

            for annotated in annotated_chunk:
                by_key[sample_cache_key(annotated)].update(annotated)

            updated = update_annotation_cache(annotation_cache, annotated_chunk, cache_version)
            print(f"Teacher chunk cached: {updated}/{len(annotated_chunk)} annotated")
            save_annotation_cache(annotation_cache, cache_path)

    output: list[dict[str, Any]] = []
    total_edges = 0
    nonempty = 0
    src_prompt_edges = 0
    src_completion_edges = 0

    for sample in tqdm.tqdm(staged, desc="build_edges", mininterval=5):
        input_ids = list(sample["input_ids"])
        labels = list(sample["labels"])
        attention_edges = build_attention_edges(
            sample,
            seq_len=len(input_ids),
            labels=labels,
            annotation_field=args.annotation_field,
        )
        total_edges += len(attention_edges)
        nonempty += int(bool(attention_edges))
        for edge in attention_edges:
            if labels[edge["src"]] == IGNORE_INDEX:
                src_prompt_edges += 1
            else:
                src_completion_edges += 1

        record = {
            "input_ids": input_ids,
            "label": labels,
            "length": len(input_ids),
            "attention_edges": attention_edges,
        }
        if args.keep_metadata:
            record.update({
                "uid": sample.get("uid"),
                "source_dataset": sample.get("source_dataset"),
                "language": sample.get("language"),
                "raw_id": sample.get("raw_id"),
            })
        output.append(record)

    print(f"Attention-edge samples: {nonempty}/{len(output)}")
    print(f"Attention edges: {total_edges}")
    if total_edges:
        print(
            "Attention edge source split: "
            f"prompt/context={src_prompt_edges} ({src_prompt_edges / total_edges:.2%}), "
            f"completion={src_completion_edges} ({src_completion_edges / total_edges:.2%})"
        )
    write_jsonl(output, args.output_path)


if __name__ == "__main__":
    main()
