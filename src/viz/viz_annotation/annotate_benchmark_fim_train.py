#!/usr/bin/env python
"""Annotate benchmark ChatML-FIM train rows without changing the benchmark schema.

Input:  /mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/sft_data/rendered_chatml_fim_train.jsonl
Output: compact JSONL used by benchmark training/viewers:
        {"input_ids": ..., "label": ..., "length": ..., "attention_edges": ...}

The original rendered data is not rewritten. We only tokenize its existing
messages, annotate the reconstructed FIM code, and align annotation edges back
to the tokenized ChatML sequence.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.annotate.neural_annot import (
    AnnotatorAgent,
    SyntacticCheckerTool,
    get_thread_local_openai_client,
    _rate_limited_chat_completion,
)
from src.annotate.utils import map_simple_to_bpe, normalize_fim_annotation_edge_direction, tokenize_code_for_annotation
from src.annotate.viz_utils import visualize_correlations

IGNORE_INDEX = -100
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
MASK_TOKEN = "[MASK]"
INCOMPLETE_CODE_MARKER = "* Incomplete Code:\n"
LANG_MAP = {
    "python": "Python", "py": "Python",
    "cpp": "CPP", "c++": "CPP", "c": "C",
    "csharp": "C#", "c#": "C#",
    "java": "Java", "go": "Go",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", default="/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/sft_data/rendered_chatml_fim_train.jsonl")
    parser.add_argument("--output_path", default="/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function/interim/benchmark_legacy_fim/sft_data/ours_graphsignal_train.json")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--max_rounds", type=int, default=6)
    parser.add_argument("--llm_context_radius", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--oneshot_prompt_style",
        choices=["verbose", "compact"],
        default="verbose",
        help=(
            "Prompt encoding for --annotation_mode oneshot. verbose keeps the exact existing JSON shape; "
            "compact carries the same token/edge information in shorter arrays to reduce request latency."
        ),
    )
    parser.add_argument(
        "--annotation_mode",
        choices=["llm", "structural", "oneshot"],
        default="llm",
        help=(
            "llm keeps the original multi-round agent flow; structural uses only local tree-sitter edges; "
            "oneshot adds one OpenAI-compatible JSON call per row on top of structural edges."
        ),
    )
    parser.add_argument(
        "--edge_types",
        default="",
        help="Comma-separated subtype allow-list after token mapping, e.g. defuse,call,return,type. Empty keeps all.",
    )
    parser.add_argument(
        "--bpe_map_mode",
        choices=["all", "first"],
        default="all",
        help="all expands a word edge to every overlapping model token; first keeps only the first overlapping token per word.",
    )
    parser.add_argument(
        "--max_edges_per_target",
        type=int,
        default=0,
        help="If >0, keep at most this many incoming edges for each target token after subtype filtering.",
    )
    parser.add_argument(
        "--edge_scope",
        choices=["answer_target", "context_response_prompt", "all"],
        default="answer_target",
        help=(
            "answer_target keeps old target-only edges; context_response_prompt keeps "
            "context->response plus prompt-internal forward edges; all keeps every mapped edge."
        ),
    )
    parser.add_argument(
        "--static_viz_dir",
        default="outputs/viz_annotation/visualization/ours_graphsignal_static",
        help="Optional per-sample static HTML visualization directory. Use '' to disable.",
    )
    parser.add_argument(
        "--include_debug_fields",
        action="store_true",
        help="Include uid/language/raw annotations in output rows. Default keeps compact training format.",
    )
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output JSONL by skipping already written rows and appending new rows.",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Automatically enable --resume when the output JSONL already exists.",
    )
    parser.add_argument(
        "--row_cache_dir",
        "--cache_dir",
        default="",
        help=(
            "Optional per-source-row JSON cache directory. Completed rows are written atomically "
            "as row_XXXXXXXX.json and reused on restart, even if they were not appended yet."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, *, offset: int, limit: int) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            if limit > 0 and len(rows) >= limit:
                break
            line = line.strip()
            if line:
                rows.append((idx, json.loads(line)))
    return rows


def count_jsonl_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in existing resume file {path} at line {lineno}: {exc}") from exc
            count += 1
    return count


def row_cache_path(cache_dir: Path, row_index: int) -> Path:
    return cache_dir / f"row_{row_index:08d}.json"


def write_row_cache(cache_dir: Path | None, row_index: int, item: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = row_cache_path(cache_dir, row_index)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, final_path)


def read_row_cache(cache_dir: Path | None, row_index: int) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = row_cache_path(cache_dir, row_index)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.loads(f.readline())


def normalize_language(language: str) -> str:
    return LANG_MAP.get((language or "").strip().lower(), language or "Python")


def get_message(messages: list[dict[str, Any]], role: str) -> str:
    for msg in messages:
        if msg.get("role") == role:
            return str(msg.get("content", ""))
    return ""


def extract_fim_parts(row: dict[str, Any]) -> tuple[str, str, str]:
    # GoSingle canonical/rendered rows store the semantic split explicitly.
    if all(k in row for k in ("prefix", "target", "suffix")):
        return str(row.get("prefix", "")), str(row.get("target", "")), str(row.get("suffix", ""))

    prompt = str(row.get("fim_prompt") or get_message(row.get("messages", []), "user"))
    p0 = prompt.find(FIM_PREFIX)
    p1 = prompt.find(FIM_SUFFIX)
    p2 = prompt.find(FIM_MIDDLE)
    if p0 != -1 and p1 != -1 and p2 != -1 and p0 < p1 < p2:
        prefix = prompt[p0 + len(FIM_PREFIX):p1]
        suffix = prompt[p1 + len(FIM_SUFFIX):p2]
        completion = str(row.get("fim_completion") or get_message(row.get("messages", []), "assistant"))
        return prefix, completion, suffix

    marker_pos = prompt.find(INCOMPLETE_CODE_MARKER)
    code_start = marker_pos + len(INCOMPLETE_CODE_MARKER) if marker_pos != -1 else 0
    mask_pos = prompt.find(MASK_TOKEN, code_start)
    if mask_pos != -1:
        prefix = prompt[code_start:mask_pos]
        suffix = prompt[mask_pos + len(MASK_TOKEN):]
        completion = str(row.get("target") or row.get("fim_completion") or get_message(row.get("messages", []), "assistant"))
        return prefix, completion, suffix

    raise ValueError("Cannot parse FIM/[MASK] markers.")


def setup_tokenizer(model_path: str, *, max_len: int, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        model_max_length=max_len,
        truncation=True,
        padding_side="right",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|im_start|>"]})
    return tokenizer


def build_chatml_sequence(messages: list[dict[str, Any]]) -> tuple[str, int, int, int, str]:
    """Return sequence and offsets for user/assistant content in the existing messages."""
    parts: list[str] = []
    user_start = user_end = assistant_start = assistant_end = -1
    user_text = ""
    for msg in messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        header = f"<|im_start|>{role}\n"
        parts.append(header)
        content_start = sum(len(x) for x in parts)
        parts.append(content)
        content_end = sum(len(x) for x in parts)
        parts.append("<|im_end|>\n")
        if role == "user":
            user_start, user_end, user_text = content_start, content_end, content
        if role == "assistant":
            assistant_start, assistant_end = content_start, content_end
    if user_start < 0 or assistant_start < 0:
        raise ValueError("Expected user and assistant messages.")
    return "".join(parts), user_start, user_end, assistant_start, assistant_end, user_text


def tokenize_chatml(tokenizer, sequence: str, assistant_start: int) -> tuple[list[dict[str, Any]], list[int], list[int], int]:
    enc = tokenizer(sequence, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    if len(input_ids) > tokenizer.model_max_length:
        input_ids = input_ids[: tokenizer.model_max_length]
        offsets = offsets[: tokenizer.model_max_length]
    tokens = [
        {"token_id": tid, "char_start": s, "char_end": e}
        for tid, (s, e) in zip(input_ids, offsets, strict=True)
    ]
    output_token_start = len(input_ids)
    for i, (_, end) in enumerate(offsets):
        # Match existing SFT convention: assistant role header is ignored, assistant content/footer is trained.
        if end > assistant_start:
            output_token_start = i
            break
    labels = [IGNORE_INDEX] * output_token_start + input_ids[output_token_start:]
    return tokens, input_ids, labels, output_token_start


def fim_positions_in_user(user_text: str) -> tuple[int, int]:
    p0 = user_text.find(FIM_PREFIX)
    p1 = user_text.find(FIM_SUFFIX)
    p2 = user_text.find(FIM_MIDDLE)
    if p0 != -1 and p1 != -1 and p2 != -1 and p0 < p1 < p2:
        return p0 + len(FIM_PREFIX), p1 + len(FIM_SUFFIX)

    marker_pos = user_text.find(INCOMPLETE_CODE_MARKER)
    code_start = marker_pos + len(INCOMPLETE_CODE_MARKER) if marker_pos != -1 else 0
    mask_pos = user_text.find(MASK_TOKEN, code_start)
    if mask_pos != -1:
        return code_start, mask_pos + len(MASK_TOKEN)

    raise ValueError("Cannot parse FIM/[MASK] markers from user message.")


def remap_full_code_tokens_to_chatml(
    *,
    simple_tokens,
    prefix_len: int,
    completion_len: int,
    user_start: int,
    assistant_start: int,
    prefix_start_in_user: int,
    suffix_start_in_user: int,
) -> list[dict[str, Any]]:
    def remap_start(offset: int) -> int:
        if offset < prefix_len:
            return user_start + prefix_start_in_user + offset
        if offset < prefix_len + completion_len:
            return assistant_start + (offset - prefix_len)
        return user_start + suffix_start_in_user + (offset - prefix_len - completion_len)

    out: list[dict[str, Any]] = []
    for tok in simple_tokens:
        start = remap_start(tok.char_start)
        out.append({
            "surface": tok.surface,
            "token_id": int(tok.token_id),
            "char_start": int(start),
            "char_end": int(start + (tok.char_end - tok.char_start)),
        })
    return out


def to_dict(obj: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return dict(obj.__dict__)


EDGE_PRIORITY = {
    "dataflow": 0,
    "defuse": 1,
    "return": 2,
    "type": 3,
    "call": 4,
    "semantic": 5,
    "api": 6,
    "bracket": 7,
    "": 8,
}


def parse_edge_types(value: str) -> set[str]:
    return {x.strip().lower() for x in value.split(",") if x.strip()}


def structural_annotations(code: str, simple_tokens, language: str) -> list[dict[str, Any]]:
    tool = SyntacticCheckerTool()
    edges = tool.get_edges(code, simple_tokens, language=language, char_offset=0)
    annotations: list[dict[str, Any]] = []
    for edge in edges:
        if edge.token_i_idx < 0 or edge.token_j_idx < 0:
            continue
        annotations.append({
            "token_i_idx": int(edge.token_i_idx),
            "token_j_idx": int(edge.token_j_idx),
            "subtype": str(edge.reason),
            "source": "Structural",
        })
    return annotations



def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _loads_json_array(text: str) -> list[Any]:
    text = _strip_json_fence(text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start:end + 1])
    if isinstance(obj, dict):
        for key in ("edges", "annotations", "relations"):
            value = obj.get(key)
            if isinstance(value, list):
                return value
    if not isinstance(obj, list):
        raise ValueError("One-shot annotator must return a JSON array or an object containing an edges array.")
    return obj


def oneshot_llm_annotations(
    *,
    code: str,
    simple_tokens,
    language: str,
    structural_edges: list[dict[str, Any]],
    completion_simple_indices: set[int],
    context_radius: int,
    edge_scope: str,
    prompt_style: str = "verbose",
) -> list[dict[str, Any]]:
    """Ask the model for non-structural edges in one request."""
    if not completion_simple_indices:
        return []

    scope = (edge_scope or "answer_target").strip().lower()
    if scope == "context_response_prompt":
        annotation_target_indices = set(range(len(simple_tokens)))
        window_min = 0
        window_max = len(simple_tokens) - 1
    else:
        annotation_target_indices = set(completion_simple_indices)
        target_min = min(completion_simple_indices)
        target_max = max(completion_simple_indices)
        window_min = max(0, target_min - max(0, context_radius))
        window_max = min(len(simple_tokens) - 1, target_max + max(0, context_radius))
    visible_indices = set(range(window_min, window_max + 1)) | annotation_target_indices
    visible_token_rows = [
        (i, tok.surface, i in annotation_target_indices, i in completion_simple_indices)
        for i, tok in enumerate(simple_tokens)
        if i in visible_indices
    ]
    structural_edge_rows = [
        (
            int(edge.get("token_i_idx", -1)),
            int(edge.get("token_j_idx", -1)),
            str(edge.get("subtype", "")),
        )
        for edge in structural_edges
        if int(edge.get("token_i_idx", -1)) in visible_indices
        and int(edge.get("token_j_idx", -1)) in visible_indices
    ][:200]

    if prompt_style == "compact":
        visible_tokens: list[Any] = [[idx, text, int(is_target), int(is_mask)] for idx, text, is_target, is_mask in visible_token_rows]
        structural_preview: list[Any] = [[src, dst, subtype] for src, dst, subtype in structural_edge_rows]
        token_schema: Any = ["idx", "text", "is_target", "is_mask_completion"]
        structural_schema: Any = ["src", "dst", "subtype"]
    else:
        visible_tokens = [
            {
                "idx": idx,
                "text": text,
                "is_target": is_target,
                "is_mask_completion": is_mask,
            }
            for idx, text, is_target, is_mask in visible_token_rows
        ]
        structural_preview = [
            {"src": src, "dst": dst, "subtype": subtype}
            for src, dst, subtype in structural_edge_rows
        ]
        token_schema = {"idx": "int", "text": "str", "is_target": "bool", "is_mask_completion": "bool"}
        structural_schema = {"src": "int", "dst": "int", "subtype": "str"}

    client = get_thread_local_openai_client()
    model = os.environ.get("ANNOTATE_MODEL", "gpt-4o-mini")
    system = (
        "You annotate code-token dependency edges for training a code completion model. "
        "Return only valid JSON. Do not use tools. Do not explain."
    )
    user = {
        "language": language,
        "task": (
            "Given a complete code snippet, token list, and existing structural edges, "
            "add only high-confidence non-structural edges. "
            "Use subtypes only from: dataflow, semantic, api. "
            "Do not duplicate structural edges. Prefer useful edges; fewer is okay. "
            "Direction: src is the cue/context token, dst is the token it helps predict. "
            "For masked completion tokens, src may come from either prefix or suffix context. "
            "For non-completion prompt/context tokens, use normal forward dependencies only."
        ),
        "return_schema": [
            {"src": "int token idx", "dst": "int target token idx", "subtype": "dataflow|semantic|api"}
        ],
        "prompt_style": prompt_style,
        "token_schema": token_schema,
        "structural_edge_schema": structural_schema,
        "target_token_indices": sorted(annotation_target_indices),
        "masked_completion_token_indices": sorted(completion_simple_indices),
        "edge_scope": scope,
        "tokens": visible_tokens,
        "structural_edges": structural_preview,
        "code": code,
    }
    response = _rate_limited_chat_completion(
        client,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "[]"
    raw_edges = _loads_json_array(content)

    allowed_subtypes = {"dataflow", "semantic", "api"}
    visible_or_target = visible_indices | annotation_target_indices
    structural_keys = {
        (
            int(edge.get("token_i_idx", -1)),
            int(edge.get("token_j_idx", -1)),
            str(edge.get("subtype", "")).lower(),
        )
        for edge in structural_edges
    }
    annotations: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict):
            continue
        try:
            src = int(raw_edge.get("src", raw_edge.get("token_i_idx", -1)))
            dst = int(raw_edge.get("dst", raw_edge.get("token_j_idx", -1)))
        except (TypeError, ValueError):
            continue
        subtype = str(raw_edge.get("subtype", "semantic")).strip().lower()
        if subtype not in allowed_subtypes:
            continue
        if src < 0 or dst < 0 or src >= len(simple_tokens) or dst >= len(simple_tokens):
            continue
        if src not in visible_or_target or dst not in annotation_target_indices or src == dst:
            continue
        src_is_completion = src in completion_simple_indices
        dst_is_completion = dst in completion_simple_indices
        if scope == "context_response_prompt":
            if src_is_completion and dst_is_completion and src > dst:
                continue
            if not src_is_completion and not dst_is_completion and src > dst:
                continue
        elif dst not in completion_simple_indices:
            continue
        key = (src, dst, subtype)
        if key in seen or key in structural_keys:
            continue
        seen.add(key)
        annotations.append({
            "token_i_idx": src,
            "token_j_idx": dst,
            "subtype": subtype,
            "source": "OneShotLLM",
        })
    return annotations


def filter_edges(
    edges: list[dict[str, Any]],
    *,
    edge_types: set[str],
    max_edges_per_target: int,
) -> list[dict[str, Any]]:
    if edge_types:
        edges = [e for e in edges if str(e.get("subtype", "")).lower() in edge_types]
    if max_edges_per_target <= 0:
        return edges

    # Keep the most training-relevant relation types first when a target token
    # has too many incoming edges. This controls noisy call/bracket explosions.
    ranked = sorted(
        enumerate(edges),
        key=lambda item: (
            int(item[1].get("dst", -1)),
            EDGE_PRIORITY.get(str(item[1].get("subtype", "")).lower(), 99),
            int(item[1].get("src", -1)),
            item[0],
        ),
    )
    kept_indices: set[int] = set()
    counts: dict[int, int] = {}
    for original_idx, edge in ranked:
        dst = int(edge.get("dst", -1))
        count = counts.get(dst, 0)
        if count >= max_edges_per_target:
            continue
        counts[dst] = count + 1
        kept_indices.add(original_idx)
    return [edge for i, edge in enumerate(edges) if i in kept_indices]


def simple_to_qwen_edges(
    *,
    simple_tokens: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    token_offsets: list[dict[str, Any]],
    completion_simple_indices: set[int],
    edge_scope: str,
    bpe_map_mode: str,
) -> list[dict[str, Any]]:
    simple_ns = [SimpleNamespace(char_start=t["char_start"], char_end=t["char_end"]) for t in simple_tokens]
    qwen_ns = [SimpleNamespace(char_start=t["char_start"], char_end=t["char_end"]) for t in token_offsets]
    sorted_indices = sorted(range(len(simple_ns)), key=lambda i: simple_ns[i].char_start)
    sorted_s2b = map_simple_to_bpe([simple_ns[i] for i in sorted_indices], qwen_ns)
    s2b = {orig_idx: sorted_s2b.get(sorted_pos, []) for sorted_pos, orig_idx in enumerate(sorted_indices)}

    scope = (edge_scope or "answer_target").strip().lower()
    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for ann in annotations:
        si = int(ann.get("token_i_idx", -1))
        sj = int(ann.get("token_j_idx", -1))
        si_is_completion = si in completion_simple_indices
        sj_is_completion = sj in completion_simple_indices
        if scope == "answer_target" and not sj_is_completion:
            continue

        subtype = str(ann.get("subtype", "") or "")
        src_tokens = s2b.get(si, [])
        dst_tokens = s2b.get(sj, [])
        if bpe_map_mode == "first":
            src_tokens = src_tokens[:1]
            dst_tokens = dst_tokens[:1]

        for qi in src_tokens:
            for qj in dst_tokens:
                if qi == qj:
                    continue

                normalized = normalize_fim_annotation_edge_direction(
                    si_is_completion=si_is_completion,
                    sj_is_completion=sj_is_completion,
                    qi=qi,
                    qj=qj,
                    edge_scope=scope,
                )
                if normalized is None:
                    continue
                src, dst = normalized
                key = (src, dst, subtype)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({"src": int(src), "dst": int(dst), "subtype": subtype})
    return edges


def annotate_one(row_index: int, row: dict[str, Any], tokenizer, args: argparse.Namespace) -> dict[str, Any]:
    prefix, completion, suffix = extract_fim_parts(row)
    full_code = prefix + completion + suffix
    sequence, user_start, _user_end, assistant_start, _assistant_end, user_text = build_chatml_sequence(row["messages"])
    prefix_start_in_user, suffix_start_in_user = fim_positions_in_user(user_text)

    token_offsets, input_ids, labels, _output_token_start = tokenize_chatml(tokenizer, sequence, assistant_start)
    if len(input_ids) > args.max_len:
        raise ValueError(f"tokenized length {len(input_ids)} exceeds --max_len={args.max_len}")

    simple_raw = tokenize_code_for_annotation(full_code)
    completion_simple_indices = {
        i for i, tok in enumerate(simple_raw)
        if tok.char_start < len(prefix) + len(completion) and tok.char_end > len(prefix)
    }
    language = normalize_language(str(row.get("language", "")))
    if args.annotation_mode == "structural":
        annotations = structural_annotations(full_code, simple_raw, language)
    elif args.annotation_mode == "oneshot":
        structural = structural_annotations(full_code, simple_raw, language)
        llm_annotations = oneshot_llm_annotations(
            code=full_code,
            simple_tokens=simple_raw,
            language=language,
            structural_edges=structural,
            completion_simple_indices=completion_simple_indices,
            context_radius=args.llm_context_radius,
            edge_scope=args.edge_scope,
            prompt_style=args.oneshot_prompt_style,
        )
        annotations = structural + llm_annotations
    else:
        agent = AnnotatorAgent(language=language, max_rounds=args.max_rounds)
        correlations = agent.annotate(
            full_code,
            simple_raw,
            target_indices=None,
            ts_code=full_code,
            ts_char_offset=0,
        )
        annotations = [to_dict(c) for c in correlations]
    simple_tokens = remap_full_code_tokens_to_chatml(
        simple_tokens=simple_raw,
        prefix_len=len(prefix),
        completion_len=len(completion),
        user_start=user_start,
        assistant_start=assistant_start,
        prefix_start_in_user=prefix_start_in_user,
        suffix_start_in_user=suffix_start_in_user,
    )
    edges = simple_to_qwen_edges(
        simple_tokens=simple_tokens,
        annotations=annotations,
        token_offsets=token_offsets,
        completion_simple_indices=completion_simple_indices,
        edge_scope=args.edge_scope,
        bpe_map_mode=args.bpe_map_mode,
    )
    edges = filter_edges(
        edges,
        edge_types=parse_edge_types(args.edge_types),
        max_edges_per_target=args.max_edges_per_target,
    )

    out = {
        "input_ids": input_ids,
        "label": labels,
        "length": len(input_ids),
        "attention_edges": edges,
    }
    if args.include_debug_fields:
        out.update({
            "uid": row.get("uid"),
            "language": row.get("language"),
            "raw_id": row.get("raw_id"),
            "annotation_meta": {
                "source_row_index": row_index,
                "language_for_annotator": language,
                "simple_tokens": len(simple_tokens),
                "annotation_mode": args.annotation_mode,
                "bpe_map_mode": args.bpe_map_mode,
                "edge_types": sorted(parse_edge_types(args.edge_types)),
                "max_edges_per_target": args.max_edges_per_target,
                "llm_context_radius": args.llm_context_radius,
                "raw_annotations": len(annotations),
                "attention_edges": len(edges),
                "edge_scope": args.edge_scope,
                "oneshot_prompt_style": args.oneshot_prompt_style,
            },
            "annotations": annotations,
        })
    return out


def write_static_viz(row_index: int, compact_row: dict[str, Any], tokenizer, output_dir: Path) -> None:
    token_texts = tokenizer.convert_ids_to_tokens(compact_row["input_ids"])
    parts: list[str] = []
    subwords: list[SimpleNamespace] = []
    cursor = 0
    for i, token in enumerate(token_texts):
        text = token.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
        start = cursor
        end = start + len(text)
        parts.append(text)
        subwords.append(SimpleNamespace(surface=text, clean=text.strip() or text, token_id=i, char_start=start, char_end=end))
        cursor = end
    correlations = []
    for e in compact_row["attention_edges"]:
        src, dst = int(e["src"]), int(e["dst"])
        correlations.append(SimpleNamespace(
            token_i=token_texts[src], token_j=token_texts[dst], source="Neural",
            subtype=e.get("subtype", ""), token_i_idx=src, token_j_idx=dst,
        ))
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_correlations(
        correlations=correlations,
        title=f"Benchmark annotation edges · sample {row_index}",
        code="".join(parts),
        subwords=subwords,
        output_path=str(output_dir / f"sample_{row_index:05d}.html"),
        open_browser=False,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if args.auto_resume and args.overwrite:
        raise ValueError("Use either --auto_resume or --overwrite, not both.")
    if args.auto_resume and output_path.exists():
        args.resume = True
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")
    if output_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"{output_path} exists. Pass --overwrite to replace it, --resume to append after existing rows, "
            "or --auto_resume to resume automatically."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_rows = count_jsonl_rows(output_path) if args.resume else 0
    if args.resume and args.limit > 0 and completed_rows >= args.limit:
        print(f"Nothing to annotate. Output already has {completed_rows} rows, reaching --limit={args.limit} -> {output_path}")
        return
    effective_offset = args.offset + completed_rows
    effective_limit = 0 if args.limit <= 0 else max(0, args.limit - completed_rows)

    if args.annotation_mode != "structural" and not os.environ.get("OPENAI_API_KEY"):
        print("[warn] OPENAI_API_KEY is not set. Set it before running annotation.", file=sys.stderr)
    print(f"[info] annotation_mode={args.annotation_mode}")
    print(f"[info] bpe_map_mode={args.bpe_map_mode}")
    if args.auto_resume:
        print(f"[info] auto_resume enabled: output_exists={output_path.exists()}")
    if args.resume:
        print(f"[info] resume enabled: existing_rows={completed_rows}, next_input_row={effective_offset}")
    if args.edge_types:
        print(f"[info] edge_types={args.edge_types}")
    if args.max_edges_per_target > 0:
        print(f"[info] max_edges_per_target={args.max_edges_per_target}")
    if args.annotation_mode == "oneshot":
        print(f"[info] oneshot_prompt_style={args.oneshot_prompt_style}")
    if os.environ.get("ANNOTATE_MODEL"):
        print(f"[info] ANNOTATE_MODEL={os.environ['ANNOTATE_MODEL']}")
    if os.environ.get("OPENAI_BASE_URL"):
        print(f"[info] OPENAI_BASE_URL={os.environ['OPENAI_BASE_URL']}")

    tokenizer = setup_tokenizer(args.model_path, max_len=args.max_len, local_files_only=args.local_files_only)
    rows = load_rows(input_path, offset=effective_offset, limit=effective_limit)
    limit_text = "all" if args.limit <= 0 else str(args.limit)
    print(
        f"Loaded {len(rows)} rows from {input_path} "
        f"(requested_offset={args.offset}, effective_offset={effective_offset}, limit={limit_text})"
    )
    if not rows:
        print(f"Nothing to annotate. Output already has {completed_rows} rows -> {output_path}")
        return

    failures: dict[int, str] = {}
    pending: dict[int, dict[str, Any]] = {}
    next_to_write = rows[0][0]
    mode = "a" if args.resume else "w"
    written_this_run = 0
    cache_dir = Path(args.row_cache_dir) if args.row_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] row_cache_dir={cache_dir}")

    rows_to_submit: list[tuple[int, dict[str, Any]]] = []
    cache_hits = 0
    for idx, row in rows:
        cached = read_row_cache(cache_dir, idx)
        if cached is None:
            rows_to_submit.append((idx, row))
        else:
            pending[idx] = cached
            cache_hits += 1

    if cache_dir is not None:
        print(f"[info] row_cache_hits={cache_hits}, row_cache_misses={len(rows_to_submit)}")

    with output_path.open(mode, encoding="utf-8") as out_f:
        while next_to_write in pending:
            item_to_write = pending.pop(next_to_write)
            out_f.write(json.dumps(item_to_write, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            written_this_run += 1
            print(f"[checkpoint:cache] wrote row={next_to_write} total_rows={completed_rows + written_this_run}")
            next_to_write += 1

        print(f"[info] submit_rows={len(rows_to_submit)}, cached_pending={len(pending)}")
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
            futures = {pool.submit(annotate_one, idx, row, tokenizer, args): idx for idx, row in rows_to_submit}
            for future in tqdm(as_completed(futures), total=len(futures), desc="annotate"):
                idx = futures[future]
                try:
                    item = future.result()
                except Exception as exc:
                    failures[idx] = repr(exc)
                    print(f"[error] row {idx}: {exc}", file=sys.stderr)
                    continue

                write_row_cache(cache_dir, idx, item)
                pending[idx] = item
                print(f"[done] row={idx} tokens={item['length']} edges={len(item['attention_edges'])}")

                while next_to_write in pending:
                    item_to_write = pending.pop(next_to_write)
                    out_f.write(json.dumps(item_to_write, ensure_ascii=False) + "\n")
                    out_f.flush()
                    os.fsync(out_f.fileno())
                    written_this_run += 1
                    print(f"[checkpoint] wrote row={next_to_write} total_rows={completed_rows + written_this_run}")
                    next_to_write += 1

    if pending:
        pending_path = output_path.with_suffix(output_path.suffix + ".pending.json")
        pending_path.write_text(json.dumps({str(k): v for k, v in pending.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[warn] {len(pending)} completed rows could not be appended because an earlier row failed. "
            f"Saved them -> {pending_path}",
            file=sys.stderr,
        )

    print(f"Wrote {written_this_run} new rows -> {output_path}")

    if args.static_viz_dir:
        print(
            "[warn] static_viz_dir is disabled for resume-safe streaming output. "
            "Build visualization after annotation finishes.",
            file=sys.stderr,
        )

    if failures:
        failure_path = output_path.with_suffix(output_path.suffix + ".failures.json")
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[warn] {len(failures)} rows failed -> {failure_path}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
