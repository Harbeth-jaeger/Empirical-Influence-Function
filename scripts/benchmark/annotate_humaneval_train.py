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
        if len(surface) <= 1 and not surface.isdigit():
            continue
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
        span_items = _word_items(span_text, start, max_items=80 - len(doc_items))
        for item in span_items:
            prefix = span_text[: max(0, int(item["start"]) - start)].lower()
            last_returns = max(prefix.rfind("returns:"), prefix.rfind("return:"), prefix.rfind(":return"))
            last_args = max(prefix.rfind("args:"), prefix.rfind("arguments:"), prefix.rfind(":param"), prefix.rfind("params:"))
            item["doc_section"] = "returns" if last_returns >= 0 and last_returns > last_args else "other"
        doc_items.extend(span_items)
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


def add_docstring_to_code_edges(*, row: dict[str, Any], record: dict[str, Any], tokenizer: Any, max_edges: int, use_llm: bool = False) -> dict[str, Any]:
    if max_edges <= 0:
        return record
    doc_items, code_items = _extract_doc_and_code_items(row)
    if not doc_items or not code_items:
        record.setdefault("annotation_meta", {})["docstring_edges_added"] = 0
        return record

    pairs = _candidate_docstring_pairs(doc_items, code_items, max_edges=max_edges)
    if use_llm:
        payload = {
            "task": "Add only high-confidence docstring return-answer to completion-code dependency edges for Python FIM annotation.",
            "rules": [
                "Edges must go from a docstring Returns/return token to a completion code token.",
                "Never create docstring->docstring edges.",
                "Never create code->docstring edges.",
                "Ignore Args/parameter descriptions; function signature def-use covers parameters.",
                "Prefer return candidate answers that are absent from prompt code but present in completion code.",
                "Return at most %d pairs as JSON: {\"pairs\":[{\"doc\":0,\"code\":1}]}" % max_edges,
            ],
            "docstring_tokens": [[item["idx"], item["text"], item.get("doc_section", "other")] for item in doc_items],
            "code_tokens": [[item["idx"], item["text"], item.get("section", "")] for item in code_items],
        }
        system = "You are a precise Python docstring-return to completion-code dependency annotator. Return JSON only."
        client = get_thread_local_openai_client()
        model = os.environ.get("ANNOTATE_MODEL", "gpt-4o-mini")
        try:
            response = _rate_limited_chat_completion(
                client,
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = _rate_limited_chat_completion(
                client,
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                temperature=0,
                max_tokens=1024,
            )
        raw = getattr(response.choices[0].message, "content", "") if not isinstance(response, str) else response
        pairs.extend({**pair, "source": "DocstringTeacher"} for pair in _parse_docstring_pairs(str(raw)))

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
                source = str(pair.get("source", "DocstringTeacher"))
                key = (src, dst, "semantic")
                if key in existing:
                    continue
                existing.add(key)
                qwen_annotations.append(
                    {
                        "token_i": qwen_tokens[src].get("surface", ""),
                        "token_j": qwen_tokens[dst].get("surface", ""),
                        "source": source,
                        "subtype": "semantic",
                        "token_i_idx": src,
                        "token_j_idx": dst,
                    }
                )
                attention_edges.append({"src": src, "dst": dst, "subtype": "semantic"})
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
    p.add_argument("--input-path", type=Path, default=Path("data/benchmark/train_data/humaneval_python_chatml.jsonl"))
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/benchmark/train_data/humaneval_python_compact_annotated.jsonl"),
    )
    p.add_argument(
        "--annotation-cache-path",
        type=Path,
        default=Path("data/benchmark/train_data/humaneval_python_annotation_cache.jsonl"),
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
