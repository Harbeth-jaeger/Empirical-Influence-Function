from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import tqdm
import transformers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sft.binarize_data import chatml_format_preprocess, setup_tokenizer

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy the ChatML message containers without deep-copying string payloads."""
    return [dict(msg) for msg in messages]

class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_run_logging(log_path: str | None) -> tuple[logging.Logger, Any | None]:
    logger = logging.getLogger("sft_data_convert")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(stream_handler)

    if not log_path:
        return logger, None

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(f"Logging to {log_path}")
    return logger, log_fh


def _model_load_kwargs(torch_dtype: Any, device_map: str | None, attn_implementation: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        # Standard HF model loading is enough here; keeping this False avoids
        # fetching repo-specific custom_generate code during from_pretrained().
        "trust_remote_code": False,
    }
    if device_map:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs

def load_jsonl(path: str) -> list[dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(data)} samples from {path}")
    return data


def write_jsonl(data: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {path}")


def setup_qwen_tokenizer(model_path: str) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=False,
        add_bos_token=False,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        cache_dir=None,
        model_max_length=8192 * 5,
        truncation=True,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer = setup_tokenizer(tokenizer)
    return tokenizer


def normalize_api_model_name(model_name: str | None, logger: logging.Logger | None = None) -> str:
    """
    Keep old experiment commands working after DeepSeek renamed the V4 endpoints.

    DeepSeek currently accepts deepseek-v4-flash / deepseek-v4-pro, while some
    older benchmark notes used deepseek-v4 to mean the fast/flash endpoint.
    """
    name = (model_name or DEFAULT_DEEPSEEK_MODEL).strip()
    aliases = {
        "deepseek-v4": DEFAULT_DEEPSEEK_MODEL,
        "deepseek-v4-fast": DEFAULT_DEEPSEEK_MODEL,
    }
    normalized = aliases.get(name, name)
    if normalized != name:
        msg = f"[api] mapped api_model_name {name!r} -> {normalized!r}"
        logger.info(msg) if logger else print(msg)
    return normalized


def binarize_samples(
    raw_samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    logger: logging.Logger | None = None,
    log_interval: int = 50,
) -> list[dict[str, Any]]:
    """把 chatml jsonl 样本 tokenize 成 {input_ids, label, length}"""
    out = []
    skipped = 0
    for i, sample in enumerate(tqdm.tqdm(raw_samples, desc="  binarize", mininterval=5)):
        if logger and (i % log_interval == 0):
            logger.info(f"binarize: {i}/{len(raw_samples)} samples processed")
        messages = sample.get("messages", [])
        if not messages:
            skipped += 1
            continue
        result = chatml_format_preprocess(
            sources=copy_messages(messages),
            tokenizer=tokenizer,
            max_len=max_len,
            only_last_turn_loss=sample.get("only_last_turn_loss", True),
        )
        if result is None:
            skipped += 1
            continue
        out.append(result)
    print(f"  binarize: {len(raw_samples)} -> {len(out)} (skipped {skipped})")
    return out


def binarize_samples_with_indices(
    raw_samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    logger: logging.Logger | None = None,
    log_interval: int = 50,
) -> tuple[list[dict[str, Any]], list[int]]:
    """
    Binarize samples and keep the source row index for every retained item.

    This matters because chatml_format_preprocess returns None for over-length
    rows. Governance operators that need both raw text and tokenized labels must
    align against the retained rows, not zip the full raw list with the shorter
    binarized list.
    """
    out: list[dict[str, Any]] = []
    source_indices: list[int] = []
    skipped = 0
    for i, sample in enumerate(tqdm.tqdm(raw_samples, desc="  binarize", mininterval=5)):
        if logger and (i % log_interval == 0):
            logger.info(f"binarize: {i}/{len(raw_samples)} samples processed")
        messages = sample.get("messages", [])
        if not messages:
            skipped += 1
            continue
        result = chatml_format_preprocess(
            sources=copy_messages(messages),
            tokenizer=tokenizer,
            max_len=max_len,
            only_last_turn_loss=sample.get("only_last_turn_loss", True),
        )
        if result is None:
            skipped += 1
            continue
        out.append(result)
        source_indices.append(i)
    print(f"  binarize: {len(raw_samples)} -> {len(out)} (skipped {skipped})")
    return out, source_indices


# ── 字段桥接：把 chatml 格式转换为算子需要的字段 ──────────────────────────────

def bridge_to_operator_format(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    chatml 样本 → 算子通用格式
    算子需要 input_ids / labels（token id 列表），以及可选的 prompt/response（文本）
    这里先把 messages 里的内容提取成文本字段，token id 由 binarize 之后填入。
    """
    out = []
    for s in samples:
        local = dict(s)
        messages = s.get("messages", [])
        local["messages"] = copy_messages(messages)
        # 提取 user prompt 和 assistant response 的文本（供 LLM 类算子使用）
        user_content = ""
        assistant_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
            elif msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")
        local["prompt"] = user_content.replace("\\n", "\n").replace("\\t", "\t")
        local["response"] = assistant_content.replace("\\n", "\n").replace("\\t", "\t")
        out.append(local)
    return out


def bridge_binarized_back(
    operator_samples: list[dict[str, Any]],
    binarized: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    把算子修改后的 labels（token 级 mask）合并回 binarized 结果。
    TokenCleaning / XTF 直接修改 sample["labels"]，需要同步回 binarized。
    """
    # 如果算子修改了 labels，用算子的结果覆盖 binarized 的 label
    out = []
    for op_s, bin_s in zip(operator_samples, binarized):
        merged = dict(bin_s)
        if "labels" in op_s:
            # 算子输出的 labels 是 token 级 list，直接替换 binarize 的 label
            merged["label"] = op_s["labels"] if isinstance(op_s["labels"], list) else op_s["labels"].tolist()
        out.append(merged)
    return out


