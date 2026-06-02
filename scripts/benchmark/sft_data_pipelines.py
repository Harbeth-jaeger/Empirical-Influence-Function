from __future__ import annotations

import itertools
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import tqdm
import transformers
from openai import OpenAI

from scripts.benchmark.apply_governance_operator import (
    add_graphsignal_teacher_annotations,
    graph_signal_operator,
    token_cleaning_operator,
    xtf_token_filter_dataset,
    clear_curation_dataset,
)
from scripts.benchmark.sft_data_utils import (
    DEFAULT_DEEPSEEK_MODEL,
    _model_load_kwargs,
    binarize_samples,
    binarize_samples_with_indices,
    bridge_binarized_back,
    bridge_to_operator_format,
    copy_messages,
)

IGNORE_INDEX = -100


def apply_none(
    samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    logger: logging.Logger | None = None,
    **_,
) -> list[dict[str, Any]]:
    """不做治理，直接 binarize"""
    return binarize_samples(samples, tokenizer, max_len, logger=logger)


def apply_graph_signal(
    samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    keep_ratio: float = 0.6,
    mode: str = "hard_mask",
    annotation_field: str = "auto",
    api_key: str | None = None,
    api_base_url: str = "https://api.deepseek.com",
    model_name: str = "deepseek-v4",
    teacher_overwrite_annotations: bool = False,
    graph_teacher_workers: int = 1,
    graph_teacher_max_tokens: int = 768,
    graph_teacher_context_chars: int = 6000,
    logger: logging.Logger | None = None,
    **_,
) -> list[dict[str, Any]]:
    """
    graph_signal 算子：基于 annotation 图的 token 重要性。
    先 binarize，再把 binarized 的 input_ids/labels 填回 sample，让算子处理。
    """
    if logger:
        logger.info("[graph_signal] binarizing first ...")
    else:
        print(f"  [graph_signal] binarizing first ...")
    binarized, source_indices = binarize_samples_with_indices(samples, tokenizer, max_len, logger=logger)

    # 把 binarized 的 input_ids/labels 填回原 sample 给算子
    op_samples = []
    for src_idx, b in zip(source_indices, binarized):
        s = samples[src_idx]
        local = dict(s)
        local["messages"] = copy_messages(s.get("messages", []))
        local["input_ids"] = b["input_ids"]
        local["labels"] = b["label"]
        op_samples.append(local)

    missing = sum(1 for s in op_samples if not (s.get("qwen_annotations") or s.get("annotations")))
    if missing:
        if logger:
            logger.info(f"[graph_signal] {missing}/{len(op_samples)} samples lack annotations; calling teacher model {model_name} ...")
        else:
            print(f"  [graph_signal] {missing}/{len(op_samples)} samples lack annotations; calling teacher model {model_name} ...")
        op_samples = add_graphsignal_teacher_annotations(
            samples=op_samples,
            tokenizer=tokenizer,
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            api_base_url=api_base_url,
            model_name=model_name,
            overwrite=teacher_overwrite_annotations,
            num_workers=graph_teacher_workers,
            teacher_max_tokens=graph_teacher_max_tokens,
            teacher_context_chars=graph_teacher_context_chars,
        )

    if logger:
        logger.info(f"[graph_signal] applying operator (mode={mode}, keep_ratio={keep_ratio}) ...")
    else:
        print(f"  [graph_signal] applying operator (mode={mode}, keep_ratio={keep_ratio}) ...")
    governed = graph_signal_operator(
        samples=op_samples,
        mode=mode,
        keep_ratio=keep_ratio,
        annotation_field=annotation_field,
    )

    # 同步回 binarized
    result = []
    for g, b in zip(governed, binarized):
        merged = dict(b)
        if "labels" in g:
            lbl = g["labels"]
            merged["label"] = lbl if isinstance(lbl, list) else lbl.tolist()
        result.append(merged)
    return result


def apply_token_cleaning(
    samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    base_model_path: str,
    ref_model_path: str,
    keep_ratio: float = 0.6,
    device: str | None = None,
    device_map: str | None = "auto",
    attn_implementation: str | None = "sdpa",
    logger: logging.Logger | None = None,
    **_,
) -> list[dict[str, Any]]:
    """TokenCleaning baseline"""
    import torch
    from transformers import AutoModelForCausalLM

    if not base_model_path:
        raise ValueError("--base_model_path is required for token_cleaning")
    if not ref_model_path:
        raise ValueError("--ref_model_path is required for token_cleaning")

    msg = f"  [token_cleaning] loading base model from {base_model_path} (device_map={device_map}, attn={attn_implementation}) ..."
    logger.info(msg) if logger else print(msg)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **_model_load_kwargs(torch.bfloat16, device_map, attn_implementation),
    ).eval()
    msg = f"  [token_cleaning] loading ref model from {ref_model_path} (device_map={device_map}, attn={attn_implementation}) ..."
    logger.info(msg) if logger else print(msg)
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        **_model_load_kwargs(torch.bfloat16, device_map, attn_implementation),
    ).eval()

    if logger:
        logger.info("[token_cleaning] binarizing ...")
    else:
        print(f"  [token_cleaning] binarizing ...")
    binarized, _source_indices = binarize_samples_with_indices(samples, tokenizer, max_len, logger=logger)

    op_samples = []
    for b in binarized:
        local = dict(b)
        local["labels"] = local.pop("label")   # 算子用 "labels" key
        op_samples.append(local)

    if logger:
        logger.info(f"[token_cleaning] applying operator (keep_ratio={keep_ratio}) ...")
    else:
        print(f"  [token_cleaning] applying operator (keep_ratio={keep_ratio}) ...")
    governed = token_cleaning_operator(
        samples=op_samples,
        base_model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        keep_ratio=keep_ratio,
        device=device,
    )

    result = []
    for g, b in zip(governed, binarized):
        merged = dict(b)
        lbl = g["labels"]
        merged["label"] = lbl if isinstance(lbl, list) else lbl.tolist()
        result.append(merged)

    del base_model, ref_model
    import gc; gc.collect()
    return result


def apply_xtf(
    samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    base_model_path: str,
    pcp_threshold: float = 0.95,
    tr_percentile: float = 20.0,
    device: str | None = None,
    device_map: str | None = "auto",
    attn_implementation: str | None = "eager",
    logger: logging.Logger | None = None,
    **_,
) -> list[dict[str, Any]]:
    """XTF baseline"""
    import torch
    from transformers import AutoModelForCausalLM

    if not base_model_path:
        raise ValueError("--base_model_path is required for xtf")

    if attn_implementation == "flash_attention_2":
        raise ValueError("XTF needs output_attentions=True; use --attn_implementation eager or sdpa for xtf.")

    msg = f"  [xtf] loading model from {base_model_path} (device_map={device_map}, attn={attn_implementation}) ..."
    logger.info(msg) if logger else print(msg)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **_model_load_kwargs(torch.bfloat16, device_map, attn_implementation),
    ).eval()

    if logger:
        logger.info("[xtf] binarizing ...")
    else:
        print(f"  [xtf] binarizing ...")
    binarized, _source_indices = binarize_samples_with_indices(samples, tokenizer, max_len, logger=logger)

    op_samples = []
    for b in binarized:
        local = dict(b)
        local["labels"] = local.pop("label")
        op_samples.append(local)

    logger.info("[xtf] applying operator ...") if logger else print(f"  [xtf] applying operator ...")
    governed = xtf_token_filter_dataset(
        samples=op_samples,
        model=model,
        pcp_threshold=pcp_threshold,
        tr_percentile=tr_percentile,
        device=device,
    )

    result = []
    for g, b in zip(governed, binarized):
        merged = dict(b)
        lbl = g["labels"]
        merged["label"] = lbl if isinstance(lbl, list) else lbl.tolist()
        result.append(merged)

    del model
    import gc; gc.collect()
    return result


def apply_llm_code_cleaning(
    samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    api_key: str,
    api_base_url: str = "https://api.deepseek.com",
    model_name: str = DEFAULT_DEEPSEEK_MODEL,
    num_workers: int = 1,
    logger: logging.Logger | None = None,
    **_,
) -> list[dict[str, Any]]:
    """
    LLM-CleanCode baseline。
    对原始 messages 里 assistant 的 fim_completion 做 3-step 清洗，
    再替换回 messages，最后 binarize。
    注意：FIM completion 是代码片段，oracle equivalence check 对片段无效，
    这里退化为只调用 LLM 做 rename/modularize/planning，不做 oracle 校验。
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("--api_key or DEEPSEEK_API_KEY is required for llm_code_cleaning")
    model_name = normalize_api_model_name(model_name, logger=logger)
    client = OpenAI(api_key=api_key, base_url=api_base_url)
    num_workers = max(1, int(num_workers or 1))

    prompts_map = {
        "rename": "Rename the variables in the following code snippet to be descriptive, meaningful, and consistent. Return only the modified code snippet.",
        "modularize": "Refactor the following code snippet making it more modular and readable. Return only the modified code snippet.",
        "planning": "Add a brief inline comment at the start of the following code snippet describing what it does. Return only the modified code snippet.",
    }

    def clean_snippet(snippet: str, language: str) -> str:
        current = snippet
        for stage, instruction in prompts_map.items():
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": f"You are an expert {language} developer. Return only code, no explanation."},
                        {"role": "user", "content": f"{instruction}\n\n```{language}\n{current}\n```"},
                    ],
                    temperature=0.3,
                    max_tokens=512,
                )
                candidate = resp.choices[0].message.content or ""
                # 提取代码块
                import re
                blocks = re.findall(r"```(?:\w+)?\n(.*?)```", candidate, re.DOTALL)
                candidate = blocks[0].strip() if blocks else candidate.strip()
                if candidate:
                    current = candidate
            except Exception as e:
                print(f"    [llm_code_cleaning] stage={stage} error: {e}")
                continue
        return current

    def clean_sample(i: int, s: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if logger and (i % 10 == 0):
            logger.info(f"llm_code_cleaning: {i}/{len(samples)} processed")
        local = dict(s)
        local["messages"] = copy_messages(local.get("messages", []))
        language = local.get("language", "python")
        fim_completion = local.get("fim_completion", "")
        if fim_completion:
            raw = fim_completion.replace("\\n", "\n").replace("\\t", "\t")
            cleaned = clean_snippet(raw, language)
            # 替换 messages 里 assistant 的 content
            for msg in local.get("messages", []):
                if msg.get("role") == "assistant":
                    msg["content"] = cleaned
            local["fim_completion"] = cleaned
        return i, local

    cleaned_samples: list[dict[str, Any] | None] = [None] * len(samples)
    if num_workers == 1:
        iterator = enumerate(tqdm.tqdm(samples, desc="  llm_code_cleaning"))
        for i, s in iterator:
            idx, local = clean_sample(i, s)
            cleaned_samples[idx] = local
    else:
        msg = f"  [llm_code_cleaning] using {num_workers} workers"
        logger.info(msg) if logger else print(msg)
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(clean_sample, i, s): i
                for i, s in enumerate(samples)
            }
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="  llm_code_cleaning"):
                idx = futures[future]
                try:
                    out_idx, local = future.result()
                except Exception as exc:
                    print(f"    [llm_code_cleaning] sample={idx} error: {exc}")
                    _, local = clean_sample(idx, samples[idx])
                    out_idx = idx
                cleaned_samples[out_idx] = local

    return binarize_samples([s for s in cleaned_samples if s is not None], tokenizer, max_len, logger=logger)


def apply_clear(
    samples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    api_key: str,
    api_base_url: str = "https://api.deepseek.com",
    model_name: str = DEFAULT_DEEPSEEK_MODEL,
    gamma: float = 0.5,
    eta: float = 0.8,
    clear_stage: str = "filter_correct",
    clear_alpha: float = 0.5,
    clear_consistency_samples: int = 3,
    clear_consistency_temperature: float = 0.8,
    clear_candidate_model_name: str | None = None,
    clear_candidate_temperature: float = 0.2,
    clear_max_tokens: int = 512,
    clear_num_workers: int = 8,
    logger: logging.Logger | None = None,
    **_,
) -> list[dict[str, Any]]:
    """
    CLEAR baseline。
    Auto-Filter: 用 BSDetector 置信度过滤低质样本。
    Auto-Correct: 生成候选 completion，若 judge 认为显著优于原 target 则替换。
    字段桥接：chatml messages → prompt/response。
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("--api_key or DEEPSEEK_API_KEY is required for clear")
    model_name = normalize_api_model_name(model_name, logger=logger)
    if clear_candidate_model_name:
        clear_candidate_model_name = normalize_api_model_name(clear_candidate_model_name, logger=logger)
    client = OpenAI(api_key=api_key, base_url=api_base_url)
    bridged = bridge_to_operator_format(samples)

    msg = (
        f"[clear] stage={clear_stage}, gamma={gamma}, eta={eta}, alpha={clear_alpha}, "
        f"model={model_name}, candidate_model={clear_candidate_model_name or model_name}"
    )
    logger.info(msg) if logger else print(f"  {msg}")
    curated_bridged = clear_curation_dataset(
        samples=bridged,
        client=client,
        model_name=model_name,
        stage=clear_stage,
        gamma=gamma,
        eta=eta,
        alpha=clear_alpha,
        consistency_samples=clear_consistency_samples,
        consistency_temperature=clear_consistency_temperature,
        candidate_model_name=clear_candidate_model_name,
        candidate_temperature=clear_candidate_temperature,
        max_tokens=clear_max_tokens,
        num_workers=clear_num_workers,
    )
    replaced = sum(1 for s in curated_bridged if s.get("clear_metadata", {}).get("replaced"))
    logger.info(f"[clear] curated: {len(bridged)} -> {len(curated_bridged)}, replaced={replaced}") if logger else print(f"  [clear] curated: {len(bridged)} -> {len(curated_bridged)}, replaced={replaced}")

    kept_samples = []
    for curated in curated_bridged:
        local = dict(curated)
        local["messages"] = copy_messages(local.get("messages", []))
        kept_samples.append(local)

    return binarize_samples(kept_samples, tokenizer, max_len, logger=logger)


# ── 主流程 ────────────────────────────────────────────────────────────────────

OPERATOR_FUNCS = {
    "none": apply_none,
    "graph_signal": apply_graph_signal,
    "token_cleaning": apply_token_cleaning,
    "xtf": apply_xtf,
    "llm_code_cleaning": apply_llm_code_cleaning,
    "clear": apply_clear,
}
