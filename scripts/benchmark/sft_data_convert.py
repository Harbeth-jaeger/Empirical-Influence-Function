"""
sft_data_convert.py
====================
读取原始 chatml jsonl → 应用治理算子 → binarize → 写出可直接喂给 train.py 的 jsonl

支持的 operator:
  graph_signal      我们自己的方法（不需要模型和 API）
  token_cleaning    TokenCleaning baseline（需要 base_model + ref_model）
  xtf               XTF baseline（需要 base_model）
  llm_code_cleaning LLM-CleanCode baseline（需要 LLM API）
  clear             CLEAR baseline（需要 LLM API）
  none              不做治理，直接 binarize（用于对照组）

用法示例见文件末尾。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import tqdm
import transformers
from openai import OpenAI
import logging

# ── 路径设置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # Empirical-Influence-Function/
sys.path.insert(0, str(ROOT))

DEFAULT_INPUT_PATH = ROOT / "data/benchmarks/sft_data/rendered_chatml_fim_train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmarks/sft_data"
DEFAULT_LOG_DIR = ROOT / "runs/benchmark/curation_data"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

from scripts.benchmark.apply_governance_operator import (
    add_graphsignal_teacher_annotations,
    graph_signal_operator,
    token_cleaning_operator,
    xtf_token_filter_dataset,
    clear_curation_dataset,
)
from src.sft.binarize_data import chatml_format_preprocess, setup_tokenizer

IGNORE_INDEX = -100


# ── 工具函数 ──────────────────────────────────────────────────────────────────

from scripts.benchmark.sft_data_utils import (
    _model_load_kwargs,
    binarize_samples,
    binarize_samples_with_indices,
    bridge_binarized_back,
    bridge_to_operator_format,
    load_jsonl,
    normalize_api_model_name,
    setup_qwen_tokenizer,
    setup_run_logging,
    write_jsonl,
)

# ── 各算子的 apply 函数 ──────────────────────────────────────────────────────

from scripts.benchmark.sft_data_pipelines import (
    OPERATOR_FUNCS,
    apply_clear,
    apply_graph_signal,
    apply_llm_code_cleaning,
    apply_none,
    apply_token_cleaning,
    apply_xtf,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply governance operator and binarize training data.")
    p.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH),
                   help="原始 chatml jsonl 路径，如 data/benchmarks/sft_data/rendered_chatml_fim_train.jsonl")
    p.add_argument("--output_path", default=None,
                   help="输出 binarized jsonl 路径，如 data/benchmarks/sft_data/token_cleaning_train.jsonl")
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR),
                   help="[operator=all] 输出目录")
    p.add_argument("--operator", default="all",
                   choices=["all"] + list(OPERATOR_FUNCS.keys()),
                   help="治理算子名称；all 会生成 5 个 baseline 文件")
    p.add_argument("--tokenizer_path", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                   help="tokenizer 路径（本地或 HuggingFace hub）")
    p.add_argument("--max_len", type=int, default=4096,
                   help="最大 token 长度，超过的样本会被过滤")

    # token_cleaning / xtf 专用
    p.add_argument("--base_model_path", default=None,
                   help="[token_cleaning/xtf] base model 路径")
    p.add_argument("--ref_model_path", default=None,
                   help="[token_cleaning] reference model 路径（通常是 SFT 后的 checkpoint）")
    p.add_argument("--keep_ratio", type=float, default=0.6,
                   help="[token_cleaning/graph_signal] 保留 token 的比例")
    p.add_argument("--device", default=None,
                   help="[token_cleaning/xtf] tensor 输入设备；默认使用模型首个参数所在设备")
    p.add_argument("--device_map", default="auto",
                   help="[token_cleaning/xtf] transformers device_map；auto 可使用多卡")
    p.add_argument("--attn_implementation", default="sdpa",
                   help="[token_cleaning/xtf] eager/sdpa/flash_attention_2；XTF 需要 attentions，建议 eager")

    # xtf 专用
    p.add_argument("--pcp_threshold", type=float, default=0.95,
                   help="[xtf] Knowledge Novelty 过滤阈值（PCP > 此值则过滤）")
    p.add_argument("--tr_percentile", type=float, default=20.0,
                   help="[xtf] Task Relevance 过滤百分位")

    # graph_signal 专用
    p.add_argument("--gs_mode", default="hard_mask", choices=["hard_mask", "soft_weight"],
                   help="[graph_signal] 治理模式")
    p.add_argument("--annotation_field", default="auto",
                   help="[graph_signal] annotation 字段名（auto/qwen_annotations/annotations）")
    p.add_argument("--graph_teacher_workers", type=int, default=1,
                   help="[graph_signal] teacher API 并发数；DeepSeek flash 可先试 16，限流则降低")
    p.add_argument("--graph_teacher_max_tokens", type=int, default=768,
                   help="[graph_signal] teacher 单次输出上限，越小越快但边可能更少")
    p.add_argument("--graph_teacher_context_chars", type=int, default=6000,
                   help="[graph_signal] 发给 teacher 的代码上下文字符数；0 表示发送完整代码")

    # LLM API 类算子专用
    p.add_argument("--api_key", default=None,
                   help="[llm_code_cleaning/clear] LLM API key")
    p.add_argument("--api_base_url", default="https://api.deepseek.com",
                   help="[llm_code_cleaning/clear] LLM API base URL")
    p.add_argument("--api_model_name", default=DEFAULT_DEEPSEEK_MODEL,
                   help="[llm_code_cleaning/clear] LLM 模型名称")
    p.add_argument("--llm_cleaning_num_workers", type=int, default=1,
                   help="[llm_code_cleaning] API 并发线程数；DeepSeek flash 可先试 8，限流则降低")
    p.add_argument("--clear_gamma", type=float, default=0.5,
                   help="[clear] Auto-Filter 置信度阈值")
    p.add_argument("--clear_eta", type=float, default=0.8,
                   help="[clear] Auto-Correct 置信度阈值")
    p.add_argument("--clear_stage", default="filter_correct",
                   choices=["filter", "correct", "filter_correct"],
                   help="[clear] 执行 filter、correct 或 filter_correct")
    p.add_argument("--clear_alpha", type=float, default=0.5,
                   help="[clear] BSDetector C = alpha*O + (1-alpha)*S")
    p.add_argument("--clear_consistency_samples", type=int, default=3,
                   help="[clear] Observed Consistency 的高温采样次数")
    p.add_argument("--clear_consistency_temperature", type=float, default=0.8,
                   help="[clear] Observed Consistency 的采样温度")
    p.add_argument("--clear_candidate_model_name", default=None,
                   help="[clear] Auto-Correct 生成候选的模型；默认同 --api_model_name")
    p.add_argument("--clear_candidate_temperature", type=float, default=0.2,
                   help="[clear] Auto-Correct 候选生成温度")
    p.add_argument("--clear_max_tokens", type=int, default=512,
                   help="[clear] API 生成/判别 max_tokens")
    p.add_argument("--clear_num_workers", type=int, default=8,
                   help="[clear] API 并发线程数；太大可能触发限流")

    # 数据量控制
    p.add_argument("--max_samples", type=int, default=0,
                   help="只处理前 N 条样本（0 = 全部），用于调试")
    p.add_argument("--teacher_overwrite_annotations", action="store_true",
                   help="[graph_signal] 即使已有 annotations 也重新调用 teacher")
    p.add_argument("--log_path", default=None,
                   help="同时写 stdout/stderr/tqdm 的日志文件；默认写到 log_dir/sft_data_convert_<operator>_<timestamp>.log")
    p.add_argument("--log_dir", default=str(DEFAULT_LOG_DIR),
                   help="默认日志目录，长任务建议保留在 runs/benchmark/curation_data")

    return p.parse_args()


def _default_output_path(operator: str, output_dir: str) -> str:
    names = {
        "graph_signal": "ours_graphsignal_train.json",
        "token_cleaning": "token_cleaning_train.json",
        "xtf": "xtf_train.json",
        "llm_code_cleaning": "llm_cleancode_train.json",
        "clear": "clear_train.json",
        "none": "none_train.json",
    }
    return str(Path(output_dir) / names[operator])


def main() -> None:
    args = parse_args()
    if args.log_path is None:
        log_operator = args.operator if args.operator != "all" else "all"
        args.log_path = str(Path(args.log_dir) / f"sft_data_convert_{log_operator}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger, log_fh = setup_run_logging(args.log_path)

    # 读数据
    samples = load_jsonl(args.input_path)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        print(f"  [debug] using only {len(samples)} samples")

    # 过滤掉没有 messages 的样本
    samples = [s for s in samples if s.get("messages")]
    print(f"  Valid samples (with messages): {len(samples)}")

    # 加载 tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer_path} ...")
    tokenizer = setup_qwen_tokenizer(args.tokenizer_path)

    if args.operator == "all":
        operators = ["graph_signal", "token_cleaning", "xtf", "llm_code_cleaning", "clear"]
    else:
        operators = [args.operator]

    # 组装 kwargs
    kwargs: dict[str, Any] = dict(
        tokenizer=tokenizer,
        max_len=args.max_len,
        # token_cleaning / xtf
        base_model_path=args.base_model_path,
        ref_model_path=args.ref_model_path,
        keep_ratio=args.keep_ratio,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        # xtf
        pcp_threshold=args.pcp_threshold,
        tr_percentile=args.tr_percentile,
        # graph_signal
        mode=args.gs_mode,
        annotation_field=args.annotation_field,
        # LLM API
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        model_name=args.api_model_name,
        num_workers=args.llm_cleaning_num_workers,
        teacher_overwrite_annotations=args.teacher_overwrite_annotations,
        graph_teacher_workers=args.graph_teacher_workers,
        graph_teacher_max_tokens=args.graph_teacher_max_tokens,
        graph_teacher_context_chars=args.graph_teacher_context_chars,
        # clear
        gamma=args.clear_gamma,
        eta=args.clear_eta,
        clear_stage=args.clear_stage,
        clear_alpha=args.clear_alpha,
        clear_consistency_samples=args.clear_consistency_samples,
        clear_consistency_temperature=args.clear_consistency_temperature,
        clear_candidate_model_name=args.clear_candidate_model_name,
        clear_candidate_temperature=args.clear_candidate_temperature,
        clear_max_tokens=args.clear_max_tokens,
        clear_num_workers=args.clear_num_workers,
        logger=logger,
    )

    failures: dict[str, str] = {}
    for operator in operators:
        output_path = args.output_path if len(operators) == 1 and args.output_path else _default_output_path(operator, args.output_dir)

        print(f"\n{'='*60}")
        print(f"  Operator : {operator}")
        print(f"  Input    : {args.input_path}")
        print(f"  Output   : {output_path}")
        print(f"  Log      : {args.log_path}")
        print(f"{'='*60}\n")

        apply_fn = OPERATOR_FUNCS[operator]
        print(f"\nApplying operator: {operator} ...")
        try:
            result = apply_fn(samples=samples, **kwargs)
        except Exception as exc:
            traceback.print_exc()
            failures[operator] = str(exc)
            continue

        print(f"\nAfter governance: {len(result)} samples")
        write_jsonl(result, output_path)
        print(f"\nDone. Output saved to: {output_path}")

    if failures:
        print("\nFailures:")
        for operator, err in failures.items():
            print(f"  {operator}: {err}")
        sys.exit(1)

    print("\nAll requested governance outputs are ready.")
    if log_fh is not None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_fh.close()


if __name__ == "__main__":
    main()
