# Benchmark 说明文档

本文档记录当前 benchmark 实验的文件作用、pipeline、常用运行命令和结果文件约定。

## 1. 目录与文件作用

### 1.1 数据文件

统一训练数据：

```text
data/benchmarks/sft_data/rendered_chatml_fim_train.jsonl
```

统一评测数据：

```text
data/benchmarks/eval_data/rendered_chatml_fim_eval.jsonl
```

每条样本统一成 ChatML + FIM 格式，核心字段包括：

```text
uid
source_dataset
split
language
task_type
raw_id
messages
fim_prompt
fim_completion
metadata
judge_payload
```

模型训练使用 `messages` 字段。评测时从 `fim_prompt` 中解析 prefix/suffix，并用：

```text
full_code = PREFIX + predicted_completion + SUFFIX
```

还原待 judge 的完整代码。

### 1.2 代码文件

Benchmark 主脚本位于：

```text
scripts/benchmark/
```

当前主要文件职责如下：

| 文件 | 作用 |
| --- | --- |
| `build_chatml_fim_unified.py` | 构建统一 ChatML + FIM train/eval 数据。 |
| `sft_data_convert.py` | 数据治理入口：读取统一 train JSONL，调用指定治理算子，输出 SFT 可读 JSON。 |
| `sft_data_utils.py` | `sft_data_convert.py` 的公共工具：JSONL 读写、logging、tokenizer、binarize、字段桥接。 |
| `apply_governance_operator.py` | 治理算子统一 dispatcher，并保留 GraphSignal hard-mask 算子。 |
| `governance_common.py` | 各治理算子共用的小工具。 |
| `graphsignal_annotations.py` | GraphSignal 标注、teacher 补边、annotation 映射逻辑。 |
| `token_cleaning_operator.py` | TokenCleaning token-level filtering。 |
| `xtf_operator.py` | XTF token-level filtering。 |
| `llm_cleaning_operator.py` | LLM-CleanCode 风格 cleaning。 |
| `clear_operator.py` | CLEAR 风格 filtering/correction。 |
| `ibft_loss.py` | IB-FT baseline 的 variational bottleneck 与辅助 loss。 |
| `train_ibft.py` | IB-FT baseline 训练入口，复用 benchmark SFT 数据和超参数，只替换 loss。 |
| `benchmark_eval.py` | 评测入口：加载模型、生成候选、调用 judge、写出 shard 结果。 |
| `eval_generation.py` | generation 相关逻辑。 |
| `eval_judges.py` | HumanEval / SAFIM / McEval judge 和代码包装逻辑。 |
| `eval_reporting.py` | pass@k 聚合与 Markdown 表格输出。 |
| `merge_benchmark_eval.py` | 合并多个 eval shard，输出 merged JSON/CSV/MD。 |
| `oracle_eval.py` | 使用 gold completion 诊断 FIM 拼接与 judge ceiling。 |

训练主入口仍然是：

```text
src/sft/train.py
```

### 1.3 输出与日志约定

正式结果放在：

```text
outputs/benchmark/
```

运行日志放在：

```text
runs/benchmark/
```

评测结果通常按 baseline 分目录：

```text
outputs/benchmark/eval_results/<baseline_dir>/
```

每个 baseline 合并后应至少包含：

```text
<Baseline>_benchmark_eval_merged.json
<Baseline>_benchmark_overall_merged.csv
<Baseline>_benchmark_by_dataset_language_merged.csv
<Baseline>_benchmark_tables_merged.md
```

当前已使用过的 baseline 目录包括：

```text
outputs/benchmark/eval_results/ours_graphsignal
outputs/benchmark/eval_results/tokencleaning
outputs/benchmark/eval_results/xtf
outputs/benchmark/eval_results/clear
outputs/benchmark/eval_results/llm_cleaning
```

注意：`outputs/benchmark/eval_results/xtf/Xtf_*` 是较早旧结果；当前对比优先使用大写 `XTF_*` 文件。

## 2. Benchmark Pipeline

完整流程是：

1. 构建统一 train/eval 数据。
2. 对 train 数据应用治理方法，输出治理后的 SFT JSON。
3. 使用同一个 Qwen base model 做混合多语言 SFT。
4. 使用统一 eval 数据评测，按 shard 并行生成和 judge。
5. 合并 shards，输出 pass@1 / pass@10。
6. 分 HumanEval、SAFIM、McEval 汇总表格。

当前 base model：

```text
Qwen/Qwen2.5-Coder-1.5B-Instruct
```

当前统一评测设置：

```text
num_samples = 10
temperature = 0.2
top_p = 0.95
infer_batch_size = 4
sample_infer_batch_size = 1
judge_workers = 8
judge_timeout_sec = 10
num_shards = 8
```

## 3. 环境

进入项目并激活环境：

```bash
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"
```

常用离线环境变量：

```bash
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 4. 数据治理命令

### 4.1 XTF

```bash
mkdir -p runs/benchmark/curation_data

nohup python scripts/benchmark/sft_data_convert.py \
  --operator xtf \
  --base_model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --keep_ratio 0.6 \
  --device_map auto \
  --attn_implementation eager \
  --log_path runs/benchmark/curation_data/xtf.log \
  > runs/benchmark/curation_data/xtf.outer.log 2>&1 &
```

默认输出：

```text
data/benchmarks/sft_data/xtf_train.json
```

### 4.2 TokenCleaning

TokenCleaning 需要 base model 和 warm-up/ref model。

```bash
mkdir -p runs/benchmark/curation_data

nohup python scripts/benchmark/sft_data_convert.py \
  --operator token_cleaning \
  --base_model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --ref_model_path outputs/benchmark/sft_qwen_tokencleaning_ref \
  --keep_ratio 0.6 \
  --device_map auto \
  --attn_implementation sdpa \
  --log_path runs/benchmark/curation_data/token_cleaning.log \
  > runs/benchmark/curation_data/token_cleaning.outer.log 2>&1 &
```

默认输出：

```text
data/benchmarks/sft_data/token_cleaning_train.json
```

### 4.3 CLEAR

```bash
mkdir -p runs/benchmark/curation_data

nohup python scripts/benchmark/sft_data_convert.py \
  --operator clear \
  --api_base_url https://api.deepseek.com \
  --api_model_name deepseek-v4 \
  --clear_stage filter_correct \
  --clear_num_workers 8 \
  --log_path runs/benchmark/curation_data/clear.log \
  > runs/benchmark/curation_data/clear.outer.log 2>&1 &
```

默认输出：

```text
data/benchmarks/sft_data/clear_train.json
```

### 4.4 LLM-CleanCode

```bash
mkdir -p runs/benchmark/curation_data

nohup python scripts/benchmark/sft_data_convert.py \
  --operator llm_code_cleaning \
  --api_base_url https://api.deepseek.com \
  --api_model_name deepseek-v4 \
  --llm_cleaning_num_workers 8 \
  --log_path runs/benchmark/curation_data/llm_cleancode.log \
  > runs/benchmark/curation_data/llm_cleancode.outer.log 2>&1 &
```

默认输出：

```text
data/benchmarks/sft_data/llm_code_cleaning_train.json
```

### 4.5 GraphSignal hard-mask 旧路线

当前 `--operator graph_signal` 是旧路线：用标注边计算 token importance，并把低分 target label 置为 `-100`。

这不是后续希望采用的 `CE + token-correlation attention loss` 路线。若做新方法实验，应优先使用 `src/curation/` 和 `scripts/curation/` 中的 attention loss 训练代码，或继续完善该路线。

旧路线命令：

```bash
mkdir -p runs/benchmark/curation_data

nohup python scripts/benchmark/sft_data_convert.py \
  --operator graph_signal \
  --keep_ratio 0.6 \
  --gs_mode hard_mask \
  --api_base_url https://api.deepseek.com \
  --api_model_name deepseek-v4 \
  --graph_teacher_workers 8 \
  --log_path runs/benchmark/curation_data/ours_graphsignal.log \
  > runs/benchmark/curation_data/ours_graphsignal.outer.log 2>&1 &
```

默认输出：

```text
data/benchmarks/sft_data/graph_signal_train.json
```

### 4.6 IB-FT 数据准备

IB-FT 是 loss-side baseline，不做 sample/token curation。它使用原始统一训练数据：

```text
data/benchmarks/sft_data/rendered_chatml_fim_train.jsonl
```

先将其直接 binarize 成 IB-FT 专用训练文件：

```bash
mkdir -p runs/benchmark/curation_data

nohup python scripts/benchmark/sft_data_convert.py \
  --operator none \
  --input_path data/benchmarks/sft_data/rendered_chatml_fim_train.jsonl \
  --output_path data/benchmarks/sft_data/ibft_train.json \
  --log_path runs/benchmark/curation_data/ibft_data.log \
  > runs/benchmark/curation_data/ibft_data.outer.log 2>&1 &
```

输出：

```text
data/benchmarks/sft_data/ibft_train.json
```

## 5. SFT 命令模板

为了公平复用当前 baseline 设置，除非特别说明，SFT 超参数应与 XTF/CLEAR/TokenCleaning/LLM-CleanCode 保持一致。

需要替换：

```text
<RUN_DIR>
<OUTPUT_DIR>
<DATA_PATH>
<MASTER_PORT>
```

命令：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p <RUN_DIR> <OUTPUT_DIR>

nohup torchrun \
  --nproc_per_node 8 \
  --master_port <MASTER_PORT> \
  --module src.sft.train \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --data_path <DATA_PATH> \
  --output_dir <OUTPUT_DIR> \
  --model_max_length 4096 \
  --truncate_source True \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --logging_strategy steps \
  --logging_steps 1 \
  --dataloader_num_workers 8 \
  --deepspeed src/sft/configs/zero2.json \
  --bf16 True \
  --tf32 True \
  --use_flash_attention False \
  --use_peft True \
  --peft_config_path src/sft/configs/lora \
  --report_to none \
  > <RUN_DIR>/train.log 2>&1 &
```

常用路径示例：

| Baseline | DATA_PATH | RUN_DIR | OUTPUT_DIR |
| --- | --- | --- | --- |
| XTF | `data/benchmarks/sft_data/xtf_train.json` | `runs/benchmark/sft_qwen_xtf` | `outputs/benchmark/sft_qwen_xtf` |
| TokenCleaning | `data/benchmarks/sft_data/token_cleaning_train.json` | `runs/benchmark/sft_qwen_tokencleaning` | `outputs/benchmark/sft_qwen_tokencleaning` |
| CLEAR | `data/benchmarks/sft_data/clear_train.json` | `runs/benchmark/sft_qwen_clear` | `outputs/benchmark/sft_qwen_clear` |
| LLM-CleanCode | `data/benchmarks/sft_data/llm_code_cleaning_train.json` | `runs/benchmark/sft_qwen_llm_cleancode` | `outputs/benchmark/sft_qwen_llm_cleancode` |
| Ours GraphSignal | `data/benchmarks/sft_data/ours_graphsignal_train.json` | `runs/benchmark/sft_qwen_ours_graphsignal` | `outputs/benchmark/sft_qwen_ours_graphsignal` |
| IB-FT | `data/benchmarks/sft_data/ibft_train.json` | `runs/benchmark/sft_qwen_ibft` | `outputs/benchmark/sft_qwen_ibft` |

IB-FT 使用单独入口：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs/benchmark/sft_qwen_ibft outputs/benchmark/sft_qwen_ibft

nohup torchrun \
  --nproc_per_node 8 \
  --master_port 6110 \
  --module scripts.benchmark.train_ibft \
  --model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --data_path data/benchmarks/sft_data/ibft_train.json \
  --output_dir outputs/benchmark/sft_qwen_ibft \
  --model_max_length 4096 \
  --truncate_source True \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --logging_strategy steps \
  --logging_steps 1 \
  --dataloader_num_workers 8 \
  --deepspeed src/sft/configs/zero2.json \
  --bf16 True \
  --tf32 True \
  --use_flash_attention False \
  --use_peft True \
  --peft_config_path src/sft/configs/lora \
  --ib_layer 20 \
  --ib_z_dim 256 \
  --ib_alpha 0.01 \
  --ib_beta 1.0 \
  --ib_max_tokens_per_batch 2048 \
  --report_to none \
  > runs/benchmark/sft_qwen_ibft/train.log 2>&1 &
```

## 6. 评测命令模板

需要替换：

```text
<MODEL_PATH>
<RUN_DIR>
<EVAL_OUTPUT_DIR>
<BASELINE_NAME>
```

8 卡 shard 评测：

```bash
mkdir -p <RUN_DIR>/shards <EVAL_OUTPUT_DIR>/shards

for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python scripts/benchmark/benchmark_eval.py \
    --model_path <MODEL_PATH> \
    --eval_path data/benchmarks/eval_data/rendered_chatml_fim_eval.jsonl \
    --output_dir <EVAL_OUTPUT_DIR>/shards/shard_$i \
    --baseline_name "<BASELINE_NAME>" \
    --num_samples 10 \
    --temperature 0.2 \
    --top_p 0.95 \
    --infer_batch_size 4 \
    --sample_infer_batch_size 1 \
    --judge_workers 8 \
    --judge_timeout_sec 10 \
    --num_shards 8 \
    --shard_index $i \
    > <RUN_DIR>/shards/eval_shard_$i.log 2>&1 &
done
```

合并结果：

```bash
nohup python scripts/benchmark/merge_benchmark_eval.py \
  --input_glob '<EVAL_OUTPUT_DIR>/shards/shard_*/<BASELINE_NAME>_benchmark_eval.json' \
  --output_dir <EVAL_OUTPUT_DIR> \
  --baseline_name "<BASELINE_NAME>" \
  > <RUN_DIR>/merge_eval.log 2>&1 &
```

示例：Ours GraphSignal

```bash
mkdir -p runs/benchmark/sft_qwen_ours_graphsignal/shards outputs/benchmark/eval_results/ours_graphsignal/shards

for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python scripts/benchmark/benchmark_eval.py \
    --model_path outputs/benchmark/sft_qwen_ours_graphsignal \
    --eval_path data/benchmarks/eval_data/rendered_chatml_fim_eval.jsonl \
    --output_dir outputs/benchmark/eval_results/ours_graphsignal/shards/shard_$i \
    --baseline_name "Ours Graphsignal" \
    --num_samples 10 \
    --temperature 0.2 \
    --top_p 0.95 \
    --infer_batch_size 4 \
    --sample_infer_batch_size 1 \
    --judge_workers 8 \
    --judge_timeout_sec 10 \
    --num_shards 8 \
    --shard_index $i \
    > runs/benchmark/sft_qwen_ours_graphsignal/shards/eval_shard_$i.log 2>&1 &
done

nohup python scripts/benchmark/merge_benchmark_eval.py \
  --input_glob 'outputs/benchmark/eval_results/ours_graphsignal/shards/shard_*/Ours Graphsignal_benchmark_eval.json' \
  --output_dir outputs/benchmark/eval_results/ours_graphsignal \
  --baseline_name "Ours Graphsignal" \
  > runs/benchmark/sft_qwen_ours_graphsignal/merge_eval.log 2>&1 &
```

## 7. 结果表读取

合并后的 Markdown 表格最适合直接看：

```text
outputs/benchmark/eval_results/<baseline_dir>/<Baseline>_benchmark_tables_merged.md
```

若要填论文/汇报表：

```text
outputs/benchmark/eval_results/<baseline_dir>/<Baseline>_benchmark_by_dataset_language_merged.csv
```

其中：

- HumanEval：读取 `source_dataset=humaneval, language=python`。
- SAFIM：读取 `source_dataset=safim` 下 Python / Java / C++ / C#。
- McEval：读取 `source_dataset=mceval` 下 C / C++ / C# / Go / Java / Python。

## 8. 结果汇总

本节汇总当前已经跑完的 benchmark 结果。`Ours Graphsignal` 暂时留空：旧 hard-mask / exploratory attention-loss 结果不纳入最终主表，等待新方法版本确认后再补。

结果来源：

- `Base Qwen`: `outputs/benchmark/eval_results/base_qwen/Base Qwen_benchmark_by_dataset_language_merged.csv`
- `TokenCleaning`: `outputs/benchmark/eval_results/tokencleaning/TokenCleaning_benchmark_by_dataset_language_merged.csv`
- `XTF`: `outputs/benchmark/eval_results/xtf/XTF_benchmark_by_dataset_language_merged.csv`
- `LLM-CleanCode`: `outputs/benchmark/eval_results/llm_cleaning/LLM-CleanCode_benchmark_by_dataset_language_merged.csv`
- `CLEAR`: `outputs/benchmark/eval_results/clear/CLEAR_benchmark_by_dataset_language_merged.csv`
- `IB-FT`: `outputs/benchmark/eval_results/ibft/IB-FT_benchmark_by_dataset_language_merged.csv`
- `Gold Oracle`: `outputs/benchmark/eval_results/Gold Oracle_oracle_tables.md`

### 8.1 Overall

| **Baseline** | **Pass@1** | **Pass@10** | **N** |
| --- | ---: | ---: | ---: |
| **Ours Graphsignal** |  |  | 15107 |
| **Base Qwen** | 0.0306 | 0.0757 | 15107 |
| **TokenCleaning** | 0.1642 | 0.2282 | 15107 |
| **XTF** | 0.1697 | 0.2470 | 15107 |
| **LLM-CleanCode** | 0.0351 | 0.0911 | 15107 |
| **CLEAR** | 0.2239 | 0.3018 | 15107 |
| **IB-FT** | 0.1593 | 0.2263 | 15107 |

### 8.2 Benchmark 1: HumanEval

| **Competitors** | **Pass@1** | **Pass@10** |
| --- | ---: | ---: |
| **Ours Graphsignal** |  |  |
| **Base Qwen** | 0.0490 | 0.1066 |
| **TokenCleaning** | 0.2277 | 0.2841 |
| **XTF** | 0.2377 | 0.3008 |
| **LLM-CleanCode** | 0.0325 | 0.0824 |
| **CLEAR** | 0.2461 | 0.3099 |
| **IB-FT** | 0.2275 | 0.2877 |

### 8.3 Benchmark 2: SAFIM

| **Baseline** | **Language** | **Pass@1** | **Pass@10** |
| --- | --- | ---: | ---: |
| **Ours Graphsignal** | Python |  |  |
|  | Java |  |  |
|  | C++ |  |  |
|  | C# |  |  |
| **Base Qwen** | Python | 0.1031 | 0.2075 |
|  | Java | 0.0056 | 0.0319 |
|  | C++ | 0.0123 | 0.0435 |
|  | C# | 0.0000 | 0.0095 |
| **XTF** | Python | 0.1441 | 0.2273 |
|  | Java | 0.1702 | 0.2820 |
|  | C++ | 0.0813 | 0.1562 |
|  | C# | 0.0699 | 0.1285 |
| **CLEAR** | Python | 0.2186 | 0.3267 |
|  | Java | 0.2525 | 0.3505 |
|  | C++ | 0.1562 | 0.2285 |
|  | C# | 0.1758 | 0.2798 |
| **LLM-CleanCode** | Python | 0.0758 | 0.1652 |
|  | Java | 0.0254 | 0.0867 |
|  | C++ | 0.0276 | 0.0753 |
|  | C# | 0.0095 | 0.0454 |
| **TokenCleaning** | Python | 0.1478 | 0.2037 |
|  | Java | 0.1444 | 0.2263 |
|  | C++ | 0.0835 | 0.1429 |
|  | C# | 0.0851 | 0.1493 |
| **IB-FT** | Python | 0.1354 | 0.2099 |
|  | Java | 0.1404 | 0.2388 |
|  | C++ | 0.0815 | 0.1371 |
|  | C# | 0.0851 | 0.1210 |

### 8.4 Benchmark 3: McEval

| **Baseline** | **Language** | **Pass@1** | **Pass@10** |
| --- | --- | ---: | ---: |
| **Ours Graphsignal** | C |  |  |
|  | C++ |  |  |
|  | C# |  |  |
|  | Go |  |  |
|  | Java |  |  |
|  | Python |  |  |
| **Base Qwen** | C | 0.0617 | 0.1605 |
|  | C++ | 0.0366 | 0.1220 |
|  | C# | 0.0361 | 0.1205 |
|  | Go | 0.0125 | 0.1000 |
|  | Java | 0.0000 | 0.0625 |
|  | Python | 0.0787 | 0.1124 |
| **TokenCleaning** | C | 0.3333 | 0.4815 |
|  | C++ | 0.4024 | 0.5122 |
|  | C# | 0.5422 | 0.5783 |
|  | Go | 0.2750 | 0.3750 |
|  | Java | 0.5729 | 0.6771 |
|  | Python | 0.4157 | 0.6404 |
| **XTF** | C | 0.3951 | 0.4198 |
|  | C++ | 0.3293 | 0.4390 |
|  | C# | 0.4096 | 0.5060 |
|  | Go | 0.2875 | 0.3750 |
|  | Java | 0.5521 | 0.6458 |
|  | Python | 0.3820 | 0.5843 |
| **CLEAR** | C | 0.4938 | 0.6173 |
|  | C++ | 0.4634 | 0.6341 |
|  | C# | 0.5422 | 0.6265 |
|  | Go | 0.4250 | 0.5250 |
|  | Java | 0.7083 | 0.8125 |
|  | Python | 0.6180 | 0.7640 |
| **LLM-CleanCode** | C | 0.1235 | 0.3086 |
|  | C++ | 0.0854 | 0.2439 |
|  | C# | 0.2048 | 0.3855 |
|  | Go | 0.0250 | 0.1375 |
|  | Java | 0.3333 | 0.4896 |
|  | Python | 0.0899 | 0.1910 |
| **IB-FT** | C | 0.3210 | 0.3704 |
|  | C++ | 0.2195 | 0.3902 |
|  | C# | 0.3976 | 0.4940 |
|  | Go | 0.2125 | 0.2875 |
|  | Java | 0.5000 | 0.6146 |
|  | Python | 0.3820 | 0.6180 |

### 8.5 Gold Oracle 诊断表

Gold Oracle 使用每条样本的 `fim_completion` 作为预测结果，主要用于诊断 FIM 拼接和 judge wrapper 的上限。该表不与模型 baseline 放在一起比较。

| **Dataset** | **Language** | **Pass@1** | **Pass@10** | **N** | **N_total** |
| --- | --- | ---: | ---: | ---: | ---: |
| humaneval | python | 1.0000 | 1.0000 | 5815 | 5815 |
| mceval | c | 1.0000 | 1.0000 | 81 | 81 |
| mceval | cpp | 1.0000 | 1.0000 | 82 | 82 |
| mceval | csharp | 0.9639 | 0.9639 | 83 | 83 |
| mceval | go | 0.8750 | 0.8750 | 80 | 80 |
| mceval | java | 0.8958 | 0.8958 | 96 | 96 |
| mceval | python | 0.9438 | 0.9438 | 89 | 89 |
| safim | cpp | 0.9815 | 0.9815 | 4968 | 4968 |
| safim | csharp | 1.0000 | 1.0000 | 529 | 529 |
| safim | java | 0.9984 | 0.9984 | 2479 | 2479 |
| safim | python | 0.9988 | 0.9988 | 805 | 805 |

说明：McEval Go / Java 的 oracle ceiling 明显低于 1.0，因此这些语言的绝对分数需要谨慎解读；相对比较仍然使用同一份 eval 文件和同一套 judge。
