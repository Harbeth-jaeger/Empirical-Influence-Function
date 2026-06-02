#!/usr/bin/env bash
set -euo pipefail

export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_CACHE="/mnt/nvme0n1/hf_hub"
export HF_HOME="/mnt/nvme0n1/hf_hub"
export WANDB_PROJECT="code-corr-saliency"

# Multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA="./data/mceval/mceval-sft.jsonl" # todo: change to your own dataset
OUTPUT="./checkpoints"

torchrun \
  --nproc_per_node=6 \
  --master_port=29500 \
  src/train/train.py \
  --model_name_or_path "${MODEL}" \
  --data_path        "${DATA}" \
  --output_dir       "${OUTPUT}" \
  --num_train_epochs 30 \
  --per_device_train_batch_size  1 \
  --gradient_accumulation_steps  4 \
  --learning_rate    2e-5 \
  --warmup_ratio     0.03 \
  --lr_scheduler_type cosine \
  --max_grad_norm    1.0 \
  --bf16             true \
  --eval_strategy    steps \
  --eval_steps       50 \
  --save_strategy    steps \
  --save_steps       100 \
  --save_total_limit 1 \
  --logging_steps    10 \
  --dataloader_num_workers 4 \
  --report_to        wandb \
  --run_name         "Qwen2.5-Coder-1.5B-Instruct-saliency" \
  --max_len          8192 \
  --remove_unused_columns False \
  --language         Python \
  --use_flash_attention False \
  --use_peft         True \
  --saliency_lambda  1.0 \
  --saliency_alpha   1.5 \
  --saliency_eps     1e-8