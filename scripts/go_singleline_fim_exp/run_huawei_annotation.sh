#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"

: "${TRAIN_DATA:?Set TRAIN_DATA to the ChatML/FIM jsonl path. Do not use TRAIN_ DATA with a space.}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the OpenAI-compatible endpoint.}"
: "${ANNOTATE_MODEL:?Set ANNOTATE_MODEL to the deployed model name.}"

REQUIRE_HUAWEI_GATEWAY="${REQUIRE_HUAWEI_GATEWAY:-1}"
if [[ "$REQUIRE_HUAWEI_GATEWAY" == "1" ]]; then
  : "${HW_APPKEY:?Set HW_APPKEY to the Huawei X-HW-APPKEY value.}"
  : "${HW_OPERATOR:?Set HW_OPERATOR to your work-id/operator value.}"
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://apigw-cn-southe2.huawei.com/api/v1}"
  export HW_ID="${HW_ID:-com.huawei.ipd.coretool.coreai}"
  export HW_APP_ID="${HW_APP_ID:-$HW_ID}"
  export HW_SCENE="${HW_SCENE:-test}"
  export HW_ENABLE_THINKING="${HW_ENABLE_THINKING:-0}"
  export ANNOTATE_HTTP_PROXY_NONE="${ANNOTATE_HTTP_PROXY_NONE:-1}"
  export ANNOTATE_VERIFY_SSL="${ANNOTATE_VERIFY_SSL:-0}"
else
  : "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL for local non-Huawei OpenAI-compatible tests.}"
  unset HW_ID HW_APPKEY HW_APP_ID HW_SCENE HW_OPERATOR
  unset HUAWEI_HW_ID HUAWEI_HW_APPKEY HUAWEI_APP_ID HUAWEI_SCENE HUAWEI_OPERATOR
  unset ANNOTATE_EXTRA_HEADERS_JSON ANNOTATE_EXTRA_BODY_JSON
  export HW_ENABLE_THINKING="${HW_ENABLE_THINKING:-0}"
  export ANNOTATE_HTTP_PROXY_NONE="${ANNOTATE_HTTP_PROXY_NONE:-0}"
  export ANNOTATE_VERIFY_SSL="${ANNOTATE_VERIFY_SSL:-1}"
fi

export ANNOTATE_TEMPERATURE="${ANNOTATE_TEMPERATURE:-0.2}"
export ANNOTATE_MIN_REQUEST_INTERVAL="${ANNOTATE_MIN_REQUEST_INTERVAL:-1.0}"
export ANNOTATE_MAX_RETRIES="${ANNOTATE_MAX_RETRIES:-8}"
export ANNOTATE_RETRY_BASE_SLEEP="${ANNOTATE_RETRY_BASE_SLEEP:-10}"
export ANNOTATE_MAX_TOKENS="${ANNOTATE_MAX_TOKENS:-2048}"

MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-Coder-7B-Instruct}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-6}"
ANNOTATION_MODE="${ANNOTATION_MODE:-agent}"
FLUSH_EVERY="${FLUSH_EVERY:-20}"

DATA_STEM="$(basename "$TRAIN_DATA")"
DATA_STEM="${DATA_STEM%.*}"
RUN_NAME="${RUN_NAME:-${DATA_STEM}_huawei_${ANNOTATION_MODE}}"

OUT_DIR="${OUT_DIR:-outputs/go_singleline_fim_exp/huawei_annotation}"
RUN_DIR="${RUN_DIR:-runs/go_singleline_fim_exp/huawei_annotation}"
mkdir -p "$OUT_DIR" "$RUN_DIR"

OUTPUT_PATH="${OUTPUT_PATH:-$OUT_DIR/${RUN_NAME}_compact.jsonl}"
CACHE_PATH="${CACHE_PATH:-$RUN_DIR/${RUN_NAME}.annotation_cache.jsonl}"

MAX_ROWS_ARGS=()
OVERWRITE_ARGS=()
if [[ "$MODE" == "check" ]]; then
  MAX_ROWS_ARGS=(--max_rows "${MAX_ROWS:-20}")
  OVERWRITE_ARGS=(--overwrite_cache)
elif [[ "$MODE" != "full" ]]; then
  echo "Usage: $0 [check|full]" >&2
  exit 2
fi

echo "mode=$MODE"
echo "input=$TRAIN_DATA"
echo "output=$OUTPUT_PATH"
echo "cache=$CACHE_PATH"
echo "model_path=$MODEL_PATH"
echo "annotate_model=$ANNOTATE_MODEL"
echo "require_huawei_gateway=$REQUIRE_HUAWEI_GATEWAY"
echo "workers=$NUM_WORKERS"

python scripts/go_singleline_fim_exp/annotate_chatml_with_src_annotate.py \
  --input_path "$TRAIN_DATA" \
  --output_path "$OUTPUT_PATH" \
  --model_name_or_path "$MODEL_PATH" \
  "${MAX_ROWS_ARGS[@]}" \
  --num_workers "$NUM_WORKERS" \
  --annotation_mode "$ANNOTATION_MODE" \
  --max_rounds "$MAX_ROUNDS" \
  --annotation_cache_path "$CACHE_PATH" \
  --flush_every "$FLUSH_EVERY" \
  "${OVERWRITE_ARGS[@]}"
