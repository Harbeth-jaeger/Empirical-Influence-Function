#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RAW_DATA="${HUAWEI_RAW_DATA:-${RAW_DATA:-data/huawei_data/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl}}"
PROCESSED_DIR="${HUAWEI_PROCESSED_DIR:-data/huawei_data/processed}"
RAW_BASENAME="$(basename "$RAW_DATA")"
RAW_STEM="${RAW_BASENAME%.*}"
CHATML_DATA="${HUAWEI_CHATML_DATA:-$PROCESSED_DIR/${RAW_STEM}_chatml.jsonl}"
CANONICAL_DATA="${HUAWEI_CANONICAL_DATA:-$PROCESSED_DIR/${RAW_STEM}_canonical.jsonl}"
REPORT_DATA="${HUAWEI_PREPARE_REPORT:-$PROCESSED_DIR/${RAW_STEM}_prepare_report.json}"
PREPARE_MAX_ROWS="${PREPARE_MAX_ROWS:-0}"
PREPARE_MAX_ACCEPTED_ROWS="${PREPARE_MAX_ACCEPTED_ROWS:-0}"
STRIP_CJK_COMMENTS="${STRIP_CJK_COMMENTS:-1}"
MAX_TARGET_NONEMPTY_LINES="${MAX_TARGET_NONEMPTY_LINES:-10}"
MAX_TARGET_ROUGH_TOKENS="${MAX_TARGET_ROUGH_TOKENS:-192}"
MAX_TARGET_CHARS="${MAX_TARGET_CHARS:-1024}"
VALIDATE_LIMIT="${VALIDATE_LIMIT:-100}"
CHECK_ROWS="${CHECK_ROWS:-50}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
ENABLE_VISUALIZE="${ENABLE_VISUALIZE:-1}"
ANNOTATION_MODE="${ANNOTATION_MODE:-agent}"
CHECK_RUN_NAME="${CHECK_RUN_NAME:-${RAW_STEM}_check_huawei_${ANNOTATION_MODE}}"
FULL_RUN_NAME="${FULL_RUN_NAME:-${RAW_STEM}_full_huawei_${ANNOTATION_MODE}}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-Coder-7B-Instruct}"
OUT_DIR="${OUT_DIR:-outputs/go_singleline_fim_exp/huawei_annotation}"
RUN_DIR="${RUN_DIR:-runs/go_singleline_fim_exp/huawei_annotation}"
VIS_OUT_DIR="${VIS_OUT_DIR:-outputs/huawei_deploy}"

usage() {
  cat <<'EOF'
Usage: bash scripts/huawei_deploy/annotate.sh [prepare|validate|check|full|visualize|all]

Modes:
  prepare    Convert Huawei raw prompt/response/task_id JSONL to project ChatML/FIM JSONL.
  validate   Validate converted ChatML/FIM rows.
  check      prepare -> validate -> annotate CHECK_ROWS rows -> visualize by default.
  full       prepare -> validate -> annotate all rows.
  visualize  Build rich-sample CSV and HTML viewer for CHECK_RUN_NAME output.
  all        prepare -> validate -> check -> optional visualize -> full annotation.

Important env vars:
  HUAWEI_RAW_DATA       Raw Huawei JSONL path. Default: data/huawei_data/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl
  HUAWEI_CHATML_DATA    Converted ChatML output path.
  HUAWEI_CANONICAL_DATA Converted canonical output path.
  FORCE_PREPARE=1      Rebuild converted data even if outputs already exist.
  PREPARE_MAX_ROWS=N   Read only N raw rows; 0 means all rows.
  PREPARE_MAX_ACCEPTED_ROWS=N Stop after N accepted rows; 0 means no accepted-row limit.
  STRIP_CJK_COMMENTS=1 Remove Go comments containing CJK characters during prepare; default: 1.
  MAX_TARGET_NONEMPTY_LINES=N  Reject long targets by non-empty line count; default: 10.
  MAX_TARGET_ROUGH_TOKENS=N    Reject long targets by rough token count; default: 192.
  MAX_TARGET_CHARS=N           Reject long targets by char count; default: 1024.
  CHECK_ROWS=50        Number of rows for check mode.
  ENABLE_VISUALIZE=1   Build viewer after check/all; default: 1.
  VIS_OUT_DIR=PATH     Visualization CSV/HTML output dir. Default: outputs/huawei_deploy

API env vars are required for check/full/all:
  OPENAI_API_KEY, OPENAI_BASE_URL, ANNOTATE_MODEL
  If REQUIRE_HUAWEI_GATEWAY=1, also require HW_APPKEY, HW_OPERATOR.
  For local non-Huawei OpenAI-compatible tests, set REQUIRE_HUAWEI_GATEWAY=0.
EOF
}

log() { printf '[huawei-annotate] %s\n' "$*"; }

prepare() {
  if [[ ! -f "$RAW_DATA" ]]; then
    echo "Raw data not found: $RAW_DATA" >&2
    exit 1
  fi
  if [[ "$FORCE_PREPARE" != "1" && -s "$CHATML_DATA" && -s "$CANONICAL_DATA" ]]; then
    log "prepare skipped: $CHATML_DATA and $CANONICAL_DATA already exist"
    return
  fi
  mkdir -p "$PROCESSED_DIR"
  local max_args=()
  if [[ "$PREPARE_MAX_ROWS" != "0" ]]; then
    max_args+=(--max-rows "$PREPARE_MAX_ROWS")
  fi
  if [[ "$PREPARE_MAX_ACCEPTED_ROWS" != "0" ]]; then
    max_args+=(--max-accepted-rows "$PREPARE_MAX_ACCEPTED_ROWS")
  fi
  if [[ "$STRIP_CJK_COMMENTS" == "1" ]]; then
    max_args+=(--strip-cjk-comments)
  fi
  if [[ "$MAX_TARGET_NONEMPTY_LINES" != "0" ]]; then
    max_args+=(--max-target-nonempty-lines "$MAX_TARGET_NONEMPTY_LINES")
  fi
  if [[ "$MAX_TARGET_ROUGH_TOKENS" != "0" ]]; then
    max_args+=(--max-target-rough-tokens "$MAX_TARGET_ROUGH_TOKENS")
  fi
  if [[ "$MAX_TARGET_CHARS" != "0" ]]; then
    max_args+=(--max-target-chars "$MAX_TARGET_CHARS")
  fi
  log "prepare raw=$RAW_DATA"
  log "prepare chatml=$CHATML_DATA"
  log "prepare canonical=$CANONICAL_DATA"
  log "prepare cleaning strip_cjk_comments=$STRIP_CJK_COMMENTS max_lines=$MAX_TARGET_NONEMPTY_LINES max_tokens=$MAX_TARGET_ROUGH_TOKENS max_chars=$MAX_TARGET_CHARS max_accepted=$PREPARE_MAX_ACCEPTED_ROWS"
  python scripts/huawei_deploy/build_huawei_fim_chatml.py \
    --input-path "$RAW_DATA" \
    --chatml-output "$CHATML_DATA" \
    --canonical-output "$CANONICAL_DATA" \
    --report-output "$REPORT_DATA" \
    "${max_args[@]}"
}

validate() {
  log "validate input=$CHATML_DATA limit=$VALIDATE_LIMIT"
  python scripts/go_singleline_fim_exp/check_chatml_fim_input.py \
    --input_path "$CHATML_DATA" \
    --limit "$VALIDATE_LIMIT"
}

require_api_env() {
  : "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the OpenAI-compatible endpoint.}"
  : "${ANNOTATE_MODEL:?Set ANNOTATE_MODEL to the deployed model name.}"
  local require_huawei="${REQUIRE_HUAWEI_GATEWAY:-1}"
  if [[ "$require_huawei" == "1" ]]; then
    : "${HW_APPKEY:?Set HW_APPKEY to the Huawei X-HW-APPKEY value, or set REQUIRE_HUAWEI_GATEWAY=0 for local non-Huawei tests.}"
    : "${HW_OPERATOR:?Set HW_OPERATOR to your work-id/operator value, or set REQUIRE_HUAWEI_GATEWAY=0 for local non-Huawei tests.}"
  else
    : "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL for local non-Huawei OpenAI-compatible tests.}"
  fi
}

annotate_check() {
  require_api_env
  log "annotate check rows=$CHECK_ROWS run=$CHECK_RUN_NAME"
  TRAIN_DATA="$CHATML_DATA" \
  RUN_NAME="$CHECK_RUN_NAME" \
  MAX_ROWS="$CHECK_ROWS" \
  MODEL_PATH="$MODEL_PATH" \
  OUT_DIR="$OUT_DIR" \
  RUN_DIR="$RUN_DIR" \
  ANNOTATION_MODE="$ANNOTATION_MODE" \
  REQUIRE_HUAWEI_GATEWAY="${REQUIRE_HUAWEI_GATEWAY:-1}" \
  bash scripts/go_singleline_fim_exp/run_huawei_annotation.sh check
}

annotate_full() {
  require_api_env
  log "annotate full run=$FULL_RUN_NAME"
  TRAIN_DATA="$CHATML_DATA" \
  RUN_NAME="$FULL_RUN_NAME" \
  MODEL_PATH="$MODEL_PATH" \
  OUT_DIR="$OUT_DIR" \
  RUN_DIR="$RUN_DIR" \
  ANNOTATION_MODE="$ANNOTATION_MODE" \
  REQUIRE_HUAWEI_GATEWAY="${REQUIRE_HUAWEI_GATEWAY:-1}" \
  bash scripts/go_singleline_fim_exp/run_huawei_annotation.sh full
}

visualize() {
  local run_name="${VIS_RUN_NAME:-$CHECK_RUN_NAME}"
  local compact="$OUT_DIR/${run_name}_compact.jsonl"
  local csv="$VIS_OUT_DIR/${run_name}_rich_samples.csv"
  local html="$VIS_OUT_DIR/${run_name}_viewer.html"
  if [[ ! -s "$compact" ]]; then
    echo "Compact output not found for visualization: $compact" >&2
    exit 1
  fi
  mkdir -p "$VIS_OUT_DIR"
  log "visualize compact=$compact"
  log "visualize csv=$csv"
  log "visualize html=$html"
  python tools/viz_annotation/find_annotation_rich_samples.py \
    --raw_data_path "$CHATML_DATA" \
    --edge_data_path "$compact" \
    --model_path "$MODEL_PATH" \
    --top_n 50 \
    --sort_by total \
    --output_csv "$csv" \
    --local_files_only
  python tools/viz_annotation/build_dynamic_annotation_viewer.py \
    --edge_data_path "$compact" \
    --samples_csv "$csv" \
    --top_n_from_csv 50 \
    --model_path "$MODEL_PATH" \
    --output_path "$html" \
    --local_files_only
  log "viewer=$html"
  log "serve from repo root: python -m http.server 8000"
  log "open: http://<server>:8000/$html"
}

case "$MODE" in
  prepare)
    prepare
    ;;
  validate)
    prepare
    validate
    ;;
  check)
    prepare
    validate
    annotate_check
    if [[ "$ENABLE_VISUALIZE" == "1" ]]; then visualize; fi
    ;;
  full)
    prepare
    validate
    annotate_full
    ;;
  visualize)
    prepare
    validate
    visualize
    ;;
  all)
    prepare
    validate
    annotate_check
    if [[ "$ENABLE_VISUALIZE" == "1" ]]; then visualize; fi
    annotate_full
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
