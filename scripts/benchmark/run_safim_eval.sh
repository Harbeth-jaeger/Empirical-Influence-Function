#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-Coder-7B-Instruct}"
RUN_NAME="${RUN_NAME:-base_qwen_7b}"

TEST_DIR="${TEST_DIR:-data/benchmark/test_data}"
PREPARED_DIR="${PREPARED_DIR:-data/benchmark/test_data/safim_prepared}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-outputs/benchmark/eval_results/${RUN_NAME}/safim_predictions.jsonl}"
SAMPLES_DIR="${SAMPLES_DIR:-data/benchmark/test_data/safim_${RUN_NAME}_official_samples}"
RESULT_DIR="${RESULT_DIR:-outputs/benchmark/eval_results/${RUN_NAME}/safim_exact}"
OFFICIAL_RESULT_DIR="${OFFICIAL_RESULT_DIR:-outputs/benchmark/eval_results/${RUN_NAME}/safim_official}"

LANGUAGES="${LANGUAGES:-}"
TASK_TYPE="${TASK_TYPE:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
K_VALUE="${K_VALUE:-1}"
PREPARE="${PREPARE:-auto}"
RUN_OFFICIAL="${RUN_OFFICIAL:-0}"

INFER_REQUESTS="${PREPARED_DIR}/safim_infer_requests.jsonl"

mkdir -p "$(dirname "${PREDICTIONS_PATH}")" "${SAMPLES_DIR}" "${RESULT_DIR}" runs

echo "[safim] model=${MODEL_PATH}"
echo "[safim] run_name=${RUN_NAME}"
echo "[safim] test_dir=${TEST_DIR}"
echo "[safim] prepared=${PREPARED_DIR}"
echo "[safim] predictions=${PREDICTIONS_PATH}"
echo "[safim] samples=${SAMPLES_DIR}"
echo "[safim] exact_results=${RESULT_DIR}"
echo "[safim] languages=${LANGUAGES:-all}"
echo "[safim] task_type=${TASK_TYPE:-all}"

prepare_args=()
postprocess_args=()
eval_args=()
if [[ -n "${LANGUAGES}" ]]; then
  prepare_args+=(--languages "${LANGUAGES}")
  postprocess_args+=(--languages "${LANGUAGES}")
  eval_args+=(--languages "${LANGUAGES}")
fi
if [[ -n "${TASK_TYPE}" ]]; then
  prepare_args+=(--task-type "${TASK_TYPE}")
  postprocess_args+=(--task-type "${TASK_TYPE}")
  eval_args+=(--task-type "${TASK_TYPE}")
fi

if [[ "${PREPARE}" == "1" || "${PREPARE}" == "true" || ( "${PREPARE}" == "auto" && ! -s "${INFER_REQUESTS}" ) ]]; then
  echo "[safim] prepare"
  python scripts/benchmark/evaluate_safim.py prepare \
    --test-dir "${TEST_DIR}" \
    --out-dir "${PREPARED_DIR}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    "${prepare_args[@]}"
else
  echo "[safim] prepare skipped: ${INFER_REQUESTS}"
fi

echo "[safim] generate"
python scripts/benchmark/generate_official_fim_predictions.py \
  --input-path "${INFER_REQUESTS}" \
  --output-path "${PREDICTIONS_PATH}" \
  --model-name-or-path "${MODEL_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --num-return-sequences "${NUM_RETURN_SEQUENCES}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --dtype "${DTYPE}" \
  --device-map "${DEVICE_MAP}"

echo "[safim] postprocess"
python scripts/benchmark/evaluate_safim.py postprocess \
  --test-dir "${TEST_DIR}" \
  --predictions-path "${PREDICTIONS_PATH}" \
  --out-dir "${SAMPLES_DIR}" \
  "${postprocess_args[@]}"

echo "[safim] evaluate exact"
python scripts/benchmark/evaluate_safim.py eval-exact \
  --test-dir "${TEST_DIR}" \
  --predictions-path "${PREDICTIONS_PATH}" \
  --out-dir "${RESULT_DIR}" \
  --k "${K_VALUE}" \
  "${eval_args[@]}"

if [[ "${RUN_OFFICIAL}" == "1" || "${RUN_OFFICIAL}" == "true" ]]; then
  echo "[safim] evaluate official"
  python scripts/benchmark/evaluate_safim.py run-official \
    --samples-dir "${SAMPLES_DIR}" \
    --out-dir "${OFFICIAL_RESULT_DIR}"
  echo "[safim] official done: ${OFFICIAL_RESULT_DIR}/safim_official_eval_outputs.json"
fi

echo "[safim] done: ${RESULT_DIR}/safim_exact_eval_summary.json"
