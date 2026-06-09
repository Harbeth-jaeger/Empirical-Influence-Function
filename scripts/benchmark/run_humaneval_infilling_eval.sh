#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-Coder-7B-Instruct}"
RUN_NAME="${RUN_NAME:-base_qwen_7b}"

TEST_PATH="${TEST_PATH:-data/benchmark/test_data/humaneval_infilling_python.jsonl}"
PREPARED_DIR="${PREPARED_DIR:-data/benchmark/test_data/humaneval_infilling_prepared}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-outputs/benchmark/eval_results/${RUN_NAME}/humaneval_predictions.jsonl}"
SAMPLES_DIR="${SAMPLES_DIR:-data/benchmark/test_data/humaneval_infilling_${RUN_NAME}_official_samples}"
RESULT_DIR="${RESULT_DIR:-outputs/benchmark/eval_results/${RUN_NAME}/humaneval_infilling}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
N_WORKERS="${N_WORKERS:-16}"
TIMEOUT="${TIMEOUT:-3.0}"
K_VALUES="${K_VALUES:-1}"
PREPARE="${PREPARE:-auto}"

INFER_REQUESTS="${PREPARED_DIR}/humaneval_infilling_infer_requests.jsonl"

mkdir -p "$(dirname "${PREDICTIONS_PATH}")" "${SAMPLES_DIR}" "${RESULT_DIR}" runs

echo "[humaneval] model=${MODEL_PATH}"
echo "[humaneval] run_name=${RUN_NAME}"
echo "[humaneval] test=${TEST_PATH}"
echo "[humaneval] prepared=${PREPARED_DIR}"
echo "[humaneval] predictions=${PREDICTIONS_PATH}"
echo "[humaneval] samples=${SAMPLES_DIR}"
echo "[humaneval] results=${RESULT_DIR}"

if [[ "${PREPARE}" == "1" || "${PREPARE}" == "true" || ( "${PREPARE}" == "auto" && ! -s "${INFER_REQUESTS}" ) ]]; then
  echo "[humaneval] prepare"
  python scripts/benchmark/evaluate_humaneval_infilling.py prepare \
    --test-path "${TEST_PATH}" \
    --out-dir "${PREPARED_DIR}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"
else
  echo "[humaneval] prepare skipped: ${INFER_REQUESTS}"
fi

echo "[humaneval] generate"
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

echo "[humaneval] postprocess"
python scripts/benchmark/evaluate_humaneval_infilling.py postprocess \
  --test-path "${TEST_PATH}" \
  --predictions-path "${PREDICTIONS_PATH}" \
  --out-dir "${SAMPLES_DIR}"

echo "[humaneval] evaluate"
# shellcheck disable=SC2086
python scripts/benchmark/evaluate_humaneval_infilling.py eval \
  --samples-dir "${SAMPLES_DIR}" \
  --out-dir "${RESULT_DIR}" \
  --k ${K_VALUES} \
  --n-workers "${N_WORKERS}" \
  --timeout "${TIMEOUT}" \
  --no-reuse-existing

echo "[humaneval] done: ${RESULT_DIR}/humaneval_eval_summary.json"
