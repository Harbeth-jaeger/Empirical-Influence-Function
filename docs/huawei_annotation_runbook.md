# Huawei Annotation Runbook

This branch annotates Go ChatML/FIM data with `src.annotate` and writes compact
training rows containing `input_ids`, `label`, `attention_edges`, `uid`, and
`annotation_meta`.

## 1. Enter repo and install deps

```bash
cd /home/model_project/Empirical-Influence-Function-go-single-7b
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-annotate.txt
```

Use `requirements.txt` only if you also need training/evaluation dependencies.

The tokenizer path must exist locally. By default the scripts use:

```bash
models/Qwen2.5-Coder-7B-Instruct
```

Override it with `MODEL_PATH=/path/to/Qwen2.5-Coder-7B-Instruct` if needed.

## 2. Export data and Huawei API config

Use `TRAIN_DATA`, not `TRAIN_ DATA`.

```bash
export TRAIN_DATA=/home/model_project/Open_CC_SFT_Eval/train/cloud_core_test_25JunJunly_Goonly_length_filter.jsonl

export OPENAI_BASE_URL=https://apigw-cn-southe2.huawei.com/api/v1
export OPENAI_API_KEY="<api_key>"
export ANNOTATE_MODEL="<deployed_model_name>"

export HW_ID=com.huawei.ipd.coretool.coreai
export HW_APPKEY="<x_hw_appkey>"
export HW_APP_ID=com.huawei.ipd.coretool.coreai
export HW_SCENE=test
export HW_OPERATOR="<your_operator_or_work_id>"

export ANNOTATE_HTTP_PROXY_NONE=1
export ANNOTATE_VERIFY_SSL=0
export HW_ENABLE_THINKING=0
export ANNOTATE_TEMPERATURE=0.2
```

The code passes Huawei values as:

- request headers: `X-HW-ID`, `X-HW-APPKEY`
- request body: `appId`, `scene`, `operator`, `chat_template_kwargs.enable_thinking`
- custom httpx transport: `proxy=None`, `verify=False` when the env vars above are set

For unsupported fields, add a raw JSON object:

```bash
export ANNOTATE_EXTRA_BODY_JSON='{"customField":"value"}'
export ANNOTATE_EXTRA_HEADERS_JSON='{"Header-Name":"value"}'
```

## 3. Validate input shape

```bash
python scripts/go_singleline_fim_exp/check_chatml_fim_input.py \
  --input_path "$TRAIN_DATA" \
  --limit 100
```

The annotator expects each row to have `uid`, Go `language`, `target`, and
`messages`, where the user message contains `[MASK]` and the assistant message
equals `target`.

## 4. Run 20-row check

Start conservatively:

```bash
export NUM_WORKERS=2
export ANNOTATION_MODE=agent
export RUN_NAME=cloud_core_go_20_huawei

nohup bash scripts/go_singleline_fim_exp/run_huawei_annotation.sh check \
  > runs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}.log 2>&1 &
```

Progress:

```bash
tail -f runs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}.log
wc -l runs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}.annotation_cache.jsonl
```

Output:

```bash
outputs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}_compact.jsonl
```

## 5. Visual inspect

```bash
python tools/viz_annotation/find_annotation_rich_samples.py \
  --raw_data_path "$TRAIN_DATA" \
  --edge_data_path outputs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}_compact.jsonl \
  --model_path "${MODEL_PATH:-models/Qwen2.5-Coder-7B-Instruct}" \
  --top_n 50 \
  --sort_by total \
  --output_csv outputs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}_rich_samples.csv \
  --local_files_only

python tools/viz_annotation/build_dynamic_annotation_viewer.py \
  --edge_data_path outputs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}_compact.jsonl \
  --samples_csv outputs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}_rich_samples.csv \
  --top_n_from_csv 50 \
  --model_path "${MODEL_PATH:-models/Qwen2.5-Coder-7B-Instruct}" \
  --output_path outputs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}_viewer.html \
  --local_files_only
```

Open a static server from the repo root if you need browser access:

```bash
python -m http.server 8000
```

Then visit:

```text
http://<server>:8000/outputs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}_viewer.html
```

## 6. Run full annotation

After the 20-row check looks good:

```bash
export RUN_NAME=cloud_core_go_full_huawei
export NUM_WORKERS=4
export ANNOTATION_MODE=agent

nohup bash scripts/go_singleline_fim_exp/run_huawei_annotation.sh full \
  > runs/go_singleline_fim_exp/huawei_annotation/${RUN_NAME}.log 2>&1 &
```

If the Huawei endpoint rate-limits, lower workers or increase interval:

```bash
export NUM_WORKERS=1
export ANNOTATE_MIN_REQUEST_INTERVAL=2.0
```

To resume after interruption, run the same full command again with the same
`RUN_NAME` and `CACHE_PATH`. Do not use `--overwrite_cache` for full runs.
