# 华为侧部署说明

## 数据处理

华为侧 raw 数据是 `<PRE>/<SUF>/<MID>` prompt + `response` 的 Go FIM 数据，但它不是我们本地 CodeSearchNet Go pipeline 产出的标准 canonical 数据。现场检查发现，部分 raw 样本存在代码层面的低质量问题，例如标识符被换行或空格打断：`logapi.Critic\nal(...)`、`acslog\n.Info(...)`、`va r i uint`、`Ta\n\nbleName()` 等。这类问题会导致拼回的 Go 代码无法通过语法解析，也会让后续因果边标注变得没有意义。

作为对照，我们用同样的 `gofmt` 语法检查过本地 Go 实验阶段数据：`go_single_train_v2_canonical.jsonl` 10000 条、CodeSearchNet valid 7611 条、CodeSearchNet test 7979 条、MCEval 45 条、HumanEval-X derived 129 条均为 `gofmt_fail=0`。说明本地 Go 数据的源质量和预处理链路是可靠的；华为侧需要在标注前补上专门的数据清洗和质量过滤。

当前 `scripts/huawei_deploy/annotate.sh` 的 prepare 阶段会自动调用 `scripts/huawei_deploy/build_huawei_fim_chatml.py`，处理顺序是：

1. 从 raw prompt 中抽取 `prefix / suffix`，从 `response` 中读取 `target`。
2. 可选去掉包含中文字符的 Go 注释，默认 `STRIP_CJK_COMMENTS=1`。
3. 按 target 长度过滤，默认仍沿用当前标准：`MAX_TARGET_NONEMPTY_LINES=10`、`MAX_TARGET_ROUGH_TOKENS=192`、`MAX_TARGET_CHARS=1024`。
4. 拼回 `full_code = prefix + target + suffix`，并用 `gofmt` 做 Go 语法质量检查；默认 `FILTER_GOFMT_VALID=1`。解析失败的样本不会进入 `data/huawei_data/processed_*`，也不会送去标注。
5. 写出 canonical/chatml/report，并在 report 中记录 rejected 数量和原因，例如 `reject_gofmt_invalid_full_code`。

如果华为机器上 `gofmt` 不在 PATH，需要先确认 Go 环境，或者显式设置：

```bash
export FILTER_GOFMT_VALID=1
export GOFMT_BIN=/path/to/gofmt
```

不建议关闭 `FILTER_GOFMT_VALID` 后跑大规模标注；除非只是为了定位 raw 数据问题。

本文用于在华为绿区机器上部署 Go single-line FIM causality annotation 流程。目标是：进入项目、拉取 Gitee 同名分支、配置华为 OpenAI-compatible API 环境变量，先用 30 条样本检查格式和标注质量，再根据限流情况调整并发，随后跑 5000 条小规模标注。

队友今日已在华为侧做了铺垫：`EIF-huawei-annotation/常用命令.md` 中保存了华为侧需要 export 的变量、API key、模型名等真实值。本文中凡是 `<...>` 占位符，明天优先从该 md 文件中复制替换。

## 部署标注

本节是华为机器现场优先执行的命令。路径、数据、tokenizer/base model、外部标注模型都以这里为准。多份训练 jsonl 用命令行位置参数传给 `scripts/huawei_deploy/annotate.sh`，脚本会按顺序串行处理，并为每份数据按文件名自动派生 prepared/cache/compact 输出名。

注意：真实 `HW_APPKEY / OPENAI_API_KEY / ANNOTATE_MODEL / HW_OPERATOR` 等值来自华为机器本地 `EIF-huawei-annotation/常用命令.md`。如果后续要公开仓库或同步给无权限环境，需要先脱敏。

关于调用模型进行在线推理、辅助标注，华为官方提供了部署在他们机器上的模型，具体调用详见 `scripts/huawei_deploy/vlm.py`。

### 0. 进入仓库和环境

```bash
cd /home/model_project/EIF-huawei-annotation
source .venv/bin/activate
conda activate EIF

export GOROOT="$PWD/.local/go"
export PATH="$GOROOT/bin:$PATH"

mkdir -p data/huawei_data/processed_30_clean
mkdir -p data/huawei_data/processed_full_clean
mkdir -p outputs/huawei_deploy
mkdir -p runs/huawei_deploy
```

### 1. 公共环境变量

```bash
# 数据路径：第 1 份是真实已确认路径，其余 3 份明天在华为机器上替换为实际 jsonl。
export TRAIN_DATA_1=/home/model_project/Open_CC_SFT_Eval/train/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl
export TRAIN_DATA_2=/home/model_project/Open_CC_SFT_Eval/train/cloud_core_test_25.JunJunly_GoGoLLTTrain.jsonl
export TRAIN_DATA_3=/home/model_project/Open_CC_SFT_Eval/train/cloud_core_test_25.JunJunly_GoOnly.jsonl
export TRAIN_DATA_4=/home/model_project/Open_CC_SFT_Eval/train/cloud_core_test_25.JunJunly_JavaOnlyTrain.jsonl

# tokenizer / base model 路径
export MODEL_PATH=/home/model_project/CCCodeGenerationTrain/infer_format/

# 华为 OpenAI-compatible API
export REQUIRE_HUAWEI_GATEWAY=1
export OPENAI_BASE_URL=https://apigw-cn-south02.huawei.com/api/v1
export OPENAI_API_KEY="com.huawei.ipd.coretool.coreai"
export ANNOTATE_MODEL="6d2c5ff6-615d-45a8-9703-2f591d6c2437"

# 华为网关身份信息
export HW_ID=com.huawei.ipd.coretool.coreai
export HW_APPKEY="WxhsDOVQJGVYpkDfQ7C2HA=="
export HW_APP_ID=com.huawei.ipd.coretool.coreai
export HW_SCENE=test
export HW_OPERATOR="h00965148"

# HTTP / SSL / 生成参数
export ANNOTATE_HTTP_PROXY_NONE=1
export ANNOTATE_VERIFY_SSL=0
export HW_ENABLE_THINKING=0
export ANNOTATE_STREAM=1
export ANNOTATE_FALLBACK_ON_CHAT_ERROR=1
export ANNOTATE_TEMPERATURE=0.2
export ANNOTATE_MIN_REQUEST_INTERVAL=1.0
export ANNOTATE_MAX_RETRIES=8
export ANNOTATE_RETRY_BASE_SLEEP=10
export ANNOTATE_MAX_TOKENS=2048

# 标注模式
export ANNOTATION_MODE=agent

# 数据清洗与质量过滤
export STRIP_CJK_COMMENTS=1
export MAX_TARGET_NONEMPTY_LINES=10
export MAX_TARGET_ROUGH_TOKENS=192
export MAX_TARGET_CHARS=1024
export FILTER_GOFMT_VALID=1
export GOFMT_BIN="$GOROOT/bin/gofmt"
```

### 2. 小样本检查

默认先用第 1 份数据跑 30 条，检查转换、过滤、标注和 HTML 可视化是否正常。这样总量就是 30 条，不会因为传入 4 份数据变成每份各 30 条。

```bash
export HUAWEI_PROCESSED_DIR=data/huawei_data/processed_30_clean
export OUT_DIR=data/huawei_data/processed_30_clean
export RUN_DIR=runs/huawei_deploy
export VIS_OUT_DIR=outputs/huawei_deploy

export CHECK_ROWS=30
export VALIDATE_LIMIT=100
export PREPARE_MAX_ACCEPTED_ROWS=30
export FORCE_PREPARE=1
export ENABLE_VISUALIZE=1
```

先试 12 workers：

```bash
export NUM_WORKERS=12
export ANNOTATE_MIN_REQUEST_INTERVAL=1.0

nohup bash scripts/huawei_deploy/annotate.sh check "$TRAIN_DATA_1" \
  > runs/huawei_deploy/huawei_go_30_clean_check_w12.pipeline.log 2>&1 &
```

查看日志：

```bash
tail -f runs/huawei_deploy/huawei_go_30_clean_check_w12.pipeline.log
```

如果 12 workers 触发 429，改成 8 重新跑：

```bash
export NUM_WORKERS=8
export ANNOTATE_MIN_REQUEST_INTERVAL=1.0

nohup bash scripts/huawei_deploy/annotate.sh check "$TRAIN_DATA_1" \
  > runs/huawei_deploy/huawei_go_30_clean_check_w8.pipeline.log 2>&1 &
```

如果 8 workers 仍触发 429，改成 4 并增加间隔：

```bash
export NUM_WORKERS=4
export ANNOTATE_MIN_REQUEST_INTERVAL=2.0

nohup bash scripts/huawei_deploy/annotate.sh check "$TRAIN_DATA_1" \
  > runs/huawei_deploy/huawei_go_30_clean_check_w4.pipeline.log 2>&1 &
```

检查输出：

```bash
ls -lh data/huawei_data/processed_30_clean
ls -lh outputs/huawei_deploy
```

打开 HTML：

```bash
python -m http.server 8000
```

浏览器访问：

```text
http://<server>:8000/outputs/huawei_deploy/huawei_go_30_clean_check_viewer.html
```

如果想确认 4 份数据都能被顺序处理，可以在小样本阶段也传入 4 个路径。注意：此时是每份数据最多 accepted 30 条，合计最多约 120 条。

```bash
nohup bash scripts/huawei_deploy/annotate.sh check \
  "$TRAIN_DATA_1" \
  "$TRAIN_DATA_2" \
  "$TRAIN_DATA_3" \
  "$TRAIN_DATA_4" \
  > runs/huawei_deploy/huawei_go_4files_30_check.pipeline.log 2>&1 &
```

### 3. 全量标注

30 条检查没问题后，跑 4 份训练数据的全量标注。这里使用命令行位置参数传入 4 个 jsonl，脚本会串行处理：第 1 份完成后自动处理第 2 份，然后第 3、4 份。

```bash
export HUAWEI_PROCESSED_DIR=data/huawei_data/processed_full_clean
export OUT_DIR=data/huawei_data/processed_full_clean
export RUN_DIR=runs/huawei_deploy
export VIS_OUT_DIR=outputs/huawei_deploy

export PREPARE_MAX_ACCEPTED_ROWS=0
export FORCE_PREPARE=1
export ENABLE_VISUALIZE=0
```

沿用小样本检查中最快且不 429 的并发。例如 8 稳定：

```bash
export NUM_WORKERS=8
export ANNOTATE_MIN_REQUEST_INTERVAL=1.0

nohup bash scripts/huawei_deploy/annotate.sh full \
  "$TRAIN_DATA_1" \
  "$TRAIN_DATA_2" \
  "$TRAIN_DATA_3" \
  "$TRAIN_DATA_4" \
  > runs/huawei_deploy/huawei_go_4files_full.pipeline.log 2>&1 &
```

如果 8 不稳，使用保守配置：

```bash
export NUM_WORKERS=4
export ANNOTATE_MIN_REQUEST_INTERVAL=2.0

nohup bash scripts/huawei_deploy/annotate.sh full \
  "$TRAIN_DATA_1" \
  "$TRAIN_DATA_2" \
  "$TRAIN_DATA_3" \
  "$TRAIN_DATA_4" \
  > runs/huawei_deploy/huawei_go_4files_full.pipeline.log 2>&1 &
```

监控进度：

```bash
tail -f runs/huawei_deploy/huawei_go_4files_full.pipeline.log
ls -lh data/huawei_data/processed_full_clean
ls -lh runs/huawei_deploy
```

全量输出会按每个 raw 文件的 basename 自动派生。例如输入文件名是 `cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl`，主要输出类似：

```text
data/huawei_data/processed_full_clean/cloud_core_test_25.JunJunly_GoOnly_length_filter_chatml.jsonl
data/huawei_data/processed_full_clean/cloud_core_test_25.JunJunly_GoOnly_length_filter_canonical.jsonl
data/huawei_data/processed_full_clean/cloud_core_test_25.JunJunly_GoOnly_length_filter_full_huawei_agent_compact.jsonl
runs/huawei_deploy/cloud_core_test_25.JunJunly_GoOnly_length_filter_full_huawei_agent.annotation_cache.jsonl
```

## 部署训练
