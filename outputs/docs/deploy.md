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

本节是华为机器现场优先执行的命令。路径、数据、tokenizer/base model、外部标注模型都以这里为准；下面同时设置 `TRAIN_DATA` 和 `HUAWEI_RAW_DATA`，避免脚本回退到本地默认路径。

注意：不要把真实 `HW_APPKEY / OPENAI_API_KEY / ANNOTATE_MODEL / HW_OPERATOR` 固化提交到仓库文档。真实值从华为机器本地 `EIF-huawei-annotation/常用命令.md` 复制，或者整理到一个不提交 git 的本地 env 文件后 `source`。

关于调用模型进行在线推理、辅助标注，华为官方提供了部署在他们机器上的模型，具体调用详见`scripts/huawei_deploy/vlm.py` 。

### 0. 进入仓库和环境

```bash
cd /home/model_project/EIF-huawei-annotation
source .venv/bin/activate
conda activate EIF
```

### 1. 公共路径、API 和模型变量

```bash
export TRAIN_DATA=/home/model_project/Open_CC_SFT_Eval/train/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl
export HUAWEI_RAW_DATA="$TRAIN_DATA"
export MODEL_PATH=/home/model_project/CCCodeGenerationTrain/infer_format/

export REQUIRE_HUAWEI_GATEWAY=1
export OPENAI_BASE_URL=https://apigw-cn-south02.huawei.com/api/v1
export OPENAI_API_KEY="com.huawei.ipd.coretool.coreai"
export ANNOTATE_MODEL="6d2c5ff6-615d-45a8-9703-2f591d6c2437"

export HW_ID=com.huawei.ipd.coretool.coreai
export HW_APPKEY="WxhsDOVQJGVYpkDfQ7C2HA=="
export HW_APP_ID=com.huawei.ipd.coretool.coreai
export HW_SCENE=test
export HW_OPERATOR="h00965148"

export ANNOTATE_HTTP_PROXY_NONE=1
export ANNOTATE_VERIFY_SSL=0
export HW_ENABLE_THINKING=0
export ANNOTATE_TEMPERATURE=0.2
export ANNOTATE_MIN_REQUEST_INTERVAL=1.0
export ANNOTATE_MAX_RETRIES=8
export ANNOTATE_RETRY_BASE_SLEEP=10
export ANNOTATE_MAX_TOKENS=2048

export ANNOTATION_MODE=agent
export STRIP_CJK_COMMENTS=1
export MAX_TARGET_NONEMPTY_LINES=10
export MAX_TARGET_ROUGH_TOKENS=192
export MAX_TARGET_CHARS=1024
export FILTER_GOFMT_VALID=1
export GOFMT_BIN=gofmt
```

### 2. 先跑 30 条检查并生成 HTML

先创建目录。注意：`nohup > runs/...log` 的重定向发生在脚本启动前，所以 `runs/huawei_deploy` 必须提前存在。

```bash
mkdir -p data/huawei_data/processed_30_clean
mkdir -p outputs/huawei_deploy
mkdir -p runs/huawei_deploy
```

配置 30 条检查路径：

```bash
export HUAWEI_PROCESSED_DIR=data/huawei_data/processed_30_clean
export HUAWEI_CHATML_DATA=data/huawei_data/processed_30_clean/huawei_go_30_clean_chatml.jsonl
export HUAWEI_CANONICAL_DATA=data/huawei_data/processed_30_clean/huawei_go_30_clean_canonical.jsonl
export HUAWEI_PREPARE_REPORT=data/huawei_data/processed_30_clean/huawei_go_30_clean_prepare_report.json

export OUT_DIR=data/huawei_data/processed_30_clean
export RUN_DIR=runs/huawei_deploy
export VIS_OUT_DIR=outputs/huawei_deploy
export CHECK_RUN_NAME=huawei_go_30_clean_check
export CHECK_ROWS=30
export VALIDATE_LIMIT=100
export PREPARE_MAX_ACCEPTED_ROWS=30
export FORCE_PREPARE=1
export ENABLE_VISUALIZE=1
```

先试 12 workers：

```bash
export NUM_WORKERS=12
nohup bash scripts/huawei_deploy/annotate.sh check   > runs/huawei_deploy/huawei_go_30_clean_check_w12.pipeline.log 2>&1 &
```

查看日志：

```bash
tail -f runs/huawei_deploy/huawei_go_30_clean_check_w12.pipeline.log
```

如果 12 workers 触发 429，改成 8 重新跑：

```bash
export NUM_WORKERS=8
nohup bash scripts/huawei_deploy/annotate.sh check   > runs/huawei_deploy/huawei_go_30_clean_check_w8.pipeline.log 2>&1 &
```

如果 8 workers 仍触发 429，改成 4 并增加间隔：

```bash
export NUM_WORKERS=4
export ANNOTATE_MIN_REQUEST_INTERVAL=2.0
nohup bash scripts/huawei_deploy/annotate.sh check   > runs/huawei_deploy/huawei_go_30_clean_check_w4.pipeline.log 2>&1 &
```

检查输出：

```bash
wc -l data/huawei_data/processed_30_clean/huawei_go_30_clean_chatml.jsonl
wc -l data/huawei_data/processed_30_clean/huawei_go_30_clean_canonical.jsonl
wc -l data/huawei_data/processed_30_clean/huawei_go_30_clean_check_compact.jsonl
ls -lh outputs/huawei_deploy/huawei_go_30_clean_check_viewer.html
```

打开 HTML：

```bash
python -m http.server 8000
```

浏览器访问：

```text
http://<server>:8000/outputs/huawei_deploy/huawei_go_30_clean_check_viewer.html
```

### 3. 30 条检查没问题后跑 5000 条

先创建目录：

```bash
mkdir -p data/huawei_data/processed_5k_clean
mkdir -p runs/huawei_deploy
```

配置 5K 路径。这里用 `full` 模式，但 prepare 阶段只收满 5000 条 accepted 样本，所以不是全量 raw。

```bash
export HUAWEI_PROCESSED_DIR=data/huawei_data/processed_5k_clean
export HUAWEI_CHATML_DATA=data/huawei_data/processed_5k_clean/huawei_go_5k_clean_chatml.jsonl
export HUAWEI_CANONICAL_DATA=data/huawei_data/processed_5k_clean/huawei_go_5k_clean_canonical.jsonl
export HUAWEI_PREPARE_REPORT=data/huawei_data/processed_5k_clean/huawei_go_5k_clean_prepare_report.json

export OUT_DIR=data/huawei_data/processed_5k_clean
export RUN_DIR=runs/huawei_deploy
export FULL_RUN_NAME=huawei_go_5k_clean
export PREPARE_MAX_ACCEPTED_ROWS=5000
export FORCE_PREPARE=1
```

沿用 30 条检查中最快且不 429 的并发。例如 8 稳定：

```bash
export NUM_WORKERS=8
export ANNOTATE_MIN_REQUEST_INTERVAL=1.0
nohup bash scripts/huawei_deploy/annotate.sh full   > runs/huawei_deploy/huawei_go_5k_clean.pipeline.log 2>&1 &
```

如果 8 不稳，使用保守配置：

```bash
export NUM_WORKERS=4
export ANNOTATE_MIN_REQUEST_INTERVAL=2.0
nohup bash scripts/huawei_deploy/annotate.sh full   > runs/huawei_deploy/huawei_go_5k_clean.pipeline.log 2>&1 &
```

监控进度：

```bash
tail -f runs/huawei_deploy/huawei_go_5k_clean.pipeline.log
wc -l runs/huawei_deploy/huawei_go_5k_clean.annotation_cache.jsonl
wc -l data/huawei_data/processed_5k_clean/huawei_go_5k_clean_compact.jsonl
```

5K 输出：

```text
data/huawei_data/processed_5k_clean/huawei_go_5k_clean_chatml.jsonl
data/huawei_data/processed_5k_clean/huawei_go_5k_clean_canonical.jsonl
data/huawei_data/processed_5k_clean/huawei_go_5k_clean_compact.jsonl
runs/huawei_deploy/huawei_go_5k_clean.annotation_cache.jsonl
runs/huawei_deploy/huawei_go_5k_clean.pipeline.log
```

## 部署训练
