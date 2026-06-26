# 标注流水线（`src/annotate`）

将原始 [CodeSearchNet](https://github.com/github/CodeSearchNet) 函数转换为带有 token 级依赖边标注的 FIM（fill-in-the-middle，中间填充）训练数据。整体分为两步：

1. **`preprocess_to_chatml.py`**：从每个函数中抽取一行作为 FIM 目标，将其包装进 FIM system prompt，并输出 ChatML。
2. **`run.py`**：为每一行 ChatML 样本标注 token 级依赖边，包括确定性的 tree-sitter 结构边，以及一次 LLM 标注。

下面的命令使用 `$lang` shell 变量。请先将它设置为 CodeSearchNet 支持的语言之一：`go`、`java`、`javascript`、`python`、`php`、`ruby`。

```bash
export lang=go      # 或：java / javascript / python / php / ruby
```

如果要处理所有语言，可以把命令包在循环里：

```bash
for lang in go java javascript python php ruby; do
  # ... 在这里运行 step-1 / step-2 的命令 ...
done
```

> 脚本自身的 `--languages` 参数也支持逗号分隔的语言列表，例如
> `--languages go,python,java`；这里使用 `$lang` 只是为了让示例命令
> 以及输出路径更具体，方便直接复制粘贴。

---

## Step 1 — 预处理为 ChatML

该步骤读取 `--raw_root/jsonl/{train,valid,test}/*.jsonl(.gz)`，从每个函数中选择一条 assignment / return / call 语句作为被 mask 的目标行，并按 split 输出 ChatML。请将 `--raw_root` 指向对应语言的原始 CSN 目录（`data/$lang/final`），这样数据发现过程只会看到原始数据，而不会扫描 `data/$lang/output_train/` 下生成的输出文件。

```bash
python src/annotate/preprocess_to_chatml.py \
  --raw_root data/$lang/final \
  --output_root data/$lang/output_train \
  --languages $lang
```

split 处理方式如下（默认 `--splits train,valid,test`）：

* 默认情况下**不会进行任何子采样**，train / valid / test 都会完整输出。
* 传入 `--per_language N` 时，只会对子采样 **train split**，得到 `N` 条均衡样本，例如 `--per_language 20000`；valid / test 始终完整输出。

输出文件如下，其中 N 表示该 split 实际产生的样本行数：

```text
data/$lang/output_train/${lang}_single/train_data/${lang}_single_train_codesearchnet_${N}_chatml.jsonl
data/$lang/output_train/${lang}_single/train_data/${lang}_single_valid_codesearchnet_${N}_chatml.jsonl
data/$lang/output_train/${lang}_single/train_data/${lang}_single_test_codesearchnet_${N}_chatml.jsonl
```

你也可以一次传入多种语言，例如 `--languages go,python,java`；也可以限制 split，例如 `--splits train`。

---

## Step 2 — 标注

默认情况下，`run.py` 只会基于 tree-sitter 生成**纯结构标注**，这是确定性的，不需要模型或网络，因此速度很快。如果希望同时调用 LLM 生成语义边 / `context->completion` 边，请添加 `--use_llm`；只有这种情况下才需要 `.env` 和模型 endpoint。

### 2a. 配置 `.env`（仅在使用 `--use_llm` 时需要）

在仓库根目录创建或编辑 `.env`，然后将其加载到当前 shell 中：

```bash
set -a && source .env && set +a
```

**方案 A — 通过 vLLM 本地运行 Qwen-Coder（推荐；不需要代理或 VPN）。**

先启动服务，该服务会在 8000 端口将模型作为 `qwen-coder` 提供：

```bash
bash src/scripts/run_local_llm.sh      # 使用 $HF_HOME/Qwen2.5-Coder-32B-Instruct
```

`.env`：

```dotenv
HF_HOME=/path/to/your/hf_hub          # tokenizer + model cache 根目录
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=dummy                   # vLLM 会忽略它，但 client 要求必须有值
ANNOTATE_MODEL=qwen-coder              # 必须和 vllm --served-model-name 一致
REQUIRE_HUAWEI_GATEWAY=0               # 重要：使用普通 OpenAI 调用，保留 JSON mode + max_tokens
# localhost / 127.0.0.1 会自动绕过代理，所以不需要额外设置
```

**方案 B — 远程 OpenAI-compatible API。**

```dotenv
HF_HOME=/path/to/your/hf_hub
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.your-provider.com/v1   # 如果使用 OpenAI 官方接口，可省略
ANNOTATE_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct     # provider 的模型 ID
REQUIRE_HUAWEI_GATEWAY=0
ANNOTATE_HTTP_PROXY_NONE=1                          # 如果 VPN 代理会劫持请求，则设置该项
```

加载环境变量：

```bash
set -a && source .env && set +a
```

### 2b. 运行标注

`run.py` 会完整标注 `--input_path` 中的每一行样本，不会进行子采样。因此需要对每个 split 分别运行一次，并为每个 split 使用单独的 cache。输入文件名中包含样本行数，因此推荐用 glob 自动解析文件名，而不是手动输入数字。这样只需要提前设置 `$lang`：

```bash
export lang=go      # 可选：go java javascript python php ruby
base=data/$lang/output_train/${lang}_single

mkdir -p "$base/annotated"
for split in train valid test; do
  in=$(ls "$base"/train_data/${lang}_single_${split}_codesearchnet_*_chatml.jsonl)
  python src/annotate/run.py \
    --input_path  "$in" \
    --output_path "$base/annotated/${split}.jsonl" \
    --annotation_cache_path "$base/annotated/${split}_cache.jsonl" \
    --num_workers 16 --gzip_output --flush_every 100
done
```

使用 `--gzip_output` 时，cache 和输出文件都会写为 `.jsonl.gz`。如果只想运行单个 split，可以去掉循环并手动设置 `split=valid` 等。

---

## 环境变量（由 `run.py` → `neural_annot.py` 读取）


| 变量                            | 默认值                          | 用途                                                                                                                          |
| ------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY`                | 也会从`data/openai_key.txt`读取 | API key；本地 vLLM 场景下可以填任意非空值                                                                                     |
| `OPENAI_BASE_URL`               | OpenAI 默认值                   | endpoint URL，本地 vLLM 或远程 provider                                                                                       |
| `ANNOTATE_MODEL`                | `gpt-4o-mini`                   | 发送给 endpoint 的模型名                                                                                                      |
| `REQUIRE_HUAWEI_GATEWAY`        | `1`，开启                       | 华为 ModelArts 请求模板。**如果使用 vLLM / OpenAI / 其他 provider，请设置为`0`**；否则`response_format`和`max_tokens`会被移除 |
| `ANNOTATE_HTTP_PROXY_NONE`      | localhost 自动开启              | 绕过`HTTP(S)_PROXY`，避免 VPN 代理劫持请求                                                                                    |
| `ANNOTATE_MAX_TOKENS`           | `1024`                          | 每次标注调用最多生成的 token 数                                                                                               |
| `ANNOTATE_MIN_REQUEST_INTERVAL` | `0`                             | 请求之间的最小间隔秒数；只有在网关限流时才需要调高                                                                            |
| `ANNOTATE_MAX_RETRIES`          | `8`                             | 遇到临时错误时的重试次数，例如 429 / 5xx / 连接错误                                                                           |
| `ANNOTATE_TEMPERATURE`          | 未设置，即 0                    | 采样温度                                                                                                                      |
| `ANNOTATE_VERIFY_SSL`           | `1`                             | 是否验证 TLS 证书                                                                                                             |
| `HF_HOME`                       | —                              | HF cache 根目录；tokenizer 默认路径为`$HF_HOME/Qwen2.5-Coder-7B-Instruct`                                                     |

`run.py` 参数：`--use_llm`（默认关闭 → 只生成结构边；开启 → 同时调用 LLM）、`--model_name_or_path`（tokenizer，默认从 `HF_HOME` 推断）、`--num_workers`、`--flush_every`、`--annotation_cache_path`、`--gzip_output`、`--overwrite_cache`、`--max_rows`、`--max_teacher_edges`、`--model_max_length`。

下面这些环境变量只有在使用 `--use_llm` 时才有意义。

---

## 注意事项

* **断点续跑**：使用相同的 `--annotation_cache_path` 重新运行时，会跳过已经存在于 cache 中的样本。失败的样本不会写入 cache，因此下次运行时会重新尝试。
* **临时错误**：429 / 502 / 连接错误会使用 backoff 机制自动重试；如果程序被杀死导致 `.gz` cache 尾部损坏，脚本会自动恢复。
* **LLM 失败 fallback**：如果 LLM 调用失败，例如 bad request，该行样本不会被丢弃，而是保留确定性的 tree-sitter 结构边。
* 标注 tokenizer（`--model_name_or_path`）只用于计算 token offset；它和负责生成依赖边的 LLM（`ANNOTATE_MODEL`）相互独立。
