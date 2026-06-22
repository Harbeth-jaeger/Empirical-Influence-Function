# CodeSearchNet FIM 数据处理脚本交接

本文记录 CodeSearchNet FIM 语料处理链路，重点面向 Go 以外的 CSN 语料，例如 Python、Java、C、C++、C#。整体链路是：

```text
CodeSearchNet raw jsonl/jsonl.gz/zip
  -> 挖单行 mask，生成 canonical + ChatML
  -> 标注 token 依赖边，生成 compact annotated/train-readable JSONL
  -> 可视化标注边，抽查质量
```

## 预先准备

从仓库根目录运行脚本，脚本里大多使用相对路径。建议使用项目已有环境：

```bash
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"
```

需要的主要包是 `transformers`、`torch`、`tqdm`、`openai`，以及项目内的 `src/annotate`、`src/sft` 模块。标注如果走新版 `target_evidence` 规则，不需要外部 LLM；如果走旧版 `oneshot/agent` 标注，则需要配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`ANNOTATE_MODEL` 等环境变量。

原始 CSN 数据默认按语言放在类似下面的位置：

```text
data/raw_data/codesearchnet/python/...
data/raw_data/codesearchnet/java/...
data/raw_data/codesearchnet/cpp/...
data/raw_data/codesearchnet/csharp/...
```

多语言构造脚本也能读取 `.jsonl`、`.jsonl.gz`，以及语言级 zip 包里的 jsonl/jsonl.gz，运行时最好显式传入输入输出路径。

## 1. 多语言 CSN 单行 FIM 构造

推荐入口：`scripts/data_process/build_codesearchnet_single.py`

这个脚本已经是 Go 单行挖 mask 逻辑的多语言泛化版，适用于 `python,java,c,cpp,csharp`。它会从 CSN 函数级代码中挖一行作为 target，并输出两份数据：

- canonical：包含 `prefix/target/suffix/full_code/target_kind/metadata`。
- chatml：在 canonical 基础上加入 `messages` 和 `only_last_turn_loss=True`，可送入标注或 tokenizer/binarize。

核心筛选逻辑：

- 过滤测试路径、生成代码、测试符号、噪声输出调用。
- 当前策略是“有注释就过滤整条函数/代码”，不是删除所有注释后继续用。
- target 只保留较稳定的单行语句：`assignment`、`return`、`call`。
- Python 会跳过 `if/for/while/def/class/import/...` 这类块头或低价值语句。
- C-like 语言要求语句通常以 `;` 结束，并跳过控制块头、花括号行等。
- 默认每种语言最多采样 `--per_language 20000`，按 `assignment:return:call = 0.45:0.30:0.25` 做近似平衡。

示例命令：

```bash
python scripts/data_process/build_codesearchnet_single.py \
  --raw_root data/raw_data/codesearchnet \
  --output_root data \
  --languages python,java,cpp,csharp \
  --per_language 20000 \
  --max_source_rows 0 \
  --seed 42 \
  --report_path outputs/docs/codesearchnet_multilang_single_report.md \
  --preview_path outputs/docs/codesearchnet_multilang_single_preview.md
```

输出路径形如：

```text
data/python_single/train_data/python_single_train_codesearchnet_<N>_canonical.jsonl
data/python_single/train_data/python_single_train_codesearchnet_<N>_chatml.jsonl
data/java_single/train_data/java_single_train_codesearchnet_<N>_canonical.jsonl
data/java_single/train_data/java_single_train_codesearchnet_<N>_chatml.jsonl
```

必须关注的超参数：

- `--languages`：处理 Go 以外语料时，建议显式写 `python,java,cpp,csharp`。
- `--per_language`：最终每个语言保留多少条。原始过滤比较严格，如果某语言不足，脚本会输出实际条数。
- `--max_source_rows`：调试时可以设小，例如 `10000`；正式构造设 `0` 表示扫完整数据。
- `--raw_root`：必须对齐本地 CSN 原始数据目录。脚本会按语言名搜索子目录或 zip。

## 2. Go 旧版构造代码（检查作用）

这章节可以不看、放进来主要是用来检查的，即如果老师觉得代码有问题、或者构造思路是怎么样的，可以看看原始设计：如何过滤 CSN、如何挖单行 mask、如何渲染 ChatML、如何写构造报告。

Go 旧实验入口：`scripts/go_singleline_fim_exp/build_go_single_data.py`

共享逻辑：`scripts/go_singleline_fim_exp/go_single_pipeline.py`

关键逻辑包括：

- `should_reject_codesearchnet_row()`：过滤测试文件、测试 import、生成代码、注释、噪声调用。
- `find_function_slice()`：抽出 Go 函数体。
- `extract_statement_candidates()`：在函数体里找 assignment/return/call 单行 target。
- `render_chatml()`：生成 Qwen ChatML FIM prompt。
- `filter_by_function_lines()`：可按函数行数上限或分位数过滤过长函数。

旧 Go 构造示例，路径需按当前仓库实际情况显式传：

```bash
python scripts/go_singleline_fim_exp/build_go_single_data.py \
  --codesearchnet-dir data/raw_data/codesearchnet/go/final/jsonl/unzip \
  --codesearchnet-glob 'go_train_*.jsonl' \
  --train-output data/go_singleline_fim_exp/train_data/go_single_train_v2_canonical.jsonl \
  --train-chatml-output data/go_singleline_fim_exp/train_data/go_single_train_v2_chatml.jsonl \
  --num-train 10000 \
  --max-train-rows 0 \
  --seed 42 \
  --report outputs/go_singleline_fim_exp/reports/go_single_v2_build_report.md \
  --samples-report outputs/go_singleline_fim_exp/reports/go_single_v2_samples.md
```

注意：这套 Go 脚本还会构造 MCEval eval 数据；如果只交接 CSN train，可以重点看 train 相关参数。

## 3. 标注：生成 compact annotated / train-readable 数据

### 推荐：LLM/agent 标注

旧入口：`scripts/go_singleline_fim_exp/annotate_chatml_with_src_annotate.py`

这套最初为 Go single-line 写的，但内部有语言 normalize，也能处理 Python/Java/C/C++/C#。它支持：

- `--annotation_mode oneshot`：结构规则 + 一次 LLM JSON 标注。
- `--annotation_mode agent`：调用 `AnnotatorAgent` 多轮工具式标注。

输出格式同样是 compact：`input_ids/label/attention_edges/...`。

示例命令：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export ANNOTATE_MODEL=...

python scripts/go_singleline_fim_exp/annotate_chatml_with_src_annotate.py \
  --input_path data/python_single/train_data/python_single_train_codesearchnet_20000_chatml.jsonl \
  --output_path data/python_single/train_data/python_single_train_codesearchnet_20000_srcannotate_compact.jsonl \
  --annotation_cache_path data/python_single/train_data/python_single_train_codesearchnet_20000_srcannotate_cache.jsonl \
  --model_name_or_path models/Qwen2.5-Coder-7B-Instruct \
  --model_max_length 4096 \
  --annotation_mode oneshot \
  --max_teacher_edges 128 \
  --num_workers 4 \
  --flush_every 100
```

这条线的注意事项：

- 会调用外部 LLM，成本、限流、失败重试都要考虑。
- `--num_workers` 过高容易被 API 限流。

### 新版：规则标注

为了应对 safim 的 file level 构思的一个标注策略，仅供参考、对 singleline FIM 的数据意义不大

推荐入口：`scripts/benchmark/annotate_safim_train.py`

Python 示例：

```bash
python scripts/benchmark/annotate_safim_train.py \
  --input-path data/python_single/train_data/python_single_train_codesearchnet_20000_chatml.jsonl \
  --output-path data/python_single/train_data/python_single_train_codesearchnet_20000_target_evidence_compact.jsonl \
  --annotation-cache-path data/python_single/train_data/python_single_train_codesearchnet_20000_target_evidence_cache.jsonl \
  --model-name-or-path models/Qwen2.5-Coder-7B-Instruct \
  --languages python \
  --task-types python_single_statement_completion \
  --annotation-mode target_evidence \
  --model-max-length 4096 \
  --max-edges 128 \
  --num-workers 4 \
  --flush-every 100
```

Java 示例：

```bash
python scripts/benchmark/annotate_safim_train.py \
  --input-path data/java_single/train_data/java_single_train_codesearchnet_20000_chatml.jsonl \
  --output-path data/java_single/train_data/java_single_train_codesearchnet_20000_target_evidence_compact.jsonl \
  --annotation-cache-path data/java_single/train_data/java_single_train_codesearchnet_20000_target_evidence_cache.jsonl \
  --model-name-or-path models/Qwen2.5-Coder-7B-Instruct \
  --languages java \
  --task-types java_single_statement_completion \
  --annotation-mode target_evidence \
  --model-max-length 4096 \
  --max-edges 128 \
  --num-workers 4 \
  --flush-every 100
```

特别注意：

- `--num-workers` 是 CPU 规则标注并发，4 到 16 都可以试。
- `--overwrite-cache` 只有在修改了标注逻辑、希望重算时才加。
- 输出 compact 文件已经可以作为 GraphSignal 训练数据读取，不需要再跑 `sft_data_convert.py`。


## 4. 后处理为 train 可直接读

有两种情况。

第一种：如果已经通过 `annotate_safim_train.py` 或 `annotate_chatml_with_src_annotate.py` 得到 compact annotated 文件，那么它已经是训练可读格式，字段包括 `input_ids`、`label`、`length`、`attention_edges`。GraphSignal 训练可以直接读这类文件。

第二种：如果只有 raw ChatML，想生成 baseline 训练 JSON，或者想套其他 baseline 的治理算子，使用 `scripts/benchmark/sft_data_convert.py`。注意它的输入应当是带 `messages` 的 ChatML JSONL，而不是 compact annotated JSONL；compact annotated 已经没有 `messages` 字段，直接拿去喂这个脚本会被过滤掉。

它的作用是：读取 ChatML JSONL，应用治理算子，tokenize/binarize，写出训练 JSONL。常用 operator：

- `none`：不做治理，只 tokenize 成 train JSON。
- `all`：跑多个 baseline，不建议一上来全量跑。

无标注 baseline 示例：

```bash
python scripts/benchmark/sft_data_convert.py \
  --input_path data/python_single/train_data/python_single_train_codesearchnet_20000_chatml.jsonl \
  --output_path data/python_single/train_data/python_single_train_codesearchnet_20000_none_train.json \
  --operator none \
  --tokenizer_path models/Qwen2.5-Coder-7B-Instruct \
  --max_len 4096 \
  --max_samples 0
```

注意：`sft_data_convert.py` 最早服务 benchmark/curation baseline，部分 operator 会加载模型或调用 API。只想要可训练的 tokenizer JSON 时，用 `--operator none` 最稳。

## 5. 标注可视化

### 推荐：单样本静态 HTML

入口：`tools/viz_annotation/visualize_annotation_edges.py`

适合把若干样本导出成单独 HTML 文件，调试看边是否合理。

```bash
python tools/viz_annotation/visualize_annotation_edges.py \
  --edge_data_path data/python_single/train_data/python_single_train_codesearchnet_20000_target_evidence_compact.jsonl \
  --sample_indices 0 1 2 3 4 5 6 7 8 9 \
  --model_path models/Qwen2.5-Coder-7B-Instruct \
  --output_dir outputs/viz_annotation/csn_python_single_edges \
  --local_files_only
```

输出形如：

```text
outputs/viz_annotation/csn_python_single_edges/annotate_sample0_ours_graphsignal.html
```

### 动态 Viewer

入口：`tools/viz_annotation/build_dynamic_annotation_viewer.py`

适合给一个可以切换样本、点击 token 看边的完整 viewer。

```bash
python tools/viz_annotation/build_dynamic_annotation_viewer.py \
  --edge_data_path data/python_single/train_data/python_single_train_codesearchnet_20000_target_evidence_compact.jsonl \
  --sample_indices 0 1 2 3 4 5 6 7 8 9 \
  --model_path models/Qwen2.5-Coder-7B-Instruct \
  --output_path outputs/viz_annotation/csn_python_single_dynamic_viewer.html \
  --max_length 4096 \
  --local_files_only
```

如果想叠加模型 attention top-k，需要加 `--with_attention`，但是这会加载模型，耗显存。

## 6. 建议交接顺序

建议给学姐上传这些文件：

```text
scripts/data_process/build_codesearchnet_single.py
scripts/go_singleline_fim_exp/build_go_single_data.py
scripts/go_singleline_fim_exp/go_single_pipeline.py
scripts/go_singleline_fim_exp/annotate_chatml_with_src_annotate.py
scripts/benchmark/annotate_safim_train.py
src/annotate/target_evidence_annot.py
scripts/benchmark/sft_data_convert.py
tools/viz_annotation/visualize_annotation_edges.py
tools/viz_annotation/build_dynamic_annotation_viewer.py
tools/viz_annotation/dynamic_annotation_viewer.html
src/annotate/viz_utils.py
```

真正建议她先跑的最短链路是：

```text
build_codesearchnet_single.py
  -> annotate_safim_train.py --annotation-mode target_evidence
  -> build_dynamic_annotation_viewer.py
```

`go_singleline_fim_exp` 目录更像方法历史和 Go 规则参考，不建议她直接拿它作为多语言主入口，除非要复现实验旧结果。
