## Benchmark 数据、治理与评测

本文整理 benchmark 相关的数据来源、字段对齐、训练/评测使用字段、baseline 分类、当前结果，以及 `scripts/benchmark` 中主要代码文件的作用。

## 一、Benchmark 的任务定义

本项目的 benchmark 是函数级或代码片段级 FIM 补全任务。每个样本都被统一成单空洞补全：

```text
prefix + target + suffix = full_code
```

模型输入不是完整代码，而是包含 `prefix` 和 `suffix` 的补全请求；模型输出 `target`。评测时再拼回：

```text
full_code = prefix + predicted_completion + suffix
```

然后用对应数据集和语言的 judge 运行测试。这个设定保证训练可以走 ChatML SFT，评测可以走代码执行或单元测试。

## 二、数据来源

### 1. 训练原始数据

训练原始数据来自 McEval-Instruct：


| 文件                                                                            |  行数 | 原始字段                                      | 作用                                             |
| ------------------------------------------------------------------------------- | ----: | --------------------------------------------- | ------------------------------------------------ |
| `data/benchmarks/sft_data/mceval_instruct/mceval_instruct_train.jsonl`          | 35943 | `language`, `instruction`, `source`, `output` | McEval-Instruct 原始训练数据。                   |
| `data/benchmarks/sft_data/mceval_instruct/mceval_instruct_train_filtered.jsonl` |  5259 | `language`, `instruction`, `source`, `output` | 过滤到 benchmark 目标语言后用于构造 SFT 训练集。 |

最终训练集只保留六种评测相关语言：


| Language | Rows |
| -------- | ---: |
| `python` |  926 |
| `csharp` |  915 |
| `go`     |  905 |
| `java`   |  875 |
| `c`      |  868 |
| `cpp`    |  770 |

McEval-Instruct 的原始形式是 instruction-to-code，不是天然 FIM。因此 `build_chatml_fim_unified.py` 会从唯一 fenced code block 中切出中间一段作为 synthetic FIM target，形成 `prefix/suffix/target`。

### 2. 评测原始数据

评测数据来自三个数据集：


| Dataset   | 原始文件                                    | 原始行数 | 原始关键字段                                                               |
| --------- | ------------------------------------------- | -------: | -------------------------------------------------------------------------- |
| HumanEval | `data/benchmarks/eval_data/humaneval.jsonl` |     5815 | `prompt`, `suffix`, `canonical_solution`, `test`, `entry_point`            |
| McEval    | `data/benchmarks/eval_data/mceval.jsonl`    |      518 | `prefix_code`, `suffix_code`, `masked_spans`, `canonical_solution`, `test` |
| SAFIM     | `data/benchmarks/eval_data/safim.jsonl`     |     8781 | `eval_prompt`, `ground_truth`, `unit_tests`, `lang`                        |

格式校验和语言过滤后，统一评测文件有 15107 条：


| Dataset   | Rendered eval rows |
| --------- | -----------------: |
| HumanEval |               5815 |
| McEval    |                511 |
| SAFIM     |               8781 |

## 三、格式对齐后的数据形态

### 1. Canonical FIM schema

所有原始数据先被对齐到 canonical FIM schema：

```text
uid
source_dataset
split
language
task_type
prefix
suffix
target
raw_id
metadata
judge_payload
```

其中 `prefix/suffix/target` 是任务语义核心，`judge_payload` 保存测试代码或单元测试，`metadata` 保存 entry point、签名、原始语言等辅助信息。

对应产物：


| 文件                                        |  行数 | 作用                                 |
| ------------------------------------------- | ----: | ------------------------------------ |
| `data/benchmarks/canonical_fim_all.jsonl`   | 20366 | train + eval 的 canonical FIM 总表。 |
| `data/benchmarks/canonical_fim_train.jsonl` |  5259 | 训练 canonical FIM。                 |
| `data/benchmarks/canonical_fim_eval.jsonl`  | 15107 | 评测 canonical FIM。                 |

### 2. Rendered ChatML-FIM schema

随后 canonical FIM 被渲染成 Qwen-Instruct 可用的 ChatML-FIM 格式：

```text
uid
source_dataset
split
language
task_type
raw_id
messages
fim_prompt
fim_completion
metadata
judge_payload
```

其中 `messages` 是真正给模型训练和推理的 ChatML：

```text
system: 代码补全助手要求
user:   包含 <|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>
assistant: target
```

`fim_prompt` 和 `fim_completion` 保留 FIM 语义，主要服务于评测、oracle 和结果分析。

对应产物：


| 文件                                                       |  行数 | 作用                     |
| ---------------------------------------------------------- | ----: | ------------------------ |
| `data/benchmarks/sft_data/rendered_chatml_fim_train.jsonl` |  5259 | 所有治理算子的共同输入。 |
| `data/benchmarks/eval_data/rendered_chatml_fim_eval.jsonl` | 15107 | 所有模型评测的共同输入。 |

## 四、训练和评测分别使用哪些字段

### 1. 训练字段

对普通 SFT baseline，治理脚本先读取：

```text
messages
fim_completion
language
```

其中 `messages` 是主字段；LLM-CleanCode、CLEAR 这类需要改写 target 的方法会读取或替换 `fim_completion` / assistant message。

经过 tokenizer binarize 后，最终训练文件使用：

```text
input_ids
label
length
```

其中 `label == -100` 的位置不会参与 next-token loss。普通 baseline 最终都对齐到这个格式。

GraphSignal 训练数据额外包含：

```text
attention_edges
```

每条边通常形如：

```text
{"src": source_token_index, "dst": target_token_index, "subtype": relation_type}
```

这些边会被 `src/train/dataset.py` 读成 `annot_pairs`，再进入 saliency loss。

最终训练数据路径：


| 方法             | 训练数据路径                                           | 行数 | 数据形态                                 |
| ---------------- | ------------------------------------------------------ | ---: | ---------------------------------------- |
| None / 普通 SFT  | `data/benchmarks/sft_data/none_train.json`             | 5259 | `input_ids/label/length`                 |
| TokenCleaning    | `data/benchmarks/sft_data/token_cleaning_train.json`   | 5259 | token-level hard mask 后的`label`        |
| XTF              | `data/benchmarks/sft_data/xtf_train.json`              | 5259 | token-level hard mask 后的`label`        |
| LLM-CleanCode    | `data/benchmarks/sft_data/llm_cleancode_train.json`    | 5259 | LLM 改写 completion 后再 binarize        |
| CLEAR            | `data/benchmarks/sft_data/clear_train.json`            |  567 | sample-level filter/correct 后保留的样本 |
| IB-FT            | `data/benchmarks/sft_data/ibft_train.json`             | 5259 | 原始 label，训练时改 loss                |
| Ours GraphSignal | `data/benchmarks/sft_data/ours_graphsignal_train.json` |  514 | `input_ids/label/length/attention_edges` |

### 2. 评测字段

评测读取 `data/benchmarks/eval_data/rendered_chatml_fim_eval.jsonl`，主要使用：


| 字段                         | 使用位置                   | 作用                                                                   |
| ---------------------------- | -------------------------- | ---------------------------------------------------------------------- |
| `messages`                   | generation                 | 构造 system + user 的 ChatML prompt，让模型生成 assistant completion。 |
| `fim_completion`             | generation length / oracle | gold target；评测模型时不喂给模型，只用于估计生成长度上限和 oracle。   |
| `fim_prompt`                 | judge                      | 解析出`prefix/suffix`，把模型输出拼回完整代码。                        |
| `judge_payload`              | judge                      | HumanEval/McEval 的`test` 或 SAFIM 的 `unit_tests`。                   |
| `source_dataset`, `language` | judge dispatch / reporting | 决定用哪个 judge，并按数据集和语言聚合。                               |

评测逻辑是：

```text
prompt = ChatML(system + user)
prediction = model.generate(prompt)
full_code = prefix + sanitize(prediction) + suffix
pass = judge(full_code, judge_payload)
```

pass@1 使用 greedy generation；pass@10 使用 10 个 sampled generation，只要任意一个通过即记为通过。

## 五、Baseline 分类

当前代码里主要有六类对照项，外加我们自己的 GraphSignal 训练路线。


| 方法             | 分类                            | 核心思路                                                                                                          |
| ---------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Base Qwen        | 不训练/不治理对照               | 直接评测 Qwen2.5-Coder-1.5B-Instruct，作为基础能力下限。                                                          |
| TokenCleaning    | token-level hard mask           | 用 base model 与 reference model 的 token loss 差异打分，保留 top ratio token，其余 supervised label 置为`-100`。 |
| XTF              | token-level hard mask           | 用 attention relevance、PCP novelty、embedding task relevance 三类信号过滤噪声 token。                            |
| LLM-CleanCode    | sample/span-level rewrite       | 用 LLM 对 assistant completion 做 rename、modularize、planning 风格清洗，然后再 SFT。                             |
| CLEAR            | sample-level filter/correct     | 用 observed consistency 和 self-reflection 估计样本质量，低置信样本过滤，高置信候选可替换原 target。              |
| IB-FT            | loss-level optimization         | 不主要改数据，而是在训练时加 variational bottleneck 辅助 loss。                                                   |
| Ours GraphSignal | annotation edge + saliency loss | 先标注 token edge，再训练时约束标注边的 contribution saliency 高于非标注边。                                      |

需要注意：`scripts/benchmark` 里仍保留一个旧的 `graph_signal_operator`，它属于 annotation graph hard-mask/soft-weight 数据治理算子；当前我们主要使用的是 `src/annotate` 生成 `attention_edges`，再由 `src/train` 的 saliency loss 训练。

## 六、当前结果

下面结果来自 `outputs/benchmark/eval_results/`。全量 baseline 的 eval scope 是 15107 条，即 HumanEval + McEval + SAFIM。当前 `Ours Graphsignal 500` 是只用约 500 条已标注训练样本训练得到的 checkpoint，因此与全量训练 baseline 横比时要标注这一限制。

### 1. Oracle 上限

Gold Oracle 使用数据里的 `fim_completion` 作为 prediction，只检查拼接和 judge 是否可靠。


| **Dataset** | **Language** | **Pass@1** | **Pass@10** | **N** | **N_total** |
| ----------- | ------------ | ---------- | ----------- | ----- | ----------- |
| humaneval   | python       | 1.0000     | 1.0000      | 5815  | 5815        |
| mceval      | c            | 1.0000     | 1.0000      | 81    | 81          |
| mceval      | cpp          | 1.0000     | 1.0000      | 82    | 82          |
| mceval      | csharp       | 0.9639     | 0.9639      | 83    | 83          |
| mceval      | go           | 0.8750     | 0.8750      | 80    | 80          |
| mceval      | java         | 0.8958     | 0.8958      | 96    | 96          |
| mceval      | python       | 0.9438     | 0.9438      | 89    | 89          |
| safim       | cpp          | 0.9815     | 0.9815      | 4968  | 4968        |
| safim       | csharp       | 1.0000     | 1.0000      | 529   | 529         |
| safim       | java         | 0.9984     | 0.9984      | 2479  | 2479        |
| safim       | python       | 0.9988     | 0.9988      | 805   | 805         |

### 2. Overall


| Method               | Eval scope                            | Pass@1 | Pass@10 | N     |
| -------------------- | ------------------------------------- | ------ | ------- | ----- |
| Base Qwen            | 全量 eval: HumanEval + McEval + SAFIM | 0.0306 | 0.0757  | 15107 |
| TokenCleaning        | 全量 eval: HumanEval + McEval + SAFIM | 0.1642 | 0.2282  | 15107 |
| XTF                  | 全量 eval: HumanEval + McEval + SAFIM | 0.1697 | 0.2470  | 15107 |
| LLM-CleanCode        | 全量 eval: HumanEval + McEval + SAFIM | 0.0351 | 0.0911  | 15107 |
| CLEAR                | 全量 eval: HumanEval + McEval + SAFIM | 0.2239 | 0.3018  | 15107 |
| IB-FT                | 全量 eval: HumanEval + McEval + SAFIM | 0.1593 | 0.2263  | 15107 |
| Ours Graphsignal 500 | HumanEval 前 1000 条                  | 0.2520 | 0.3230  | 1000  |
| Ours Graphsignal 500 | HumanEval 全量                        | 0.2468 | 0.3221  | 5815  |

### 3. HumanEval


| Method               | Pass@1 | Pass@10 | N    |
| -------------------- | ------ | ------- | ---- |
| Base Qwen            | 0.0490 | 0.1066  | 5815 |
| TokenCleaning        | 0.2277 | 0.2841  | 5815 |
| XTF                  | 0.2377 | 0.3008  | 5815 |
| LLM-CleanCode        | 0.0325 | 0.0824  | 5815 |
| CLEAR                | 0.2461 | 0.3099  | 5815 |
| IB-FT                | 0.2275 | 0.2877  | 5815 |
| Ours Graphsignal 500 | 0.2468 | 0.3221  | 5815 |

### 4. SAFIM


| Method        | Language | Pass@1 | Pass@10 | N    |
| ------------- | -------- | ------ | ------- | ---- |
| Base Qwen     | python   | 0.1031 | 0.2075  | 805  |
| Base Qwen     | java     | 0.0056 | 0.0319  | 2479 |
| Base Qwen     | cpp      | 0.0123 | 0.0435  | 4968 |
| Base Qwen     | csharp   | 0.0000 | 0.0095  | 529  |
| TokenCleaning | python   | 0.1478 | 0.2037  | 805  |
| TokenCleaning | java     | 0.1444 | 0.2263  | 2479 |
| TokenCleaning | cpp      | 0.0835 | 0.1429  | 4968 |
| TokenCleaning | csharp   | 0.0851 | 0.1493  | 529  |
| XTF           | python   | 0.1441 | 0.2273  | 805  |
| XTF           | java     | 0.1702 | 0.2820  | 2479 |
| XTF           | cpp      | 0.0813 | 0.1562  | 4968 |
| XTF           | csharp   | 0.0699 | 0.1285  | 529  |
| LLM-CleanCode | python   | 0.0758 | 0.1652  | 805  |
| LLM-CleanCode | java     | 0.0254 | 0.0867  | 2479 |
| LLM-CleanCode | cpp      | 0.0276 | 0.0753  | 4968 |
| LLM-CleanCode | csharp   | 0.0095 | 0.0454  | 529  |
| CLEAR         | python   | 0.2186 | 0.3267  | 805  |
| CLEAR         | java     | 0.2525 | 0.3505  | 2479 |
| CLEAR         | cpp      | 0.1562 | 0.2285  | 4968 |
| CLEAR         | csharp   | 0.1758 | 0.2798  | 529  |
| IB-FT         | python   | 0.1354 | 0.2099  | 805  |
| IB-FT         | java     | 0.1404 | 0.2388  | 2479 |
| IB-FT         | cpp      | 0.0815 | 0.1371  | 4968 |
| IB-FT         | csharp   | 0.0851 | 0.1210  | 529  |

### 5. McEval


| Method        | Language | Pass@1 | Pass@10 | N  |
| ------------- | -------- | ------ | ------- | -- |
| Base Qwen     | c        | 0.0617 | 0.1605  | 81 |
| Base Qwen     | cpp      | 0.0366 | 0.1220  | 82 |
| Base Qwen     | csharp   | 0.0361 | 0.1205  | 83 |
| Base Qwen     | go       | 0.0125 | 0.1000  | 80 |
| Base Qwen     | java     | 0.0000 | 0.0625  | 96 |
| Base Qwen     | python   | 0.0787 | 0.1124  | 89 |
| TokenCleaning | c        | 0.3333 | 0.4815  | 81 |
| TokenCleaning | cpp      | 0.4024 | 0.5122  | 82 |
| TokenCleaning | csharp   | 0.5422 | 0.5783  | 83 |
| TokenCleaning | go       | 0.2750 | 0.3750  | 80 |
| TokenCleaning | java     | 0.5729 | 0.6771  | 96 |
| TokenCleaning | python   | 0.4157 | 0.6404  | 89 |
| XTF           | c        | 0.3951 | 0.4198  | 81 |
| XTF           | cpp      | 0.3293 | 0.4390  | 82 |
| XTF           | csharp   | 0.4096 | 0.5060  | 83 |
| XTF           | go       | 0.2875 | 0.3750  | 80 |
| XTF           | java     | 0.5521 | 0.6458  | 96 |
| XTF           | python   | 0.3820 | 0.5843  | 89 |
| LLM-CleanCode | c        | 0.1235 | 0.3086  | 81 |
| LLM-CleanCode | cpp      | 0.0854 | 0.2439  | 82 |
| LLM-CleanCode | csharp   | 0.2048 | 0.3855  | 83 |
| LLM-CleanCode | go       | 0.0250 | 0.1375  | 80 |
| LLM-CleanCode | java     | 0.3333 | 0.4896  | 96 |
| LLM-CleanCode | python   | 0.0899 | 0.1910  | 89 |
| CLEAR         | c        | 0.4938 | 0.6173  | 81 |
| CLEAR         | cpp      | 0.4634 | 0.6341  | 82 |
| CLEAR         | csharp   | 0.5422 | 0.6265  | 83 |
| CLEAR         | go       | 0.4250 | 0.5250  | 80 |
| CLEAR         | java     | 0.7083 | 0.8125  | 96 |
| CLEAR         | python   | 0.6180 | 0.7640  | 89 |
| IB-FT         | c        | 0.3210 | 0.3704  | 81 |
| IB-FT         | cpp      | 0.2195 | 0.3902  | 82 |
| IB-FT         | csharp   | 0.3976 | 0.4940  | 83 |
| IB-FT         | go       | 0.2125 | 0.2875  | 80 |
| IB-FT         | java     | 0.5000 | 0.6146  | 96 |
| IB-FT         | python   | 0.3820 | 0.6180  | 89 |

## 七、代码文件结构及作用

### 数据构建与格式对齐


| 文件                                                      | 主要作用                                                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `scripts/benchmark/build_chatml_fim_unified.py`           | 把 McEval-Instruct、HumanEval、McEval、SAFIM 统一成 canonical FIM，再渲染成 ChatML-FIM。   |
| `scripts/benchmark/sft_data_utils.py`                     | JSONL 读写、Qwen tokenizer 初始化、ChatML binarize、字段桥接和日志工具。                   |
| `scripts/benchmark/sft_data_convert.py`                   | 治理数据生成主入口；读取 rendered train，调用指定 governance operator，输出最终训练 JSON。 |
| `scripts/benchmark/sft_data_pipelines.py`                 | 各治理方法的 apply 层，负责把样本转成算子需要的格式，再转回训练格式。                      |
| `scripts/benchmark/sample_rendered_chatml_by_language.py` | 按语言抽样 rendered ChatML-FIM，便于检查数据格式和标注效果。                               |

### 治理算子


| 文件                                             | 主要作用                                                                          |
| ------------------------------------------------ | --------------------------------------------------------------------------------- |
| `scripts/benchmark/apply_governance_operator.py` | 治理算子的公共实现入口，包含旧 GraphSignal、TokenCleaning、XTF、CLEAR 等调用。    |
| `scripts/benchmark/governance_common.py`         | 多个治理算子共享的 label、device、文本抽取、tensor 转换工具。                     |
| `scripts/benchmark/graphsignal_annotations.py`   | 旧 GraphSignal 算子的 annotation 解析、teacher 标注和图重要性辅助逻辑。           |
| `scripts/benchmark/token_cleaning_operator.py`   | TokenCleaning：用 base/ref token loss 差异过滤 supervised token。                 |
| `scripts/benchmark/xtf_operator.py`              | XTF：用 RI、PCP、TR 三类 token 信号过滤 label。                                   |
| `scripts/benchmark/llm_cleaning_operator.py`     | LLM-CleanCode 相关逻辑；当前主要通过`sft_data_pipelines.py` 中的 apply 函数调用。 |
| `scripts/benchmark/clear_operator.py`            | CLEAR：样本过滤和候选修正逻辑。                                                   |
| `scripts/benchmark/ibft_loss.py`                 | IB-FT 的 variational bottleneck auxiliary loss。                                  |
| `scripts/benchmark/train_ibft.py`                | IB-FT baseline 的训练入口。                                                       |

### 评测与结果合并


| 文件                                        | 主要作用                                                                               |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| `scripts/benchmark/benchmark_eval.py`       | 评测主入口：加载模型、读取 eval、生成 greedy/sample candidates、调用 judge、写出结果。 |
| `scripts/benchmark/eval_generation.py`      | 构造与训练一致的手写 ChatML prompt，并封装 greedy / sampled generation。               |
| `scripts/benchmark/eval_judges.py`          | 解析`fim_prompt`、拼接完整代码，并根据 HumanEval/McEval/SAFIM 和语言运行 judge。       |
| `scripts/benchmark/eval_reporting.py`       | 聚合 pass@1/pass@10，并生成 CSV/Markdown 表格。                                        |
| `scripts/benchmark/merge_benchmark_eval.py` | 合并多 shard 评测结果，输出 merged JSON/CSV/Markdown。                                 |
| `scripts/benchmark/oracle_eval.py`          | 用 gold`fim_completion` 做 oracle 评测，检查数据拼接和 judge 上限。                    |

# Benchmark V2

## train data

benchmarkV2 数据路线：

- 数据获取：

  - train 使用 CodeSearchNet Go：`data/go_single/raw_data/codesearchnet`，原始规模约 317K。
  - test 只使用 MCEval Go：`data/go_single/raw_data/mceval`，原始规模约 400。不要额外构造 CodeSearchNet heldout eval，避免评测口径分散。
  - MCEval 的 `completion/single` 不是我们定义的 singleline/single-statement，它更接近 single mask/span，很多样本实际是多行或控制流片段。因此 test 侧不要直接相信原始 `[MASK]`，而是从 `prompt + canonical_solution` 中重新抽取干净的 single statement，并保留原始 `test / entry_point / task_id` 用于评测。
- 清洗过滤：

  - 文件级过滤：拒绝 `_test.go`，拒绝路径包含 `testdata/vendor/mock/mocks/fixture/fixtures/example/examples/generated`，拒绝 `package xxx_test`，拒绝 `import testing`、`testify`、`assert`、`require`，拒绝明显 generated file，例如 `Code generated`、`DO NOT EDIT`。
  - 注释必须严格过滤：函数内部不能有行注释、块注释、docstring 或注释残留；target 语句和 prefix/suffix 所在函数片段都不能包含注释。原因是后续 annotation、saliency 和数据治理依赖代码 token 关系，注释会显著干扰归因和治理。
  - 函数级过滤：只保留 `function_declaration` / `method_declaration`，函数体必须存在，函数行数和 token 长度控制在短代码范围内，拒绝函数名或内容包含 `Test/Benchmark/Example/assert/require/testcase/expected/actual`。
  - target 级过滤：只保留完整单行、完整单条 statement；类型限定为 assignment、return、call expression。拒绝 `if/for/switch/select` 等 block header，拒绝包含 `{` 或 `}` 的 target，拒绝空 return、`return nil`、`return err`、`i++`、`continue`、`break`、`panic` 等信息量低或控制流强的 target，拒绝 `fmt.Print`、`log.Print`、测试相关符号。
  - 去重与泄漏控制：对 `full_code`、函数片段、target 做 exact hash 和 normalized hash 去重；train 侧剔除与 MCEval eval 过近的样本，降低 train-test leakage 风险。
- singleline 构造：

  - 对 CodeSearchNet，解析 Go AST，从函数体中遍历候选 statement，选出合法 target；`prefix` 是同一函数片段中 target 前源码，`target` 是该 statement 原文，`suffix` 是 target 后源码，`full_code = prefix + target + suffix`。
  - 对 MCEval，先构造 `full_code = prompt + canonical_solution`，在 `entry_point` 对应函数内重新选择合法 single statement；`prefix/target/suffix` 由 AST byte span 切分，`judge_payload` 保留原始 `test / entry_point / task_id / raw_mceval`。
  - prefix/suffix 优先使用函数级片段而不是整文件，保证上下文足够、长度可控、annotation 更干净。
- 数据格式：

  - 第一层保存 canonical schema：`uid/source_dataset/split/language/task_type/prefix/target/suffix/full_code/target_kind/metadata/judge_payload`。这一层是数据真相层，不绑定 ChatML、tokenizer 或具体模型。
  - 第二层保存 rendered ChatML schema：保留 canonical 字段，并新增 `messages`。`system` 写 Go code completion assistant；`user` 给出带 `[MASK]` 的 Go 代码片段并要求只返回缺失代码；`assistant` 只存 `target`。
  - 后续 tokenized 训练格式再从 rendered ChatML 生成，`label != -100` 只落在 assistant target 区域，GraphSignal 的 annotation edge target 也应集中到 assistant target token。
- 评测方式：

  - 模型使用 Qwen2.5-Coder-7B-Instruct 系列，训练和推理都走 ChatML 格式。
  - 推理时读取 rendered ChatML eval，去掉 assistant message，让模型生成缺失 target。
  - MCEval-derived eval 的评测由我们显式拼接 `candidate_code = prefix + prediction + suffix`，再结合原始 `test` 运行 Go judge，统计 pass@1/pass@k 和 CodeBleu 。MCEval 官方评测思想和 judge 逻辑可以参考或复用，但插入位置必须由我们自己控制，因为 benchmarkV2 的 mask 是重新抽取的 single statement，不是 MCEval 原始 mask。
  - 先写 inspect/build 脚本输出候选统计、reject reason、target_kind 分布、长度分布和随机样本报告；确认样本质量后再生成正式 10K train 和 MCEval clean eval。

## Internal test data

Internal test 主要用于和当前 Go single-statement 训练任务保持同分布，核心目标不是追求外部 benchmark 权威性，而是回答：

```text
模型在我们自己的 Go function-level completion 数据分布上，completion 质量和 saliency alignment 是否提升？
```

### 1. 数据来源与定位

当前 internal test 使用 CodeSearchNet Go heldout 数据构造，原始数据来自：

```text
data/go_single/raw_data/codesearchnet/go/final/jsonl
```

构造后的 eval 数据放在：

```text
data/go_single/eval_data/
```

已构造过的代表性文件包括：

```text
data/go_single/eval_data/codesearchnet_go_valid_1000_chatml.jsonl
data/go_single/eval_data/codesearchnet_go_test_1000_chatml.jsonl
data/go_single/eval_data/codesearchnet_go_test_1000_graphsignal_500_compact.json
```

这里的 CodeSearchNet internal test 没有 unit test / judge payload，因此不能报告真正意义上的 pass@k。它适合报告：

```text
CodeBLEU
Exact Match@1/@k
Edit Similarity
parse/gofmt success rate
saliency Recall@10
saliency Precision@10
saliency mAP@10
```

其中 saliency 指标只对已经有 GraphSignal annotation edge 的样本计算。

### 2. 清洗与过滤

Internal test 沿用 train 侧 Go single-statement 构造逻辑，但 split 必须和训练数据分开：

- 拒绝 `_test.go`、`testdata/vendor/mock/fixture/example/generated` 等测试、mock、样例、生成代码。
- 拒绝 `package xxx_test`、`import testing`、`testify/assert/require` 等测试相关样本。
- 只保留 Go function / method 内部的单行、单 statement target。
- target 优先保留 assignment、return、call expression 等信息量较高的语句。
- 拒绝 `return nil`、`return err`、`break/continue/panic`、简单自增等弱信息 target。
- 对 `full_code`、函数片段、target 做 exact / normalized 去重。

Internal test 的目标是形成干净、可解释、和训练任务一致的 Go completion 分布；它不是外部论文 benchmark，因此结果应该标注为 internal data。

### 3. 格式转换

保持两层格式：

```text
canonical schema:
uid/source_dataset/split/language/task_type/prefix/target/suffix/full_code/metadata/judge_payload
```

```text
rendered ChatML schema:
uid/source_dataset/split/language/messages/prefix/target/suffix/full_code/metadata/judge_payload
```

对模型推理，`messages` 只保留 system + user，不喂 assistant target。user prompt 使用当前 Go completion 模板：

```text
Fill the [MASK] in the Go function. Return only the missing Go code.
```

### 4. 评测方式

Internal test 的 completion 质量不报告 pass@k，除非样本明确带有可执行 judge。当前 CodeSearchNet internal test 的推荐主表：


| Metric              | 含义                                                                     |
| ------------------- | ------------------------------------------------------------------------ |
| `CodeBLEU`          | 与 gold target 的代码相似度。                                            |
| `Exact Match@1/@10` | 预测 target 规范化后是否与 gold 完全一致。                               |
| `Edit Similarity`   | 基于编辑距离的字符串相似度。                                             |
| `Parse/gofmt rate`  | 预测拼回`prefix + prediction + suffix` 后是否能被 Go parser/gofmt 接受。 |
| `Recall@10`         | saliency top-10 是否覆盖 annotation source。                             |
| `Precision@10`      | saliency top-10 中 annotation source 占比。                              |
| `mAP@10`            | annotation source 在 saliency 排名中的平均精度。                         |

需要注意：Exact Match 不是 pass@k。CodeSearchNet 没有 unit test，不能用“至少一个生成能通过测试”定义 pass@k。

## External test benchmark

External test 主要用于补足外部权威 benchmark。这里不要强行要求 Go 语言；benchmark 的语言由 benchmark 本身决定。当前优先考虑三个 completion 领域常用 benchmark：

```text
CodeXGLUE Code Completion
RepoBench
CrossCodeEval
```

### 1. 总览


| Benchmark                 | 语言                         | 任务形态                              | 是否有 unit test | 推荐报告指标                             | 优先级 |
| ------------------------- | ---------------------------- | ------------------------------------- | ---------------- | ---------------------------------------- | ------ |
| CodeXGLUE Code Completion | Python, Java                 | token-level / line-level completion   | 否               | EM, Edit Similarity                      | S      |
| RepoBench                 | Python, Java                 | repository-level next-line completion | 否               | EM, Edit Similarity, CodeBLEU            | A      |
| CrossCodeEval             | Python, Java, TypeScript, C# | cross-file code completion            | 否               | EM, Edit Similarity, Identifier Match/F1 | A      |

这三个 benchmark 都不是 execution benchmark，因此默认不报告 pass@k。pass@k 只适合 HumanEval、McEval、SAFIM、xCodeEval 这类带 unit test 或 judge payload 的数据。

### 2. CodeXGLUE Code Completion

CodeXGLUE completion 是传统 code completion 最稳的公开基准之一。completion 主体包含：

```text
PY150
GitHub Java Corpus
```

任务包括 token-level 和 line-level。对于 Qwen-Instruct 这类生成式模型，优先使用 line-level completion。

数据获取：

```text
data/go_single/raw_data/codexglue/CodeXGLUE
```

推荐只 sparse checkout completion 相关目录：

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/microsoft/CodeXGLUE \
  data/go_single/raw_data/codexglue/CodeXGLUE

cd data/go_single/raw_data/codexglue/CodeXGLUE
git sparse-checkout set Code-Code/CodeCompletion-line Code-Code/CodeCompletion-token
```

处理原则：

- 不修改官方 `test.json` / `answers.json`。
- 只生成我们自己的 rendered ChatML eval 文件。
- 保留官方 raw id，便于 prediction 和 answer 对齐。
- 对 prompt 超长样本单独记录 skip reason，不在原始 benchmark 文件上改动。

建议输出：

```text
data/go_single/eval_data/codexglue_py150_line_chatml.jsonl
data/go_single/eval_data/codexglue_java_line_chatml.jsonl
outputs/go_single/codexglue/predictions/
outputs/go_single/codexglue/results/
```

评测：

- 优先调用 CodeXGLUE 官方 evaluator。
- line-level completion 主报告 EM 和 Edit Similarity。
- CodeBLEU 可作为补充，但不要替代官方指标。

### 3. RepoBench

RepoBench 是 ICLR 2024 repository-level code auto-completion benchmark，更接近真实 IDE 场景。支持语言：

```text
Python
Java
```

主要 setting：


| Setting             | 含义                                   |
| ------------------- | -------------------------------------- |
| `in_file`           | 预测不依赖跨文件信息的当前文件下一行。 |
| `cross_file_first`  | mask 第一次使用跨文件模块的行。        |
| `cross_file_random` | mask 随机跨文件依赖行。                |

数据获取：

```text
tianyang/repobench_python_v1.1
tianyang/repobench_java_v1.1
```

同时 clone 官方 repo，保留 `eval.py`：

```text
data/go_single/raw_data/repobench_repo
```

处理方式：

- 使用 HuggingFace `datasets.load_dataset` 拉取 Python/Java v1.1。
- 原始 dataset 保存在 `data/go_single/raw_data/repobench/`。
- 不改官方字段，只生成我们的 ChatML rendered copy。
- prompt 由 cross-file context、import statement、cropped code 组成。
- target 是官方 `next_line`。
- 生成时只取下一行，避免模型继续生成后续多行污染 EM。

建议输出：

```text
data/go_single/eval_data/repobench_python_chatml.jsonl
data/go_single/eval_data/repobench_java_chatml.jsonl
outputs/go_single/repobench/predictions/
outputs/go_single/repobench/results/
```

评测：

- 调用 RepoBench 官方 `eval.py`。
- 按 language / setting / level 聚合。
- 主报告 EM、Edit Similarity、CodeBLEU。
- 不报告 pass@k，因为 RepoBench 没有 unit test。

今晚优先级：先跑 RepoBench Python subset，每个 setting 各取 200-500 条，确认 pipeline 后再扩到 Java 和全量。

### 4. CrossCodeEval

CrossCodeEval 是 NeurIPS 2023 Datasets & Benchmarks 的 cross-file code completion benchmark。支持语言：

```text
Python
Java
TypeScript
C#
```

它的重点是：样本需要跨文件上下文才能完成，适合评估模型是否能利用 retrieved context。

官方 setting 包括：


| Setting                      | 含义                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------- |
| in-file only                 | 只给当前文件上下文。                                                         |
| retrieved cross-file context | 给检索到的跨文件上下文。                                                     |
| retrieval with reference     | 使用 reference 辅助检索，主要作为 retrieval 上界，不适合作为真实模型主结果。 |

数据获取：

```text
data/go_single/raw_data/crosscodeeval_repo
```

处理方式：

- clone `amazon-science/cceval`。
- 解压官方 crosscodeeval 数据。
- 不修改官方 prompt / reference。
- 生成我们的 ChatML rendered copy。
- prompt 保留官方 in-file context 和 retrieved cross-file context。
- target 使用官方 reference completion。

建议输出：

```text
data/go_single/eval_data/crosscodeeval_python_chatml.jsonl
data/go_single/eval_data/crosscodeeval_java_chatml.jsonl
data/go_single/eval_data/crosscodeeval_typescript_chatml.jsonl
data/go_single/eval_data/crosscodeeval_csharp_chatml.jsonl
outputs/go_single/crosscodeeval/predictions/
outputs/go_single/crosscodeeval/results/
```

评测：

- 优先调用 CrossCodeEval 官方评测脚本。
- 主报告 code match：EM、Edit Similarity。
- 同时报告 identifier match：Identifier Precision / Recall / F1。
- 不报告 pass@k，因为 CrossCodeEval 没有 unit test。

今晚优先级：先跑 Python，必要时再加 Java；TypeScript/C# 依赖 parser/tokenizer 环境更多，放到后续补齐。

### 5. 当前推荐实验顺序

在时间紧张的情况下，推荐顺序：

```text
1. Internal CodeSearchNet-Go: CodeBLEU + saliency metrics
2. CodeXGLUE PY150 line-level: EM + Edit Similarity
3. RepoBench Python subset: EM + Edit Similarity + CodeBLEU
4. CrossCodeEval Python retrieved-context: EM + ES + Identifier F1
```

最终报告中应明确区分：

```text
Internal data:
  同分布 Go completion，主要看 CodeBLEU 和 saliency alignment。

External benchmark:
  权威 completion benchmark，主要看 EM / ES / CodeBLEU / Identifier F1。

Execution benchmark:
  只有带 judge/unit test 的数据才能报告 pass@k。
```
