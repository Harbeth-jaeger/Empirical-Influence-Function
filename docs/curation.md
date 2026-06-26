# 数据治理与模型训练

本文只整理 ours causality curation 路线的算法逻辑和代码结构。这里 causality curation 是我们的方法名，ours 表示“我们的方法”。本文关注标注如何产生、saliency 如何定义、loss 如何设计，以及这些信息如何进入训练。这里的 curation 不是简单清洗样本，而是把“哪些 token 关系应该被模型重视”显式写进训练监督。

## 一、整体数学思路与算法链路

causality curation 的训练链路可以概括为：

```text
原始 SFT / ChatML-FIM 数据
  -> 代码片段级 tokenization
  -> tree-sitter 结构边标注
  -> LLM 语义边补充
  -> word-level edge 映射到 Qwen BPE token edge
  -> AnnotatedSFTDataset 读入 input_ids / labels / annot_pairs
  -> next-token loss + saliency loss 联合训练
```

其中最关键的设计是把训练样本从普通序列：

```text
x = (x_1, x_2, ..., x_T)
```

扩展为带图约束的序列：

```text
(x, E),  E = {(i, j, r)}
```

一条边 `(i, j, r)` 表示第 `i` 个 token 是预测或理解第 `j` 个 token 的重要线索，`r` 是关系类型。训练时不会改变原始 ChatML 训练序列本身，而是在 loss 中额外要求这些边对应的 token 贡献更强。

## 二、相关文件结构及其作用

这一节只列和 causality curation 标注、数据构造、训练诊断、评测可视化强相关的代码。整体上可以按六条线理解：

```text
数据处理 scripts/data_process, scripts/go_singleline_fim_exp
  -> 标注 src/annotate, tools/viz_annotation, scripts/go_singleline_fim_exp/annotate_*.py
  -> 训练 src/train
  -> 小样本 saliency 实验 scripts/saliency_exp
  -> 训练前后 saliency 对齐可视化 tools/viz_saliency
  -> benchmark / in-domain 错误案例可视化 tools/viz_failure
```

### 1. `src/annotate/`：标注核心库


| 文件                      | 主要作用                                                                                                                                                   |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `utils.py`                | 定义`SubwordToken`、`TokenCorrelation`；实现代码级 tokenizer、Qwen BPE tokenizer、simple-token 到 BPE-token 的字符区间映射；也包含 FIM 边方向归一化逻辑。  |
| `neural_annot.py`         | 标注核心。包含 tree-sitter 结构边提取器和 tool-calling LLM agent；结构边负责`bracket/defuse/call/return/type`，LLM 负责补充 `dataflow/semantic/api` 等边。 |
| `postprocessing.py`       | 将 MCEval / SFT 风格标注结果重建为 ChatML 序列，完成 word edge 到 Qwen BPE edge 的映射，并生成可视化数据。                                                 |
| `postprocessing_safim.py` | SAFIM 数据格式的后处理版本，逻辑与`postprocessing.py` 类似，但重建代码和 mask span 的方式不同。                                                            |
| `main.py`                 | MCEval 风格数据的标注入口，串联 tokenization、annotator、保存原始标注。                                                                                    |
| `main_safim.py`           | SAFIM 风格数据的标注入口。                                                                                                                                 |
| `viz_utils.py`            | 将 token、边和 attention/top-k 信息组织成可视化页面需要的结构。                                                                                            |
| `web_search.py`           | 可选 API 文档检索工具，供 LLM 判断 API 语义边时使用。                                                                                                      |

### 2. `src/train/`：训练数据、loss 与 Trainer


| 文件          | 主要作用                                                                                                              |
| ------------- | --------------------------------------------------------------------------------------------------------------------- |
| `dataset.py`  | 读取 compact benchmark format 或 legacy annotated format，统一转换成`input_ids/labels/annot_pairs`。                  |
| `loss.py`     | 实现最后一层 contribution saliency、dense/sparse 两种 saliency loss 计算，以及训练日志诊断指标。                      |
| `train.py`    | 自定义 HuggingFace Trainer，组合 next-token CE loss 和 saliency loss，加载 Qwen、LoRA、dataset、callback 并执行训练。 |
| `attn_viz.py` | 训练期间抽取部分样本 attention，用于观察模型关注位置。                                                                |

### 3. `src/scripts/`


| 文件       | 主要作用                                                                                                                         |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `train.sh` | 官方训练脚本模板，展示多卡 Qwen causality curation SFT 的默认参数组织方式；实际 benchmark 可以按数据路径、GPU 数、输出目录调整。 |

### 4. `scripts/data_process/`：通用 CodeSearchNet 数据处理


| 文件                                  | 主要作用                                                                                                                                                                                           |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `build_codesearchnet_single.py`       | 多语言 CodeSearchNet single-statement 构造脚本。对 Python/Java/C/C++/C# 等语言做过滤、抽取 prefix/target/suffix、生成 canonical 与 ChatML 两份数据，输出到`data/<lang>_single/train_data/`。       |
| `build_codesearchnet_go_full_eval.py` | CodeSearchNet-Go valid/test 全量 eval 构造脚本。复用 GoSingle 的 AST/过滤逻辑，输出`codesearchnet_go_{split}_full_canonical.jsonl` 与 `codesearchnet_go_{split}_full_chatml.jsonl`，并写构造报告。 |

这一层的职责是“构造干净任务数据”，不负责 LLM 标注，也不负责训练。输出的 canonical schema 是数据真相层，ChatML schema 是模型输入层。

### 5. `scripts/go_singleline_fim_exp/`：Go single-statement 场景脚本


| 文件                                    | 主要作用                                                                                                                                                                                       |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `go_single_pipeline.py`                 | GoSingle 数据构造底层库。包含 CodeSearchNet/MCEval 读取、Go AST 候选 statement 抽取、过滤、去重、canonical sample 构造和 ChatML render。                                                       |
| `build_go_single_data.py`               | GoSingle benchmarkV2 数据构造入口，生成 train canonical/chatml 与 eval canonical/chatml，并输出 build report 和 samples report。                                                               |
| `annotate_chatml_with_src_annotate.py`  | GoSingle ChatML 标注入口。读取 ChatML-FIM 样本，调用`src/annotate` 结构边与 LLM 补边，映射到 Qwen BPE compact 格式；支持 cache、seed compact、oneshot/agent 模式。                             |
| `export_src_annotate_cache_preview.py`  | 从`.annotation_cache.jsonl` 导出已完成样本的 compact preview 和对齐 raw preview，用于不中断长标注进程时做可视化抽检。                                                                          |
| `evaluate_go_single_predictions.py`     | MCEval-derived GoSingle 评测脚本。把模型 prediction 回填为`prefix + prediction + suffix`，运行 Go judge，统计 pass@1/pass@k，可选 CodeBLEU。                                                   |
| `oracle_eval_go_single.py`              | 用 gold target 做 oracle judge，用于检查 eval 数据、judge payload、拼接逻辑是否自洽。                                                                                                          |
| `evaluate_codesearchnet_go_internal.py` | CodeSearchNet-Go internal eval。由于没有 unit test，completion 指标是 normalized exact match / CodeBLEU；同时可在 teacher forcing 下评估 saliency 与 annotation edge 的 Recall/Precision/mAP。 |
| `overfit_multi_target.py`               | GoSingle 对`scripts/saliency_exp/overfit_multi_target.py` 的场景封装，用于单样本多 target saliency overfit。                                                                                   |

GoSingle 目录是当前项目最直接的工程主线：数据构造、标注、oracle 验证、in-domain eval、saliency alignment eval 都在这里串起来。

### 6. `tools/viz_annotation/`：标注边质量可视化


| 文件                                  | 主要作用                                                                                                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `annotate_benchmark_fim_train.py`     | benchmark ChatML-FIM 训练数据标注脚本。输入 rendered ChatML-FIM JSONL，输出训练/viewer 共用的 compact JSONL；支持 structural/oneshot/agent、resume、row cache、静态 HTML 抽检。 |
| `find_annotation_rich_samples.py`     | 从 raw + compact edge 数据里挑选 annotation 边丰富的样本，输出 CSV，常用于挑 viewer 展示样本。                                                                                  |
| `build_dynamic_annotation_viewer.py`  | 构建动态 annotation viewer。可点击 token 查看入边/出边，按边类型过滤，适合检查 causality curation 标注是否合理。                                                                |
| `visualize_annotation_edges.py`       | 对指定 sample 生成独立静态 HTML，快速看 tokenizer-aligned annotation edges。                                                                                                    |
| `build_annotation_showcase_viewer.py` | 自动筛选 annotation-rich 样本并构建 showcase viewer，强调 edge density、prompt-to-completion target、边类型多样性。                                                             |
| `build_attention_report.py`           | 对模型跑 attention/saliency report，并把 annotation parents 标到 completion target 上，用于比较模型 attention top-k 与 annotation source 是否一致。                             |
| `dynamic_annotation_viewer.html`      | 动态 viewer 模板，脚本会把 viewer data 嵌入其中。                                                                                                                               |

这个目录用于回答：“标注本身有没有问题？”重点看边方向、边类型、prompt/completion 区域、tokenizer 映射是否对齐。

### 7. `tools/viz_failure/`：错误案例与 benchmark 失败分析


| 文件                                    | 主要作用                                                                                                                     |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `eval_dump.py`                          | 运行 benchmark generation/judge，并把每个样本的 prompt、target、prediction、judge 结果 dump 成 JSONL，作为错误分析基础数据。 |
| `build_comparison_data.py`              | 合并两个模型的 eval dump，按 pass@1/pass@k 关系分类，例如 ours pass/CLEAR fail、CLEAR pass/ours fail、both fail 等。         |
| `build_failure_viewer.py`               | 构建 ours-vs-CLEAR failure analysis 的静态 HTML viewer。                                                                     |
| `build_failure_viewer_v2.py`            | failure viewer v2，支持嵌入 ours saliency 数据，点击失败样本时能同时看预测差异和 saliency 状态。                             |
| `build_saliency_manifest.py`            | 从 failure viewer 数据生成 saliency manifest，供后续 saliency 计算脚本读取。                                                 |
| `build_dual_model_saliency_manifest.py` | 构建双模型预测/saliency viewer 的 sample manifest。                                                                          |
| `compute_dual_model_saliency.py`        | 对两个模型的 selected samples 计算 clickable saliency，支持 cache、partial write，用于失败案例归因。                         |
| `build_dual_model_saliency_viewer.py`   | 构建双模型 prediction + saliency 对比 viewer，可点击 completion token 查看两个模型的 top-k source tokens。                   |

这个目录用于回答：“模型为什么在 in-domain test 或外部 benchmark 上失败？”它关注 prediction、judge、错误类别和 saliency 解释，不直接参与训练。

### 8. `tools/viz_saliency/`：训练前后 saliency 与 annotation 对齐


| 文件                                                                                     | 主要作用                                                                                                                                                                  |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compute_teacher_forcing_annotation_saliency.py`                                         | 在固定 ground-truth completion 的 teacher-forcing 条件下，分别计算 Base 与 SFT/ours 的 saliency，并与 annotation edges 对齐；这是检查训练是否真的改变内部归因的核心脚本。 |
| `analyze_row_saliency_alignment.py`                                                      | 对单个 row 的 saliency/annotation alignment 做统计分析，输出 JSON/Markdown，例如 Recall@k、Precision@k、mAP 等。                                                          |
| `build_base_vs_ours_annotation_viewer.py`                                                | 构建 Base-vs-ours annotation alignment viewer，左右对比训练前后模型 saliency top-k 是否更贴近 annotation source。                                                         |
| `build_rich_annotation_saliency_data.py`                                                 | 将 annotation-rich samples 与 Base/ours saliency 合并成 viewer 数据。                                                                                                     |
| `build_saliency_comparison_viewer.py`                                                    | 构建多模型/多样本 saliency comparison viewer。                                                                                                                            |
| `compute_model_saliency.py`                                                              | 预计算 free-run ALTI saliency top-k，用于动态 viewer；适合看模型生成过程中的 saliency，而不是固定答案 teacher forcing。                                                   |
| `select_saliency_viz_samples.py`                                                         | 为 saliency viewer 选择样本 manifest。                                                                                                                                    |
| `overfit_go_single_row_saliency.py`                                                      | 对单个 GoSingle row 做 saliency overfit，并报告 alignment dynamics。                                                                                                      |
| `fit_single_saliency_showcase.py`                                                        | 演示型工具：选一个 annotation-rich sample，用小 LoRA 适配器拟合 saliency loss，并追加到展示 viewer。                                                                      |
| `export_intervention_saliency_format.py`                                                 | 把当前 visual-saliency 数据转成 legacy intervention experiment 的`latest_saliency.json` 风格。                                                                            |
| `export_legacy_saliency_by_sample.py` / `export_legacy_saliency_all_models_by_sample.py` | 按 sample/model 导出 legacy saliency 文件，方便和旧可视化或干预实验代码对接。                                                                                             |

这个目录用于回答：“训练以后，模型 saliency 分布是否真的更靠近 annotation label？”其中 teacher-forcing alignment 是训练目标的直接诊断，free-run saliency 更接近真实生成行为诊断。

### 9. `scripts/saliency_exp/`：小规模 saliency 实验与梯度诊断


| 文件                                   | 主要作用                                                                                                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `select_single_target.py`              | 从 annotated train data 中选择一个适合 overfit 的样本和 target token，要求存在严格 causal annotation source。                    |
| `overfit_single_target.py`             | 单样本、单 completion target 的 saliency overfit 实验，用于验证 loss 能否把指定 source saliency 拉高。                           |
| `overfit_multi_target.py`              | 单样本、多 completion target 的纯 saliency overfit 实验，更接近真实 causality curation 训练目标。                                |
| `base_saliency_distribution.py`        | 在 base model 上扫描 annotated samples，统计初始 teacher-forcing saliency 分布、positive/negative 分布和 floor 相关统计。        |
| `estimate_saliency_floor_quantiles.py` | 估计 saliency floor quantile，用于旧版 InfoNCE/floor 方案的超参分析。                                                            |
| `probe_saliency_autograd.py`           | 单样本 autograd 健康检查：确认 saliency tensor 是否连在计算图上，比较 CE 与 saliency 梯度范数和 cosine，检查 LoRA 权重读取问题。 |
| `trace_row_token_training.py`          | 带 trace 的小规模训练入口，按 step 记录 row/token 级 saliency、loss、梯度和 top-k 变化。                                         |
| `render_row_token_trace_report.py`     | 将`trace.jsonl` 渲染成更易读的 Markdown/图表报告。                                                                               |

这个目录用于回答：“loss 的数学设计和梯度是否真的能工作？”它比正式训练更小、更可控，适合在大规模训练前排查 OOM、梯度断开、CE/saliency loss 打架、floor/margin 超参问题。

## 三、调试阶段实验所用数据的获取与处理

这一章讨论 go single-line fim 作为调试、探索阶段的主实验，所用的数据及其相关处理

把数据分成三层：train data、internal test data、external test benchmark。三者不要混用，因为它们回答的问题不同。

由于是 Qwen 系统模型作为基模，我们希望实现的效果是，ChatML 格式用于训练和推理，至于 ChatML 的内容内核来自 FIM 场景，即 [MASK] 的 prefix 和 suffix，举例如下：

```
<|im_start|>system
You are a Go code completion assistant.<|im_end|>
<|im_start|>user
Fill the [MASK] in the Go function. Return only the missing Go code, without Markdown fences or explanation.

* Incomplete Code:
func (w *Window) GetFrameSize() (left, top, right, bottom int) {
	var l, t, r, b C.int
	C.glfwGetWindowFrameSize(w.data, &l, &t, &r, &b)
[MASK]
	return int(l), int(t), int(r), int(b)
}<|im_end|>
<|im_start|>assistant
	panicError()<|im_end|>
```

具体而言，每个数据集的作用不同：

```text
train data
  用来训练 causality curation：需要 ChatML-FIM 输入、assistant completion label、annotation edges。

internal test data
  用来检查同分布收益：可以有 annotation，因此能同时看 saliency alignment 和少量 completion quality。

external test benchmark
  用来报告外部主结果：不依赖我们的 annotation，只关注 completion quality，主指标必须包含 pass@k。
```

下文即围绕这个思路展开

### 1. 为什么同时需要 internal 和 external

Internal test 和 external test 的分工如下。


| 数据层        | 核心问题                                                                                | 是否需要标注 | 主指标                                                                              | 作用                                                  |
| ------------- | --------------------------------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------- | ----------------------------------------------------- |
| train data    | 模型从哪些 token 关系中学习 causality curation 监督？                                   | 需要         | train loss、saliency loss 诊断                                                      | 训练本体。                                            |
| internal test | 在和训练同分布的 Go single-statement completion 上，模型内部归因是否更贴近 annotation？ | 需要         | saliency Recall/Precision/mAP，辅以 CodeBLEU、EM、edit similarity、parse/gofmt rate | 证明 causality curation 真的改变了模型内部 saliency。 |
| external test | 在公认 benchmark 上，模型 completion 能力是否提升？                                     | 不需要       | pass@k，辅以 CodeBLEU/EM/edit similarity                                            | 对外报告模型能力，避免只在自建数据上自洽。            |

因此，internal test 不能替代 external benchmark：internal 更适合解释“为什么模型变了”，external 更适合回答“模型是否真的更会写代码”。反过来，external benchmark 通常没有我们的 annotation，也不能用来直接计算 saliency alignment。

### 2. Train Data：训练数据

当前训练数据来自 CodeSearchNet-Go，任务被统一成 Go single-statement FIM completion。

原始数据位置：

```text
data/go_single/raw_data/codesearchnet/go/final/jsonl/train
```

推荐输出位置：

```text
data/go_single/train_data/
```

训练数据至少需要三份逻辑格式。


| 格式              | 作用                                   | 典型字段                                                                                |
| ----------------- | -------------------------------------- | --------------------------------------------------------------------------------------- |
| canonical         | 数据真相层，不绑定模型模板。           | `uid/source_dataset/split/language/prefix/target/suffix/full_code/target_kind/metadata` |
| ChatML-FIM        | 模型输入层，用于 SFT 和标注。          | `messages/prefix/target/suffix/full_code`                                               |
| compact annotated | 训练层，已经 tokenized 并映射 BPE 边。 | `input_ids/label/attention_edges/language/annotation_meta`                              |

清洗过滤原则：

- 文件级过滤：拒绝 `_test.go`，拒绝路径包含 `testdata/vendor/mock/mocks/fixture/fixtures/example/examples/generated`，拒绝 `package xxx_test`，拒绝 `import testing`、`testify`、`assert`、`require`，拒绝明显 generated file，例如 `Code generated`、`DO NOT EDIT`。
- 注释过滤：函数内部、target 语句、prefix/suffix 片段都不能包含普通注释、块注释、docstring 或注释残留。annotation 和 saliency 监督依赖代码 token 关系，注释会显著污染归因。
- 函数级过滤：只保留 `function_declaration` / `method_declaration`，函数体必须存在，函数长度和 token 长度控制在短代码范围内，拒绝函数名或内容包含 `Test/Benchmark/Example/assert/require/testcase/expected/actual`。
- target 级过滤：只保留完整单行、完整单条 statement；优先 assignment、return、call expression。拒绝 `if/for/switch/select` 等 block header，拒绝包含 `{` 或 `}` 的 target，拒绝 `return nil`、`return err`、`i++`、`continue`、`break`、`panic` 等信息量低或控制流强的 target。
- 去重与泄漏控制：对 `full_code`、函数片段、target 做 exact hash 和 normalized hash 去重；train 侧剔除与 eval 过近的样本，降低 train-test leakage 风险。

单条样本构造逻辑：

```text
full_code = prefix + target + suffix
prompt/context = prefix + [MASK] + suffix
completion = target
```

`prefix` 和 `suffix` 优先使用函数级片段，而不是整文件。这样上下文足够、长度可控，annotation 也更干净。

训练时，`label != -100` 只落在 assistant completion 区域；annotation edge 的 target 也应主要集中在 completion token，尤其是 `prompt -> completion` 边。

### 3. Internal Test Data：同分布诊断集

Internal test 使用和训练同源、同构造逻辑的 Go single-statement heldout 数据。它的定位是“训练目标诊断集”，不是外部 benchmark。

原始数据位置：

```text
data/go_single/raw_data/codesearchnet/go/final/jsonl/valid
data/go_single/raw_data/codesearchnet/go/final/jsonl/test
```

推荐输出位置：

```text
data/go_single/eval_data/
```

代表性文件包括：

```text
data/go_single/eval_data/codesearchnet_go_valid_1000_chatml.jsonl
data/go_single/eval_data/codesearchnet_go_test_1000_chatml.jsonl
data/go_single/eval_data/codesearchnet_go_test_1000_graphsignal_500_compact.json
```

这里最后一个文件名里的 `graphsignal` 是历史文件名，不代表当前方法名；正文统一称为 causality curation。

Internal test 的处理原则：

- 复用 train data 的 Go AST 抽取、过滤、去重和 ChatML-FIM render 逻辑。
- split 必须和训练数据分开，不能从 train 样本里抽 internal test。
- 每个原始函数最多构造一个 eval sample，避免一个函数被多个 mask 反复评测，导致样本数虚高。
- 可以做 annotation，因为 internal test 需要检查 saliency 是否和 annotation label 对齐。
- 由于 CodeSearchNet 没有 unit test / judge payload，不能报告真正意义上的 pass@k。

Internal test 推荐报告两类指标。

Saliency alignment 指标：


| Metric                         | 含义                                                    |
| ------------------------------ | ------------------------------------------------------- |
| `Recall@k`                     | annotation source 是否被模型 saliency top-k 覆盖。      |
| `Precision@k`                  | saliency top-k 中 annotation source 的占比。            |
| `mAP@k`                        | annotation source 在 saliency 排名中的平均精度。        |
| positive/negative saliency gap | annotation source 与 non-annotation source 的贡献差距。 |

Completion quality 辅助指标：


| Metric                     | 含义                                                           |
| -------------------------- | -------------------------------------------------------------- |
| `CodeBLEU`                 | 预测 target 与 gold target 的代码相似度。                      |
| `Exact Match@1/@k`         | 规范化后是否和 gold target 完全一致。                          |
| `Edit Similarity`          | 基于编辑距离的字符串相似度。                                   |
| `parse/gofmt success rate` | `prefix + prediction + suffix` 是否能被 Go parser/gofmt 接受。 |

Internal test 的主叙事应是：causality curation 是否让模型在同分布任务上更关注 annotation source。CodeBLEU/EM 只能作为 completion quality 辅助观察，不能替代 pass@k。

### 4. External Test Benchmark：外部主评测

External test 用来报告外部 benchmark 主结果，应该尽量选择带 unit test 或 judge payload 的 benchmark。原因很简单：代码生成最主流、最有说服力的指标是 pass@k，它要求每个任务能判断生成代码是否真的通过测试。

External benchmark 的原则：

- 不改 benchmark 原始数据和官方 test；raw benchmark 只放在 `data/raw_data/`，不直接覆盖或重写。
- 可以从 raw benchmark 派生我们的 ChatML/FIM 推理输入，保留官方 id，方便 prediction 回填。
- 不做 causality curation annotation，也不报告 saliency alignment。
- 主报告 completion quality，尤其是 pass@k。
- 如果 benchmark 没有 unit test，则只能作为补充相似度评测，不能作为 external 主结果。

推荐 external 主表优先使用 execution benchmark。


| Benchmark                           | 语言        | 是否有 unit test / judge | 推荐主指标    | 备注                                                                                                                     |
| ----------------------------------- | ----------- | ------------------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| MCEval-Go / Go subset               | Go          | 有                       | pass@1/pass@k | 和当前 Go completion 场景最接近，适合短期主结果。                                                                        |
| MultiPL-E Go body-mask derived eval | Go / 多语言 | 有                       | pass@1/pass@k | MultiPL-E HF 数据不含 Go canonical solution，当前先构造函数体`[MASK]` 评测样本；论文中必须说明不是官方 generation 协议。 |
| HumanEval-X Go derived FIM          | Go / 多语言 | 有                       | pass@1/pass@k | 从 HumanEval-X Go 样本派生单语句 FIM；保留官方 test，使用自建兼容 judge。                                                |
| xCodeEval-Go / ExecEval             | Go / 多语言 | 有                       | pass@1/pass@k | 更标准的可执行代码评测，适合外部 benchmark。                                                                             |
| HumanEval / MBPP 类 benchmark       | Python 为主 | 有                       | pass@1/pass@k | 权威但语言和当前 Go 训练不完全一致，可作补充。                                                                           |

没有 unit test 的 completion benchmark 可以作为 external supplementary，不建议放在主表。


| Benchmark                 | 语言                         | 是否有 unit test | 可报指标                           | 定位                         |
| ------------------------- | ---------------------------- | ---------------- | ---------------------------------- | ---------------------------- |
| CodeXGLUE Code Completion | Python, Java                 | 否               | EM、Edit Similarity、CodeBLEU      | 传统 completion 相似度补充。 |
| RepoBench                 | Python, Java                 | 否               | EM、Edit Similarity、CodeBLEU      | repo-level completion 补充。 |
| CrossCodeEval             | Python, Java, TypeScript, C# | 否               | EM、Edit Similarity、Identifier F1 | cross-file completion 补充。 |

External pass@k 的计算逻辑：

```text
对每个 task 生成 k 个 candidate。
只要至少 1 个 candidate 回填到上下文后通过官方 judge/unit test，则该 task pass@k = 1。
最终 pass@k = 通过任务数 / 总任务数。
```

对于 FIM/single-statement 任务，评测时必须显式拼接：

```text
candidate_code = prefix + prediction + suffix
```

然后把 `candidate_code` 和 benchmark 原始 `test/judge_payload` 一起交给官方或兼容 judge。不能用字段精确匹配冒充 pass@k。

#### 当前外部 benchmark 来源

当前短期外部评测优先使用两个权威多语言 execution benchmark，并只在实验协议层面派生 Go `[MASK]` 任务；其中 HumanEval-X 可派生 single-statement FIM，MultiPL-E 当前只能派生 body-mask eval。

1. **MultiPL-E**

   - 官网 / 文档页：`https://nuprl.github.io/MultiPL-E/`
   - GitHub：`https://github.com/nuprl/MultiPL-E`
   - 论文：`https://arxiv.org/pdf/2208.08227`
   - 数据来源：Hugging Face dataset `nuprl/MultiPL-E`
   - 原始定位：将 HumanEval 和 MBPP 翻译到多种编程语言，用 unit tests 评估多语言 code generation。
2. **HumanEval-X**

   - CodeGeeX GitHub：`https://github.com/zai-org/CodeGeeX`
   - evaluator 说明：`https://github.com/open-compass/code-evaluator`
   - 论文页：`https://huggingface.co/papers/2303.17568`
   - arXiv：`https://arxiv.org/abs/2303.17568`
   - 数据来源：Hugging Face dataset `zai-org/humaneval-x`
   - 原始定位：基于 HumanEval 构建的多语言 benchmark，包含 Python、C++、Java、JavaScript、Go，每个样本带 solution 和 test cases。

原始数据下载到：

```text
data/raw_data/multipl_e/
data/raw_data/humaneval_x/
```

推荐下载命令为：

```bash
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function

export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

mkdir -p data/raw_data/multipl_e
hf download nuprl/MultiPL-E \
  --repo-type dataset \
  --include "humaneval-*/*.parquet" \
  --include "mbpp-*/*.parquet" \
  --local-dir data/raw_data/multipl_e

mkdir -p data/raw_data/humaneval_x
hf download zai-org/humaneval-x \
  --repo-type dataset \
  --include "data/*/data/*.jsonl" \
  --local-dir data/raw_data/humaneval_x
```

这里下载的是 Hugging Face dataset 中的数据文件，不是整个 GitHub 仓库。MultiPL-E 的 `.parquet` 是 Hugging Face 常用的列式数据格式，可以直接用 `pandas.read_parquet` 读取，也可以转换为 JSONL。

#### Derived Go FIM 协议

由于 MultiPL-E 和 HumanEval-X 的官方协议主要是 unit-test-driven code generation，而我们的训练任务是 Go single-statement FIM completion，因此外部 benchmark 不直接按官方 generation prompt 评测，而是构造成 derived benchmark：

```text
benchmark raw sample
  -> 抽取 prompt / declaration / canonical_solution / tests / entry_point / language
  -> 拼出 reference full function 或 full program
  -> 定位 Go entry function
  -> 去掉或过滤含注释、docstring 的函数
  -> HumanEval-X: 从函数体内抽取完整单行 statement 作为 target，构造 prefix / target / suffix
  -> MultiPL-E: 因 raw 数据无 canonical_solution，构造 func signature + [MASK] + closing brace 的 body-mask 样本
  -> render 成项目统一 ChatML-MASK 格式
  -> model.generate 得到 completion
  -> 后处理 completion
  -> 拼回 candidate_code = prefix + prediction + suffix
  -> 运行官方 tests 或自建兼容 Go judge
  -> 统计 pass@1 / pass@k / CodeBLEU
```

论文中应明确说明：这里报告的是 **MultiPL-E-derived Go body-mask eval** 和 **HumanEval-X-derived Go FIM**，不是官方 MultiPL-E / HumanEval-X generation protocol 的直接结果。MultiPL-E 的 Hugging Face parquet 只有 `prompt/tests/stop_tokens`，没有 Go `canonical_solution`，因此不能从 raw 数据直接抽取带 gold target 的单语句 FIM；如果后续额外接入可靠 Go reference solution map，可以再升级为 single-statement derived FIM。

派生后的数据放在：

```text
data/go_single_fim/test_data/multipl_e/
data/go_single_fim/test_data/humaneval_x/
```

建议输出文件为：

```text
data/go_single_fim/test_data/multipl_e/multipl_e_go_bodymask_canonical.jsonl
data/go_single_fim/test_data/multipl_e/multipl_e_go_bodymask_chatml.jsonl
data/go_single_fim/test_data/humaneval_x/humaneval_x_go_derived_canonical.jsonl
data/go_single_fim/test_data/humaneval_x/humaneval_x_go_derived_chatml.jsonl
```

canonical schema 建议统一为：

```json
{
  "uid": "humaneval_x_go_xxx",
  "source_dataset": "humaneval_x_go",
  "split": "test",
  "language": "go",
  "task_type": "go_single_statement_completion_derived",
  "prefix": "...",
  "target": "...",
  "suffix": "...",
  "full_code": "...",
  "target_kind": "assignment|return|call",
  "metadata": {
    "raw_task_id": "...",
    "entry_point": "...",
    "source_benchmark": "HumanEval-X",
    "derivation": "comment_free_single_statement_mask",
    "target_line": 12,
    "filters": ["no_comments", "single_line", "assignment"]
  },
  "judge_payload": {
    "kind": "derived_humaneval_x_go_test",
    "tests": "...",
    "entry_point": "...",
    "raw_sample": {}
  }
}
```

ChatML 渲染必须沿用训练数据中的固定话术：

```text
system:
You are a Go code completion assistant.

user:
Fill the [MASK] in the Go function. Return only the missing Go code, without Markdown fences or explanation.

* Incomplete Code:
{prefix}[MASK]{suffix}

assistant:
{target}
```

评测时只把 system/user 喂给模型，assistant 的 `target` 只用于 teacher forcing、CodeBLEU 或 debug。

#### 派生样本过滤与后处理

派生样本应尽量复用 GoSingle 的过滤原则：

- 只保留 Go。
- 只保留能定位到 entry function 的样本。
- 函数体、target 语句和模型输入里不应含普通注释或 docstring。
- 只抽完整单行 statement，优先 `assignment`、`return`、`call`。
- 拒绝 `if/for/switch/select` 等 block header，拒绝包含 `{` 或 `}` 的 target。
- 拒绝 `return nil`、`return err`、`break`、`continue`、`panic` 等信息量低或控制流强的 target。
- HumanEval-X 每个 raw benchmark task 建议最多保留 `per_task=1` 个派生 mask，避免一个原始题目被多个 mask 放大权重；内部分析可使用 `per_task=3`。
- MultiPL-E 当前是 body-mask eval，`target` 和 `full_code` 为空，`metadata.target_available=false`，只用于 pass@k，不用于 span-level CodeBLEU。

生成后处理建议：

```text
raw_generation
  -> 截断 <|im_end|> / <|endoftext|> / <|im_start|> 等 special token
  -> 去掉 Markdown fences
  -> 如果输出重复了 prompt 或 [MASK] 上下文，抽取缺失片段
  -> 对 single-statement derived task，只保留第一条完整 Go statement
  -> candidate_code = prefix + clean_prediction + suffix
  -> gofmt 只做格式化，不做语义修复
  -> Go unit tests / compatible judge
```

自建 evaluator 应至少区分：

```text
ok
format_error
compile_error
test_failure
timeout
unsupported
missing_prediction
```

pass@k 的统计方式保持 execution benchmark 口径：

```text
对每个 derived sample 生成 k 个 candidate。
只要至少 1 个 candidate 拼回后通过 tests，则该 sample pass@k = 1。
最终 pass@k = 通过样本数 / 可评测样本数。
```

如果一个 raw task 派生多个 mask，报告时应额外说明统计单位是 `derived sample` 还是 `raw task`；论文主表建议 HumanEval-X 优先使用 `per_task=1`，使二者基本一致。

### 5. 数据处理与评测输出约定

推荐目录组织：

```text
data/go_single/train_data/          # train canonical/chatml/compact annotated
data/go_single/eval_data/           # internal eval canonical/chatml/compact annotated
data/go_single/raw_data/            # 原始 benchmark 或 raw corpus
outputs/go_singleline_fim_exp/<benchmark>/       # prediction、result、report、viewer
runs/go_single/                      # 长任务日志
```

评测报告建议分成两张表。

Internal table：


| 类别               | 必报/建议报                                                |
| ------------------ | ---------------------------------------------------------- |
| saliency alignment | Recall@k、Precision@k、mAP@k、positive/negative gap        |
| completion quality | CodeBLEU、Exact Match、Edit Similarity、parse/gofmt rate   |
| 不应报告           | pass@k，除非该 internal 样本明确有 unit test/judge payload |

External table：


| 类别                        | 必报/建议报                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------ |
| completion quality 主指标   | pass@1、pass@k                                                                             |
| completion quality 辅助指标 | CodeBLEU、EM、Edit Similarity、parse/compile rate                                          |
| 不应报告                    | saliency alignment，除非额外对 external 样本做了独立 annotation 且明确标注为 analysis-only |

简而言之：internal 解释 causality curation 是否学到了对的 token 关系；external 证明模型写代码的最终效果是否提升。

## 四、标注数据

### 1. 标注 token 与边的定义

标注阶段先使用代码级 tokenizer，而不是直接使用 Qwen BPE。原因是结构规则和 LLM 更容易理解代码 word，例如变量名、关键字、括号、调用名、类型名。

代码 token 记录三类信息：

```text
surface: token 字符串
char_start / char_end: 在原始代码或指令中的字符区间
token_id: 标注阶段通常为 -1，BPE 后处理阶段才填 Qwen token id
```

边的基本结构是：

```text
token_i_idx -> token_j_idx
```

语义方向统一解释为：

```text
token_i 是 cue / source，token_j 是 consequence / target
```

也就是说，看到 `token_i` 应当有助于模型预测或理解 `token_j`。例如：


| 边类型     | 方向含义                                                   |
| ---------- | ---------------------------------------------------------- |
| `bracket`  | 左括号或块开始符号 -> 匹配的右括号或块结束符号。           |
| `defuse`   | 变量定义位置 -> 后续使用位置。                             |
| `call`     | 函数或方法调用名 -> 参数表达式 token。                     |
| `return`   | `return` 关键字 -> 返回表达式 token。                      |
| `type`     | 类型标注 token -> 被声明的变量名。                         |
| `dataflow` | 值的生产位置 -> 值的消费位置。                             |
| `semantic` | 控制流、语法配对、语义约束 token -> 被约束 token。         |
| `api`      | API 使用模式中的前置线索 -> 后续调用、资源释放或结果使用。 |

标注 tokenizer 默认会过滤普通注释和 docstring，但保留 C/C++ 预处理指令，因为 `#include`、`#define` 等可能真实影响代码依赖和符号定义。

需要注意，`humaneval` 训练数据是一个例外。它本身是 documented Python function 风格，docstring 是模型输入中真实可见的任务说明，不应像普通代码注释一样整体丢弃。当前处理方式是：

```text
普通代码注释 / 非任务 docstring:
  仍然过滤，避免把自然语言噪声写进代码结构标注。

humaneval documented function docstring:
  保留在 ChatML prompt 中，只抽取 Args / Parameters / Returns / @param / @return 等与代码生成直接相关的信息。
```

这类 docstring 边只允许从前文可见信息指向后续 code token，尤其是 assistant completion token；不生成 docstring 内部互指，也不生成 code token 反向指向 docstring 的边。当前规则重点补充两类信息：

- `type` 边：docstring 中的 `int/str/list/dict/set/boolean/array/optional/Any/pandas DataFrame`、reST `:class:\`str\``、CamelCase 类名等类型 token 指向对应参数、返回变量或返回表达式。
- `semantic` 边：Args / Returns 描述中和 prefix/suffix/completion 代码同名且有生成约束的信息指向后续代码；Examples / doctest / Traceback 区域不参与，避免示例文本误连到代码。

### 2. 结构化规则：不调用 LLM 的边

结构化规则由 tree-sitter 完成。它的作用是先用确定性方式提取稳定、便宜、可复现的代码关系，让 LLM 不必浪费 token 去重复标括号、定义使用、调用等基础结构。

结构化规则主要有五类。

#### `bracket`：括号和块结构

算法维护一个括号栈，按源码顺序扫描 tree-sitter 节点：

```text
遇到左括号: 入栈
遇到右括号: 从栈顶向前找匹配左括号，形成 left -> right
```

它不仅处理 `()`、`[]`、`{}`，也对泛型或模板参数中的 `< >` 做专门处理，例如 Java/C#/C++ 的 `List<T>`、`foo<T>()`。

#### `defuse`：定义到使用

这是一个轻量级两遍扫描：

```text
第一遍：在声明、参数、循环变量、赋值、短变量声明等节点中收集声明位置
第二遍：扫描所有 identifier 叶子节点，如果名字命中声明表，则产生 decl -> use
```

这不是完整的数据流分析，而是跨语言的近似 def-use 规则。它能覆盖函数参数、局部变量、循环变量、Go range、C/C++ 声明、C# local declaration 等常见模式。

#### `call`：调用名到参数

对 call expression 类节点，先找到 callee：

```text
function / name / method / 第一个 child
```

再找到 argument list，生成：

```text
callee -> 每个参数表达式的起始 token
```

因此 `foo(a, b + c)` 会产生 `foo -> a`、`foo -> b` 一类边，表达函数名对参数结构和参数语义的约束。

#### `return`：返回关键字到返回值

对 return statement，定位 `return` 关键字，再把它连到返回表达式：

```text
return -> expression
```

这条边的含义不是值从 `return` 流向 expression，而是 `return` 这个语法动作提示后面应出现返回值表达式。

#### `type`：类型到变量名

对 typed declaration，先排除函数或方法声明，避免把“返回类型 -> 函数名”误当成变量类型标注。然后定位类型节点和变量名节点，生成：

```text
type token -> variable name
```

例如 `int count` 中是 `int -> count`，泛型或 qualified type 会把类型内部的相关 token 都连向变量名。

### 3. LLM 补边：需要语义判断的边

LLM 标注器是一个 tool-calling agent。它必须先调用结构工具得到确定性边，然后再调用 `emit_correlations` 提交补充边。结构边会自动放入最终结果，LLM 不需要也不应该重复提交。

LLM 主要补三类规则难以稳定覆盖的边。

#### `dataflow`

表示值从生产位置到消费位置，例如：

```text
RHS expression -> LHS variable
function call result -> stored variable
loop init / update -> loop condition or body use
accumulator update -> later accumulator read
index variable -> array access
condition value -> branch body中的相关使用
```

结构规则中的 `defuse` 只看同名定义和使用，而 `dataflow` 更关注值依赖。例如 `total = total + x` 中，`x` 和旧 `total` 都对新 `total` 有值贡献，这类关系需要语义判断。

#### `semantic`

表示非数值流动但有语义约束的关系，例如：

```text
if -> else
try -> catch / finally
throw -> exception object
switch -> case
async -> await
import / using -> imported identifier
```

这些关系不一定是变量定义或值流，但会影响模型理解代码结构。

#### `api`

表示库函数或资源生命周期约束，例如：

```text
open -> close
malloc -> free
acquire -> release
API type -> method call
library call -> returned value usage
```

代码中保留了可选的 API 文档检索工具，但默认可以关闭，让 LLM 只根据代码上下文和结构边判断。

### 4. 目标边类型

在 benchmark 适配里，原始样本包含 `chatml`、`fim`、`test` 等字段。训练和推理使用 `chatml`，测试 pass 使用 `fim` 拼接完整代码和 `test`。标注时需要明确哪些边应该进入 causality curation，避免把无关 prompt token 拉入 loss，也避免漏掉 FIM 任务最关键的上下文到答案依赖。

FIM 场景里，`prompt/context` 不是只有 `[MASK]` 前缀，而是由两部分组成：

```text
prompt/context = prefix_before_mask + suffix_after_mask
completion     = masked_span / assistant answer
```

因此合法标注边不是简单地“只标 completion 内部”，而是限制为三类方向：


| 边范围                     | 方向                                        | 作用                                                              |
| -------------------------- | ------------------------------------------- | ----------------------------------------------------------------- |
| `prompt -> prompt`         | prompt 内部按正常代码顺序从前文到后文。     | 保留上下文内部的结构、def-use、调用、类型和语义约束。             |
| `prompt -> completion`     | prefix 和 suffix 都可以指向 completion。    | 最关键的监督边，表示哪些上下文 token 直接帮助预测被 mask 的答案。 |
| `completion -> completion` | completion 内部按正常代码顺序从前文到后文。 | 保留答案内部的局部结构、数据流和语义连续性。                      |

这里的 `prompt -> completion` 是核心：prefix 中的函数签名、变量定义、前置控制流，以及 suffix 中的后续使用、返回约束、错误处理、API 形态，都可能决定 completion 应该生成什么。因此 suffix 不能被当成“completion 之后不可见的未来文本”丢掉；在 FIM 任务里，suffix 本来就是 prompt 的一部分。

也就是说，禁止的是 `completion -> prompt`，因为模型生成 completion 时不能用答案 token 反过来解释 prompt；但 `prompt -> completion` 必须保留，而且 prefix 与 suffix 都属于 prompt。

具体实现上仍保留全局 token index，方便可视化和训练对齐完整 ChatML 序列；但边过滤时只保留上述三类边。这样 loss 既能集中到模型真正需要生成的 completion 区域，也不会丢失 FIM 里最重要的上下文到答案依赖。

### 5. Word Edge 到 Qwen BPE Edge

标注边最初在 word/code-token 层面，而训练输入是 Qwen BPE token。后处理阶段的核心是字符区间映射。

给定 simple token 和 BPE token 的字符 span：

```text
simple token s: [s_start, s_end)
BPE token b:    [b_start, b_end)
```

如果满足：

```text
s_start < b_end  且  b_start < s_end
```

就认为这个 simple token 与该 BPE token 重叠。

因此 word-level 边：

```text
simple_i -> simple_j
```

会映射为 BPE-level 边。当前后处理更偏保守地保留全部重叠 BPE 组合：

```text
BPE(simple_i) × BPE(simple_j)
```

也就是说，如果一个变量名被切成多个 BPE token，那么边会展开到多个 BPE 对。随后去重、丢弃自环、过滤越界边，得到训练数据中的 `attention_edges` 或 legacy 格式中的 `qwen_annotations`。

## 五、Saliency 的定义

训练中的 saliency 不是 attribution 模块里的全层 rollout ALTI，而是最后一层 decoder block 的 token contribution score。它的目标是回答：

```text
在最后一层中，source token s 对 query token q 的表示贡献有多大？
```

对最后一层某个 query 位置 `q` 和 source 位置 `s`，代码使用 attention 概率和 value/output projection 构造贡献向量：

```text
v_s^h = W_V^h · LN(x_s)
z_{q,s} = Σ_h A^h_{q,s} · W_O^h v_s^h
```

其中：

```text
A^h_{q,s}: 最后一层第 h 个 head 中 q 对 s 的 attention 概率
W_V^h: 第 h 个 head 的 value projection
W_O^h: 第 h 个 head 对应的 output projection 切片
x_s: 最后一层输入 hidden state
```

对自环位置还会加入 residual 分支：

```text
z_{q,q} = z_{q,q} + x_q
```

然后用 query 位置的 RMSNorm 尺度归一化，并取 L2 norm 得到标量贡献：

```text
c_{q,s} = || z_{q,s} / sigma_q ||_2
```

这个 `c_{q,s}` 就是训练 loss 中使用的 saliency。它比直接看 attention 更强，因为它不仅考虑“看了谁”，也考虑被看的 token 经过 `W_V` 和 `W_O` 后实际给 hidden state 带来的向量贡献。

为了节省显存，实际训练默认使用稀疏实现：不构造完整 `[B, T, T, D]` 贡献张量，而是只对有标注边的 query 行计算 `c_{q,*}`。

## 六、Saliency Loss 的设计

Saliency loss 的目标是让 annotation edge 对应的 source token 在模型内部贡献更强。对某个 query token $q$，先定义三个集合：

$$
A_q = \{s \mid s \rightarrow q \text{ is an annotation edge},\ s < q\}
$$

$$
M_q = \{s \mid s < q\}
$$

$$
N_q = M_q \setminus A_q
$$

其中 $A_q$ 是标注正样本 source，$M_q$ 是所有 causal source，$N_q$ 是非标注负样本 source。实际参与 loss 的 query 集合为：

$$
Q = \{q \mid |A_q| > 0 \ \text{and} \ |N_q| > 0\}
$$

### 1. 旧版本 loss：InfoNCE / multi-positive NLL

旧版本先把 saliency 转成 log-score：

$$
r_{q,s} = \log(C_{q,s} + \epsilon_{\text{reg}})
$$

然后用 temperature $\tau$ 缩放：

$$
\ell_{q,s} = \frac{r_{q,s}}{\tau}
$$

代码里这个 $\tau$ 曾经叫 `alpha`，因此要注意：这里的 `alpha` 不是 margin，而是 temperature。

在所有 causal source 上做 softmax。带负样本 floor 的形式可以理解为：

$$
p_{q,s} = \frac{\exp(\ell_{q,s})}{\exp(\ell_{q,s}) + \sum_{r \in N_q} \exp(\max(\ell_{q,r}, \epsilon))}
$$

其中 $\epsilon$ 相当于 negative floor，决定负样本最低有多强。直观上，$\epsilon$ 越大，负样本 logits 被抬得越高，softmax 分母里负样本更强；但如果 floor 和动态 saliency 分布耦合不当，也会让大量负样本梯度被截断或变得很弱。

对于一个 query $q$，multi-positive loss 是：

$$
\mathcal{L}_q = - \frac{1}{|A_q|} \sum_{s \in A_q} \log p_{q,s}
$$

总 saliency loss 是：

$$
\mathcal{L}_{sal} = \frac{1}{|Q|} \sum_{q \in Q} \mathcal{L}_q
$$

直观理解是：对每个 target token $q$，把所有前文 causal token 都作为候选 source，让 annotation source 在 softmax 分布里获得更高概率。它不只是要求 annotation saliency 高于平均 non-annotation saliency，而是让 annotation source 和所有 causal source 竞争排名。

### 2. 旧版本 loss 的问题

旧版 InfoNCE / multi-positive NLL 在实验里暴露了几个问题。

1. 动态 floor quantile 会和当前 saliency 分布耦合。例如使用 0.75 quantile 时，大约 75% 的 negative 可能被 floor 机制截断或弱化，导致它们几乎没有有效梯度。
2. 固定 floor，例如 floor = -10 后，loss 下降更明显，recall 和 precision 也有提高，但对 negative 的压低仍然不显著。主要原因是 positive 和 negative 数量极不均衡：negative 很多，softmax 梯度分摊到每个 negative 上后非常小。
3. 常数 floor 对拉高 positive mean 有帮助，但会推动部分 positive 越来越大，导致 positive 分布变得更不均衡。实验中 positive variance 从约 1.6 升高到约 3.2，说明 loss 更倾向于继续放大已经很强的 positive，而不是稳定地区分所有 positive 和 negative。

因此，旧版 loss 的主要缺陷不是“不能拉高 positive”，而是 negative 侧压力太弱、不均匀，并且 floor 超参数会显著影响梯度形态。

### 3. 新版本 loss：pairwise contrastive hinge

新版使用 pairwise contrastive hinge loss。仍然先在 log saliency 上比较正负样本：

$$
\mathcal{L}_q = \frac{1}{|A_q||N_q|} \sum_{s \in A_q} \sum_{s' \in N_q} \max\left(0, m + \frac{\log C_{q,s'} - \log C_{q,s}}{\tau}\right)
$$

其中 $m$ 是 margin，$\tau$ 是 temperature。每个 positive source $s$ 都会和每个 negative source $s'$ 组成 pair；只有当 negative 比 positive 过强，或者二者差距还没有超过 margin 时，hinge 才会激活。

这等价于要求：

$$
\log C_{q,s} - \log C_{q,s'} \ge m\tau
$$

如果该条件已经满足，则该 positive-negative pair 不再产生梯度。

### 4. 为什么 negative 侧更均匀

从梯度上看，对每个 active 的 $(s, s')$ pair，negative 侧梯度大小相同。记 $\ell_{q,s} = \log C_{q,s} / \tau$，则：

$$
\frac{\partial \mathcal{L}_q}{\partial \ell_{q,s'}} = \mathbf{1}[\text{active}] \cdot \frac{1}{\tau |A_q||N_q|}
$$

$$
\frac{\partial \mathcal{L}_q}{\partial \ell_{q,s}} = - \sum_{s' \in N_q} \mathbf{1}[\text{active}] \cdot \frac{1}{\tau |A_q||N_q|}
$$

这说明新版 loss 的 negative 压力来自显式 pairwise 比较：只要某个 negative 和某个 positive 的 margin 没拉开，它就会收到相同尺度的压低梯度。相比 InfoNCE 中所有 negative 共享 softmax 分母、梯度被大量 negative 稀释，pairwise hinge 对每个 negative pair 都有独立监督。

### 5. 为什么改用 contrastive loss

改用 contrastive loss 的核心优势有两点。

1. 每个 positive pair 和 negative pair 都有独立梯度，因此可以在拉高 positive 的同时压低 negative，而不是只依赖 softmax 分母间接压低 negative。
2. Hinge 天然带有“够好就不再惩罚”的机制，不需要额外设计常数 floor。虽然它仍然需要 margin $m$，但 margin 的含义更直接：只要 positive 比 negative 高出足够间隔，该 pair 就停止施加 loss；这通常比手工 floor 更 robust。

因此，新版 saliency loss 更符合 causality curation 的目标：不是单纯让 annotated source 越大越好，而是让 annotated source 相对 non-annotated source 有稳定、可解释的排序优势。

## 七、Train 的具体流程

训练入口会依次完成以下步骤。

### 1. 参数解析

训练参数分三组：


| 参数组                 | 作用                                                                          |
| ---------------------- | ----------------------------------------------------------------------------- |
| `ModelArguments`       | 模型路径、是否使用 PEFT/LoRA、LoRA 超参。                                     |
| `DataArguments`        | 数据路径、最大长度、可选语言过滤。                                            |
| `SFTTrainingArguments` | HuggingFace TrainingArguments 加上 saliency loss 的`lambda`、`alpha`、`eps`。 |

### 2. tokenizer 初始化

tokenizer 使用 Qwen / Qwen-Coder 的 chat token 设置：

```text
pad_token = <|endoftext|>
eos_token = <|im_end|>
additional_special_tokens = <|im_end|>, <|im_start|>
```

这样 dataset 中的 ChatML token 与模型 special token 保持一致。

### 3. dataset 与 collator

`AnnotatedSFTDataset` 读取带边训练数据，产出 `input_ids`、`labels`、`annot_pairs`。`DataCollatorForAnnotatedSFT` 做 padding，并把每个样本的边列表保留给 trainer。

### 4. 模型加载

模型必须以：

```text
attn_implementation = eager
```

加载。原因是 saliency loss 需要真实的 attention probability `A^h_{q,s}`。FlashAttention 或 SDPA 往往不会返回完整 attention 矩阵，所以即使命令里打开 flash attention，训练代码也会回落到 eager。

如果启用 PEFT，则使用 LoRA 训练以下模块：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

### 5. forward 与 loss 组合

每个 batch 中 trainer 会：

```text
1. 从 inputs 中取出 annot_pairs
2. 前向计算，并要求 output_attentions=True、output_hidden_states=True
3. 从模型输出得到 next-token loss
4. 用最后一层 attention 和 hidden_states[-2] 计算 saliency loss
5. 返回 total_loss = ntp_loss + lambda * saliency_loss
```

其中 `hidden_states[-2]` 是最后一层 decoder block 的输入，正好对应 contribution 公式中的 `x`。

训练日志会记录：

```text
ntp_loss
saliency_loss
C_bar / N_bar / ratio / 有效 query 数
```

这些指标可以帮助判断标注边是否真的在模型内部贡献中变强。

### 6. 保存与可视化回调

训练过程会挂载 attention visualization callback，把部分样本的 attention 信息输出到 `attn_viz`。训练结束后保存 trainer state 和模型权重。

## 八. 待修复的bug

### 标注的 Go selector / qualified type 覆盖不足

Go 里 `.` 是 selector，用于包名、类型、字段和方法选择，例如：

```go
dockerclient.APIEvents
result.dom.On(...)
result.attr.markDirty()
```

当前 structural annotation 对表达式位置的 selector 覆盖较好，例如 `obj.Method`、`obj.Field` 通常会产生 `api` 边。但对类型位置的 selector 覆盖不稳定，尤其是：

```go
chan *pkg.Type
[]pkg.Type
map[string]pkg.Type
```

这类代码在 tree-sitter 中可能被解析为 `qualified_type` 或相关 type node，而不是普通 `selector_expression`。因此当前标注有时只能通过外围调用边间接覆盖，例如 `make -> APIEvents`，但缺少更直接的：

```text
pkg -> Type
Type -> declared variable / field
```

这属于 annotation coverage bug / quality gap，后续可以做一个 Go selector annotation v2：

1. 在 structural checker 中补充 Go `qualified_type` / type-position selector 的边。
2. 考虑在 leaf token 提取时跳过单独的 `.`，避免出现 `. -> On` 这类语义不自然的边。
3. 用 500 条 old-vs-fixed A/B 比较 selector coverage、edge 数、训练 mAP/R@K，再决定是否重标更大规模数据。

## 九、Baselines 整理

当前代码里主要有六类对照项，外加我们自己的治理方法的训练路线。


| 方法                      | 分类                            | 核心思路                                                                                                          |
| ------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Base Qwen 和 Only CE Loss | 2个对照组                       | 直接评测 Qwen2.5-Coder-7B-Instruct 和使用交叉熵进行 SFT 后的模型，作为基础能力下限。                              |
| TokenCleaning             | token-level hard mask           | 用 base model 与 reference model 的 token loss 差异打分，保留 top ratio token，其余 supervised label 置为`-100`。 |
| XTF                       | token-level hard mask           | 用 attention relevance、PCP novelty、embedding task relevance 三类信号过滤噪声 token。                            |
| LLM-CleanCode             | sample/span-level rewrite       | 用 LLM 对 assistant completion 做 rename、modularize、planning 风格清洗，然后再 SFT。                             |
| CLEAR                     | sample-level filter/correct     | 用 observed consistency 和 self-reflection 估计样本质量，低置信样本过滤，高置信候选可替换原 target。              |
| IB-FT                     | loss-level optimization         | 不主要改数据，而是在训练时加 variational bottleneck 辅助 loss。                                                   |
| Ours Causality Cuartion   | annotation edge + saliency loss | 先标注 token edge，再训练时约束标注边的 contribution saliency 高于非标注边。                                      |

需要注意：`scripts/benchmark` 里仍保留一个旧的 `graph_signal_operator`，它属于 annotation graph hard-mask/soft-weight 数据治理算子；当前我们主要使用的是 `src/annotate` 生成 `attention_edges`，再由 `src/train` 的 saliency loss 训练。

如下主要给出同类型 paper 的整理，正式评测时还需要加入 base model 和 only ce loss sft model 作为对照组

### 1

==**Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning**==  ✅✅
Jinlong Pang, Na Di, Zhaowei Zhu, Jiaheng Wei, Hao Cheng, Chen Qian, Yang Liu
[[2502.01968] Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning](https://arxiv.org/abs/2502.01968)
ICML 2025，引用数 27 ，代码和数据公开，内容聚焦 token-level error detection & cleaning 提升 code generalization 能力

***通俗理解***
通过引入一个更强悍的参考模型（这个模型不是外部的 teacher 模型比如 gpt 系列，而是 base model 通过一定程度的 fine-tuning 得到的）计算出训练数据中每个样本每个 token 的分数，对数据从 token 层面上进行清理，进而提高 fine-tuning 效果

**1.提取文章摘要（直接给出无删减、未翻译的摘要全段）** （摘自论文第1页 Abstract，原样复制）

Recent studies show that in supervised fine-tuning (SFT) of large language models (LLMs), data quality matters more than quantity. While most data cleaning methods concentrate on filtering entire samples, the quality of individual tokens within a sample can vary significantly. After pre-training, even in high-quality samples, patterns or phrases that are not task-related can be redundant, uninformative, or even harmful. Continuing to fine-tune on these patterns may offer limited benefit and even degrade downstream task performance. In this paper, we investigate token quality from a noisy-label perspective and propose a generic token cleaning pipeline for SFT tasks. Our method filters out uninformative tokens while preserving those carrying key task-specific information. Specifically, we first evaluate token quality by examining the influence of model updates on each token, then apply a threshold-based separation. The token influence can be measured in a single pass with a fixed reference model or iteratively with self-evolving reference models. The benefits and limitations of both methods are analyzed theoretically by error upper bounds. Extensive experiments show that our framework consistently improves downstream performance. Code is available at [https://github.com/UCSC-REAL/TokenCleaning](https://github.com/UCSC-REAL/TokenCleaning).

**2.概括文章（领域 + 问题/目标 + 方法 + 核心创新/优势）** （对应论文 Introduction / Chapter 1 + Figure 1 + Related Work / Chapter 2）

* **领域**：Data-Centric AI 在 LLM Supervised Fine-Tuning (SFT) / Instruction Tuning 中的细粒度数据治理（聚焦 token-level selection，与我们 Qwen Go 代码补全任务高度同构）。
* **针对问题/目标**：SFT 数据集中即使是“高质量样本”，内部仍存在大量 uninformative tokens（pre-training 已学过的 common patterns、高频短语、冗余结构），这些 token 会稀释 task-specific 信号、引入噪声梯度，导致 downstream 性能下降；传统方法仅做 sample-level cleaning（过滤整条样本），忽略样本内部 token 质量差异。
* **使用方法**：从 noisy-label 视角出发，提出通用 **Token Cleaning Pipeline**，通过 influence-guided scoring（模型更新对每个 token 的 loss 影响）+ threshold-based 分离，实现“敲除 uninformative tokens + 保留 task-specific tokens”。
* **核心创新与优势**：
  1. 首次系统性将 token-level noisy label cleaning 引入 SFT（超越 sample-level），支持两种实现：**Fixed-Model Cleaning**（单次全局 ranking）和 **Self-Evolving Cleaning**（迭代自进化参考模型）。
  2. 提供严格理论框架（Theorem 5.1 + Corollary 5.2），证明 token cleaning 可在 noise rate 降低 vs. token 数量减少之间取得最优 trade-off，并解释两种策略的稳定 vs. Matthew Effect。
  3. 全局 ranking（而非 per-sample local ranking）避免低质样本污染，实验在 3B/8B/7B 模型 + 7 个 OpenLLM Leaderboard 任务上，Self-Evolving 策略较 full tokens 平均提升 6.3%（3B 模型），Fixed-Model 也稳定优于 RHO baseline（Figure 1、Table 1、Chapter 6）。
  4. 开源代码 + 50k 高质量数据池，可直接复用。

**3.详细展开技术路线（对应 Chapter 4 + Figure 1 + Algorithm 1 + Chapter 5）**
Token Cleaning Pipeline 核心思想：将 SFT 中的 response tokens 视为 noisy labels（**\\tilde{y}\_{i,j}=1** 但实际部分为 uninformative），通过 influence score 量化每个 token 的“任务相关性”。

注意，本文所说的 xy 和平时所说的不同
x\_{i,j} 表示第 **i** 个样本中 response 的第 **j** 个 token
x\_{i,:j} 表示该样本的完整 prefix = prompt 的全部 tokens + response 的前 **(j-1)** 个 tokens
y 表示一种伪标签，作为一种额外的标签，代表这个 token 是否 informative

**3.1 评分函数（Chapter 4.2.1）** 给定 base model **\\theta** 和 reference model **\\theta'**，token influence 定义为：

**\\text{Infl}(x\_{i,j}|x\_{i,:j}; \\theta, \\theta') := \\ell(x\_{i,j}|x\_{i,:j}; \\theta') - \\ell(x\_{i,j}|x\_{i,:j}; \\theta)**

其中，**\\ell(x\_j, \\theta) = -\\log P\_\\theta(x\_j \\mid x\_{<j})**

Token quality score（越高越 informative）：

**\\text{Score}(x\_{i,j}|x\_{i,:j}; \\theta, \\theta') = -\\text{Infl}(x\_{i,j}|x\_{i,:j}; \\theta, \\theta')**

（更负 influence = reference model 对该 token 预测更自信 = 该 token 对 task 更关键）

**3.2 阈值分离（Chapter 4.2.2）** 全局排序所有 token score，按固定比例 **k\\%**（论文实验推荐 50%-70%）保留 top-k% 为 informative（**\\hat{y}\_{i,j}=1**），其余过滤（**\\hat{y}\_{i,j}=0**）。

**3.3 两种实现策略（Chapter 4.3 + Figure 1 + Algorithm 1）**

* **Fixed-Model Cleaning**（单次全局）：

base **\\theta** = 较弱模型，reference **\\theta'** = 预先 warm-up 的较强模型（或 DS² 选出的 10k 高质样本 fine-tune 得到）。
对全数据集 **\\tilde{D}** 一次性计算 score → 全局 top-k% 保留 → 仅在 cleaned tokens 上 fine-tune 最终模型。
优势：稳定（data quality 固定，quantity 可控），理论上 error upper bound 持续降低（Theorem 5.1）。

* **Self-Evolving Cleaning**（迭代自进化，推荐策略）：   将 **\\tilde{D}** 均匀切分为 **T+1** 子集 **\\{\\tilde{D}\_0, \\dots, \\tilde{D}\_T\\}**。
  1. Warm-up：**\\theta\_0** 在 **\\tilde{D}\_0**（full tokens）上 fine-tune 得 **\\theta\_1**（初始 reference）。
  2. for **t=1** to **T**:
     用最新 reference **\\theta\_t** + base **\\theta\_0** 计算 **\\tilde{D}\_t** 的 scores → top-**k\_{\\text{self-evol}}\\%** 清洗 → 在 cleaned **\\tilde{D}\_t** 上 fine-tune 得 **\\theta\_{t+1}**。
     最终输出 **\\theta\_{T+1}**。
     优势：参考模型随迭代 progressively 提升 supervision 信号（Matthew Effect），但需警惕 poor group 的“poor get poorer”（Chapter 5.3）。

**理论支撑（Chapter 5）**：

全 tokens 的 error upper bound：

**L\_D(\\hat{\\theta}\_{\\tilde{D}}) \\leq \\eta(\\tilde{D}) + \\sqrt{\\frac{2\\log(4/\\delta)}{M}}**

（**\\eta** = noise rate，**M** = token 数量）。Token cleaning 优于 full tokens 的条件见 Corollary 5.2

本质上是在说明， cleaning 后 **\\eta(\\tilde{D})** 显著下降（只保留 top-k% informative tokens），同时 M M 只轻微减少（论文推荐保留 50%-70%，M 仍足够大），整个泛化误差上界大幅降低，模型泛化更好

**实例可视化**（Figure 3 + Appendix F）：迭代中 informative tokens（红色）逐渐向左上角聚集（base loss 高、reference loss 低），common tokens 被有效过滤。

**4.文章使用的数据和代码链接****代码**：论文 Abstract 明确给出开源仓库

**[https://github.com/UCSC-REAL/TokenCleaning](https://github.com/UCSC-REAL/TokenCleaning)**（包含完整 pipeline、Fixed / Self-Evolving 实现、实验脚本）。

**数据集**（Chapter 6.1 + Appendix D.1）：

* 50k 高质量 SFT 数据池，由以下 5 个公开数据集经 LLM-driven quality rating curation 得到（Pang et al., 2024b）：
  * Flan v2 (Longpre et al., 2023)
  * Open Assistant 1 (Köpf et al., 2024)
  * Stanford Alpaca (Taori et al., 2023)
  * Dolly (Databricks, 2023)
  * WizardLM (Xu et al., 2023)
* 采用 Tulu template 标准化（Wang et al., 2023）。
* 统计信息（Table 4）：平均 prompt / completion / overall length 已给出，无需额外下载（仓库中应包含处理后数据）。
* 评估基准：7 个 OpenLLM Leaderboard 任务（MMLU, TruthfulQA, TydiQA, HellaSwag, ARC-Challenge, BoolQ, LogiQA），使用 lm-eval-harness 仓库。

### 2

**==XTF: Explainable Token-level Noise Filtering for LLM Fine-tuning Datasets==** ✅✅
Yuchen Yang, Wenze Lin, Enhao Huang, Zhixuan Chu, Hongbin Zhou, Lan Tao, Yiming Li, Zhan Qin, Kui Ren
[http://arxiv.org/pdf/2602.14536v2](http://arxiv.org/pdf/2602.14536v2)
ICLR 2026，引用数暂时为 0（论文刚于3月上线）

***通俗理解***
为每个 token 定义三个属性，一个是和 attn 相关的推理重要性、一个是和 prop 有关的知识新颖性、一个是和 embedding 特征向量相关的任务相关度，在这三个属性上筛选（各有各的筛选规则）但凡一个属性不好就直接把这个 token 过滤掉，“过滤”的方式是把 label 标为-100，即 mask 掉不参与相关 loss 和 grad 的计算

---

***1. 提取文章摘要（直接给出无删减、未翻译的摘要全段）***

Large Language Models (LLMs) have seen remarkable advancements, achieving state-of-the-art results in diverse applications. Fine-tuning, an important step for adapting LLMs to specific downstream tasks, typically involves further training on corresponding datasets. However, a fundamental discrepancy exists between current fine-tuning datasets and the token-level optimization mechanism of LLMs: most datasets are designed at the sentence-level, which introduces token-level noise, causing negative influence to final performance. In this paper, we propose XTF, an explainable token-level noise filtering framework. XTF decomposes the complex and subtle contributions of token-level data to the fine-tuning process into three distinct and explicit attributes (reasoning importance, knowledge novelty, and task relevance), which can be assessed using scoring methods, and then masks the gradients of selected noisy tokens accordingly to optimize the performance of fine-tuned LLMs. We conduct extensive experiments on three representative downstream tasks (math, code and medicine) across 7 mainstream LLMs. The results demonstrate that XTF can significantly improve downstream performance by up to 13.7% compared to regular fine-tuning. Our work highlights the importance of token-level dataset optimization, and demonstrates the potential of strategies based on attribute decomposition for explaining complex training mechanisms.

---

***2. 概括文章（领域 + 问题/目标 + 方法 + 核心创新/优势）***

（对应论文 Introduction / Chapter 1 + Figure 2 + Methodology / Chapter 3 + Related Work / Chapter 2）

* **领域**：Data-Centric AI 在 LLM Fine-Tuning 数据治理中的 **token-level** 优化（聚焦 supervised fine-tuning 中的 token-level noise，与我们 Qwen Go 代码补全任务高度类似）。
* **针对问题/目标**：现有 fine-tuning 数据集均为 sentence-level 设计，但 LLM 实际以 token-by-token 方式计算 loss 并更新参数，导致大量 token-level noise（对下游性能无贡献甚至有害的 token），误导优化方向，最终降低 fine-tuned 模型在目标任务上的表现（math/code/medicine 等）。
* **使用方法**：提出 **XTF**（Explainable Token-Level Noise Filtering）框架，将 token 对 fine-tuning 的贡献分解为三个可解释属性（Reasoning Importance RI、Knowledge Novelty KN、Task Relevance TR），分别设计可控计算成本的评分机制（attention / PCP / embedding distance），再通过保守策略过滤 noisy tokens 并在训练时 mask 其梯度。
* **核心创新与优势**：
  1. 首次系统性地从 **token-level** 视角解决 fine-tuning 数据噪声问题（超越 sample-level filtering / augmentation），并给出三个属性分解的理论证明（Appendix A）。
  2. 三个属性互补（重叠率 <58.3%），覆盖 base model 认知、知识新颖性和任务相关性，实现“复杂问题拆解为多个简单问题”（Figure 4）。
  3. 在 3 大下游任务（math/code/medicine）+ 7 个主流 LLM 上，XTF 最高提升 13.7%（medicine）、13.3%（math）、6.3%（code pass@10），显著优于 Normal FT、DF、DA、SLM、TC 等 baseline（Table 1、Figure 1、Chapter 4）。
  4. 完全即插即用（LoRA + gradient masking），计算开销可控，无需额外 reference model 训练。

---

***3. 详细展开技术路线***
（对应 Chapter 3 + Figure 2 + Figure 3 + Appendix A）

XTF 流水线分为三阶段（Figure 2），核心是“属性分解 → 评分 → 梯度 mask”。

3.1 理论基础：哪些 token 是 noise？（Chapter 3.1 + Appendix A）

Fine-tuning 本质上是 base model 与任务数据集的对齐。token 对 fine-tuning 的贡献可分解为三个正向属性：

* **Reasoning Importance (RI)**：该 token 是否显著影响 base model 的推理结果。
* **Knowledge Novelty (KN)**：该 token 是否为 base model 未学过的新知识。
* **Task Relevance (TR)**：该 token 是否与任务目标相关。

**噪声定义**（Eq. 1）：

**D\_{\\text{noise}} = (D\_{RI\\downarrow}) \\cup (D\_{KN\\downarrow}) \\cup (D\_{TR\\downarrow})**

其中：

* **D\_{\\text{noise}}**：**token-level noise 数据集**（即所有需要被过滤/梯度 masking 的有害 token 集合）
* **D\_{\\text{RI}\_\\downarrow}**：**缺乏 Reasoning Importance（推理重要性）的 token 集合**
* **D\_{\\text{KN}\_\\downarrow}**：**缺乏 Knowledge Novelty（知识新颖性）的 token 集合**
* **D\_{\\text{TR}\_\\downarrow}**：**缺乏 Task Relevance（任务相关性）的 token 集合**
* **\\cup**：**并集**（只要满足任意一条“↓”条件，就属于噪声）
  3.2 属性评分机制（Chapter 3.2）

**RI → Attention Score**（Eq. 2）：代表 **base model 已有的认知能力**。如果 attention 很低，说明模型在生成后续内容时几乎不依赖该 token。在您的 Go 代码补全场景中，这类 token 属于“不影响注意力流的背景噪声”。

```
$$S_{RI}(O_k) = A(\theta, I+O)[l_I + k]$$
```

* **A(\\theta, \\cdot)**：base model 的 attention 函数（多头平均或最后一层 attention map）。
* **I / **O****：input tokens（提示）与 output label tokens（答案）。
* **l\_I**：input 长度（将完整文本 **I + O** 一次性喂给模型）。
* **O\_k**：output label 中第 **k** 个 token。

**过滤阈值（**\\downarrow**）**（论文公式 5）：使用 **quantile + IQR**（四分位距）自适应识别异常低分：

**O\_k \\in D\_{RI\_\\downarrow} \\quad \\text{if} \\quad S\_{RI}(O\_k) < Q\_1 - (Q\_3 - Q\_1)**

**KN → PCP Score**（Eq. 3）：代表 **base model 与任务数据集的知识差异**。PCP（预测概率）越高，**S\_{KN}** 越低，意味着模型**已经掌握**了该知识点。微调时保留高 PCP 的 token 不仅学不到新东西，反而可能强化错误的训练偏差（Correlation）。

```
$$S_{KN}(O_k) = 1 - P(O_k | I + [O_0, \dots, O_{k-1}])$$
```

* **P(\\cdot)**：base model 在 teacher-forcing 下对正确 token 的**预测概率**。

**过滤阈值（**\\downarrow**）**（论文公式 6）：

**O\_k \\in D\_{KN\_\\downarrow} \\quad \\text{if} \\quad S\_{KN}(O\_k) < 0.05 \\quad \\text{(即 PCP > 0.95 )} **

**TR → Distance Score**（Eq. 4）：代表 **token 与下游任务目标的语义相关性**。如果一个 token（如注释中的无关单词）距离 Go 代码补全的任务中心太远，即使它具有推理重要性，也属于“题外话”噪声。

**V(\\text{Domain}) = \\frac{1}{n\_w} \\sum E(\\theta, \\text{exp}\_w)**

**S\_{TR}(O\_k) = 1 - \\text{Normalize}\\bigl(D\\bigl(E(O\_k), V(\\text{Domain})\\bigr)\\bigr)**

* **E(\\theta, \\cdot)**：base model embedding 层输出（Context-free）。
* **V(\\text{Domain})**：整个微调数据集所有 token 的平均向量（语义中心）。

**过滤阈值（**\\downarrow**）**（论文公式 7）：采用 **Multi-Otsu 多阈值聚类**。通常取**第 2 小均值簇**（排除掉空格等极低分簇后的次低分集合）：

**O\_k \\in D\_{TR\_\\downarrow} \\quad \\text{if} \\quad S\_{TR}(O\_k) \\in \\mathcal{M}(S\_{TR})^{2nd}**

3.3 过滤与训练汇总（Chapter 3.3）

* **RI 过滤**：Quantile (Interquartile Range) 过滤极低分 token（Eq. 5）。
* **KN 过滤**：启发式阈值 PCP > 95%（Eq. 6）。
* **TR 过滤**：Multi-Otsu 聚类，取第二小均值簇（Eq. 7）。
* **训练时**：对 noisy token 在 label 中标记默认值（-100），实现 gradient mask（Eq. 8）。

**效果验证**（Figure 3-4）：三个属性分布互补，重叠率低；过滤后模型在 math/code/medicine 任务上显著优于 baseline。

---

***4. 文章使用的数据和代码链接***

**数据集**（Chapter 4.1）：

* **Math**：GSM8K（fine-tune & eval）、NuminaMath-CoT（fine-tune）+ MATH-500（eval）、FIQA（finance）。
* **Code**：CodeExercise（fine-tune）+ HumanEval（eval）。
* **Medicine**：PubMedQA（fine-tune & eval）。   （均为公开标准基准，无需额外链接。）

**代码**：论文正文、附录及参考文献中均未提供任何公开代码仓库、GitHub 链接或实现脚本（包括三个属性评分及 gradient mask 的具体代码）。仅给出详细算法描述、超参（Table 3）和实验设置，未开源。

### 3

==**IB-FT: Breaking Memorization Barriers in LLM Code Fine-Tuning**== ✅✅
[http://arxiv.org/pdf/2510.16022v1](http://arxiv.org/pdf/2510.16022v1)
Changsheng Wang, Xin Chen, Sijia Liu, Ke Ding
预印本，无引用

***通俗理解***
通过引入互信息中的 IB 原理，设计一个新颖的 loss 函数，该 loss 函数的本质是通过 KL 散度让构造的后验分布符合先验分布、进而实现信息压缩，打破模型训练中的记忆壁垒（这个记忆壁垒，大抵意思就是说模型训练会把训练样本整个“记下来”，相当于在死背答案）

---

**1. 提取文章摘要（直接给出无删减、未翻译的摘要全段）** （摘自论文第1页 Abstract，原样复制）

Adapting pretrained large language models (LLMs) to code domains via supervised fine-tuning (FT) has been commonly used for code generation. However, we identify a previously underappreciated failure mode, the memorization barrier, where strong memorization of downstream code data in the base model could trap optimization and prevent the standard FT from effectively acquiring new, generalizable code knowledge. To overcome this barrier, we propose the information bottleneck (IB)-guided fine-tuning, termed IB-FT, which applies an IB penalty on hidden representations of the code data to compress spurious, memorized features while preserving task-relevant information. Extensive experiments on two code benchmarks (OriGen and Evol-CodeAlpaca-V1) show that IB-FT substantially alleviates the memorization barrier, improves top-1 performance (Pass@1), and yields far more stable gains under the stricter multi-sample metric Pass@k(m) (a problem counts as solved only if at least m of k samples pass unit tests) compared with conventional FT.

---

**2. 概括文章（领域 + 问题/目标 + 方法 + 核心创新/优势）** （对应论文 Introduction / Chapter 1 + Figure 1 + Figure 3-5 + Related Work / Chapter 2）

* **领域**：LLM Code Fine-Tuning / Code Generation 中的 Data-Centric 优化（聚焦 supervised fine-tuning 中的 memorization 问题，与我们 Qwen Go 代码补全任务高度同构）。
* **针对问题/目标**：传统 FT 在代码领域存在“memorization barrier”——base model 已强烈记忆 fine-tuning 数据，导致优化被困在局部最优，无法有效习得新、可泛化的代码知识；表现为 Pass@1 急剧下降、Pass@k(m)（m>1）极不稳定（Figure 2、Chapter 3）。
* **使用方法**：提出 **IB-FT**（Information Bottleneck-guided Fine-Tuning），在 hidden representations 上施加 IB penalty，压缩 spurious memorized features，同时保留 task-relevant 信息（Chapter 5）。
* **核心创新与优势**：
  1. 首次正式定义并系统分析“memorization barrier”（cross-setting 视角：base model 对 fine-tuning 数据的预先记忆），区别于以往仅视 memorization 为 privacy/contamination 问题（Chapter 4）。
  2. 将 IB 原理首次应用于 code FT（无须额外 data pruning/attribution），通过 variational IB regularizer（Eq. (IB-FT)）使不同记忆强度的样本被“平等对待”，实现更均匀学习（Figure 5 表征分析）。
  3. 在 OriGen（Verilog）和 Evol-CodeAlpaca-V1 两个基准上，IB-FT 显著提升 Pass@1，并在大 m 的 Pass@k(m) 下提供远更稳定的增益（Table 1、Figure 6），同时对 temperature 变化鲁棒（Chapter 6）。
  4. 无需修改数据或模型架构，即插即用（LoRA + IB penalty），计算开销可控。

---

**3. 详细展开技术路线（对应 Chapter 4 + Chapter 5 + Figure 3-5 + Appendix A）**

**核心目标**：在 fine-tuning 过程中，通过信息瓶颈（Information Bottleneck, IB）约束隐藏表示，动态压缩输入特定的、无关的或过度记忆的特征（memorized spurious features），同时保留任务相关信号，提升泛化能力。

3.1 Memorization Barrier 诊断（Chapter 4）

* **Min-K% Prob 方法**：用于量化 base model **\\theta\_0** 对 fine-tuning 数据 **D\_{\\text{code}}** 的记忆程度。
  * K=20%，取概率最低的 20% token，计算平均负 log-likelihood。
  * **lower score → stronger memorization**。
  * 可视化：概率密度函数，峰值越靠左说明模型记忆越强 。
* **比较三个分布**：
  * **p\_0(D\_{\\text{code}}; \\theta\_0)**：base model 对数据的预测
  * **p\_1(D\_{\\text{code}}; \\theta\_{\\text{code}})**：fine-tuned model 对数据的预测
  * **p\_2(D\_{\\text{TOFU}}; \\theta\_0)**：弱记忆参考（TOFU 数据集，synthetic / fictitious）
* **实验验证**：
  * 移除 top 10% 最 memorized 数据点 → Pass@1 显著提升（Figure 4）。
  * 说明 memorization 构成了 fine-tuning 的 barrier 。

**纠正理解**：

* Memorization barrier 不只是过拟合或数据泄漏，而是 base model 预训练阶段已经强记忆了 fine-tuning 数据，使得标准 FT 很难跳出局部最优。
* 去掉 top-memorized 数据只是诊断手段，而不是 IB-FT 方法本身的核心。
  3.2 IB-FT 技术路线（Chapter 5）
* **IB 原理**：
  **\\min I(X;Z) - \\beta I(Z;Y)**
  * **Z**：bottleneck representation，从 LLM 中间层（如第 20 层 hidden state）提取。
  * 目标：压缩输入特定 spurious 信息，保留预测相关信号。
* **Variational 实现**：
  **\\ell\_{\\text{IB}} = \\ell\_{\\text{IB}}^{\\text{compress}} - \\beta \\ell\_{\\text{IB}}^{\\text{predict}}**
  **\\min\_{\\theta,\\phi} \\ell\_{\\text{FT}}(\\theta) + \\alpha \\ell\_{\\text{IB}}(\\theta,\\phi)**

Compress Loss

**\\ell\_{\\text{IB}}^{\\text{compress}} = \\mathbb{E}[D\_{\\text{KL}}(q\_\\phi(z|h\_\\theta(x)) || p(z))]**

* **h\_\\theta(x)**：LLM 中间层隐藏表示（第 20 层）。
* **q\_\\phi(z|h\_\\theta(x))**：learnable 变分编码器，输出 Gaussian 分布 bottleneck **z\\in (B, N, D\_z)**
* **p(z) = \\mathcal{N}(0, I)**：先验分布
* KL 散度惩罚模型在 bottleneck 中保留过多输入特定信息（memorized spurious features）。

Predict Loss

**\\ell\_{\\text{IB}}^{\\text{predict}} = \\mathbb{E}[\\log p\_\\theta(y|z)]**

* 最大化这个 loss，防止过压缩，确保任务信号保留。

**纠正理解**：

* IB-FT 并不是单纯丢掉 memorized 数据或 token，而是通过压缩 hidden representation **Z** 达到“对不同记忆强度样本平滑学习”的效果 。
* **Z** 压缩的是隐藏表示，而不是直接输入 token。

实现细节

* LoRA (r=32) + IB penalty (**\\alpha, \\beta**)
* 训练 3 epochs, batch=4, LR=5e-5\~1e-4
* 适用于 DeepSeek-Coder 和 CodeLlama base models
* Appendix A 给了更详细的 GPU、数据和 LR 配置 。
  3.3 效果验证
* **Representation analysis**（Figure 5）：
  * 计算 most-memorized vs. least-memorized 样本的 **ℓ\_2** 距离和角度差异
  * IB-FT 压缩 >50%，比 FT 更均匀学习。
* **多温度鲁棒性**（Figure 6）：
  * T ∈ {0.2, 0.6, 1.0}
  * IB-FT 在 Pass@k(m) 下均优于 FT，表现更稳定。
* **严格 Pass@10(m)**：
  * IB-FT 在 m>1 时提升明显，显示其更可靠的多样本一致性。

**纠正理解**：

* IB-FT 成功的核心不是简单提升 Pass@1，而是**缓解 memorization barrier、压缩隐藏表示、平滑样本间表征差异**。
* 数据移除(top-memorized pruning)是对 barrier 的诊断，而 IB-FT 通过 IB 正则化自动处理，不依赖手动删除 。

---

**4. 文章使用的数据和代码链接****数据集**（Chapter 6.1）：

* **OriGen**（Cui et al., 2024）：Verilog 指令-响应数据集（222,075 examples），用于 RTL 代码生成。
* **Evol-CodeAlpaca-V1**（Luo et al., 2024）：多语言指令-to-code 数据集（\~111k examples，覆盖 Python/C++/Java 等）。
* **评估基准**：OriGen → VerilogEval（Eval-Human + Eval-Machine）；Evol-CodeAlpaca-V1 → HumanEval。   （论文未提供直接下载链接，但均为公开基准，可通过原论文引用获取。）

**代码**：论文正文、附录及参考文献中均未提及任何公开代码仓库、GitHub 链接或实现脚本（包括 IB penalty 的具体实现）。仅给出 fine-tuning 超参（LoRA r=32、α/β 值等），未开源。

### 4

==**LLM-Assisted Code Cleaning For Training Accurate Code Generators**==  ✅✅
Naman Jain, Tianjun Zhang, Wei-Lin Chiang, Joseph E. Gonzalez, Koushik Sen, Ion Stoica
[[2311.14904] LLM-Assisted Code Cleaning For Training Accurate Code Generators](https://arxiv.org/abs/2311.14904)
UC Berkeley 的 Ion Stoica、Joseph Gonzalez，ICLR 2024，引用数 52

***通俗理解***
通过定义三种固定的 prompt ，借用强大的 LLM 对数据进行三个维度的清洗（重命名使代码更加易读、模块化划分出多个函数使其结构清晰、给出位于代码段首的注释来清晰描述代码）

---

**1.提取文章摘要（直接给出无删减、未翻译的摘要全段）**

（对应论文第 1 页 Abstract）

Natural language to code generation is an important application area of LLMS and has received wide attention from the community. The majority of relevant studies have exclusively concentrated on increasing the quantity and functional correctness of training sets while disregarding other stylistic elements of programs. More recently, data quality has garnered a lot of interest and multiple works have showcased its importance for improving performance. In this work, we investigate data quality for code and find that making the code more structured and readable leads to improved code generation performance of the system. We build a novel data-cleaning pipeline that uses these principles to transform existing programs by 1.) renaming variables, 2.) modularizing and decomposing complex code into smaller helper sub-functions, and 3.) inserting natural-language based plans via LLM based transformations. We evaluate our approach on two challenging algorithmic code generation benchmarks and find that fine-tuning CODELLAMA-7B on our transformed modularized programs improves the performance by up to 30% compared to fine-tuning on the original dataset. Additionally, we demonstrate improved performance from using a smaller amount of higher-quality data, finding that a model fine-tuned on the entire original dataset is outperformed by a model trained on 15% of our cleaned dataset. Even in comparison to closed-source models, our models outperform the much larger ALPHACODE models (Li et al., 2022).

---

**2.概括文章（领域 + 问题/目标 + 方法 + 做了什么 + 核心创新/优势）**

（对应 Section 1 Introduction + Section 4 Experimental Results + Figure 1 + Table 3/4）

* **领域**：自然语言到代码生成（Natural Language to Code Generation），属于 LLM 在算法代码生成（algorithmic code generation）的子领域。
* **针对的问题/目标**：现有工作只关注训练集的**数量与功能正确性**，忽略程序的**风格化元素**（readability、structuring、modularity），导致模型学到有害 correlation，生成错误预测。
* **用什么方法**：提出 **LLM-assisted 三步数据清洗管道**（使用 instruction-tuned LLM + oracle equivalence checker 保证功能等价）。
* **做了一件什么事情**：将现有数据集（APPS + CODE-CONTESTS）并行转化为“cleaned”版本（**D\_{rename} \\rightarrow D\_{modular} \\rightarrow D\_{planning}**），然后在 CODELLAMA-7B 上 fine-tune，验证下游代码生成性能（PASS@K）。
* **核心创新/优势**：
  * **创新**：首次系统证明“结构化 + 可读性”本身就是数据质量的关键属性；提出**可重复、可验证的清洗流水线**。
  * **优势**：
    1. 性能提升高达 30%（PASS@1/25/100）。
    2. **数据效率极高**：仅用 cleaned 数据集的 **15%** 即可超越 full original 数据集（Figure 3）。
    3. 在开放模型上超越更大规模的闭源模型（如 ALPHACODE-41B）。

---

**3.详细展开技术路线**（对应 Section 2 Methodology + Figure 1 + Table 2 + Appendix B）

论文的核心是**迭代式 LLM 提示 + oracle 功能等价检查**。

整体流程（Figure 1）：

1. **输入**：原始问题描述（problem statement）+ 原始 Python 程序 + 测试用例（用于 oracle）。
2. **三步顺序清洗**（Table 2）：
   * **Step 1: Rename variables** (**D\_{rename}**)   *Prompt*：“Rename the variables in the program to be descriptive, meaningful, and consistent.”
   * **Step 2: Modularize functions** (**D\_{modular}**)   *Prompt*：“Refactor the above program making it more modular with smaller and meaningful helper functions...”   结果：平均插入 \~2.6 个 helper function，如 `dfs`、`build_graph`、`gcd` 等。
   * **Step 3: Plan annotations** (**D\_{planning}**)   *Prompt*：“Generate a natural language description for the following functions in the program within four lines each.”   将 top-down 函数总结作为 comment 预置到程序头部。
3. **Oracle Equivalence Checker**：用数据集自带测试用例验证功能等价；失败则最多重试 5 次（温度 0.3）。
4. **下游验证**：
   * Full fine-tuning (CL-7B, 2 epochs APPS, **lr=5 \\times 10^{-5}**)。
   * 对照组：**D\_{distill}**（纯合成数据）与 **D\_{planning} + gold\\ plan**。

**4.搜集文章使用的数据和代码并给出链接**

* **数据集**：
  * **APPS** (Hendrycks et al., 2021): [https://github.com/hendrycks/apps](https://github.com/hendrycks/apps)
  * **CODE-CONTESTS** (Li et al., 2022): [https://github.com/google-deepmind/code\_contests](https://github.com/google-deepmind/code_contests)
  * *注：论文未发布清洗后的并行数据集（如 \$D*{modular}\$）。\_
* **代码**：
  * 论文**未公开**任何官方代码仓库、清洗脚本或 fine-tuning 实现。
  * 所有实验细节（Prompt、超参）均在正文 + Appendix B 中完整描述，可供复现。

### 5

**==CLEAR: Automated Data Curation for Robust Language Model Fine-Tuning==** ✅✅
Jiuhai Chen（University of Maryland, Cleanlab）、Jonas Mueller（Cleanlab）
[https://arxiv.org/pdf/2403.12776](https://arxiv.org/pdf/2403.12776)
曾作为Anonymous ACL submission提交（OpenReview链接可见），尚未正式发表于特定会议或期刊，引用数 39

***通俗理解***
通过引入 BSDetector 框架，对样本的质量进行“置信度”评估，实现先过滤后治理。具体展开为，先用 base model 过滤置信度低的样本，再用过滤后的样本微调 base model ， 之后用 fine-tuned 的 model 为每个样本生成新的 response 并进行置信度评估，如果高置信度就替换、否则就过滤

---

**1.提取文章摘要**

（摘自论文第 1 页 Abstract，原样复制）

Large Language Models have become the de facto approach to sequence-to-sequence text generation tasks, but for specialized tasks/domains, a pretrained LLM lacks specific capabilities to produce accurate or well-formatted responses. Supervised fine-tuning specializes a LLM by training it on dataset of example prompts with target responses, but real-world data tends to be noisy. While many fine-tuning algorithms exist, here we consider a data-centric AI perspective on LLM fine-tuning, studying how to systematically curate the training dataset to improve the LLM produced via any fine-tuning algorithm.

We introduce an automated data curation pipeline CLEAR (Confidence-based LLM Evaluation And Rectification) for instruction tuning datasets, that can be used with any LLM and fine-tuning procedure. CLEAR estimates which training data is low-quality and either filters or corrects it. Automatically identifying which data to filter or correct is done via LLM-derived confidence estimates, to ensure only confident modifications to the dataset. Unlike existing data curation techniques, CLEAR is a comprehensive framework that can improve a dataset (and trained model outputs) without additional fine-tuning computations. We don’t assume access to a stronger LLM than the model being fine-tuned (e.g. relying on GPT-4 when fine-tuning GPT-3.5), to see whether CLEAR can meaningfully improve the capabilities of any LLM. Experiments reveal that CLEAR consistently improves the performance of fine-tuned models across many datasets and models (like GPT-3.5 and Llama2).

---

**2.概括文章**

（领域 + 问题/目标 + 方法 + 核心创新/优势）

（对应论文 Introduction / Chapter 1 + Figure 1 + Related Work / Chapter 2）

* **领域**：Data-Centric AI 在 LLM Instruction Tuning / Supervised Fine-Tuning 中的应用（聚焦 sequence-to-sequence text generation 任务，与我们 Qwen 代码补全任务高度同构）。
* **针对问题/目标**：真实 instruction tuning 数据集普遍 noisy（错误/低质响应、prompt 不完整、格式问题等），导致 fine-tuned LLM 输出错误、格式不佳或无关；传统工作聚焦 modeling / fine-tuning 算法迭代，而非系统性迭代数据集本身。
* **使用方法**：提出完整自动化数据治理流水线 **CLEAR**（Confidence-based LLM Evaluation And Rectification），仅依赖待 fine-tune 的同一 LLM（不假设更强 teacher LLM），通过 **Auto-Filter + Auto-Correct** 两阶段实现“敲除不良样本 + 降权/修正可修复样本”。
* **核心创新与优势**：
  * **首次将 BSDetector**（Chen & Mueller, 2023）confidence 估计（而非直接 LLM scoring）用于数据质量评估，保证所有修改均高置信（避免引入新偏差）。
  * **无需额外 fine-tuning 计算**即可显著提升数据集质量；若允许多次 fine-tune，还可迭代 virtuous cycle。
  * **与任何 LLM + 任何 fine-tuning 算法即插即用**（实验覆盖 Llama-2-7b、GPT-3.5-Turbo）。
  * 在 3 个 noisy 基准数据集上，Auto-Filter 已优于原数据集，Auto-Correct 进一步提升，最终 fine-tuned 模型超越 GPT-4 few-shot（Figure 1、Table 2、Chapter 4-5）。

---

**3.详细展开技术路线**

（对应 Chapter 3 + Figure 1 + Figure 2 + Figure 3-5）

CLEAR 流水线分为两个互补阶段，核心均基于 BSDetector 提供的置信度估计（同时考虑 observed consistency + self-reflection certainty，无需访问 LLM 参数，可 API 调用）。

其中，BSDetector 全称 Bad and Speculative Detector，是一个方法/算法框架，正式发表于 ACL 2024 文章，其置信度分数 **C** 是由两个互补因子加权组合得到的，即 **O =** Observed Consistency（观测一致性），以及 **S =** Self-reflection Certainty（自反思确定性）

C = \\alpha \\cdot O + (1 - \\alpha) \\cdot S

具体展开而言，

* **观测一致性 **O** (Observed Consistency)**：给定固定的 prompt，进行多次独立推理（在高温度下），如果给出了相似的答案，说明模型对此知识点的掌握是稳定的。具体公式为

O = \\frac{1}{N(N - 1)} \\sum\_{i \\ne j} \\text{sim}(y\_i, y\_j)

* **自反思确定性 **S** (Self-reflection Certainty)**：将刚才的输入输出拼接起来，直接询问模型这个代码是否正确（即 Yes / No 的 logits）。数学直觉为

S = P\_{\\text{model}}(\\text{Correct} \\mid \\text{Input, Output})

3.1 Auto-Filter 阶段（Chapter 3.1）

1. **输入**：原始 instruction tuning 数据集 **\\Gamma = \\{(x\_i, y\_i)\\}\_{i=1}^n**。
2. 使用 **base pretrained LLM**（未 fine-tune）运行 BSDetector，对每个 **(x\_i, y\_i)** 打置信分 **c\_i \\in [0,1]**。
3. 设定阈值 **\\gamma**（论文实验中取数据集 median confidence），过滤掉 **c\_i < \\gamma** 的样本：
   **F = \\{(x\_i, y\_i) \\mid c\_i > \\gamma\\}**
4. 仅在剩余高置信数据 **F** 上 fine-tune LLM（无需额外计算）。

* **优势**：直接提升数据集质量，实验显示优于 random 50% 过滤和 score-based evaluator（Table 3、Figure 2）。
  3.2 Auto-Correct 阶段（Chapter 3.2）

1. 先在 Auto-Filter 后的数据上 fine-tune 得到 specialized LLM。
2. 对原始数据集中所有 **x\_i**，让该 fine-tuned LLM 生成候选响应 **y'\_i**。
3. 使用 base pretrained LLM（via LLM-as-judge prompt，Table 1）+ BSDetector 判断 **y'\_i** 是否显著优于原始 **y\_i**，得到置信度（threshold **\\eta = 0.8**）。
4. 若置信度 **> \\eta**，则保留 prompt **x\_i**，替换 target 为 **y'\_i**（即修正）；否则直接过滤（Auto-Filter）。
5. 最终得到 curated 数据集，再次 fine-tune（可迭代）。

* **实验验证**：使用 fine-tuned LLM 生成 **y'** 优于直接用 base LLM 生成（Table 4）。

**实例可视化（Figure 3-5，对应 DROP-N、SQuAD-N、Email-N）：**

* **高置信样本**\\rightarrow Keep
* **低置信但 **y'** 优于 **y**（置信 **>0.8**）**\\rightarrow Auto-Correct
* **低置信且 **y'** 也不优**\\rightarrow Auto-Filter（彻底敲除）

---

**4.文章使用的数据和代码链接**

数据集（Chapter 4）：

* **SQuAD-N**：基于 SQuAD (Rajpurkar et al., 2016) 构造的 noisy 版本（20% 样本被随机替换为上下文中的错误句子）。
* **DROP-N**：基于 DROP (Dua et al., 2019) 构造的 noisy 版本（同上扰动方式）。
* **Emails-N**：Enron 邮件分类数据集（主题分类 + 长度适配响应），noisy 版本为随机 swap target response。

**公开链接（论文脚注）：**[HuggingFace - Enron Labeled Emails](https://huggingface.co/datasets/neelblabla/enron_labeled_emails_with_subjects-llama2-7b_finetuning)

代码：

论文正文及附录中**未提供**任何公开代码仓库、GitHub 链接或实现脚本（包括 BSDetector 的复用）。仅描述了实验细节（fine-tuning 超参、温度 0、max tokens 512、Adam LR 1e-5 等），未开源实现。

## 十、正式评测阶段所用数据的获取及处理

这一章记录正式 paper 阶段使用的外部 FIM benchmark。与第三章的 Go single-line 调试场景不同，正式评测阶段不再把 HumanEval-X / MultiPL-E 这类 text-to-code benchmark 转成 derived FIM 作为主表，而是优先选择原生或近原生的 code infilling / FIM benchmark。

正式评测 pipeline 的目标是：

```text
official benchmark raw data
  -> 统一 benchmark test schema
  -> prepare: 构造 ChatML/FIM 模型输入
  -> model predict: 生成 missing completion
  -> postprocess: 回填并包装成官方 evaluator 接收格式
  -> evaluate: 使用官方或兼容 evaluator 计算指标
```

### 1. Benchmark 选定

正式主评测暂定使用两个 benchmark family。


| Benchmark           | 选择原因                                                                                                      | 语言                     | 子任务 / 场景                                                       | 主要指标                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------- | --------------------------------------------------- |
| HumanEval-Infilling | 经典 HumanEval FIM 派生 benchmark，输入天然是`prefix + suffix -> missing code`，可用 unit tests 计算 pass@k。 | Python                   | `single_line`、`multi_line`、`random_span`、`random_span_light`     | pass@1 / pass@k                                     |
| SAFIM               | Syntax-Aware FIM benchmark，专门评估结构化 infilling，包括算法块、控制流表达式和 API 调用。                   | Python / Java / C++ / C# | `algorithmic_block`、`control_flow_expression`、`api_function_call` | official pass@1；当前先用 exact sanity 检查 adapter |

选择原则：

- 任务类型必须是 code completion / code infilling，而不是纯 text-to-code generation。
- benchmark 不应和 CodeSearchNet 训练语料同源，降低数据污染风险。
- benchmark 应尽量有官方 evaluator 或可执行 judge，正式主指标优先使用 pass@k。
- train data 可以从 CodeSearchNet 等 corpus 构造，但 test benchmark 本身不改语义、不改 official id、不改官方测试协议。

### 2. 原始数据与统一 test data

官方原始数据下载后放在：

```text
data/raw_data/
```

从 official raw data 构造出的统一 test data 放在：

```text
data/benchmark/test_data/
```

当前统一后的 5 份正式 benchmark test data 为：

```text
data/benchmark/test_data/humaneval_infilling_python.jsonl   # 8652
data/benchmark/test_data/safim_python.jsonl                 # 1736
data/benchmark/test_data/safim_java.jsonl                   # 4999
data/benchmark/test_data/safim_cpp.jsonl                    # 9901
data/benchmark/test_data/safim_csharp.jsonl                 # 1084
```

构造报告：

```text
data/benchmark/test_data/build_report.json
```

统一 test schema：

```json
{
  "uid": "...",
  "benchmark": "humaneval_infilling|safim",
  "source": "official",
  "language": "python|java|cpp|csharp",
  "task_type": "single_line|multi_line|random_span|random_span_light|algorithmic_block|control_flow_expression|api_function_call",
  "official_task_id": "...",
  "entry_point": "...",
  "prefix": "...",
  "suffix": "...",
  "target": "...",
  "test": "...",
  "unit_tests": "...",
  "official_prompt": "...",
  "official_eval_prompt": "...",
  "raw_fields": {}
}
```

字段说明：

- `uid` 是项目内部唯一 id，用于 prediction 和 test row 对齐。
- `official_task_id` 是官方 evaluator 识别任务的 id，postprocess 时必须保留。
- `prefix / suffix / target` 是 FIM 核心三元组，完整参考代码为 `prefix + target + suffix`。
- `official_prompt / official_eval_prompt / test / unit_tests` 保留官方评测相关信息，方便追溯。
- `raw_fields` 是官方 raw data 的原始字段备份，用于 debug 和审计。

相关构造与分析脚本：


| 文件                                                             | 作用                                                                            |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `scripts/data_process/download_official_fim_benchmarks.py`       | 下载 HumanEval-Infilling 和 SAFIM 官方 raw data。                               |
| `scripts/data_process/build_official_fim_benchmark_test_data.py` | 从 raw data 构造统一 benchmark test jsonl。                                     |
| `scripts/data_process/analyze_official_fim_benchmark_data.py`    | 统计各 benchmark / language / task type 的数量、比例、mask/target 长度分布。    |
| `tools/viz_data/`                                                | 轻量可视化 benchmark 样本，按子任务查看`prefix + [MASK] + suffix` 和 `target`。 |

### 3. Prepare Data：模型输入准备

`prepare` 是模型推理前的数据准备阶段。它不会运行模型，也不会评测，只负责把统一 test schema 转成 ChatML/FIM 输入。

HumanEval-Infilling：

```bash
python scripts/benchmark/evaluate_humaneval_infilling.py prepare
```

SAFIM：

```bash
python scripts/benchmark/evaluate_safim.py prepare
```

输出目录：

```text
data/benchmark/test_data/humaneval_infilling_prepared/
data/benchmark/test_data/safim_prepared/
```

每个 benchmark 会生成两类文件：

```text
*_chatml.jsonl
*_infer_requests.jsonl
*_prepare_report.json
```

其中 `*_chatml.jsonl` 保留结构化 ChatML messages，适合训练、检查、后续 tokenizer/binarize。典型字段：

```text
uid
benchmark
benchmark_name / completion_type
language
task_type
official_task_id
prefix
suffix
target
messages
only_last_turn_loss
```

`messages` 的模板为：

```text
system:
You are a {language} code infilling assistant.

user:
Fill the [MASK] in the {language} code. Return only the missing {language} code, without Markdown fences or explanation.

* Incomplete Code:
{prefix}[MASK]{suffix}
```

`*_infer_requests.jsonl` 是 ChatML messages 已经 render 后的推理请求文本，还不是二进制文件。典型字段：

```text
uid
benchmark
benchmark_name / completion_type
language
task_type
official_task_id
prefix
suffix
target
prompt
max_new_tokens
```

更严谨的模型输入链路是：

```text
*_chatml.jsonl
  -> render ChatML prompt
  -> *_infer_requests.jsonl
  -> tokenizer(prompt)
  -> input_ids / attention_mask tensor
  -> model.generate
```

当前 tokenizer 和 tensor 构造是在推理脚本运行时即时完成；如果后续需要加速全量评测，可以再增加 `*.pt` / `*.mmap` 形式的 binary cache。

### 4. Model Predict：模型生成 completion

统一推理脚本：

```text
scripts/benchmark/generate_official_fim_predictions.py
```

输入：

```text
data/benchmark/test_data/humaneval_infilling_prepared/humaneval_infilling_infer_requests.jsonl
data/benchmark/test_data/safim_prepared/safim_infer_requests.jsonl
```

输出 prediction jsonl，例如：

```text
outputs/benchmark/eval_results/base_qwen_7b/humaneval_predictions.jsonl
outputs/benchmark/eval_results/base_qwen_7b/safim_predictions.jsonl
```

prediction schema：

```json
{
  "uid": "...",
  "prediction": "...",
  "predictions": ["..."],
  "model_name_or_path": "models/Qwen2.5-Coder-7B-Instruct"
}
```

说明：

- `uid` 用于和统一 test data 对齐。
- `prediction` 是第一条生成结果。
- `predictions` 是多采样列表，用于 pass@k。
- 当前默认是 greedy `temperature=0.0`、`num_return_sequences=1`；如果需要 pass@10，需要设置 `NUM_RETURN_SEQUENCES=10` 并开启采样。

基模全量评测已封装为两个 shell 脚本：

```text
scripts/benchmark/run_humaneval_infilling_eval.sh
scripts/benchmark/run_safim_eval.sh
```

默认模型：

```text
models/Qwen2.5-Coder-7B-Instruct
```

默认输出：

```text
outputs/benchmark/eval_results/base_qwen_7b/
```

推荐后台运行：

```bash
nohup bash scripts/benchmark/run_humaneval_infilling_eval.sh \
  > runs/base_qwen_7b_humaneval_full.log 2>&1 &

nohup bash scripts/benchmark/run_safim_eval.sh \
  > runs/base_qwen_7b_safim_full.log 2>&1 &
```

### 5. Output Postprocess：包装成官方 evaluator 输入

postprocess 是模型输出后的格式转换。它不再给模型看，而是把 `prediction` 映射回官方 evaluator 接收的 submission/sample 格式。

HumanEval-Infilling：

```bash
python scripts/benchmark/evaluate_humaneval_infilling.py postprocess \
  --predictions-path outputs/benchmark/eval_results/base_qwen_7b/humaneval_predictions.jsonl \
  --out-dir data/benchmark/test_data/humaneval_infilling_base_qwen_7b_official_samples
```

SAFIM：

```bash
python scripts/benchmark/evaluate_safim.py postprocess \
  --predictions-path outputs/benchmark/eval_results/base_qwen_7b/safim_predictions.jsonl \
  --out-dir data/benchmark/test_data/safim_base_qwen_7b_official_samples
```

official samples schema：

```json
{
  "task_id": "...",
  "completion": "..."
}
```

转换逻辑：

```text
uid
  -> 找回 test row
  -> 读取 official_task_id
prediction / predictions
  -> 清洗 special tokens / Markdown fences
  -> 保留必要缩进和换行
  -> 写成 official completion
```

这里特别注意：Python benchmark 的 `target` 和模型输出常常带前导缩进，清洗时不能随意 `.strip()`，否则会破坏语法，导致 HumanEval official evaluator 报 syntax error。

### 6. Evaluate：正式评测与 oracle 检查

#### HumanEval-Infilling

HumanEval-Infilling 已经可以直接使用官方 unit-test evaluator。

评测命令：

```bash
python scripts/benchmark/evaluate_humaneval_infilling.py eval \
  --samples-dir data/benchmark/test_data/humaneval_infilling_base_qwen_7b_official_samples \
  --out-dir outputs/benchmark/eval_results/base_qwen_7b/humaneval_infilling \
  --k 1 \
  --n-workers 16 \
  --timeout 3.0 \
  --no-reuse-existing
```

输出：

```text
outputs/benchmark/eval_results/base_qwen_7b/humaneval_infilling/humaneval_eval_summary.json
```

oracle 验证已经通过：

```text
outputs/benchmark/eval_results/humaneval_infilling_oracle/humaneval_eval_summary.json
```

结果为：

```text
single-line         pass@1 = 1.0, pass@10 = 1.0
multi-line          pass@1 = 1.0, pass@10 = 1.0
random-span         pass@1 = 1.0, pass@10 = 1.0
random-span-light   pass@1 = 1.0, pass@10 = 1.0
```

这说明 HumanEval-Infilling 的 `prepare -> oracle prediction -> postprocess -> official evaluator` 链路自洽。

#### SAFIM

SAFIM 当前有两层 evaluator：

1. `eval-exact`：轻量 sanity evaluator，只检查 `sanitize(prediction) == target`。
2. `run-official`：调用 SAFIM 官方 `evaluate.py`，需要额外 ExecEval Docker 执行服务。

`eval-exact` 命令：

```bash
python scripts/benchmark/evaluate_safim.py eval-exact \
  --predictions-path outputs/benchmark/eval_results/base_qwen_7b/safim_predictions.jsonl \
  --out-dir outputs/benchmark/eval_results/base_qwen_7b/safim_exact \
  --k 1
```

oracle exact 验证已经通过：

```text
outputs/benchmark/eval_results/safim_oracle_pass1/safim_exact_eval_summary.json
outputs/benchmark/eval_results/safim_oracle_pass10/safim_exact_eval_summary.json
```

结果为：

```text
all: 17720 / 17720, pass_rate = 1.0
```

这说明 SAFIM 的 `prepare -> oracle prediction -> postprocess/exact check` adapter 是自洽的。

但是，SAFIM official evaluator 暂时作为遗留项放置。原因是 SAFIM 官方评测不是纯 Python 脚本，它通过 ExecEval 服务执行多语言代码：

```text
SAFIM evaluate.py
  -> exec_utils.py
  -> http://localhost:5000/api/execute_code
  -> ExecEval Docker service
  -> 编译/运行 Python, C++, Java, C# 代码
```

官方 README 要求额外环境：

```text
tree-sitter parser
datasets
Docker
ExecEval daemon
```

目前尝试构建 ExecEval 官方 Docker 镜像时，连续在外部下载依赖处超时：

```text
Oracle JDK download timeout
Go 1.19.2 download timeout
```

这不是 SAFIM data adapter 的问题，而是 ExecEval 官方 Dockerfile 是全语言执行环境，包含 Go、Kotlin、PyPy、Node、Rust、Ruby、PHP、libseccomp 等大量下载项；其中 Go 等环境并不是 SAFIM 必需项，但仍会阻塞镜像构建。

后续解决方向：

- 短期：继续使用 `eval-exact` 做 adapter sanity，不把它作为论文最终 SAFIM official 指标。
- 中期：构建 `Dockerfile.safim-lite`，只保留 SAFIM 需要的 Python / C++ / Java / C# 环境，并将 SAFIM Python runtime 从 `PyPy 3` 调整为 `Python 3`，先跑通 official evaluator。
- 长期：如果论文需要完全复现 SAFIM leaderboard 协议，应使用官方 ExecEval 环境，或在文中明确说明使用了 safim-lite execution backend。

### 7. Official Evaluator 获取方式

`official_evaluators` 不建议作为普通项目源码提交到 git。原因是它们本质上是上游官方仓库快照，包含较多外部代码、数据和 tree-sitter 语法源码；如果直接提交，会增加仓库体积，也容易引入嵌套 `.git`、缓存文件和无关 assets。当前 `.gitignore` 可以继续忽略：

```text
scripts/benchmark/official_evaluators
```

需要复现实验时，再按下面命令在本地下载到约定目录。

#### HumanEval-Infilling evaluator

官方链接：

```text
https://github.com/openai/human-eval-infilling
```

下载命令：

```bash
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
mkdir -p scripts/benchmark/official_evaluators

rm -rf scripts/benchmark/official_evaluators/human_eval_infilling

git clone --depth 1 \
  https://github.com/openai/human-eval-infilling.git \
  scripts/benchmark/official_evaluators/human_eval_infilling

rm -rf scripts/benchmark/official_evaluators/human_eval_infilling/.git
```

注意：HumanEval-Infilling 官方 `execution.py` 默认会把真正执行 generated code 的 `exec(...)` 注释掉，要求使用者先阅读安全提示。若要运行 official unit tests，需要确认安全风险后启用执行逻辑。我们当前 wrapper 使用的是：

```text
scripts/benchmark/official_evaluators/human_eval_infilling/human_eval_infilling/evaluation.py
scripts/benchmark/official_evaluators/human_eval_infilling/human_eval_infilling/execution.py
scripts/benchmark/official_evaluators/human_eval_infilling/data/*.jsonl.gz
```

#### SAFIM evaluator

官方链接：

```text
https://github.com/gonglinyuan/safim
https://huggingface.co/datasets/gonglinyuan/safim
https://arxiv.org/abs/2403.04814
```

下载命令：

```bash
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
mkdir -p scripts/benchmark/official_evaluators

rm -rf scripts/benchmark/official_evaluators/safim

git clone --depth 1 \
  https://github.com/gonglinyuan/safim.git \
  scripts/benchmark/official_evaluators/safim

rm -rf scripts/benchmark/official_evaluators/safim/.git
```

SAFIM 需要 tree-sitter parser，可按官方说明构建：

```bash
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function/scripts/benchmark/official_evaluators/safim

python -m pip install \
  "Jinja2==3.1.2" \
  "openai==0.28.1" \
  "tiktoken==0.5.2" \
  "tqdm==4.64.1" \
  "tree-sitter==0.20.4" \
  "requests==2.28.1" \
  "datasets==2.18.0"

bash setup_tree_sitter.bash
```

SAFIM official evaluator 还依赖 ExecEval execution backend。官方链接：

```text
https://github.com/ntunlp/ExecEval
```

官方推荐方式是：

```bash
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
mkdir -p external
cd external

git clone --depth 1 https://github.com/ntunlp/ExecEval.git
cd ExecEval

docker build . -t exec-eval:1.0

docker run --rm \
  -p 5000:5000 \
  -e NUM_WORKERS=8 \
  exec-eval:1.0
```

当前遗留问题是：ExecEval 官方 Dockerfile 是全语言执行环境，包含 Go、Kotlin、PyPy、Node、Rust、Ruby、PHP、libseccomp 等大量下载项；在当前机器网络环境下已经出现 Oracle JDK 和 Go 下载超时。因此 SAFIM official evaluator 暂缓，短期只使用 `eval-exact` 验证 adapter 自洽。

### 8. 代码文件作用汇总


| 文件                                                          | 作用                                                                                                                |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `scripts/benchmark/benchmark_official_common.py`              | 公共工具：JSONL 读写、task type 映射、ChatML render、prediction 清洗、prediction map 加载。                         |
| `scripts/benchmark/evaluate_humaneval_infilling.py`           | HumanEval-Infilling wrapper：`prepare`、`make-oracle`、`postprocess`、`eval`、`eval-subset`。                       |
| `scripts/benchmark/evaluate_safim.py`                         | SAFIM wrapper：`prepare`、`make-oracle`、`postprocess`、`eval-exact`、`run-official`。                              |
| `scripts/benchmark/generate_official_fim_predictions.py`      | 通用 HF 基模推理入口，读取`*_infer_requests.jsonl`，输出 predictions jsonl。                                        |
| `scripts/benchmark/run_humaneval_infilling_eval.sh`           | HumanEval-Infilling 全流程 shell：prepare(auto) -> generate -> postprocess -> evaluate。                            |
| `scripts/benchmark/run_safim_eval.sh`                         | SAFIM 全流程 shell：prepare(auto) -> generate -> postprocess -> eval-exact；可选`RUN_OFFICIAL=1` 跑官方 evaluator。 |
| `scripts/benchmark/official_evaluators/human_eval_infilling/` | 本地下载的 HumanEval-Infilling 官方 evaluator，可由上文命令恢复；默认不建议提交到 git。                             |
| `scripts/benchmark/official_evaluators/safim/`                | 本地下载的 SAFIM 官方 evaluator，可由上文命令恢复；默认不建议提交到 git。                                           |

### 9. train data 准备

本节记录正式 benchmark 阶段对应的训练数据获取与处理思路。下文统一使用简称：

```text
humaneval = HumanEval-Infilling
safim     = SAFIM
```

当前 benchmark pipeline 状态：

```text
humaneval:
  prepare / postprocess / official evaluator 均已跑通。
  oracle pass@1/pass@10 全为 1。
  正式只测 single_line 和 multi_line，不使用 random_span / random_span_light。

safim:
  prepare / postprocess / exact oracle 已跑通。
  official evaluator 依赖 ExecEval Docker，当前因外部依赖下载超时暂缓。
```

因此，train data 的构造目标不是复用 Go single-line FIM 调试数据，而是分别围绕 `humaneval` 和 `safim` 构造 train-test task match 的 FIM mixture。benchmark test data 本身不改 official id、不改测试协议；训练数据可以从其他高质量代码数据中派生，但派生后的输入输出形式必须贴近目标 benchmark。

#### 9.1 正式 train mixture 的当前决策

正式训练数据分成两个 benchmark family：

```text
humaneval train:
  data source: SelfCodeAlign seed function pool (`bigcode/python-stack-v1-functions-filtered-sc2`)
  language: Python
  task ratio: single_line 15%, multi_line 85%

safim train:
  data source: The Stack / The Stack v2
  language: Python, Java, C++, C#
  task ratio: algorithmic_block 45%, control_flow_expression 45%, api_function_call 10%
```

比例依据：

- `humaneval` 正式只测 `single_line` 和 `multi_line`。当前 benchmark 统计中二者比例约为 `15.09% / 84.91%`，因此训练侧按 `15% / 85%` 对齐；`random_span` 和 `random_span_light` 质量不稳定，正式训练 mixture 中先不纳入。
- `safim` benchmark 中 `algorithmic_block` 和 `control_flow_expression` 占绝大多数，`api_function_call` 尤其在 Java / C++ / C# 中占比很低。训练侧使用 `45% / 45% / 10%`，相当于对 API call 做轻度 oversampling，避免该子任务在训练中被完全淹没；评测时仍按 official benchmark 原始分布和 per-task breakdown 报告。

训练产物建议统一放在：

```text
data/benchmark/train_data/humaneval/
data/benchmark/train_data/safim/
```

每条训练链路至少产出三层文件：

```text
canonical jsonl:
  prefix / target / suffix / language / task_type / source_dataset / raw_meta

chatml jsonl:
  messages / prefix / target / suffix / only_last_turn_loss

compact annotated jsonl:
  input_ids / label / attention_edges / annotation_meta
```

其中 canonical 是数据真相层，ChatML 是模型输入层，compact annotated 是 ours causality curation 的训练层。

#### 9.2 humaneval train data 获取与处理

`humaneval` 是 Python-only、function-level FIM benchmark。正式评测只关注：

```text
single_line
multi_line
```

当前主训练源改为 SelfCodeAlign 的 documented Python seed function pool：

```text
bigcode/python-stack-v1-functions-filtered-sc2
```

选择它的原因是：这份数据已经由 SelfCodeAlign 从 The Stack Python 中抽取 top-level functions with docstrings，并经过 return filtering、import inference、Pyright type checking、benchmark contamination filtering 和 StarCoder2 docstring/code quality filtering。它比从 `MBPP + APPS` 人工拼 docstring 更贴近 `humaneval` 的 documented function 分布，也能避免 APPS 题面过长、风格偏竞赛题的问题。

`MBPP + APPS` 仍可作为 backup / ablation source，但不再作为 humaneval 主训练源。

原始数据建议放在：

```text
data/raw_data/selfcodealign_python_functions/
```

派生后的训练数据仍覆盖到统一路径：

```text
data/benchmark/train_data/humaneval_python_canonical.jsonl
data/benchmark/train_data/humaneval_python_chatml.jsonl
data/benchmark/train_data/humaneval_python_build_report.json
```

构造流程：

1. 下载 `bigcode/python-stack-v1-functions-filtered-sc2` 到本地 raw data 目录。
2. 读取每条样本的 `content` 字段，作为一个 documented Python function。
3. 使用 `ast.parse` 验证函数可解析，并保留原始 docstring；不再从 APPS / MBPP 题面重构 docstring。
4. 对 `content` 做轻量二次去污染：和当前 `humaneval` official test 的 `prefix + target + suffix` 做 normalized full-code exact hash 匹配，命中则丢弃。
5. 构造 `single_line`：mask 函数体中的一整行有效 statement，优先 assignment、return、API call、条件内部关键更新语句。
6. 构造 `multi_line`：mask 连续多行 statement 或完整 block，优先循环体、if/else body、try/with body、连续状态更新逻辑。
7. 验证 `prefix + target + suffix == full_code`，并用 `ast.parse` 检查回填后的 Python 代码仍可解析。
8. 做长度过滤，避免训练时被 tokenizer 截断；当前默认 `prompt <= 6000 chars`、`full_code <= 8000 chars`。
9. 按 `single_line 15% / multi_line 85%` 采样，保留 `source_dataset`、`task_type`、`raw_task_id`、`target_line_range` 等字段。
10. 渲染为与 benchmark prepare 阶段一致的 ChatML `[MASK] -> assistant target` 格式。
11. 后续调用 `scripts/benchmark/annotate_humaneval_train.py` 完成结构边、LLM 语义补边、docstring fact 边补充，并映射为 Qwen BPE compact annotated 格式。

推荐主构造命令：

```bash
python scripts/data_process/build_humaneval_train_data.py \
  --source selfcodealign \
  --out-dir data/benchmark/train_data \
  --max-source-rows 8000 \
  --max-samples 10000 \
  --docstring-mode preserve \
  --max-prompt-chars 6000 \
  --max-full-chars 8000
```

当前标注策略采用“结构规则 + LLM agent + docstring fact v2”的折中版本：

```text
结构规则:
  bracket / defuse / call / return / type 等可以由 AST 或 tokenizer 稳定捕捉的边，不交给 LLM 重复判断。

LLM agent:
  只补充 dataflow / semantic / api 等需要语义判断的代码边。

docstring fact v2:
  对 humaneval 的 documented function docstring 做确定性补边，重点覆盖 Args / Parameters / Returns / @param / @return 中的 type 和少量 semantic 信息。
  Examples / doctest / Traceback 不参与 docstring fact，避免示例文本污染训练监督。
```

全量 10K 标注建议直接覆盖标准输出路径：

```text
data/benchmark/train_data/humaneval_python_compact_annotated.jsonl
data/benchmark/train_data/humaneval_python_annotation_cache.jsonl
```

由于 cache 中保存的是已经加过 docstring edges 的 compact record，如果此前用旧 docstring 规则标过抽检样本，全量重标时应使用 `--overwrite-cache`，确保 10K 样本全部按当前 docfact v2 规则重新生成。

推荐 10K 全量标注命令：

```bash
nohup python scripts/benchmark/annotate_humaneval_train.py \
  --input-path data/benchmark/train_data/humaneval_python_chatml.jsonl \
  --output-path data/benchmark/train_data/humaneval_python_compact_annotated.jsonl \
  --annotation-cache-path data/benchmark/train_data/humaneval_python_annotation_cache.jsonl \
  --model-name-or-path models/Qwen2.5-Coder-7B-Instruct \
  --samples-per-task 0 \
  --task-types single_line multi_line \
  --selection-seed 42 \
  --annotation-mode agent \
  --num-workers 12 \
  --max-rounds 6 \
  --max-teacher-edges 128 \
  --docstring-edges \
  --docstring-type-edges \
  --no-docstring-llm-edges \
  --max-docstring-edges 120 \
  --overwrite-cache \
  --flush-every 50 \
  > runs/benchmark/annotate_humaneval_python_10k_docfact_v2.log 2>&1 &
```

#### 9.3 safim train data 获取与处理

当前 `safim` train data 的目标已经从早期方案收敛为：**用 The Stack v1 的真实多语言 file-level 源码，构造和 SAFIM official test 对齐的 `[MASK]` 补全样本；同时做必要的长度、注释、license、去污染和噪声过滤，保证训练数据高质量、无明显 test 泄漏。** 这一节只记录 train data 获取与处理，不写后续 annotation 方案。

SAFIM official test 是 syntax-aware file-level FIM，当前重点对齐三类子任务：

```text
algorithmic_block
control_flow_expression
api_function_call
```

和 `humaneval` train data 不同，`humaneval` 更偏 Python function-level docstring/function infilling，而 `safim` 是 Python / Java / C++ / C# 四语言 file-level / script-level infilling。因此 `safim` train 样本不能只截取孤立函数签名或很短函数片段，而要尽量保留 target 周围的真实文件级上下文，例如 import/include/using、类和函数外壳、局部变量定义、前后状态更新、API receiver/type 信息以及后续使用。

当前落地版本使用 The Stack v1，而不是 The Stack v2。原因是 The Stack v1 在 Hugging Face 数据集中直接包含源码 `content` 字段，可以 streaming 读取并边读边过滤；The Stack v2 更新、更规范，但 HF 侧主要是 metadata / SWHIDs，真实源码内容还需要 Software Heritage S3 访问权限，不适合当前快速闭环。正式构造不全量下载四语言原始数据，而是按语言 streaming 读取，过滤后只缓存被用到的 source pool：

```text
data/raw_data/the_stack_v1/safim_source_pool/python_sources.jsonl
data/raw_data/the_stack_v1/safim_source_pool/java_sources.jsonl
data/raw_data/the_stack_v1/safim_source_pool/cpp_sources.jsonl
data/raw_data/the_stack_v1/safim_source_pool/csharp_sources.jsonl
```

最终训练数据统一放在：

```text
data/benchmark/train_data/safim/safim_python_train_canonical.jsonl
data/benchmark/train_data/safim/safim_java_train_canonical.jsonl
data/benchmark/train_data/safim/safim_cpp_train_canonical.jsonl
data/benchmark/train_data/safim/safim_csharp_train_canonical.jsonl
data/benchmark/train_data/safim/safim_*_train_chatml.jsonl
data/benchmark/train_data/safim/safim_train_canonical.jsonl
data/benchmark/train_data/safim/safim_train_chatml.jsonl
data/benchmark/train_data/safim/build_report.json
```

当前已构造规模是每个语言 `10K`，总计 `40K`。语言内部子任务比例固定为：

```text
algorithmic_block        45%  每语言 4500
control_flow_expression  45%  每语言 4500
api_function_call        10%  每语言 1000
```

这个比例是有意设计的：SAFIM official 中 `algorithmic_block` 和 `control_flow_expression` 是主体，`api_function_call` 数量很少；训练侧给 API call 10% 的轻度 oversampling，可以避免 API 子任务完全被淹没，同时仍然保持整体任务形态接近 official benchmark。

需要注意，`source_pool` 的行数会小于最终样本数，这是正常现象。当前 builder 允许一个高质量源文件贡献多个不重叠 FIM span，最多对应三类 subtask 各一条，因此一个 source file 最多产出 3 条训练样本。当前 10K 构造中大致是每语言约 6K 个 source file 产出 10K 条样本，例如：

```text
python:  6125 sources -> 10000 samples, max 3 samples/source
java:    6336 sources -> 10000 samples, max 3 samples/source
cpp:     6271 sources -> 10000 samples, max 3 samples/source
csharp:  5945 sources -> 10000 samples, max 3 samples/source
```

这和 SAFIM 的 file-level 设置是匹配的：一个真实文件中不同位置、不同语法结构都可以成为有效 FIM 监督。为了避免单个文件刷出过多近似样本，当前上限自然受三类 task 限制，不会从同一文件密集切出几十条样本。

文件清洗先在源文件级别做，目标是尽早排除不适合 SAFIM FIM 的低质文件。当前主要过滤包括：

- 只保留 Python / Java / C++ / C# 四种语言。
- 默认要求 permissive license，避免 unknown / non-permissive license 混入。
- 按官方 SAFIM 长度分布约束样本上下文，使用 `official_p95` profile 控制 `prefix + suffix` 长度。
- 源文件长度和行数做上下限过滤，默认 `min_file_chars=800`、`max_file_chars=12000`、`min_file_lines=20`。
- 过滤 generated / vendored / config / fixture / benchmark 路径，以及超长单行、平均行长异常、重复文件。
- 过滤明显与 SAFIM official test 相关的 repo/path，例如 `safim`、`human-eval`、`humaneval`、`benchmark` 等。
- 对 full code 和 target 做 normalized hash 去污染，避免 exact overlap。

语言特定过滤也已经落地：Python 过滤纯配置或无逻辑脚本；Java 过滤纯 POJO、interface-only 和 annotation-heavy 文件；C++ 过滤只有声明的 header 或无函数体文件；C# 过滤纯 model / auto-property 风格文件。The Stack 文件开头经常有 license banner 或超长说明块，而 SAFIM official test 通常没有这种开局大块注释，所以当前默认会剥离**文件开头的大块注释 / license header / shebang / Python module docstring**，再计算长度、抽 span 和构造样本；中间的普通注释会保留，因为它属于真实 file-level 上下文。

候选 span 抽取遵循 `prefix + target + suffix == full_code`。Python 使用 `ast` 抽取稳定语法 span；Java / C++ / C# 当前使用保守的文本、brace 和 statement scanner，而不是完整 AST/parser，原因是三种语言的工程化 AST 依赖更重，且 The Stack 单文件常缺少 project/classpath 上下文。当前策略优先保证 span 边界保守、target 可读、任务类型贴近 SAFIM；如果单个脏文件触发 scanner 异常，会计入 `candidate_exception:*` 并跳过，不让整轮构造中断。

三类子任务的抽取原则如下：

- `algorithmic_block`：抽取 loop/if/else/try/catch 等结构内的短 block，或函数/方法内连续 1-5 行 statement group。优先 assignment、augmented assignment、return、collection update、accumulator update 等有真实状态变化的片段。过滤空白、纯注释、单独括号、极低信息 target。
- `control_flow_expression`：抽取 `if` / `while` / `for` / `switch` / `catch` 等控制结构里的 condition、iterable 或控制表达式。要求 target 是完整表达式，不能切半个 token 或破坏括号结构。
- `api_function_call`：抽取完整 call expression / method invocation / constructor/factory call。优先带 receiver、namespace、module 或类型上下文的调用，例如 `torch.arange(...)`、`obj.method(...)`、`Files.readAllBytes(...)`。过滤 `print/log/debug/assert/len/str/super` 等低价值调用。

上下文长度不是随意截断，而是参考 official SAFIM 的实际长度分布。当前 `official_p95` profile 记录的 `prefix + suffix` 上限为：

```text
python:  block 2511, control 2404, api 3682
java:    block 2900, control 2942, api 4540
cpp:     block 1877, control 1972, api 4444
csharp:  block 10922, control 10991, api 3170
```

因此 train 样本整体长度会贴近 official test，而不会被 The Stack 中很长的工业文件拖大。构造后用 viewer 做人工对比，确认 train 和 official test 在 file-level 上下文、mask 位置、target 长度和任务类型上基本对齐。

当前构造命令为：

```bash
nohup python scripts/data_process/build_safim_train_data.py \
  --samples-per-language 10000 \
  --task-ratios algorithmic_block=0.45,control_flow_expression=0.45,api_function_call=0.10 \
  --out-dir data/benchmark/train_data/safim \
  --test-dir data/benchmark/test_data \
  --prepare-viewer-data \
  --viewer-dir outputs/benchmark/safim_stack_probe/viewer_data \
  --length-profile official_p95 \
  --strip-leading-comments \
  --seed 42 \
  > runs/build_safim_train_10k.log 2>&1 &
```

构造完成后生成 paired viewer，用于逐语言、逐 subtask 对比左侧 train 和右侧 official test：

```bash
python tools/viz_data/build_benchmark_subtask_viewer.py \
  --test-dir outputs/benchmark/safim_stack_probe/viewer_data \
  --out outputs/benchmark/safim_stack_probe/safim_train_vs_official_paired_viewer.html \
  --samples-per-task 10 \
  --seed 42 \
  --paired-safim \
  --task-types algorithmic_block control_flow_expression api_function_call \
  --title "SAFIM Train 10K vs Official Paired Viewer"
```

人工检查重点是：`[MASK]` 是否自然、target 是否完整、task type 是否对标、开头 license/banner 是否已去除、是否仍有题面文字或非代码噪声、train/test 长度是否明显错位。当前版本没有强制接入多语言编译检查；原因是 Python 可以 `ast` 检查，但 Java/C++/C# 单文件往往缺少 project 依赖，直接编译会误杀很多真实源码。后续如果发现 C++/Java/C# 噪声仍偏多，可以增加可选 `--syntax-check` 或 tree-sitter parser sanity，但不作为当前 10K 版本的前置条件。

#### 9.4 后续优先级

当前不建议马上构造很大的 train mixture。更稳妥的交接路径是：

1. 先为 `humaneval` 构造 MBPP + APPS 小规模 mixture，例如 1k-5k 条，确认 `15% / 85%` 比例、ChatML render、annotation 和 compact 格式都自洽。
2. 使用已有 humaneval official evaluator 跑一次 base / CE / ours 的小规模闭环，因为这一条 evaluation 链路目前最完整。
3. 再为 `safim` 构造 The Stack 小规模 multilingual mixture，每种语言先各取少量样本，重点检查 AST mask 是否真的贴近 `algorithmic_block / control_flow_expression / api_function_call`。
4. 对 train data 构建轻量 viewer 或抽样报告，人工检查 `prefix + [MASK] + suffix`、`target`、`task_type`、annotation edge 是否合理。
5. 小规模训练和评测方向正确后，再扩大正式训练规模；扩大前需要固定 random seed、source split、dedup 规则和 build report。
6. 如果 `humaneval` 的 multi-line 质量不足，再考虑加入 CodeContests / Project CodeNet 的 Python accepted solution 作为补充；如果 `safim` 的 algorithmic block 不足，再考虑专门从 CodeContests 补充算法块样本。
