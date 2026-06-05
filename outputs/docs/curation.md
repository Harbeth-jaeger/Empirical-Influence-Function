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

## 二、相关代码文件结构及其作用

这一节只列和 causality curation 标注、数据构造、训练诊断、评测可视化强相关的代码。整体上可以按六条线理解：

```text
数据处理 scripts/data_process, scripts/go_single
  -> 标注 src/annotate, tools/visual_annotation, scripts/go_single/annotate_*.py
  -> 训练 src/train
  -> 小样本 saliency 实验 scripts/saliency_exp
  -> 训练前后 saliency 对齐可视化 tools/visual_saliency
  -> benchmark / in-domain 错误案例可视化 tools/visual_failure
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

### 5. `scripts/go_single/`：Go single-statement 场景脚本


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

### 6. `tools/visual_annotation/`：标注边质量可视化


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

### 7. `tools/visual_failure/`：错误案例与 benchmark 失败分析


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

### 8. `tools/visual_saliency/`：训练前后 saliency 与 annotation 对齐


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

## 三、数据获取与处理

这一章把数据分成三层：train data、internal test data、external test benchmark。三者不要混用，因为它们回答的问题不同。

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

- 不改 benchmark 原始数据和官方 test。
- 只额外生成我们的 ChatML/FIM 推理输入，保留官方 id，方便 prediction 回填。
- 不做 causality curation annotation，也不报告 saliency alignment。
- 主报告 completion quality，尤其是 pass@k。
- 如果 benchmark 没有 unit test，则只能作为补充相似度评测，不能作为 external 主结果。

推荐 external 主表优先使用 execution benchmark。


| Benchmark                     | 语言        | 是否有 unit test / judge | 推荐主指标    | 备注                                              |
| ----------------------------- | ----------- | ------------------------ | ------------- | ------------------------------------------------- |
| MCEval-Go / Go subset         | Go          | 有                       | pass@1/pass@k | 和当前 Go completion 场景最接近，适合短期主结果。 |
| xCodeEval-Go / ExecEval       | Go / 多语言 | 有                       | pass@1/pass@k | 更标准的可执行代码评测，适合外部 benchmark。      |
| HumanEval / MBPP 类 benchmark | Python 为主 | 有                       | pass@1/pass@k | 权威但语言和当前 Go 训练不完全一致，可作补充。    |

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

### 5. 数据处理与评测输出约定

推荐目录组织：

```text
data/go_single/train_data/          # train canonical/chatml/compact annotated
data/go_single/eval_data/           # internal eval canonical/chatml/compact annotated
data/go_single/raw_data/            # 原始 benchmark 或 raw corpus
outputs/go_single/<benchmark>/       # prediction、result、report、viewer
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

标注 tokenizer 会过滤普通注释和 docstring，但保留 C/C++ 预处理指令，因为 `#include`、`#define` 等可能真实影响代码依赖和符号定义。

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


| 方法                    | 分类                            | 核心思路                                                                                                          |
| ----------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Base Qwen               | 不训练/不治理对照               | 直接评测 Qwen2.5-Coder-1.5B-Instruct，作为基础能力下限。                                                          |
| TokenCleaning           | token-level hard mask           | 用 base model 与 reference model 的 token loss 差异打分，保留 top ratio token，其余 supervised label 置为`-100`。 |
| XTF                     | token-level hard mask           | 用 attention relevance、PCP novelty、embedding task relevance 三类信号过滤噪声 token。                            |
| LLM-CleanCode           | sample/span-level rewrite       | 用 LLM 对 assistant completion 做 rename、modularize、planning 风格清洗，然后再 SFT。                             |
| CLEAR                   | sample-level filter/correct     | 用 observed consistency 和 self-reflection 估计样本质量，低置信样本过滤，高置信候选可替换原 target。              |
| IB-FT                   | loss-level optimization         | 不主要改数据，而是在训练时加 variational bottleneck 辅助 loss。                                                   |
| Ours Causality Cuartion | annotation edge + saliency loss | 先标注 token edge，再训练时约束标注边的 contribution saliency 高于非标注边。                                      |

需要注意：`scripts/benchmark` 里仍保留一个旧的 `graph_signal_operator`，它属于 annotation graph hard-mask/soft-weight 数据治理算子；当前我们主要使用的是 `src/annotate` 生成 `attention_edges`，再由 `src/train` 的 saliency loss 训练。
