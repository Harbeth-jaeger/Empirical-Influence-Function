# 可视化工具说明

本文说明当前项目中三类前端可视化工具的目的、数据构造方式、数据格式、文件结构和主要代码逻辑。当前可视化分为三类：annotation edge 可视化、saliency 分布可视化、failure case 对比可视化。

## 一、总体设计

三类可视化都采用静态 HTML viewer 的形式。也就是说，最终展示页面是一个 `.html` 文件，浏览器打开后即可交互浏览；构建时会把需要展示的数据嵌入 HTML，或者从旁边的 JSON 数据构建为自包含 HTML。

这种方式和 React/Vite 动态前端不同：

- 不需要启动后端服务。
- 不需要数据库。
- 不需要浏览器向 API 请求数据。
- 适合把实验结果打包发给导师、同学或远程服务器展示。

如果通过 `python -m http.server` 访问，本质上只是用一个轻量静态文件服务器把 HTML 文件暴露出来；页面本身没有后端逻辑。

## 二、Annotation Edge Viewer

### 1. 目的

Annotation viewer 用来检查我们方法中的标注边质量。它展示一条训练样本的 token 序列，并把 annotation edge 画成 token 之间的连线。通过这个界面可以直观看到：

- 哪些 token 被标注为 source token。
- 哪些 completion token 被标注为 target/query token。
- 边类型是什么，例如 `call`、`return`、`type`、`dataflow`、`semantic` 等。
- 标注边是否过密、过稀，或者是否漏掉了关键依赖关系。

### 2. 数据来源

Annotation viewer 的输入来自已经标注好的训练数据，例如：

```text
data/benchmarks/sft_data/ours_graphsignal_train.json
```

该文件由 annotation pipeline 生成，核心字段包括：

```text
input_ids
label
length
attention_edges
messages / chatml / fim 相关辅助字段
```

其中最关键的是 `attention_edges`。每条边通常包含：

```text
src      source token index
dst      target/query token index
subtype  relation type
```

在可视化构建时，脚本会重新使用 Qwen tokenizer，把 ChatML 序列映射为 token，并把 word-level / token-level annotation edge 对齐到展示 token 上。

### 3. 当前展示数据

组会展示过的随机样本 viewer 是：

```text
outputs/visual_annotation/random5_from500/viewer.html
```

它来自前 500 条带 LLM 语义边的已标注训练数据，每种语言随机抽取若干条样本。这个 HTML 是自包含文件，数据已经嵌入页面中。

### 4. 代码结构

```text
tools/visual_annotation/
```

主要文件作用：

| 文件 | 作用 |
| --- | --- |
| `annotate_benchmark_fim_train.py` | 适配 benchmark 的 ChatML-FIM 训练数据，调用 annotation pipeline 生成 GraphSignal 训练数据。 |
| `build_dynamic_annotation_viewer.py` | 把标注结果构造成动态 HTML viewer。 |
| `build_annotation_showcase_viewer.py` | 从已标注数据中选样本并生成展示用 viewer，例如随机样本或高边密度样本。 |
| `find_annotation_rich_samples.py` | 根据 token 数、edge 数、edge/token ratio 等指标筛选标注边丰富的样本。 |
| `visualize_annotation_edges.py` | 早期/辅助的 annotation edge 可视化脚本。 |
| `dynamic_annotation_viewer.html` | annotation viewer 的前端模板。 |

代码逻辑可以概括为：

```text
读取已标注训练数据
→ 还原 ChatML/FIM 文本
→ tokenizer 得到 token、offset 和 completion mask
→ 读取 attention_edges
→ 对齐 edge 到 token index
→ 生成内嵌数据的 HTML viewer
```

## 三、Saliency Comparison Viewer

### 1. 目的

Saliency viewer 用来比较不同模型在同一个 completion token 上关注了哪些 source token。它支持选择语言、样本、模型和 source scope，点击某个 completion token 后，会显示 saliency top-k source token 以及贡献值。

这个 viewer 用来回答的问题是：

> 对同一个目标 token，我们的模型是否比 baseline 更关注语义上合理的 prompt token？

例如目标 token 是 `else` 时，如果我们的模型 top source 是 `if`，而 baseline 关注的是无关符号或局部噪声，那么可以作为我们 curation 方法更合理的定性证据。

### 2. Saliency 数据构造

Saliency 数据由下面脚本计算：

```text
tools/visual_saliency/compute_model_saliency.py
```

输入包括：

```text
outputs/visual_saliency/saliency_viz_samples.json
```

以及多个模型 checkpoint，例如：

```text
Ours GraphSignal
CLEAR
XTF
IB-FT
TokenCleaning
Base Qwen
LLM-CleanCode
```

计算逻辑是：

```text
选定展示样本
→ 对每个模型 free-run generate
→ 每生成一个 target token，就计算该 token 对历史 source token 的 ALTI saliency
→ 记录 top-k source token
→ 对所有模型和所有样本汇总成 saliency_comparison_data.json
```

这里采用的是 free-run generate，而不是 teacher-forcing 一次性计算完整答案。也就是说，第 t 个生成 token 的 saliency 是在模型已经生成前 t-1 个 token 的上下文下计算的，更接近真实推理过程。

### 3. 数据格式

核心数据文件是：

```text
outputs/visual_saliency/saliency_comparison_data.json
```

顶层字段主要包括：

```text
version
generation_mode
saliency_definition
target_semantics
source_scopes
top_k
models
samples
```

每个 sample 里包含：

```text
sample_id
row_index
uid
source_dataset
language
raw_id
models
```

每个模型结果里包含：

```text
tokens     当前模型 free-run 后的完整 token 序列
targets    每个生成 target token 的 saliency 结果
```

每个 target 里包含：

```text
target_idx       target token 在序列中的位置
token            target token 文本
step             free-run 生成步数
scopes           不同 source scope 下的 top-k saliency
```

`scopes` 中常用 scope 包括：

| Scope | 含义 |
| --- | --- |
| `prompt_code` | 只看 FIM prefix/suffix 代码区域的 source token。 |
| `prompt_all` | 看整个 prompt 范围内的 source token。 |
| `all_causal` | 看 target token 之前所有 causal source token。 |

### 4. 当前展示数据

组会展示的 saliency viewer 是：

```text
outputs/visual_saliency/saliency_comparison_viewer.html
```

对应的原始结构化数据是：

```text
outputs/visual_saliency/saliency_comparison_data.json
```

当前 HTML 已经把 JSON 数据嵌入页面，因此单独打开 HTML 也能看；但为了让别人复用数据或接入新的前端，最好同时保留 `saliency_comparison_data.json`。

### 5. 代码结构

```text
tools/visual_saliency/
```

主要文件作用：

| 文件 | 作用 |
| --- | --- |
| `select_saliency_viz_samples.py` | 从训练/展示数据中选择用于 saliency 对比的样本。 |
| `compute_model_saliency.py` | 对多个模型和多个样本计算 free-run ALTI saliency。 |
| `build_saliency_comparison_viewer.py` | 把 JSON 数据嵌入 HTML 模板，生成最终 viewer。 |
| `saliency_comparison_viewer.html` | saliency viewer 的前端模板。 |
| `export_intervention_saliency_format.py` | 把当前 saliency 数据转成队友旧格式兼容的单 JSON。 |
| `export_legacy_saliency_by_sample.py` | 导出单模型、按样本拆分的 legacy saliency 文件。 |
| `export_legacy_saliency_all_models_by_sample.py` | 导出多模型、按模型和样本拆分的 legacy saliency 文件，便于队友集成。 |

代码逻辑可以概括为：

```text
选择展示样本
→ 加载多个模型
→ 对每个样本 free-run 生成
→ 每个生成 token 调用 ALTI saliency 计算
→ 保存 top-k source token
→ 生成 comparison JSON
→ 构建自包含 HTML viewer
```

## 四、Failure Comparison Viewer

### 1. 目的

Failure viewer 用来对比 Ours GraphSignal 和 CLEAR 在同一批 HumanEval 样本上的生成结果与 judge 结果。它重点展示：

- Ours pass@10 但 CLEAR fail@10 的样本。
- CLEAR pass@10 但 Ours fail@10 的样本。
- Ours 和 CLEAR 都失败的样本。
- greedy 与 sampled candidates 的通过情况。

这个 viewer 用来分析：

> 我们的方法在哪些样本上优于强 baseline，又在哪些样本上输给 CLEAR？失败是否来自语法错误、补全位置理解错误、测试难度，还是生成不稳定？

### 2. 数据构造

Failure viewer 的数据不是直接复用 benchmark CSV，而是重新 dump 每个模型在前 1000 条 HumanEval 样本上的生成候选和 judge 结果。

主要流程是：

```text
对 Ours GraphSignal 跑前 1000 条 HumanEval
→ 保存 greedy + sampled candidates + judge 结果
→ 对 CLEAR 跑同一批样本
→ 保存 greedy + sampled candidates + judge 结果
→ 按 uid 对齐两个模型结果
→ 生成对比 JSON
→ 构建自包含 HTML viewer
```

相关中间数据包括：

```text
outputs/visual_failure/dumps/ours_graphsignal_500_humaneval1000.jsonl
outputs/visual_failure/dumps/clear_humaneval1000.jsonl
```

最终对比数据是：

```text
outputs/visual_failure/ours_vs_clear_humaneval1000.json
```

最终展示页面是：

```text
outputs/visual_failure/ours_vs_clear_failure_viewer.html
```

HTML 已经内嵌对比 JSON，因此页面本身可独立打开。

### 3. 数据格式

对比 JSON 顶层主要包含：

```text
models
num_common
categories
category_counts
samples
```

每个 sample 包含：

```text
key
uid
raw_id
language
prefix
suffix
ground_truth
ours
clear
categories
```

其中 `ours` 和 `clear` 都包含：

```text
pass1
greedy
pass10
samples
```

`greedy` 和每个 sampled candidate 内部包含：

```text
prediction
pass
error_summary
```

为了界面整洁，当前 viewer 只保留简短错误摘要，不展示完整 traceback。

### 4. 代码结构

```text
tools/visual_failure/
```

主要文件作用：

| 文件 | 作用 |
| --- | --- |
| `eval_dump.py` | 跑指定模型在指定 eval 子集上的生成与 judge，并保存每个 candidate 的结果。 |
| `build_comparison_data.py` | 对齐 Ours 和 CLEAR 的 dump 结果，构造 failure comparison JSON。 |
| `build_failure_viewer.py` | 把 comparison JSON 嵌入 HTML，生成最终 failure viewer。 |

代码逻辑可以概括为：

```text
模型生成与 judge dump
→ 两个模型结果按 uid 对齐
→ 根据 pass@1 / pass@10 关系分类样本
→ 构造可视化用 JSON
→ 生成自包含 HTML viewer
```

## 五、推荐 Git 提交流程

如果只希望把三类可视化源码和说明文档推到远程，不建议提交大体积输出数据。当前建议提交范围是：

```text
tools/visual_annotation/
tools/visual_failure/
tools/visual_saliency/
outputs/docs/visualization.md
```

如果需要让学姐直接打开三个页面，则还需要额外提供这些输出文件：

```text
outputs/visual_annotation/random5_from500/viewer.html
outputs/visual_failure/ours_vs_clear_failure_viewer.html
outputs/visual_saliency/saliency_comparison_viewer.html
```

如果需要让队友复用 saliency 原始数据，则还需要：

```text
outputs/visual_saliency/saliency_comparison_data.json
```

但这些输出数据较大，更适合通过压缩包、网盘、Release Asset 或单独数据分支传递，而不是混在源码 commit 中。
