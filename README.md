# 项目概览

该项目名为“Qwen 代码语言模型训练”，围绕如下逻辑链展开

Train (模型训练与识别错题) → Attribution (双线归因：找错因与找错题来源) → Curation (数据治理与损失干预) 这三步走战略

1. 数据准备与模型训练：Qwen 大模型训练，模型任务是 Go 语言代码补全，由于产生了一些不明所以的错误输出，由此引出对训练数据 Attribution 和 Curation 的探索
2. 双线归因与可视化分析 Data Attribution
   1. 数据归因 Samples Attribution: 哪些 harmful 样本导致了模型学习了错误的 correlation
   2. 特征归因 Token Attribution: 某一次预测或者推理中，哪些 token 导致了模型产生了错误的输出
3. 数据治理 Data Curation：敲除或者降权不良样本，重新微调模型

# 技术路线

### Attribution

### Curation

本项目当前主线方法叫 **causality curation**，核心目标不是简单清洗样本，而是把“哪些 token 关系应该被模型重视”显式写进训练监督。整体流程如下：

```text
ChatML-FIM 代码补全样本
  -> 结构化边标注 + LLM 语义补边
  -> word-level edge 映射到 Qwen BPE token edge
  -> CE loss + saliency loss 联合训练
  -> 用 saliency alignment 和 completion quality 评估效果
```

#### 边标注 annotate

标注结果是一张 token-level 有向图：

```text
source token -> target token
```

语义是：source token 是预测或理解 target token 的重要线索。边类型主要包括：

| 类型 | 含义 |
| ---- | ---- |
| `bracket` | 括号、代码块等结构配对。 |
| `defuse` | 变量定义到后续使用。 |
| `call` | 调用名到参数。 |
| `return` | `return` 关键字到返回表达式。 |
| `type` | 类型 token 到变量名。 |
| `dataflow` | 值的生产位置到消费位置。 |
| `semantic` | 控制流、语义约束或结构呼应。 |
| `api` | API 调用模式、资源生命周期等关系。 |

实现上，`src/annotate` 先用 tree-sitter 抽取稳定结构边，再用 LLM 补充 `dataflow/semantic/api` 等需要语义判断的边。FIM 样本里合法边主要限制为三类：`prompt -> prompt`、`prompt -> completion`、`completion -> completion`，其中 `prompt -> completion` 是最关键的训练监督。

#### saliency 定义

训练中的 saliency 衡量：模型在生成某个 target token 时，某个 source token 对最后一层 hidden state 的贡献强度。当前实现不是完整 ALTI rollout，而是最后一层 contribution saliency：结合 attention weight、value projection、output projection 和 residual contribution，得到标量贡献分数：

```text
C_{q,s} = source token s 对 query token q 的贡献强度
```

它比直接看 attention 更可靠，因为它不仅看“模型看了谁”，也看被看的 token 经过投影后实际给 hidden state 带来了多大向量贡献。

#### loss 设计

训练目标是：annotation source 的 saliency 应该高于 non-annotation source。当前推荐使用 pairwise contrastive hinge loss：

```text
positive source = annotation edge 指向 target 的 source token
negative source = causal 前文中非 annotation 的 source token
```

对每个 target token，loss 显式比较 positive/negative pair，鼓励：

```text
log C_positive - log C_negative >= margin
```

相比旧版 InfoNCE / multi-positive NLL，pairwise contrastive loss 对 negative 侧压力更直接，不依赖手工 floor，适合缓解 positive/negative 数量严重不均衡的问题。

更完整的数学定义、标注边范围和训练诊断见：`outputs/docs/curation.md`。

# 文件结构及其作用

本节只说明根目录下各文件夹/文件的职责，方便快速判断东西应该放在哪里。更细的代码地图见 `outputs/docs/curation.md`。

| 路径 | 作用 |
| ---- | ---- |
| `src/` | 核心源码目录。当前主线包括 `src/annotate` 标注、`src/train` 训练、`src/attribution` 归因等。 |
| `scripts/` | 可运行脚本目录。放数据处理、Go single-line FIM 流程、saliency 小实验、benchmark/baseline 相关脚本。 |
| `tools/` | 可视化和人工检查工具目录，例如标注边、saliency、失败案例、prediction saliency viewer。 |
| `data/` | 数据目录。`raw_data` 放原始拉取数据，`<lang>_<task>` 放处理后的同源任务数据，`benchmark` 放处理后的外部 benchmark 数据。 |
| `models/` | 本地下载的基座模型目录，例如 Qwen / Code model 权重。原则上不放训练输出。 |
| `outputs/` | 实验产物目录。放训练模型、评测结果、可视化页面、报告文档和 debug 中间产物。 |
| `runs/` | 日志目录。长时间运行的训练、标注、数据处理、评测日志都放这里，便于 SSH 断连后复盘。 |
| `README.md` | 项目入口说明。只保留高层路线和目录导航。 |
| `AGENTS.md` | 协作/代码代理说明，记录项目约定、环境命令和安全边界。 |
| `requirements.txt` | Python 依赖列表。 |
| `.gitattributes` / `.gitignore` | Git 文件追踪和忽略规则。 |

隐藏目录说明：`.git/` 是版本管理目录；`.vscode/` 是 IDE 配置；`.micromamba/`、`.local/`、`.venv/` 是本地运行环境，不属于算法流程本身。
