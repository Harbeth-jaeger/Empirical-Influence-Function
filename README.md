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
| `src/` | 核心 Python 包。放可复用实现，不直接堆长命令脚本。 |
| `src/annotate/` | 结构边与 LLM 语义边标注核心代码。 |
| `src/data_process/` | FIM 数据 adapter、清洗、ChatML 构造、标注后处理等核心数据处理逻辑。 |
| `src/train/` | saliency loss、训练器和训练相关核心模块。 |
| `src/baseline/` | baseline 方法实现，例如 token/data filter 和 loss 类 baseline 的可复用逻辑。 |
| `src/attribution/` | 数据归因、token/feature 归因相关实现。 |
| `src/viz/` | 可视化和人工检查工具，例如标注边、saliency、失败案例、prediction saliency viewer。 |
| `scripts/` | 可直接运行的入口脚本。一个功能尽量一个脚本，调用 `src/` 中的核心逻辑。 |
| `scripts/data_process/` | 通用数据处理与标注 pipeline 入口。 |
| `scripts/train/` | 训练入口脚本，例如 `train.sh`。 |
| `scripts/baseline/` | baseline 数据构造、打分、过滤、训练入口。 |
| `scripts/huawei_deploy/` | 华为侧部署用脚本，包含网关调用、华为数据 adapter、小样本检查和全量标注入口。 |
| `configs/` | 实验、路径、训练和方法配置。 |
| `data/` | 软链接，指向 `/mnt/nvme0n1/wenhao/datasets/Empirical-Influence-Function`。仓库内不实际保存大数据。 |
| `models/` | 软链接，指向 `/mnt/nvme0n1/wenhao/models/Empirical-Influence-Function`。仓库内不实际保存模型权重。 |
| `outputs/` | 仓库内实验产物、报告文档、可视化页面和 debug 中间产物。`outputs/runs/` 也是指向外部 runs 的软链接。 |
| `runs/` | 软链接，指向 `/mnt/nvme0n1/wenhao/runs/Empirical-Influence-Function`。长任务日志写这里。 |
| `README.md` | 项目入口说明。只保留高层路线和目录导航。 |
| `AGENTS.md` | 协作/代码代理说明，记录项目约定、环境命令和安全边界。 |
| `requirements.txt` | 依赖入口；按场景拆分的手写依赖放在 `requirements/`。 |
| `locks/` | 当前环境的冻结依赖，例如 `pip freeze` 结果，只用于复现实验环境。 |
| `.env.example` | 本地路径、模型路径、API 和训练默认环境变量模板。真实密钥写 `.env`，不要提交。 |
| `pyproject.toml` | Python 项目元数据和工具配置。 |
| `.gitattributes` / `.gitignore` | Git 文件追踪和忽略规则。 |

隐藏目录说明：`.git/` 是版本管理目录；`.vscode/` 是 IDE 配置。当前采用“仓库代码 + 外部大文件”的布局：数据、模型、长日志、HF cache 和 micromamba 环境默认放在 `/mnt/nvme0n1/wenhao/{datasets,models,runs,hf_cache,envs}`。仓库内的 `data/`、`models/`、`runs/`、`.micromamba/`、`outputs/runs/` 都是本地软链接入口，便于代码仍然使用短路径，同时避免把大文件提交到 Git。

查看软链接本身用：

```bash
ls -ld data models runs outputs/runs .micromamba
readlink data
```

注意 `ls data/` 会进入软链接目标目录，看起来像普通目录；确认是否为软链接要看 `ls -ld data` 第一列是否以 `l` 开头。
