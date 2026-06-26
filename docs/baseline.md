# Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning

Jinlong Pang, Na Di, Zhaowei Zhu, Jiaheng Wei, Hao Cheng, Chen Qian, Yang Liu
[[2502.01968] Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning](https://arxiv.org/abs/2502.01968)
ICML 2025，引用数 27 ，代码和数据公开，内容聚焦 token-level error detection & cleaning 提升 code generalization 能力

***通俗理解***
通过引入一个更强悍的参考模型（这个模型不是外部的 gpt 系列，而是 base model 通过一定程度的 fine-tuning 得到的）计算出训练数据中每个样本每个 token 的分数，对数据从 token 层面上进行清理，进而提高 fine-tuning 效果

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

# Explainable Token-level Noise Filtering for LLM Fine-tuning Datasets

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

# Breaking Memorization Barriers in LLM Code Fine-Tuning

[http://arxiv.org/pdf/2510.16022v1](http://arxiv.org/pdf/2510.16022v1)
Changsheng Wang, Xin Chen, Sijia Liu, Ke Ding
预印本，无引用

***通俗理解***
通过引入互信息中的 IB 原理，设计一个新颖的 loss 函数，该 loss 函数的本质是通过 KL 散度让构造的后验分布符合先验分布、进而实现信息压缩，打破模型训练中的记忆壁垒（这个记忆壁垒，大抵意思就是说模型训练会把训练样本整个“记下来”，相当于在死背答案）
（貌似和我们的项目关系没那么大）

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

IB-FT 流水线核心是“在 fine-tuning 过程中动态压缩 memorized spurious features”。

3.1 诊断 Memorization Barrier（Chapter 4）

* 绘图：使用 Min-K% Prob（K=20）量化 base model **\\theta\_0** 对 fine-tuning 数据 **D\_{\\text{code}}** 的记忆强度（lower score = stronger memorization），具体表现为概率密度函数图，峰值越靠左说明模型对数据的“记忆力”越强（Figure 3）
* 对比：**p\_0(D\_{\\text{code}};\\theta\_0)**（base）、**p\_1(D\_{\\text{code}};\\theta\_{\\text{code}})**（fine-tuned）、**p\_2(D\_{\\text{TOFU}};\\theta\_0)**（弱记忆参考）。
* 验证：移除 top 10% 最 memorized 数据即可显著提升 Pass@1（Figure 4），证明 memorization 确实构成 barrier（Chapter 4）。
  3.2 IB-FT 技术路线（Chapter 5）
* **IB 原理**：最小化 **I(X;Z) - \\beta I(Z;Y)**，压缩 input-specific spurious 信息，保留 prediction-relevant 信号。
  * **Z**：隐表示（bottleneck representation，从 LLM 中间层如第 20 层提取的 hidden state）。
  * **I(\\cdot;\\cdot)**：互信息（mutual information），衡量两个变量共享的信息量。
* **Variational 实现**（Eq. (3)-(6)）： **IB 正则**为**\\ell\_{\\text{IB}} = \\ell\_{\\text{IB}}^{\\text{compress}} - \\beta \\ell\_{\\text{IB}}^{\\text{predict}}**。**总目标（IB-FT）** 为**\\min\_{\\theta,\\phi} \\ell\_{\\text{FT}}(\\theta) + \\alpha \\ell\_{\\text{IB}}(\\theta,\\phi)**。
  * **Compress loss**：**\\ell\_{\\text{IB}}^{\\text{compress}} = \\mathbb{E} [D\_{\\text{KL}}(q\_\\phi(z|h\_\\theta(x)) || p(z))]**（hidden rep **h\_\\theta(x)** 取第 20 层，**p(z)=\\mathcal{N}(0,I)**）。这个 loss 要最小化，即 KL 惩罚 **q\_{\\phi}** 偏离 **p(z)** → 强迫 Z 不能保留太多 input-specific 的细节（spurious memorized features）
    * **h\_{\\theta}(x)**：LLM 当前参数 **\\theta** 下，输入 **x** 在**指定中间层**（paper 固定第 20 层）的 hidden representation（这是要“压缩”的对象）。
    * **q\_{\\phi}(z \\mid h\_{\\theta}(x))**：**变分编码器**（learnable 参数 **\\phi**），把 hidden rep 映射成一个概率分布（论文中是 Gaussian），产生 bottleneck **z \\in (B, N, D\_z)**
    * **p(z) = \\mathcal{N}(0, I)**：**先验分布**（标准正态），简单、无结构。
    * **D\_{\\text{KL}}**：Kullback-Leibler 散度，衡量两个分布有多“不同”。
  * **Predict loss**：**\\ell\_{\\text{IB}}^{\\text{predict}} = \\mathbb{E} [\\log p\_\\theta(y|z)]**（保留任务信号）。这个 loss 要最大化，防止“过压缩”——如果只压缩而不保留任务信号，模型会彻底学废
* **实现细节**（Appendix A）：LoRA（r=32）+ IB penalty（**\\alpha=0.1**, **\\beta=0.02/0.01**），3 epochs，batch=4，LR=5e-5 / 1e-4。
  3.3 效果验证
* **表征分析**（Figure 5）：IB-FT 显著压缩 most-memorized vs. least-memorized 样本间的 **\\ell\_2** 距离和 angle disparity（>50% 压缩），实现更均匀学习。
* **多温度鲁棒性**（Figure 6）：IB-FT 在 T∈{0.2,0.6,1.0} 下均稳定优于 FT。

---

**4. 文章使用的数据和代码链接****数据集**（Chapter 6.1）：

* **OriGen**（Cui et al., 2024）：Verilog 指令-响应数据集（222,075 examples），用于 RTL 代码生成。
* **Evol-CodeAlpaca-V1**（Luo et al., 2024）：多语言指令-to-code 数据集（\~111k examples，覆盖 Python/C++/Java 等）。
* **评估基准**：OriGen → VerilogEval（Eval-Human + Eval-Machine）；Evol-CodeAlpaca-V1 → HumanEval。   （论文未提供直接下载链接，但均为公开基准，可通过原论文引用获取。）

**代码**：论文正文、附录及参考文献中均未提及任何公开代码仓库、GitHub 链接或实现脚本（包括 IB penalty 的具体实现）。仅给出 fine-tuning 超参（LoRA r=32、α/β 值等），未开源。

# LLM-Assisted Code Cleaning For Training Accurate Code Generators

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

---

# Automated Data Curation for Robust Language Model Fine-Tuning

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

其中，BSDetector 全称 Bad and Speculative Detector，是一个方法/算法框架，正式发表于 ACL 2024 文章（Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics），其置信度分数 C 是由两个互补因子加权组合得到的，即O = Observed Consistency（观测一致性）、以及S = Self-reflection Certainty（自反思确定性）

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
