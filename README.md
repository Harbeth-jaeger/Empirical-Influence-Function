# 项目概览

我正在进行一个名为“Qwen 代码语言模型训练”的项目，项目信息如下

Train (模型训练与识别错题) → Attribution (双线归因：找错因与找错题来源) → Curation (数据治理与损失干预) 这三步走战略

1. 数据准备与模型训练：Qwen 大模型训练，模型任务是 Go 语言代码补全，由于产生了一些不明所以的错误输出，由此引出对训练数据 Attribution 和 Curation 的探索
2. 双线归因与可视化分析 Data Attribution
   1. 数据归因 Samples Attribution: 哪些 harmful 样本导致了模型学习了错误的 correlation
   2. 特征归因 Token Attribution: 某一次预测或者推理中，哪些 token 导致了模型产生了错误的输出
3. 数据治理 Data Curation：敲除或者降权不良样本，重新微调模型

# 技术路线

### Attribution

### Curation

#### 边标注 annotate

设一段代码经过标注 tokenizer 后得到 token 序列：

**t\_1, t\_2, \\dots, t\_n**

每个 token 带有：

**\\text{t}\_i = (\\text{surface}\_i, \\text{char\\\_start}\_i, \\text{char\\\_end}\_i)**

其中：

* **surface\_i**：表示 token 的文本内容，例如 `total`、`return`、`Math`、`(`。
* **[char\_start\_i, char\_end\_i)**：表示这个 token 在原始代码字符串里的字符区间。

标注的最终目标是得到一个有向图 **G = (V, E)**，其中：

* **V = \\{t\_1, t\_2, \\dots, t\_n\\}**
* 边集合为 **E = \\{(i, j, r)\\}**

含义是：

**\\text{token}\_i \\xrightarrow{r} \\text{token}\_j**

也就是：**token\_i 是预测 token\_j 时的语义相关线索。**

这里的 `r` 是边类型，例如：`defuse`, `dataflow`, `call`, `return`, `type`, `semantic`, `api`, `bracket`。

代码里对应的数据结构是：

```Python
TokenCorrelation(
    token_i: str,
    token_j: str,
    source: str,
    subtype: str,
    token_i_idx: int,
    token_j_idx: int,
)
```

标注边的几种类型

3.1 defuse：定义到使用

例子：

```Python
total = 0
for x in nums:
    total += x
return total
```

可能有边：`total(定义处) -> total(累加处)`。含义是后面再次出现变量时，前面的定义是重要上下文。

---

3.2 dataflow：值的生产到消费

例子：

```Python
distance = abs(a - b)
if distance < best:
    best = distance
```

可能有边：`abs -> distance`, `a -> distance`。这代表了真实的语义相关性。

---

3.3 call：函数名到参数

例子：`sorted(nums, reverse=True)`

可能有边：`sorted -> nums`。提示参数位置应该服从该 API 的调用模式。

---

3.4 return：return 到返回表达式

含义：`return` 关键字提示后面应该出现返回值表达式。

---

3.5 type：类型到变量名

例子：`int count = 0;` -> `int -> count`。

---

3.6 semantic：控制流语义

例子：`if -> else`, `if -> condition`。控制流关键字之间的语义配对。

---

3.7 api：库/API 使用模式

例子：`open -> read`, `f -> close`。学习 API 的生命周期和调用模式。

---

3.8 bracket：括号匹配

例如 `(` -> `)`。属于较弱的结构信号。

代码里的标注流程

4.1 结构化标注：tree-sitter

`SyntacticCheckerTool` 用 tree-sitter 做确定性结构分析，提取 `bracket`, `defuse`, `call` 等不依赖 LLM 的稳定结构边。

4.2 神经标注：LLM agent

`AnnotatorAgent` 先获取结构边，再补充 LLM 发现的 `dataflow`, `semantic`, `api` 等复杂语义边。

4.3 映射到 Qwen BPE token

SFT 时模型看到的是 BPE token。后处理需要根据字符区间重叠情况，将简单代码 token **s\_i** 映射到 Qwen token **q\_k**。最终训练使用 `qwen_annotations`。

#### saliency 定义

**1. 宏观直觉**

Saliency 衡量的是：模型在生成某个目标 token 时，前文哪些 token 对它的贡献最大。

给定输入序列

**x\_1, x\_2, \\ldots, x\_T**

当模型在位置 **q** 的 hidden state 用来预测下一个 token 时，我们希望知道前文某个 source token **x\_s** 对这个预测有多大贡献，其中 **s \\le q**。

可以把 saliency 理解为：

**\\mathrm{Saliency}(q, s)=\\text{source token } x\_s \\text{ 对 query position } q \\text{ 的贡献强度}**

如果模型要生成 `else`，而高 saliency token 是 `if`、condition、前一个 branch，那说明模型内部依赖比较合理；如果高 saliency token 是注释、无关符号、随机变量名，就说明 attribution 可能不理想。

**2. 数学定义**

ALTI 的核心思想是，从 Transformer 的前向传播中，把 token 间的信息传递抽象成一个 contribution matrix，把每一层的 token-to-token contribution 沿层数做 rollout，得到最终贡献矩阵：

**C = C^{(L)} C^{(L-1)} \\cdots C^{(1)}**

但是我们现在代码不是完整 ALTI rollout。当前实现是：只看最后一层 self-attention 的 ALTI contribution。

以下是具体推导：

设最后一层输入 hidden state 为：

**x\_1,\\ldots,x\_T,\\quad x\_i \\in \\mathbb{R}^D**

对 source token (**s**)，先过最后一层 attention 的 value projection。代码里还乘了 RMSNorm 的缩放参数 (**\\gamma**)：

**v\_s^h = W\_V^h(\\gamma \\odot x\_s)**

然后经过对应 head 的 output projection slice：

**z\_s^h = W\_O^h v\_s^h**

对 query token (**q**)，attention 权重为：

**A^h\_{q,s}**

于是 source (**s**) 对 query (**q**) 的向量贡献大概是：

**T\_{q,s} = \\frac{1}{\\sigma\_q} \\left( \\sum\_h A^h\_{q,s} z\_s^h + \\mathbf{1}\_{q=s}x\_q \\right)**

其中：

**\\sigma\_q = \\sqrt{\\frac{1}{D}\\sum\_d x\_{q,d}^2 + \\epsilon}**

最后 saliency 不是概率，而是这个贡献向量的 L2 norm：

**C\_{q,s}=|T\_{q,s}|\_2**

于是最终 saliency 可以写成：

**\\mathrm{Saliency}(q,s) = C\_{q,s}, \\quad s \\le q**

也就是说，**C\_{q,s}** 衡量的是：经过所有 Transformer 层传播后，token **x\_s** 对位置 **q** 的最终贡献。

#### loss v1

我们的训练数据里有 annotation edges：

**E = \\{(s,q,r)\\}**

其中 **s** 是 source token，**q** 是 target/query token，**r** 是边类型。训练时我们只关心：标注边里的 source token 是否比其他 token 更重要。

因此，对每个 query token **q**，定义标注 source 集合：

**A(q) = \\{s \\mid (s,q,r) \\in E\\}**

以及非标注 causal source 集合：

**N(q) = \\{s \\mid s < q,\\ s \\notin A(q)\\}**

然后计算两组平均贡献：

**\\bar{C}\_{A}(q)=\\frac{1}{|A(q)|}\\sum\_{s \\in A(q)}C\_{q,s}**

**\\bar{C}\_{N}(q)=\\frac{1}{|N(q)|}\\sum\_{s \\in N(q)}C\_{q,s}**

我们的 saliency loss 希望：

**\\bar{C}\_{A}(q) > \\bar{C}\_{N}(q)**

具体写成 margin ratio loss：

**\\mathcal{L}\_{\\mathrm{sal}}(q)=\\max\\left(0,\\, \\alpha - \\frac{\\bar{C}\_{A}(q)}{\\bar{C}\_{N}(q) + \\varepsilon}\\right)**

最终训练目标是：

**\\mathcal{L}=\\mathcal{L}\_{\\mathrm{SFT}}+ \\gamma \\mathcal{L}\_{\\mathrm{sal}}**

其中：

* **\\mathcal{L}\_{\\mathrm{SFT}}**：普通 next-token prediction loss。
* **\\mathcal{L}\_{\\mathrm{sal}}**：让模型更依赖 annotation edge 指定的 source token。
* **\\gamma**：控制 saliency loss 权重。
* **\\alpha**：margin，要求标注边贡献至少比非标注边高到一定比例。
* **\\varepsilon**：防止分母为零。

**5. 一句话总结**

Saliency 的数学本质是 token-to-token contribution：

**\\mathrm{Saliency}(q,s)=C\_{q,s}**

ALTI 给出了完整的跨层 contribution rollout；而训练时我们不计算完整 **T \\times T** 矩阵，只对 annotation edge 涉及的 query token 做 sparse saliency 计算，从而把 saliency supervision 放进实际 SFT 训练中。

#### loss v2

annotation pair 输入后，代码先强制因果化：

**s=\\min(p\_0,p\_1),\\quad q=\\max(p\_0,p\_1)**

也就是说无论原始边方向怎么写，loss 里都会当成“前文 source 指向后文 query”。

对每个 query token (**q**)，设 annotation source 集合为：

**A\_q=\\{s:(s,q)\\in E\_{\\text{annot}}\\}**

causal source 集合为：

**P\_q=\\{s:s<q\\}**

non-annotation causal source 为：

**N\_q=P\_q\\setminus A\_q**

计算非标注token的平均值：

**\\bar N\_q=\\frac{1}{|N\_q|}\\sum\_{s\\in N\_q}C\_{q,s}**

新版真正优化的是逐 annotation source 的 margin：

**\\mathcal{L}\_q = \\frac{1}{|A\_q|} \\sum\_{s\\in A\_q} \\max(0,\\alpha \\bar N\_q - C\_{q,s})**

总 saliency loss：

**\\mathcal{L}\_\\text{sal} = \\frac{1}{|Q|} \\sum\_q \\mathcal{L}\_q**

最终训练目标是：

**\\mathcal{L}=\\mathcal{L}\_{\\mathrm{SFT}}+ \\gamma \\mathcal{L}\_{\\mathrm{sal}}**

其中：

* **\\mathcal{L}\_{\\mathrm{SFT}}**：普通 next-token prediction loss。
* **\\mathcal{L}\_{\\mathrm{sal}}**：让模型更依赖 annotation edge 指定的 source token。
* **\\gamma**：控制 saliency loss 权重。
* **\\alpha**：margin，要求标注边贡献至少比非标注边高到一定比例。
* **\\varepsilon**：防止分母为零。

#### loss v3

对某个 query token **q**，定义它的 annotation source 集合为：

**A\_q = \\{s \\mid s \\rightarrow q \\text{ is an annotation edge},\\ s < q\\}**

所有 causal source 集合为：

**M\_q = \\{s \\mid s < q\\}**

非 annotation source 集合为：

**N\_q = M\_q \\setminus A\_q**

新版 loss 先把 saliency 转成 log-score：

**r\_{q,s} = \\log(C\_{q,s} + \\epsilon\_{\\text{reg}})**

然后用 temperature **\\tau** 缩放：

**\\ell\_{q,s} = \\frac{r\_{q,s}}{\\tau}**

在所有 causal source 上做 softmax：

**p\_{q,s} = \\frac{ \\exp(\\ell\_{q,s}) }{ \\sum\_{r \\in M\_q} \\exp(\\ell\_{q,r})}**

对于一个 query **q**，multi-positive loss 是：

**\\mathcal{L}\_q = - \\frac{1}{|A\_q|} \\sum\_{s \\in A\_q} \\log p\_{q,s}**

总 saliency loss 是：

**\\mathcal{L}\_{sal} = \\frac{1}{|Q|} \\sum\_{q \\in Q} \\mathcal{L}\_q**

其中 **Q** 是满足下面条件的 query token 集合：

**|A\_q| > 0 \\quad \\text{and} \\quad |N\_q| > 0**

#### loss v4

对某个 query token **q**，定义它的 annotation source 集合为：

**A\_q = \\{s \\mid s \\rightarrow q \\text{ is an annotation edge},\\ s < q\\}**

所有 causal source 集合为：

**M\_q = \\{s \\mid s < q\\}**

非 annotation source 集合为：

**N\_q = M\_q \\setminus A\_q**

然后用 temperature **\\tau** 缩放：

**\\ell\_{q,s} = \\frac{C\_{q,s}}{\\tau}**

在所有 causal source 上做 softmax：

**p\_{q,s} = \\frac{ \\exp(\\ell\_{q,s}) }{ \\sum\_{r \\in M\_q} \\exp(\\ell\_{q,r})}**

对于一个 query **q**，multi-positive loss 是：

**\\mathcal{L}\_q = - \\frac{1}{|A\_q|} \\sum\_{s \\in A\_q} \\log p\_{q,s}**

总 saliency loss 是：

**\\mathcal{L}\_{sal} = \\frac{1}{|Q|} \\sum\_{q \\in Q} \\mathcal{L}\_q**

其中 **Q** 是满足下面条件的 query token 集合：

**|A\_q| > 0 \\quad \\text{and} \\quad |N\_q| > 0**

#### 最新 loss 设计

InfoNCE / multi-positive NLL。

对某个 query token **q**，定义它的 annotation source 集合为：

**A\_q = \\{s \\mid s \\rightarrow q \\text{ is an annotation edge},\\ s < q\\}**

所有 causal source 集合为：

**M\_q = \\{s \\mid s < q\\}**

非 annotation source 集合为：

**N\_q = M\_q \\setminus A\_q**

新版 loss 先把 saliency 转成 log-score：

**r\_{q,s} = \\log(C\_{q,s} + \\epsilon\_{\\text{reg}})**

然后用 temperature **\\tau** 缩放：

**\\ell\_{q,s} = \\frac{r\_{q,s}}{\\tau}**

在所有 causal source 上做 softmax：

**p\_{q,s} = \\frac{ \\exp(\\ell\_{q,s}) }{\\sum\_{r \\in N\_q} \\exp(\\max(\\ell\_{q,r},\\epsilon))+\\exp(\\ell\_{q,s}) }**

其中，eps 相当于负样本 floor，决定了负样本最低有多强，从数学形式上解析，eps 越大，对 neg 的惩罚力度越小。因此 eps 是关键超参数，影响到 loss 对负样本的打压力度

对于一个 query **q**，multi-positive loss 是：

**\\mathcal{L}\_q = - \\frac{1}{|A\_q|} \\sum\_{s \\in A\_q} \\log p\_{q,s}**

总 saliency loss 是：

**\\mathcal{L}\_{sal} = \\frac{1}{|Q|} \\sum\_{q \\in Q} \\mathcal{L}\_q**

其中 **Q** 是满足下面条件的 query token 集合：

**|A\_q| > 0 \\quad \\text{and} \\quad |N\_q| > 0**

直观理解就是：

对每个 target token **q**，把所有前文 causal token 都作为候选 source，让 annotation source 在 softmax 分布里获得更高概率。它不只是要求 annotation saliency 高于平均 non-annotation saliency，而是让 annotation source 和所有 causal source 竞争排名。

***附录***：下面具体展开关于 floor 这个超参数的选定

符号说明：

* **\\theta\_{t-1}**：进入第 **t** 个 training step 前的模型参数
* 第 **t** step 前向传播得到 sal 分布：

**S\_t = S(\\theta\_{t-1}, \\mathcal{B}\_t)**	用这个 step 的 loss 做反传和更新，得到 **\\theta\_t**

* **\\epsilon\_t**：第 **t** step 训练时实际用的 eps

也就是说第 **t** step 可以这样：

1. 用 **\\theta\_{t-1}** 前向，得到当前 batch 的 sal 分布：

**S\_t=\\{c\_{q,r}(\\theta\_{t-1})\\}**

2. 计算当前 batch quantile：

**\\hat{\\epsilon}\_t = Q\_{0.75}(S\_t)**	这个0.75也是一个可以调整的超参数，目前看下来0.75最佳

3. EMA 得到当前 step 实际使用的 eps：这样 eps 会平滑一些，loss 不至于抖动的太厉害；另外一开始不太稳定，所有不准备加 eps

**\\epsilon\_t = \\begin{cases} 0, & t < T\_{warmup}\\\\ \\hat{\\epsilon}\_t, & t = T\_{warmup}\\\\ \\beta\\epsilon\_{t-1} + (1-\\beta)\\hat{\\epsilon}\_t, & t > T\_{warmup} \\end{cases}**

4. 用这个 **\\epsilon\_t** 算当前 step 的 sal loss：

**\\mathcal{L}^{(t)}\_{sal} = \\mathcal{L}\_{sal}(\\theta\_{t-1}; \\epsilon\_t)**

5. 反传更新到 **\\theta\_t**

经过实验结果验证可知，其他超参数最优为

**Q=0.75,\\quad \\beta=0.95,\\quad T\_{warmup}=10**

# 工作整理

### Model Training / Curation

#### pipeline

idea 构筑与修改、迭代——idea 实现（代码debug）——idea work效果（实验验证）

整个idea：annotate 、sal 、sal loss

先自洽、再和其他人比

#### annotate & sal & loss

work的判断标准：在ce+sal loss训练下，label和sal能对齐，且ce占主导地位

实验验证：单样本过拟合，小样本训练前后模型sal分布可视化[训练前后saliency分布对比及与annotation对齐状况](http://1.95.134.216/saliency/base_vs_floorp75_annotation_saliency_viewer_tf.html)

***已完成***

#### benchmark

work的判断标准：ours跑分最高、且sal具有可解释性，pass rate和codebleu整体数值合

具体做法：

1. 获取并处理数据，要求10K（用于7B基模），单语言go，短代码singleline（只挖assignment语句，return语句，函数调用的等等），无docstring和注释，无unit test和test case，可视化10个例子发给刘老师看看。***目前进度在这***
2. 小样本标注数据，要求annotate 代码修改、对简单样本可以尽量dense一些，annotation 可视化合理
3. 小样本benchmark，记得baseline包括base、ce loss和其他paper，oracle检验评测代码是否合理，要求ours跑分最高、且sal具有可解释性[双模型预测与 saliency 对比](http://1.95.134.216/saliency/ce500_vs_floorp75_saliency_viewer.html)，pass rate和codebleu整体数值合
4. 全量benchmark，==完整结束==

### baseline

#### 1

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

#### 2

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

#### 3

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

#### 4

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

#### 5

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

# 文件结构及其作用

本节说明根目录下主要目录、GoSingle 实验链路和常用命令，便于快速判断代码、数据、结果和日志分别放在哪里。

## 总体目录


| 路径                       | 作用                                                                                                                                                           |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/`                     | 项目核心代码。当前最关键的是`src/annotate/` 和 `src/train/`。                                                                                                  |
| `src/annotate/`            | GraphSignal 标注代码。负责 simple code token 切分、结构/语义 annotation edge 生成、simple token 到 Qwen BPE token 的映射。                                     |
| `src/train/`               | SFT 与 saliency loss 训练代码。`train.py` 是训练入口，`dataset.py` 读取带 `attention_edges` 的样本，`loss.py` 实现 saliency 与 softmax / softmax-margin loss。 |
| `scripts/`                 | 实验脚本入口。包括数据构建、benchmark 评测、saliency 诊断、治理算子等。                                                                                        |
| `scripts/go_single/`       | GoSingle 专用脚本，包括数据构建、oracle 检查、已有 prediction 的 GoSingle 评测。                                                                               |
| `scripts/benchmark/`       | 通用 benchmark 评测框架，包括生成、judge、pass@k 聚合、CodeBLEU 和分片合并。                                                                                   |
| `scripts/saliency_exp/`    | Saliency 诊断实验，包括 base saliency 分布统计、row/token 训练追踪、trace 可读报告渲染。                                                                       |
| `tools/visual_annotation/` | Annotation edge 可视化与标注辅助工具。                                                                                                                         |
| `tools/visual_saliency/`   | 训练前后 saliency 分布可视化工具，用于检查 saliency top-k 与 annotation source 的对齐情况。                                                                    |
| `data/`                    | 数据目录。保存原始数据、训练数据、评测数据、已标注数据。大数据不建议提交 git。                                                                                 |
| `outputs/`                 | 实验输出目录。保存模型/LoRA adapter、评测结果、可视化 HTML/JSON、诊断图表和中间产物。大模型和大输出不建议提交 git。                                            |
| `runs/`                    | 运行日志目录。长任务的训练、评测、标注、可视化日志通常放在这里。                                                                                               |
| `AGENTS.md`                | 面向协作者或代码代理的项目说明，包含项目目标、工作流、目录约定和协作注意事项。                                                                                 |
| `README.md`                | 项目总览文档，说明项目目标、整体流程、文件结构和关键命令。                                                                                                     |
| `requirements.txt`         | Python 依赖列表。实际实验优先使用项目内`.micromamba/envs/eif-bench` 环境。                                                                                     |

隐藏目录如 `.git/`、`.vscode/`、`.venv/`、`.micromamba/`、`.local/` 主要服务于版本管理、IDE 配置或本地环境，不属于项目算法流程本身。

## 通用环境命令

所有训练、评测、可视化命令建议先进入项目环境：

```bash
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"
```

长任务建议使用 `nohup bash -lc '...' > runs/xxx.log 2>&1 &`，日志统一放在 `runs/`，输出统一放在 `outputs/`。

## GoSingle

GoSingle 是当前主要实验任务：从 Go 函数里挖出单行 statement，把该行替换为 `[MASK]`，让 Qwen 补全缺失 Go 代码。GraphSignal 版本额外提供 `attention_edges`，训练时用 saliency loss 约束模型在预测 target token 时更依赖 annotation source token。

### 关键数据文件


| 路径                                                                        | 作用                                                                                                                          |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `data/go_single/raw_data/`                                                  | GoSingle 原始数据，包括 CodeSearchNet Go 和 MCEval Go。                                                                       |
| `data/go_single/train_data/go_single_train_v2_canonical.jsonl`              | 从 CodeSearchNet Go 构建出的 canonical 训练样本，含`prefix`、`target`、`suffix`、`target_kind`。                              |
| `data/go_single/train_data/go_single_train_v2_chatml.jsonl`                 | canonical 训练样本渲染为 ChatML 后的版本。                                                                                    |
| `data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json` | 已标注 500 条 GoSingle 训练数据，是当前 saliency loss 训练的主输入。每行含`input_ids`、`label`、`length`、`attention_edges`。 |
| `data/go_single/eval_data/mceval_go_single_v2_canonical.jsonl`              | 从 MCEval Go 构建出的 canonical 评测样本，含 judge 所需`judge_payload`。                                                      |
| `data/go_single/eval_data/mceval_go_single_v2_chatml.jsonl`                 | MCEval GoSingle 评测样本的 ChatML 版本，用于`scripts/benchmark/benchmark_eval.py`。                                           |
| `outputs/go_single/models/`                                                 | GoSingle 训练出的模型或 LoRA adapter。                                                                                        |
| `outputs/go_single/eval_results/`                                           | GoSingle 专用评测结果。                                                                                                       |
| `outputs/benchmark/go_single_*`                                             | 通用 benchmark evaluator 的 GoSingle/MCEval 输出。                                                                            |
| `outputs/visual_saliency/`                                                  | Saliency 可视化 JSON 和 HTML。                                                                                                |
| `outputs/saliency_exp/`                                                     | Saliency 诊断实验输出。                                                                                                       |
| `runs/go_single/`                                                           | GoSingle 相关日志。                                                                                                           |

### 数据构建

相关代码：


| 路径                                         | 作用                                                                                                                               |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/go_single/build_go_single_data.py`  | GoSingle 数据构建入口。读取 CodeSearchNet Go 和 MCEval Go，输出 train/eval canonical 与 ChatML 文件。                              |
| `scripts/go_single/go_single_pipeline.py`    | 数据构建核心逻辑：函数切片、statement 分类、过滤 test/generated/comment/noisy call、去重、MCEval judge payload 拼接、report 生成。 |
| `scripts/go_single/oracle_eval_go_single.py` | 用 gold target 做 oracle 评测，检查 GoSingle eval/judge 本身是否可靠。                                                             |

构建 GoSingle 数据：

```bash
nohup bash -lc '
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

python scripts/go_single/build_go_single_data.py \
  --codesearchnet-dir data/go_single/raw_data/codesearchnet/go/final/jsonl/unzip \
  --codesearchnet-glob "go_train_*.jsonl" \
  --mceval-root data/go_single/raw_data/mceval \
  --num-train 10000 \
  --train-output data/go_single/train_data/go_single_train_v2_canonical.jsonl \
  --train-chatml-output data/go_single/train_data/go_single_train_v2_chatml.jsonl \
  --eval-output data/go_single/eval_data/mceval_go_single_v2_canonical.jsonl \
  --eval-chatml-output data/go_single/eval_data/mceval_go_single_v2_chatml.jsonl \
  --report outputs/go_single/reports/go_single_v2_build_report.md \
  --samples-report outputs/go_single/reports/go_single_v2_samples.md
' > runs/go_single/build_go_single_data.log 2>&1 &
```

检查 eval judge 的 oracle 上限：

```bash
nohup bash -lc '
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

python scripts/go_single/oracle_eval_go_single.py \
  --eval-data data/go_single/eval_data/mceval_go_single_v2_canonical.jsonl \
  --output outputs/go_single/eval_results/oracle/mceval_go_single_v2_oracle_results.jsonl \
  --summary outputs/go_single/eval_results/oracle/mceval_go_single_v2_oracle_summary.json \
  --table-md outputs/go_single/eval_results/oracle/mceval_go_single_v2_oracle_table.md \
  --table-csv outputs/go_single/eval_results/oracle/mceval_go_single_v2_oracle_table.csv \
  --compute-codebleu
' > runs/go_single/oracle_eval_go_single.log 2>&1 &
```

### Annotation

相关代码：


| 路径                                                         | 作用                                                                                      |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `src/annotate/utils.py`                                      | 标注基础工具：代码 tokenization、BPE 映射、FIM annotation edge 方向归一化。               |
| `src/annotate/neural_annot.py`                               | GraphSignal 标注核心：结构边、LLM/teacher 语义边、OpenAI-compatible API 调用、edge 合并。 |
| `src/annotate/viz_utils.py`                                  | annotation edge 静态可视化函数。                                                          |
| `tools/visual_annotation/annotate_benchmark_fim_train.py`    | 对 ChatML-FIM 训练样本做 annotation，输出紧凑训练格式。                                   |
| `tools/visual_annotation/build_dynamic_annotation_viewer.py` | 构建 annotation edge 动态 HTML viewer。                                                   |

对 500 条 GoSingle 训练样本做 annotation。这个命令需要 API key，且耗时较长：

```bash
nohup bash -lc '
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

export OPENAI_API_KEY="你的 API key"
export OPENAI_BASE_URL="你的 OpenAI-compatible base url"
export ANNOTATE_MODEL="你的标注模型名"

python tools/visual_annotation/annotate_benchmark_fim_train.py \
  --input_path data/go_single/train_data/go_single_train_v2_chatml.jsonl \
  --output_path data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json \
  --model_path models/Qwen2.5-Coder-7B-Instruct \
  --limit 500 \
  --annotation_mode oneshot \
  --oneshot_prompt_style compact \
  --edge_scope context_response_prompt \
  --bpe_map_mode first \
  --num_workers 4 \
  --auto_resume \
  --row_cache_dir outputs/go_single/annotation_cache/graphsignal_500
' > runs/go_single/annotate_go_single_graphsignal_500.log 2>&1 &
```

构建 annotation viewer：

```bash
python tools/visual_annotation/build_dynamic_annotation_viewer.py \
  --edge_data_path data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json \
  --sample_indices 232 34 289 344 490 416 460 479 \
  --model_path models/Qwen2.5-Coder-7B-Instruct \
  --output_path outputs/visual_annotation/go_single_annotation_viewer.html \
  --local_files_only
```

### Train

相关代码：


| 路径                                | 作用                                                                                                                                      |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `src/train/train.py`                | 训练主入口。支持`ce_only`、`ce_saliency`、`saliency_only`。训练结束会保存模型和 `saliency_training_config.json`。                         |
| `src/train/dataset.py`              | 读取 GoSingle compact 数据，把`attention_edges` 转为训练用 `annot_pairs`。                                                                |
| `src/train/loss.py`                 | Saliency 与 loss 实现。`softmax` 是 raw saliency softmax loss；`softmax_margin` 是 log-saliency + negative floor 的 softmax-margin loss。 |
| `src/train/saliency_diagnostics.py` | 训练时详细 saliency 诊断，统计 recall@k、precision@k、mAP@k 和 query/edge 级数据。                                                        |

CE baseline：

```bash
nohup bash -lc '
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/train/train.py \
  --model_name_or_path models/Qwen2.5-Coder-7B-Instruct \
  --data_path data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json \
  --output_dir outputs/go_single/models/ce_loss_500_run1 \
  --use_peft True \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --loss_mode ce_only \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 100 \
  --learning_rate 2e-4 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none
' > runs/go_single/train_ce_loss_500_run1.log 2>&1 &
```

CE + saliency。当前默认推荐使用 `softmax_margin`：

```bash
nohup bash -lc '
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/train/train.py \
  --model_name_or_path models/Qwen2.5-Coder-7B-Instruct \
  --data_path data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json \
  --output_dir outputs/go_single/models/ce_sal_softmax_margin_500_run1 \
  --use_peft True \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --loss_mode ce_saliency \
  --saliency_loss_type softmax_margin \
  --saliency_lambda 0.1 \
  --saliency_alpha 1.5 \
  --saliency_eps 1e-8 \
  --saliency_floor_eps_mode ema_quantile \
  --saliency_floor_quantile 0.75 \
  --saliency_floor_warmup_steps 10 \
  --saliency_detail_log_path runs/go_single/ce_sal_softmax_margin_500_run1_detail.jsonl \
  --saliency_detail_log_steps 1 \
  --saliency_detail_top_k 20 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 100 \
  --learning_rate 2e-4 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none
' > runs/go_single/train_ce_sal_softmax_margin_500_run1.log 2>&1 &
```

固定 logit floor `eps=-10`、temperature `tau=0.1` 的 CE + saliency：

```bash
nohup bash -lc '
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/train/train.py \
  --model_name_or_path models/Qwen2.5-Coder-7B-Instruct \
  --data_path data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json \
  --output_dir outputs/go_single/models/ce_sal_softmax_margin_logit_eps_m10_tau0p1_500_run1 \
  --use_peft True \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --loss_mode ce_saliency \
  --saliency_loss_type softmax_margin \
  --saliency_lambda 0.1 \
  --saliency_alpha 0.1 \
  --saliency_eps 1e-8 \
  --saliency_floor_eps_mode fixed \
  --saliency_floor_logit_eps -10 \
  --saliency_floor_warmup_steps 0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 100 \
  --learning_rate 2e-4 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none
' > runs/go_single/train_ce_sal_softmax_margin_logit_eps_m10_tau0p1_500_run1.log 2>&1 &
```

只训练 saliency loss：

```bash
nohup bash -lc '
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/train/train.py \
  --model_name_or_path models/Qwen2.5-Coder-7B-Instruct \
  --data_path data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json \
  --output_dir outputs/go_single/models/sal_only_softmax_margin_500_run1 \
  --use_peft True \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --loss_mode saliency_only \
  --saliency_loss_type softmax_margin \
  --saliency_alpha 1.5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 100 \
  --learning_rate 2e-4 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none
' > runs/go_single/train_sal_only_softmax_margin_500_run1.log 2>&1 &
```

### Saliency 可视化与诊断

相关代码：


| 路径                                                                   | 作用                                                                                                       |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `tools/visual_saliency/compute_teacher_forcing_annotation_saliency.py` | 对 base 和 SFT model 做 teacher-forcing saliency 计算，输出 recall@k、precision@k、mAP@k 和 top-k source。 |
| `tools/visual_saliency/build_base_vs_ours_annotation_viewer.py`        | 把 saliency JSON 渲染成单文件 HTML viewer。                                                                |
| `scripts/saliency_exp/base_saliency_distribution.py`                   | 统计 base model 在 500 个样本上的初始 saliency 分布。                                                      |
| `scripts/saliency_exp/trace_row_token_training.py`                     | 正常训练 500 样本，同时追踪某个样本/某个 token 的 loss、梯度和 top-k saliency 变化。                       |
| `scripts/saliency_exp/render_row_token_trace_report.py`                | 把 trace JSONL 渲染成可读 Markdown / txt。                                                                 |

生成训练前后 saliency 对齐数据：

```bash
nohup bash -lc '
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tools/visual_saliency/compute_teacher_forcing_annotation_saliency.py \
  --data_path data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json \
  --base_model models/Qwen2.5-Coder-7B-Instruct \
  --sft_model outputs/go_single/models/ce_sal_softmax_margin_logit_eps_m10_tau0p1_500_run1 \
  --base_name "Base Qwen" \
  --sft_name "CE+SAL Softmax-Margin eps=-10 tau=0.1" \
  --output_path outputs/visual_saliency/go_single_train_base_vs_cesal_softmax_margin_logit_eps_m10_tau0p1_teacher_saliency_data.json \
  --row_indices 232,34,289,344,490,416,460,479 \
  --max_samples 8 \
  --max_targets_per_sample 24 \
  --target_scope all \
  --top_k 20 \
  --device cuda:0 \
  --dtype bf16 \
  --language go \
  --local_files_only 1
' > runs/go_single/visual_saliency_cesal_softmax_margin_logit_eps_m10_tau0p1_top20.log 2>&1 &
```

渲染 HTML：

```bash
python tools/visual_saliency/build_base_vs_ours_annotation_viewer.py \
  --data_path outputs/visual_saliency/go_single_train_base_vs_cesal_softmax_margin_logit_eps_m10_tau0p1_teacher_saliency_data.json \
  --output_path outputs/visual_saliency/go_single_train_base_vs_cesal_softmax_margin_logit_eps_m10_tau0p1_teacher_saliency_viewer.html
```

追踪 row 479 的 `Service` token：

```bash
nohup bash -lc '
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/saliency_exp/trace_row_token_training.py \
  --model_name_or_path models/Qwen2.5-Coder-7B-Instruct \
  --data_path data/go_single/train_data/go_single_train_v2_graphsignal_500_compact.json \
  --output_dir outputs/go_single/models/ce_sal_softmax_margin_logit_eps_m10_tau0p1_500_trace_row479_service_run1 \
  --trace_output_dir outputs/saliency_exp/row479_service_trace_eps_m10_tau0p1_run1 \
  --use_peft True \
  --loss_mode ce_saliency \
  --saliency_loss_type softmax_margin \
  --saliency_lambda 0.1 \
  --saliency_alpha 0.1 \
  --saliency_floor_eps_mode fixed \
  --saliency_floor_logit_eps -10 \
  --saliency_floor_warmup_steps 0 \
  --trace_sample_index 479 \
  --trace_query_token_index 251 \
  --trace_query_token_text Service \
  --trace_top_k 20 \
  --trace_every_n_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 100 \
  --learning_rate 2e-4 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none
' > runs/go_single/train_trace_row479_service_eps_m10_tau0p1_run1.log 2>&1 &
```

把 trace 输出转成可读文本：

```bash
python scripts/saliency_exp/render_row_token_trace_report.py \
  --trace-jsonl outputs/saliency_exp/row479_service_trace_eps_m10_tau0p1_run1/trace.jsonl \
  --max-tokens 20
```

### Eval / Benchmark

相关代码：


| 路径                                                  | 作用                                                                                                                       |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `scripts/benchmark/benchmark_eval.py`                 | 通用 benchmark 评测入口。模型先 greedy 生成得到 pass@1，再采样`num_samples` 个候选统计 pass@10；可选计算 greedy CodeBLEU。 |
| `scripts/benchmark/eval_generation.py`                | 生成逻辑，包括 greedy generation、sample generation、上下文长度处理。                                                      |
| `scripts/benchmark/eval_judges.py`                    | judge 逻辑，根据数据集和语言执行测试。                                                                                     |
| `scripts/benchmark/eval_reporting.py`                 | pass@1/pass@10 聚合和 Markdown/CSV 表格输出。                                                                              |
| `scripts/benchmark/merge_benchmark_eval.py`           | 合并 sharded benchmark 输出。                                                                                              |
| `scripts/go_single/evaluate_go_single_predictions.py` | GoSingle 专用 prediction JSONL 评测入口。如果已经有外部生成文件，可以直接统计 pass@1、pass@k、CodeBLEU。                   |

用通用 evaluator 跑 MCEval GoSingle，记录 pass@1、pass@10、CodeBLEU：

```bash
nohup bash -lc '
cd /mnt/nvme0n1/wenhao/Empirical-Influence-Function
export PATH="$PWD/.local/bin:$PATH"
eval "$($PWD/.local/bin/micromamba shell hook -s bash 2>/dev/null || micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$PWD/.micromamba"
micromamba activate "$PWD/.micromamba/envs/eif-bench"

CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/benchmark/benchmark_eval.py \
  --model_path outputs/go_single/models/ce_sal_softmax_margin_logit_eps_m10_tau0p1_500_run1 \
  --eval_path data/go_single/eval_data/mceval_go_single_v2_chatml.jsonl \
  --output_dir outputs/benchmark/go_single_ce_sal_softmax_margin_logit_eps_m10_tau0p1_500_run1 \
  --baseline_name ce_sal_softmax_margin_logit_eps_m10_tau0p1_500_run1 \
  --languages go \
  --source_datasets mceval_go \
  --num_samples 10 \
  --temperature 0.2 \
  --top_p 0.95 \
  --infer_batch_size 4 \
  --sample_infer_batch_size 2 \
  --judge_workers 8 \
  --judge_timeout_sec 10 \
  --compute_codebleu
' > runs/go_single/mceval_ce_sal_softmax_margin_logit_eps_m10_tau0p1_500_run1.log 2>&1 &
```

对比其他模型时，只需要替换下面三个参数：

```bash
--model_path models/Qwen2.5-Coder-7B-Instruct
--output_dir outputs/benchmark/go_single_base_qwen_7b
--baseline_name base_qwen_7b
```

```bash
--model_path outputs/go_single/models/ce_loss_500_run1
--output_dir outputs/benchmark/go_single_ce_loss_500_run1
--baseline_name ce_loss_500_run1
```

```bash
--model_path outputs/go_single/models/ce_sal_softmax_margin_500_run1
--output_dir outputs/benchmark/go_single_ce_sal_softmax_margin_500_run1
--baseline_name ce_sal_softmax_margin_500_run1
```

分片评测时增加：

```bash
--num_shards 4 \
--shard_index 0
```

合并分片结果：

```bash
python scripts/benchmark/merge_benchmark_eval.py \
  --input_glob "outputs/benchmark/go_single_ce_sal_softmax_margin_500_run1_shards/shard*/ce_sal_softmax_margin_500_run1_shard*_benchmark_eval.json" \
  --output_dir outputs/benchmark/go_single_ce_sal_softmax_margin_500_run1_merged \
  --baseline_name ce_sal_softmax_margin_500_run1
```

如果已经有 prediction JSONL，可以用 GoSingle 专用 evaluator：

```bash
python scripts/go_single/evaluate_go_single_predictions.py \
  --eval-data data/go_single/eval_data/mceval_go_single_v2_canonical.jsonl \
  --predictions outputs/go_single/predictions/example_predictions.jsonl \
  --output outputs/go_single/eval_results/example_eval_results.jsonl \
  --summary outputs/go_single/eval_results/example_eval_summary.json \
  --pass-k 10 \
  --compute-codebleu
```
