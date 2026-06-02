# 数据治理与标注训练逻辑

本文只整理 GraphSignal curation 路线的算法逻辑和代码结构：标注如何产生、saliency 如何定义、loss 如何设计，以及这些信息如何进入训练。这里的 curation 不是简单清洗样本，而是把“哪些 token 关系应该被模型重视”显式写进训练监督。

## 一、整体算法链路

GraphSignal 的训练链路可以概括为：

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

## 二、标注逻辑

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

### 4. 目标区域限制

在 benchmark 适配里，原始样本包含 `chatml`、`fim`、`test` 等字段。训练和推理使用 `chatml`，测试 pass 使用 `fim` 拼接完整代码和 `test`。标注时可以只让 edge 落在需要治理的 answer/completion 区域，避免把大量 prompt 或上下文 token 都拉入 loss。

具体做法是保留全局 token index，但在提交边时用 `target_indices` 约束边的 target 区域。这样可视化和训练仍使用完整序列坐标，同时 loss 更集中到模型真正需要生成的部分。

## 三、Word Edge 到 Qwen BPE Edge

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

## 四、训练数据读取逻辑

训练集读取支持两种格式。

第一种是 benchmark compact format，通常由可视化/适配脚本产出：

```text
input_ids
label 或 labels
attention_edges
language / annotation_meta
```

这种格式已经把 ChatML 序列 tokenize 成 Qwen token，并且把边映射到 BPE token index。dataset 只需要校验长度、过滤越界边，然后把边读成 `annot_pairs`。

第二种是 legacy annotated format：

```text
sft_input
qwen_tokens
qwen_annotations
```

这种格式会根据 ChatML 模板判断 assistant 输出从哪里开始。输出开始前的 label 全部设为 `-100`，只对 assistant 部分做 next-token prediction。

无论哪种格式，单个训练样本最终都是：

```text
input_ids:   [T]
labels:      [T]，prompt 部分为 -100
annot_pairs: [N, 2]，每行是一条 BPE token edge
```

collator 会 padding `input_ids` 和 `labels`，但 `annot_pairs` 保持 ragged list。这样每个样本可以拥有不同数量的标注边。

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

新版 loss 现在不是 hinge loss，而是 InfoNCE / multi-positive NLL。

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

代码里这个 **\\tau** 仍然叫 `alpha`，所以要注意：现在 `alpha` 不是 margin，而是 temperature。

在所有 causal source 上做 softmax：

**p\_{q,s} = \\frac{ \\exp(\\ell\_{q,s}) }{ \\sum\_{r \\in N\_q} \\exp(\\max(\\ell\_{q,r},\\epsilon)+\\exp(\\ell\_{q,s}) }**

其中，eps 相当于负样本 floor，决定了负样本最低有多强，从数学形式上解析，eps 越大，对 neg 的惩罚力度越小。因此 eps 是关键超参数，影响到 loss 对负样本的打压力度

消融实验可以使用固定 eps、动态eps，主实验采取 EMA 动态 eps 的计算方式，如下

**\\hat{\\varepsilon}\_t = Q\_{0.75}(S\_t)**

**\\varepsilon\_t = \\beta\\varepsilon\_{t-1} + (1-\\beta)\\hat{\\varepsilon}\_t**

其中：t 代表一个 batch
S\_t=\\{c\_{q,r}(\\theta\_t)\\mid q\\in\\mathcal{B}\_t,\\ r<q\\}

经过实验结果验证可知，其他超参数最优为
Q=0.75,\\quad \\beta=0.95,\\quad \\varepsilon\_0=\\hat{\\varepsilon}\_1

对于一个 query **q**，multi-positive loss 是：

**\\mathcal{L}\_q = - \\frac{1}{|A\_q|} \\sum\_{s \\in A\_q} \\log p\_{q,s}**

总 saliency loss 是：

**\\mathcal{L}\_{sal} = \\frac{1}{|Q|} \\sum\_{q \\in Q} \\mathcal{L}\_q**

其中 **Q** 是满足下面条件的 query token 集合：

**|A\_q| > 0 \\quad \\text{and} \\quad |N\_q| > 0**

直观理解就是：

对每个 target token **q**，把所有前文 causal token 都作为候选 source，让 annotation source 在 softmax 分布里获得更高概率。它不只是要求 annotation saliency 高于平均 non-annotation saliency，而是让 annotation source 和所有 causal source 竞争排名。


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

## 八、代码文件结构及作用

### `src/annotate/`


| 文件                      | 主要作用                                                                                                                                                   |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `utils.py`                | 定义`SubwordToken`、`TokenCorrelation`，实现代码级 tokenizer、Qwen BPE tokenizer、simple-token 到 BPE-token 的字符区间映射。                               |
| `neural_annot.py`         | 标注核心。包含 tree-sitter 结构边提取器和 tool-calling LLM agent；结构边负责`bracket/defuse/call/return/type`，LLM 负责补充 `dataflow/semantic/api` 等边。 |
| `postprocessing.py`       | 将 MCEval / SFT 风格标注结果重建为 ChatML 序列，完成 word edge 到 Qwen BPE edge 的映射，并生成可视化数据。                                                 |
| `postprocessing_safim.py` | SAFIM 数据格式的后处理版本，逻辑与`postprocessing.py` 类似，但重建代码和 mask span 的方式不同。                                                            |
| `main.py`                 | MCEval 风格数据的标注入口，串联 tokenization、annotator、保存原始标注。                                                                                    |
| `main_safim.py`           | SAFIM 风格数据的标注入口。                                                                                                                                 |
| `viz_utils.py`            | 将 token、边和 attention/top-k 信息组织成可视化页面需要的结构。                                                                                            |
| `web_search.py`           | 可选 API 文档检索工具，供 LLM 判断 API 语义边时使用。                                                                                                      |

### `src/train/`


| 文件          | 主要作用                                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------------------------ |
| `dataset.py`  | 读取 compact benchmark format 或 legacy annotated format，统一转换成`input_ids/labels/annot_pairs`。               |
| `loss.py`     | 实现最后一层 contribution saliency、dense/sparse 两种 saliency loss 计算，以及训练日志诊断指标。                   |
| `train.py`    | 自定义 HuggingFace Trainer，组合 next-token loss 和 saliency loss，加载 Qwen、LoRA、dataset、callback 并执行训练。 |
| `attn_viz.py` | 训练期间抽取部分样本 attention，用于观察模型关注位置。                                                             |

### `src/scripts/`


| 文件       | 主要作用                                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| `train.sh` | 官方训练脚本模板，展示多卡 Qwen GraphSignal SFT 的默认参数组织方式；实际 benchmark 可以按数据路径、GPU 数、输出目录调整。 |
