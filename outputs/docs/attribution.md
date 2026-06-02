# 归因与 ALTI Saliency

本文只整理 `src/attribution` 中通过 ALTI saliency 做归因的实际算法逻辑，以及相关代码文件的职责。

## 一句话目标

这套代码不是只做“看 attention 热力图”。它要回答的是：

```text
测试样本中某个错误 token 为什么会被模型生成？
训练集中哪些样本、哪些 source -> target token 关系，
在模型内部机制上最像这个错误 token 的 source -> target 关系？
```

因此它把问题拆成两层：

```text
token attribution：测试错误 token 依赖了哪些 source tokens
sample/pair attribution：训练集中哪些 token correlation pair 与这个错误机制最相似
```

最终输出的核心对象不是单纯样本分数，而是按相似度排序的 pair：

```text
train_source -> train_target
匹配
test_source -> test_wrong_token
```

## ALTI Saliency 的核心计算

### 1. Causal LM 中 target token 的解释位置

对 decoder-only causal LM 来说，序列位置 `t` 的 token 是由前一个 hidden state 预测的。代码中 `target_idx_in_seq` 表示“被预测的 token 位置”，实际解释的是 prefix：

```text
input_ids[:, :target_idx_in_seq]
query position = target_idx_in_seq - 1
```

所以如果要解释 token `x_t`，模型只看 `x_0 ... x_{t-1}`，然后解释最后一个 prefix hidden state 对 `x_t` 的预测。

对应代码：

| 函数 | 作用 |
| --- | --- |
| `compute_alti_saliency_vector` | 输入 batch 和 `target_idx_in_seq`，返回长度为 `target_idx_in_seq` 的 source saliency 向量。 |
| `NewInferenceFunction.infer` | 找到 assistant answer 起点，生成模型输出，并可对 ground truth 或 generated sequence 的每个 response token 计算 saliency。 |

### 2. 单层 ALTI contribution matrix

对每一层 transformer decoder，代码构造一张 token-to-token contribution matrix：

```text
C_l[q, s] = source token s 对 query token q 在第 l 层 residual stream 中的贡献比例
```

它不是直接使用 attention weight，而是使用：

```text
attention_probs
value projection W_V
output projection W_O
residual connection
input layernorm
```

每个 source token 的 value 向量先经过 `W_V`，再经过对应 head 的 `W_O` 切片，得到 source 对 residual stream 的向量贡献。随后对每个 query `q`：

```text
source_vectors[q, s]
  = sum_h attention_h[q, s] * W_O^h W_V^h LN(x_s)
```

再把 self residual 加回 source 等于 query 的位置：

```text
source_vectors[q, q] += x_q
```

最后把向量贡献转成标量贡献比例。

对应代码：

| 函数 | 作用 |
| --- | --- |
| `_compute_qwen_alti_layer_matrix` | 为某一层构造完整 `C_l`，行是 query，列是 source。 |
| `_infer_qwen_attention_layout` | 推断 Qwen 的 attention head、head_dim、KV head，用于兼容 GQA/MQA。 |
| `_repeat_kv_for_alti` | 把 KV heads repeat 到 query heads，保证 value states 与 attention heads 对齐。 |
| `_normalize_alti_importance` | 将 source contribution vectors 归一化成每个 query 行上的 contribution distribution。 |

### 3. ALTI 标量归一化

代码没有直接对贡献向量取范数作为最终权重，而是使用一种 min-sum 风格归一化。设某个 query 的 source contribution vectors 为：

```text
T_q(s)
```

其合成输出为：

```text
y_q = sum_s T_q(s)
```

代码为每个 source 计算：

```text
score(q, s) = max(||y_q||_p - ||T_q(s) - y_q||_p, 0)
```

然后在 source 维度归一化：

```text
C_l[q, s] = score(q, s) / sum_j score(q, j)
```

直觉是：如果去掉某个 source 的贡献后，离最终合成输出更远，则这个 source 对当前 query 更重要。

### 4. 跨层 rollout

每层得到一张 `C_l` 后，代码做逐层 rollout：

```text
R_1 = C_1
R_l = C_l @ R_{l-1}
```

最终取目标 query 行：

```text
saliency_vector = R_L[target_idx - 1, :target_idx]
```

这个向量的第 `s` 个值表示：source token `s` 通过多层 attention/residual mixing 后，对目标 token 预测位置的贡献。

对应代码：

| 函数 | 作用 |
| --- | --- |
| `compute_alti_saliency_vector` | 逐层调用 `_compute_qwen_alti_layer_matrix`，并用矩阵乘法 rollout，最终返回 target row。 |

## 从测试错误 token 到训练样本的归因流程

主流程在 `run_causal_intervention_experiment` 中。它有 single-token 和 all-tokens 两种模式。

### 1. 构造测试序列

代码先读取一个 test sample，找到 assistant answer 起点，然后让模型从 prompt 开始 greedy generate。接着把：

```text
prompt tokens + generated tokens
```

拼成新的 `test_batch`。这个 batch 的 prompt 部分 label 被 mask，只把 generated response 视作要分析的目标区域。

这样做的原因是：归因目标通常不是 gold answer 中的 token，而是模型实际生成出来的错误 token。

对应代码：

| 位置 | 作用 |
| --- | --- |
| `NewInferenceFunction.infer` | 生成模型输出，返回 prompt length、生成 token、完整 token 序列等信息。 |
| `run_causal_intervention_experiment` | 用 prompt 和 generated tokens 重新组装 `test_batch`。 |

### 2. Stage 1：测试侧 ALTI source 选择

对于目标错误 token `t`，代码调用：

```text
compute_alti_saliency_vector(model, test_batch, t)
```

得到所有 prefix source tokens 对 `t` 的 saliency。随后过滤掉无语义 token：

```text
特殊 token、role label、空白、括号、逗号、分号、单个无意义符号
```

再取 top-k source tokens，形成测试侧 correlation pairs：

```text
test_source_i -> test_wrong_token
```

对应代码：

| 函数 | 作用 |
| --- | --- |
| `is_trivial_token` | 判断 token 是否应从 saliency source 候选中移除。 |
| `top_nontrivial_saliency_sources` | 从 saliency vector 中选出 top-k 非 trivial source tokens。 |

### 3. Stage 1b：测试侧 pair 的参数空间特征

单纯知道 `test_source -> wrong_token` 的 saliency 高还不够。代码进一步把每条测试侧 correlation pair 表示成一个参数空间向量：

```text
feature(test_source -> test_target)
  = grad_theta ALTI(test_source -> test_target)
```

这里的 `theta` 默认不是全模型参数，而是最后若干层 attention projection 参数。当前默认是：

```text
最后 1 层 self-attention 的 q_proj / k_proj
```

这个特征的含义是：如果微调这些参数，这条 ALTI correlation score 会怎样变化。它比原始 saliency 分数更适合拿来和训练样本中的 correlation pair 做机制匹配。

对应代码：

| 函数 | 作用 |
| --- | --- |
| `make_attention_projection_filter` | 选择 fine matching 使用的 attention projection 参数。 |
| `compute_alti_correlation_gradient` | 对单条 `source -> target` ALTI score 求参数梯度，返回 flatten 后的 feature vector。 |
| `_compute_alti_correlation_gradient_retry` | 在 OOM 时降低 chunk size 重试，仍失败则跳过该 pair。 |

### 4. Stage 2：训练样本粗筛

对全训练集直接做 ALTI-gradient pair matching 很贵，所以代码先做 coarse prescreen。

粗筛使用 LM-head CE gradient 的相似度。对测试目标 token 构造只在该 token 位置有 label 的 batch，计算：

```text
G_test = grad CE(test_token) / grad W_lm_head
```

对每个训练样本也计算类似的 LM-head gradient，并用 cosine similarity 打分：

```text
score(train_sample)
  = cosine(G_test, G_train_sample)
```

为了提速，代码对 causal LM 的 LM-head 梯度使用解析式，不对 transformer 做 backward：

```text
grad_logits = softmax(logits) - one_hot(label)
grad_W = grad_logits^T @ hidden
```

还可以用 TensorSketch 把巨大的 `vocab_size x hidden_size` 梯度压缩成低维 sketch，并缓存训练集 sketch。

对应代码：

| 函数 | 作用 |
| --- | --- |
| `compute_lm_head_ce_gradient_no_backward` | 解析式计算单个 batch 的 LM-head CE gradient。 |
| `compute_lm_head_ce_gradient_scores_no_backward` | 批量扫描训练样本，直接输出与 test gradient 的 cosine 分数。 |
| `compute_lm_head_ce_gradient_sketches_no_backward` | 计算低维 TensorSketch，用于缓存和快速检索。 |
| `_load_or_build_prescreen_sketch_cache` | 构建或读取训练集 sketch cache。 |
| `_score_prescreen_sketch_cache` | 用 test sketch 与缓存 sketch 做矩阵乘法评分。 |
| `_screen_training_set` | 不使用 sketch 时，对训练集做精确 LM-head gradient 粗筛。 |

### 5. Stage 3：训练侧 candidate pair 构造

粗筛得到 top train samples 后，代码进入每个候选训练样本内部寻找 token correlation pair。

对一个训练样本：

1. 找到 assistant answer 起点。
2. 从 answer 起点开始扫描最多 `TOP_TARGETS` 个非 trivial target tokens。
3. 对每个 train target token `t_train` 计算 ALTI saliency。
4. 从 saliency vector 中选出 `TOP_K_SOURCE_PER_TARGET` 个非 trivial source tokens。
5. 形成训练侧候选 pair：

```text
train_source -> train_target
```

对应代码：

| 函数/位置 | 作用 |
| --- | --- |
| `_find_subseq_start` | 在 token 序列中定位 assistant answer 起点。 |
| `find_first_valid_token_index` | 跳过换行、空白、括号等，找到第一个有效 answer token。 |
| `_process_train_sample_stage3` | 对单个训练样本构造 train-side ALTI candidate pairs，并保存每个 target 的 saliency。 |
| `get_context_window` | 为 source/target token 保存局部上下文，便于后续人工查看。 |

### 6. Stage 3b：train/test pair fine matching

对每条训练侧候选 pair，代码同样计算：

```text
feature(train_source -> train_target)
  = grad_theta ALTI(train_source -> train_target)
```

然后与每条测试侧 pair feature 做 cosine similarity：

```text
cos_sim
  = cosine(feature_train_pair, feature_test_pair)
```

最终所有 pair 按 `cos_sim` 降序排列。分数越高，表示训练样本中的这条 token correlation 在参数空间里越像测试错误 token 的 correlation。

这就是当前代码中最核心的“通过 ALTI saliency 做归因”：

```text
不是只比较样本 CE gradient，
而是先用 ALTI 找 token pair，
再用 ALTI score 的参数梯度比较 pair-level 机制相似性。
```

输出记录中主要字段含义：

| 字段 | 含义 |
| --- | --- |
| `top_correlations` | 测试错误 token 的 top ALTI source tokens。 |
| `correlation_pairs` | train pair 与 test pair 的 fine matching 结果。 |
| `cos_sim` | train/test ALTI-gradient pair feature 的 cosine similarity。 |
| `coarse_cos_sim` | Stage 2 中训练样本与测试 token 的粗筛相似度。 |
| `train_sample_details` | 候选训练样本的 token、answer 起点、saliency 和上下文信息。 |

## Single-token 模式与 All-tokens 模式

### Single-token 模式

Single-token 模式只解释一个指定错误 token：

```text
TOKEN_INDEX_TO_RETRIEVE
```

算法路径是：

```text
指定 test token
-> 计算 test ALTI top sources
-> 计算 test pair ALTI-gradient features
-> 全训练集 coarse prescreen
-> top train samples 内构造 train ALTI pairs
-> train/test pair cosine matching
```

输出以一个目标 token 为中心，适合分析“第一个错误 token”这种明确问题。

### All-tokens 模式

All-tokens 模式分析模型生成 response 中的多个非 trivial token。为了避免每个 token 都全量扫描训练集，代码先做一次全局粗筛：

```text
full response CE gradient -> top COARSE_POOL_SIZE train samples
```

然后对每个 test output token：

```text
在 coarse pool 内重新按该 token 的 CE gradient rerank
-> 做该 token 的 ALTI pair fine matching
```

也就是：

```text
1 次全量 prescreen + 多个 token 的 pool 内 rerank + pair-level matching
```

这比每个 token 都扫描全训练集更便宜，同时仍保留每个 token 的单独 attribution 结果。

## 代码文件结构及作用

### `src/attribution/saliency.py`

| 函数 | 主要作用 |
| --- | --- |
| `compute_alti_saliency_vector` | 前向计算目标 token 的 ALTI source saliency vector。 |
| `_compute_qwen_alti_layer_matrix` | 构造单层 token-to-token contribution matrix。 |
| `_compute_qwen_alti_layer_relevance` | 不显式保存完整 rollout，计算一层对 relevance vector 的传播。 |
| `_compute_qwen_alti_layer_target_relevance` | 只计算最后一层目标 query 行，用于降低 ALTI-gradient 显存。 |
| `compute_alti_correlation_gradient` | 对一条 `source -> target` ALTI score 求参数梯度，作为 pair-level matching feature。 |
| `compute_lm_head_ce_gradient_no_backward` | 用解析式计算 LM-head CE gradient，用于粗筛。 |
| `compute_lm_head_ce_gradient_scores_no_backward` | 批量计算训练样本与测试 CE gradient 的 cosine 分数。 |
| `compute_lm_head_ce_gradient_sketches_no_backward` | 生成 LM-head gradient 的 TensorSketch，用于缓存和快速检索。 |

### `src/attribution/intervention_experiment.py`

| 函数/模块 | 主要作用 |
| --- | --- |
| `run_causal_intervention_experiment` | 当前 ALTI correlation matching 主流程入口。 |
| `make_attention_projection_filter` | 选择 fine matching 的参数子空间，例如最后一层 `q_proj/k_proj`。 |
| `top_nontrivial_saliency_sources` | 从 ALTI saliency 中过滤 trivial token 并取 top-k source。 |
| `_screen_training_set` | 用 LM-head CE gradient 对训练集做 coarse prescreen。 |
| `_load_or_build_prescreen_sketch_cache` | 构建或读取训练集 prescreen sketch cache。 |
| `_process_train_sample_stage3` | 对候选训练样本生成 train-side ALTI pairs，并与 test pairs 做 fine matching。 |

### `src/attribution/NIF.py`

| 类/函数 | 主要作用 |
| --- | --- |
| `NewInferenceFunction` | 封装推理、生成、saliency 计算和旧版 gradient influence 流程。 |
| `NewInferenceFunction.infer` | 找 assistant 起点，生成模型输出，并组织 ground truth/generated sequence 的 token 与 saliency。 |
| `load_model_and_tokenizer` | 加载归因实验使用的模型和 tokenizer。 |
| `build_train_dataset` / `build_single_sample_dataset` | 将原始样本转换成归因实验需要的 tokenized dataset。 |
| `influence_gradient_single` | 旧版样本级 gradient influence 逻辑；当前代码注释也提示更推荐使用 ALTI correlation matching。 |

### `src/attribution/process_data.py`

| 函数/类 | 主要作用 |
| --- | --- |
| `process_func_chatml` | 把样本转成 ChatML tokenization 格式，并只对 assistant answer 部分保留 labels。 |
| `CustomCollator` | 在 HuggingFace collator 基础上保留 `sample_index`，使归因结果能映射回训练样本。 |
| `extract_code_content` | 从原始 prompt 中抽取与代码补全最相关的结构定义和目标函数内容。 |

### `src/attribution/auto_annotate.py`

| 函数 | 主要作用 |
| --- | --- |
| `annotate_sample` / `annotate_samples` | 使用外部 LLM 标注训练样本中可能影响答案的上下文片段，属于早期辅助分析流程。 |
