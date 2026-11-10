# 06 Transformer 与大模型原理

> 本章目标：从“下一个 token 预测”出发，理解 LLM 的结构、训练流程、推理过程和能力边界。不要死记模型版本号——架构和训练范式比版本更重要，版本每月都在变。

---

## 6.1 LLM 的本质：自回归下一个 token 预测

给定 token 序列 `x_1, x_2, ..., x_t`，模型学习：

```
P(x_{t+1} | x_1, ..., x_t)
```

训练目标是最小化负对数似然：

```
L = -Σ log P(x_{t+1} | x_≤t)
```

生成时：

```
采样 x_{t+1} → 拼回序列 → 预测 x_{t+2} → ...
```

所有“对话、写作、代码、推理”能力，都是这个目标在大规模数据和大参数量下涌现出来的。理解这一点，就能理解 LLM 的几个固有特性：

- 它是概率模型，不是数据库，因此会“编”；
- 它只看到上下文，没有长期记忆；
- 它的知识截止于训练数据；
- 它的输出质量高度依赖提示和上下文。

### 2026 年的模型格局（示例，非穷举）

| 家族 | 代表 | 特点 |
|---|---|---|
| DeepSeek | V3/R1 及后续版本 | MoE、推理能力强、开源权重 |
| Qwen | Qwen3 及后续 | 尺寸全、中文强、生态好 |
| Llama | Llama 3/4 系列 | 社区生态、工具支持广 |
| GLM | GLM 系列 | 中文、开源 |
| Kimi | Kimi 系列 | 长上下文 |
| Claude | Claude 系列 | 长上下文、工具调用、安全性 |
| GPT | GPT 系列 | 通用能力、生态 |
| Gemini | Gemini 系列 | 多模态、长上下文 |

> 选型时看：许可证、上下文长度、中文能力、工具调用、成本、可部署性、评测数据是否可信。不要只看排行榜。

---

## 6.2 Tokenizer：文本如何变成数字

### 常见算法

| 算法 | 代表模型 | 特点 |
|---|---|---|
| BPE | GPT、Llama | 从字符开始合并高频对 |
| WordPiece | BERT | 基于似然的子词合并 |
| SentencePiece | Llama、T5、Qwen | 语言无关，支持 BPE/Unigram |
| Unigram | T5、ALBERT | 概率模型选择子词 |

### 关键概念

- **词表大小**：常见 32k–256k；越大，单 token 信息越多，但 Embedding 和输出层越大；
- **特殊 token**：`<bos>`、`<eos>`、`<pad>`、`<|im_start|>`、工具调用标记等；
- **中文 token 效率**：不同 tokenizer 对中文的压缩率差异很大，影响成本、上下文长度和推理速度；
- **Token 计数**：计费、上下文预算、RAG 切分都按 token 算，不按字符算。

```python
# 以 transformers 为例
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
ids = tok("你好，世界", return_tensors="pt").input_ids
print(ids, tok.convert_ids_to_tokens(ids[0]))
```

### 工程影响

- 同样的中文文本，不同模型 token 数可能差 30%–100%；
- 上下文窗口是 token 数，不是字数；
- 截断、分块、max_tokens 都要按目标模型的 tokenizer 计算；
- 特殊 token 处理错会导致对话格式错乱、工具调用失败。

---

## 6.3 Embedding 与位置编码

### Token Embedding

每个 token 映射为 `d_model` 维向量。输出层通常与 Embedding 权重共享（weight tying），减少参数。

### 位置编码

自注意力本身对顺序不敏感，必须注入位置信息。

| 方法 | 特点 |
|---|---|
| 正弦位置编码 | 固定、可外推，早期 Transformer |
| 可学习位置编码 | 简单，但受最大长度限制 |
| RoPE（旋转位置编码） | 相对位置、外推性好，LLaMA/Qwen 主流 |
| ALiBi | 线性偏置，外推性好 |
| 相对位置偏置 | T5 等 |

### RoPE 的直觉

把 Q/K 向量按位置旋转一个角度，使点积只依赖相对位置差。位置越远，旋转差越大。上下文扩展时常用：

- **位置插值（PI）**：把长位置压缩到训练范围内；
- **NTK-aware 插值**：调整频率基；
- **YaRN**：更精细的频率缩放，支持更长上下文。

> 长上下文不是“把窗口调大”就行。模型需要在长序列上训练过，否则注意力会退化。实际使用要看“有效上下文长度”，不是标称值。

---

## 6.4 注意力机制

### 缩放点积注意力

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

- Q：查询（我在找什么）
- K：键（我有什么）
- V：值（我提供什么）
- `√d_k`：防止点积过大导致 softmax 饱和

### 多头注意力

```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
MultiHead = Concat(head_1, ..., head_h) W^O
```

不同头关注不同关系（语法、指代、位置、语义）。头数、头维度、总维度要匹配。

### 因果掩码

Decoder-only 模型不能让 token 看到未来：

```
mask[i, j] = 0 if j <= i else -inf
```

softmax 后未来位置的权重为 0。

### 复杂度

```
时间：O(n² · d)
显存：O(n² · heads)（朴素实现）
```

长上下文的核心成本在注意力。优化方向：

| 方法 | 思路 |
|---|---|
| FlashAttention | 分块计算、减少 HBM 读写，不改变数学结果 |
| 稀疏/滑动窗口 | 只关注局部或部分位置 |
| 线性注意力 | 用核函数近似，复杂度 O(n) |
| GQA/MQA | 多个 Q 头共享 K/V 头，减少 KV Cache |
| 位置外推 | RoPE 缩放、YaRN |

### GQA / MQA

| 类型 | K/V 头数 | 效果 |
|---|---|---|
| MHA | 与 Q 头相同 | 质量最好，KV Cache 最大 |
| MQA | 1 组 K/V | KV Cache 最小，质量略降 |
| GQA | 分组共享 | 折中，主流 LLM 常用 |

KV Cache 大小与 K/V 头数成正比。GQA 是长上下文和高并发部署的关键设计。

### 一个最小注意力实现

```python
import torch, math

def attention(q, k, v, causal=True):
    # q: (B, H, T, D), k/v: (B, H, T, D)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        T = q.size(-2)
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v
```

---

## 6.5 Transformer 结构

### Decoder-only 块

```
x = x + Attention(LN(x))      # Pre-LN
x = x + FFN(LN(x))
```

| 组件 | 作用 | 常见选择 |
|---|---|---|
| LayerNorm | 稳定训练 | RMSNorm（LLaMA 系） |
| 注意力 |  token 间信息交换 | MHA/GQA |
| FFN | 逐 token 非线性变换 | SwiGLU，隐藏维约 3.5–4× |
| 残差 | 梯度高速公路 | 标准 |
| Dropout | 正则 | 大模型预训练常关 |
| 位置编码 | 顺序信息 | RoPE |

### MoE（混合专家）

```
y = Σ_{i∈TopK} g_i(x) · Expert_i(x)
```

- 总参数大、每 token 只激活少数专家（如 8 选 2）；
- 优点：同等算力下容量更大；
- 难点：负载均衡、通信开销、显存占用、训练稳定性；
- 例子：DeepSeek 系列、Qwen MoE、Llama 4、Mixtral。

**看 MoE 模型要看“总参数”和“激活参数”两个数。** 671B 总参数、37B 激活，推理算力按激活参数估，但显存要装下更多（或做专家卸载）。

---

## 6.6 预训练

### 数据

| 类型 | 例子 | 注意 |
|---|---|---|
| 网页 | Common Crawl | 清洗、去重、质量过滤 |
| 代码 | GitHub | 许可证、去重、语法 |
| 书籍/论文 | 出版物、arXiv | 版权 |
| 对话/指令 | 社区数据、合成数据 | 质量与多样性 |
| 多语言 | 各语种语料 | 配比影响能力 |

### 数据处理流水线

```
抓取 → 解析 → 语言识别 → 质量过滤 → 去重（精确/近似）
→ 敏感内容过滤 → 分词 → 打包（packing）→ 训练
```

**去重非常重要**：重复数据会导致记忆、评测污染、训练效率下降。

### 训练

- 目标：next token prediction；
- 规模：数千到数万 GPU/加速器，数月；
- 并行：数据并行 + 张量并行 + 流水并行 + 序列并行 + ZeRO/FSDP；
- 精度：BF16/FP16，FP8 在新硬件上常用；
- 稳定性：loss spike、梯度裁剪、检查点、回滚；
- 成本：数百万到数千万美元量级（大模型）。

### Scaling Laws

- Kaplan 等（2020）：loss 随参数量、数据量、算力呈幂律下降；
- Chinchilla（2022）：计算最优时，参数量和训练 token 数应同比增加，约 **20 tokens/参数**；
- 后续实践：为了推理成本，常“过度训练”小模型（更多 token、更小参数）。

这解释了为什么现在有很多 1B–8B 模型能在端侧可用：它们用了远超 Chinchilla 比例的数据训练。

---

## 6.7 后训练：从“会续写”到“会对话”

### SFT（监督微调）

用“指令 → 回答”数据继续训练：

```
<system>你是一个助手</system>
<user>问题</user>
<assistant>答案</assistant>
```

只对回答部分计算损失（或按权重）。数据质量比数量重要，几千到几万条高质量样本就能显著改变行为。

### RLHF

```
1. 训练奖励模型（人类偏好排序）
2. 用 PPO 优化策略模型，使奖励最大化，同时约束不要偏离太远
```

优点：对齐人类偏好；缺点：复杂、不稳定、成本高。

### DPO

直接用手偏好对（chosen/rejected）优化，不需要单独奖励模型和 RL 循环：

```
L = -log σ(β · (log π(chosen) - log π_ref(chosen)
             - log π(rejected) + log π_ref(rejected)))
```

更简单、更稳定，广泛用于开源模型。

### GRPO / 可验证奖励

- GRPO：对同一问题采样多个回答，用组内相对奖励估计优势，省去 critic；
- RLVR（可验证奖励）：数学、代码等有确定答案的任务，用规则验证给奖励；
- 推理模型（如 DeepSeek-R1 类）大量使用这类方法，训练模型生成长链思考。

### 对齐与安全

- 拒绝有害请求；
- 防止越狱；
- 减少偏见；
- 工具调用权限控制；
- 输出审核。

后训练不能消除所有问题，安全需要系统级防护（输入过滤、权限、沙箱、人工复核）。

---

## 6.8 推理过程

### Prefill 与 Decode

| 阶段 | 输入 | 特点 | 瓶颈 |
|---|---|---|---|
| Prefill | 整个 prompt | 并行计算所有 token | 算力（compute-bound） |
| Decode | 逐 token 生成 | 串行、每步只算一个 token | 显存带宽（memory-bound） |

这解释了：

- 长 prompt 的首 token 延迟（TTFT）高；
- 生成长文本时，每 token 延迟（TPOT）受显存带宽限制；
- 批量增大能提高吞吐，但会增加单请求延迟。

### KV Cache

自回归生成时，已计算的 K/V 可以缓存，避免重复计算：

```
KV bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × batch × bytes
```

以 Llama 3 8B（32 层、8 个 KV 头、head_dim 128、FP16）为例：

```
每 token = 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB
8K 上下文 ≈ 1 GB
```

KV Cache 是长上下文和高并发的显存大头。优化：GQA、PagedAttention、KV 量化、滑动窗口、前缀缓存。

### 采样策略

| 策略 | 做法 | 效果 |
|---|---|---|
| Greedy | 取 argmax | 确定、易重复 |
| Beam Search | 保留 top-k 序列 | 翻译/摘要常用 |
| Temperature | 调整分布锐度 | 控制随机性 |
| Top-k | 只从 top-k 采样 | 截断长尾 |
| Top-p | 累积概率 p 内采样 | 动态候选集 |
| Repetition Penalty | 惩罚已出现 token | 减少复读 |

代码/数学任务通常 temperature 低（0–0.3），创意写作高（0.7–1.0）。

### 投机解码（Speculative Decoding）

用小模型快速草拟多个 token，大模型并行验证，接受匹配的部分。能在不改变输出的前提下加速 1.5–3 倍（取决于草稿命中率）。

### 推理优化

- 连续批处理（continuous batching）；
- PagedAttention（分页 KV Cache）；
- 前缀缓存（prefix caching）；
- 量化（INT8/INT4/FP8）；
- 张量并行/流水并行；
- MoE 专家并行；
- CUDA Graph / 算子融合。

详见第 08 章。

---

## 6.9 评测

### 自动指标

| 指标 | 用途 | 局限 |
|---|---|---|
| Perplexity | 语言建模 | 不完全对应任务效果 |
| 准确率/EM/F1 | 问答、抽取 | 依赖数据集 |
| Pass@k | 代码 | 需要执行环境 |
| BLEU/ROUGE | 生成重叠 | 与人类判断相关性有限 |
| LLM-as-judge | 主观质量 | 有偏、需校准 |

### 常见基准

- MMLU / MMLU-Pro：多学科知识；
- GSM8K / MATH：数学推理；
- HumanEval / MBPP / SWE-bench：代码；
- HellaSwag / ARC：常识推理；
- C-Eval / CMMLU：中文知识；
- MT-Bench / Arena：对话偏好；
- RAGAS：RAG 质量（忠实度、相关性）。

### 评测的坑

- **数据污染**：测试集出现在训练数据里；
- **提示敏感**：换个 prompt 分数差很多；
- **评测口径不一致**：few-shot 数、temperature、答案解析方式不同；
- **过拟合榜单**：只优化 benchmark，不解决真实任务。

**正确做法：建立自己的业务评测集（100–500 条），固定 prompt 和参数，人工标注 + 自动指标结合。**

---

## 6.10 能力边界与常见误解

| 误解 | 现实 |
|---|---|
| “模型知道所有知识” | 知识有截止日期，且可能记错 |
| “上下文窗口大就能记住所有内容” | 中间内容易被忽略（lost in the middle） |
| “推理能力强等于不会算错” | 仍会犯低级错误，需要工具/校验 |
| “微调可以注入新知识” | 微调更适合学格式/风格，知识注入优先 RAG |
| “模型输出是确定的” | 采样有随机性，同一输入可能不同 |
| “开源模型一定便宜” | 自部署有硬件、运维、人力成本 |

### 幻觉的成因

- 训练目标是“像文本”，不是“真”；
- 知识不足时倾向于编造；
- 提示模糊、缺少依据；
- 解码随机性；
- 评测/对齐奖励了“自信的回答”。

缓解：RAG、引用来源、工具校验、降低温度、要求“不知道就说不知道”、人工复核。

---

## 6.11 自检问题

1. 为什么 LLM 会幻觉？从训练目标角度解释。
2. Tokenizer 如何影响成本、上下文和中文效果？
3. 注意力的复杂度是多少？FlashAttention 优化了什么？
4. GQA 为什么能减少 KV Cache？代价是什么？
5. Prefill 和 Decode 的瓶颈分别是什么？
6. KV Cache 的显存怎么估算？8B 模型 8K 上下文大约多少？
7. SFT、RLHF、DPO 的区别是什么？
8. MoE 的“总参数”和“激活参数”分别影响什么？
9. 为什么“上下文窗口大”不等于“能有效利用长上下文”？
10. 如何设计一个可信的 LLM 业务评测？

---

## 6.12 推荐练习

1. 用 PyTorch 实现一个单头因果注意力，验证 mask 正确；
2. 用 `transformers` 加载一个小模型，观察 tokenizer 对中英文的切分；
3. 手算一个 8B 模型在 FP16/INT8/INT4 下的权重显存；
4. 对比 temperature=0 和 1 时同一 prompt 的输出；
5. 在一个小语料上训练一个 mini GPT（nanoGPT 风格），理解完整流程；
6. 用同一组问题评测两个模型，记录准确率、延迟、成本。
