# 08 LLM 推理部署与端侧 LLM

> 本章目标：理解 LLM 推理的成本结构，掌握量化、推理引擎、并发优化和端侧部署的基本方法。核心公式只有几个：显存 ≈ 权重 + KV Cache + 开销；解码速度 ≈ 内存带宽 / 每 token 读取量。

---

## 8.1 关键指标

| 指标 | 含义 | 影响 |
|---|---|---|
| TTFT | Time To First Token，首 token 延迟 | 交互体验、prefill 性能 |
| TPOT | Time Per Output Token，每 token 延迟 | 生成速度、decode 性能 |
| TPS | Tokens Per Second | 单请求生成速度 |
| Throughput | 系统每秒总 token 数 | 并发能力、成本 |
| Concurrency | 同时处理的请求数 | 显存、批处理 |
| Goodput | 满足 SLA 的有效吞吐 | 真实服务能力 |
| 成本 | $ / 1M tokens | 商业可行性 |

**注意区分**：单请求快 ≠ 系统吞吐高。增大 batch 能提高吞吐，但可能增加单请求延迟。要同时报告 TTFT、TPOT、吞吐和并发数。

---

## 8.2 显存估算

### 权重

```
weights_bytes = params × bytes_per_param
```

| 模型 | FP16 | INT8 | INT4 |
|---|---:|---:|---:|
| 0.5B | 1 GB | 0.5 GB | 0.25 GB |
| 1.5B | 3 GB | 1.5 GB | 0.75 GB |
| 3B | 6 GB | 3 GB | 1.5 GB |
| 7B | 14 GB | 7 GB | 3.5 GB |
| 8B | 16 GB | 8 GB | 4 GB |
| 13B | 26 GB | 13 GB | 6.5 GB |
| 70B | 140 GB | 70 GB | 35 GB |

MoE 模型要按“总参数”算显存（所有专家都要加载），按“激活参数”估算算力。

### KV Cache

```
KV_bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × batch × bytes
```

示例：Qwen2.5-1.5B（28 层、2 个 KV 头、head_dim 128、FP16）：

```
每 token = 2 × 28 × 2 × 128 × 2 = 28,672 B ≈ 28 KB
4K 上下文 ≈ 114 MB
```

Llama 3 8B（32 层、8 个 KV 头、head_dim 128、FP16）：

```
每 token ≈ 128 KB
8K 上下文 ≈ 1 GB
```

### 总显存

```
总显存 ≈ 权重 + KV Cache + 激活 + 框架开销 + 碎片
```

经验：

- 推理框架开销约 1–3 GB（CUDA context、workspace）；
- 长上下文/高并发时 KV Cache 可能超过权重；
- 留 10%–20% 余量，避免 OOM。

### 训练显存

全量微调（Adam + FP16）大约需要 **16–20 bytes/参数**（权重、梯度、优化器状态、激活）。7B 模型全量微调通常需要 80GB 级 GPU；LoRA 可降到 16–24GB；QLoRA 更低。

---

## 8.3 量化

### 精度与格式

| 格式 | 每参数位数 | 特点 |
|---|---:|---|
| FP32 | 32 | 训练/高精度 |
| FP16/BF16 | 16 | 推理常用 |
| FP8 | 8 | H100/新硬件，精度好 |
| INT8 | 8 | 广泛支持，需校准 |
| INT4 | 4 | 端侧/消费级 GPU，精度需验证 |
| NF4 | 4 | QLoRA 常用 |

### 主流方法

| 方法 | 类型 | 特点 | 适用 |
|---|---|---|---|
| GPTQ | 训练后 | 逐层量化，3/4/8bit | GPU 推理 |
| AWQ | 训练后 | 保护显著权重，4bit 质量好 | GPU/端侧 |
| GGUF | 格式+量化 | llama.cpp 生态，Q2–Q8 多档 | CPU/边缘 |
| SmoothQuant | 训练后 | 平滑激活离群值，W8A8 | 服务器 INT8 |
| bitsandbytes | 训练/推理 | NF4/INT8，QLoRA | 低显存微调 |
| QAT | 训练中 | 量化感知训练，精度最好 | 高要求端侧 |

### 量化对能力的影响

- 聊天/摘要：4bit 通常可接受；
- 数学/代码/长链推理：对精度更敏感，优先 8bit 或 FP8；
- 小模型（< 1B）：量化损失相对更明显；
- 长上下文：KV Cache 量化会放大误差，需实测。

**必须用自己的评测集验证**，不要只看困惑度。

### KV Cache 量化

把 KV Cache 存成 FP8/INT8，可显著降显存、提并发。风险是长上下文精度下降。vLLM、TensorRT-LLM 等支持。

---

## 8.4 推理引擎

| 引擎 | 特点 | 适合 |
|---|---|---|
| vLLM | PagedAttention、连续批处理、OpenAI 兼容、生态好 | 服务端通用 |
| SGLang | RadixAttention 前缀缓存、结构化生成、快 | 高并发/Agent |
| TensorRT-LLM | NVIDIA 深度优化、编译式、性能强 | NVIDIA GPU 生产 |
| TGI | Hugging Face 生态、生产特性 | HF 模型服务 |
| llama.cpp | C++、GGUF、CPU/ARM/GPU、轻量 | 端侧/本地 |
| Ollama | 基于 llama.cpp，易用 | 开发/个人 |
| MLC-LLM | TVM 编译，跨平台 | 多端部署 |
| ExLlamaV2 | 消费级 GPU、量化快 | 本地 GPU |
| LMDeploy | 国产模型生态、高效 | 服务端 |

### 选型建议

- 服务端 NVIDIA GPU：vLLM 或 TensorRT-LLM；
- 需要极致吞吐/前缀复用：SGLang；
- 本地/端侧：llama.cpp / Ollama / MLC；
- 国产 NPU：厂商 SDK（RKLLM、QNN、OpenVINO 等）。

---

## 8.5 服务端优化

### 连续批处理（Continuous Batching）

传统 batch 要等最长请求结束；连续批处理在每个 decode step 动态加入/移除请求，大幅提高 GPU 利用率。

### PagedAttention

把 KV Cache 分成固定大小的 block，按需分配，像操作系统分页一样，减少碎片和浪费。vLLM 的核心技术。

### 前缀缓存（Prefix Caching）

相同 system prompt / 文档前缀只算一次，后续请求复用 KV。对 RAG、Agent、多轮对话收益很大。SGLang 的 RadixAttention 是典型实现。

### 投机解码

小模型草拟 + 大模型验证。适合延迟敏感、草稿命中率高的场景。

### 并行

| 方式 | 切分对象 | 场景 |
|---|---|---|
| 张量并行（TP） | 单层内矩阵 | 单机多卡、降延迟 |
| 流水并行（PP） | 按层切分 | 多机、超大模型 |
| 数据并行（DP） | 复制模型 | 提高吞吐 |
| 专家并行（EP） | MoE 专家 | MoE 大模型 |

### Chunked Prefill

把长 prompt 的 prefill 分块，与 decode 交错执行，降低长请求对短请求的阻塞。

### 其他

- CUDA Graph：减少 kernel launch 开销；
- 算子融合；
- 量化；
- 请求调度（优先级、超时、限流）；
- 自动扩缩容。

---

## 8.6 端侧 LLM

### 约束

| 约束 | 影响 |
|---|---|
| 内存 | 权重 + KV Cache 必须装下 |
| 带宽 | decode 速度受内存带宽限制 |
| 算力 | prefill 受 NPU/CPU 算力限制 |
| 功耗/散热 | 持续推理会降频 |
| 算子支持 | 动态 shape、注意力、RoPE 可能不支持 |
| 存储 | 模型文件大小、加载时间 |
| 生态 | 量化/转换工具链成熟度 |

### 端侧速度估算

解码阶段每生成一个 token，需要读取一遍模型权重（近似）：

```
TPS ≤ 内存带宽 / 每 token 读取字节数
```

例如 3B INT4 模型约 1.5 GB，若有效带宽 25 GB/s：

```
理论上限 ≈ 25 / 1.5 ≈ 16.7 tokens/s
```

实际会因算子效率、KV Cache、调度开销更低。这解释了为什么端侧 LLM 的瓶颈通常是**内存带宽**，而不是标称 TOPS。

### 模型选择

| 场景 | 建议模型规模 |
|---|---|
| 手机/低端 SoC | 0.5B–1.5B（INT4） |
| RK3588/RK3576 | 1.5B–3B（W8A8/W4A16） |
| Jetson Orin | 3B–8B（INT4/FP16） |
| Jetson Thor / 高端 GPU | 8B–70B+ |
| DeepX DX-M2（目标） | 最高约 20B |

### 端侧工具链

| 平台 | 方案 | 说明 |
|---|---|---|
| Rockchip | RKLLM-Toolkit + RKLLM Runtime | HuggingFace/GGUF → .rkllm，NPU 推理 |
| Rockchip 新方案 | RK182X + RKNN3 | 协处理器，支持 0.5B–8B |
| Qualcomm | QNN + Genie | 骁龙 HTP/NPU |
| Intel | OpenVINO GenAI | CPU/iGPU/NPU |
| NVIDIA | TensorRT-LLM / llama.cpp | Jetson |
| 通用 ARM | llama.cpp / MLC-LLM | CPU/GPU，GGUF |
| Android | MediaPipe LLM / ExecuTorch | 手机端 |
| NXP | eIQ + 社区方案 | 以视觉/小模型为主 |
| Realtek | 以视觉/音频小模型为主 | 不适合跑 LLM |
| DeepX | DXNN SDK | DX-M 系列加速卡 |

### RKLLM 流程（概念）

```bash
# PC 端：转换 + 量化
rkllm_toolkit convert \
  --model Qwen2.5-1.5B-Instruct \
  --target rk3588 \
  --quant w8a8 \
  --output qwen.rkllm

# 板端：加载 + 推理
./rkllm_demo qwen.rkllm
```

实际参数以 RKLLM 版本为准。注意：

- 模型结构要在支持列表内；
- 量化类型影响精度/速度；
- 上下文长度受内存限制；
- 多核 NPU 可并行，但要注意调度；
- 首 token 延迟与 prompt 长度强相关。

### 端侧 LLM 的体验设计

- 流式输出，降低感知延迟；
- 限制上下文，定期摘要；
- 短 prompt、少样本；
- 本地小模型 + 云端大模型混合；
- 离线兜底（规则/模板）；
- 明确告知用户能力边界。

---

## 8.7 部署流程

```
1. 明确需求：模型能力、上下文、延迟、并发、成本、隐私
2. 选模型：尺寸、许可证、语言、量化友好度
3. 选硬件：显存/带宽/算力/功耗
4. 选引擎：vLLM/TensorRT-LLM/llama.cpp/厂商 SDK
5. 量化：精度与速度权衡
6. 转换：ONNX/GGUF/厂商格式
7. 压测：TTFT、TPOT、吞吐、并发、长上下文
8. 评测：业务指标、幻觉、拒答、安全
9. 集成：API、流式、重试、降级、监控
10. 迭代：根据线上数据优化 prompt/检索/模型
```

---

## 8.8 容量规划

### GPU 显存

```
显存 = 权重 + KV Cache × 并发 + 激活/workspace + 框架开销
```

示例：8B INT4（4 GB）+ 8K 上下文 KV（约 1 GB/请求）+ 2 GB 开销：

- 并发 1：约 7 GB；
- 并发 4：约 10 GB；
- 并发 8：约 14 GB。

实际还要留余量，考虑碎片和峰值。

### 吞吐与成本

```
成本 / 1M tokens = 硬件小时成本 / (每小时吞吐 tokens / 1M)
```

自部署不一定比 API 便宜：小流量时 API 更划算；大流量、稳定负载、数据合规要求高时自部署更有优势。

### 云 vs 端 vs 混合

| 方案 | 优点 | 缺点 | 适合 |
|---|---|---|---|
| 云 API | 无运维、模型强、弹性 | 成本随量增长、数据出域 | 快速验证、低流量 |
| 自部署云 | 可控、可定制、数据可控 | 运维、GPU 成本 | 中高流量、合规 |
| 端侧 | 隐私、离线、低延迟 | 模型小、算力有限 | 手机/设备/隐私场景 |
| 混合 | 兼顾能力与隐私 | 架构复杂 | 端侧小模型 + 云端大模型 |

---

## 8.9 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| OOM | 权重 + KV + 并发超显存 | 量化、减并发、缩上下文、张量并行 |
| 首 token 很慢 | prompt 太长、prefill 算力不足 | 前缀缓存、缩短 prompt、chunked prefill |
| 生成很慢 | 带宽不足、量化不够、batch 太大 | 量化、换引擎、调 batch |
| 吞吐低 | 没有连续批处理、并发不足 | vLLM/SGLang、增并发 |
| 输出乱码/重复 | 量化损失、采样参数、模板错 | 换量化、调 temperature、检查 chat template |
| 长上下文效果差 | 训练长度不足、KV 量化 | 缩短上下文、换模型、RAG |
| 工具调用失败 | chat template/特殊 token 错 | 用官方模板、检查 tokenizer |
| 精度下降 | 4bit 量化、校准差 | 8bit/FP8、AWQ、QAT、混合精度 |

---

## 8.10 自检问题

1. TTFT 和 TPOT 分别由什么决定？
2. 8B 模型 FP16/INT8/INT4 的权重显存各是多少？
3. KV Cache 怎么估算？为什么 GQA 能省显存？
4. 为什么 decode 阶段是 memory-bound？
5. PagedAttention 解决什么问题？
6. 前缀缓存对 RAG 为什么收益大？
7. 端侧 LLM 的速度上限由什么决定？
8. GPTQ、AWQ、GGUF 分别适合什么场景？
9. 如何为一个 8B 模型做容量规划？
10. 什么情况下不应该自部署模型？

---

## 8.11 推荐练习

1. 用 llama.cpp 在本地跑一个 4B GGUF 模型，测不同量化档的 tokens/s；
2. 用 vLLM 部署一个 7B 模型，测不同并发下的 TTFT/吞吐；
3. 手算一个模型的权重、KV Cache、总显存；
4. 对比 FP16 与 INT4 在数学题上的准确率；
5. 在 RK3588 或 Jetson 上部署一个 1.5B–3B 模型，记录首 token 延迟；
6. 实现一个流式输出 + 上下文摘要的对话 demo。
