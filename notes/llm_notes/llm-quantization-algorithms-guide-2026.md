**TECHNICAL REFERENCE · 2026**

# 主流 LLM 量化算法技术指南

*从 INT8 / INT4 到 FP8 / FP4、KV Cache 与原生低比特训练*

**算法原理 · 工程实现 · 框架生态 · 选型方法**

**适用读者**

模型工程师、推理平台工程师、算法研究人员与技术决策者

版本基线：2026 年 7 月

*说明：框架与硬件支持变化很快，生产部署前应以目标版本实测为准。*

## 执行摘要

> **一句话结论**　截至 2026 年，4-bit weight-only 仍是通用本地/单卡推理的主力；FP8 已成为新一代数据中心 GPU 的成熟路线；FP4/MXFP4/NVFP4 正随硬件原生支持进入生产；2–3 bit 仍应视为高压缩、强验证的专项方案。

- GPTQ 与 AWQ：成熟的校准式 W4A16 后训练量化。GPTQ偏重二阶误差补偿；AWQ偏重激活感知的显著通道保护。

- SmoothQuant：经典 W8A8 路线，通过等价缩放处理激活异常值，适合拥有高效 INT8 内核的服务器推理。

- bitsandbytes NF4：低显存加载和 QLoRA 微调的事实标准之一，但“方便加载”不等同于目标平台上的最高推理吞吐。

- GGUF K-quants/IQ：面向 llama.cpp 生态的格式与量化类型集合，特别适合 CPU、Apple Silicon 和 CPU/GPU 混合部署；GGUF不是单一算法。

- FP8 / FP4：是数值格式与硬件执行路线，不是单一 PTQ 算法。其收益高度依赖 Hopper、Blackwell 或其他支持相应低精度计算的硬件与内核。

- KV Cache 量化：长上下文和大批量服务中不可忽略；仅压权重可能无法解决运行时显存瓶颈。

### 阅读导航

| 章节 | 主题 | 解决的问题 |
| --- | --- | --- |
| 01 | 基础概念与统一术语 | 理解 W/A/KV、PTQ/QAT、粒度与映射 |
| 02 | 主流部署型算法 | GPTQ、AWQ、SmoothQuant、LLM.int8、HQQ |
| 03 | 微调与本地生态 | NF4/QLoRA、GGUF K/IQ quants |
| 04 | 硬件原生与前沿路线 | FP8、FP4、MXFP4/NVFP4、旋转量化、2–3 bit |
| 05 | 工程选型与评测 | 场景决策、框架匹配、指标与验收清单 |

## 1. 基础概念与统一术语

### 1.1 量化到底改变了什么

量化把连续的高精度数值映射到有限的离散表示。对线性量化，常见形式是 q = clamp(round(x / s) + z)，反量化近似为 x̂ = s(q − z)。其中 s 为缩放系数，z 为零点。低比特并不会自动改变模型结构，但会改变参数存储、内存搬运和矩阵乘的执行方式。

| 记法 | 含义 | 典型价值 | 常见例子 |
| --- | --- | --- | --- |
| W4A16 | 权重4-bit，激活16-bit | 显著降低权重容量与解码带宽 | GPTQ、AWQ |
| W8A8 | 权重和激活均8-bit | 利用INT8矩阵计算，提升吞吐 | SmoothQuant |
| W4A8 | 权重4-bit，激活8-bit | 兼顾prefill计算与decode带宽 | QQQ、部分AWQ/GPTQ配方 |
| FP8 W8A8 | 权重/激活使用FP8 | 较宽动态范围，适配新GPU | E4M3/E5M2配方 |
| KV8 / KV4 | KV Cache降至8/4-bit | 降低长上下文运行时显存 | FP8 KV、KIVI类方法 |

*表 1　W/A/KV 表示法。A16 通常指 FP16 或 BF16；具体实现需查看框架定义。*

### 1.2 三条正交轴

- 量化对象：仅权重、权重+激活、KV Cache，或者训练中的权重/激活/梯度/优化器状态。

- 量化时机：训练后量化（PTQ）、量化感知训练（QAT）、量化基础模型上的参数高效微调（如QLoRA），以及从头原生低比特训练。

- 执行机制：低比特存储后反量化计算，或直接进入INT8/FP8/FP4张量核。前者主要省容量与带宽，后者才更直接增加计算吞吐。

### 1.3 粒度与映射

| 维度 | 选项 | 影响 |
| --- | --- | --- |
| 粒度 | per-tensor / per-channel / per-group / per-token | 越细通常误差越低，但scale元数据与内核复杂度越高 |
| 映射 | 对称 / 非对称 | 非对称增加zero-point，适合偏移分布；对称计算更简单 |
| 参数估计 | min-max / clipping / percentile / MSE / Hessian | 决定异常值与主体分辨率之间的权衡 |
| 静态性 | 静态校准 / 动态量化 | 动态量化适应输入，但每次推理要计算scale |
| 编码 | 整数、普通浮点、NormalFloat、codebook | 决定可表示范围、分辨率与硬件支持 |

> **关键提醒**　“4-bit 模型”信息不足以支持选型。至少还要知道量化对象、group size、scale/zero-point精度、是否混合精度、运行内核和目标硬件。

## 2. 算法版图与成熟度

下面将算法与格式按工程成熟度分层。这里的“主流”同时考虑论文影响、开源模型供给、主流框架集成和可用内核，而不是只按榜单精度排序。

| 方法 | 定位 | 典型精度 | 校准/训练 | 工程状态 |
| --- | --- | --- | --- | --- |
| RTN / MinMax | 基础PTQ | W8/4A16 | 无或少量 | 成熟基础 |
| LLM.int8() | 混合精度PTQ | INT8+FP16 | 通常无需离线校准 | 成熟 |
| SmoothQuant | 激活平滑PTQ | W8A8 | 需要 | 成熟生产 |
| GPTQ | 二阶补偿PTQ | W4/3A16 | 需要 | 成熟生产 |
| AWQ | 激活感知PTQ | W4A16 / W4A8 | 需要 | 成熟生产 |
| HQQ | 优化式快速PTQ | W8/4/3/2A16 | 不要求校准集 | 较成熟 |
| NF4 + QLoRA | 量化微调 | 4-bit基础权重 | 训练数据 | 成熟微调 |
| GGUF K/IQ | 格式+分块量化族 | 约2–8 bit | 可选重要性矩阵 | 成熟本地 |
| FP8 | 硬件低精度 | W8A8 / KV8 | 静态或动态 | 成熟度上升 |
| FP4/MXFP4/NVFP4 | 微缩放浮点 | W4A4 / W4A16 | 配方相关 | 新硬件生产 |
| OmniQuant/QuaRot | 高级PTQ | W4A4至W2A16 | 需要/可选 | 研究到工程 |
| AQLM/VPTQ/QuIP# | 码本/极低比特 | 2–3 bit | 需要且耗时 | 专项部署 |
| KIVI类 | KV Cache量化 | KV2/4 | 无需训练 | 长上下文专项 |
| BitNet b1.58 | 原生低比特训练 | 三值权重 | 需从头训练 | 前沿/专用 |

*表 2　主流量化路线总览。成熟度为截至 2026-07 的综合判断，不代表所有模型与硬件均已覆盖。*

### 2.1 为什么 W4A16 长期流行

自回归解码通常受权重读取和显存带宽限制。W4A16把权重压缩到约四分之一，可减少每生成一个token时的权重搬运；激活仍保持FP16/BF16，避开最棘手的激活异常值。其代价是prefill阶段未必像真正的W8A8或FP8那样获得完整低精度矩阵计算收益。

### 2.2 为什么“算法排名”不能脱离内核

- 同一GPTQ/AWQ权重可能由不同打包布局和内核运行，例如Marlin、ExLlama类内核或框架自有kernel。

- 如果运行时先把INT4解码到BF16再做普通GEMM，显存下降不等于吞吐成比例上升。

- batch size、输入/输出长度与并发会改变瓶颈：prefill更偏计算，decode更偏内存带宽，KV Cache则随上下文和batch增长。

## 3. 主流部署型 PTQ 算法

### 3.1 GPTQ：二阶信息与逐步误差补偿

GPTQ沿袭二阶量化思想，使用校准样本近似层输入的Hessian信息，逐列或分块量化权重，并把已产生的误差补偿到尚未量化的权重中。它关心的是量化前后层输出误差，而不是只最小化权重本身的欧氏距离。[1]

- 优势：4-bit精度通常显著优于朴素RTN；预量化模型丰富；GPU生态成熟。

- 代价：需要代表性校准数据；量化耗时和内存高于直接舍入；打包格式与kernel兼容性需要核对。

- 适合：NVIDIA GPU上的单模型推理、已有高质量GPTQ权重、需要稳定W4A16质量时。

### 3.2 AWQ：保护激活感知的显著通道

AWQ认为权重重要性应由真实激活响应判断。它寻找少量显著通道，通过数学等价的缩放降低这些通道的量化误差，避免硬件不友好的大范围混合精度，并且不依赖反向传播或逐层重构。[2]

- 优势：校准成本通常低于复杂重构法；对指令模型和多模态模型有良好适配记录；W4A16生态成熟。

- 代价：仍依赖校准集；算法名称相同不保证scale搜索、group size和kernel布局完全一致。

- 适合：端侧或GPU部署、强调4-bit质量与量化效率、推理框架明确支持AWQ时。

### 3.3 GPTQ 与 AWQ 的工程差异

| 比较项 | GPTQ | AWQ |
| --- | --- | --- |
| 误差控制 | Hessian近似与序列误差补偿 | 激活统计识别显著通道并缩放保护 |
| 校准数据 | 需要，且影响Hessian估计 | 需要，用于激活统计与scale搜索 |
| 量化成本 | 通常较高 | 通常较低 |
| 典型目标 | 高质量W4A16，亦可更低比特 | 硬件友好的W4A16/W4A8 |
| 最终速度 | 取决于打包与kernel | 同样取决于kernel；不能仅凭算法名判断 |

> **选型原则**　同一硬件、同一模型、同一任务上，同时测质量、峰值显存、prefill吞吐、decode吞吐和并发吞吐。不要用单一困惑度代替应用验收。

## 4. 激活量化与快速无校准路线

### 4.1 SmoothQuant：把激活难题迁移到权重

SmoothQuant针对LLM激活中的持久异常通道，引入等价的逐通道缩放：激活除以scale，同时权重乘以scale。这样可让激活更容易量化，而权重吸收的幅度通常更可控，最终实现训练无关的W8A8 PTQ。[3]

- 优势：权重和激活均为INT8，适合有高效INT8 Tensor Core/kernel的高吞吐服务。

- 代价：需要校准并选择平滑强度；W8A8在权重容量上不如W4A16；不同模型的异常值特征不同。

- 更适合prefill或较大batch；对极低batch的decode，需与W4A16实测。

### 4.2 LLM.int8()：异常维度走高精度

LLM.int8()把大多数特征用向量级INT8矩阵乘处理，同时将少量系统性异常维度单独用FP16计算，再合并结果。论文报告超过99.9%的值进入8-bit路径，用混合精度避免异常值破坏整体分辨率。[4]

> **定位**　LLM.int8()的主要价值是稳健地把16-bit模型降到约8-bit权重级别；如果目标是极致压缩，它通常不是最终答案。

### 4.3 HQQ 与普通 RTN：追求快速、在线量化

HQQ使用半二次优化思想估计量化参数，可在不依赖校准数据的情况下进行快速低比特权重量化。当前Transformers选型指南把HQQ归入无需校准、可在线量化的常用方案；PyTorch原生torchao也提供HQQ参数选择路径。[12][13]

| 方案 | 校准集 | 转换速度 | 典型质量 | 推荐用途 |
| --- | --- | --- | --- | --- |
| 朴素RTN | 不需要 | 最快 | 8-bit稳；4-bit依模型而定 | 基线、快速试验 |
| HQQ | 不需要 | 快 | 通常优于直接舍入 | 在线转换、灵活部署 |
| GPTQ/AWQ | 需要 | 中到慢 | 4-bit通常更稳 | 追求质量、可离线准备 |

## 5. 微调与本地推理生态

### 5.1 NF4 + QLoRA：低显存微调

NF4是一种针对近似正态分布权重设计的4-bit非均匀数据类型。QLoRA冻结NF4量化的基础权重，让梯度穿过量化权重，只训练低秩LoRA适配器；同时引入double quantization以进一步压缩量化常数，并用paged optimizer控制显存峰值。[5]

- 适合：消费级GPU或有限显存上的LoRA微调、原型验证与多适配器训练。

- 不要混淆：NF4是表示形式；QLoRA是微调流程；bitsandbytes是常见实现。

- 推理注意：QLoRA训练完成后，可继续以量化基础模型+adapter运行，也可合并后重新按目标推理格式量化。后者需重新评测。

### 5.2 QAT：训练中模拟量化误差

QAT在训练或恢复训练期间插入fake quantization，使模型适应目标量化网格。它通常能修复PTQ在小模型、4-bit激活、极低比特或任务敏感模型上的质量损失，但需要额外训练预算。torchao现已提供INT4 weight-only及INT8动态激活+INT4权重等QAT流程。[14]

### 5.3 GGUF 与 K-quants / I-quants

GGUF是ggml/llama.cpp生态的模型容器格式，负责存放张量和元数据；Q4_K_M、Q5_K_M、Q8_0等才是量化类型。K-quants通常按块处理，并可对不同张量采用不同精度；importance matrix（imatrix）可利用样本统计改善重要权重的量化分配。[6]

| 常见类型 | 大致定位 | 质量/容量倾向 | 典型选择 |
| --- | --- | --- | --- |
| Q8_0 | 8-bit分块 | 高质量、容量约为16-bit一半 | 质量优先或再量化源 |
| Q6_K | 6-bit K-quant | 接近高精度、容量中等 | 代码/推理敏感任务 |
| Q5_K_M | 混合5-bit K-quant | 稳健折中 | 本地高质量 |
| Q4_K_M | 混合4-bit K-quant | 容量与质量均衡 | 通用默认起点 |
| Q3/Q2与IQ族 | 低于4-bit | 极小但风险明显上升 | 内存极限、需专项评测 |

> **实践建议**　本地部署通常从Q4_K_M开始；若代码、数学、工具调用或长上下文质量下降，先升到Q5_K_M/Q6_K，而不是立即换更复杂的采样参数。

## 6. FP8、FP4 与硬件原生低精度

### 6.1 FP8：数据中心推理的主流新路线

FP8保留浮点指数，因此动态范围通常优于同位宽整数。常见E4M3偏重精度，E5M2偏重范围；实际配方会采用per-tensor、per-row或block scaling，并可能对权重、激活和KV Cache分别选择策略。TensorRT-LLM和torchao均提供多种FP8推理路径。[7][13]

- 优势：在原生支持FP8的GPU上，可同时降低带宽与增加矩阵计算吞吐。

- 局限：旧硬件可能只能把FP8当存储格式；scale粒度、累加精度与异常值策略仍决定质量。

- 典型场景：Hopper/Blackwell级GPU上的高吞吐在线服务、FP8预量化模型、FP8 KV Cache。

### 6.2 FP4、MXFP4 与 NVFP4

4-bit浮点格式通过很少的指数与尾数位表达数值。MXFP4采用microscaling思想，让一个小块共享scale；NVFP4也属于细粒度缩放的4-bit路线。它们的精度并不只由“4 bit”决定，还依赖块大小、二级scale、累加类型、随机舍入及校准/微调配方。OCP MX规范定义了MXFP8、MXFP6、MXFP4与MXINT8等格式。[8]

| 路线 | 数值特点 | 主要收益 | 硬件依赖 | 当前定位 |
| --- | --- | --- | --- | --- |
| INT8 | 统一整数网格 | 成熟、工具链广 | 广泛支持 | 稳健生产 |
| FP8 | 指数+尾数 | 动态范围与吞吐平衡 | 新一代加速器 | 生产主流 |
| INT4 W-only | 分组整数权重 | 容量与decode带宽 | 需高效解包kernel | 通用主流 |
| MXFP4/NVFP4 | 4-bit值+块级scale | 更高计算密度 | Blackwell等新硬件 | 快速进入生产 |

> **不要把格式当算法**　FP8、MXFP4和NVFP4定义“如何表示与计算”；校准、scale选择、异常值处理、QAT及kernel实现才构成完整量化配方。

## 7. 高级 PTQ 与 2–3 bit 极低比特方法

### 7.1 OmniQuant：可学习的裁剪与等价变换

OmniQuant在PTQ预算内，以少量校准样本优化可学习权重裁剪（LWC）和可学习等价变换（LET），覆盖W4A4、W4A16、W3A16甚至W2A16等配置。它比简单校准更重，但远低于完整训练成本。[9]

### 7.2 QuaRot / 旋转量化：先消除异常值再量化

QuaRot利用保持模型函数等价的Hadamard类旋转，让隐藏状态和激活中的异常值分散，从而把权重、激活和KV Cache端到端量化到4-bit。旋转类方法展示了W4A4潜力，但真正收益取决于旋转开销是否能被融合进kernel。[10]

### 7.3 AQLM、VPTQ、QuIP#：码本与极限压缩

AQLM使用多个学习码本的加性组合近似权重，并跨Transformer block联合优化，目标是2–3 bit区间的精度—容量前沿。这类方法能在极限内存约束下保留更大参数规模，但量化耗时、模型转换复杂度和kernel覆盖通常高于主流INT4。[11]

| 方法族 | 主要技巧 | 目标精度 | 优势 | 主要风险 |
| --- | --- | --- | --- | --- |
| OmniQuant | 学习裁剪+等价变换 | W4A4～W2A16 | 低比特精度强 | 校准优化成本高 |
| QuaRot/Spin类 | 正交旋转分散异常值 | 端到端4-bit | 激活/KV也可低比特 | 需要融合kernel |
| AQLM/VPTQ | 多码本/向量量化 | 2–3 bit | 容量极低 | 转换慢、生态较窄 |
| QuIP#类 | 非相干变换+码本 | 2 bit附近 | 极限压缩精度改善 | 实现复杂 |

> **经验法则**　固定内存预算下，“更小模型的4-bit”和“更大模型的2-bit”谁更好没有普遍答案。2-bit尤其容易损伤代码、数学、事实记忆与指令遵循，必须按业务任务比较。

## 8. KV Cache 量化与原生低比特模型

### 8.1 KV Cache 为什么会成为新瓶颈

权重内存与请求数基本无关，而KV Cache大致随层数、batch、上下文长度和KV头维度线性增长。在长上下文或连续批处理服务中，即使权重已经INT4，KV Cache仍可能占据主要显存，并限制可并发请求数。

### 8.2 KIVI 与运行时 KV 量化

KIVI根据统计观察对Key采用per-channel量化、对Value采用per-token量化，并保留近期token的高精度残差窗口，形成无需调优的2-bit KV Cache方案。论文报告峰值内存和吞吐收益，但生产环境仍需关注长文本召回、位置敏感任务与特定注意力结构。[15]

- KV8/FP8：通常更稳，适合先获得低风险的容量收益。

- KV4：收益更明显，但长上下文质量、注意力峰值和模型结构更敏感。

- KV2：属于激进配置，应对needle-in-a-haystack、多轮一致性和长文档问答做专项验收。

### 8.3 BitNet b1.58：量化模型还是原生模型

BitNet b1.58把权重约束为三值 {−1, 0, 1}，理论信息量约为log₂3≈1.58 bit。它强调从头训练时让网络适应极低比特，不是把任意现成FP16模型直接后量化成1.58-bit。公开2B4T模型证明了原生低比特路线的可行性，但其模型供给与专用kernel生态仍不同于通用PTQ。[16][17]

> **边界**　原生低比特训练可能从根本上改变未来模型设计，但在现阶段，它不能替代对现有开源权重进行GPTQ/AWQ/GGUF量化的通用工作流。

## 9. 框架与格式匹配

量化选型应从最终运行时反推。先确认框架和硬件能高效执行目标格式，再选择量化算法；否则可能得到一个更小但更慢的模型。

| 生态/框架 | 常见量化路线 | 优势场景 | 注意事项 |
| --- | --- | --- | --- |
| Transformers | bitsandbytes、GPTQ、AWQ、HQQ、Quanto、torchao等 | 研究、微调、统一加载 | 方便不等于最高吞吐；核对HfQuantizer和后端 |
| vLLM | AWQ、GPTQ/Marlin、FP8、GGUF、AQLM、compressed-tensors等 | GPU高吞吐服务 | 支持矩阵依GPU代际和版本变化 |
| TensorRT-LLM | FP8、FP4、W4A8/W4A16 AWQ/GPTQ、KV量化 | NVIDIA生产服务 | 与Model Optimizer及GPU代际强相关 |
| llama.cpp | GGUF Q/I/K-quants、CPU/GPU混合 | 本地CPU、Mac、边缘设备 | 量化类型和offload配置共同决定速度 |
| torchao | INT4/INT8/FP8、MX/NVFP4、QAT、torch.compile | PyTorch原生优化 | 部分低精度配置仍为prototype |
| ExecuTorch | 8da4w、int8、低比特embedding等 | 移动端/嵌入式 | 后端与设备算子覆盖决定效果 |

*表 3　框架生态概览。框架版本和硬件支持变化快，应以目标版本官方兼容矩阵为准。[12][13][18][19]*

### 9.1 文件名不能替代量化配置

- 记录原始模型版本、量化算法版本、校准数据来源和样本数。

- 记录bit width、group size、对称/非对称、scale与zero-point类型、跳过量化的层。

- 记录打包格式、运行kernel、目标GPU/CPU、框架及驱动版本。

- 对于KV Cache，记录KV精度、scale粒度、残差窗口和最大上下文。

## 10. 场景化选型建议

### 10.1 快速决策表

| 场景 | 优先候选 | 备选 | 决策重点 |
| --- | --- | --- | --- |
| Mac / CPU本地聊天 | GGUF Q4_K_M | Q5_K_M / Q6_K | 内存、首token延迟、质量 |
| 消费级NVIDIA单卡推理 | AWQ或GPTQ W4A16 | bitsandbytes 4-bit / HQQ | kernel支持、VRAM、decode速度 |
| 低显存LoRA微调 | NF4 + QLoRA | INT8 LoRA | 训练显存、adapter质量 |
| A100等成熟INT8服务器 | SmoothQuant W8A8 | AWQ/GPTQ W4A16 | batch、prefill/decode比例 |
| Hopper数据中心推理 | FP8 | INT8或INT4 W-only | 动态scale、吞吐、KV精度 |
| Blackwell级硬件 | FP4/NVFP4或FP8 | W4A8 AWQ/GPTQ | Model Optimizer与kernel成熟度 |
| 超长上下文服务 | 权重量化+KV8/FP8 | 验证后KV4/KIVI类 | 长文召回、并发、KV容量 |
| 极端内存限制 | 高质量3-bit或AQLM/VPTQ | 2-bit专项方案 | 任务质量优先于名义bit |

### 10.2 选择流程

1. 确定目标：省模型容量、降低峰值显存、提高prefill吞吐、提高decode吞吐，还是增加并发？

1. 锁定硬件与运行时：只有目标kernel实际支持的格式才进入候选集。

1. 从风险最低的精度开始：FP8/INT8 → 4-bit → 3-bit → 2-bit，逐级测量收益。

1. 为模型准备代表性校准集，覆盖语言、上下文长度、系统提示词、工具调用与业务输入分布。

1. 统一基准：相同模型、prompt、采样参数、batch、输入/输出长度和并发。

1. 设置质量与性能双阈值；任一不达标都不进入生产。

> **推荐默认**　若没有更具体约束：本地CPU/Mac从GGUF Q4_K_M开始；NVIDIA单卡从AWQ/GPTQ W4A16开始；数据中心新GPU优先评估FP8；微调从NF4 + QLoRA开始。

## 11. 评测与验收方法

### 11.1 质量指标

- 基础语言建模：困惑度（PPL），只能作为回归信号，不能代表全部任务质量。

- 任务能力：代码、数学、知识问答、指令遵循、结构化输出、工具调用和多语言。

- 长上下文：检索召回、跨段推理、位置偏差、多轮一致性；KV量化必须重点测试。

- 鲁棒性：不同prompt模板、温度、序列长度和输入分布；校准集之外的数据尤其重要。

- 生成退化：重复、乱码、格式错误、EOS异常、拒答率变化和置信度漂移。

### 11.2 系统指标

| 指标 | 应如何测 | 常见误区 |
| --- | --- | --- |
| 模型磁盘大小 | 含scale/zero-point/元数据后的实际文件 | 只用参数量×bit估算 |
| 峰值显存 | 加载、prefill、decode、并发分别记录 | 只看模型加载后静态值 |
| TTFT | 固定输入长度与并发 | 与输出吞吐混在一起 |
| Prefill吞吐 | tokens/s，覆盖短/长输入 | 只测batch=1短prompt |
| Decode吞吐 | 单请求与并发均测 | 忽略KV Cache与调度 |
| 端到端成本 | 功耗、GPU时、并发与SLA综合 | 只比较峰值算力 |

### 11.3 最小验收矩阵

- 至少一个通用基准 + 两个核心业务数据集 + 一个长尾/安全数据集。

- 至少三种输入长度、两种输出长度、单请求与生产目标并发。

- 与BF16/FP16基线逐样本对比；对关键任务设置硬性不回退指标。

- 保存量化产物、配置、校准集哈希、评测脚本版本和运行日志，确保可复现。

> **停止条件**　如果低比特方案只减少文件大小，却没有降低峰值显存或提高目标负载吞吐，应检查反量化路径、kernel回退、CPU/GPU搬运与不支持的层，而不是继续降低bit数。

## 12. 常见误区与趋势判断

### 12.1 常见误区

| 误区 | 正确理解 |
| --- | --- |
| INT4一定比FP16快 | 不一定。没有低比特kernel时，解包与反量化可能抵消收益。 |
| 同为4-bit就可直接比较 | NF4、INT4、FP4、Q4_K_M的网格、粒度与执行方式不同。 |
| 量化只影响PPL | 代码、数学、工具调用、长上下文和结构化输出可能更敏感。 |
| 校准数据随便选 | 语言、长度和任务分布不匹配会降低GPTQ/AWQ/SmoothQuant效果。 |
| 权重压缩解决全部显存 | 长上下文和并发场景中，KV Cache可能成为主导。 |
| 重新量化无损 | 从已量化权重再次量化会叠加误差，应尽量从FP16/BF16源模型转换。 |

### 12.2 2026 年的趋势判断

- 4-bit weight-only继续作为通用部署基线，但W4A8和真正的端到端4-bit会随kernel成熟扩大使用。

- FP8从“新硬件特性”转向标准生产精度；FP4/MXFP4/NVFP4在Blackwell级硬件与相应工具链上加速落地。

- 量化从只处理权重扩展到全栈：激活、KV Cache、MoE专家、训练状态与通信。

- 算法与kernel联合设计越来越重要；可部署的近似最优，通常胜过论文精度更高但缺少高效kernel的方案。

- QAT与低比特恢复训练会用于修复PTQ难以承受的W4A4、2–3 bit和小模型退化。

- 原生低比特训练（如BitNet）是结构性方向，但短期仍与现成权重PTQ生态并行。

### 12.3 最终建议

> **工程结论**　量化是“模型 × 数据 × 格式 × kernel × 硬件 × 工作负载”的联合优化问题。先确定目标瓶颈和运行时，再选算法；先用4/8-bit成熟路线建立基线，再探索更激进精度。

## 附录 A：术语速查

| 术语 | 定义 |
| --- | --- |
| PTQ | Post-Training Quantization，训练完成后量化 |
| QAT | Quantization-Aware Training，训练中插入伪量化以适应误差 |
| RTN | Round-to-Nearest，舍入到最近量化点 |
| Scale | 浮点范围与量化整数/低精度值之间的比例参数 |
| Zero-point | 非对称量化中表示实数零的整数偏移 |
| Group size | 共享一组量化参数的连续权重数量 |
| Weight-only | 只量化权重，激活保持较高精度 |
| Static quant | 通过校准预先确定激活量化参数 |
| Dynamic quant | 根据运行时输入动态计算激活量化参数 |
| Outlier | 幅度显著高于主体分布、容易主导scale的值或通道 |
| KV Cache | 自回归注意力中缓存历史token的Key和Value |
| Prefill | 处理输入prompt并构建初始KV Cache的阶段 |
| Decode | 逐token生成输出的阶段 |
| GGUF | ggml/llama.cpp生态的模型张量与元数据容器格式 |
| NF4 | 针对近似正态权重设计的4-bit NormalFloat表示 |
| MX | Microscaling，多值共享细粒度scale的低精度格式族 |

## 参考资料

以下优先列出原始论文、标准组织和官方框架文档。访问日期：2026-07-28。框架API与支持矩阵会持续更新，生产决策应再次核对目标版本。

**[1]** GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers — arXiv. [链接](https://arxiv.org/abs/2210.17323)。GPTQ原始论文。

**[2]** AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration — arXiv. [链接](https://arxiv.org/abs/2306.00978)。AWQ原始论文。

**[3]** SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models — arXiv. [链接](https://arxiv.org/abs/2211.10438)。SmoothQuant与W8A8。

**[4]** LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale — arXiv. [链接](https://arxiv.org/abs/2208.07339)。异常维度混合精度。

**[5]** QLoRA: Efficient Finetuning of Quantized LLMs — arXiv. [链接](https://arxiv.org/abs/2305.14314)。NF4、double quantization和paged optimizer。

**[6]** llama.cpp Quantization README — ggml-org / GitHub. [链接](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)。GGUF量化工具、K-quants及importance matrix说明。

**[7]** TensorRT-LLM Quantization — NVIDIA. [链接](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)。FP8、FP4、AWQ/GPTQ与KV量化支持。

**[8]** OCP Microscaling Formats (MX) Specification — Open Compute Project. [链接](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。MXFP8/MXFP6/MXFP4/MXINT8格式定义。

**[9]** OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models — arXiv. [链接](https://arxiv.org/abs/2308.13137)。LWC与LET。

**[10]** QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs — arXiv. [链接](https://arxiv.org/abs/2404.00456)。旋转消除异常值与端到端4-bit。

**[11]** Extreme Compression of Large Language Models via Additive Quantization — arXiv. [链接](https://arxiv.org/abs/2401.06118)。AQLM及2–3 bit码本量化。

**[12]** Selecting a Quantization Method — Hugging Face Transformers. [链接](https://huggingface.co/docs/transformers/main/quantization/selecting)。主流推理与微调量化方案对比。

**[13]** Quantized Inference — PyTorch torchao. [链接](https://docs.pytorch.org/ao/stable/workflows/inference.html)。INT4/INT8/FP8/MX/NVFP4推理配置。

**[14]** Quantization-Aware Training — PyTorch torchao. [链接](https://docs.pytorch.org/ao/stable/workflows/qat.html)。INT4与8da4w QAT流程。

**[15]** KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache — arXiv. [链接](https://arxiv.org/abs/2402.02750)。Key per-channel、Value per-token的2-bit KV方案。

**[16]** The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits — arXiv. [链接](https://arxiv.org/abs/2402.17764)。BitNet b1.58原始工作。

**[17]** BitNet b1.58 2B4T Technical Report — arXiv. [链接](https://arxiv.org/abs/2504.12285)。公开原生1.58-bit模型技术报告。

**[18]** Quantization — vLLM Documentation. [链接](https://docs.vllm.ai/en/latest/features/quantization/)。vLLM量化实现与硬件支持矩阵。

**[19]** Exporting LLMs — PyTorch ExecuTorch. [链接](https://docs.pytorch.org/executorch/stable/llm/export-llm.html)。移动端8da4w、int8和低比特导出。
