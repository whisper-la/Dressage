**TECHNICAL REFERENCE · 2026**

# 从 Safe Softmax 到 Online Softmax 再到 FlashAttention：精确注意力的 IO 优化演进详解

*Safe Softmax → Online Softmax → Memory-Efficient Attention → FlashAttention-1/2/3*

**面向教学的逐步推导 · 公式细节 · 硬件视角 · 演进逻辑**

**适用读者**

希望理解"为什么注意力不能按教科书方式在 GPU 上实现"、以及 FlashAttention 系列如何绕开内存墙的研究人员与工程师

版本基线：2026 年 8 月

姊妹篇：[从 MHA 到 MQA、GQA 再到 MLA](./mha-to-mqa-gqa-to-mla.md) 讲如何压缩 KV Cache 的**表示**；[从 Full Attention 到 Linear Attention 到 GDN 再到 KDA](./linear-attention-to-gdn-to-kda.md) 讲如何用固定大小状态**消灭** softmax。本文讲第三条路线：**不改变注意力的数学定义，只改变它在硬件上的执行方式**——结果与教科书实现逐位对齐，速度成倍提升，显存从平方降到线性。

---

## 执行摘要

> **一句话结论**　Softmax 是一个 memory-bound 的多遍扫描算子，Online Softmax 用"运行最大值 $m$ + 运行归一化因子 $l$"的增量更新把它压成一遍流式扫描；FlashAttention 把这一思想嵌入注意力的分块（tiling）计算中，让 $N \times N$ 注意力矩阵**永不写入 HBM**，把标准注意力的 $\Theta(Nd + N^2)$ HBM 访问降到 $\Theta(N^2 d^2 / M)$（$M$ 为 SRAM 容量）——**数学上完全等价，工程上快 2-4 倍、显存线性**。FlashAttention-2 通过延迟归一化、Q 外循环并行与 warp 重排再快约 2 倍；FlashAttention-3 借助 Hopper 异步特性与 FP8 把硬件利用率推到 75%。

| 算法 | 核心思想 | softmax 扫描方式 | HBM 访问量 | 与精确 softmax 的关系 |
| --- | --- | --- | --- | --- |
| Naive Softmax | 直接对 logits 取 exp | 1 遍（但会溢出） | $\Theta(N)$ | 数值不安全 |
| Safe Softmax | 先减最大值再 exp | 3 遍 | $\Theta(N)$ | 数学恒等 |
| Online Softmax | 运行 $(m, l)$ 增量更新 | 1 遍流式（+1 遍归一化） | $\Theta(N)$ | 数学恒等 |
| Memory-Efficient Attention | 分块 + online softmax | 融合进分块 | 显存线性，IO 非最优 | 数学恒等 |
| FlashAttention-1 | 分块 + online softmax + 反向重算 | 融合进 matmul | $\Theta(N^2 d^2 / M)$，理论最优 | 数学恒等 |
| FlashAttention-2 | 延迟归一化 + 并行粒度/warp 重排 | 融合 | 同阶，常数项更优 | 数学恒等 |
| FlashAttention-3 | warp 专精 + GEMM/softmax 异步重叠 + FP8 | 融合 | 同阶，利用率约 75% | FP16 恒等；FP8 有量化误差 |

### 阅读导航

| 章节 | 主题 | 教学重点 |
| --- | --- | --- |
| 01 | Softmax 的数值稳定与三遍扫描 | naive softmax 为什么溢出？safe softmax 为什么要扫三遍数据？ |
| 02 | Online Softmax | 运行状态 $(m, l)$ 如何增量更新？为什么数学上与 safe softmax 严格一致？ |
| 03 | 标准 Attention 的内存墙 | GPU 内存层次什么样？教科书注意力为什么被 HBM 带宽拖死？ |
| 04 | FlashAttention-1 | tiling 如何与 online softmax 融合？反向传播为什么"重算"反而更快？ |
| 05 | FlashAttention-2 | 延迟归一化、并行粒度与 warp 划分如何再榨出 2 倍？ |
| 06 | FlashAttention-3 | Hopper 的异步执行、GEMM-softmax 重叠与 FP8 各解决什么问题？ |
| 07 | 工程细节与本仓库实现 | exp2 技巧、causal 跳块、FlashDecoding、手写实现代码走读 |
| 08 | 总结 | 演进逻辑、设计权衡、与其他技术线的关系 |

---

## 1. Softmax：一个看似平凡却不平凡的算子

FlashAttention 的全部故事都埋在 softmax 里。先看清楚这个算子本身的两个问题：**数值安全**与**内存访问模式**。

### 1.1 Naive softmax 与上溢

softmax 把任意实数向量 $x \in \mathbb{R}^N$ 归一化为概率分布：

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

问题出在指数函数的增长速度上。浮点数有表示上限：

| 精度 | 最大值 | $e^x$ 溢出的临界 $x$ |
| --- | --- | --- |
| FP16 | 65,504 | $x > \ln 65504 \approx 11.09$ |
| BF16 | $\approx 3.39 \times 10^{38}$ | $x > 88.72$ |
| FP32 | $\approx 3.40 \times 10^{38}$ | $x > 88.72$ |

注意力的 logits 是缩放点积 $\mathbf{q}^\top \mathbf{k} / \sqrt{d}$——内积的尺度随维度增长，训练初期权重较大时，$|x| > 11$ 毫不稀奇。一旦某个 $e^{x_i}$ 变成 `inf`，分子分母同时溢出，`inf/inf = NaN`，整个前向宣告报废。

### 1.2 Safe softmax：减最大值，三遍扫描

标准修复是利用 logsumexp 恒等式：对任意常数 $c$，分子分母同乘 $e^{-c}$ 不改变结果。取 $c = m = \max_j x_j$：

$$
\boxed{\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{j=1}^{N} e^{x_j - m}}, \qquad m = \max_{j} x_j}
$$

这样最大的指数参数是 $x_i - m = 0$，$e^0 = 1$，**永远不会上溢**（下溢到 0 是无害的——该位置的权重本来就趋近于 0）。这就是"safe softmax"，数学上与 naive softmax **严格恒等**，只是浮点安全的写法。

但代价藏在计算结构上。safe softmax 需要**三遍扫描**数据：

| 遍次 | 操作 | 内存访问 |
| --- | --- | --- |
| 第 1 遍 | 求最大值 $m$ | 读 $N$ 个元素 |
| 第 2 遍 | 求归一化因子 $l = \sum_j e^{x_j - m}$ | 读 $N$ 个元素 |
| 第 3 遍 | 写出 $y_i = e^{x_i - m} / l$ | 读 $N$ 个 + 写 $N$ 个 |

合计约 $4N$ 次 HBM 访问。为什么必须分三遍？因为每一遍的计算都依赖前一遍的**全局**结果：没有全局最大值就不敢取 exp，没有全局求和就不敢归一化。**全局依赖 = 必须等所有数据过完一遍才能开始下一遍。**

### 1.3 为什么三遍扫描是大问题：softmax 是 memory-bound 算子

GPU 的吞吐瓶颈分两种：算力（FLOPs/s）与带宽（bytes/s）。一个算子受哪个瓶颈约束，由**算术强度**（arithmetic intensity = FLOPs ÷ 访存字节数）决定：

$$
\text{算术强度} > \text{盈亏平衡点} \Rightarrow \text{compute-bound}；\quad \text{算术强度} < \text{盈亏平衡点} \Rightarrow \text{memory-bound}
$$

以 A100 为例，盈亏平衡点约为 $312\ \text{TFLOPS} \div 1.6\ \text{TB/s} \approx 195$ FLOP/byte（FP16 张量核口径）。

softmax 每读一个字节只做几次简单运算（减、exp、加、除），算术强度在 1 FLOP/byte 量级——**比盈亏平衡点低两个数量级**。这意味着 GPU 跑 softmax 时，99% 的周期在等数据从 HBM 搬进来，计算单元绝大部分时间空转。

单看一个向量的 softmax，三遍扫描无非多浪费几倍时间；但在注意力里，softmax 作用在 $N \times N$ 的分数矩阵 $S$ 上——**三遍扫描 = 三次 $\Theta(N^2)$ 的 HBM 读写**。序列一长，这就是注意力的真实瓶颈（第 3 节定量分析）。

> **核心问题（之一）**　能否只扫一遍数据就算出 softmax？

---

## 2. Online Softmax：一边读数据，一边改答案

答案是肯定的。Online Softmax（Milakov & Gimelshein, NVIDIA, 2018）的思路是：**不等到看完全局再动手，而是维护一个"截至目前的最好答案"，每来一个新数据就修正一次。**

### 2.1 核心思路：运行最大值 + 运行归一化因子

对数据流 $x_1, x_2, \ldots$ 维护两个标量状态：

$$
m_k = \max_{1 \le i \le k} x_i \quad (\text{截至 } k \text{ 的运行最大值}), \qquad l_k = \sum_{i=1}^{k} e^{x_i - m_k} \quad (\text{以 } m_k \text{ 为基准的运行求和})
$$

难点在于：$l_k$ 是**以当前最大值 $m_k$ 为基准**累加的。如果第 $k+1$ 个元素刷新了最大值，之前累加的每一项 $e^{x_i - m_k}$ 的基准都过时了——难道要全部重算？

不需要。因为基准换算只是一个乘法：

$$
e^{x_i - m_{k+1}} = e^{x_i - m_k} \cdot e^{m_k - m_{k+1}}
$$

旧累加值整体乘一个修正因子 $\alpha = e^{m_k - m_{k+1}}$，就完成了"换基准"。

### 2.2 更新公式推导

新元素 $x_{k+1}$ 到来时：

$$
m_{k+1} = \max(m_k,\ x_{k+1})
$$

$$
l_{k+1} = \sum_{i=1}^{k+1} e^{x_i - m_{k+1}} = \underbrace{\sum_{i=1}^{k} e^{x_i - m_k} \cdot e^{m_k - m_{k+1}}}_{\text{旧部分换基准}} + \underbrace{e^{x_{k+1} - m_{k+1}}}_{\text{新元素}} = \alpha \cdot l_k + e^{x_{k+1} - m_{k+1}}
$$

于是一遍扫描的递推就是：

$$
\boxed{m_{k+1} = \max(m_k,\ x_{k+1}), \qquad l_{k+1} = e^{m_k - m_{k+1}} \cdot l_k + e^{x_{k+1} - m_{k+1}}}
$$

扫描结束后，任意位置的 softmax 值为 $y_i = e^{x_i - m_N} / l_N$。

**修正因子的性质**：由于 $m_{k+1} \ge m_k$，恒有 $\alpha = e^{m_k - m_{k+1}} \in (0, 1]$。修正只会"缩小"旧值，永不放大——这个方向性对数值稳定至关重要（见 2.3 节）。

### 2.3 正确性：为什么和 safe softmax 严格一致

**命题**：扫描完 $N$ 个元素后，$m_N = \max_j x_j$，$l_N = \sum_j e^{x_j - m_N}$。

**证明**（对 $k$ 归纳）：$k = 1$ 时 $m_1 = x_1$，$l_1 = e^{x_1 - x_1} = 1$，成立。假设 $k$ 时成立，即 $l_k = \sum_{i=1}^{k} e^{x_i - m_k}$，则 2.2 节的推导直接给出 $l_{k+1} = \sum_{i=1}^{k+1} e^{x_i - m_{k+1}}$，且 $m_{k+1} = \max_{i \le k+1} x_i$ 由定义成立。$\square$

因此最终 $y_i = e^{x_i - m_N} / l_N$ 与 safe softmax **逐点相同**——这不是近似算法，是同一数学对象的另一种求值顺序（浮点求和顺序不同会带来最后一位的差异，但误差有界且实测无影响）。

**数值安全性**，逐条检查所有 exp 的参数：

| exp 出现处 | 参数上界 | 结论 |
| --- | --- | --- |
| $e^{x_{k+1} - m_{k+1}}$ | $x_{k+1} \le m_{k+1} \Rightarrow$ 参数 $\le 0$ | 不上溢 |
| $\alpha = e^{m_k - m_{k+1}}$ | $m_k \le m_{k+1} \Rightarrow$ 参数 $\le 0$ | 不上溢，且 $\alpha \in (0,1]$ |
| 最终 $e^{x_i - m_N}$ | 参数 $\le 0$ | 不上溢 |

另外，由于最大值元素本身贡献 $e^0 = 1$，恒有 $l_N \ge 1 > 0$——**除零也不可能发生**（空输入除外；全 mask 行的处理见 7.6 节的代码走读）。

### 2.4 分块合并形式：softmax 的"结合律"

逐元素递推可以自然推广为**块级合并**。设集合 $A$、$B$ 各自已经算好局部状态 $(m_A, l_A)$、$(m_B, l_B)$，则合并状态为：

$$
\boxed{m = \max(m_A, m_B), \qquad l = l_A \cdot e^{m_A - m} + l_B \cdot e^{m_B - m}}
$$

证明与 2.2 节完全相同：把基准较小的那一侧乘上修正因子即可。这个 merge 运算满足结合律，意味着：

- **任意分块策略都正确**：一块一块流式扫、两路并行再合并、树形归约——结果一致
- **它是 GPU 并行的通行证**：warp 内、block 内、甚至跨设备（见 7.4 节 FlashDecoding 与 Ring Attention）都可以用同一套代数合并部分结果

换一个等价视角：定义 logsumexp 状态 $\text{LSE} = m + \log l$。merge 运算在 LSE 空间就是对 $(\text{LSE}_A, \text{LSE}_B)$ 再做一次 logsumexp——**LSE 是 softmax 分块合并的完备压缩状态**，记住这个量，它在 FlashAttention 的反向传播（4.5 节）和 decode 优化（7.4 节）中会反复出现。

### 2.5 从三遍到两遍：向量 kernel 的优化与局限

Online softmax 把 safe softmax 的 3 遍扫描压缩为 2 遍：第 1 遍流式算出 $(m, l)$，第 2 遍写出 $y_i = e^{x_i - m} / l$。访存量从 $4N$ 降到 $3N$，原论文报告在 V100 上对 softmax kernel 最高约 1.3 倍加速。

但必须诚实指出它的局限：**这个优化救不了注意力。** 注意力的问题不是 softmax 内部多扫一遍，而是 softmax 的输入——$N \times N$ 分数矩阵 $S$——本身要物化到 HBM（写一次、读多次）。只要 $S$ 落盘，$\Theta(N^2)$ 的访存就一分不少。真正的解法是**把 softmax 彻底融化进注意力的分块计算里**，让 $S$ 根本不离开片上内存——这正是 FlashAttention 的故事。

> **核心问题（之二）**　注意力的 $N \times N$ 中间矩阵能不能永不写入 HBM？

---

## 3. 标准 Attention 的内存墙

### 3.1 GPU 内存层次：SRAM 与 HBM 差两个数量级

GPU 的存储是一座金字塔，越靠近算力越小越快。以 A100（FlashAttention-1 的目标硬件）为例：

| 层级 | 容量 | 带宽 | 角色 |
| --- | --- | --- | --- |
| SRAM（片上共享内存/寄存器） | 每 SM 192 KB × 108 SM ≈ 20 MB | 约 19 TB/s（估算） | kernel 内的工作区 |
| HBM（高带宽显存） | 40-80 GB | 1.5-2.0 TB/s | 张量的"家" |
| DRAM（CPU 内存） | TB 级 | 约 0.1 TB/s |  offload 才会用到 |

两个关键事实：

1. **带宽差一个数量级**（19 TB/s vs 1.6 TB/s）：同样的数据，在 SRAM 里读比在 HBM 里读快 10 倍以上
2. **容量差四个数量级**（20 MB vs 40 GB）：SRAM 装不下任何像样的 $N \times N$ 矩阵——$N = 8192$、FP16 的方阵就是 134 MB

于是 GPU 编程的黄金法则是：**让数据在 SRAM 里被尽可能多地复用，把 HBM 访问压到最少**。大矩阵乘法（GEMM）天然满足这一点（算术强度 $\Theta(d)$，几百 FLOP/byte），所以 GEMM 是 compute-bound 的；而 softmax、dropout、mask、逐元素加减乘除都是 memory-bound 的。

### 3.2 教科书注意力的 HBM 账本

教科书实现 $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d}) V$ 是三个独立 kernel 的串行，中间结果全部经由 HBM 中转。设序列长度 $N$、头维 $d$，统计单个（batch, head）的 HBM 访问量：

| 步骤 | 读 | 写 | 量级 |
| --- | --- | --- | --- |
| $S = QK^\top/\sqrt{d}$ | $Q, K$：$2Nd$ | $S$：$N^2$ | $\Theta(N^2)$ |
| $P = \text{softmax}(S)$ | $S$：$N^2$（safe softmax 实际多遍） | $P$：$N^2$ | $\Theta(N^2)$ |
| $O = PV$ | $P$：$N^2$，$V$：$Nd$ | $O$：$Nd$ | $\Theta(N^2)$ |
| 合计 | — | — | $\Theta(N^2 + Nd)$ |

也就是说，**真正消耗时间的不是矩阵乘（compute-bound，跑得快），而是 $S$ 和 $P$ 这两个 $N \times N$ 矩阵在 HBM 上的反复读写**。用 roofline 语言说：标准注意力把两个 compute-bound 的 GEMM 用一个 memory-bound 的 softmax 粘起来，结果整体被拖到 memory-bound 的沟里。

训练场景还要再补一刀：反向传播需要 $S$ 和 $P$，标准实现会把它们从前向**存下来**供反向使用——$\Theta(N^2)$ 的显存占用随序列长度平方增长。$N = 64$K、batch 32、32 头的模型，光注意力中间矩阵就要 PB 级——这就是长序列训练"显存爆炸"的真正来源。

### 3.3 核心问题

> 能否计算**精确**注意力，同时 (a) 不把 $S$、$P$ 写入 HBM，(b) 显存占用随 $N$ 线性而非平方？

约束条件很苛刻：SRAM 只有约 100 KB 量级，装不下整行 $S$。所以唯一的出路是**分块（tiling）**：每次只在 SRAM 里算一小块 $S_{ij}$，算完立刻消费掉（乘上 $V$ 累加进输出），绝不落盘。但 softmax 需要全局归一化——块与块之间怎么衔接？答案就是第 2 节的 online softmax。

**历史脉络**：这条路线不是 FlashAttention 首创。Rabe & Staats（2021）的 *Self-attention Does Not Need $\mathcal{O}(n^2)$ Memory* 已经用"分块 + online softmax + 反向重算"实现了线性显存的精确注意力（xformers 的 `memory_efficient_attention` 即此 lineage，PyTorch SDPA 的 mem-efficient 后端同源）。但该工作面向 TPU/推理场景，**没有做精细的 IO 分析与 kernel 级融合**，速度收益有限。FlashAttention 的贡献是把 IO 复杂度作为一等公民：用定理刻画最优 HBM 访问量，并用 CUDA kernel 把这个最优值真正跑出来。

---

## 4. FlashAttention-1：让注意力矩阵永不落盘

FlashAttention-1（Dao, Fu, Ermon, Rudra, Ré, NeurIPS 2022）由三根支柱构成：**分块计算（tiling）**、**在线 softmax（online softmax 的块级应用）**、**反向重算（recomputation）**。

### 4.1 分块方案：把 SRAM 当缓存用

把 $Q$ 沿行切成 $T_r = N / B_r$ 个块，$K, V$ 沿行切成 $T_c = N / B_c$ 个块：

$$
Q = \begin{bmatrix} Q_1 \\ \vdots \\ Q_{T_r} \end{bmatrix}, \quad Q_i \in \mathbb{R}^{B_r \times d}; \qquad K = \begin{bmatrix} K_1 \\ \vdots \\ K_{T_c} \end{bmatrix}, \quad V = \begin{bmatrix} V_1 \\ \vdots \\ V_{T_c} \end{bmatrix}, \quad K_j, V_j \in \mathbb{R}^{B_c \times d}
$$

块大小按 SRAM 预算 $M$ 选取：$B_c = \lceil M / 4d \rceil$，$B_r = \min(\lceil M / 4d \rceil,\ d)$（保证 $K_j, V_j, Q_i, O_i$ 四块同时放得下）。实践中 $B_r, B_c$ 常取 64-128。

**循环结构**：对每个 $Q$ 块 $Q_i$，依次扫过所有 KV 块 $(K_j, V_j)$，在 SRAM 里完成 $S_{ij} = Q_i K_j^\top$、块级 online softmax、$\tilde{P}_{ij} V_j$ 累加——全程只有块大小的中间量，$N \times N$ 矩阵从不存在。

### 4.2 前向算法：每个 Q 块的流式 KV 扫描

对每个 $Q$ 块 $Q_i$，维护三个运行状态（全部在 SRAM/寄存器里）：

- $m_i \in \mathbb{R}^{B_r}$：截至当前的**逐行最大值**，初始 $-\infty$
- $l_i \in \mathbb{R}^{B_r}$：以 $m_i$ 为基准的**逐行 exp 和**，初始 0
- $O_i \in \mathbb{R}^{B_r \times d}$：**已归一化**的输出累加器，初始 0

处理第 $j$ 个 KV 块时：

**第一步：算块内分数**。$S_{ij} = Q_i K_j^\top / \sqrt{d} \in \mathbb{R}^{B_r \times B_c}$（张量核 GEMM，SRAM 内完成）。

**第二步：更新运行最大值**。

$$
\tilde{m}_{ij} = \text{rowmax}(S_{ij}), \qquad m_i^{new} = \max(m_i,\ \tilde{m}_{ij})
$$

**第三步：以新最大值为基准，算未归一化权重**。

$$
\tilde{P}_{ij} = e^{S_{ij} - m_i^{new}} \in \mathbb{R}^{B_r \times B_c} \quad (\text{元素均} \in (0, 1]\text{，安全})
$$

**第四步：修正因子换基准**（这就是 2.2 节的 $\alpha$，现在是逐行的）。

$$
\alpha = e^{m_i - m_i^{new}} \in (0, 1]^{B_r}, \qquad l_i^{new} = \alpha \cdot l_i + \text{rowsum}(\tilde{P}_{ij})
$$

**第五步：累加输出**。FA1 的选择是**每步都归一化**，维护不变量"$O_i$ 等于已见前缀上的精确注意力输出"：

$$
O_i \leftarrow \frac{\alpha \, l_i}{l_i^{new}} \cdot O_i + \frac{1}{l_i^{new}} \cdot \tilde{P}_{ij} V_j
$$

**正确性直觉**：旧输出 $O_i$ 原本是除以旧 $l_i$ 归一化的，先把分母"退回分子"（乘 $\alpha l_i$），并入本块的贡献 $\tilde{P}_{ij} V_j$，再统一除以新分母 $l_i^{new}$——与 2.4 节的块合并代数完全一致。扫完全部 $T_c$ 个 KV 块后，$O_i$ 就是该 $Q$ 块的精确注意力输出，写回 HBM。

### 4.3 IO 复杂度：从 $\Theta(N^2)$ 到 $\Theta(N^2 d^2 / M)$

**标准实现**：$\Theta(Nd + N^2)$ 次 HBM 访问（3.2 节的账本）。

**FlashAttention**：外循环 $T_c$ 个 KV 块每个被全部 $T_r$ 个 $Q$ 块读一遍（或等价地，KV 常驻、Q 流式），HBM 访问总量为：

$$
\Theta\!\left(\frac{N}{B_c} \cdot \frac{N}{B_r} \cdot B_c d + \frac{N}{B_r} \cdot B_r d\right) = \Theta\!\left(\frac{N^2 d}{B_r} + Nd\right) = \Theta\!\left(\frac{N^2 d^2}{M}\right)
$$

直观理解：**每个 KV 块被重复读取 $N/B_r$ 次**，而 $B_r \approx M/d$ 由 SRAM 容量决定——SRAM 越大，重复读越少。论文还证明了匹配的下界：对精确注意力，任何算法在 $M \in [d, Nd]$ 范围内都需要 $\Omega(N^2 d^2 / M)$ 次 HBM 访问——**FlashAttention 的 IO 复杂度在同阶意义下已是最优**。

代入典型数字（$N = 8192$，$M \approx 50$K 元素）：

| 头维 $d$ | $N^2 d^2 / M$ ÷ $N^2$ | 相对标准实现的 HBM 访问比 |
| --- | --- | --- |
| 64 | $4096 / 50000 \approx 0.08$ | 约 1/12 |
| 128 | $16384 / 50000 \approx 0.33$ | 约 1/3 |

再叠加标准实现对 $S/P$ 的多遍读写（3-5 次 $\Theta(N^2)$），实际 HBM 流量差距在 9-20 倍——与论文实测的 wallclock 加速一致。

### 4.4 反向传播：存储 logsumexp，重算注意力

反向需要 $P$ 来计算梯度，但前向恰恰没有存 $P$（这正是省显存的关键）。FA1 的选择：**重算（recomputation）**。

前向只额外保存两样小东西：输出 $O$（$\Theta(Nd)$）和逐行 logsumexp 统计量（$\Theta(N)$）：

$$
\boxed{L_i = m_i + \log l_i \quad \in \mathbb{R}^{N}}
$$

反向时，对每个块重新计算 $S_{ij} = Q_i K_j^\top$，然后用 $L$ 一步到位地重建**已归一化**的 $P$：

$$
P_{ij} = e^{S_{ij} - L_i}
$$

（验证：$e^{S_{ij} - L_i} = e^{S_{ij} - m_i} / e^{\log l_i} = e^{S_{ij} - m_i} / l_i$，正是全局 softmax。）之后的梯度按 softmax 求导恒等式在块内完成：

$$
dP_{ij} = dO_i\, V_j^\top, \qquad dS_{ij} = P_{ij} \circ (dP_{ij} - D_i), \qquad D_i = \text{rowsum}(dO_i \circ O_i)
$$

$$
dV_j \mathrel{+}= P_{ij}^\top dO_i, \qquad dK_j \mathrel{+}= dS_{ij}^\top Q_i, \qquad dQ_i \mathrel{+}= dS_{ij} K_j
$$

两个精妙细节：

1. **$D_i$ 不需要 $P$**：softmax 反向里的对角项 $D_i = \text{rowsum}(dP \circ P)$ 可以等价写成 $\text{rowsum}(dO \circ O)$——而 $dO$ 和 $O$ 本来就存着。于是反向全程不依赖任何前向的 $N \times N$ 中间量。
2. **重算反而更快**：重算 $S, P$ 增加了 FLOPs，但注意力是 memory-bound——省下的 $N \times N$ HBM 读写远大于多算的 GEMM。这就是"用计算换带宽"的经典权衡，在 memory-bound 区间永远是赚的。

显存账本：标准实现反向需存 $S, P$ 共 $\Theta(N^2)$；FA 只存 $O, L$ 共 $\Theta(Nd)$——**显存从平方降为线性**，长序列训练由此可行。

### 4.5 精确性与实测收益

必须强调：**FlashAttention 是精确注意力，不是近似**。它的输出与教科书实现在数学上恒等（浮点求和顺序差异除外），不引入任何近似误差、不需要重训、不改变模型行为——这也是为什么它能无摩擦替换所有 Transformer 的注意力实现。

论文报告的收益（A100）：

| 指标 | 数字 |
| --- | --- |
| 注意力 kernel wallclock | 比标准实现快 2-4 倍 |
| 端到端训练 | BERT-large 比 MLPerf 1.1 纪录快 15%；GPT-2 比 HuggingFace 基线快约 3 倍；Long-Range-Arena 快约 2.4 倍 |
| 显存 | 10-20 倍缩减，随 $N$ 线性增长 |
| 可用序列长度 | 同显存下显著更长（GPT-2 上 4K+ 上下文成为常态） |

---

## 5. FlashAttention-2：榨干 GPU 的剩余 60%

FA1 的 IO 已达理论最优，但实测只跑到 A100 理论峰值算力的 **25-40%**。FlashAttention-2（Dao, 2023）的诊断是：**瓶颈从 HBM 带宽转移到了"非矩阵乘 FLOPs"与"并行度不足"**。三项优化都围绕这两点。

### 5.1 优化一：延迟归一化，减少非矩阵乘 FLOPs

GPU 上两类运算的吞吐相差悬殊（A100）：张量核 GEMM 312 TFLOPS（FP16/BF16），而普通 CUDA 核上的逐元素运算（exp、除法、乘加）只有 19.5 TFLOPS——**相差 16 倍**。FA1 每处理一个 KV 块都要做一次除法归一化（4.2 节第五步：除以 $l_i^{new}$）和一次 $O$ 的整体 rescale，这些全是昂贵的非矩阵乘运算。

FA2 的改法——**延迟归一化（deferred normalization）**：内循环里只维护**未归一化**的累加器 $\tilde{O}_i$，把除法推迟到所有 KV 块扫完之后只做一次：

$$
\tilde{O}_i \leftarrow \alpha \cdot \tilde{O}_i + \tilde{P}_{ij} V_j \quad (\text{内循环}), \qquad O_i = \tilde{O}_i / l_i \quad (\text{扫完后一次})
$$

对比 4.2 节：省掉了每步的除法，且 $O$ 的 rescale 从"乘 $\alpha l_i / l_i^{new}$"简化为"乘 $\alpha$"。数学上等价（把 $O_i \cdot l_i$ 这个不变量维护到底再除掉）。前向结束时顺带输出 $L = m + \log l$ 供反向使用。

### 5.2 优化二：并行粒度——外循环换成 Q

FA1 的 kernel 网格只在 (batch, head) 维度上并行，块内是 KV 外循环、Q 内循环。问题来了：长序列场景 batch × heads 往往很小（比如 batch 4、8 头 = 32 个并行单元），而 A100 有 108 个 SM——**大部分 SM 在空转**。

FA2 把 $Q$ 块也纳入网格：每个 thread block 负责一个 $Q$ 块，**外循环固定 Q、内循环流式扫 KV**。好处有两个：

1. **并行单元数乘以 $T_r$**：(batch, head, Q块) 三维网格，长序列下轻松填满所有 SM
2. **零跨块通信**：每个 $Q$ 块的输出 $O_i$ 只由自己的 KV 扫描决定，块间无需任何同步或归约

对 causal 注意力还有附带红利：第 $i$ 个 $Q$ 块只需扫到对角线位置的 KV 块，**整块跳过上三角**——块数近似减半，且无一块需要做细粒度掩码的块外计算（对角块内部才需要逐元素 mask）。

### 5.3 优化三：warp 间工作划分

thread block 内部 4 个 warp 的分工也重排了：

| | FA1：split-KV | FA2：split-Q |
| --- | --- | --- |
| 分工 | 4 个 warp 各切一片 $K, V$，$Q$ 全员共享 | 4 个 warp 各切一片 $Q$，$K, V$ 全员共享 |
| 汇合 | 各 warp 算出部分 $O_i$，需写共享内存 + 同步 + 跨 warp 归约 | 各 warp 独立算完自己那片 $Q$ 的完整输出 |
| 通信 | 需要 | **完全不需要** |

FA1 的 split-KV 之所以需要归约，是因为不同 warp 持有同一批 query 对不同 KV 段的部分结果（各自的 $m, l$ 基准不同，合并要走 2.4 节的 merge 代数）；split-Q 让 merge 只发生在 warp 内部的 KV 流式扫描中，跨 warp 通信清零。

### 5.4 收益

| 指标 | FA1 | FA2 |
| --- | --- | --- |
| A100 理论峰值利用率 | 25-40% | **50-73%** |
| 相对 FA1 加速 | — | **约 2 倍** |
| 相对标准实现 | 2-4 倍 | 约 5-9 倍 |

至此，注意力在 FP16/BF16 下已经非常接近"纯 GEMM"的速度——memory-bound 的帽子被彻底摘掉。

---

## 6. FlashAttention-3：Hopper 时代的异步与低精度

FA2 在 A100 上已接近极限，但同样的 kernel 搬到 H100 后利用率反而只剩约 35%——H100 的张量核算力（FP16 约 989 TFLOPS）比 A100 翻了 3 倍多，而执行 exp 的 SFU/CUDA 核并没有同步变快，**GEMM 与 softmax 串行执行、互相等待**成为新瓶颈。FlashAttention-3（Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, 2024）的答案是 Hopper 架构的异步特性。

### 6.1 Warp 专精与乒乓调度

Hopper 引入 TMA（Tensor Memory Accelerator，异步批量拷贝引擎）并强化了 warp 级异步。FA3 把 thread block 内的 warp 分成**生产者**（用 TMA 搬下一块 K/V）与**消费者**（计算），并让两个消费者 warpgroup **乒乓执行**：一个做 GEMM 时，另一个同时做 softmax。

### 6.2 GEMM 与 softmax 重叠

关键观察：GEMM 跑在**张量核**上，softmax 的 exp/乘加跑在 **CUDA 核/SFU** 上——这是两套独立的硬件单元，可以真正并行。FA3 在指令级把"第 $j$ 块的 softmax（含 online 更新）"与"第 $j+1$ 块的 $QK^\top$ / $PV$ GEMM"交错发射，softmax 的延迟被 GEMM 完全掩盖。4.2 节的 online 更新步骤一个没少，只是**不再占用关键路径**。

### 6.3 FP8：块量化 + 不相干处理

H100 的 FP8 张量核算力再翻一倍（约 2 PFLOPS），但 FP8（e4m3）只有约 3 位有效尾数，注意力 logits 中的 outlier 会被量化误差显著放大。FA3 用两项技术压误差：

1. **逐块量化**：每个块独立缩放因子，而非整张量一个 scale
2. **不相干处理（incoherent processing）**：给 $Q, K$ 同乘一个随机 Hadamard 矩阵 $H$。由于 $H$ 正交，$(QH)(KH)^\top = Q H H^\top K^\top = QK^\top$——内积数学不变，但 outlier 的能量被打散到所有维度，量化误差显著下降

### 6.4 数字与适用边界

| 指标 | 数字 |
| --- | --- |
| H100 FP16 | 约 740 TFLOPS（理论峰值的 75%），比 FA2 快 1.5-2 倍 |
| H100 FP8 | 约 1.2 PFLOPS，误差显著低于朴素 FP8 注意力 |

注意 FA3 依赖 Hopper 专有特性（TMA、warp 专精），并不向后兼容：本仓库 [geo3k_vlm/README.md](../../slime/examples/geo3k_vlm/README.md) 就记录了 Blackwell 当前不支持 FA3、需回退 `--attn-implementation flash_attention_2` 的实践经验；slime 的[可复现性文档](../../slime/docs/zh/advanced/reproducibility.md)也提到追求确定性训练时需要卸载 FA3。

---

## 7. 工程细节、数值性质与本仓库实现

### 7.1 数值稳定性与确定性清单

| 性质 | 结论 |
| --- | --- |
| 上溢 | 不可能：所有 exp 的参数 $\le 0$（2.3 节已逐条验证） |
| 除零 | 正常输入不可能：最大值元素贡献 $e^0 = 1$，故 $l \ge 1$；全 mask 行需特判（见 7.6 节代码走读） |
| 前向确定性 | 确定（固定分块与归约顺序） |
| 反向确定性 | 默认实现用 atomicAdd 累加 $dK/dV$，浮点层面**不可复现**；`deterministic=True` 可换可复现性（有性能代价） |
| 舍入偏差 | 有偏的舍入误差确实存在：Kimi K3 技术报告（见本仓库 [kimi.md](./kimi.md)）记录的对策是**训练时把注意力输出 tile 保持在 FP32**，代价是片上占用翻倍，需要重新设计 kernel 的缓冲重叠 |

### 7.2 base-2 技巧：为什么是 exp2 而不是 exp

GPU 的 SFU 指数指令 `MUFU.EX2` 是**以 2 为底**的。flash-attn 系列因此全程在 base-2 下工作：预先把 $\log_2 e$ 折进 softmax scale（即 $S' = QK^\top \cdot \text{scale} \cdot \log_2 e$），于是 $e^{x - m} = 2^{(x' - m')}$，每次 exp 省一条换算指令；仅在对 API 输出 LSE 时换算回自然对数。这类细节的堆叠，正是 FA 系列把"常数项"一路压下去的方式。

### 7.3 Causal 与滑窗的块级处理

causal 掩码在分块视角下非常干净：**对角线以上的 KV 块整块跳过**（迭代块数近似减半），对角块内部才需要逐元素 mask。sliding window 同理——窗口外整块跳过，窗口边界块做逐元素判断。本仓库 [gemma4.py](../../slime/slime_plugins/models/gemma4.py) 即用 `flash_attn_varlen_func` 的 `window_size` 参数承载滑窗注意力（并记录了 head_dim=512 超出 flash-attn 2.x 支持范围时回退 SDPA 的兼容路径）。

### 7.4 同一套代数的三个应用：tiling、FlashDecoding、Ring Attention

2.4 节的块合并代数（$(m, l)$ / LSE 的 merge）在三个不同尺度上复用，值得集中对比：

| 应用 | 分块对象 | 合并时机 |
| --- | --- | --- |
| FlashAttention 分块（单卡） | KV 沿长度切块，SRAM 内流式扫 | 每个 KV 块处理完即更新 $(m, l, O)$ |
| FlashDecoding（单卡 decode） | KV 切成 $s$ 段分给不同 SM **并行** | 各段算完 $(O_s, \text{LSE}_s)$ 后集中合并 |
| Ring Attention（多卡） | KV 分片驻留各设备，绕环轮转 | 每收到一个 KV 分片即更新 $(m, l, O)$ |

decode 场景的问题与 5.2 节同源：batch × heads 太小填不满 SM。FlashDecoding 把 KV 长度维也切开并行，每个 split 产出一个**已归一化**的部分输出 $O_s$ 和 $\text{LSE}_s$，最终合并为：

$$
L = \log \sum_s e^{\text{LSE}_s}, \qquad O = \sum_s e^{\text{LSE}_s - L} \cdot O_s
$$

Ring Attention（Liu et al. 2023）则是分布式版：每台设备固定持有自己的 $Q$ 块，KV 分片绕设备环传递，每收到一片就执行一次 4.2 节的 online 更新——**FlashAttention 的内循环被拉到了设备间**。本仓库 [attention_rl_handwrite.py](../attention_rl_handwrite.py) 中的 `cp_ring_attention` 正是这一思想的教学实现。

### 7.5 变长序列：varlen packing

生产训练几乎不用 padding，而是把一个 batch 的多条序列**拼接成一条大序列**（THD 格式），用 `cu_seqlens`（累积长度数组）描述边界，配合块级掩码防止跨序列泄漏。flash-attn 的对应接口是 `flash_attn_varlen_func`——本仓库 slime 的注意力封装 [flash_dot_product_attention.py](../../slime/slime_plugins/models/flash_dot_product_attention.py) 全线使用它。

### 7.6 本仓库实现参考

**（1）教学版手写实现**：[attention_rl_handwrite.py](../attention_rl_handwrite.py) 中的 `flash_attention_simple`（第 1175 行起）。

- **输入**：`q` [B, N, Sq, D]、`k` [B, N, Sk, D]、`v` [B, N, Sk, Dv]，可选 `causal` 与块大小
- **输出**：[B, N, Sq, Dv]，与 PyTorch SDPA 参考实现逐位对齐（文件内冒烟测试用 `rtol=1e-5` 验证）

逻辑与第 4-5 节一一对应：

| 代码 | 对应公式 |
| --- | --- |
| `torch.matmul(qb, kb.transpose(-2, -1)) / math.sqrt(d)` | $S_{ij} = Q_i K_j^\top / \sqrt{d}$ |
| `scores.amax(dim=-1)` | $\tilde{m}_{ij} = \text{rowmax}(S_{ij})$ |
| `new_m = torch.maximum(m, block_max)` | $m_i^{new} = \max(m_i, \tilde{m}_{ij})$ |
| `alpha = torch.exp(m - new_m_safe)` | $\alpha = e^{m_i - m_i^{new}}$ |
| `p_block = torch.exp(scores - new_m_safe)` | $\tilde{P}_{ij} = e^{S_{ij} - m_i^{new}}$ |
| `acc = alpha * acc + torch.matmul(p_block, vb)` | $\tilde{O}_i \leftarrow \alpha \tilde{O}_i + \tilde{P}_{ij} V_j$ |
| `l = alpha * l + p_block.sum(...)` | $l_i^{new} = \alpha l_i + \text{rowsum}(\tilde{P}_{ij})$ |
| `acc / l.clamp_min(1e-30)`（循环结束后一次） | $O_i = \tilde{O}_i / l_i$ |

注意：这份手写实现采用的正是 **FA2 风格的延迟归一化**——内循环只做未归一化累加，除法在扫完全部 KV 块后只做一次（5.1 节）。

比教科书公式多出来的三行，全是数值边界处理：

- `new_m_safe`：尚未遇到任何合法 token 时 $m = -\infty$，`scores - (-inf) = NaN`；用 0 替代后 $e^{-\infty - 0} = 0$，块贡献自然为零
- `alpha` 的 `where(isfinite(m), ...)`：首个块时 $m = -\infty$，强制 $\alpha = 0$ 以清空初始累加器
- `l > 0` 判断 + `clamp_min`：**全 mask 行输出 0 而非 NaN**（冒烟测试中有对应用例）

生产 kernel 反而没有这些分支——causal 跳块保证第一个被处理的块至少含一个合法元素，$m$ 必然有限。**教学版用分支换通用性，生产版用调度设计换分支**，这是 kernel 工程里很典型的取舍。

**（2）生产调用与 LSE 的活用**：[learnable_softmax_attention.py](../../slime/slime_plugins/models/learnable_softmax_attention.py)。

- 调用 `flash_attn_varlen_func(..., return_attn_probs=True)` 拿到 `(out, softmax_lse, _)`；其中 `softmax_lse` 就是 4.4 节的 $L = m + \log l$，形状 `(nheads, total_q)`，FP32
- learnable softmax 给 logits 附加一个 per-head 可学习 offset 作为 attention sink，其等价变换 $P = \text{softmax}(QK^\top/\sqrt{d}) \cdot \text{sigmoid}(\text{LSE} - \text{offset})$ **完全建立在 flash-attn 暴露 LSE 的前提上**；反向通过自定义 autograd 把 sigmoid 缩放折进 $dO$，原样复用 flash-attn 的 backward kernel

这是"LSE 是 softmax 的完备压缩状态"（2.4 节）在生产代码里的直接证据：拿到 LSE，就等于拿到了整行 softmax 的全部归一化信息。

**（3）框架接线**：HF 模型侧通过 `_attn_implementation = "flash_attention_2"` 接入 FA2（[hf_attention.py](../../slime/slime_plugins/models/hf_attention.py)、[qwen3_5.py](../../slime/slime_plugins/models/qwen3_5.py)）。

---

## 8. 总结

### 8.1 演进逻辑

```
Naive Softmax
 │ ❌ exp(x) 直接溢出（FP16 下 x > 11 即 inf）
 │  减最大值 → 数学恒等且数值安全
 ▼
Safe Softmax
 │ ✅ 所有 exp 参数 ≤ 0，永不上溢
 │ ❌ 三遍扫描：max / sum / normalize，遍间是全局依赖
 │  维护运行状态 (m, l)，新数据到来时换基准（α = e^(m_old - m_new)）
 ▼
Online Softmax (Milakov & Gimelshein 2018)
 │ ✅ 一遍流式扫描算出 (m, l)，访存 4N → 3N
 │ ✅ merge 满足结合律 → 任意分块/并行粒度可合并
 │ ❌ 救不了注意力：N×N 分数矩阵本身仍要落 HBM
 │  把 online softmax 嵌进注意力分块，S/P 永不落盘
 ▼
Memory-Efficient Attention (Rabe & Staats 2021)
 │ ✅ 分块 + online softmax + 反向重算 → 显存线性
 │ ❌ 无 IO 复杂度分析，kernel 未深度融合，速度收益有限
 │  IO 一等公民：Θ(N²d²/M) 且证明同阶最优
 ▼
FlashAttention-1 (Dao et al. 2022)
 │ ✅ HBM 访问理论最优，wallclock 快 2-4 倍
 │ ✅ 反向只存 O 与 L = m + log l → 显存线性
 │ ❌ 只用上 A100 算力的 25-40%：非矩阵乘 FLOPs 与并行度是瓶颈
 │  延迟归一化 + Q 外循环三维网格 + split-Q warp 划分
 ▼
FlashAttention-2 (Dao 2023)
 │ ✅ 再快约 2 倍，A100 利用率 50-73%
 │ ❌ 搬到 H100 只剩约 35%：GEMM 与 softmax 串行互等
 │  warp 专精 + GEMM/softmax 异步重叠 + FP8 不相干处理
 ▼
FlashAttention-3 (Shah et al. 2024)
 │ ✅ H100 FP16 约 740 TFLOPS（75% 利用率），FP8 约 1.2 PFLOPS
 │ ⚠ 依赖 Hopper 专有特性；Blackwell 需回退 FA2
```

### 8.2 核心设计权衡

| 设计维度 | 选择 | 理由 |
| --- | --- | --- |
| softmax 求值顺序 | 在线更新 $(m, l)$ | 把全局依赖折叠为流式状态，与分块天然兼容 |
| 归一化时机 | 延迟到扫完后一次除法（FA2） | 除法/exp 是慢 16 倍的非矩阵乘运算，越少越好 |
| 中间矩阵 | $S/P$ 永不落 HBM，反向重算 | memory-bound 区间"用计算换带宽"必赚 |
| 反向存储 | 只存 $O$ 与 $L = m + \log l$ | LSE 是完备压缩状态；$D = \text{rowsum}(dO \circ O)$ 恒等式消去对 $P$ 的依赖 |
| 并行粒度 | (batch, head, Q块) 三维网格 | 长序列小 batch 下填满所有 SM |
| warp 划分 | split-Q 而非 split-KV | 跨 warp 归约清零 |
| 硬件跟进 | 异步/专精/FP8（FA3） | 张量核与 SFU 算力失衡时，用重叠把 softmax 藏进 GEMM 的影子里 |

### 8.3 与其他技术线的关系

三条注意力优化路线**正交且可组合**：

| 路线 | 层次 | 手段 | 与本文的关系 |
| --- | --- | --- | --- |
| FlashAttention 系列 | 执行层 | 不改数学定义，改硬件执行方式 | 本文 |
| MQA/GQA/MLA | 表示层 | 压缩 KV Cache 的存储 | 训练/长序列前向仍跑 FA kernel（DeepSeek-V2 基于改进版 FA2），见[姊妹篇](./mha-to-mqa-gqa-to-mla.md) |
| Linear Attention/GDN/KDA | 模型层 | 消灭 softmax，换固定大小状态 | 另一条路线，见[姊妹篇](./linear-attention-to-gdn-to-kda.md)；混合架构（Kimi Linear 3:1）中的精确检索层照旧走 flash kernel |

### 8.4 一句话总结

> Online softmax 把 softmax 的全局依赖折叠成 $(m, l)$ 两个可增量修正的运行状态；FlashAttention 把这一代数嵌入注意力分块，让 $N \times N$ 矩阵永不落盘，再用反向重算把显存也降到线性——**数学上一步不让，工程上把 memory-bound 的注意力跑出了 compute-bound 的速度**。FA2/FA3 不再改算法，只改调度与硬件特性的使用方式，把 GPU 利用率一路推到 75%。

---

## 参考文献

- Milakov & Gimelshein (2018). *Online normalizer calculation for softmax.* (Online Softmax) [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- Rabe & Staats (2021). *Self-attention Does Not Need $\mathcal{O}(n^2)$ Memory.* (Memory-Efficient Attention) [arXiv:2112.05682](https://arxiv.org/abs/2112.05682)
- Dao, Fu, Ermon, Rudra & Ré (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Shah, Bikshandi, Zhang, Thakkar, Ramani & Dao (2024). *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- Dao, Haziza, Ermon & Ré (2023). *Flash-Decoding for long-context inference.* (PyTorch Blog，split-KV decode)
- Liu et al. (2023). *Ring Attention with Blockwise Transformers for Near-Infinite Context.* [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
- Ye et al. (2025). *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.* [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)
- Vaswani et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
