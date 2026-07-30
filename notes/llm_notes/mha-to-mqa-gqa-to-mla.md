**TECHNICAL REFERENCE · 2026**

# 从 MHA 到 MQA、GQA 再到 MLA：KV Cache 压缩算法演进详解

*Multi-Head Attention → Multi-Query Attention → Grouped-Query Attention → Multi-head Latent Attention*

**面向教学的逐步推导 · 公式细节 · 直觉建立 · 演进逻辑**

**适用读者**

希望深入理解大模型推理效率优化与注意力机制演进脉络的研究人员与工程师

版本基线：2026 年 8 月

姊妹篇：[从 Full Attention 到 Linear Attention 到 GDN 再到 KDA](./linear-attention-to-gdn-to-kda.md)——那条路线用固定大小状态彻底消灭 KV Cache；本文讲的是另一条路线：保留 softmax 精确检索，转而压缩 KV Cache 的表示本身。

---

## 执行摘要

> **一句话结论**　MHA 为每个头独立缓存 KV，检索最强但缓存随头数线性膨胀；MQA 让所有查询头共享一组 KV，缓存压到极限但质量受损；GQA 分组共享取得折中，本质仍是"砍头"；MLA 转换思路——不减少头数，而是把所有头的 KV **联合压缩进一个低维潜向量**，配合**矩阵吸收**与**解耦 RoPE** 两大技术，以约等于 GQA-2.25 的缓存量实现**反超 MHA** 的质量。

| 方法 | 核心思路 | KV Cache / Token | 质量 | 关键突破 | 关键缺陷 |
| --- | --- | --- | --- | --- | --- |
| MHA | 每头独立 K/V | $2 n_h d_h l$ | 强 | 逐头独立检索 | 缓存随头数膨胀 |
| MQA | 全部头共享 1 组 K/V | $2 d_h l$ | 弱 | 缓存最小 | 单组 KV 成信息瓶颈 |
| GQA | $n_g$ 组内共享 K/V | $2 n_g d_h l$ | 中 | 平滑插值 MHA↔MQA | 仍是砍头，难任务掉点 |
| MLA | K/V 联合低秩压缩为潜向量 | $(d_c + d_h^R) l \approx \frac{9}{2} d_h l$ | **更强** | 缓存≈GQA-2.25 且质量超 MHA | 训练需解压，RoPE 需解耦 |

### 阅读导航

| 章节 | 主题 | 教学重点 |
| --- | --- | --- |
| 01 | MHA 与 KV Cache | KV Cache 从何而来？为什么说 decode 是带宽游戏？ |
| 02 | MQA | 共享 KV 如何把缓存压到极限？质量代价是什么？ |
| 03 | GQA | 分组如何插值 MHA 与 MQA？uptraining 如何低成本转换？ |
| 04 | MLA | 低秩联合压缩如何工作？矩阵吸收为什么关键？RoPE 为什么要解耦？ |
| 05 | 总结 | 完整对比、演进逻辑、设计权衡与后续演进 |

---

## 1. MHA 与 KV Cache：一切的起点

### 1.1 推理时为什么需要 KV Cache

回顾标准 Multi-Head Attention（MHA）。设嵌入维度为 $d$，头数为 $n_h$，每头维度为 $d_h$，第 $t$ 个 token 的注意力输入为 $\mathbf{h}_t \in \mathbb{R}^{d}$。MHA 先通过三个投影矩阵生成 query、key、value：

$$
\mathbf{q}_t = W^Q \mathbf{h}_t, \quad \mathbf{k}_t = W^K \mathbf{h}_t, \quad \mathbf{v}_t = W^V \mathbf{h}_t, \quad W^Q, W^K, W^V \in \mathbb{R}^{d_h n_h \times d}
$$

然后切分为 $n_h$ 个头分别计算注意力，最后拼接并经输出投影：

$$
\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^\top \mathbf{k}_{j,i}}{\sqrt{d_h}}\right) \mathbf{v}_{j,i}, \quad \mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \ldots; \mathbf{o}_{t,n_h}]
$$

**朴素自回归生成的问题**：生成第 $t$ 个 token 时，如果从零开始重算整段前缀，则每步要对全部 $t$ 个位置重新计算 K/V 投影，总复杂度高达 $\mathcal{O}(L^3)$。

**关键观察**：位置 $j \leq t$ 的 $\mathbf{k}_j, \mathbf{v}_j$ 只取决于 $\mathbf{h}_j$，一旦算出就**不再变化**。因此可以把它们缓存起来——这就是 KV Cache。有了缓存，每步只需为新 token 计算一次 K/V 投影（$\mathcal{O}(d \cdot d_h n_h)$），注意力部分降为 $\mathcal{O}(t \cdot d_h n_h)$。

**代价**：缓存本身成为新的状态。MHA 每个 token 需要缓存的元素数为：

$$
\boxed{\text{KV Cache / token} = \underbrace{2}_{K \text{ 与 } V} \times n_h \times d_h \times l}
$$

其中 $l$ 是层数。注意它**与序列长度无关**——这是"每 token"的存储单价，总缓存随上下文长度 $L$ 线性增长。

### 1.2 KV Cache 的规模测算

用 DeepSeek-V2 规模的配置（$n_h = 128$, $d_h = 128$, $l = 60$）做一个具体测算：

| 项目 | 计算 | 结果 |
| --- | --- | --- |
| 单 token 单层 | $2 \times 128 \times 128$ | 32,768 元素 |
| 单 token 全模型 | $32{,}768 \times 60$ | $\approx 1.97 \times 10^6$ 元素 |
| 单 token 体积（BF16） | $1.97 \times 10^6 \times 2$ 字节 | $\approx 3.9$ MB |
| 128K 上下文单序列 | $3.9\text{ MB} \times 131{,}072$ | $\approx 515$ GB |

515 GB 已经超过一个 8×80GB H800 节点的全部显存（640 GB）——而这还只是**一条序列**的 KV Cache，batch size 大于 1 时还要翻倍。

> **直觉理解**　KV Cache 像图书馆的"开架书"：每来一个新读者（新 token），管理员都要把之前所有上架的书（历史 K/V）翻一遍。书架上每多一本书，所有人的查询都变慢一分；而 MHA 的书架层数 = 头数，书多得放不下。

### 1.3 瓶颈分析：Decode 是带宽游戏

训练时的注意力是 $\mathcal{O}(L^2 d)$ 的大矩阵乘，计算密度高，GPU 利用率高。但**逐 token 解码**完全不同——它是一个**内存带宽受限**（memory-bound）的过程。

**算术强度分析**（单步 decode，单层，上下文长度 $t$）：

- **FLOPs**：QK 内积 $2 t d_h n_h$ + 加权求和 $2 t d_h n_h$ $= 4 t d_h n_h$
- **内存读取**：读入全部 K、V 缓存（BF16）$= 2 \times 2 t d_h n_h = 4 t d_h n_h$ 字节

$$
\text{算术强度} = \frac{4 t d_h n_h \text{ FLOPs}}{4 t d_h n_h \text{ bytes}} = 1 \text{ FLOP/byte}
$$

作为对比，H100 的盈亏平衡点（ridge point）约为 295 FLOP/byte（989 TFLOPS BF16 ÷ 3.35 TB/s）。算术强度 1 ≪ 295，意味着 **GPU 99% 的时间在等数据，而不是在计算**。

**批处理能救吗？** 模型权重可以被 batch 内所有序列共享读取（摊薄），但 KV Cache 是**每条序列私有**的，无法摊薄。于是 KV Cache 的大小直接决定了三件事：

1. **最大 batch size**：缓存越大，能并发的序列越少
2. **最大上下文长度**：缓存随 $L$ 线性增长，撞到显存上限
3. **解码延迟**：每步都要完整读一遍缓存

> **核心问题**　能否缩小 KV Cache 而不牺牲 softmax 注意力的精确检索能力？

---

## 2. MQA：砍掉头的激进方案

### 2.1 核心思想：所有头共享一组 KV

Multi-Query Attention（MQA, Shazeer 2019）的出发点非常直接：既然缓存随头数 $n_h$ 线性膨胀，那就**让全部 query 头共享同一组 K 和 V**：

$$
\mathbf{q}_{t,i} = W_i^Q \mathbf{h}_t \ (\text{每头独立}), \quad \mathbf{k}_t = W^K \mathbf{h}_t, \quad \mathbf{v}_t = W^V \mathbf{h}_t \ (\text{全局唯一})
$$

$$
\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^\top \mathbf{k}_j}{\sqrt{d_h}}\right) \mathbf{v}_j
$$

query 仍然保留 $n_h$ 个头（模型容量主要靠 query 侧维持），但 key 和 value 各只有**一份**。

**缓存变化**：

$$
\boxed{\text{KV Cache / token} = 2 d_h l}
$$

用同样的 V2 规模配置测算：$2 \times 128 \times 60 = 15{,}360$ 元素/token，是 MHA（约 197 万元素）的 **1/128**。

### 2.2 为什么快

- **缓存读取量 ÷$n_h$**：每步 decode 的 KV 读取从 $2 t d_h n_h$ 降为 $2 t d_h$，带宽压力消失两个数量级
- **batch size 上限大增**：缓存省下的显存可以全部换成并发序列数，吞吐成倍提升
- Shazeer 在原论文（encoder-decoder 场景）报告增量解码提速最高达一个数量级

### 2.3 质量代价：单组 KV 成为信息瓶颈

MQA 的问题出在表达容量上。MHA 的 $n_h$ 组 K/V 允许不同的头**同时**从不同角度索引历史——一个头追踪指代关系，另一个头追踪数字，第三个头追踪实体名称。MQA 把全部历史信息强行压进**单组** $d_h$ 维的 K/V 表示：

1. **检索视角塌缩**：所有 query 头面对的是同一份"记忆索引"，头间行为趋于同质化，失去分工
2. **信息容量受限**：整段上下文无论多长，都要编码进固定的单组 K/V 序列中，长上下文的细节保真度下降

DeepSeek-V2 论文附录 D.1 的 7B 消融实验（1.33T tokens 训练，参数量对齐）量化了这一代价：

| Benchmark | MQA (7.1B) | GQA-8 (6.9B) | MHA (6.9B) |
| --- | --- | --- | --- |
| BBH (EM, 3-shot) | 33.2 | 35.6 | **37.0** |
| MMLU (Acc, 5-shot) | 37.9 | 41.2 | **45.2** |
| C-Eval (Acc, 5-shot) | 30.0 | 37.7 | **42.9** |
| CMMLU (Acc, 5-shot) | 34.6 | 38.4 | **43.5** |

在 MMLU 上 MQA 落后 MHA 达 7.3 分——缓存省了 64 倍，但智能水平的损失不可接受。

> **演进动机**　"全共享"太激进。能否只共享一部分，在缓存与质量之间找一个平滑的折中点？

---

## 3. GQA：质量与效率的折中

### 3.1 分组共享

Grouped-Query Attention（GQA, Ainslie et al. 2023）将 $n_h$ 个 query 头分成 $n_g$ 组，每组内的 query 头共享一组 K/V：

$$
\mathbf{q}_{t,i} = W_i^Q \mathbf{h}_t, \quad \mathbf{k}_{t,g(i)} = W_{g(i)}^K \mathbf{h}_t, \quad \mathbf{v}_{t,g(i)} = W_{g(i)}^V \mathbf{h}_t
$$

其中 $g(i) = \lfloor i \cdot n_g / n_h \rfloor$ 是头 $i$ 所属的组。缓存为：

$$
\boxed{\text{KV Cache / token} = 2 n_g d_h l}
$$

### 3.2 MHA 与 MQA 都是 GQA 的特例

GQA 的价值在于它给出了一个**单参数插值族**：

$$
\underbrace{\text{MQA}}_{n_g = 1} \;\xleftarrow{\;n_g\ \text{增大}\;}\; \underbrace{\text{GQA-}n_g}_{1 < n_g < n_h} \;\xrightarrow{\;n_g = n_h\;}\; \underbrace{\text{MHA}}_{n_g = n_h}
$$

- $n_g = 1$：退化为 MQA
- $n_g = n_h$：退化为 MHA
- 中间值：缓存与质量的连续权衡。工业界常用 GQA-8（如 Llama-2/3-70B 的 64 query 头配 8 KV 头）

### 3.3 Uptraining：从 MHA 检查点低成本转换

GQA 论文的另一个贡献是**转换协议**：已经训练好的 MHA 模型不必从头重训。

1. **结构转换**：组内各 K 头取平均（mean-pooling）得到该组的共享 K，V 同理
2. **继续预训练**：仅用原预训练量约 5% 的计算量微调，质量即可恢复大半

这使得 GQA 可以"搭 MHA 的便车"，是它能迅速普及（Llama、Mistral、Qwen 等）的重要原因。

### 3.4 质量-效率权衡的真相

回看 3.1 节的消融表：GQA-8 在 MMLU 上仍落后 MHA 4.0 分，C-Eval 落后 5.2 分。这揭示了"砍头"路线的根本局限：

| 路线 | 缓存 | 质量 | 交换比 |
| --- | --- | --- | --- |
| MHA → GQA-8 | ÷16 | MMLU −4.0 | 缓存省得多，质量掉得少，但**仍然掉** |
| GQA-8 → MQA | ÷8 | MMLU −3.3 | 越往极限压，单位缓存节省的质量代价越大 |

**问题的本质**：无论 MQA 还是 GQA，都是在**减少存储的 K/V 向量份数**（结构共享），被删掉的信息通道无法恢复。缓存与质量的交换曲线太陡峭。

> **演进动机**　"砍头"路线撞墙了。换一种思路——**不减少头数，而是压缩每个 token 的 K/V 表示本身**，能否用远小于 $2 n_h d_h$ 的存储，承载全部 $n_h$ 个头的 K/V 信息？

---

## 4. MLA：低秩联合压缩

Multi-head Latent Attention（MLA）由 DeepSeek-V2（2024 年 5 月）提出，是 KV Cache 压缩路线的分水岭。它的两个支柱是：**低秩联合压缩**（把全部头的 K/V 压进一个共享潜向量）与**矩阵吸收**（推理时无需把 K/V 解压出来）；再加上一个关键补丁：**解耦 RoPE**。

### 4.1 思路转变：从"减少头数"到"压缩维度"

MQA/GQA 与 MLA 的分歧在于对"KV Cache 里到底有什么"的不同回答：

- **MQA/GQA 视角**：缓存 = $n_g$ 份独立的 K/V 向量 → 压缩就是**减少份数**（结构共享）
- **MLA 视角**：缓存 = 每个 token 的一段**信息** → 压缩就是对信息本身做**有损编码**

关键洞察：一个 token 的全部 K/V 拼接起来是 $2 n_h d_h = 32{,}768$ 维的向量，但其中存在大量冗余——不同头的 K/V 源自同一个 $\mathbf{h}_t$，有效自由度远小于 $32{,}768$。因此可以用一个 $d_c = 512$ 维的**潜向量**（latent vector）近似承载，压缩比高达 32 倍。

> **类比**　MQA 是"把图书馆的副本扔掉只留一本"；MLA 是"每本书都保留，但全部转成 zip 存放，读的时候在内存里解压"。前者丢失副本里的独立批注，后者几乎无损。

### 4.2 KV 联合压缩公式

MLA 的第一步是把 $\mathbf{h}_t$ 下投影为压缩潜向量，需要 K/V 时再上投影解压：

$$
\boxed{\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c}}
$$

$$
\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}
$$

各矩阵的维度（以 DeepSeek-V2 配置为例，$d = 5120$, $n_h d_h = 16384$, $d_c = 512$）：

| 符号 | 维度 | 作用 |
| --- | --- | --- |
| $W^{DKV}$ | $d_c \times d = 512 \times 5120$ | 下投影：隐藏态 → 潜向量 |
| $W^{UK}$ | $d_h n_h \times d_c = 16384 \times 512$ | 上投影：潜向量 → 全部头的 K |
| $W^{UV}$ | $d_h n_h \times d_c = 16384 \times 512$ | 上投影：潜向量 → 全部头的 V |

**"联合"（joint）的含义**：K 和 V 从**同一个**潜向量 $\mathbf{c}_t^{KV}$ 解压，而不是各自拥有一个潜向量。这迫使模型把"检索特征"（K 的职责）与"内容特征"（V 的职责）编码进共享的低秩子空间。

**缓存量**：推理时只需缓存 $\mathbf{c}_t^{KV}$，每 token 每层仅 $d_c = 512$ 个元素（先不考虑 RoPE，见 4.4 节）：

$$
\text{KV Cache / token} = d_c \, l \quad (\ll 2 n_h d_h l)
$$

注意一个反直觉的事实：缓存的不是 K，也不是 V，而是"**可以推出 K 和 V 的东西**"。这是 MLA 与一切"砍头"方案的本质区别。

### 4.3 矩阵吸收：推理时根本不需要"算出" K 和 V

如果只做到 4.2，推理时每步还要把 $\mathbf{k}^C, \mathbf{v}^C$ 从潜向量解压出来才能算注意力，计算量反而比 MHA 大。MLA 的第二个支柱——**矩阵吸收**（weight absorption）——让解压步骤在推理时**完全消失**。

**（1）QK 侧：$W^{UK}$ 被吸收进 query 通路**

先给 query 也配上低秩压缩（详见 4.5 节）：$\mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q$，$\mathbf{c}_t^Q \in \mathbb{R}^{d_c'}$。第 $i$ 个头在位置 $t$ 与 $j$ 之间的注意力分数为：

$$
s_{t,j}^C = (\mathbf{q}_{t,i}^C)^\top \mathbf{k}_{j,i}^C = (W_i^{UQ} \mathbf{c}_t^Q)^\top (W_i^{UK} \mathbf{c}_j^{KV}) = (\mathbf{c}_t^Q)^\top \underbrace{(W_i^{UQ})^\top W_i^{UK}}_{\text{两个权重相乘，可离线合并}} \mathbf{c}_j^{KV}
$$

定义离线合并后的矩阵与"潜空间 query"：

$$
\tilde{W}_i = (W_i^{UQ})^\top W_i^{UK} \in \mathbb{R}^{d_c' \times d_c}, \quad \tilde{\mathbf{q}}_{t,i} = \tilde{W}_i^\top \mathbf{c}_t^Q \in \mathbb{R}^{d_c}
$$

则分数直接在潜空间计算：

$$
\boxed{s_{t,j}^C = \tilde{\mathbf{q}}_{t,i}^\top \mathbf{c}_j^{KV}}
$$

**全程没有物化** $\mathbf{k}^C$。$\tilde{W}_i$ 在部署前一次性算好，推理时零额外成本。

**（2）AV 侧：$W^{UV}$ 被吸收进输出投影**

注意力输出 $\mathbf{o}_{t,i} = \sum_j a_{t,j} \mathbf{v}_{j,i}^C$，代入 $\mathbf{v}_{j,i}^C = W_i^{UV} \mathbf{c}_j^{KV}$：

$$
\mathbf{o}_{t,i} = \sum_j a_{t,j} W_i^{UV} \mathbf{c}_j^{KV} = W_i^{UV} \underbrace{\sum_j a_{t,j} \mathbf{c}_j^{KV}}_{\tilde{\mathbf{o}}_{t,i} \in \mathbb{R}^{d_c}}
$$

最终输出 $\mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \ldots; \mathbf{o}_{t,n_h}]$ 中，$W^O$ 的第 $i$ 个列块 $W_i^O$ 与 $W_i^{UV}$ 相邻相乘，同样可以离线合并为 $\tilde{W}_i^O = W_i^O W_i^{UV}$，于是：

$$
\boxed{\mathbf{u}_t = \sum_i \tilde{W}_i^O \, \tilde{\mathbf{o}}_{t,i}}
$$

同样**全程没有物化** $\mathbf{v}^C$。

**（3）吸收后的推理形态：潜空间上的 MQA**

把两步合起来看，吸收后的 decode 注意力长这样：

$$
\text{分数} = \tilde{\mathbf{q}}_{t,i}^\top \mathbf{c}_j^{KV}, \quad \text{输出} = \text{Softmax}(\text{分数}) \text{ 加权 } \mathbf{c}_j^{KV}
$$

所有头读写**同一份**缓存 $\{\mathbf{c}_j^{KV}\}$（这正是它被允许小的原因），但每个头有自己独立的潜空间 query $\tilde{\mathbf{q}}_{t,i}$——**形态上等价于 head_dim = $d_c + d_h^R = 576$ 的 MQA**，可以复用高度优化的 MQA decode kernel；而表达能力上每个头的"查询接口"宽达 512 维，远非真正的 MQA 可比。

| 维度 | MHA decode | MLA 吸收后 decode |
| --- | --- | --- |
| 每 token 每层缓存读取 | $2 n_h d_h = 32{,}768$ 元素 | $d_c = 512$ 元素 |
| 注意力计算空间 | $n_h$ 个 $d_h$ 维子空间 | $n_h$ 个 $d_c$ 维潜空间 |
| Kernel 形态 | MHA/GQA kernel | 类 MQA kernel |

### 4.4 RoPE 危机与解耦 RoPE

**（1）RoPE 为什么破坏吸收**

RoPE 给 query 和 key 施加位置相关的旋转 $R_t$（分块对角正交矩阵），利用性质 $R_t^\top R_j = R_{t-j}$ 使注意力分数只依赖相对位置。如果对解压后的 K 施加 RoPE：

$$
\mathbf{k}_{j,i} = R_j W_i^{UK} \mathbf{c}_j^{KV}, \quad \mathbf{q}_{t,i} = R_t W_i^{UQ} \mathbf{c}_t^Q
$$

则分数变为：

$$
s_{t,j} = (\mathbf{c}_t^Q)^\top (W_i^{UQ})^\top \underbrace{R_t^\top R_j}_{R_{t-j}} W_i^{UK} \mathbf{c}_j^{KV}
$$

问题一目了然：**位置相关的矩阵 $R_{t-j}$ 夹在 $(W_i^{UQ})^\top$ 与 $W_i^{UK}$ 之间**。矩阵乘法不满足交换律，无法把 $R_{t-j}$ 挪走，也就无法离线合并两个权重。后果是灾难性的——每生成一个 token，都要为**全部前缀**重新计算带新相对位置的 K，推理效率崩塌。这就是论文所说的 "RoPE is incompatible with low-rank KV compression"。

**（2）解耦方案：内容与位置分道走**

DeepSeek 的解法是把注意力分数拆成两个通道——**内容通道走潜空间（不带 RoPE），位置通道走独立的小维度（带 RoPE）**：

$$
\mathbf{q}_t^R = \text{RoPE}(W^{QR} \mathbf{c}_t^Q) \in \mathbb{R}^{d_h^R n_h} \ (\text{逐头}), \quad \mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \in \mathbb{R}^{d_h^R} \ (\text{全头共享})
$$

$$
\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R], \quad \mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R]
$$

拼接后内积自然分解为两项：

$$
\mathbf{q}_{t,i}^\top \mathbf{k}_{j,i} = \underbrace{(\mathbf{q}_{t,i}^C)^\top \mathbf{k}_{j,i}^C}_{\text{内容/语义分：可吸收}} + \underbrace{(\mathbf{q}_{t,i}^R)^\top \mathbf{k}_j^R}_{\text{位置分：RoPE}}
$$

Softmax 的缩放因子相应变为 $\sqrt{d_h + d_h^R}$。几个设计细节值得注意：

- **$\mathbf{k}^R$ 全头共享**：位置通道只有一份（MQA 式），因此缓存只需额外增加 $d_h^R = 64$ 维，而非 $n_h \times 64$
- **$\mathbf{k}^R$ 从 $\mathbf{h}_t$ 直接投影**：不经过低秩瓶颈。位置信息必须高保真，且 $\mathbf{k}^R$ 本身就是要被缓存的对象，从 $\mathbf{h}_t$ 一步投影路径最短
- **$\mathbf{q}^R$ 从 $\mathbf{c}_t^Q$ 投影**：query 不缓存，低秩路径无妨
- **长上下文扩展只需作用于 $\mathbf{k}^R$**：DeepSeek-V2 把 YaRN 只施加在解耦共享 key 上，因为全部位置信息都由它承载

**（3）最终缓存量**

$$
\boxed{\text{KV Cache / token} = (d_c + d_h^R)\, l = (512 + 64) \times 60 = 34{,}560 \text{ 元素}}
$$

用 $d_h$ 表示：$d_c = 4 d_h$，$d_h^R = \frac{1}{2} d_h$，合计 $\frac{9}{2} d_h l$——**恰好等于 GQA-2.25 组的缓存量**。

### 4.5 Query 侧压缩

query 不需要缓存，压缩它省不了 KV Cache，但 DeepSeek 仍对 query 做了对称的低秩处理：

$$
\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \in \mathbb{R}^{d_c'}, \quad \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q
$$

其中 $d_c' = 1536$（$d_c' \ll d_h n_h = 16384$）。动机有两个：

1. **减少训练激活内存**：反向传播要保存每层的 $\mathbf{q}$，16384 维 → 1536 维的瓶颈显著降低激活占用
2. **统一潜空间接口**：Q 侧也有了"下投影-上投影"结构，4.3 节的吸收合并 $\tilde{W}_i = (W_i^{UQ})^\top W_i^{UK}$ 才成为 $d_c' \times d_c$ 的小矩阵运算

### 4.6 MLA 完整计算流程

把全部组件串起来（对应 DeepSeek-V2 论文附录 C 的完整公式）：

**写入侧（每个新 token 执行一次）**：

$$
\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \;(\to \text{缓存}), \quad \mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \;(\to \text{缓存})
$$

**读取侧（训练/物化形式）**：

$$
\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t, \quad \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q, \quad \mathbf{q}_t^R = \text{RoPE}(W^{QR} \mathbf{c}_t^Q)
$$

$$
\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}
$$

$$
\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R], \quad \mathbf{k}_{j,i} = [\mathbf{k}_{j,i}^C; \mathbf{k}_j^R]
$$

$$
\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^\top \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}}\right) \mathbf{v}_{j,i}^C, \quad \mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \ldots; \mathbf{o}_{t,n_h}]
$$

**读取侧（推理/吸收形式）**：见 4.3 节，$\mathbf{k}^C, \mathbf{v}^C$ 永不物化，注意力直接在 $\{\mathbf{c}_j^{KV}\}$ 上进行。

**参数量对比**（按 V2 配置估算，$d = 5120$, $n_h d_h = 16384$, $d_c = 512$, $d_c' = 1536$, $d_h^R = 64$）：

| 组件 | MHA | MLA |
| --- | --- | --- |
| Q 通路 | $W^Q$：$16384 \times 5120 = 83.9$M | $W^{DQ} + W^{UQ} + W^{QR}$：$7.9 + 25.2 + 12.6 = 45.7$M |
| KV 通路 | $W^K, W^V$：$2 \times 83.9 = 167.8$M | $W^{DKV} + W^{UK} + W^{UV} + W^{KR}$：$2.6 + 8.4 + 8.4 + 0.3 = 19.7$M |
| 输出 | $W^O$：83.9M | $W^O$：83.9M |
| **合计** | **≈ 335.5M** | **≈ 149.3M** |

注意：**MLA 的注意力参数量反而只有 MHA 的 44%**——低秩分解不仅是缓存压缩，也是参数压缩。省下的参数预算在 DeepSeek-V2 中被重新分配给 MoE 与层数。

### 4.7 KV Cache 对比与部署收益

**论文 Table 1 的四种机制对比**（每 token 缓存元素数）：

| 机制 | KV Cache / Token | 等效 GQA 组数 | 能力 |
| --- | --- | --- | --- |
| MHA | $2 n_h d_h l$ | $n_h$（128） | 强 |
| GQA | $2 n_g d_h l$ | $n_g$ | 中 |
| MQA | $2 d_h l$ | 1 | 弱 |
| **MLA** | $(d_c + d_h^R) l \approx \frac{9}{2} d_h l$ | **2.25** | **更强** |

**DeepSeek-V2 实际数字**：

| 项目 | MHA（同规模假想） | MLA（实际） | 压缩比 |
| --- | --- | --- | --- |
| 单 token 全模型缓存 | $1{,}966{,}080$ 元素 | $34{,}560$ 元素 | **≈ 57×** |
| 单 token 体积（BF16） | ≈ 3.9 MB | ≈ 69 KB | ≈ 57× |
| 128K 上下文单序列 | ≈ 515 GB | **≈ 9 GB** | 单节点从放不下到从容部署 |

**官方部署口径**（对比 DeepSeek 67B 的实际服务）：KV Cache 减少 **93.3%**，单节点 8×H800 生成吞吐超过 50K tokens/s，是 DeepSeek 67B 最大生成吞吐的 **5.76 倍**（该数字还包含 FP8 权重转换与 6-bit KV Cache 量化等部署优化的贡献）。

> **教学要点**　57× 是"同规模 MHA vs MLA"的纯架构对比；93.3% 是"DeepSeek 67B 实际部署 vs DeepSeek-V2 实际部署"的系统级对比，两者口径不同，引用时注意区分。

### 4.8 为什么压缩后反而更强

MLA 最反直觉的性质是：**缓存只有 GQA-2.25 的水平，质量却反超满配 MHA**。DeepSeek-V2 附录 D.2 的消融（参数量对齐的 MoE 模型）给出了硬数据：

| 规模 | 指标 | MHA | MLA | MLA KV Cache 占比 |
| --- | --- | --- | --- | --- |
| Small MoE（≈16B，1.33T tokens） | BBH / MMLU / CMMLU | 37.9 / 48.7 / 52.3 | **39.0 / 50.0 / 53.4** | 14%（15.6K vs 110.6K 元素） |
| Large MoE（≈250B，420B tokens） | BBH / MMLU / CMMLU | 46.6 / 57.5 / 60.7 | **50.7 / 59.0 / 62.5** | 4%（34.6K vs 860.2K 元素） |

论文只给出结论（"equipped with low-rank key-value joint compression, MLA achieves better performance than MHA"），机制层面没有完全解释。以下是几种主流解读（**属于讨论与猜想，非论文结论**）：

1. **联合压缩的信息共享**：K 和 V 共用同一潜空间基底，逼迫"检索特征"与"内容特征"对齐到共享低秩子空间，消除 MHA 中 K/V 各自为政的冗余表示。

2. **低秩正则化**：瓶颈结构约束了注意力映射的有效秩，类似 LoRA 的正则效应——参数更少的假设空间在小数据上反而泛化更好。4.6 节的参数对比支持这一点：MLA 注意力参数仅为 MHA 的 44%。

3. **参数预算再分配**：对齐总参数量的消融里，MLA 省下的参数被投给了更多层/更大 MoE，质量收益可能部分来自预算的重新配置而非注意力本身。

4. **更宽的查询-记忆接口**：吸收形式下每个头的有效查询维度是 $d_c = 512$，是 MHA 单头 $d_h = 128$ 的 4 倍；每次"查询"能与记忆发生更丰富的交互。

> **演进动机之外的一问**　GQA/MQA 的降质证明"信息瓶颈不能开在头数上"；MLA 的反超说明"瓶颈开在 token 表示的秩上"是安全的——因为文本的 K/V 表示本就低秩。**压缩的对象选对了，瓶颈就不是瓶颈。**

### 4.9 工程实现：训练与推理的不对称

MLA 有一个容易踩坑的工程特性：**训练形态与推理形态不对称**。

| 阶段 | 计算形态 | 原因 |
| --- | --- | --- |
| 训练 | **物化形式**：解压出完整 $\mathbf{k}^C, \mathbf{v}^C$，走 FlashAttention（V2 论文：基于改进版 FlashAttention-2 优化） | 训练是批量矩阵乘，$\mathcal{O}(L^2)$ 注意力主导；上投影的额外 FLOPs 占比小；物化形式可直接复用成熟 kernel |
| 推理 Prefill | 通常也走物化形式 | 同训练，长序列并行计算 |
| 推理 Decode | **吸收形式**：注意力在潜空间进行，等价 head_dim=576 的类 MQA kernel | 逐 token 时带宽是唯一瓶颈，缓存读取必须最小 |

**上投影带来的额外训练 FLOPs**（本仓库 slime 的 FLOPs 统计实现，见 [flops_utils.py](../../slime/slime/utils/flops_utils.py)）：

$$
\text{QKV 投影} = 2s\, d_c'\big(d + n_h(d_h + d_h^R)\big) + 2s\Big(d_c\big(d + n_h(d_h + d_v)\big) + d\, d_h^R\Big)
$$

对比 MHA 的 $2s \cdot d \cdot 3 n_h d_h$，MLA 的投影 FLOPs 略高，但注意力主体不变，整体训练开销增加有限。

**权重吸收是离线操作**：部署时把 $(W_i^{UQ})^\top W_i^{UK}$ 与 $W_i^O W_i^{UV}$ 预先合并存好，推理二进制里根本不存在 $W^{UK}, W^{UV}$ 的独立拷贝。

**本仓库的实现参考**：slime 的 GLM-5 插件实现了完整的 MLA（并叠加了 DSA 稀疏化，见第 5.5 节），见 [glm5.py](../../slime/slime_plugins/models/glm5/glm5.py) 中的 `DSAMLASelfAttention`：

| 论文组件 | 代码对应 |
| --- | --- |
| $W^{DQ}$ / $W^{UQ}$ | `linear_q_down_proj` / `linear_q_up_proj` |
| $W^{DKV}$（输出含 $\mathbf{k}^R$ 的 $d_h^R$ 维） | `linear_kv_down_proj`（输出维度 = `kv_lora_rank + qk_pos_emb_head_dim`） |
| $W^{UK}, W^{UV}$ | `linear_kv_up_proj`（输出 $n_h \times (d_h + d_v)$） |
| 潜空间归一化 | `q_layernorm` / `kv_layernorm`（对 $\mathbf{c}^Q, \mathbf{c}^{KV}$ 做 RMSNorm——这是 DeepSeek-V2 论文之后工程实践沉淀出的稳定化技巧） |
| 矩阵吸收 | `torch.einsum("thd,hdm->thm", q_no_pe, w_kc)`：query 直接与 $W^{UK}$ 权重收缩，得到潜空间 query |
| 解耦 RoPE | `fuse_rope` 分别施加于 `q_pos_emb` / `k_pos_emb`，再 `cat` 回 no-pe 部分 |

一个值得注意的细节：slime 的实现中**训练也直接使用吸收形式**（`key = cat([kv_compressed, k_pos_emb])`，key 就是潜向量本身），因为它配套的是 DSA 稀疏注意力——top-k 选择在潜空间打分，注意力也在潜空间执行，物化形式反而没有必要。

---

## 5. 演进对比与总结

### 5.1 完整对比表

| 方法 | 缓存内容 | KV Cache / Token | 等效 GQA 组数 | 质量 | 代表模型 |
| --- | --- | --- | --- | --- | --- |
| MHA | 每头独立 K、V | $2 n_h d_h l$ | $n_h$ | 强（基线） | GPT-3、DeepSeek 67B |
| MQA | 全局 1 组 K、V | $2 d_h l$ | 1 | 弱 | PaLM、Falcon |
| GQA | $n_g$ 组 K、V | $2 n_g d_h l$ | $n_g$（常取 8） | 中强 | Llama-2/3-70B、Qwen、Mistral |
| **MLA** | 潜向量 $\mathbf{c}^{KV}$ + 共享 $\mathbf{k}^R$ | $(d_c + d_h^R) l$ | **2.25** | **反超 MHA** | DeepSeek-V2/V3、Kimi K2、GLM-5 |

### 5.2 演进逻辑

```
MHA
 │ ✅ 每头独立 KV，检索能力最强
 │ ❌ KV Cache = 2·n_h·d_h·l，随头数线性膨胀
 │    （V2 规模：~2M 元素/token，128K 上下文 ~515 GB）
 │
 │  所有 query 头共享一组 KV → 缓存 ÷ n_h
 ▼
MQA
 │ ✅ 缓存最小 (2·d_h·l)，decode 带宽压力消失
 │ ❌ 单组 KV 成信息瓶颈 → 7B 消融 MMLU −7.3
 │
 │  分组共享 n_g 组 → 缓存与质量平滑插值
 ▼
GQA
 │ ✅ 单参数插值族 (GQA-1=MQA, GQA-n_h=MHA)
 │ ✅ uptraining：mean-pool + 5% 计算量即可从 MHA 转换
 │ ❌ 本质仍是"砍头"：GQA-8 仍落后 MHA（MMLU −4.0）
 │
 │  换思路：不砍头，压缩表示本身
 │  → 低秩联合压缩 + 矩阵吸收 + 解耦 RoPE
 ▼
MLA
 │ ✅ 缓存 = GQA-2.25（约 9/2·d_h·l，同规模 MHA 的 1/57）
 │ ✅ 质量反超 MHA（16B/250B 两档消融一致成立）
 │ ✅ 权重离线吸收 → decode 等价 head_dim=576 的类 MQA
 │ ✅ 参数量反降至 MHA 的 44%，预算可再分配
 │ ⚠ RoPE 必须解耦；训练需物化解压（训练/推理不对称）
```

### 5.3 核心设计权衡

| 设计维度 | MLA 的选择 | 理由 |
| --- | --- | --- |
| 压缩对象 | token 的 K/V **表示**（而非头数） | K/V 表示天然低秩，瓶颈开在这里无损 |
| 压缩结构 | K、V **联合**压缩（共享潜向量） | 检索与内容特征共享基底，消除冗余 |
| 推理形态 | 矩阵吸收，注意力在潜空间执行 | 解压计算离线合并进 query/output 通路，decode 零物化 |
| 位置编码 | 解耦 RoPE：内容 NoPE + 独立 64 维共享 $\mathbf{k}^R$ | RoPE 矩阵夹在权重间阻断吸收，位置通道必须独立 |
| Query 处理 | 同样低秩压缩（$d_c' = 1536$） | 省训练激活内存 + 统一潜空间接口 |
| 缓存构成 | $d_c$（潜向量）+ $d_h^R$（共享位置 key） | 位置 key 全头共享，只加 64 维而非 $64 \times n_h$ |

### 5.4 性能数据

**MHA / GQA / MQA 7B 消融**（DeepSeek-V2 附录 D.1，1.33T tokens，参数量对齐）：

| Benchmark | MQA | GQA-8 | MHA |
| --- | --- | --- | --- |
| BBH (EM) | 33.2 | 35.6 | **37.0** |
| MMLU (Acc) | 37.9 | 41.2 | **45.2** |
| C-Eval (Acc) | 30.0 | 37.7 | **42.9** |
| CMMLU (Acc) | 34.6 | 38.4 | **43.5** |

**MLA vs MHA MoE 消融**（DeepSeek-V2 附录 D.2）：

| 规模 | 指标 | MHA | MLA | KV Cache 缩减 |
| --- | --- | --- | --- | --- |
| Small MoE（≈16B 总参，1.33T tokens） | BBH / MMLU / CMMLU | 37.9 / 48.7 / 52.3 | **39.0 / 50.0 / 53.4** | 110.6K → 15.6K 元素（86%） |
| Large MoE（≈250B 总参，420B tokens） | BBH / MMLU / CMMLU | 46.6 / 57.5 / 60.7 | **50.7 / 59.0 / 62.5** | 860.2K → 34.6K 元素（96%） |

**系统级收益**（DeepSeek-V2 vs DeepSeek 67B 实际部署）：KV Cache −93.3%，训练成本 −42.5%，最大生成吞吐 ×5.76。

### 5.5 后续演进与生态

**（1）DeepSeek-V3（2024.12）：MLA 的规模化验证**

DeepSeek-V3 沿用完全相同的 MLA 配置（$d_c = 512$, $d_c' = 1536$, $d_h^R = 64$, $n_h = 128$），将模型规模推至 671B 总参 / 37B 激活，14.8T tokens 训练。MLA 从"V2 的创新点"变成"DeepSeek 系的标准件"，证明了其在超大规模下的稳定性。

**（2）DSA（2025）：MLA × 稀疏注意力**

DeepSeek-V3.2 引入 DeepSeek Sparse Attention：在 MLA 的潜空间之上加一个轻量 **lightning indexer**，为每个 query 从全部历史 token 中选出 top-k 个再做注意力，把 $\mathcal{O}(L^2)$ 的注意力计算降为 $\mathcal{O}(Lk)$，进一步降低长上下文成本。关键协同在于：**top-k 打分与注意力都在 512 维潜空间进行**，indexer 的打分成本也因此极低。本仓库 slime 的 [GLM-5 插件](../../slime/slime_plugins/models/glm5/glm5.py) 实现了这一路线（`DSAMLASelfAttention` + `IndexerFunction`）。

**（3）与线性注意力路线的交汇：Kimi Linear 混合架构**

MLA 也成为了另一条技术路线的"精确检索组件"：Kimi Linear 以 3:1 的比例交替堆叠 KDA（线性注意力）与 MLA 层——KDA 层用固定大小状态承担绝大部分序列建模，MLA 层周期性提供无损的全局精确检索。详见姊妹篇 [从 Full Attention 到 Linear Attention 到 GDN 再到 KDA](./linear-attention-to-gdn-to-kda.md)。

**（4）两条 KV Cache 路线的哲学对比**

| 维度 | 线性注意力路线（GDN/KDA） | MLA 路线 |
| --- | --- | --- |
| 哲学 | **压缩历史**：把过去压进固定大小状态 | **压缩表示**：保留全部 token，压缩每 token 的存储 |
| 状态/缓存 | $\mathcal{O}(d^2)$ 固定，与 $L$ 无关 | $\mathcal{O}(L (d_c + d_h^R))$，随 $L$ 线性但系数小 57 倍 |
| 遗忘 | 必须遗忘（有损，靠门控管理） | 不遗忘（潜向量近似无损重构 K/V） |
| 检索精度 | 受限（记忆碰撞是固有上界） | 保持 softmax 精确检索 |
| 超长上下文 | 内存恒定，理论上无界 | 线性增长，但 1M 级已工程可行 |
| 交汇点 | — | Kimi Linear 3:1 混合架构（KDA + MLA） |

**（5）其他变体**

- **TransMLA**：将已训练好的 GQA 模型结构等价转换为 MLA，复用存量检查点
- **量化友好性**：潜向量 $d_c$ 维共享表示比分散的逐头 K/V 更适合低比特量化，DeepSeek-V2 部署即采用 6-bit KV Cache 量化

### 5.6 一句话总结

> MQA/GQA 在**头数**上做减法，缓存与质量线性交换；MLA 在**表示的秩**上做减法，配合矩阵吸收把"解压"折叠进相邻权重、用解耦 RoPE 绕开位置编码的阻断——最终以 GQA-2.25 的缓存实现了超越 MHA 的质量。**压缩的对象选对了，瓶颈就不是瓶颈。**

---

## 参考文献

- Vaswani et al. (2017). *Attention Is All You Need.* (Transformer / MHA) [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Shazeer (2019). *Fast Transformer Decoding: One Write-Head is All You Need.* (MQA) [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* (GQA / Uptraining) [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Su et al. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* (RoPE) [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Pope et al. (2023). *Efficiently Scaling Transformer Inference.* (Decode 的内存带宽分析) [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
- DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* (MLA) [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report.* [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek-AI (2025). *DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention.* (DSA)
- Peng et al. (2023). *YaRN: Efficient Context Window Extension of Large Language Models.* (YaRN) [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* (KDA + MLA 混合架构) [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
