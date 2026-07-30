**TECHNICAL REFERENCE · 2026**

# 从 Full Attention 到 Linear Attention 到 GDN 再到 KDA：算法演进详解

*Full (Softmax) Attention → Linear Attention → DeltaNet → Gated DeltaNet → Kimi Delta Attention*

**面向教学的逐步推导 · 公式细节 · 直觉建立 · 演进逻辑**

**适用读者**

希望深入理解高效注意力机制演进脉络的研究人员与工程师

版本基线：2026 年 8 月

---

## 执行摘要

> **一句话结论**　Full Attention 用 softmax 实现精确检索但推理 $\mathcal{O}(L^2)$；Linear Attention 移除 softmax 降为 $\mathcal{O}(Ld^2)$ 但只能加不能擦；DeltaNet 引入 Delta 规则实现"先擦后写"但无全局遗忘；GDN 统一标量遗忘门与 Delta 规则；KDA 将标量门升级为逐通道对角门并约束 DPLR 结构实现 2× 算子加速。

| 方法 | 核心公式 | 推理复杂度 | 关键突破 | 关键缺陷 |
| --- | --- | --- | --- | --- |
| Full Attention | $\mathbf{o}_t = \sum_j \text{softmax}(\mathbf{q}_t^\top \mathbf{k}_j) \mathbf{v}_j$ | $\mathcal{O}(L^2 d)$ | 精确检索 | KV Cache 线性增长 |
| Linear Attention | $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | $\mathcal{O}(Ld^2)$ | 固定大小状态 | 无法遗忘，记忆碰撞 |
| DeltaNet | $\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$ | $\mathcal{O}(Ld^2)$ | 先擦后写，纠正误差 | 无全局遗忘 |
| GDN | $\mathbf{S}_t = \mathbf{S}_{t-1}(\alpha_t(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$ | $\mathcal{O}(Ld^2)$ | 统一遗忘与精准更新 | 门控粒度太粗 |
| KDA | $\mathbf{S}_t = (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)\text{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top$ | $\mathcal{O}(Ld^2)$ | 逐通道门控 + 2× 加速 | 需混合全注意力 |

### 阅读导航

| 章节 | 主题 | 教学重点 |
| --- | --- | --- |
| 01 | Full Attention | 为什么需要注意力？softmax 的作用是什么？瓶颈在哪？ |
| 02 | Linear Attention | 如何移除 softmax？结合律如何改变一切？为什么记忆会碰撞？ |
| 03 | DeltaNet | Delta 规则从何而来？如何从在线学习推导？如何分块并行？ |
| 04 | GDN | 遗忘门与 Delta 规则如何统一？在线学习目标如何变化？ |
| 05 | KDA | 逐通道门控为什么更强？DPLR 约束如何加速？ |
| 06 | 总结 | 完整对比、演进逻辑与设计权衡 |

---

## 1. Full Attention：一切的起点

### 1.1 注意力机制要解决什么问题

在处理序列（如文本）时，模型需要一种机制来**根据当前 token 的需求，从历史信息中选择性地提取相关内容**。例如，在句子"The cat sat on the mat because **it** was tired"中，理解"it"需要回溯到"cat"。

Full Attention（标准 Softmax Attention）通过三个矩阵实现这一目标：

- **Query（查询）** $\mathbf{Q} \in \mathbb{R}^{L \times d}$：每个位置发出"我要找什么"的查询
- **Key（键）** $\mathbf{K} \in \mathbb{R}^{L \times d}$：每个位置提供"我有什么"的标签
- **Value（值）** $\mathbf{V} \in \mathbb{R}^{L \times d}$：每个位置提供"我的内容"的载荷

其中 $L$ 是序列长度，$d$ 是头维度。

### 1.2 计算公式：逐步拆解

**第一步：计算相似度矩阵** $\mathbf{Q}\mathbf{K}^\top$

$$
\mathbf{Q}\mathbf{K}^\top = \begin{pmatrix} \mathbf{q}_1^\top \mathbf{k}_1 & \mathbf{q}_1^\top \mathbf{k}_2 & \cdots & \mathbf{q}_1^\top \mathbf{k}_L \\ \mathbf{q}_2^\top \mathbf{k}_1 & \mathbf{q}_2^\top \mathbf{k}_2 & \cdots & \mathbf{q}_2^\top \mathbf{k}_L \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{q}_L^\top \mathbf{k}_1 & \mathbf{q}_L^\top \mathbf{k}_2 & \cdots & \mathbf{q}_L^\top \mathbf{k}_L \end{pmatrix} \in \mathbb{R}^{L \times L}
$$

第 $(t, j)$ 个元素 $\mathbf{q}_t^\top \mathbf{k}_j$ 表示位置 $t$ 的查询与位置 $j$ 的键之间的点积相似度。

**第二步：因果掩码** $\odot \mathbf{M}$

在自回归模型中，位置 $t$ 只能看到位置 $\leq t$ 的信息。因果掩码 $\mathbf{M}$ 是一个下三角矩阵：

$$
\mathbf{M}_{ij} = \begin{cases} 1 & \text{if } i \geq j \\ 0 & \text{if } i < j \end{cases}
$$

掩码后，$(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})_{ij} = 0$ 当 $i < j$（未来信息被屏蔽）。

**第三步：softmax 归一化**

$$
\text{softmax}(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})_{t,j} = \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j)}{\sum_{l=1}^{t} \exp(\mathbf{q}_t^\top \mathbf{k}_l)}
$$

softmax 的作用是**将原始相似度转换为概率分布**：每个位置的注意力权重非负且归一化为 1。这使得模型能进行**竞争性选择**——相似度最高的 key 获得最大权重。

**第四步：加权求和**

$$
\mathbf{O} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V} \in \mathbb{R}^{L \times d}
$$

每个输出 $\mathbf{o}_t$ 是所有历史 value 的加权和，权重由 softmax 决定。

**推理时的逐 token 形式**：

$$
\mathbf{o}_t = \sum_{j=1}^{t} \underbrace{\frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j)}{\sum_{l=1}^{t} \exp(\mathbf{q}_t^\top \mathbf{k}_l)}}_{\text{注意力权重 } a_{t,j}} \mathbf{v}_j
$$

### 1.3 为什么 softmax 带来精确检索

考虑一个极端情况：query $\mathbf{q}_t$ 与某个 key $\mathbf{k}_{j^*}$ 的相似度远大于其他 key。softmax 的指数放大效应会使得 $a_{t,j^*} \approx 1$，其他权重 $\approx 0$，从而 $\mathbf{o}_t \approx \mathbf{v}_{j^*}$。

**具体例子**：假设 $d = 3$，$\mathbf{q}_t = [1, 0, 0]$，三个 key 为 $\mathbf{k}_1 = [1, 0, 0]$, $\mathbf{k}_2 = [0, 1, 0]$, $\mathbf{k}_3 = [0, 0, 1]$。

$$
\mathbf{q}_t^\top \mathbf{k}_1 = 1, \quad \mathbf{q}_t^\top \mathbf{k}_2 = 0, \quad \mathbf{q}_t^\top \mathbf{k}_3 = 0
$$

$$
a_{t,1} = \frac{e^1}{e^1 + e^0 + e^0} = \frac{e}{e + 2} \approx 0.576, \quad a_{t,2} = a_{t,3} = \frac{1}{e + 2} \approx 0.212
$$

如果相似度差距更大，如 $\mathbf{q}_t^\top \mathbf{k}_1 = 5$，其他为 0：

$$
a_{t,1} = \frac{e^5}{e^5 + 2} \approx 0.993, \quad a_{t,2} = a_{t,3} \approx 0.003
$$

可见 softmax 的指数效应将微小的相似度差异放大为压倒性的权重分配，实现近似精确检索。

### 1.4 瓶颈分析

**训练复杂度**：需要计算 $L \times L$ 的注意力矩阵，复杂度 $\mathcal{O}(L^2 d)$。当 $L = 128\text{K}$ 时，该矩阵有约 $1.6 \times 10^{10}$ 个元素。

**推理复杂度**：生成第 $t$ 个 token 时，需要计算 $\mathbf{q}_t$ 与所有历史 $\mathbf{k}_j$ 的相似度，复杂度 $\mathcal{O}(t \cdot d)$。生成 $L$ 个 token 的总复杂度 $\mathcal{O}(L^2 d)$。

**KV Cache**：推理时需要缓存所有历史 key-value 对，空间复杂度 $\mathcal{O}(L \cdot d)$。以 $L = 1\text{M}$、$d = 128$、128 层、8 头为例，KV Cache 大小为 $1\text{M} \times 128 \times 128 \times 8 \times 2 \times 2\text{ bytes} \approx 64\text{GB}$（FP16）。

| 维度 | Full Attention |
| --- | --- |
| 训练时间复杂度 | $\mathcal{O}(L^2 d)$ |
| 推理时间复杂度（逐 token） | $\mathcal{O}(t \cdot d)$ |
| 推理空间复杂度 | $\mathcal{O}(L \cdot d)$ |
| 训练并行度 | 高（$L \times L$ 矩阵运算） |
| 检索精度 | 高（softmax 竞争性选择） |

> **核心问题**　能否设计一种机制，保留 key-value 关联记忆的能力，但将推理复杂度从 $\mathcal{O}(L^2)$ 降为线性甚至常数？

---

## 2. Linear Attention：移除 Softmax 的代价与收获

### 2.1 核心洞察：softmax 阻碍了结合律

回到推理时的逐 token 计算：

$$
\mathbf{o}_t = \sum_{j=1}^{t} \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j)}{\sum_{l=1}^{t} \exp(\mathbf{q}_t^\top \mathbf{k}_l)} \mathbf{v}_j
$$

**问题在于**：分母 $\sum_l \exp(\mathbf{q}_t^\top \mathbf{k}_l)$ 依赖于所有 key 的全局信息，使得我们**无法将计算重新排列**。具体来说，无法把 $\mathbf{q}_t$ 提取到求和号外面。

**Linear Attention 的做法**：移除 softmax 的**指数函数**，用原始点积作为注意力权重。但这里有一个重要的区分——原始 Linear Attention（Katharopoulos et al., 2020）**保留了归一化分母**，而后续的 DeltaNet/GDN/KDA 论文则**连分母也去掉了**。我们需要分两步理解。

### 2.2 第一步：原始 Linear Attention（带归一化分母）

Katharopoulos et al. (2020) 的原始公式引入特征映射 $\phi(\cdot)$ 并保留归一化：

$$
\mathbf{o}_t = \frac{\sum_{j=1}^{t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_j) \cdot \mathbf{v}_j}{\sum_{j=1}^{t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_j)}
$$

**分子**可以用结合律重排（与 softmax 不同，因为这里没有指数函数）：

$$
\text{分子} = \sum_{j=1}^{t} \mathbf{v}_j \left(\phi(\mathbf{k}_j)^\top \phi(\mathbf{q}_t)\right) = \left(\sum_{j=1}^{t} \mathbf{v}_j \phi(\mathbf{k}_j)^\top\right) \phi(\mathbf{q}_t) = \mathbf{S}_t \phi(\mathbf{q}_t)
$$

**分母**同样可以重排：

$$
\text{分母} = \phi(\mathbf{q}_t)^\top \sum_{j=1}^{t} \phi(\mathbf{k}_j) = \phi(\mathbf{q}_t)^\top \mathbf{z}_t
$$

因此原始 Linear Attention 的 RNN 形式为：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \phi(\mathbf{k}_t)^\top, \quad \mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t), \quad \mathbf{o}_t = \frac{\mathbf{S}_t \phi(\mathbf{q}_t)}{\mathbf{z}_t^\top \phi(\mathbf{q}_t)}
$$

> **关键区别**：softmax 的分母 $\sum_l \exp(\mathbf{q}_t^\top \mathbf{k}_l)$ 中 $\exp$ 使得分母与 $\mathbf{q}_t$ 以非线性方式纠缠，无法分离；而 Linear Attention 的分母 $\sum_j \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_j) = \phi(\mathbf{q}_t)^\top \mathbf{z}_t$ 是线性的，可以干净地分离为 $\phi(\mathbf{q}_t)$ 和 $\mathbf{z}_t$。这就是为什么 Linear Attention 能写成 RNN 而 softmax 不能。

### 2.3 第二步：去掉分母（DeltaNet/GDN/KDA 的选择）

后续的 DeltaNet、GDN、KDA 论文**直接去掉了归一化分母**，使用更简洁的形式：

$$
\mathbf{o}_t = \sum_{j=1}^{t} (\mathbf{q}_t^\top \mathbf{k}_j) \mathbf{v}_j
$$

（同时也省略了特征映射 $\phi$，直接用原始 $\mathbf{q}, \mathbf{k}$，并通过 L2 归一化保证数值稳定性。）

**为什么可以去掉分母？**

1. **Delta 规则自带"归一化"效果**：DeltaNet 的 MSE 损失 $\|\mathbf{S}\mathbf{k}_t - \mathbf{v}_t\|^2$ 本质上是在做回归，状态 $\mathbf{S}$ 的大小由学习率 $\beta_t$ 控制，不需要额外的归一化来约束输出范围。

2. **L2 归一化替代**：GDN/KDA 对 $\mathbf{q}, \mathbf{k}$ 做 L2 归一化（$\|\mathbf{q}\| = \|\mathbf{k}\| = 1$），使得点积 $\mathbf{q}^\top \mathbf{k} \in [-1, 1]$，输出范围天然有界。

3. **简化算法**：去掉分母后只需维护一个状态 $\mathbf{S}_t$（而非 $\mathbf{S}_t$ 和 $\mathbf{z}_t$ 两个），分块并行算法更简洁。

4. **性能无损**：实验表明，在 Delta 规则框架下，有无分母对语言建模性能影响很小。

> **教学要点**：原始 Linear Attention 保留分母是为了模拟 softmax 的归一化效果（使输出不受 key 数量的影响）。但在 Delta 规则框架下，状态更新本身具有自校正能力（先擦后写），分母的归一化作用变得冗余，因此可以安全地省略。

### 2.4 结合律：改变一切的关键

去掉分母后，$(\mathbf{q}_t^\top \mathbf{k}_j)$ 是一个标量，可以自由移动位置。利用标量与向量的交换律 $\alpha \mathbf{v} = \mathbf{v} \alpha$：

$$
\mathbf{o}_t = \sum_{j=1}^{t} (\mathbf{q}_t^\top \mathbf{k}_j) \mathbf{v}_j = \sum_{j=1}^{t} \mathbf{v}_j (\mathbf{k}_j^\top \mathbf{q}_t)
$$

现在 $\mathbf{k}_j^\top \mathbf{q}_t$ 是一个标量，可以提取到求和号外：

$$
\mathbf{o}_t = \left(\sum_{j=1}^{t} \mathbf{v}_j \mathbf{k}_j^\top\right) \mathbf{q}_t
$$

这就是**结合律的威力**：$\sum_j \mathbf{v}_j (\mathbf{k}_j^\top \mathbf{q}_t) = (\sum_j \mathbf{v}_j \mathbf{k}_j^\top) \mathbf{q}_t$。

> **为什么 softmax 不能这样做？** 因为 $\frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j)}{\sum_l \exp(\mathbf{q}_t^\top \mathbf{k}_l)}$ 中分母对 $\mathbf{q}_t$ 是非线性的（指数求和），无法将 $\mathbf{q}_t$ 提取出来。

### 2.5 从结合律到线性 RNN

定义**状态矩阵**（memory state）：

$$
\mathbf{S}_t = \sum_{j=1}^{t} \mathbf{v}_j \mathbf{k}_j^\top \in \mathbb{R}^{d_v \times d_k}
$$

则输出变为：

$$
\mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t
$$

关键观察：状态可以**递推更新**：

$$
\mathbf{S}_t = \sum_{j=1}^{t} \mathbf{v}_j \mathbf{k}_j^\top = \underbrace{\sum_{j=1}^{t-1} \mathbf{v}_j \mathbf{k}_j^\top}_{\mathbf{S}_{t-1}} + \mathbf{v}_t \mathbf{k}_t^\top = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top
$$

因此 Linear Attention 是一个**矩阵值状态的线性 RNN**：

$$
\boxed{\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top, \quad \mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t}
$$

**复杂度变化**：

| 维度 | Full Attention | Linear Attention |
| --- | --- | --- |
| 推理时间（逐 token） | $\mathcal{O}(t \cdot d)$（扫描历史 KV） | $\mathcal{O}(d^2)$（矩阵-向量乘） |
| 推理空间 | $\mathcal{O}(L \cdot d)$（KV Cache） | $\mathcal{O}(d^2)$（固定状态） |
| 训练并行 | $\mathcal{O}(L^2)$ | $\mathcal{O}(L)$（chunkwise） |

当 $t \gg d$ 时（长序列推理），$\mathcal{O}(d^2) \ll \mathcal{O}(t \cdot d)$，且无需 KV Cache。

### 2.6 训练时的并行形式

训练时需要并行计算所有位置的输出。Full Attention 的并行形式是 $\mathbf{O} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V}$。

Linear Attention 的并行形式为：

$$
\mathbf{O} = (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V}
$$

这里 $\odot \mathbf{M}$ 实现因果掩码（上三角置零）。虽然形式上仍是 $\mathcal{O}(L^2 d)$，但可以通过**分块并行**（chunkwise）降为 $\mathcal{O}(Ld^2)$。

### 2.7 固有局限：记忆碰撞

**问题**：状态 $\mathbf{S}$ 只能不断累加 $\mathbf{v}_t \mathbf{k}_t^\top$，**没有擦除机制**。随着序列增长，越来越多的外积叠加在一起，导致"记忆碰撞"。

**详细推导**：假设所有 key 已归一化为单位长度 $\|\mathbf{k}_j\| = 1$。当用 $\mathbf{k}_j$ 查询记忆时：

$$
\mathbf{S}\mathbf{k}_j = \left(\sum_i \mathbf{v}_i \mathbf{k}_i^\top\right) \mathbf{k}_j = \sum_i \mathbf{v}_i (\mathbf{k}_i^\top \mathbf{k}_j) = \mathbf{v}_j \cdot \underbrace{1}_{\mathbf{k}_j^\top \mathbf{k}_j} + \sum_{i \neq j} \mathbf{v}_i \underbrace{(\mathbf{k}_i^\top \mathbf{k}_j)}_{\text{交叉项}}
$$

理想情况下，$\mathbf{S}\mathbf{k}_j = \mathbf{v}_j$（精确检索）。但交叉项 $\sum_{i \neq j} (\mathbf{k}_i^\top \mathbf{k}_j) \mathbf{v}_i$ 构成**检索误差**。

要使误差为零，需要所有 key 两两正交：$\mathbf{k}_i^\top \mathbf{k}_j = 0, \forall i \neq j$。但在 $d$ 维空间中最多只有 $d$ 个正交向量。**当序列长度 $L > d$ 时，记忆碰撞不可避免**。

> **直觉理解**　想象一个有 $d$ 个抽屉的文件柜。Linear Attention 不断往抽屉里塞文件，但不清理旧文件。当文件数量远超抽屉容量时，不同文件混在同一抽屉里，找不到所需的那份。Full Attention 相当于每次都翻遍所有文件——慢但准确。

### 2.8 门控变体：引入遗忘

为缓解信息过载，研究者在状态更新中引入**遗忘门**（gating / decay）：

$$
\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top
$$

其中 $\mathbf{G}_t$ 是遗忘门矩阵，控制旧信息的保留比例。不同模型对 $\mathbf{G}_t$ 的参数化：

| 模型 | $\mathbf{G}_t$ | 衰减粒度 | 特点 |
| --- | --- | --- | --- |
| RetNet | $\alpha \mathbf{I}$（$\alpha$ 常数） | 全局 | 数据无关，简单但僵化 |
| Mamba2 | $\alpha_t \mathbf{I}$（$\alpha_t$ 数据相关） | 逐头 | 自适应衰减，但所有通道同等衰减 |
| GLA | $\mathbf{1} \boldsymbol{\alpha}_t^\top$ | 逐通道 | 外积结构，不同通道不同衰减 |

**门控的局限**：遗忘是**无差别的**——衰减所有 key-value 关联，无法只遗忘特定的过时关联。如果模型需要遗忘 key $\mathbf{k}_j$ 对应的关联，所有关联都会被同等衰减。这与 Full Attention 的精确选择性形成鲜明对比。

> **演进动机**　能否设计一种机制，既能像门控一样快速遗忘过时信息，又能像 Delta 规则一样精准修改特定的 key-value 关联？

---

## 3. DeltaNet：引入 Delta 规则

### 3.1 从 Linear Attention 的问题出发

Linear Attention 的状态更新 $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ 的问题在于：它只是**盲目地将新的外积堆加上去**，不考虑当前 key $\mathbf{k}_t$ 是否已经与某个旧 value 关联。

如果我们能**先检查当前 key 在记忆中对应什么旧值，再用新值校正**，就能避免记忆碰撞。

### 3.2 Delta 规则：误差纠正学习

**Delta 规则**（Widrow-Hoff, 1960）是神经网络中最古老的误差纠正学习原则之一。其核心思想：

> 用目标值与当前预测值之间的**差异**（delta）来调整参数。

**应用到 Linear Attention**：

1. **当前预测**：用 key $\mathbf{k}_t$ 查询记忆，得到旧值 $\mathbf{v}_t^{\text{old}} = \mathbf{S}_{t-1} \mathbf{k}_t$
2. **误差**：$\boldsymbol{\delta}_t = \mathbf{v}_t^{\text{old}} - \mathbf{v}_t$（旧值与新值的差异）
3. **更新**：沿误差方向修正状态

DeltaNet 的状态更新公式：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \underbrace{(\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t)}_{\text{误差} = \mathbf{v}_t^{\text{old}} - \mathbf{v}_t} \mathbf{k}_t^\top
$$

展开重组：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \mathbf{S}_{t-1} \mathbf{k}_t \mathbf{k}_t^\top + \beta_t \mathbf{v}_t \mathbf{k}_t^\top
$$

$$
\boxed{\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top}
$$

其中 $\beta_t \in (0,1)$ 是学习率（写入强度），控制更新的激进程度。

### 3.3 "先擦后写"的直观理解

定义旧值和新值：

$$
\mathbf{v}_t^{\text{old}} = \mathbf{S}_{t-1} \mathbf{k}_t, \quad \mathbf{v}_t^{\text{new}} = (1 - \beta_t) \mathbf{v}_t^{\text{old}} + \beta_t \mathbf{v}_t
$$

则更新可分解为两步：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \underbrace{\mathbf{v}_t^{\text{old}} \mathbf{k}_t^\top}_{\text{擦除：删除旧关联}} + \underbrace{\mathbf{v}_t^{\text{new}} \mathbf{k}_t^\top}_{\text{写入：添加新关联}}
$$

**三种边界情况**：

| $\beta_t$ 值 | 行为 | 等价于 |
| --- | --- | --- |
| $\beta_t = 0$ | $\mathbf{v}_t^{\text{new}} = \mathbf{v}_t^{\text{old}}$，擦除与写入抵消 | 记忆保持不变 |
| $\beta_t = 1$ | $\mathbf{v}_t^{\text{new}} = \mathbf{v}_t$，完全替换 | 用新值覆盖旧关联 |
| $\beta_t \in (0,1)$ | 新旧值加权组合 | 软更新 |

### 3.4 从在线学习严格推导

DeltaNet 的更新规则可以从**在线学习**（online learning）框架严格推导。

**设定**：将状态 $\mathbf{S}$ 视为"快速权重"（fast weight），每一步接收一个新的样本 $(\mathbf{k}_t, \mathbf{v}_t)$，目标是让 $\mathbf{S}$ 学会映射 $\mathbf{k}_t \mapsto \mathbf{v}_t$。

**在线回归损失**（MSE）：

$$
\mathcal{L}_t(\mathbf{S}) = \frac{1}{2} \|\mathbf{S} \mathbf{k}_t - \mathbf{v}_t\|^2
$$

**梯度**：

$$
\nabla_\mathbf{S} \mathcal{L}_t(\mathbf{S}) = (\mathbf{S} \mathbf{k}_t - \mathbf{v}_t) \mathbf{k}_t^\top
$$

**一步梯度下降**（学习率 $\beta_t$）：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \nabla_\mathbf{S} \mathcal{L}_t(\mathbf{S}_{t-1}) = \mathbf{S}_{t-1} - \beta_t (\mathbf{S}_{t-1} \mathbf{k}_t - \mathbf{v}_t) \mathbf{k}_t^\top
$$

这正是 DeltaNet 的更新规则。

**对比 Linear Attention**：Linear Attention 对应的损失是**负内积**（无界相关损失）：

$$
\mathcal{L}'_t(\mathbf{S}) = -\langle \mathbf{S} \mathbf{k}_t, \mathbf{v}_t \rangle
$$

其梯度下降为：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \eta_t \nabla \mathcal{L}'_t = \mathbf{S}_{t-1} + \eta_t \mathbf{v}_t \mathbf{k}_t^\top
$$

令 $\eta_t = 1$ 就得到 Linear Attention。**关键区别**：负内积损失只能"强化"（加法），无法"纠正"（减法）；MSE 损失能根据误差方向调整，既能加也能减。

### 3.5 Householder 变换

DeltaNet 的状态转移矩阵 $(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$ 是**广义 Householder 变换**。

**标准 Householder 变换**：$\mathbf{H} = \mathbf{I} - 2\mathbf{u}\mathbf{u}^\top$（$\|\mathbf{u}\| = 1$），将向量关于法向量 $\mathbf{u}$ 反射。

**DeltaNet 的推广**：$\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$（$\beta_t \in (0,1)$），是"部分反射"——不完全是反射，而是部分投影。

### 3.6 分块并行训练：WY 表示

**问题**：DeltaNet 的递推形式无法并行训练。需要展开递推并找到高效的矩阵形式。

**分块展开**：将序列分成大小为 $C$ 的块。对第 $t$ 个块内的第 $r$ 步：

$$
\mathbf{S}_{[t]}^r = \mathbf{S}_{[t]} \underbrace{\prod_{i=1}^r (\mathbf{I} - \beta_{[t]}^i \mathbf{k}_{[t]}^i \mathbf{k}_{[t]}^{i\top})}_{:= \mathbf{P}_{[t]}^r} + \underbrace{\sum_{i=1}^r \beta_{[t]}^i \mathbf{v}_{[t]}^i \mathbf{k}_{[t]}^{i\top} \prod_{j=i+1}^r (\mathbf{I} - \beta_{[t]}^j \mathbf{k}_{[t]}^j \mathbf{k}_{[t]}^{j\top})}_{:= \mathbf{H}_{[t]}^r}
$$

其中 $\mathbf{S}_{[t]} = \mathbf{S}_{[t]}^0$ 是块的初始状态（上一块的最终状态）。

**WY 表示**（Bischof & Loan, 1985）：将一系列 Householder 变换的连乘压缩为紧凑形式：

$$
\mathbf{P}_{[t]}^r = \mathbf{I} - \sum_{i=1}^r \mathbf{w}_{[t]}^i \mathbf{k}_{[t]}^{i\top}
$$

其中辅助向量 $\mathbf{w}_{[t]}^r$ 通过递推计算：

$$
\mathbf{w}_{[t]}^r = \beta_{[t]}^r \left(\mathbf{k}_{[t]}^r - \sum_{i=1}^{r-1} \mathbf{w}_{[t]}^i (\mathbf{k}_{[t]}^{i\top} \mathbf{k}_{[t]}^r)\right)
$$

**直觉**：$\mathbf{w}_{[t]}^r$ 可以理解为"经过之前所有 Householder 变换修正后的第 $r$ 个 key"，它编码了当前 key 与之前所有 key 的交互关系。

类似地，$\mathbf{H}_{[t]}^r$ 可表示为：

$$
\mathbf{H}_{[t]}^r = \sum_{i=1}^r \mathbf{u}_{[t]}^i \mathbf{k}_{[t]}^{i\top}
$$

$$
\mathbf{u}_{[t]}^r = \beta_{[t]}^r \left(\mathbf{v}_{[t]}^r - \sum_{i=1}^{r-1} \mathbf{u}_{[t]}^i (\mathbf{k}_{[t]}^{i\top} \mathbf{k}_{[t]}^r)\right)
$$

### 3.7 UT 变换：矩阵化

**UT 变换**（Joffrain et al., 2006）将上述递推转化为矩阵运算，充分利用 Tensor Core。

定义矩阵 $\mathbf{T}_{[t]}$：

$$
\mathbf{T}_{[t]} = \left[\mathbf{I} + \text{StrictTril}\left(\text{diag}(\beta_{[t]}) \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top\right)\right]^{-1} \text{diag}(\beta_{[t]}) \in \mathbb{R}^{C \times C}
$$

其中 $\text{StrictTril}$ 取严格下三角部分（不含对角线），$\text{diag}(\beta_{[t]})$ 是以 $\beta$ 值为对角线的对角矩阵。

辅助矩阵：

$$
\mathbf{W}_{[t]} = \mathbf{T}_{[t]} \mathbf{K}_{[t]} \in \mathbb{R}^{C \times d_k}, \quad \mathbf{U}_{[t]} = \mathbf{T}_{[t]} \mathbf{V}_{[t]} \in \mathbb{R}^{C \times d_v}
$$

**分块状态更新**：

$$
\mathbf{S}_{[t+1]} = \mathbf{S}_{[t]} + (\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]}^\top)^\top \mathbf{K}_{[t]}
$$

**分块输出**：

$$
\mathbf{O}_{[t]} = \mathbf{Q}_{[t]} \mathbf{S}_{[t]}^\top + (\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M})(\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]}^\top)
$$

其中第一项 $\mathbf{Q}_{[t]} \mathbf{S}_{[t]}^\top$ 是块间贡献（用块初始状态计算），第二项是块内贡献（用因果掩码 $\mathbf{M}$ 处理块内自回归）。

### 3.8 DeltaNet 的局限

DeltaNet **只修改单个 key-value 关联**（通过 Householder 变换），缺乏快速清除大量过时信息的能力。在上下文切换场景中（如从讨论天气切换到讨论代码），旧上下文的全部信息需要被批量清除，但 DeltaNet 只能逐个修正，效率极低。

> **演进动机**　需要一种机制，既保留 Delta 规则的精准更新能力，又能快速遗忘大量过时信息。

---

## 4. Gated DeltaNet (GDN)：遗忘门与 Delta 规则的统一

### 4.1 核心思想：两种互补的机制

回顾前两节的演进：

- **门控（Mamba2 等）**：能快速遗忘，但是**无差别**衰减所有关联
- **Delta 规则（DeltaNet）**：能精准更新单个关联，但**无法批量遗忘**

GDN 的洞察：这两种机制是**互补的**——门控适合快速清除过时上下文，Delta 规则适合精准修正特定关联。将二者统一可以获得两者的优势。

### 4.2 Gated Delta Rule：公式推导

**出发点**：DeltaNet 的更新规则是：

$$
\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top
$$

**引入标量遗忘门** $\alpha_t \in (0,1)$：在 Householder 变换矩阵上乘以 $\alpha_t$：

$$
\boxed{\mathbf{S}_t = \mathbf{S}_{t-1} \left(\alpha_t (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)\right) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top}
$$

**展开理解**：

$$
\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top
$$

- $\alpha_t \mathbf{S}_{t-1}$：对旧状态做标量衰减（遗忘）
- $\alpha_t \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$：在衰减后的状态上做 Delta 规则更新
- $+ \beta_t \mathbf{v}_t \mathbf{k}_t^\top$：写入新的 key-value 关联

**三种边界情况**：

| $\alpha_t$ 值 | 行为 | 等价于 |
| --- | --- | --- |
| $\alpha_t \to 0$ | 旧状态被清零，只剩 $\beta_t \mathbf{v}_t \mathbf{k}_t^\top$ | 快速遗忘 + 写入新信息（类似 Mamba2） |
| $\alpha_t \to 1$ | 旧状态完整保留，退化为纯 Delta 规则 | DeltaNet |
| $\alpha_t \in (0,1)$ | 在遗忘与精准更新之间动态平衡 | GDN 的完整能力 |

### 4.3 在线学习视角：统一框架

在 Liu et al. (2024) 的在线学习框架下，所有线性 RNN 的状态更新都可以从**带正则化的在线优化目标**推导。

**一般形式**：

$$
\mathcal{L}_t(\mathbf{S}) = \underbrace{\|\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1}\|_F^2}_{\text{正则项：控制与旧状态的偏离}} - 2 \underbrace{\langle \mathbf{S}_t \mathbf{k}_t, \beta_t (\mathbf{v}_t - \alpha_t \mathbf{S}_{t-1} \mathbf{k}_t) \rangle}_{\text{损失项：学习 key→value 映射}}
$$

- 正则项 $\|\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1}\|_F^2$ 中的 $\alpha_t$ 控制"允许多大程度的遗忘"
- 损失项中的 $\beta_t(\mathbf{v}_t - \alpha_t \mathbf{S}_{t-1}\mathbf{k}_t)$ 是"在衰减后的旧状态上的预测误差"

**各方法的统一对比**：

| 方法 | 在线学习目标 | 更新规则 |
| --- | --- | --- |
| Linear Attention | $\|\mathbf{S}_t - \mathbf{S}_{t-1}\|_F^2 - 2\langle \mathbf{S}_t \mathbf{k}_t, \mathbf{v}_t \rangle$ | $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ |
| Mamba2 | $\|\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1}\|_F^2 - 2\langle \mathbf{S}_t \mathbf{k}_t, \mathbf{v}_t \rangle$ | $\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ |
| DeltaNet | $\|\mathbf{S}_t - \mathbf{S}_{t-1}\|_F^2 - 2\langle \mathbf{S}_t \mathbf{k}_t, \beta_t(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t) \rangle$ | $\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$ |
| **GDN** | $\|\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1}\|_F^2 - 2\langle \mathbf{S}_t \mathbf{k}_t, \beta_t(\mathbf{v}_t - \alpha_t \mathbf{S}_{t-1}\mathbf{k}_t) \rangle$ | $\mathbf{S}_t = \mathbf{S}_{t-1}(\alpha_t(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$ |

**如何理解这张表**：

1. **正则项的区别**：Linear Attention 和 DeltaNet 用 $\|\mathbf{S}_t - \mathbf{S}_{t-1}\|_F^2$（$\alpha_t = 1$，不允许遗忘）；Mamba2 和 GDN 用 $\|\mathbf{S}_t - \alpha_t \mathbf{S}_{t-1}\|_F^2$（$\alpha_t < 1$，允许可控遗忘）。

2. **损失项的区别**：Linear Attention 和 Mamba2 用 $-\langle \mathbf{S}_t \mathbf{k}_t, \mathbf{v}_t \rangle$（负内积，只能强化）；DeltaNet 和 GDN 用 $-\langle \mathbf{S}_t \mathbf{k}_t, \beta_t(\mathbf{v}_t - \cdot) \rangle$（基于误差的损失，能纠正）。

3. **GDN = Mamba2 的遗忘 + DeltaNet 的纠错**：GDN 的正则项来自 Mamba2（有 $\alpha_t$），损失项来自 DeltaNet（有 $\beta_t$ 和误差项）。

**从 TTT 视角**：将 $\mathbf{S}$ 视为测试时训练的"快速权重"，GDN 相当于在 SGD 更新中加入自适应权重衰减 $\alpha_t$——这是深度学习中广泛使用的技术（Krogh & Hertz, 1991）。

### 4.4 S-NIAH 案例：遗忘与记忆的互补性

GDN 论文通过 Single Needle-in-a-Haystack (S-NIAH) 基准验证了两种机制的互补性：

| 设置 | 测试能力 | DeltaNet | Mamba2 | GDN |
| --- | --- | --- | --- | --- |
| S-NIAH-1 (passkey) | 长期记忆保持 | 接近完美 | 2K 后退化 | 轻微退化 |
| S-NIAH-2 (number) | 高效记忆管理 | 长序列下降 | 短序列好 | 保持良好 |
| S-NIAH-3 (UUID) | 复杂模式记忆 | 长序列下降 | 快速退化 | 保持较好 |

**三个关键结论**：

1. **衰减损害记忆保持**（S-NIAH-1）：无差别衰减会误伤需要保留的信息。DeltaNet 无衰减，表现最好；Mamba2 衰减太快，长序列丢失针。
2. **门控促进过滤**（S-NIAH-2）：真实上下文中有大量无关信息，需要过滤。DeltaNet 无清除机制，信息堆积导致碰撞；Mamba2 和 GDN 通过门控过滤。
3. **Delta 规则有助于记忆**（S-NIAH-3）：复杂模式需要精准写入。Mamba2 的纯加法无法精确记忆 UUID；GDN 的 Delta 规则能精准写入。

**结论**：GDN 在三个维度上都表现最好，验证了"遗忘门 + Delta 规则"互补统一的有效性。

### 4.5 分块并行训练算法

GDN 的分块算法在 DeltaNet 基础上引入衰减项。关键修改是将 DeltaNet 的 WY 表示中的 $\mathbf{k}$ 替换为衰减后的版本。

**累积衰减**：定义块内累积衰减：

$$
\gamma_{[t]}^j = \prod_{k=tC+1}^{tC+j} \alpha_k, \quad \gamma_{[t]}^1 = \alpha_{tC+1}
$$

**箭头记号**（表示衰减方向）：

$$
\overleftarrow{\mathbf{q}_{[t]}^r} = \gamma_{[t]}^r \mathbf{q}_{[t]}^r \quad \text{（衰减到块首）}
$$

$$
\overrightarrow{\mathbf{k}_{[t]}^r} = \frac{\gamma_{[t]}^C}{\gamma_{[t]}^r} \mathbf{k}_{[t]}^r \quad \text{（衰减到块尾）}
$$

$$
\overrightarrow{\mathbf{S}_{[t]}} = \gamma_{[t]}^C \mathbf{S}_{[t]} \quad \text{（状态在整块上衰减）}
$$

**修正后的 UT 变换**（$\widetilde{\mathbf{U}_{[t]}}$ 融入衰减掩码 $\Gamma_{[t]}$）：

$$
\widetilde{\mathbf{U}_{[t]}} = \left[\mathbf{I} + \text{StrictTril}\left(\text{diag}(\beta_{[t]}) (\Gamma_{[t]} \odot \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top)\right)\right]^{-1} \text{diag}(\beta_{[t]}) \mathbf{V}_{[t]}
$$

其中 $(\Gamma_{[t]})_{ij} = \frac{\gamma_{[t]}^i}{\gamma_{[t]}^j}$ 是衰减感知的因果掩码。

**分块状态更新与输出**：

$$
\mathbf{S}_{[t+1]} = \overrightarrow{\mathbf{S}_{[t]}} + (\widetilde{\mathbf{U}_{[t]}} - \overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}_{[t]}^\top)^\top \overrightarrow{\mathbf{K}_{[t]}}
$$

$$
\mathbf{O}_{[t]} = \overleftarrow{\mathbf{Q}_{[t]}} \mathbf{S}_{[t]}^\top + (\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M})(\widetilde{\mathbf{U}_{[t]}} - \overleftarrow{\mathbf{W}_{[t]}} \mathbf{S}_{[t]}^\top)
$$

与 DeltaNet 的分块算法相比，唯一区别是所有涉及跨时间步的量都乘以了相应的衰减因子。

### 4.6 神经参数化

GDN 的 block 设计遵循 Llama 宏架构：

| 组件 | 计算路径 |
| --- | --- |
| $\mathbf{q}, \mathbf{k}$ | 线性投影 → ShortConv → SiLU → L2 归一化 |
| $\mathbf{v}$ | 线性投影 → ShortConv → SiLU |
| $\alpha, \beta$ | 仅线性投影 |
| 输出 | 归一化 + 门控 → 输出投影 |

L2 归一化确保 key 的范数为 1，保证 Householder 变换的数值稳定性。

---

## 5. KDA (Kimi Delta Attention)：细粒度门控

### 5.1 动机：为什么标量门不够好

GDN 使用标量遗忘门 $\alpha_t \in (0,1)$，即**同一个衰减率应用于所有通道**。这类似于 RoPE（旋转位置编码）如果只使用单一旋转频率——无法区分不同维度的位置信息。

**RoPE 的启发**：RoPE 为每对维度分配不同的旋转频率，实现类似非均匀傅里叶变换的细粒度位置编码。GDN 的逐头标量衰减缺乏这种逐维度多样性。

**KDA 的方案**：将标量 $\alpha_t$ 升级为**逐通道向量** $\boldsymbol{\alpha}_t \in [0,1]^{d_k}$，即每个特征维度有独立的遗忘率。

### 5.2 核心公式

$$
\boxed{\mathbf{S}_t = \left(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top\right) \text{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top}
$$

**逐步拆解**：

1. $\text{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1}$：对旧状态的每个通道独立衰减（类似 GLA 的细粒度门控）
2. $(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \cdot$：在衰减后的状态上做 Householder 变换（Delta 规则的擦除-写入）
3. $+ \beta_t \mathbf{k}_t \mathbf{v}_t^\top$：写入新的 key-value 关联

**与 GDN 的对比**：

| 维度 | GDN | KDA |
| --- | --- | --- |
| 遗忘门 | $\alpha_t \in (0,1)$（标量） | $\boldsymbol{\alpha}_t \in [0,1]^{d_k}$（向量） |
| 衰减操作 | $\alpha_t \mathbf{S}_{t-1}$（标量乘） | $\text{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1}$（对角矩阵乘） |
| 门控粒度 | 逐头 | 逐通道 |
| 位置编码 | 粗粒度 | 类似 RoPE 的细粒度 |
| 对标 | Mamba2 的标量衰减 | GLA 的对角门控 |

### 5.3 与 DPLR 的关系：约束带来效率

**DPLR（Diagonal-Plus-Low-Rank）**是一类更一般的状态转移结构：

$$
\mathbf{S}_t = (\mathbf{D} - \mathbf{a}_t \mathbf{b}_t^\top) \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top
$$

其中 $\mathbf{D}$ 是对角矩阵，$\mathbf{a}_t \mathbf{b}_t^\top$ 是秩-1修正。

**KDA 作为 DPLR 的约束变体**：通过将 $\mathbf{a}, \mathbf{b}$ 绑定为 $\mathbf{k}$ 的函数：

$$
\mathbf{D} = \text{Diag}(\boldsymbol{\alpha}_t), \quad \mathbf{a}_t = \beta_t \mathbf{k}_t, \quad \mathbf{b}_t = \mathbf{k}_t \odot \boldsymbol{\alpha}_t
$$

**验证**：代入 DPLR 公式：

$$
(\mathbf{D} - \mathbf{a}_t \mathbf{b}_t^\top) \mathbf{S}_{t-1} = \left(\text{Diag}(\boldsymbol{\alpha}_t) - \beta_t \mathbf{k}_t (\mathbf{k}_t \odot \boldsymbol{\alpha}_t)^\top\right) \mathbf{S}_{t-1}
$$

$$
= \text{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1} - \beta_t \mathbf{k}_t (\mathbf{k}_t \odot \boldsymbol{\alpha}_t)^\top \mathbf{S}_{t-1}
$$

由于 $(\mathbf{k}_t \odot \boldsymbol{\alpha}_t)^\top = \mathbf{k}_t^\top \text{Diag}(\boldsymbol{\alpha}_t)$：

$$
= \text{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top \text{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1}
$$

$$
= (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \text{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1}
$$

这正是 KDA 的公式。**约束的核心**：通过共享 $\boldsymbol{\alpha}_t$，将其提取到 Householder 变换之前，先做乘性衰减再做 Delta 规则更新。

### 5.4 效率优势：KDA vs DPLR

| 优化维度 | DPLR（一般形式） | KDA（约束形式） | 节省 |
| --- | --- | --- | --- |
| 二级分块 | 4 次 | 2 次 | 减少 2 次 |
| 额外矩阵乘法 | — | — | 减少 3 次 |
| 算子速度 | 基线 | ~2× 加速 | 在 64k 序列验证 |

**为什么 DPLR 更慢**：

1. **数值不稳定的除法**：DPLR 的一般形式在分块计算中涉及 $1/\Gamma$ 的除法运算。当 $\Gamma$ 接近 0 时数值不稳定。GLA 通过在对数域计算 + 二级分块（全精度）来解决，但阻止了半精度矩阵乘的充分利用。

2. **更多的矩阵运算**：DPLR 有两个独立的低秩向量 $\mathbf{a}$ 和 $\mathbf{b}$，需要分别处理。KDA 通过固定 $\mathbf{a} = \beta_t \mathbf{k}$ 和 $\mathbf{b} = \mathbf{k} \odot \boldsymbol{\alpha}$，将两者统一为 $\mathbf{k}$ 的函数，消除了冗余计算。

### 5.5 分块并行算法

#### 累积衰减定义

KDA 使用**逐通道**的累积衰减（注意：与 GDN 的标量累积不同）：

$$
\text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to j}) := \prod_{k=i}^{j} \text{Diag}(\boldsymbol{\alpha}_{[t]}^k)
$$

以及矩阵堆叠形式 $\boldsymbol{\Gamma}_{[t]}^{i \to j} \in \mathbb{R}^{C \times d_k}$，其中每行是对应时间步的 $\boldsymbol{\gamma}$ 向量。

#### WY 表示（Comba 形式）

$$
\mathbf{P}_{[t]}^r = \text{Diag}(\boldsymbol{\gamma}_{[t]}^r) - \sum_{i=1}^r \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^i \mathbf{w}_{[t]}^{i\top}
$$

$$
\mathbf{H}_{[t]}^r = \sum_{i=1}^r \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^i \mathbf{u}_{[t]}^{i\top}
$$

辅助向量递推（注意衰减项 $\text{Diag}(\boldsymbol{\gamma})$ 融入内积）：

$$
\mathbf{w}_{[t]}^r = \beta_{[t]}^r \left(\text{Diag}(\boldsymbol{\gamma}_{[t]}^r) \mathbf{k}_{[t]}^r - \sum_{i=1}^{r-1} \mathbf{w}_{[t]}^i \left(\mathbf{k}_{[t]}^{i\top} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^r\right)\right)
$$

$$
\mathbf{u}_{[t]}^r = \beta_{[t]}^r \left(\mathbf{v}_{[t]}^r - \sum_{i=1}^{r-1} \mathbf{u}_{[t]}^i \left(\mathbf{k}_{[t]}^{i\top} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^r\right)\right)
$$

**与 GDN 的区别**：GDN 的衰减是标量 $\gamma$ 乘以向量；KDA 的衰减是对角矩阵 $\text{Diag}(\boldsymbol{\gamma})$ 乘以向量，每个通道独立衰减。

#### UT 变换

$$
\mathbf{M}_{[t]} = \left(\mathbf{I} + \text{StrictTril}\left(\text{Diag}(\beta_{[t]}) (\boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{K}_{[t]}) \left(\frac{\mathbf{K}_{[t]}}{\boldsymbol{\Gamma}_{[t]}^{1 \to C}}\right)^\top\right)\right)^{-1} \text{Diag}(\beta_{[t]})
$$

$$
\mathbf{W}_{[t]} = \mathbf{M}_{[t]} (\boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{K}_{[t]}), \quad \mathbf{U}_{[t]} = \mathbf{M}_{[t]} \mathbf{V}_{[t]}
$$

#### 分块状态更新

$$
\mathbf{S}_{[t+1]} = \text{Diag}(\boldsymbol{\gamma}_{[t]}^C) \mathbf{S}_{[t]} + (\boldsymbol{\Gamma}_{[t]}^{i \to C} \odot \mathbf{K}_{[t]})^\top (\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]})
$$

#### 分块输出

$$
\mathbf{O}_{[t]} = \underbrace{(\boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{Q}_{[t]}) \mathbf{S}_{[t]}}_{\text{块间贡献}} + \underbrace{\text{Tril}\left((\boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{Q}_{[t]}) \left(\frac{\mathbf{K}_{[t]}}{\boldsymbol{\Gamma}_{[t]}^{1 \to C}}\right)^\top\right)}_{\text{块内贡献}} \underbrace{(\mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]})}_{\text{伪 value 项}}
$$

### 5.6 神经参数化

KDA 每个头 $h$ 的输入计算：

$$
\mathbf{q}_t^h, \mathbf{k}_t^h = \text{L2Norm}(\text{Swish}(\text{ShortConv}(\mathbf{W}_{q/k}^h \mathbf{x}_t))) \in \mathbb{R}^{d_k}
$$

$$
\mathbf{v}_t^h = \text{Swish}(\text{ShortConv}(\mathbf{W}_v^h \mathbf{x}_t)) \in \mathbb{R}^{d_v}
$$

$$
\boldsymbol{\alpha}_t^h = f(\mathbf{W}_\alpha^{\uparrow} \mathbf{W}_\alpha^{\downarrow} \mathbf{x}_t) \in [0,1]^{d_k}
$$

$$
\beta_t^h = \text{Sigmoid}(\mathbf{W}_\beta^h \mathbf{x}_t) \in [0,1]
$$

其中 $d_k = d_v = 128$。$\boldsymbol{\alpha}$ 采用**低秩投影**参数化（$\mathbf{W}_\alpha^{\downarrow} \in \mathbb{R}^{r \times d}$, $\mathbf{W}_\alpha^{\uparrow} \in \mathbb{R}^{d_k \times r}$, $r = d_k$），$f(\cdot)$ 是类似 GDN/Mamba 的衰减函数。

输出经过 head-wise RMSNorm + Sigmoid 门控：

$$
\mathbf{o}_t = \mathbf{W}_o \left(\text{Sigmoid}(\mathbf{W}_g^{\uparrow} \mathbf{W}_g^{\downarrow} \mathbf{x}_t) \odot \text{RMSNorm}(\text{KDA}(\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t, \boldsymbol{\alpha}_t, \beta_t))\right)
$$

### 5.7 Kimi Linear 混合架构

KDA 不单独使用，而是以 **3:1 比例**与全局注意力（MLA）交替堆叠：

```
[KDA] → [KDA] → [KDA] → [MLA] → [KDA] → [KDA] → [KDA] → [MLA] → ...
```

**设计理由**：

| 设计选择 | 理由 |
| --- | --- |
| 3:1 比例 | 质量与效率的最优平衡（实验验证） |
| MLA 用 NoPE | 位置信息全部由 KDA 承担，避免 RoPE 外推问题 |
| 层间混合（非头间） | 系统简单，训练稳定，推理调度方便 |

**效率收益**：

- KV Cache 减少 75%（3/4 的层用固定大小状态替代 KV Cache）
- 1M 上下文解码吞吐量提升 6.3×
- 在 1.4T token 训练后，短上下文、长上下文和 RL 任务均超越全注意力基线

### 5.8 复杂度分析

**训练 FLOPs**（单头，头维度 $d_h$，分块大小 $C = 64$）：

$$
\text{FLOPs}_{\text{KDA}}(T; C, d_h) = 6Td_h^2 + 3TCd_h + TC^2
$$

**全注意力 FLOPs**（对比）：

$$
\text{FLOPs}_{\text{Attn}}(T; d_h) = 2T^2 d_h
$$

当 $T \gg d_h$ 时（如 $T = 128\text{K}$, $d_h = 128$），KDA 的 $\mathcal{O}(Td_h^2)$ 远小于全注意力的 $\mathcal{O}(T^2 d_h)$。

**推理**：KDA 维持固定大小状态（$d_k \times d_v$ 每头），与序列长度无关。prefill 阶段使用分块 kernel，autoregressive 生成切换到递推 kernel（$\mathcal{O}(d^2)$ 每 token）。

### 5.9 KDA 作为可学习位置编码

GDN/KDA 的门控机制可以解释为一种**数据相关的乘性位置编码**，放松了 RoPE 的正交约束。

Full Attention 的注意力分数可以写成：

$$
s_{t,j} = \mathbf{q}_t^\top \left(\prod_{k=i+1}^t \mathbf{R}_k\right) \mathbf{k}_i
$$

其中 $\mathbf{R}_k$ 是 RoPE 的旋转矩阵（固定频率、正交）。

GDN/KDA 的对应形式为：

$$
s_{t,j} = \mathbf{q}_t^\top \left(\prod_{k=j+1}^t \mathbf{A}_k (\mathbf{I} - \beta_k \mathbf{k}_k \mathbf{k}_k^\top)\right) \mathbf{k}_j
$$

其中 $\mathbf{A}_k$ 是遗忘门（GDN 中 $\alpha_k \mathbf{I}$，KDA 中 $\text{Diag}(\boldsymbol{\alpha}_k)$），$(\mathbf{I} - \beta_k \mathbf{k}_k \mathbf{k}_k^\top)$ 是 Delta 规则的 Householder 变换。

**关键区别**：RoPE 的 $\mathbf{R}_k$ 是固定的、正交的；GDN/KDA 的转移矩阵是**数据相关的、可学习的**，因此更具表达力。这也是 KDA 层可以替代 RoPE 的理论依据——KDA 本身就承担了位置感知的角色。

---

## 6. 演进对比与总结

### 6.1 状态更新规则完整对比

| 方法 | 状态更新规则 | 遗忘机制 | 更新机制 | 门控粒度 |
| --- | --- | --- | --- | --- |
| **Full Attention** | $\mathbf{o}_t = \sum_j \text{softmax}(\mathbf{q}_t^\top \mathbf{k}_j) \mathbf{v}_j$ | softmax 归一化 | 全 KV 精确检索 | 逐 token |
| Linear Attention | $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | 无 | 纯累积 | — |
| Mamba2 | $\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\top$ | 标量全局衰减 | 纯累积 | 逐头 |
| DeltaNet | $\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$ | 无 | Delta 规则 | — |
| **GDN** | $\mathbf{S}_t = \mathbf{S}_{t-1}(\alpha_t(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)) + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$ | 标量衰减 | Delta 规则 | 逐头 |
| **KDA** | $\mathbf{S}_t = (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)\text{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top$ | 逐通道衰减 | Delta 规则 | **逐通道** |

### 6.2 演进逻辑

```
Full Attention
  │ ✅ 精确检索 (softmax 竞争性选择)
  │ ❌ O(L²) 推理复杂度, KV Cache 线性增长
  │
  │  移除 softmax → 利用结合律 → 矩阵值 RNN
  ▼
Linear Attention
  │ ✅ O(Ld²) 推理, O(d²) 固定状态, 无 KV Cache
  │ ❌ 只能加不能擦 → 记忆碰撞 → 检索退化
  │
  │  引入 Delta 规则: 在线 MSE 梯度下降 → 先擦后写
  ▼
DeltaNet
  │ ✅ 精准更新单个 KV 关联, 纠正检索误差
  │ ❌ 无全局遗忘 → 上下文切换困难
  │
  │  引入标量遗忘门 α_t → 统一遗忘与精准更新
  ▼
Gated DeltaNet (GDN)
  │ ✅ α→0 快速清空; α→1 精准更新; 动态平衡
  │ ❌ 门控粒度太粗 (逐头, 所有通道同等衰减)
  │
  │  逐通道对角门 + DPLR 约束 a=b=k
  ▼
KDA (Kimi Delta Attention)
  │ ✅ 逐通道门控 (类 RoPE 细粒度位置感知)
  │ ✅ 2× 算子加速 (消除 2 次二级分块 + 3 次矩阵乘法)
  │ ✅ 3:1 混合 MLA → 全注意力替代方案
```

### 6.3 核心设计权衡

| 设计维度 | 选择 | 理由 |
| --- | --- | --- |
| 损失函数 | MSE 重构损失（非内积损失） | 支持误差纠正，改善关联检索 |
| 遗忘门 | 逐通道对角 $\text{Diag}(\boldsymbol{\alpha}_t)$ | 细粒度记忆管理 + 位置感知 |
| DPLR 约束 | $\mathbf{a} = \beta_t \mathbf{k}$, $\mathbf{b} = \mathbf{k} \odot \boldsymbol{\alpha}$ | 消除数值不稳定 + 减少矩阵乘法 |
| 混合策略 | 3:1 KDA:MLA 交替 | 质量-效率最优平衡 |
| 位置编码 | KDA 承担位置感知，MLA 用 NoPE | 避免 RoPE 外推问题 |

### 6.4 性能数据

在 1.4T token 公平训练对比中（Kimi Linear 48B 总参数 / 3B 激活参数）：

| 指标 | MLA (全注意力) | GDN-H | Kimi Linear (KDA) |
| --- | --- | --- | --- |
| MMLU-Pro (4k) | 47.2 | 47.9 | **51.0** |
| RULER (128k) | 81.3 | 80.5 | **84.3** |
| 1M 解码加速 | 1× | ~5.7× | **6.3×** |
| KV Cache 使用 | 100% | ~25% | ~25% |

---

## 参考文献

- Vaswani et al. (2017). *Attention Is All You Need.* (Transformer)
- Katharopoulos et al. (2020). *Linear Transformers Are Secretly Fast Weight Programmers.* (Linear Attention)
- Schlag et al. (2021). *Linear Transformers Are Secretly Fast Weight Programmers.* (DeltaNet)
- Yang et al. (2024). *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* (DeltaNet chunkwise, NeurIPS'24)
- Dao & Gu (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* (Mamba2)
- Yang, Kautz, Hatamizadeh (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* (GDN, ICLR 2025) [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture.* (KDA) [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
- Bischof & Loan (1985). *The WY Representation for Products of Householder Matrices.*
- Joffrain et al. (2006). *Accumulation of Householder Transformations, Revisited.*
- Liu et al. (2024). *Longhorn: State Tracking Models.*
- Yang et al. (2024). *Gated Linear Attention Transformers with Hardware-Efficient Training.* (GLA)
- Widrow & Hoff (1960). *Adaptive Switching Circuits.* (Delta Rule)
