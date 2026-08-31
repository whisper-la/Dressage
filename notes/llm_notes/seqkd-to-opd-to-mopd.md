**TECHNICAL REFERENCE · 2026**

# 从 SeqKD 到 OPD、MOPD 再到 G-OPD：On-Policy Distillation 算法家族演进详解

*离线蒸馏 → GKD → On-Policy Distillation → Multi-Teacher OPD → Generalized OPD*

**面向教学的逐步推导 · 公式细节 · 工业实践 · 演进逻辑**

**适用读者**

希望理解"蒸馏、RL、on-policy 训练三者如何汇合"、以及 2025-2026 年头部实验室为何纷纷把多专家融合押注在 OPD 上的研究人员与工程师

版本基线：2026 年 8 月

姊妹篇：[从 Safe Softmax 到 Online Softmax 再到 FlashAttention](./online-softmax-to-flashattention.md) 讲执行层的 IO 优化；本文讲**训练信号层**的演进——学习信号从哪来、有多稠密、由几个教师提供。RL 算法本体（PPO/GRPO/GSPO/DAPO 等）见 [llm-rl-algorithms-zh.md](../llm-rl-algorithms-zh.md)，OPD 与之正交组合（其 §9.4 有对应小节）。

---

## 执行摘要

> **一句话结论**　On-Policy Distillation（OPD）让学生模型在**自己生成的轨迹**上接受教师**逐 token 的稠密分布监督**，从而同时获得 RL 的 on-policy 相关性与蒸馏的信号密度。家族演进沿三条轴线展开：**数据来源**（离线 → on-policy，GKD 奠基）、**教师组织方式**（单教师 → 多教师 MOPD，"specialize-then-unify" 成为头部实验室多能力融合的标配）、**目标函数的理论统一**（G-OPD 证明 OPD 是"奖励:KL 权重锁死 1:1"的稠密 KL 约束 RL 特例；解耦权重、奖励外推后，ExOPD 让学生**反超教师**）。

| 算法 | 数据分布 | 监督信号 | 教师数 | 学生天花板 | 代表 |
| --- | --- | --- | --- | --- | --- |
| SeqKD / logit 蒸馏 | 教师生成（off-policy） | 稠密（forward KL 方向） | 1 | 教师 | Hinton 2015；Alpaca 式 SFT |
| GKD | 学生生成（on-policy，可混合） | 稠密（fKL / rKL / JSD 可选） | 1 | 教师 | Agarwal et al.（Google DeepMind, ICLR 2024） |
| **OPD** | 学生生成 | 稠密（逐 token reverse KL） | 1 | 教师 | Qwen3 报告、Thinking Machines Tinker、slime |
| **MOPD** | 学生生成 + metadata 路由 | 稠密，多教师分域打分 | N | 各域教师（融合体） | KAT-Coder-V2、Kimi K3、DeepSeek-V4、Dressage |
| **G-OPD / ExOPD** | 学生生成 | 稠密 + 奖励缩放因子 $\lambda$ | 1 / N | **可超越教师** | arXiv 2602.12125（2026） |
| OPSD | 学生生成 | 稠密，教师是自己 | self | 恢复自身能力、防遗忘 | Thinking Machines 持续学习实验 |

### 阅读导航

| 章节 | 主题 | 教学重点 |
| --- | --- | --- |
| 01 | 两个老问题 | 离线蒸馏的 exposure bias 是什么？RL 的奖励为什么"稀疏"？ |
| 02 | GKD | on-policy 蒸馏的学术框架：数据 × 散度 × 混合比三个自由度 |
| 03 | OPD | 核心定义、两种等价视角、reverse KL 三性质、关键实验数字 |
| 04 | MOPD | 多教师路由如何工作？token 级 vs 全词表 KL 的分叉在哪？ |
| 05 | G-OPD | 为什么说 OPD 是 RL 的特例？奖励外推如何让学生超越教师？ |
| 06 | 本仓库实现 | slime OPD 代码走读、Dressage MOPD 架构、教学 demo |
| 07 | 总结 | 演进逻辑、设计权衡、与 RL 算法家族的关系 |

---

## 1. 两个老问题：离线蒸馏的分布偏移 与 RL 的信号稀疏

后训练（post-training）的方法可以沿两个正交维度分类：**数据从哪来**（采样是否 on-policy）与**监督信号有多密**：

| 方法 | 采样 | 奖励信号 | 问题 |
| --- | --- | --- | --- |
| SFT / 离线蒸馏 | off-policy（学别人的轨迹） | 稠密（逐 token） | 分布偏移 |
| RL | on-policy（学自己的轨迹） | 稀疏（序列级） | 信号效率低 |
| **OPD** | **on-policy** | **稠密** | 两者兼得 |

OPD 的定位一目了然：它占据的是前两种方法各自空出来的那个象限。下面分别看清两个老问题。

### 1.1 离线蒸馏：exposure bias 与"模仿风格而非准确性"

经典知识蒸馏（Hinton et al. 2015）在 LLM 时代的标准形态是 **SeqKD**：教师生成一批完整轨迹，学生对这些轨迹做 SFT。写成目标函数，这是 forward KL 方向的蒙特卡洛估计：

$$
\mathcal{L}_{\text{SeqKD}}(\theta) = \mathbb{E}_{y \sim \pi_T(\cdot|x)} \big[ -\log \pi_\theta(y \mid x) \big] \;\approx\; D_{\text{KL}}\big(\pi_T \,\|\, \pi_\theta\big) + \text{const}
$$

它便宜、稳定、信号稠密，但有两个结构性缺陷：

1. **Exposure bias / 复合误差**：学生学的是"教师会走到的状态"，而推理时学生走的是自己的路。一旦早期犯了教师从不会犯的错，学生就进入训练分布之外的状态，误差逐步放大——序列越长越致命。**学生必须学会从自己的错误中恢复，而离线数据里没有这种样本。**
2. **模仿风格而非事实准确性**：研究（*The False Promise of Imitating Proprietary LLMs*）发现，对闭源模型输出做模仿学习，学生学到的是教师的语气与自信，而不是事实能力。

### 1.2 RL：每个 episode 只教 $O(1)$ 个比特

RL 走另一条路：学生自己 rollout，环境（或奖励模型）给整条轨迹一个标量奖励。on-policy 保证了学习信号与学生实际行为分布一致，但反馈**稀疏**——Thinking Machines 的信息论视角说得很直白：**RL 每个 training episode 只教 $O(1)$ 个比特**（"这条轨迹好/不好"），而蒸馏每个 episode 教 $O(N)$ 个比特（$N$ = token 数）。

用他们的数学作业例子：学生算出"21"被判错，它知道这条轨迹不对，但不知道是运算顺序错了还是算术错了——credit assignment 只能靠大量采样去"撞"出来。国际象棋比喻同样精准：纯 RL 像没有教练的对弈（输赢一盘只反馈一次），离线蒸馏像看大师下棋（棋招极高明，但局面不是新手会走到的）。

### 1.3 核心问题

> 能否让学生**在自己的棋局里**，由大师**逐步点评每一手**？

这就是 OPD。它的思想谱系可以追溯到模仿学习中的 DAGGER（Ross et al. 2010：让专家标注**学习者实际访问到的状态**，迭代聚合数据），以及过程奖励模型（Lightman et al. 2023：给思维链的每一步打分）。OPD 的特殊之处在于：**"逐步点评"不需要训练任何新模型——教师自身的 logprob 就是奖励。**

---

## 2. GKD：on-policy 蒸馏的学术源头

Generalized Knowledge Distillation（Agarwal, Vieillard 等，Google DeepMind；arXiv:2306.13649，ICLR 2024，原标题即 *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*）是把"on-policy 蒸馏"系统化的第一个框架。

### 2.1 核心思想：从学生自己犯的错误中学习

GKD 让**学生**在训练中采样输出序列，教师在这些序列上提供监督——正是 1.3 节"在自己的棋局里被点评"的第一次完整形式化。

### 2.2 三个自由度

GKD 把蒸馏目标泛化为三个可独立选择的旋钮：

$$
\mathcal{L}_{\text{GKD}}(\theta) = \mathbb{E}_{y \sim \mathcal{D}_{\text{mix}}} \; D\big(\pi_T(\cdot|x, y_{<t}) \,\big\|\, \pi_\theta(\cdot|x, y_{<t})\big)
$$

| 旋钮 | 选项 | 效果 |
| --- | --- | --- |
| 数据来源 $\mathcal{D}_{\text{mix}}$ | 固定数据集 / 学生采样 / 两者按比例混合 | 全学生采样 = 纯 on-policy；混合是平滑过渡 |
| 散度 $D$ | forward KL / reverse KL / 广义 JSD($\beta$) | 决定"对齐"的行为语义（见 2.3） |
| 与 RL 的组合 | 可叠加环境奖励 | 论文已验证 on-policy GKD + RL 反馈可提升事实一致性 |

其中广义 JSD 的定义为 $\text{JSD}_\beta(P, Q) = \beta\, D_{\text{KL}}(P \| M) + (1-\beta)\, D_{\text{KL}}(Q \| M)$，$M = \beta P + (1-\beta) Q$。

### 2.3 Forward KL 与 reverse KL 的行为差异

散度方向是蒸馏里最反直觉也最重要的选择：

| | forward KL：$D_{\text{KL}}(\pi_T \| \pi_\theta)$ | reverse KL：$D_{\text{KL}}(\pi_\theta \| \pi_T)$ |
| --- | --- | --- |
| 采样分布 | 教师轨迹（off-policy） | 学生轨迹（on-policy） |
| 行为 | **mode-covering**：学生的概率要覆盖教师的所有模式，覆盖不了就摊薄 | **mode-seeking**：学生锁定教师的一个高概率模式集中火力 |
| 典型后果 | 输出"平均化"、模糊 | 尖锐、贴近教师的强项行为 |

对小模型蒸馏，reverse KL 通常更优：容量有限时，"把教师最强的一种行为学精"优于"把所有行为都学个大概"。同期的 MiniLLM（Gu et al. 2023）正是用 reverse KL + policy gradient 做蒸馏——已经是非常接近 OPD 的形态。

### 2.4 GKD 的局限

GKD 给出了完整框架，但验证停留在学术规模（摘要、翻译、算术推理等任务、较小模型）。它留下三个工程问题等待工业界回答：教师打分怎么组织进现有 RL 基建？多教师怎么调度？目标函数还有没有更本质的理解？——这分别对应第 3、4、5 节。

---

## 3. OPD：工业范式的确立

2025 年，on-policy 蒸馏完成了从学术框架到工业范式的跳跃：Qwen3 技术报告用它以 RL 十分之一的成本反超 RL，Thinking Machines 的博客（Kevin Lu，2025 年 10 月）则用 Tinker 平台完整复现并系统化了这套配方。slime 的实现注释里直接引用了 Tinker cookbook 的参考实现——血统清晰。

### 3.1 核心定义与四步流程

**OPD：学生从自身策略采样轨迹，教师对轨迹上每个 token 提供对数概率，以逐 token reverse KL 为稠密监督信号更新学生。** 流程：

1. **学生采样**：$y \sim \pi_\theta(\cdot|x)$，与 RL 完全一样（学生 logprob $\log \pi_\theta(y_t|x, y_{<t})$ 在 rollout 时顺手记录，重要性采样损失本来就要用）
2. **教师打分**：对同一条轨迹计算 $\log \pi_T(y_t|x, y_{<t})$——教师只需**一次前向传播**（compute_logprobs），不需要生成任何东西
3. **计算逐 token reverse KL**：

$$
d_t = \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_T(y_t \mid x, y_{<t})
$$

4. **更新**：把 $-d_t$ 当作逐 token 优势（或叠加到任务优势上），走标准的 policy gradient / 重要性采样损失

### 3.2 两个等价视角

**KL 蒸馏视角**：最小化学生轨迹上的 reverse KL（Thinking Machines 版式，折扣因子取 0，每步只优化当下这一个 token）：

$$
\min_\theta \; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot|x)} \Big[ D_{\text{KL}}\big(\pi_\theta(\cdot|x, y_{<t}) \,\big\|\, \pi_T(\cdot|x, y_{<t})\big) \Big]
$$

当学生与教师行为一致时 reverse KL 为 0——目标清晰、有下界、可监控。

**RL 稠密奖励视角**：把 reverse KL 塞进 RL 的奖励/优势通道。slime 的做法是**优势级融合**（见 6.1 节代码走读）：

$$
\hat{A}_t = A_t - \lambda_{\text{opd}} \cdot \big( \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_T(y_t \mid x, y_{<t}) \big)
$$

由此得到一个重要洞察：**OPD 是 RL 实现的"一行改动"**——任何带 KL 正则的 RL 训练栈，把正则的参考模型换成教师模型即得 OPD。两个极端帮助定位这个旋钮：

- $\lambda_{\text{opd}} \to 0$：退化为普通 RL（教师只留一点约束）
- 任务奖励置 0 + $\lambda_{\text{opd}}$ 调大：退化为**纯 on-policy 蒸馏**，学习信号全部来自教师

### 3.3 为什么是 reverse KL：三个好性质

Thinking Machines 总结了 reverse KL 在此场景的三个独特优势：

1. **不可作弊（unhackable）**：KL 低当且仅当学生行为在教师视角下是高概率的好行为。与学习型奖励模型不同，这里没有可被策略钻空子的近似器——奖励黑客（reward hacking）无处下手
2. **Mode-seeking**：学生学精教师的一种强项行为，而不是在多个次优选项间摊薄概率（与 2.3 节对照）
3. **治 exposure bias**：训练分布 = 学生的实际行为分布，1.1 节的复合误差问题在结构上消失

### 3.4 工程红利

OPD 在工程上还有一串容易被忽视的优点：

| 红利 | 原因 |
| --- | --- |
| 教师开销极小 | 教师只做前向打分，生成全部由更小的学生完成 |
| 天然兼容 partial rollout | 没有"序列终点奖励"的概念，每个 token 自带奖励——轨迹不需要采完就能训（与本仓库 Dressage 的 partial rollout / multi-segment 基建无缝兼容） |
| 无需独立奖励模型 | 奖励 = 两个 logprob 之差，省掉 RM 的训练与部署 |
| 数据可复用 | RL 对同一 prompt 多 epoch 会背答案；OPD 学的是教师的完整分布，同一 prompt 反复采样训练也不崩 |

### 3.5 关键实验数字

**Qwen3 技术报告（Table 21，8B 量级，AIME'24 / GPQA-Diamond）**：

| 方法 | AIME'24 | GPQA-D | GPU 小时 |
| --- | --- | --- | --- |
| 离线蒸馏（SFT） | 55.0% | 55.6% | 未报告 |
| + RL | 67.6% | 61.3% | 17,920 |
| + OPD | **74.4%** | **63.3%** | **1,800** |

OPD 用 RL **十分之一**的成本反超 RL 近 7 个点——这是整个范式最有说服力的广告。

**Thinking Machines 复现**（学生 Qwen3-8B-Base，教师 Qwen3-32B，任务数学推理）：从 400K prompt 的 SFT checkpoint（AIME'24 60%）出发，OPD 约 150 步（77K prompt × 4 采样）即达 70%。成本对比（达到约 70% 的口径）：

| 方法 | AIME'24 | 相对成本 |
| --- | --- | --- |
| SFT-2M（外推） | ~70% | 1× |
| RL | ~68% | ≈1× |
| OPD | 70% | **1/9 - 1/30** |

**自蒸馏实验**（最能说明信号密度价值的实验）：先用 RL 把 Qwen3-8B-Base 训成教师，再用 OPD 把这个教师蒸回同一个基座——**7-10 倍更少的梯度步**即复现教师水平，叠加"蒸馏可用更短上下文与更小 batch"两个因素，累积算力效率提升 **50-100 倍**。

**持续学习实验**（教师=学生自己的历史版本，即 OPSD）：Qwen3-8B 在内部文档上 midtrain 后 IF-eval 从 85% 掉到 45-79%（灾难性遗忘）；以 midtrain 前的自己为教师做 OPD，IF-eval 恢复到 83% 且新学的知识不丢。对照实验很有说服力：即使拿模型**自己采样**的数据（KL=0）做 SFT，只要学习率非零，有限批次效应仍会让行为逐步退化——SFT 的更新方向不对齐当前策略，而 OPD 始终 on-policy。

### 3.6 哲学：RL 是搜索，蒸馏是抄近路

TM 博客给出的最深一层解释值得记住：**RL 的大部分算力并不花在梯度更新上，而是花在搜索上**——在语义策略的空间里试错、分配 credit。一旦好策略被找到，蒸馏就是一条抄近路的学习通道：OPD 不需要复现 RL 课程中的所有中间策略，只需要学最终那一个。

> **类比**　科研发现需要漫长的探索，但一旦答案被找到，把它教给别人只需要用自然语言讲一遍——RL 是前者，OPD 是后者。反之，如果目标不是复制已知策略而是**发现新策略**（没有教师覆盖的区域），RL 不可替代——这为第 5 节"超越教师"埋下伏笔。

---

## 4. MOPD：多教师与 Specialize-then-Unify

### 4.1 问题与范式

2025-2026 年头部实验室收敛到同一个后训练范式——**Specialize-then-Unify**：

1. 按领域把能力拆开，各领域专家**独立**做 SFT + RL（避免多领域混训的梯度冲突与负迁移）
2. 用多教师 OPD（MOPD）把 $N$ 个专家在**分布层面**融合成单一可部署模型

为什么不用权重合并（model merging）？因为权重空间合并是粗暴的算术平均，而蒸馏在**分布层面**融合——学生学到的是各教师在真实轨迹上的行为分布，能力保真度更高。形式化地，给定 $N$ 个专家 $\{\pi_{E_1}, \ldots, \pi_{E_N}\}$，目标是：

$$
\mathcal{L}_{\text{MOPD}}(\theta) = \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}\big(\pi_\theta \,\|\, \pi_{E_i}\big)
$$

### 4.2 KAT-Coder-V2：五领域专家的融合

KAT-Coder-V2（本仓库 [kat-coder-v2.md](./kat-coder-v2.md)）把智能体编程拆成 5 个专家领域（SWE / WebCoding / Terminal / WebSearch / General），各自独立 SFT + RL，最后用 OPD 融合成一个可部署模型——"Specialize-then-Unify"的完整落地案例。

### 4.3 Kimi K3：逐 token 对数比奖励

Kimi K3（本仓库 [kimi.md](./kimi.md) §4.1.3）的场景更极致：RL 训出 9 个专家（3 个领域 × 3 档推理努力），MOPD 负责合并回一个统一模型。给定领域 $d$ 与采样到的努力档 $e$，用对应教师 $\pi^{(d,e)}_{\text{teacher}}$ 指导学生，逐 token OPD 奖励（报告 Eq. 15）：

$$
r^d_{\text{opd}}(y_t \mid e, x, y_{<t}) = \mathrm{clip}\!\left( \mathrm{sg}\!\left( \log \frac{\pi^{(d,e)}_{\text{teacher}}(y_t \mid x, y_{<t})}{\pi_\theta(y_t \mid e, x, y_{<t})} \right),\; -R_{\max},\; R_{\max} \right)
$$

逐符号解读：

- $\log(\pi_T / \pi_\theta)$：教师比学生更喜欢这个 token（比值 > 1）→ 奖励为正 → 鼓励学生提高其概率
- $\mathrm{sg}(\cdot)$（stop-gradient）：奖励当常数，不对教师反传梯度
- $\mathrm{clip}(\cdot, \pm R_{\max})$：裁掉极端优势信号，稳定训练

两个工程判断值得注意：这个稠密逐 token 奖励**无缝接入 K3 现有 RL 框架**，天然享受 partial rollout 等基建优化；报告还提到他们试过更细粒度的 top-k 蒸馏目标，但在其设定下收敛速度与最终性能均无显著优势，故从简。**简单形式的逐 token 对数比在超大规模上已经足够好**——这是一个重要的负结果。

### 4.4 Dressage MOPD：本仓库的 metadata 路由实现

本仓库的 MOPD 实现（详见 [mopd-architecture.md](../mopd-architecture.md)）在 slime 单教师 OPD 之上做了三点扩展：

1. **Metadata 路由**：每条训练样本携带 `teacher_id`，指定该样本的权威教师；多个数据集按权重混合采样，各绑定一个教师
2. **权重轮转打分**：所有教师权重复用学生 actor 的 GPU 缓冲区——教师权重一次性加载到 pinned-CPU 的 `TensorBackuper`，训练时按教师分组，依次把对应教师权重恢复到 GPU、只对该教师的样本子集计算 logprob，再散射回原 batch 顺序。**GPU 显存不随教师数增长，只有 CPU 内存线性增长**
3. **零 slime 源码修改**：通过 slime 原生的 `actor_cls` factory hook 注入自定义 actor

硬约束：**学生与所有教师必须同架构、同 tokenizer、同词表**——否则逐 token logprob 不可比。若需异构教师（不同架构/尺寸），应走 slime OPD 的 `sglang` 模式（教师在外部推理服务上，见 6.1 节）。

### 4.5 DeepSeek-V4：全词表 OPD 与"为什么别人不做"

DeepSeek-V4（本仓库 [deepseek-v4.md](./deepseek-v4.md) §8.4）把 MOPD 推到 10+ 个万亿/千亿级教师的规模，并做出一个关键的技术分叉选择：**全词表 logit 蒸馏**。

| | token 级 KL 估计（3.2 节形式） | 全词表 KL（V4） |
| --- | --- | --- |
| 估计方式 | 只看实际生成的那个 token 的对数比，当优势塞进策略损失 | 保留教师在整个词表上的完整分布，解析计算 KL |
| 梯度性质 | 采样噪声大、训练不稳 | 解析地消掉采样噪声，更稳 |
| 工程成本 | 便宜，复用 RL 框架 | 昂贵——需要专门调度 |

V4 让全词表 OPD 可扩展的四项工程（§5.2.2）：

1. **教师权重 FP4 离线存储**，ZeRO 式分片、按需加载——10+ 个巨模型教师不同时占显存
2. **只缓存教师最后一层隐状态**（而非全词表 logits，后者是"词表 × 序列长"的天文数字），训练时经对应教师的预测头**即时重建 logits**
3. **按教师索引排序训练样本**：保证一个 mini-batch 内每种教师的预测头只加载一次
4. **TileLang 专用内核**计算精确 KL 散度

配套的还有 1M 上下文 RL/OPD 的数据工程（轻量元数据全局调度 + 逐 token 重型字段即用即释放）。一句话：**不是别人不想做全词表，是存储与调度成本不划算；V4 改掉成本结构后，更优的统计性质才变得可用。**

---

## 5. G-OPD：OPD 是 RL 的一个特例

前面所有工作的隐含假设是"学生的天花板 = 教师"。G-OPD（*Learning Beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation*，arXiv:2602.12125，2026）用一个理论观察打破了这个天花板。

### 5.1 理论：奖励与 KL 被锁死成 1:1

标准 OPD 目标（$\pi_*$ 为教师）：

$$
J_{\text{OPD}}(\theta) = \min_\theta \; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta} \big[ D_{\text{KL}}\big(\pi_\theta(y|x) \,\|\, \pi_*(y|x)\big) \big]
$$

引入**任意**参考模型 $\pi_{\text{ref}}$ 做代数改写，可以重排为：

$$
J_{\text{OPD}}(\theta) = \max_\theta \; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta} \left[ \underbrace{\log \frac{\pi_*(y|x)}{\pi_{\text{ref}}(y|x)}}_{\text{隐式奖励}} - \underbrace{D_{\text{KL}}\big(\pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x)\big)}_{\text{KL 正则}} \right]
$$

这正是**稠密 KL 约束 RL**的标准形态（隐式奖励的形式与 DPO 的 $\log(\pi/\pi_{\text{ref}})$ 同源），只不过 OPD 把两样东西**锁死**了：

- 奖励项与 KL 项的相对权重恒为 **1:1**
- 参考模型可以是任意模型，但标准 OPD 没有利用这个自由度

一旦看出这一点，自然的问题就是：为什么要锁死？

### 5.2 G-OPD 的两个旋钮

G-OPD 解耦这两个自由度：

$$
J_{\text{G-OPD}}(\theta) = \max_\theta \; \mathbb{E} \left[ \lambda \log \frac{\pi_*(y|x)}{\pi_{\text{ref}}(y|x)} - D_{\text{KL}}\big(\pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x)\big) \right]
$$

| $\lambda$ 取值 | 名称 | 行为 |
| --- | --- | --- |
| $0 < \lambda < 1$ | 奖励内插 | 学生行为在参考模型与标准 OPD 之间插值，可控缩放 |
| $\lambda = 1$ | 标准 OPD | 恢复原始目标，天花板 = 教师 |
| $\lambda > 1$ | **奖励外推（ExOPD）** | 鼓励学生沿"教师相对参考模型的改进方向"继续走，可**超越教师** |

直觉：$\log(\pi_* / \pi_{\text{ref}})$ 度量的是"教师从参考模型出发学到了什么"；$\lambda > 1$ 就是让学生在这个**改进方向**上比教师走得更远。

### 5.3 关键实验数字

实验设置：Qwen3 系列模型；数学（DeepMath 数据，AIME / HMMT 评测）与代码（Eurus-RL-Code 数据，HumanEval+ / MBPP+ / LiveCodeBench 评测）两个领域；教师由 GRPO 训出。

| 发现 | 数字 |
| --- | --- |
| 单教师 ExOPD（$\lambda = 1.25$ 最佳） | 平均超教师 **+2.0%**（数学）、**+0.9%**（代码） |
| 多教师融合 | ExOPD 是**唯一**让统一学生全面超过所有领域专科教师的方法；SFT 与标准 OPD 融合后通常仍不如专科教师 |
| 强到弱蒸馏（30B → 4B）+ 奖励修正 | 用教师的 RL 前基座做 $\pi_{\text{ref}}$，比标准 OPD 再 **+2.7%** |

**奖励修正**的动机：$\log(\pi_* / \pi_{\text{teacher\_base}})$ 捕捉的是**教师自己的学习轨迹**，比用容量不匹配的学生基座做参考更准确。代价是需要访问教师的 RL 前版本，且计算开销更大。

### 5.4 边界与风险

- **过度外推失稳**：$\lambda = 1.5$ 时训练可能不稳定，疑似长度偏差与对噪声隐式奖励的过拟合
- **行为漂移**：ExOPD 的响应一致性地更长、熵更高——多样性提升，但也提示外推在放大探索倾向
- **天花板依然存在**：没有可靠任务奖励时，外推方向的"正确性"仍由教师的学习轨迹背书

### 5.5 前沿速览

- **Uni-OPD**（2026）：用结果引导的边界校准统一 OPD 目标，报告以比传统 RL 更少的优化步数收敛，并缓解教师信号在分布外区域的退化（媒体解读口径，细节以原文为准）
- **OPSD**（On-Policy Self-Distillation）：教师取学生自身（或历史 checkpoint），用于持续学习/防遗忘——3.5 节的持续学习实验即此形态
- **训练动态分析**：*Rethinking On-Policy Distillation of Large Language Models*（arXiv:2604.13016，2026）开始系统研究 OPD 的训练动力学
- 多教师 OPD 的工业采用还在继续扩大：MiMo-v2-flash 技术报告（arXiv:2601.02780）等同样以 OPD 做领域专家融合

---

## 6. 本仓库实现参考

### 6.1 slime OPD：两个文件看懂全链路

**（1）教师打分**：[on_policy_distillation.py](../../slime/slime/rollout/on_policy_distillation.py)（`sglang` 模式）。

- `reward_func`：**输入**一个 `Sample`（含 `tokens`）；**输出**教师服务返回的逐 token logprob JSON。逻辑：向教师 SGLang 服务发一个纯打分请求——`sampling_params` 里 `temperature=0, max_new_tokens=0`（一个 token 都不生成），`return_logprob=True, logprob_start_len=0`（全序列返回 logprob）
- `post_process_rewards`：**输入**样本列表 + 教师响应；**输出**两件事——把教师 logprob 截尾对齐到 response 长度后挂到 `sample.teacher_log_probs`，以及返回标量奖励 `0.0`。**奖励为 0 不是 bug**：纯蒸馏场景下学习信号全部来自后续的 OPD KL 惩罚（代码注释明确说明）

**（2）优势融合**：[loss.py](../../slime/slime/backends/megatron_utils/loss.py) 的 `apply_opd_kl_to_advantages`。

- **输入**：`args`（含 `opd_kl_coef`）、`rollout_data`（含 `teacher_log_probs`）、`advantages` 列表、学生 `student_log_probs`
- **输出**：无返回值——**原地**改写 `advantages`
- **逻辑**：逐样本 `reverse_kl = student_logp - teacher_logp`，然后 `adv -= opd_kl_coef * reverse_kl`；把 `reverse_kl` 存进 `rollout_data["opd_reverse_kl"]` 供监控。docstring 直接引用 Thinking Machines 的 [tinker-cookbook 参考实现](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py)

它与优势估计器完全解耦——`compute_advantages_and_returns` 先用 GRPO/GSPO/PPO 等算出 $A_t$，再叠加 OPD 项，下游 loss 构造无感。

**两种教师接入模式**：

| 模式 | 教师位置 | logprob 计算时机 | 架构要求 |
| --- | --- | --- | --- |
| `sglang` | 外部 SGLang 推理服务 | rollout 阶段经 HTTP 获取 | 可异构 |
| `megatron` | 训练进程内（`--opd-teacher-load`） | 训练前向时 | 教师与学生同架构 |

### 6.2 Dressage MOPD

多教师扩展的完整架构见 [mopd-architecture.md](../mopd-architecture.md)（权重轮转、metadata 路由、`actor_cls` hook，4.4 节已展开），测试入口为 [test_mopd.py](../../tests/test_mopd.py) 与 [test_mopd_metrics.py](../../tests/test_mopd_metrics.py)。

### 6.3 教学 demo

[kimi.md](./kimi.md) §4.1.3 内嵌了一个可运行的 `mopd_reward.py` 教学 demo，用四个案例把 Eq. 15 讲透：教师=学生时奖励 ≈ 0；教师更自信时奖励为正；极端不匹配时 clip 恰好生效；`detach` 实现 stop-gradient。适合作为手写练习的起点。

---

## 7. 总结

### 7.1 演进逻辑

```
SeqKD / logit 蒸馏（Hinton 2015 → Alpaca 式 SFT）
 │ ✅ 稠密监督、便宜稳定
 │ ❌ off-policy：exposure bias；模仿风格而非事实准确性
 │  让学生自己采样，教师在学生轨迹上打分
 ▼
GKD（Agarwal et al., Google DeepMind, ICLR 2024）
 │ ✅ on-policy 蒸馏的系统框架：数据 × 散度 × 混合比三自由度
 │ ❌ 学术规模验证，未接入工业 RL 基建
 │  简化定型：per-token reverse KL + 复用 RL 框架（"RL 的一行改动"）
 ▼
OPD（Qwen3 报告 / Thinking Machines，2025）
 │ ✅ 1/10 成本反超 RL（AIME 74.4 vs 67.6）；自蒸馏 50-100x 算力效率
 │ ✅ reverse KL 三性质：unhackable、mode-seeking、治 exposure bias
 │ ✅ 天然兼容 partial rollout；教师只需前向打分
 │ ❌ 单教师；天花板 = 教师
 │  多教师分域打分 → specialize-then-unify
 ▼
MOPD（KAT-Coder-V2 / Kimi K3 / DeepSeek-V4 / Dressage，2025-2026）
 │ ✅ N 个专家 → 1 个可部署模型，分布级融合
 │ ✅ V4 全词表 KL 解析消噪（FP4 存教师 + 隐状态缓存 + TileLang 内核）
 │ ❌ 天花板仍 = 教师（融合体）
 │  理论统一：OPD ≡ 奖励:KL = 1:1 锁死的稠密 KL 约束 RL
 ▼
G-OPD / ExOPD（arXiv 2602.12125，2026）
 │ ✅ 解耦 λ：λ>1 奖励外推 → 学生反超教师（数学 +2.0%）
 │ ✅ 奖励修正：以教师的 RL 前基座为参考，强到弱再 +2.7%
 │ ⚠ λ 过大失稳；修正需访问教师 RL 前版本
```

### 7.2 核心设计权衡

| 设计维度 | 选项 | 权衡 |
| --- | --- | --- |
| 数据分布 | off-policy（教师生成）vs on-policy（学生生成） | on-policy 治 exposure bias，但要付采样成本 |
| 散度方向 | forward KL vs reverse KL | 蒸馏小模型优先 reverse KL（mode-seeking）；forward KL 用于 SFT 冷启动扩充支持集 |
| 信号粒度 | token 级估计 vs 全词表 KL | token 级便宜但有采样噪声；全词表稳但调度昂贵（V4 才付得起） |
| 教师组织 | 单教师 vs MOPD 路由 | MOPD 支持 specialize-then-unify，硬约束是同架构/同词表 |
| 奖励-KL 权重 | 锁死 1:1（OPD）vs 解耦 λ（G-OPD） | λ>1 可超越教师，但失稳风险随 λ 增大 |
| 与任务奖励的关系 | 纯蒸馏（奖励置 0）vs 优势级混合 | 同一优势通道加权，$\lambda_{\text{opd}}$ 是连续旋钮 |

### 7.3 与 RL 算法家族的关系

OPD 不是 PPO/GRPO/GSPO/DAPO 的替代品，而是**叠加在它们之上的信号通道**（见 [llm-rl-algorithms-zh.md](../llm-rl-algorithms-zh.md) §9.4）：优势估计器负责"方向对不对"（任务奖励），OPD 负责"老师具体怎么做"（分布对齐），两者在同一优势里加权。G-OPD 的理论则进一步说明：蒸馏与 RL 本来就是同一个目标函数族上的不同参数点——**蒸馏是稠密奖励的特例，RL 是 λ 自由的蒸馏**。

### 7.4 一句话总结

> OPD 的洞察是"教师的 logprob 就是免费的稠密奖励"：学生在自己的轨迹上被逐 token 点评，兼得 RL 的 on-policy 相关性与蒸馏的信号密度；MOPD 把它扩展成多专家的分布级融合，成为 2026 年旗舰模型能力合并的标准件；G-OPD 再把它统一进 KL 约束 RL 的理论框架，用奖励外推捅破教师天花板——**蒸馏的终点不是复制教师，而是把教师变成搜索方向的初值。**

---

## 参考文献

- Hinton, Vinyals & Dean (2015). *Distilling the Knowledge in a Neural Network.* [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Ross, Gordon & Bagnell (2010). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* (DAGGER) [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
- Agarwal et al. (2023, ICLR 2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes / GKD.* [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- Gu et al. (2023). *MiniLLM: Knowledge Distillation of Large Language Models.* [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- Gudibande et al. (2023). *The False Promise of Imitating Proprietary LLMs.* [arXiv:2305.15717](https://arxiv.org/abs/2305.15717)
- Lightman et al. (2023). *Let's Verify Step by Step.* (过程奖励模型) [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* (隐式奖励 $\log \pi / \pi_{\text{ref}}$) [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Qwen Team (2025). *Qwen3 Technical Report.* (Table 21：OPD 以 1/10 成本反超 RL) [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- Lu, Kevin & Thinking Machines Lab (2025). *On-Policy Distillation.* (Connectionism 博客，Tinker cookbook 实现) [doi:10.64434/tml.20251026](https://thinkingmachines.ai/blog/on-policy-distillation/)
- DeepSeek-AI (2025). *DeepSeek-V3.2-Exp.* (DSA；专家蒸馏合并)（见本仓库 [deepseek-v4.md](./deepseek-v4.md)）
- Kimi Team. *Kimi K3 技术报告.*（MOPD：9 专家 → 1 学生，逐 token 对数比奖励）（见本仓库 [kimi.md](./kimi.md)）
- KAT Team. *KAT-Coder-V2 技术报告.*（Specialize-then-Unify：5 领域专家 + OPD 融合）（见本仓库 [kat-coder-v2.md](./kat-coder-v2.md)）
- DeepSeek-AI. *DeepSeek-V4 技术报告.*（全词表多教师 OPD 与工程调度）（见本仓库 [deepseek-v4.md](./deepseek-v4.md)）
- *Learning Beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation.* (G-OPD / ExOPD) [arXiv:2602.12125](https://arxiv.org/abs/2602.12125)，代码：[RUCBM/G-OPD](https://github.com/RUCBM/G-OPD)
- *Rethinking On-Policy Distillation of Large Language Models.* (OPD 训练动态分析) [arXiv:2604.13016](https://arxiv.org/abs/2604.13016)
- Xiao et al. (2026). *MiMo-v2-flash Technical Report.*（多教师 OPD 融合实例） [arXiv:2601.02780](https://arxiv.org/abs/2601.02780)
- Zheng et al. (2025). *Model Extrapolation Expedites Alignment.* (ExPO，权重外推基线) ACL 2025
