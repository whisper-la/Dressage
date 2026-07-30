# LLM 训练中的强化学习算法详解

> **本文档目的**：系统讲解大语言模型（LLM）训练中用到的强化学习算法——从策略梯度的数学基础，到 PPO / GRPO / GSPO / DAPO / DPO 等主流算法的原理、损失函数与工程取舍，并映射到 Dressage/slime 中实际使用的训练参数。
>
> **适用读者**：希望搞清楚"LLM 的 RL 训练到底在优化什么、每个算法/超参数在做什么"的研究者和工程师。
>
> **与 [agentic-rl-training-zh.md](agentic-rl-training-zh.md) 的关系**：那篇文档讲的是 Dressage 的**系统架构**（Proxy / Paddock / Rollout / 轨迹构建）；本篇讲的是**算法本身**（优势估计、损失函数、KL、clip）。两篇互补，建议配合阅读。
>
> **阅读建议**：第一至二章建立数学基础；第三章是 RLHF 的经典方案 PPO；第四至六章是当前 LLM RL 主流的"去 critic"路线（GRPO/GSPO/CISPO/DAPO）；第七章讲不需要 rollout 的 DPO；第八章横向拆解所有算法共享的损失组件；第九章把这些算法映射到 Dressage/slime 的真实 CLI 参数。

---

## 目录

- [一、总览：LLM 为什么要用 RL](#一总览llm-为什么要用-rl)
- [预备知识：信息熵与 KL 散度](#预备知识信息熵与-kl-散度)
- [二、强化学习基础（LLM 语境）](#二强化学习基础llm-语境)
  - [2.1 把 LLM 生成建模成 MDP](#21-把-llm-生成建模成-mdp)
  - [2.2 策略梯度定理](#22-策略梯度定理)
  - [2.3 REINFORCE 与 baseline](#23-reinforce-与-baseline)
  - [2.4 价值函数、优势函数与 GAE](#24-价值函数优势函数与-gae)
  - [2.5 重要性采样：原理深化与工程落点](#25-重要性采样原理深化与工程落点)
- [三、RLHF 的经典方案：PPO](#三rlhf-的经典方案ppo)
  - [3.1 RLHF 三阶段流程](#31-rlhf-三阶段流程)
  - [3.2 PPO 的裁剪目标](#32-ppo-的裁剪目标)
  - [3.3 KL 惩罚：不要跑太远](#33-kl-惩罚不要跑太远)
  - [3.4 PPO 完整目标与四个模型](#34-ppo-完整目标与四个模型)
- [四、去掉 critic：GRPO](#四去掉-criticgrpo)
  - [4.1 核心思想：组内相对优势](#41-核心思想组内相对优势)
  - [4.2 GRPO 目标函数](#42-grpo-目标函数)
  - [4.3 GRPO 与 PPO 的对比](#43-grpo-与-ppo-的对比)
  - [4.4 近亲：RLOO 与 Reinforce++ baseline](#44-近亲rloo-与-reinforce-baseline)
  - [4.5 GRPO 的数学偏差与 Dr. GRPO](#45-grpo-的数学偏差与-dr-grpo)
- [五、GSPO 与 CISPO：重要性采样的稳定化](#五gspo-与-cispo重要性采样的稳定化)
  - [5.1 GSPO：序列级重要性采样](#51-gspo序列级重要性采样)
  - [5.2 CISPO：裁剪 IS 权重，保留所有 token 的梯度](#52-cispo裁剪-is-权重保留所有-token-的梯度)
- [六、DAPO：面向长 CoT 的工程改进](#六dapo面向长-cot-的工程改进)
- [七、免 Reward Model 的偏好优化：DPO](#七免-reward-model-的偏好优化dpo)
  - [7.1 DPO 的变体家族](#71-dpo-的变体家族)
- [八、横向拆解：损失函数的公共组件](#八横向拆解损失函数的公共组件)
  - [8.1 重要性采样比率](#81-重要性采样比率)
  - [8.2 裁剪与 dual-clip](#82-裁剪与-dual-clip)
  - [8.3 KL 散度的三种估计](#83-kl-散度的三种估计)
  - [8.4 熵正则](#84-熵正则)
  - [8.5 优势归一化](#85-优势归一化)
  - [8.6 损失聚合：token-mean vs sequence-mean](#86-损失聚合token-mean-vs-sequence-mean)
  - [8.7 训练-推理不一致与离线策略修正](#87-训练-推理不一致与离线策略修正)
  - [8.8 MoE 模型 RL 的路由稳定性](#88-moe-模型-rl-的路由稳定性)
- [九、这些算法在 Dressage/slime 中的落地](#九这些算法在-dressageslime-中的落地)
  - [9.1 参数到概念的映射表](#91-参数到概念的映射表)
  - [9.2 一个真实的 GRPO 配置](#92-一个真实的-grpo-配置)
  - [9.3 Dressage 的优势后处理](#93-dressage-的优势后处理)
  - [9.4 OPD：on-policy 蒸馏与 RL 的混合](#94-opdon-policy-蒸馏与-rl-的混合)
  - [9.5 Agentic 场景的 credit assignment](#95-agentic-场景的-credit-assignment)
- [十、算法选型与常见陷阱](#十算法选型与常见陷阱)
- [十一、总结与参考](#十一总结与参考)

---

## 一、总览：LLM 为什么要用 RL

**结论：SFT（监督微调）只能模仿"标注给出的答案"，无法直接优化"我们真正在意的目标"（人类偏好、答案是否正确、代码是否通过测试）。RL 让模型以一个标量奖励为目标去探索、试错，从而优化那些无法逐 token 标注、只能事后评价好坏的目标。**

三条主线贯穿 LLM 的 RL 训练，它们的差异本质是**奖励从哪来**和**优势怎么估计**：

| 路线 | 代表算法 | 奖励来源 | 是否需要 critic | 是否需要 rollout |
|---|---|---|---|---|
| RLHF（偏好对齐） | PPO | 训练好的 Reward Model | 需要（value model） | 需要 |
| 可验证奖励 RLVR | GRPO / GSPO / CISPO / DAPO | 规则/验证器（答案对错、测试通过） | 不需要（组内基线） | 需要 |
| 离线偏好优化 | DPO / IPO / KTO | 偏好对 `(chosen, rejected)` | 不需要 | 不需要（离线） |

算法谱系（按"如何降低策略梯度方差 / 如何估计优势"演化）：

```
                        策略梯度定理 ∇J = E[∇logπ · A]
                                   │
                     REINFORCE（A = 蒙特卡洛回报）
                                   │  加 baseline 降方差
                     REINFORCE + baseline
                          ┌────────┴─────────┐
              学一个 value 网络             用一组样本的均值当 baseline
                    │                              │
                  Actor-Critic                 无 critic
                    │                    ┌────────┼─────────┐
                   PPO                 RLOO     GRPO      Reinforce++
              (clip + GAE + KL)                  │
                                        ┌────────┬────────┬──────────┐
                                      GSPO    CISPO     DAPO     Dr. GRPO
                                 (序列级重要性)(裁剪IS权重)(clip-higher/ (去 std 归一化
                                                           动态采样/    + token 级聚合)
                                                           token-level loss)
```

Dressage 走的是**RLVR 路线**：底层训练引擎 slime 的 `--advantage-estimator` 支持六种取值——`grpo`、`gspo`、`cispo`、`reinforce_plus_plus`、`reinforce_plus_plus_baseline`、`ppo`（choices 定义见 [arguments.py](../slime/slime/utils/arguments.py) 第 931-946 行；仅 `ppo` 会启用 critic），示例脚本默认用 GRPO。Dressage 自己的优势后处理层对其中 `grpo`/`gspo`/`reinforce_plus_plus_baseline` 三种做组内归一化（见 [reward_post_process.py](../dressage/training/reward_post_process.py) 第 118-125 行的判断）。因此本文重点讲 GRPO/GSPO/CISPO/DAPO，同时把它们的"祖先"PPO 讲透，这样每个超参数的来历都能说清楚。

---

## 预备知识：信息熵与 KL 散度

**结论：信息熵衡量单个概率分布的"不确定性"，KL 散度衡量两个分布之间的"差异"。它们是理解后续 PPO/GRPO 损失函数中 `entropy_loss`（熵正则）和 `kl_loss`（KL 惩罚）的基础。两者通过交叉熵联系：`KL(P‖Q) = H(P, Q) − H(P)`。**

### 信息熵

信息熵衡量分布 `P` 的不确定性（随机性）：

$$H(P) = -\sum_x P(x)\log P(x)$$

- 分布越**均匀**（越不确定）→ 熵越大
- 分布越**尖锐**（越确定，如 one-hot）→ 熵越小（最小为 0）
- 在 RL 中，策略 `π_θ` 的熵 `H(π_θ)` 就是 `entropy_loss`，鼓励策略保持输出多样性、避免过早坍缩为确定性输出

### 交叉熵

交叉熵衡量"用分布 `Q` 编码来自分布 `P` 的数据"的平均代价：

$$H(P, Q) = -\sum_x P(x)\log Q(x)$$

- 当 `P = Q` 时，`H(P, Q) = H(P)`（退化为自熵）
- 当 `P` 固定时，最小化交叉熵等价于让 `Q` 逼近 `P`

### KL 散度

KL 散度（Kullback-Leibler divergence）衡量两个分布之间的差异：

$$\mathrm{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} = H(P, Q) - H(P)$$

关键性质：

| 性质 | 说明 |
|---|---|
| **非负** | `KL(P‖Q) ≥ 0`，当且仅当 `P = Q` 时为 0 |
| **不对称** | `KL(P‖Q) ≠ KL(Q‖P)`（所以叫"散度"而非"距离"） |
| **与交叉熵的关系** | `KL = 交叉熵 − 自熵`；当 `P` 固定时 `H(P)` 为常数，最小化交叉熵与最小化 KL 等价 |

**KL 散度与交叉熵的区别**：分类任务中标签 `P` 是 one-hot（`H(P) = 0`），所以 `KL = 交叉熵`，两者完全相等——这就是为什么 PyTorch 的 `CrossEntropyLoss` 在 one-hot 标签下等价于 KL 散度。RL 中参考策略 `π_ref` 不是 one-hot（`H(π_ref) > 0`），所以 KL 与交叉熵差一个常数——优化等价但数值不同。用 KL 而非纯交叉熵的原因是 KL 有明确的"距离"语义（≥0，相同为 0），更适合做监控指标。

**在 RL 中的两个角色**（后续章节详述）：

| 概念 | 在损失中的角色 | 方向 | Dressage 参数 |
|---|---|---|---|
| `H(π_θ)`（信息熵） | 熵正则 `entropy_loss`：鼓励探索，防止策略过早坍缩 | 要**大** | `--entropy-coef` |
| `KL(π_new‖π_ref)` | KL 惩罚 `kl_loss`：约束策略不要偏离参考模型太远 | 要**小** | `--use-kl-loss --kl-loss-coef` |

> **符号约定**：RL 的原始目标是**最大化**的；在目标函数前面加上负号即为训练 loss（`loss = -objective`），框架通过梯度下降最小化 loss。因此 entropy_loss 在目标中为正（要熵大），在 loss 中为负；kl_loss 在目标中为负（要 KL 小），在 loss 中为正。详见第 3.3 节和第 8 章。

---

## 二、强化学习基础（LLM 语境）

### 2.1 把 LLM 生成建模成 MDP

**结论：把"生成一个 token"看成一个动作，把"已生成的前缀"看成状态，那么"生成一整段回复"就是一条轨迹（trajectory）。整段回复结束后拿到的奖励，需要被"分配"回每一个 token 的决策上。**

马尔可夫决策过程（MDP）的四元组在 LLM 语境下的对应：

- **状态 `s_t`**：prompt + 已经生成的 token 前缀 `(x, y_{<t})`
- **动作 `a_t`**：从词表中选出的下一个 token `y_t`
- **策略 `π_θ(a_t | s_t)`**：模型在当前前缀下对下一个 token 的概率分布——这就是 LLM 本身
- **奖励 `r_t`**：绝大多数场景是**稀疏奖励**——只有整段回复结束时才有一个标量奖励 `R`（答案对不对、RM 打多少分），中间每个 token 的即时奖励为 0

一段长度为 `T` 的回复就是一条轨迹 `τ = (s_0, a_0, s_1, a_1, ..., s_{T-1}, a_{T-1})`。轨迹的总回报通常就是终点奖励：`R(τ) = R`。

> 在 Agentic RL 中，一条轨迹还包含多轮对话和工具调用，`T` 会长得多，稀疏奖励问题也更严重——这正是 Dressage 要解决的场景，见 [agentic-rl-training-zh.md](agentic-rl-training-zh.md) 第一章。

### 2.2 策略梯度定理

**结论：策略梯度告诉我们如何调整参数 θ 来提高期望回报——把"好轨迹"里每个动作的概率调高，"坏轨迹"里的调低。核心公式是 $\nabla J(\theta) = \mathbb{E}[\nabla\log\pi_\theta(a|s)\cdot A]$。**

我们的优化目标是最大化期望回报：

```
J(θ) = E_{τ~π_θ} [ R(τ) ]
```

要展开这个期望，首先要弄清楚**一条轨迹 τ 的出现概率 p_θ(τ) 是什么**——这是策略梯度推导的基石。

#### 轨迹的出现概率 p_θ(τ)

**结论：一条轨迹 τ = (s_0, a_0, s_1, a_1, ..., s_T, a_T) 的出现概率，等于"初始状态分布 × 各步（策略选动作概率 × 环境转移概率）的连乘"。它由 MDP 的马尔可夫性 + agent/environment 两环节独立 共同决定。**

**轨迹的形态**：轨迹是 agent 与环境交互产生的完整序列，状态与动作交替出现：

$$\tau = (s_0, a_0, s_1, a_1, s_2, a_2, \ldots, s_T, a_T)$$

- `s_0`：环境给的初始状态
- `a_0`：agent 在 s_0 下选的动作
- `s_1`：环境根据 (s_0, a_0) 转移到的新状态
- ... 直到 `s_T, a_T`

> 下标含义：状态和动作交错，`a_t` 发生在 `s_t` 之后、`s_{t+1}` 之前；时间步 t 数的是 `(s_t, a_t)` 对的个数。

**MDP 的两个角色**：讲轨迹概率时只需关心两个角色，它们各管一个环节：

| 角色 | 控制什么 | 用什么刻画 | 是否依赖 θ |
|---|---|---|---|
| Agent（策略） | 在某状态下选哪个动作 | `π_θ(a|s)`：状态→动作分布 | 依赖 θ（训练对象） |
| Environment（环境） | 给定动作后转到哪个状态 | `P(s'|s,a)`：状态转移 | 与 θ 无关（世界规律，固定） |

外加初始状态分布 `p(s_0)`（也叫 `ρ_0`），也与 θ 无关。每一步生成都分两个独立子环节——agent 先用策略选动作，环境再用转移函数给下一个状态——所以每步的联合概率就是两个因子相乘。

**用链式法则展开**：对 `p(s_0, a_0, s_1, a_1, ...)` 用条件概率链式法则逐层写：

$$p(\tau) = p(s_0)\cdot p(a_0|s_0)\cdot p(s_1|s_0,a_0)\cdot p(a_1|s_0,a_0,s_1)\cdot p(s_2|s_0,a_0,s_1,a_1)\cdots$$

逐项简化：

- `p(s_0)`：初始状态分布 `ρ_0`。
- `p(a_0|s_0)`：给定 `s_0` 选 `a_0` 的概率，正是策略 → `π_θ(a_0|s_0)`。
- `p(s_1|s_0,a_0)`：给定 `(s_0,a_0)` 转到 `s_1`，正是环境转移 → `P(s_1|s_0,a_0)`。
- `p(a_1|s_0,a_0,s_1)`：agent 选 `a_1` 时只看当前状态 `s_1`，历史 `(s_0,a_0)` 不影响决策（马尔可夫性）→ `π_θ(a_1|s_1)`。
- `p(s_2|s_0,a_0,s_1,a_1)`：转到 `s_2` 只依赖当前 `(s_1,a_1)`（马尔可夫性）→ `P(s_2|s_1,a_1)`。

**关键：马尔可夫性把所有冗长的历史条件砍掉了**。每对 `(s_t, a_t)` 只贡献两个因子：

$$\underbrace{\pi_\theta(a_t|s_t)}_{\text{agent 子环节}}\cdot\underbrace{P(s_{t+1}|s_t,a_t)}_{\text{environment 子环节}}$$

于是整条轨迹的概率就是把它们连乘起来，再乘上初始分布：

$$p_\theta(\tau) = p(s_0)\cdot\prod_{t=0}^{T}\big[\pi_\theta(a_t|s_t)\cdot P(s_{t+1}|s_t,a_t)\big]$$

**几个细节**：

1. **连乘上界 T 还是 T-1**：取决于轨迹是否在终止状态 `s_T` 结束。最干净的写法是拆开——"所有动作的策略概率"连乘 `T+1` 次、"所有转移的概率"连乘 `T` 次（最后一步没有"下一个状态"）：

   $$p_\theta(\tau) = p(s_0)\cdot\prod_{t=0}^{T}\pi_\theta(a_t|s_t)\cdot\prod_{t=0}^{T-1}P(s_{t+1}|s_t,a_t)$$

2. **环境项 P 与 θ 无关**：θ 是 agent（策略）的参数；环境是"客观世界"，`P` 是固定的物理规律（棋类规则、物理仿真器方程、API 响应分布），不随 agent 训练而变。
3. **初始分布 p(s_0) 也与 θ 无关**：由任务设定决定，所以 `∇ log p(s_0) = 0`。
4. **P 可以是确定性的**：很多环境（棋类、代码沙箱）转移是确定函数 `s_{t+1}=f(s_t,a_t)`，此时 `P` 退化为 δ 函数，该因子恒为 1，公式形式不变。
5. **连续空间同理**：把离散求和换成积分、概率换成密度，公式形式完全一样，所以策略梯度定理在连续控制（机器人）中也成立。

**为什么这个公式关键**：策略梯度推导的"破局一步"是对 `log p_θ(τ)` 求梯度。把它写成：

$$\log p_\theta(\tau) = \underbrace{\log p(s_0)}_{\text{与 θ 无关，求导=0}} + \underbrace{\sum_t\log\pi_\theta(a_t|s_t)}_{\text{依赖 θ}} + \underbrace{\sum_t\log P(s_{t+1}|s_t,a_t)}_{\text{与 θ 无关，求导=0}}$$

所以：

$$\nabla_\theta\log p_\theta(\tau) = \sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)$$

**两项与 θ 无关的因子（`p(s_0)` 和 `P`）直接消失**，只剩策略项的累加——这就是策略梯度定理"不需要环境模型"的根源，也是 model-free RL 的理论基石。

#### 回到策略梯度定理

由上，$\nabla_\theta\log p_\theta(\tau) = \sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)$。代入 log-derivative trick `∇p = p·∇log p`，得策略梯度定理：

$$\nabla J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\!\left[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t)\cdot\Psi_t\right]$$

其中 `Ψ_t` 是"这个动作有多好"的度量，可以有多种选择：

- $\Psi_t = R(\tau)$：整条轨迹的总回报（最朴素的 REINFORCE）
- $\Psi_t = \sum_{t'\ge t} r_{t'}$：从 t 时刻起的未来回报（reward-to-go，降方差）
- $\Psi_t = A_t$：**优势函数**（advantage），当前 LLM RL 的标准选择

**全家福对照：各算法的 Ψ_t 与正负号含义**

下表汇总所有主流算法对 $\Psi_t$ 的取法（后续章节逐一展开），它们共享同一个梯度骨架 $\nabla\log\pi\cdot\Psi_t$，区别只在 $\Psi_t$ 怎么估：

| 算法 | $\Psi_t$ | "好/差"相对什么 |
|---|---|---|
| REINFORCE | $G_t$ | 绝对回报（无 baseline，区分度差） |
| REINFORCE + baseline | $G_t - b(s)$ | 相对 baseline $b$ |
| TD / Actor-Critic | $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | 相对 critic 预测 $V(s_t)$ |
| PPO | $A_t^{\mathrm{GAE}}$ | 相对 critic（多步平滑） |
| GRPO | $r_i - \mathrm{mean}(\text{group})$ | 相对组内均值 |

**正负号含义**（减了 baseline 之后才成立）：$\Psi_t > 0$ → 该动作比平均好 → 梯度上升推高其概率；$\Psi_t < 0$ → 比平均差 → 压低概率；$\Psi_t = 0$ → 不更新。所以 $\Psi_t$ 本质是"**相对 baseline 的好坏度量**"，而非绝对回报。

**每一项的物理直觉**：把策略梯度公式拆开看，每一项在做不同的事：

| 项 | 含义 | 作用 |
|---|---|---|
| `∇_θ log π_θ(a_t|s_t)` | 方向向量：增大"在 s_t 下选 a_t 的对数概率"的参数更新方向 | 决定"往哪推 θ" |
| `Ψ_t`（或 `R(τ)`） | 权重：这条轨迹/这个动作有多好 | 决定"推多大力、正还是负" |
| `Σ_t` | 沿轨迹所有决策点累加 | 把终点奖励分配回每个决策 |
| `E_{τ~π_θ}` | 按当前策略采样的期望 | 实践用 Monte Carlo：N 条采样轨迹平均 |

梯度上升时：`Ψ_t > 0`（好动作）→ θ 朝增大 `logπ(a_t|s_t)` 的方向走 → 该动作概率被推高；`Ψ_t < 0`（坏动作）→ θ 反向走 → 概率被压低。这就是"`∇log π` 是提高该 token 概率的方向，乘上 `Ψ_t`（正=好，负=坏）后把好动作推高、坏动作推低"的完整含义。

**为什么是 `log π` 而不是 `π`？** 因为 $
abla\log\pi = 
abla\pi / \pi$ 自动做了**重要性权重归一**——概率本就越小的动作，它的 `∇π` 也会小，除以 `π` 后能“公平”地获得梯度信号。这本质上是 **likelihood ratio / score function 估计**：$
abla\mathbb{E}_p[f] = \mathbb{E}_p[f\cdot
abla\log p]$，策略梯度定理只是把它套到 $p = p_\theta(\tau)$、$f = R(\tau)$ 的特例。

### 2.3 REINFORCE 与 baseline

**结论：直接用回报 `R` 当权重方差极大，训练不稳。减去一个与动作无关的基线 `b`（baseline）可以在不引入偏差的前提下大幅降低方差。用什么当 baseline，是区分各类算法的关键。**

朴素 REINFORCE 用蒙特卡洛回报：

$$\nabla J(\theta) \approx \frac{1}{N}\sum_i\sum_t \nabla\log\pi_\theta(a_t|s_t)\cdot R(\tau_i)$$

问题：`R(τ)` 数值大且波动大，梯度方差高。改进是减去一个 baseline `b(s_t)`：

$$\nabla J(\theta) \approx \mathbb{E}\!\left[\sum_t \nabla\log\pi_\theta(a_t|s_t)\cdot (R - b)\right]$$

只要 `b` 不依赖当前动作 `a_t`，减去它**不改变梯度的期望**（无偏），但能显著降低方差。`(R - b)` 本质上就是优势 `A` 的估计。

**另一个降方差技巧：reward-to-go（因果性）**。除了 baseline，还可以把 `Ψ_t = R(τ) = Σ_{t'} r_{t'}` 换成 reward-to-go `Ψ_t = Σ_{t'≥t} r_{t'}`。理由是**因果性**：`t` 时刻的动作 `a_t` 不可能影响 `t` 之前已经发生的奖励 `Σ_{t'<t} r_{t'}`，所以这些项与 `a_t` 无关，并入 `Ψ_t` 只会增加方差、不改变期望，可以安全丢掉。这就是 2.2 节 `Ψ_t` 列表里"reward-to-go，降方差"的来历。

#### 重要性采样：用旧策略的样本更新新策略

**结论：策略梯度定理要求用当前策略 π_θ 采样，但实践中用旧策略 π_old 采的一批轨迹来更新当前参数（样本复用、提效），于是出现分布偏移。用重要性采样比率 `ρ = π_θ(τ)/π_old(τ)` 修正，可在旧策略分布下无偏估计新策略的期望。**

**为什么需要**：定理给出的是 `E_{τ~π_θ}[·]`，要求"用当前策略采样"。但每更新一次梯度就重新采样成本太高——我们想用一批旧策略采的轨迹做多步梯度更新（提高样本效率）。此时样本来自 `π_old`，却要评估 `π_θ`，存在分布偏移。

**修正**：用重要性采样比率修正分布差异：

$$\rho(\tau) = \frac{\pi_\theta(\tau)}{\pi_{\mathrm{old}}(\tau)}$$

代入后，旧策略分布下的期望等于新策略分布下的期望：

$$\mathbb{E}_{\tau\sim\pi_{\mathrm{old}}}\!\left[\rho(\tau)\cdot f(\tau)\right] = \mathbb{E}_{\tau\sim\pi_\theta}\!\left[f(\tau)\right]$$

**比率的展开**：把 `p_θ(τ)` 的分解代入，环境项 `P` 与 θ 无关，在 `π_θ` 和 `π_old` 中完全相同，相除后抵消，只剩策略项：

$$\rho(\tau) = \prod_t \frac{\pi_\theta(a_t|s_t)}{\pi_{\mathrm{old}}(a_t|s_t)}$$

即"各 token 重要性比率的连乘"——环境模型再次消失，这是 model-free 的另一体现。

**token 级比率**：实践中常直接用每个 token 的比率（PPO/GRPO 的做法）：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\mathrm{old}}(a_t|s_t)}$$

**比率偏离 1 越远，说明新旧策略差异越大**，越需要约束——这正是 PPO 裁剪（clip）、KL 惩罚、dual-clip 等技巧的动机（详见第八章）。但它们仍活在策略梯度定理的框架内，只是把"用旧样本评估新策略"这件事做稳。

> 本节是最小必要版。IS 的完整讨论——一般形式与支撑覆盖条件、轨迹级比率为何方差爆炸、token 级比率的一阶近似本质、比率的诊断恒等式、`π_old` 的三种语境含义、裁剪的偏差-方差权衡——见 [2.5 节](#25-重要性采样原理深化与工程落点)。

不同算法对 baseline 的选择：

| 算法 | baseline `b` |
|---|---|
| REINFORCE + value baseline | 学一个价值网络 `V(s)` |
| PPO | GAE，基于 value 网络 |
| RLOO | 同组其他样本回报的均值（留一法） |
| **GRPO** | **同一 prompt 采样的一组回复的奖励均值** |

### 2.4 价值函数、优势函数与 GAE

**结论：优势函数 $A(s,a) = Q(s,a) - V(s)$ 衡量"在状态 s 下选动作 a，比平均水平好多少"。PPO 用 GAE（广义优势估计）在偏差和方差间做权衡来估计它；GRPO 则用组内均值绕开了对价值网络的依赖。**

三个价值量：

- **状态价值 `V(s)`**：从状态 s 出发，按策略走下去的期望回报
- **动作价值 `Q(s,a)`**：在 s 选了动作 a 后的期望回报
- **优势 $A(s,a) = Q(s,a) - V(s)$**：动作 a 相对"平均"的增量

PPO 用 **GAE（Generalized Advantage Estimation）** 估计优势。下面从 REINFORCE 出发，一步步推出 GAE，看清楚每一步在用什么。

**（1）REINFORCE：直接用回报 G_t**

回顾策略梯度定理 `∇J = E[Σ_t ∇log π · Ψ_t]`，REINFORCE 选 `Ψ_t = G_t`（reward-to-go，从 t 时刻起的回报）：

$$G_t = \sum_{l=0}^{\infty}\gamma^l\cdot r_{t+l}$$

`G_t` 是真实累积回报，无偏但高方差。问题：能否用价值函数 `V` 降方差？

**（2）TD 误差：优势的一步估计**

由优势定义 `A(s,a) = Q(s,a) - V(s)` 和 Bellman 方程 $Q(s_t,a_t) = \mathbb{E}[r_t + \gamma\cdot V(s_{t+1})\mid s_t,a_t]$，得：

$$A(s_t,a_t) = \mathbb{E}[r_t + \gamma\cdot V(s_{t+1})\mid s_t,a_t] - V(s_t)$$

用单次采样 `(r_t, s_{t+1})` 近似期望，得到 A 的一步估计：

$$\hat A_t = r_t + \gamma\cdot V(s_{t+1}) - V(s_t) := \delta_t$$

这就是 **TD 误差** `δ_t`。注意它**不是回报 R**，而是"即时奖励 `r_t` + 价值预测 `γ·V(s_{t+1})` − 价值预测 `V(s_t)`"。用真实 `V^π` 时 `E[δ_t | s,a] = A(s,a)`（无偏）；但实际用学的 `V̂ ≠ V^π`，所以 `δ_t` 有偏，偏差来自 critic 不准。

**（3）n-step 优势：多步真实回报，降低对 critic 的依赖**

把 TD 误差累加 n 步：

$$\hat A_t^{(n)} = \sum_{l=0}^{n-1}\gamma^l\cdot\delta_{t+l}$$

代入 `δ` 定义、递推抵消中间的 `V`，化简为：

$$\hat A_t^{(n)} = \sum_{l=0}^{n-1}\gamma^l\cdot r_{t+l} + \gamma^n\cdot V(s_{t+n}) - V(s_t)$$

- `n=1`：`δ_t`（纯 TD，critic 用得多，偏差大方差小）
- `n→∞`：`G_t - V(s_t)`（MC，critic 只在终点用一次，偏差小方差大；**此时才和回报 `G_t` 直接相关**）

**（4）GAE：把所有 n-step 优势做指数加权平均**

GAE 不固定 n，而是对所有 n-step 优势按 `(1-λ)λ^{n-1}` 加权求和（权重和为 1）：

$$A_t^{\mathrm{GAE}} = \sum_{n=1}^{\infty}(1-\lambda)\cdot\lambda^{n-1}\cdot\hat A_t^{(n)}$$

代入 `Â_t^{(n)} = Σ_{l=0}^{n-1} γ^l · δ_{t+l}` 并交换求和顺序，化简为简洁形式：

$$A_t^{\mathrm{GAE}} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\cdot\delta_{t+l}$$

**λ 的两个极端**（关键考点）：

- $\lambda=0$：$A_t^{\mathrm{GAE}} = \delta_t$（退化为纯 TD，方差小、偏差大）
- $\lambda=1$：$A_t^{\mathrm{GAE}} = \sum\gamma^l\delta_{t+l} = G_t - V(s_t)$（退化为 MC，偏差小、方差大）

所以 **`λ` 是"信 critic（TD）"和"信真实回报（MC）"之间的旋钮**，`0<λ<1` 取折中（实践常取 0.95）。

- GAE 需要价值网络 `V`（critic）来估计，这就是 PPO 要额外训练 critic 的原因；GRPO 用组内均值绕开了它。

> **LLM 里的简化**：由于奖励只在终点出现、且序列内 `γ` 常取 1，很多实现把整段回复视为一个"动作"。这正是 GRPO 能用"一个标量优势广播到整段回复所有 token"这一简化的前提。

### 2.5 重要性采样：原理深化与工程落点

**结论：重要性采样（IS）是一切"样本复用"的数学基础——它让用旧策略 `π_old` 采的数据可以（近似）无偏地更新新策略 `π_θ`。但原始 IS 的方差随序列长度指数爆炸，LLM RL 的工程主线就是把这个方差闸住：token 级比率替代轨迹级比率、clip 替代无界比率、TIS 截断替代裸比率。本节回答四个问题：IS 的一般形式是什么、为什么必须用 token 级比率、比率有哪些诊断恒等式、`π_old` 到底是谁。**

**（1）IS 的一般形式**

期望可以在另一个分布下估计，只要乘上密度比：

$$\mathbb{E}_{x\sim p}[f(x)] = \sum_x p(x)f(x) = \sum_x q(x)\cdot\frac{p(x)}{q(x)}\cdot f(x) = \mathbb{E}_{x\sim q}\!\left[\frac{p(x)}{q(x)}f(x)\right]$$

唯一前提是**支撑覆盖**：`p(x) > 0` 的地方必须 `q(x) > 0`（旧策略必须"有可能"产出新策略会产出的样本）。温度 > 0 的 LLM 采样天然满足——任何 token 序列都有非零概率。2.3 节已把这条恒等式套用到轨迹分布（`p = p_θ(τ)`、`q = p_old(τ)`），并推导出环境项抵消后的 `ρ(τ) = ∏_t r_t`。

**（2）为什么不能用轨迹级比率：方差爆炸**

`ρ(τ)` 是 T 个 token 比率的连乘。连乘的统计性质是：哪怕每个比率只偏离 1 一点点，乘积也会指数级发散。

数值直觉：假设新旧策略差异小到每个 token 的比率只有 1% 漂移（`r_t ≈ 1.01`），一条 4000 token 的轨迹（agentic 场景算短的）：

$$\rho(\tau) = 1.01^{4000} \approx e^{40} \approx 10^{17}$$

更一般地，若各 token 比率近似独立、期望为 1、单个方差为 `σ²`，则轨迹比率的方差为：

$$\mathrm{Var}\!\left(\prod_t r_t\right) = \prod_t \mathbb{E}[r_t^2] - 1 = (1+\sigma^2)^T - 1$$

**随 T 指数增长**。batch 里只要混入一条这种"巨比率"轨迹，梯度就被它一家主导，其余样本全部失效。

**（3）token 级比率为什么合法：一阶近似**

出路藏在 MDP 形式化里。2.2 节的策略梯度定理把梯度写成**逐决策点求和**：`∇J = E[Σ_t ∇log π(a_t|s_t)·Ψ_t]`——每个 `(s_t, a_t)` 都是独立决策点，不是"一整条轨迹是一个决策"。既然期望能逐点分解，IS 修正也能逐点进行。

但严格地说，逐点修正需要**前缀比率** `∏_{t'≤t} r_{t'}`——因为 t 时刻的状态访问分布本身也随策略变了。只保留当前点的 `r_t` 是**一阶近似**：策略变化足够小时，状态分布的变化可以忽略。PPO 的推导正是这个口径，而 clip 进一步把"策略变化小"从假设强制为现实。所以完整的逻辑链是：

```text
token 级比率 = 一阶近似（忽略状态访问分布变化）
clip        = 把近似成立的前提（策略变化小）变成约束
```

这就是 PPO/GRPO 的工程标准：**不修正"整条轨迹来自哪个分布"，而是修正"每个 token 来自哪个分布"，再用裁剪为近似兜底。**

**（4）比率的两个诊断恒等式**

- $\mathbb{E}_{\pi_{old}}[r_t] = 1$：旧分布下比率期望恒为 1（`∫ π_old·(π_θ/π_old) = ∫ π_θ = 1`）。监控比率均值是否 ≈1 是最基本的 sanity check——显著偏离说明实现或数据有错。
- $\mathbb{E}_{\pi_{old}}[\log r_t] = -\mathrm{KL}(\pi_{old}\|\pi_\theta)$：对数比率的均值就是新旧策略 KL 的负估计——**比率本身就是 KL 探针**。

slime 的实现正好利用第二点：[loss.py](../slime/slime/backends/megatron_utils/loss.py) 第 976 行计算 `ppo_kl = old_log_probs - log_probs`（即 `−log r_t`），[ppo_utils.py](../slime/slime/utils/ppo_utils.py) 的 `compute_policy_loss` 第 132 行 `ratio = (-ppo_kl).exp()` 还原比率做裁剪；而 `ppo_kl` 本身同时作为监控指标上报。一个量，既是 loss 的输入又是健康度指标。

**（5）方差、有效样本量与三条降方差路线**

IS 估计的精度由权重分布的离散度决定：有效样本量 `ESS = (Σw)²/Σw²`，若少数样本权重远大于其余，ESS 会坍缩到远小于 batch size——其余样本白采。三条工程路线：

| 路线 | 做法 | 代表 |
|---|---|---|
| 自归一化 | 权重除以组内均值，吸收绝对尺度 | 与 GRPO 组均值 baseline 的思想同源（4.1） |
| 截断 | 比率钳在区间内，超限部分丢弃或归零 | PPO clip（8.2）、TIS clamp 到 `[tis_clip_low, tis_clip]`（8.7） |
| 换粒度 | 整条序列一个比率，或换掉修正对象 | GSPO（5.1）、CISPO（5.2） |

**（6）`π_old` 到底是谁：三种语境三种答案**

| 语境 | `π_old` 指什么 | slime 对应 |
|---|---|---|
| 同步 PPO/GRPO（默认） | 采样完成、训练步开始那一刻的 actor 权重（训练引擎前向） | `log_probs`（[loss.py](../slime/slime/backends/megatron_utils/loss.py) 第 912 行默认分支） |
| 引擎不一致修正 | rollout 引擎（SGLang）实际采样时的 logprob | `--use-rollout-logprobs` 后的 `rollout_log_probs` |
| 异步训练 | 混合策略——不同 token 可能产生于不同权重版本 | staleness 追踪 + TIS/OPSM（8.7 节） |

最常见的误读是把 `π_old` 和 `π_ref` 混为一谈：**`π_old` 是"采这批数据的策略"**，每个 batch 都在变；**`π_ref` 是 KL 惩罚的锚点**（通常冻结为 SFT 模型，3.3 节）。前者活在比率里，后者活在 KL 惩罚里。

**（7）裁剪的本质：用偏差换方差**

clip 之后目标函数不再是真梯度的无偏估计——这是**故意**引入偏差来闸住方差。本文各章出现的机制，都是同一个"偏差-方差"权衡在不同位置的落点：

| 机制 | 落点 | 引入的偏差 | 换来的方差收益 |
|---|---|---|---|
| PPO clip（3.2 / 8.2） | 逐 token 比率限在 `[1±ε]` | 被裁 token 梯度归零 | 单步更新有界 |
| Clip-higher（6.1） | 上界放宽 `ε_high > ε_low` | 同上，但保留探索 | 防熵坍缩 |
| GSPO（5.1） | 序列级比率整体 clip | 整条序列限幅 | 长序列方差大降 |
| CISPO（5.2） | 裁 IS 权重（stop-gradient） | 权重封顶 | 被裁 token 仍有梯度 |
| TIS（8.7） | 引擎间比率截断 | 截断偏差 | 控制引擎不一致噪声 |

一句话记住本节：**IS 让旧数据能再用，token 级比率让它在 LLM 上可用，clip/TIS 让它在长序列上敢用。**

---

## 三、RLHF 的经典方案：PPO

### 3.1 RLHF 三阶段流程

**结论：RLHF（基于人类反馈的强化学习）是把 RL 引入 LLM 的里程碑，分三步：SFT → 训练 Reward Model → 用 PPO 优化策略。PPO 阶段的奖励来自 RM，而不是环境。**

1. **SFT（监督微调）**：用高质量示范数据得到初始策略 `π_SFT`，同时作为后续 KL 惩罚的参考策略 `π_ref`。
2. **训练奖励模型（RM）**：收集人类对同一 prompt 多个回复的偏好排序，训练一个打分模型 `r_φ(x, y)`。典型损失是 Bradley-Terry 成对损失：

   $$L_{\mathrm{RM}} = -\mathbb{E}\!\left[\log\sigma\!\left(r_\phi(x, y_{\mathrm{chosen}}) - r_\phi(x, y_{\mathrm{rejected}})\right)\right]$$
3. **PPO 优化**：以 `r_φ` 的打分为奖励，用 PPO 更新策略，使其生成更受偏好的回复，同时用 KL 约束不要偏离 `π_ref` 太远。

### 3.2 PPO 的裁剪目标

**结论：PPO 用"重要性采样比率 + 裁剪"来实现"用一批旧策略采的数据，安全地做多步梯度更新"。裁剪把比率限制在 `[1-ε, 1+ε]`，防止单步更新把策略推得太远导致崩溃。**

由于用当前策略 `π_θ` 评估、却是用旧策略 `π_θ_old` 采的样，需要重要性采样比率来修正分布差异：

$$r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}$$

PPO 的裁剪代理目标（clipped surrogate objective）：

$$L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\cdot A_t,\ \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\cdot A_t\right)\right]$$

逐项理解：

- `A_t > 0`（好动作）：想提高其概率，但 `r_t` 一旦超过 `1+ε` 就被裁掉——单步不许把概率推太高
- `A_t < 0`（坏动作）：想降低其概率，但 `r_t` 一旦低于 `1-ε` 也被裁掉
- `min(...)` 取的是"更保守"的那一项，本质是给目标加了一个悲观下界，避免过大更新

`ε` 就是 `eps-clip`，典型值 0.2。

### 3.3 KL 惩罚：不要跑太远

**结论：为防止策略为了迎合 RM 而"钻空子"（reward hacking）跑到语言退化的区域，PPO 在奖励里额外加一个对参考策略 `π_ref` 的 KL 惩罚，把策略拴在合理语言分布附近。**

有两种加法（实现上二选一或都用）：

1. **加进奖励**（token 级）：`r_t' = r_φ - β·KL(π_θ || π_ref)`
2. **加进损失**（作为独立 loss 项）：`L = L^CLIP - β·KL(π_θ || π_ref)`

slime/Dressage 用的是第二种（`--use-kl-loss` + `--kl-loss-coef`），KL 的具体估计形式见 [8.3 节](#83-kl-散度的三种估计)。

### 3.4 PPO 完整目标与四个模型

**结论：PPO 的完整目标 = 裁剪策略损失 + 价值损失 + 熵奖励 − KL 惩罚。它在训练时需要同时持有四个模型（Actor / Critic / Reward / Reference），显存开销大——这正是 GRPO 要砍掉 Critic 的动机。**

完整目标函数：

$$L^{\mathrm{PPO}} = \mathbb{E}\!\left[L^{\mathrm{CLIP}}(\theta) - c_1\cdot L^{\mathrm{VF}}(\theta) + c_2\cdot H(\pi_\theta) - \beta\cdot\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})\right]$$

- `L^VF = (V_ψ(s_t) - R_t)^2`：价值网络的回归损失
- `H(π_θ)`：策略熵，鼓励探索（`c2` 即 `entropy-coef`）
- `β`：KL 系数

PPO 训练时同时存在的四个模型：

| 模型 | 作用 | 是否更新 |
|---|---|---|
| Actor（策略 `π_θ`） | 要训练的 LLM | ✅ 更新 |
| Critic（价值 `V_ψ`） | 估计状态价值，用于 GAE | ✅ 更新 |
| Reward Model `r_φ` | 提供奖励信号 | ❌ 冻结 |
| Reference `π_ref` | KL 惩罚的锚点 | ❌ 冻结 |

四个模型（其中两个与 Actor 同尺寸）带来的显存和工程复杂度，是"去 critic"算法兴起的直接原因。

---

## 四、去掉 critic：GRPO

### 4.1 核心思想：组内相对优势

**结论：GRPO（Group Relative Policy Optimization）是 PPO 的变体，用"对同一个 prompt 采样一组回复、用组内奖励均值当 baseline"替代价值网络。这样彻底省掉了 critic，把训练所需模型从四个减到三个（甚至两个）。**

流程：对每个 prompt `x`，用当前策略采样一组 `G` 个回复 `{y_1, ..., y_G}`（`G` 就是 `n-samples-per-prompt`），给每个回复打一个终点奖励 `r_i`。优势直接由组内统计得到：

$$\hat A_i = \frac{r_i - \mathrm{mean}(r_1,\ldots,r_G)}{\mathrm{std}(r_1,\ldots,r_G)}$$

- **分子** `r_i - mean`：组内相对好坏——比同组平均好就是正优势，反之为负。这就是用组均值当 baseline。
- **分母** `std`：可选的标准差归一化（Dressage 里由 `grpo_std_normalization` 控制），把不同 prompt 的奖励尺度拉齐。

同一个回复 `y_i` 内的**所有 token 共享同一个标量优势 `Â_i`**（因为奖励在终点、序列内无中间信号）。这就是为什么 Dressage 的 [reward_post_process.py](../dressage/training/reward_post_process.py) 只需为每个 sample 算一个标量优势，再由 slime 广播到该回复的所有 token。

### 4.2 GRPO 目标函数

**结论：GRPO 保留了 PPO 的"裁剪比率"和"KL 损失"，只是把 GAE 优势换成了组内相对优势。**

$$L^{\mathrm{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_i\frac{1}{|y_i|}\sum_t\min\!\left(r_{i,t}(\theta)\cdot\hat A_i,\ \mathrm{clip}(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon)\cdot\hat A_i\right)\right] - \beta\cdot\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})$$

其中 `r_{i,t}(θ) = π_θ(y_{i,t}|·) / π_θ_old(y_{i,t}|·)` 仍是 token 级重要性比率，`Â_i` 是该回复的组内优势。

关键点：GRPO 依然是 token 级的裁剪与重要性采样，只是优势来自组内相对值。

> **符号约定**：以上目标函数 `L` 是要最大化的 RL 目标；在目标函数前面加上负号即为训练 loss（`loss = -L`），框架通过梯度下降最小化 loss，等价于最大化目标。KL 项在目标中为负（要 KL 小），熵项在目标中为正（要熵大）。

### 4.3 GRPO 与 PPO 的对比

| 维度 | PPO | GRPO |
|---|---|---|
| 优势估计 | GAE（需 critic） | 组内相对（`r_i - mean`） |
| 训练模型数 | 4（含 Critic） | 3 或 2（无 Critic） |
| baseline | 学习得到的 `V(s)` | 一组样本奖励均值 |
| 每 prompt 采样数 | 可为 1 | 必须一组（`G≥2`） |
| 适合场景 | 稠密/可学 value 的奖励 | 可验证的稀疏奖励（RLVR） |
| 显存/工程 | 重 | 轻 |

**目标函数并列对比**

PPO：

$$L^{\mathrm{PPO}} = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\cdot A_t,\ \mathrm{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)\cdot A_t\right) - c_1 L^{\mathrm{VF}} + c_2 H(\pi_\theta)\right] - \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})$$

GRPO：

$$L^{\mathrm{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_i\frac{1}{|y_i|}\sum_t\min\!\left(r_{i,t}(\theta)\cdot\hat A_i,\ \mathrm{clip}(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon)\cdot\hat A_i\right)\right] - \beta\cdot\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})$$

**逐项对照**：

- **裁剪项（结构完全相同）**：PPO $\min(r_t A_t,\ \mathrm{clip}(r_t) A_t)$ ↔ GRPO $\min(r_{i,t} \hat A_i,\ \mathrm{clip}(r_{i,t}) \hat A_i)$ —— 都是“min(比率×优势， clip(比率)×优势)”，唯一差别在优势来源：PPO 用 GAE $A_t$（需 critic），GRPO 用组内相对 $\hat A_i$（免 critic）。
- **PPO 独有**：$-c_1 L^{\mathrm{VF}}$（价值损失，训 critic 用）+ $+c_2 H(\pi_\theta)$（熵奖励，鼓励探索）——这两项 GRPO 都没有。
- **共同**：KL 惩罚 $-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})$ 放在期望外。

> 一句话：GRPO = PPO − 价值损失 − 熵奖励，且把 GAE 优势换成组内相对优势。

GRPO 的代价：需要对每个 prompt 采样一组（`G` 倍的采样成本），且当一组内奖励全相同时优势全为 0（无梯度信号）——后者正是 DAPO"动态采样"要解决的问题。

### 4.4 近亲：RLOO 与 Reinforce++ baseline

同属"无 critic、用样本统计当 baseline"的家族：

- **RLOO（REINFORCE Leave-One-Out）**：baseline 用"同组**其他**样本"的均值（留一法），即 `b_i = mean(r_{j≠i})`，理论上更无偏。
- **Reinforce++**：给朴素 REINFORCE 加上 PPO 的工程技巧（token 级 KL、全局 batch 优势归一化、裁剪），但不强制分组。slime 的 `reinforce_plus_plus_baseline` 是其带组内 baseline 的变体——在 [reward_post_process.py](../dressage/training/reward_post_process.py) 第 118-125 行与 `grpo`/`gspo` 走同一套均值归一化逻辑。

### 4.5 GRPO 的数学偏差与 Dr. GRPO

**结论：GRPO 的目标函数里藏着两个不易察觉的数学偏差——组内 std 归一化会放大"过易/过难"组的权重（难度偏置）；`1/|y_i|` 的序列内平均会稀释长回复的梯度（长度偏置，对长错误回复惩罚不足 → 回复越训越长）。Dr. GRPO（Understanding R1-Zero-Like Training，arXiv 2503.20783）的修正是"做减法"：去掉 std 归一化、改用 token 级聚合。slime 中分别对应 `--disable-grpo-std-normalization` 与 `--calculate-per-token-loss`。**

**偏差 1：std 归一化 → 难度偏置**

4.1 节的优势公式 `Â_i = (r_i − mean) / std` 中，当一组回复的奖励几乎全同（全对或全错）时 std ≈ 0，除法会人为放大该组的优势。而这类"饱和组"恰恰是信息量最少的组，却被赋予了更大的有效权重——这就是难度偏置。Dr. GRPO 建议直接去掉分母，只用 `Â_i = r_i − mean`。

- slime 中 std 归一化**默认开启**，需显式加 `--disable-grpo-std-normalization` 关闭（[arguments.py](../slime/slime/utils/arguments.py) 第 995-1000 行，help 文本直接引用了 Dr. GRPO 论文）。
- Dressage 的后处理层同样遵循 `grpo_std_normalization` 开关（[reward_post_process.py](../dressage/training/reward_post_process.py) 第 151-155 行，std < 1e-6 时跳过除法防爆）。

**偏差 2：`1/|y_i|` 序列内平均 → 长度偏置**

4.2 节目标函数中的 `1/|y_i|` 表示"每条回复内部先对 token 取平均、再对样本取平均"。后果是长回复里每个 token 的梯度贡献被稀释；尤其当长回复是**错误**答案时，逐 token 的惩罚被摊薄，模型倾向于把回复越写越长。修正方向是把聚合单位从"序列"换成"token"——即 8.6 节的 token-mean（`--calculate-per-token-loss`），与 DAPO 的 token-level loss（6.3 节）殊途同归。

> Dressage 的示例脚本目前均未显式设置这两个开关，即保持 slime 默认（std 归一化开启、sequence-mean 聚合）——解读实验结果时应注意这一点。

---

## 五、GSPO 与 CISPO：重要性采样的稳定化

本章讲 GRPO 的两个"稳定化"变体，它们都改动了重要性采样比率的处理方式：GSPO 把比率从 token 级升到序列级，CISPO 则改变裁剪的对象（裁 IS 权重而非丢弃梯度）。两者都面向长序列/长 CoT 场景，slime 均通过 `--advantage-estimator` 直接启用。

### 5.1 GSPO：序列级重要性采样

**结论：GSPO（Group Sequence Policy Optimization）在 GRPO 基础上，把"token 级"的重要性比率换成"序列级"（并做长度归一化）。这显著降低了长序列/长 CoT 场景下的梯度方差和训练不稳定，是 GRPO 的稳定化变体。slime 通过 `--advantage-estimator gspo` 支持。**

问题：GRPO 的 token 级比率 `r_{i,t}` 在长序列里会连乘出剧烈波动——个别 token 的比率异常会被裁剪机制放大，导致训练抖动甚至崩溃。

先回顾 **GRPO 的 token 级比率**（每个 token 一个，各自独立裁剪）：

$$r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}\mid\cdot)}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid\cdot)}$$

**GSPO 的做法**：改在**整条序列**上定义一个长度归一化的重要性比率（每条序列一个）：

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i\mid x)}\right)^{1/|y_i|} = \exp\!\left(\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t}\mid\cdot)}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid\cdot)}\right)$$

直观对比：GRPO 是“逐 token 算比率、逐 token 裁剪”——长序列里单 token 的比率异常会被裁剪放大；GSPO 是“整条序列算一个比率、整条序列裁剪一次”——把所有 token 的对数比率平均后取指数，抹平单 token 异常，长序列更稳。

- 括号内是整段回复的联合概率比；`1/|y_i|` 是按 token 数做几何平均（长度归一化），消除长度带来的量纲差异。
- 目标函数把裁剪也放到序列级：`min( s_i·Â_i, clip(s_i, 1-ε, 1+ε)·Â_i )`。

**GRPO vs GSPO 对比**：

| | GRPO | GSPO |
|---|---|---|
| 比率 | $r_{i,t}$（逐 token） | $s_i$（整条序列） |
| 归一化 | 无 | 开 $\lvert y_i\rvert$ 次方（几何平均） |
| 裁剪粒度 | 每个 token 一次 | 整条序列一次 |
| 长序列稳定性 | 一般 | 更好 |
| 优势来源 | 组内相对（相同） | 组内相对（相同） |

在 Dressage 中，`grpo` 与 `gspo` 共用同一套组内均值归一化的优势后处理（[reward_post_process.py](../dressage/training/reward_post_process.py)），差异体现在 slime 内部如何用这个优势构造 loss（token 级 vs 序列级比率）。

### 5.2 CISPO：裁剪 IS 权重，保留所有 token 的梯度

**结论：CISPO（Clipped Importance Sampling Policy Optimization，MiniMax-M1 提出，arXiv 2506.13585）与 PPO/GRPO 的根本区别在于"裁剪的是什么"：PPO 家族裁剪的是更新量（比率×优势），被裁 token 的梯度直接归零；CISPO 裁剪的是重要性采样权重本身——把 clip 后的比率在 stop-gradient 下当作固定权重，梯度经由 log π 流动，因此包括比率越界 token 在内的所有 token 都始终贡献梯度。slime 通过 `--advantage-estimator cispo` 支持。**

问题：PPO/GRPO 中比率 `r_t` 一旦越出 `[1-ε, 1+ε]`，目标取 clip 分支后不再随 θ 变化——**该 token 的梯度为零**，相当于被静默丢弃。而比率越界的 token 往往正是探索最激进的 token，长 CoT 训练中成片丢弃它们会损失大量学习信号。

**CISPO 的做法**（MiniMax-M1 论文 Eq. 4-5）：

$$L^{\mathrm{CISPO}}(\theta) = -\mathbb{E}\!\left[\mathrm{sg}\!\left(\mathrm{clip}(r_t,\ 1-\varepsilon_{\mathrm{low}},\ 1+\varepsilon_{\mathrm{high}})\right)\cdot \hat A_t \cdot \log\pi_\theta(a_t\mid s_t)\right]$$

- `sg(·)` 是 stop-gradient：clip 后的比率只作为**固定权重**乘在前面，自身不产生梯度。
- 梯度项是 `log π_θ`：无论比率是否被裁，每个 token 都照常获得"方向 ∇log π、幅度 = 截断权重 × 优势"的梯度。
- 一句话对比：PPO 说"步子太大的 token 就别再动了"（梯度归零）；CISPO 说"步子太大的 token 权重封顶，但梯度照传"。

slime 的实现见 [ppo_utils.py](../slime/slime/utils/ppo_utils.py) 的 `compute_cispo_loss`：核心一行是 `-ratio_truncated.detach() * advantages * log_probs`（`.detach()` 即 stop-gradient），与 `compute_policy_loss` 的 `min(r·A, clip(r)·A)` 结构形成鲜明对照。

**三种裁剪哲学对比**：

| | PPO / GRPO | GSPO | CISPO |
|---|---|---|---|
| 比率粒度 | 逐 token | 整条序列（长度归一化） | 逐 token |
| 裁剪的对象 | 更新量（比率×优势） | 序列级比率 | IS 权重（stop-gradient） |
| 被裁 token 的梯度 | 无（静默丢弃） | 无（整条限幅） | **有**（照常反传） |
| 面向的场景 | 通用 | 长序列 / MoE 稳定化 | 长 CoT 保留学习信号 |

**slime 落地**：

- `--advantage-estimator cispo` 与 `grpo`/`gspo` 共用同一套组内均值/std 归一化（[rollout.py](../slime/slime/ray/rollout.py) 第 668-685 行），差异只在 loss 构造（[loss.py](../slime/slime/backends/megatron_utils/loss.py) 第 978-981 行的分支）。
- 原版 CISPO 是**单边**的（只保留上界裁剪）：在 slime 中令 `--eps-clip >= 1.0` 即可关闭下界，否则启动时会打印 "CISPO is canonically single-sided" 的 warning（[arguments.py](../slime/slime/utils/arguments.py) 第 1826-1828 行）。

---

## 六、DAPO：面向长 CoT 的工程改进

**结论：DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）不是全新算法，而是在 GRPO 基础上打了四个关键补丁，专门解决"长思维链 RL"中的熵坍缩、无效梯度、长度偏置等问题。Dressage 的示例数据（`dressage_dapo_prompts*.jsonl`）和 `--eps-clip-high`/`--eps-clip-c` 参数正对应其中的技巧。**

四个改进：

### 6.1 Clip-Higher（解耦上下裁剪边界）

标准 PPO 上下用同一个 `ε`。DAPO 发现这会压制低概率 token 的概率增长，导致**熵坍缩**（模型越训越"死板"、丧失探索）。方案是解耦上下界：

$$\min\!\left(r_t\cdot A_t,\ \mathrm{clip}(r_t, 1-\varepsilon_{\mathrm{low}}, 1+\varepsilon_{\mathrm{high}})\cdot A_t\right),\quad \varepsilon_{\mathrm{high}} > \varepsilon_{\mathrm{low}}$$

调高 `ε_high` 给低概率 token 更多上升空间，保持探索。这对应 Dressage 脚本里的：

```bash
--eps-clip 0.2        # ε_low
--eps-clip-high 0.2   # ε_high（DAPO 中通常调高，如 0.28）
```

### 6.2 Dynamic Sampling（动态采样）

一组回复若奖励全对（全 1）或全错（全 0），组内均值 = 每个值 → 优势全为 0 → **该组不产生任何梯度**，白白浪费算力。动态采样会过滤掉这类"零信号"的 prompt 组，并重新采样补足 batch，保证每个 batch 都有有效梯度。

### 6.3 Token-Level Policy Gradient Loss（token 级损失聚合）

GRPO 默认按"样本平均"聚合损失，长回复里每个 token 的权重被稀释，导致长序列的学习信号偏弱、且长度容易失控。DAPO 改为在整个 batch 的 token 上直接做平均（token-mean），让长回复的每个 token 都获得同等权重。详见 [8.6 节](#86-损失聚合token-mean-vs-sequence-mean)。

### 6.4 Overlong Reward Shaping（超长惩罚）

对因触达长度上限而被截断的回复，直接给固定负奖励或屏蔽其损失，避免模型学到"写不完但不受罚"的坏行为。

此外，DAPO 在很多配置下**移除了 KL 惩罚项**——因为在可验证奖励场景，reward hacking 风险低，去掉 KL 反而给模型更大的优化自由度。

> `--eps-clip-c`（Dressage 示例中为 3.0）对应 **dual-clip PPO** 的常数 `c`：当优势为负且比率过大时，用 `max(min(r·A, clip(r)·A), c·A)` 再套一层下界，防止负优势样本产生过大的破坏性更新。它与 DAPO 的 clip-higher 是正交的两个裁剪技巧，见 [8.2 节](#82-裁剪与-dual-clip)。

---

## 七、免 Reward Model 的偏好优化：DPO

**结论：DPO（Direct Preference Optimization）用一个巧妙的数学等价，把"训练 RM + 跑 PPO"两步合并成一步监督式损失，直接在偏好对数据上优化策略。它不需要采样 rollout、不需要 RM、不需要 critic，因此训练极其简单——但它是离线的，无法像 GRPO 那样通过在线试错发现新策略。**

DPO 的核心洞见：RLHF 的最优策略与奖励之间存在闭式关系，反解出奖励可以用策略本身表示，从而把 RM 训练与 PPO 合并为一个损失：

$$L^{\mathrm{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right]$$

其中 `y_w` 是偏好回复（win），`y_l` 是较差回复（lose），`β` 控制与参考策略的偏离程度。

DPO vs PPO/GRPO：

| 维度 | DPO（离线） | PPO/GRPO（在线） |
|---|---|---|
| 是否需要 RM | 否 | PPO 需要 |
| 是否需要 rollout | 否（用固定偏好数据） | 需要 |
| 是否需要 critic | 否 | PPO 需要 |
| 能否探索出新策略 | 弱（受限于数据分布） | 强（在线试错） |
| 训练稳定性/成本 | 高 / 低 | 相对低 / 高 |
| 适合场景 | 偏好对齐、成本敏感 | 可验证奖励、需要探索 |

### 7.1 DPO 的变体家族

DPO 之后出现了一批针对其短板改进的离线偏好优化算法，核心差异在"参考模型是否必需、数据形态、目标函数"：

| 变体 | 针对的问题 | 核心改动 | 数据形态 |
|---|---|---|---|
| **IPO** | DPO 在确定性偏好数据上过拟合（logit 差距被无限拉大） | 把 log-odds 目标换成对奖励差值的平方回归，自带正则 | 偏好对 |
| **KTO** | 成对偏好标注昂贵 | 基于 Kahneman-Tversky 前景理论建模"得失效用"，只需好/坏二元标注，无需配对 | 单条 + 好坏标签 |
| **ORPO** | reference model 占显存、流程繁琐 | 免参考模型：SFT 损失 + 对 rejected 样本的 odds 惩罚，单阶段完成 | 偏好对 |
| **SimPO** | DPO 的隐式奖励与生成时的度量（平均 logprob）不一致 | 免参考模型：直接用平均 logprob 当隐式奖励，并引入目标奖励 margin | 偏好对 |

共同趋势：**去掉 reference model**（ORPO/SimPO）与**放宽数据形态**（KTO），进一步压低偏好优化的工程门槛。但它们都仍是离线范式，不改变下文"Dressage 不用 DPO"的判断。

> **为什么 Dressage 不用 DPO**：Dressage 面向的是**agentic、可验证奖励**的场景（代码是否通过测试、答案是否正确、任务是否完成），必须让模型在真实环境里多轮交互、试错——这依赖在线 rollout，正是 GRPO 家族的主场，而非 DPO 的离线范式。DPO 列在此处是为了让 RL 算法图谱完整。

---

## 八、横向拆解：损失函数的公共组件

上面这些在线算法（PPO/GRPO/GSPO/DAPO）共享同一套"零件"。理解这些零件，就能看懂任何一份训练脚本里的 RL 超参数。

### 8.1 重要性采样比率

**作用**：修正"用旧策略采样、用新策略评估"的分布偏移，使得一批数据可以做多步梯度更新（提高样本效率）。

$$\text{token 级（PPO/GRPO）:}\quad r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}$$

$$\text{序列级（GSPO）:}\quad s_i(\theta) = \left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i\mid x)}\right)^{1/|y_i|}$$

比率偏离 1 越远，说明新旧策略差异越大，越需要裁剪来约束。

> 重要性采样的完整原理——一般形式与支撑覆盖、轨迹级比率的方差爆炸、token 级比率的一阶近似本质、`π_old` 的三种语境含义、clip 的偏差-方差权衡——见 [2.5 节](#25-重要性采样原理深化与工程落点)。

### 8.2 裁剪与 dual-clip

**作用**：限制单步更新幅度，防止策略被少数高比率样本带崩。

- **标准裁剪**：`clip(r, 1-ε, 1+ε)`，上下对称。
- **Clip-Higher（DAPO）**：`clip(r, 1-ε_low, 1+ε_high)`，`ε_high>ε_low`，缓解熵坍缩（对应 `--eps-clip` 与 `--eps-clip-high`）。
- **Dual-Clip**：当 `A_t<0` 时再加一层下界 `c·A_t`（`c>1`），即

  $$L_t = \max\!\left(\min(r_t\cdot A_t,\ \mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)\cdot A_t),\ c\cdot A_t\right)$$

  防止"负优势 + 大比率"产生极端大的破坏性梯度（对应 `--eps-clip-c`，示例中 `c=3.0`）。

### 8.3 KL 散度的三种估计

**作用**：约束策略不要偏离参考策略 `π_ref` 太远，防止 reward hacking 与语言退化。

设比率 `ρ = π_ref/π_θ`，工程上有三种无需展开整个词表的 KL 估计（Schulman 的近似），slime 额外提供 k3 的数值稳定变体：

| 类型 | 公式 | 特点 | slime `--kl-loss-type` 取值 |
|---|---|---|---|
| k1 | `-log ρ` | 无偏但方差大，可能为负 | `k1`（**默认值**） |
| k2 | `0.5·(log ρ)^2` | 恒非负，有偏 | `k2` |
| k3 | `ρ - 1 - log ρ` | 无偏且恒非负、低方差 | `k3` |
| **k3 + 数值截断** | `clamp(ρ - 1 - log ρ, -10, 10)` | 同 k3，额外 clamp 防数值爆炸 | **`low_var_kl`** |

choices 定义见 [arguments.py](../slime/slime/utils/arguments.py) 第 924-929 行；k3 与 low_var_kl 共用同一公式、仅差一层 `clamp(min=-10, max=10)`，实现见 [ppo_utils.py](../slime/slime/utils/ppo_utils.py) 的 `compute_approx_kl`。另有开关 `--use-unbiased-kl`：给 KL 估计乘上重要性采样比率做无偏修正（DeepSeek-V3.2 的技巧），可与上述四种形式组合。

Dressage 示例用的正是 `--kl-loss-type low_var_kl`（k3 + clamp 变体）+ `--kl-loss-coef 0.001`（很小的系数，说明只做轻微约束）。

### 8.4 熵正则

**作用**：`H(π_θ) = -Σ π log π`，鼓励策略保持多样性、避免过早收敛到确定性输出（探索 vs 利用）。系数即 `entropy-coef`。

注意 Dressage 示例把 `--entropy-coef 0.00`，即**关闭独立熵奖励**——因为 DAPO 的 clip-higher 已经从另一条路径缓解熵坍缩，再叠加熵奖励可能引入不稳定。

### 8.5 优势归一化

**作用**：把优势数值缩放到统一尺度，稳定梯度。两个层面：

- **组内归一化**（GRPO 必备）：`Â_i = r_i - mean(group)`，可选再除以 `std(group)`（`grpo_std_normalization`）。
- **全局 batch 归一化**（Reinforce++ 常用）：对整个 batch 的优势做标准化。

Dressage 的 [reward_post_process.py](../dressage/training/reward_post_process.py) 实现的是组内均值归一化（第 144–158 行），并在多段轨迹场景把同一父轨迹的优势广播到所有 segment。

### 8.6 损失聚合：token-mean vs sequence-mean

**作用**：决定长短回复在总损失中的权重，直接影响长度偏置与长 CoT 学习效果。

- **sequence-mean（样本平均）**：先在每条序列内对 token 取平均，再对序列取平均。长短回复权重相同 → 长回复里单个 token 被稀释。
- **token-mean（token 平均，DAPO 推荐）**：对整个 batch 的所有有效 token 直接取平均。长回复贡献更多 token → 每个 token 权重相等，长序列学习信号更强。

"有效 token"由 `loss_mask` 决定——在 Dressage 中，哪些 token 参与损失由 [convert_samples.py](../dressage/rollout/convert_samples.py) 和 proxy 侧的掩码构建器控制，详见 [agentic-rl-training-zh.md](agentic-rl-training-zh.md) 的 2.8 节（loss_mask）。

### 8.7 训练-推理不一致与离线策略修正

**结论：即使权重完全同步，rollout 引擎（SGLang，推理优化 kernel + bf16）与训练引擎（Megatron，训练 kernel）对同一条序列算出的 logprob 也存在数值级差异——"采样分布 ≠ 训练侧认为的采样分布"，构成隐式 off-policy，且随序列变长累积放大。slime 提供三档对策：IS 比率分母直接复用 rollout 侧 logprob（`--use-rollout-logprobs`）、截断重要性采样 TIS（`--use-tis`）、离线程度过大时整条序列掩码 OPSM（`--use-opsm`）。**

问题来源有两层：

1. **引擎数值不一致**：推理引擎为吞吐优化（融合 kernel、bf16），训练引擎为精度优化；同一权重、同一 token，两边 logprob 有微小差异，逐 token 累积后在长序列上不可忽视。MoE 模型还叠加路由不一致，见 [8.8 节](#88-moe-模型-rl-的路由稳定性)。
2. **异步 staleness**：异步训练中 in-flight 请求跨越权重版本，`π_old` 本身就不是单一策略，见第十章陷阱 #6。

三档对策（按数据代价递增）：

| 对策 | 做法 | slime 参数 | 代价/备注 |
|---|---|---|---|
| 用 rollout 侧 logprob | IS 比率的分母直接用 rollout 引擎返回的 logprob，与真实采样分布严格一致 | `--use-rollout-logprobs` | 需回传 logprob；与 TIS 互斥（[arguments.py](../slime/slime/utils/arguments.py) 第 1801-1802 行的断言） |
| **TIS（截断重要性采样）** | 给 pg_loss 乘逐 token 权重 `w_t = clamp(exp(logp_train − logp_rollout), tis_clip_low, tis_clip)`，修正分布偏差的同时截断控方差；另有自定义函数变体（RS 风格：越界 token 直接掩码） | `--use-tis --tis-clip 2.0 --tis-clip-low 0`，可 `--custom-tis-function-path` 自定义 | 截断引入轻微偏差；实现见 [loss.py](../slime/slime/backends/megatron_utils/loss.py) 第 840-852 行 |
| OPSM（离线序列掩码） | 计算序列级新旧策略 KL，偏离超过阈值 δ 的整条序列 loss 置零——宁可不用也不硬修 | `--use-opsm --opsm-delta 1e-4` | 丢弃部分数据；实现见 [ppo_utils.py](../slime/slime/utils/ppo_utils.py) 的 `compute_opsm_mask` |

观测手段：`--get-mismatch-metrics` 统计 `tis` / `tis_clipfrac` / `train_rollout_logprob_abs_diff` 等指标，量化训练-推理不一致的程度（需配合 `--custom-tis-function-path`）。

### 8.8 MoE 模型 RL 的路由稳定性

**结论：MoE 模型的专家路由是离散选择，rollout 与训练两次前向即使加载同一权重，top-k 路由也可能选中不同专家——这让 logprob 波动远大于稠密模型，token 级 IS 比率更容易越界。这既是 GSPO 序列级比率的动机之一，也催生了更直接的解法：路由重放（routing replay），训练前向不重算路由、而是回放 rollout 时记录的路由决策。slime 对应 `--use-routing-replay` 与 `--use-rollout-routing-replay`。**

问题机理：

- MoE 每层 router 对 token 做 top-k 专家选择，是 argmax 式的离散决策
- 框架/精度/kernel 差异 → router logit 微小变化 → 选中的专家组合可能不同 → 同一 token 的 logprob 出现跳变（8.7 节数值不一致在 MoE 上的放大版）
- 后果：IS 比率 `r_t = π_θ/π_old` 在 MoE 上偏离 1 的幅度和频率都显著高于稠密模型，裁剪被频繁触发，训练不稳

实测量级（Dressage 在 Qwen3.5-35B-A3B，40 层 MoE / 256 experts / top-8 上的统计）：约 45% 的 (token, layer, slot) 专家选择不一致，约 85% 的 token 至少在一层存在路由差异——绝大多数训练样本的 logprob 都带系统性噪声。

对策谱系：

| 对策 | 思路 | slime 参数 |
|---|---|---|
| GSPO | 换粒度：序列级比率抹平单 token 跳变 | `--advantage-estimator gspo` |
| **R3（Routing Replay）** | 换因果：训练前向直接**回放** rollout 时记录的路由决策（`scores.gather` 替代 `torch.topk`），消除两次前向的路由差异 | `--use-routing-replay`（arXiv 2507.18071） |
| Rollout 侧路由回放 | 反向变体：rollout 引擎侧回放路由 | `--use-rollout-routing-replay`（arXiv 2510.11370） |

> Dressage 在真实 MoE 模型上系统实践了 R3：rollout 侧捕获路由（base64 int32 编码随轨迹传输）、训练侧 monkey-patch `compute_topk` 回放，训练-推理 logprob 绝对差异降低约 60%，额外开销接近零。详见 [02-r3-moe-routing-replay.md](final/02-r3-moe-routing-replay.md)。

---

## 九、这些算法在 Dressage/slime 中的落地

### 9.1 参数到概念的映射表

**结论：slime 的 RL 相关 CLI 参数几乎一一对应本文讲的算法组件。看懂下表，就能把训练脚本和算法原理对上号。**

| slime CLI 参数 | 对应算法概念 | 本文章节 |
|---|---|---|
| `--advantage-estimator grpo\|gspo\|cispo\|reinforce_plus_plus\|reinforce_plus_plus_baseline\|ppo` | 优势估计方法（仅 `ppo` 启用 critic） | 三、四、五 |
| `--n-samples-per-prompt 8` | GRPO 组大小 `G` | 4.1 |
| `--disable-rewards-normalization` | 关闭优势均值归一化（默认开启） | 8.5 |
| `--disable-grpo-std-normalization` | 关闭组内 std 归一化（默认开启；Dr. GRPO 建议关闭） | 4.1 / 4.5 / 8.5 |
| `--eps-clip 0.2` | 裁剪下界 `ε_low` | 3.2 / 8.2 |
| `--eps-clip-high 0.2` | 裁剪上界 `ε_high`（clip-higher） | 6.1 / 8.2 |
| `--eps-clip-c 3.0` | dual-clip 常数 `c` | 8.2 |
| `--use-kl-loss` | 启用 KL 惩罚项 | 3.3 |
| `--kl-loss-coef 0.001` | KL 系数 `β` | 3.3 / 8.3 |
| `--kl-loss-type k1\|k2\|k3\|low_var_kl` | KL 估计形式（默认 `k1`） | 8.3 |
| `--use-unbiased-kl` | IS 比率加权 KL（DeepSeek-V3.2） | 8.3 |
| `--entropy-coef 0.00` | 熵正则系数 `c2` | 8.4 |
| `--calculate-per-token-loss` | token-mean 损失聚合（DAPO 推荐） | 6.3 / 8.6 |
| `--use-rollout-logprobs` | IS 比率分母用 rollout 侧 logprob | 8.7 |
| `--use-tis` / `--tis-clip` / `--tis-clip-low` | 截断重要性采样（离线修正） | 8.7 |
| `--use-opsm` / `--opsm-delta` | 离线序列掩码 | 8.7 |
| `--get-mismatch-metrics` | 观测训练-推理 logprob 不一致 | 8.7 |

### 9.2 一个真实的 GRPO 配置

以下取自 [run_hotpotqa_whitebox_agent_qwen3.5_4b_async.sh](../examples/scripts/run_hotpotqa_whitebox_agent_qwen3.5_4b_async.sh)（第 224–232 行）：

```bash
GRPO_ARGS=(
   --advantage-estimator grpo    # 用 GRPO 组内相对优势（无 critic）
   --use-kl-loss                 # 启用对参考策略的 KL 约束
   --kl-loss-coef 0.001          # KL 系数很小 → 只做轻微约束
   --kl-loss-type low_var_kl     # k3 低方差 KL 估计
   --entropy-coef 0.00           # 关闭独立熵奖励（靠 clip 结构维持探索）
   --eps-clip 0.2                # 裁剪下界
   --eps-clip-high 0.2           # 裁剪上界（可调高以缓解熵坍缩）
   --eps-clip-c 3.0              # dual-clip：约束负优势样本的更新幅度
)
```

配合采样配置：

```bash
--n-samples-per-prompt 8         # 每个 prompt 采 8 个回复组成一个 GRPO 组
--rollout-batch-size 32          # 每个 rollout batch 的 prompt 数
--rollout-max-response-len 4096  # 单条回复最大长度（超长会被截断）
```

这套配置可以读作："用 GRPO（每 prompt 8 采样）+ 轻量 KL 约束 + 三重裁剪（下界/上界/dual-clip）来训练一个 4B 模型的 HotpotQA agent"。

### 9.3 Dressage 的优势后处理

**结论：Dressage 在 [reward_post_process.py](../dressage/training/reward_post_process.py) 中实现了 GRPO/GSPO/Reinforce++ 共用的组内均值归一化，并额外处理了"多段轨迹"的优势广播——这是普通 RL 库没有的、为 agentic 多轮场景定制的一层。**

关键逻辑：

1. **提取原始奖励**：`raw_rewards = [s.reward for s in samples]`（第 108–114 行）。
2. **按 `group_index` 分组做均值归一化**：`normalized_i = r_i - mean(group)`，可选再除以 `std`（第 142–158 行）。
3. **多段轨迹广播**：当一条 agent 轨迹因历史重写被切成多个 segment 时，只有 anchor segment（`segment_index` 最大者）携带终点奖励；归一化后把该优势**广播到同一父轨迹的所有 segment**（`_broadcast_to_segments`，第 51–76 行），保证同一轨迹的所有训练片段拿到一致的优势信号。
4. **原始奖励刻意不广播**：`raw_rewards` 保持稀疏（只有 anchor 非零），以便下游按轨迹求和还原终点奖励，用于 wandb 的轨迹级指标统计（第 95–104 行注释）。

> **重要工程约束**：Dressage 因 slime 的一个 quirk，把优势归一化放进了 `convert_samples_to_train_data` 的第一步，因此**不要**再单独注册 `--custom-reward-post-process-path`（会重复处理或跳过归一化）。详见 [agentic-rl-training-zh.md](agentic-rl-training-zh.md) 的 3.4 与 6.1 节。

### 9.4 OPD：on-policy 蒸馏与 RL 的混合

**结论：slime 的 OPD（On-Policy Distillation）把"teacher 蒸馏"与"RL 优势"拧进同一条通道：学生照常 rollout，teacher 对同一批轨迹算 logprob，两者逐 token 的 KL 作为惩罚项**就地叠加到优势**上（而非独立的 loss 项）。它与任意优势估计器正交，由 `--use-opd --opd-type sglang|megatron --opd-kl-coef` 启用。**

工作方式（输入 → 输出）：

- **输入**：rollout 样本 + teacher 逐 token logprob。teacher 的提供方式由 `--opd-type` 决定：`sglang` 表示 teacher 跑在独立的 SGLang 服务上、rollout 阶段顺便取 logprob；`megatron` 表示用 `--opd-teacher-load` 加载 teacher 权重、训练阶段前向计算。
- **输出**：改写后的优势——`apply_opd_kl_to_advantages` 把 reverse KL（student − teacher）按 `--opd-kl-coef`（默认 1.0）加权后加进 `rollout_data["advantages"]`（[loss.py](../slime/slime/backends/megatron_utils/loss.py) 第 766-768 行），下游 loss 构造完全无感。

两个极端用法有助于理解它的定位：

- `--opd-kl-coef` 调小 → 退化为普通 RL（只在优势上带一点 teacher 约束）
- 任务奖励置 0 + 较大 `--opd-kl-coef` → 退化为**纯 on-policy 蒸馏**，学习信号全部来自 OPD KL 惩罚（[on_policy_distillation.py](../slime/slime/rollout/on_policy_distillation.py) 的注释明确说明了这一点）

直觉：RL 奖励告诉学生"方向对不对"，蒸馏告诉学生"老师具体怎么做"，OPD 让两者在同一优势通道里加权，省去独立的蒸馏 loss 和额外训练阶段。

> Dressage 在 OPD 之上扩展了多教师版本 **MOPD**：按样本 metadata 把轨迹路由到不同 teacher，分组计算 log-prob 后再交给 slime 的 OPD loss，详见 [mopd-architecture.md](mopd-architecture.md)。

### 9.5 Agentic 场景的 credit assignment

**结论：多轮 agentic 轨迹把"稀疏终点奖励"推到极致——数十轮交互、数万 token，只有一个终点标量。Dressage 的答案是"标量优势 + 结构化广播"：组内相对优势广播到全轨迹所有 token；历史重写导致轨迹断裂时，锚段（anchor segment）优势广播到所有兄弟 segment；聚合时按 prompt 等权（prompt-equal）避免多段轨迹权重膨胀。**

为什么 PPO 式 per-token value 在这里难训：

1. 价值网络要对"含工具返回的超长前缀"估值，输入分布极不均匀，critic 本身训不稳
2. 多轮轨迹的状态转移由外部环境（沙箱、API）决定，critic 的回归目标天然高方差
3. critic 与 actor 同尺寸 → 显存翻倍（3.4 节的老问题在长轨迹下更严重）

因此"组内标量优势"几乎成了 agentic RL 的默认选择，但要补两层结构化处理：

- **段级广播**：agent 重写历史会把轨迹切成多个 segment（各自是合法 token 序列），只有携带终点奖励的 anchor 段参与组内归一化，再把优势广播给兄弟段——见 9.3 节与 [multi-segment-design.md](multi-segment-design.md)。
- **prompt-equal 聚合**：同一轨迹产生的多个 segment 在聚合时只占一份权重，避免"断得越碎、梯度权重越大"的畸变，详见 [agentic-rl-training-zh.md](agentic-rl-training-zh.md) 的 prompt-equal 章节。

研究前沿的方向是在"标量广播"与"per-token value"之间找中间粒度——例如按轮切分的 turn-level advantage、分两层做组内比较的 GiGPO（Group-in-Group Policy Optimization）等。Dressage 的 multi-segment 机制为这类更细粒度的优势分配保留了数据结构上的接口（segment 边界本身就是天然的 turn 切分点）。

---

## 十、算法选型与常见陷阱

**结论：没有"最好"的算法，只有"最匹配你的奖励类型和资源"的算法。下表给出选型指引。**

| 你的场景 | 推荐算法 | 理由 |
|---|---|---|
| 有偏好排序数据、想对齐人类偏好 | RLHF + PPO | 经典成熟，RM 能建模模糊偏好 |
| 成本敏感、只有偏好对、不想跑 rollout | DPO | 免 RM/critic/rollout，训练最省 |
| 奖励可验证（答案对错/测试通过），要探索 | GRPO | 无 critic、省显存，RLVR 主流 |
| 长 CoT、训练不稳/熵坍缩 | GSPO / CISPO / DAPO | 序列级比率 / 保留被裁 token 梯度 / clip-higher 更稳 |
| 多轮 agent + 工具调用（Dressage 场景） | GRPO/GSPO + agentic rollout | 在线试错发现工具使用策略 |

常见陷阱：

1. **GRPO 组内奖励全同 → 零梯度**：一组回复全对或全错时优势全为 0，该组白采。用 DAPO 的动态采样过滤/补采（[6.2 节](#62-dynamic-sampling动态采样)）。
2. **KL 系数过大 → 学不动**：`β` 太大会把策略死死拴在参考模型上。Dressage 用 0.001 的极小值，甚至 DAPO 直接去掉 KL。
3. **熵坍缩**：对称裁剪会压制低概率 token，模型越训越死板。用 clip-higher（调高 `--eps-clip-high`）缓解。
4. **长度偏置**：sequence-mean 聚合会稀释长回复的 token 权重，导致长 CoT 学不好或长度失控。改用 token-mean（[8.6 节](#86-损失聚合token-mean-vs-sequence-mean)）。
5. **重要性比率爆炸**：token 级比率在长序列里剧烈波动。改用 GSPO 的序列级归一化比率。
6. **rollout 与训练权重错位**：异步训练中 in-flight 请求可能跨越权重版本，导致 logprob 与实际权重不一致。Dressage 用 pause/resume + staleness 追踪解决，见 [agentic-rl-training-zh.md](agentic-rl-training-zh.md) 的 4.5 节；算法层的配套修正（TIS / OPSM / rollout-logprobs）见 [8.7 节](#87-训练-推理不一致与离线策略修正)。

---

## 十一、总结与参考

### 一句话总结每个算法

- **PPO**：用"裁剪比率 + GAE 优势 + KL 约束"安全地做策略更新，需要 critic，是 RLHF 的基石。
- **GRPO**：用"组内相对优势"替代 critic，省显存，可验证奖励场景的主流选择。
- **GSPO**：把重要性比率从 token 级升到序列级（长度归一化），长序列训练更稳。
- **CISPO**：裁剪 IS 权重（stop-gradient）而非丢弃被裁 token 的梯度，长 CoT 下保留更多学习信号。
- **DAPO**：GRPO 的四个工程补丁（clip-higher / 动态采样 / token-level loss / 超长惩罚），面向长 CoT。
- **Dr. GRPO**：指出并移除 GRPO 的两个数学偏差（std 难度偏置、`1/|y|` 长度偏置），"做减法"的无偏修正。
- **DPO**：把 RM + PPO 合并成一步离线监督损失，免 rollout/critic，成本最低但探索弱。

### 核心脉络

所有这些算法都在回答同一个问题：**如何用一个稀疏的终点奖励，稳定地把策略往"好"的方向推，同时不让它跑崩？** 答案由几块拼图组成——优势怎么估（critic vs 组内 vs 偏好对）、更新幅度怎么限（裁剪）、偏离怎么约束（KL）、探索怎么保持（熵/clip-higher）。看懂了这几块拼图，任何新算法都只是它们的重新组合。

### 在 Dressage 中的位置

Dressage 本身**不实现 RL 算法**——算法（GRPO/GSPO/Reinforce++、裁剪、KL）由底层的 slime 训练引擎提供。Dressage 的职责是把 **agentic rollout（多轮交互 + 工具调用 + 沙箱）** 的轨迹，转换成 slime 能消费的训练样本，并做好 agentic 特有的优势后处理（组归一化 + 多段广播）。因此本文的算法与 [agentic-rl-training-zh.md](agentic-rl-training-zh.md) 的系统架构合起来，才是 Dressage 训练全貌。

### 延伸阅读

- 本仓库：[agentic-rl-training-zh.md](agentic-rl-training-zh.md)（系统架构）、[reward_post_process.py](../dressage/training/reward_post_process.py)（优势后处理实现）
- 经典论文：PPO（Schulman et al., 2017）、InstructGPT/RLHF（Ouyang et al., 2022）、GAE（Schulman et al., 2015）、DPO（Rafailov et al., 2023）、DeepSeekMath/GRPO（Shao et al., 2024）、GSPO（Qwen 团队, 2025）、DAPO（ByteDance/清华, 2025）
