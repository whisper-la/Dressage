# Multi-Segment 训练设计文档

> 对应代码：`dressage/proxy/server.py`、`dressage/proxy/session_manager.py`、`dressage/proxy/trajectory_store.py`、`dressage/proxy/tito/tito_tokenizer.py`、`dressage/rollout/multi_segment.py`、`dressage/rollout/convert_samples.py`、`dressage/rollout/artifacts/samples.py`、`dressage/training/reward_post_process.py`、`slime/slime/backends/megatron_utils/cp_utils.py`
>
> 文档结构：全文按"总—分—总"组织——第 1 章是全局总览，第 2-6 章分层展开，第 7 章收束为设计契约；每章内部同样按"章首结论 → 逐节展开 → 本章小结"组织。

## 1. 总览：token 层分段，轨迹层归因

### 1.1 一句话定位

Multi-Segment 训练解决的是长轨迹 Agent RL 里的一个核心工程问题：**一条轨迹在语义上仍然是一次完整尝试，但它的 token 序列在中途不一定还能作为一个连续样本来训练**。

在 SWE-Gym、Claw 这类任务里，Agent 会多轮调用 LLM、执行工具、观察环境、更新计划。训练侧理想上希望得到一条完整 token 序列：prompt、response、logprobs、loss mask 全部对齐，然后用终端 reward 更新模型。但真实 Agent 会压缩上下文、重写历史、改变工具定义，导致后续请求重新渲染出来的 token 前缀和之前记录的 token 对不上。

Multi-Segment 的设计是：

```text
无法保证连续拼接的地方切开
每个 segment 内部保证 token/logprob/loss_mask 对齐
训练时仍把多个 segment 视为同一条轨迹的兄弟样本
只对最后一个 segment 计算终端 reward
把归一化后的 advantage 广播给所有 segment
再用 prompt-equal loss 分母避免多段轨迹被重复放大
```

一句话总结：

**Multi-Segment 是"token 层分段、轨迹层归因"的训练机制。它让历史重写后的长轨迹不丢 token、不硬拼错误 token，也不因为 segment 数变多而获得不公平梯度权重。**

### 1.2 设计哲学：三个解耦

整个机制可以拆成三对相互独立的决策，它们分别回答三个不同层次的问题：

| 解耦 | 回答的问题 | 实现手段 |
|------|------------|----------|
| **数据表示 vs 奖励归因** | token 怎么切 与 reward 怎么算，是两件事 | segment 管切分；anchor + 广播管归因 |
| **统计 vs 训练** | 同一个 reward，统计口径和训练口径不同 | raw_reward 保持稀疏（统计）；advantage 广播（训练） |
| **段数 vs 权重** | 切几段是数据表示的偶然，不应影响梯度权重 | prompt-equal loss 分母 |

理解这三对解耦，就理解了后文所有具体规则"为什么是那样"。

### 1.3 全景数据流

```text
Agent ──/v1/chat/completions──▶ Proxy（server.py）
                                  │ 逐步记录 StepRecord
                                  │ （route / lineage / boundary / concat_* 数组）
                                  ▼
                          SessionManager（lineage 管理 + boundary 检测，fail closed）
                                  │ rollout 结束，session finalize
                                  ▼
                          TrajectoryStore（双 view：lineage / timeline，0..N-1 完整性校验）
                                  │ 训练侧读取（默认 lineage view）
                                  ▼
                          multi_segment.expand_segments_to_samples（每段一个 Sample，共享 rollout_id）
                                  │ reward_fn 只对 anchor 执行
                                  ▼
                          reward_post_process（anchor 代表轨迹归一化 → advantage 广播兄弟段）
                                  ▼
                          convert_samples（prompt-equal 分母 rollout_mask_sums）
                                  ▼
                          slime 训练（sum_of_sample_mean reducer → 梯度）
```

### 1.4 全文地图

| 章节 | 回答的问题 |
|------|------------|
| 2 问题背景 | 为什么必须切段——token 对齐为什么是 loss 定义的前提 |
| 3 核心对象 | 切段涉及哪些概念——Segment / Lineage / Boundary / View / Anchor |
| 4 端到端流程 | 段怎么产生、怎么变成训练样本、怎么公平聚合（八步） |
| 5 例子 | 以上机制在一个 batch 里的完整数字演算 |
| 6 边界与局限 | 适用场景、常见误区、uniform advantage 的理论地位 |
| 7 收束 | 七条设计契约（每条对应一个"违反后果"） |

## 2. 问题背景：为什么长轨迹 Agent 拼不成一条 token 序列

**本章结论先行：Agent 的每次 LLM 调用都要把 messages 重新渲染成 token；一旦 Agent 压缩历史、回退重试或更换工具，新渲染结果与旧 token 序列的前缀关系就断了。而 logprob 是以前缀为条件的量 $\log\pi(y_t \mid y_{\lt t})$，前缀一断，其后所有 token 的训练信号在定义上失效——所以必须在断点处切开。切错方向的代价极不对称（错拼静默污染梯度、多切几乎无害），因此切段一律 fail closed。**

### 2.1 从 messages 到 tokens：渲染管线为什么天然脆弱

每次 LLM 调用，Agent 发出的是 **messages**（结构化对话列表），模型消费的是 **token ids**。中间隔着两步变换：

```text
messages ──chat template──▶ 渲染文本 ──tokenizer──▶ token ids
```

这条管线有两个结构性脆弱点：

1. **chat template 是全局渲染**。tools schema 会被渲染进 system prompt、每条消息有 role 包装和特殊分隔 token。messages 里任何一处改动——哪怕改的是中间一条旧消息——渲染出的文本都会从改动点之后完全不同。
2. **tokenizer 是文本驱动的**。两段文本只要字符有差异，token 序列就不同；甚至同一字符流，前缀边界不同时 BPE 合并结果也可能不同。

但比渲染更根本的是 **logprob 的语义**。训练需要的是每个 response token 的条件对数概率：

$$
\log\pi_\theta(y_t \mid y_{\lt t})
$$

它**以前缀为条件**。前缀任何一个 token 变了，之后所有 token 的 logprob 在定义上就不是同一个量——不是数值差一点，而是语义失效。

PPO/GRPO 的 loss 又逐 token 计算重要性采样比率（见 [llm-rl-algorithms-zh.md](llm-rl-algorithms-zh.md) 8.1 节）：

$$
r_t(\theta) = \frac{\pi_\theta(y_t \mid y_{\lt t})}{\pi_{\theta_{old}}(y_t \mid y_{\lt t})}
$$

这要求 rollout 记录的 $\pi_{old}$ 与训练时前向的 $\pi_\theta$ **逐 token 一一对应**。错位一个 token，其后所有 $r_t$ 全部失真——而且这种错误不会抛任何异常，只会静默污染梯度。

**结论：token 对齐不是工程洁癖，而是 loss 定义的前提。"能不能拼"必须由客观判据决定，不能靠侥幸。**

### 2.2 为什么长轨迹 Agent 会破坏 token 连续性

普通 RLHF 或数学题训练常常是一条 prompt 对一条 response，token 序列从输入到输出比较稳定。但 Agent 训练不一样：每一步 LLM 调用都可能重新构造完整 messages，再经过 chat template 渲染成 token。

以下行为都会破坏 append-only 假设：

| 场景 | 发生了什么 | 为什么 token 会断 |
|------|------------|-------------------|
| 上下文压缩 | Agent 把早期对话摘要成短消息 | 新 messages 不再包含旧 messages 的完整前缀 |
| 策略回退 | Agent 回到早期状态重新规划 | 同一 turn 可能不再只是追加 |
| 工具变更 | tools schema 增删改 | tools 参与 system prompt 渲染，prompt token 改变 |
| chat template 差异 | 模板渲染细节变化 | 文本相似也可能 token 前缀不同 |
| TITO 后缀差分失败 | `rendered_with` 不是 `rendered_without` 的前缀 | 无法提取增量 token |

一个具体的压缩例子：

```text
Step 3 的 messages:                Step 4 的 messages:
  [system]                           [system]
  [user] 任务                        [user] 任务
  [assistant] 分析 A   ──压缩──▶     [assistant] <摘要：之前尝试了 A、B…>
  [tool] 结果 X                      [assistant] 新计划 C
  [assistant] 分析 B
  [tool] 结果 Y
```

Step 4 的渲染文本从第 3 条消息起与 Step 3 完全不同 → token 前缀在此分歧 → Step 4 的 response logprob 不能接在 Step 1-3 的 token 序列后面训练。

**TITO 的 suffix 差分原理**（[tito_tokenizer.py](../dressage/proxy/tito/tito_tokenizer.py)）：TITO 让每步只记录**增量** token，前提是"新渲染结果以旧渲染结果为文本前缀"：

```python
rendered_without = render(base_messages)                        # 旧上下文
rendered_with    = render(base_messages + appended_messages)    # 旧 + 新增
if not rendered_with.startswith(rendered_without):
    raise ValueError("rendered suffix diff failed")             # 前缀不成立 → 切段信号
incremental_text = rendered_with[len(rendered_without):]
return encode(incremental_text)   # 本步的 concat_token_ids
```

每步产出的 `concat_token_ids` / `concat_response_logprobs` / `concat_response_mask` 就是"这一步新增"的 token 及其对齐数组。差分前提一旦不成立，增量在定义上不存在，只能切段。

### 2.3 没有 Multi-Segment 会怎样

没有这个机制时，系统通常只能在几个坏方案中选一个：

| 方案 | 看起来简单 | 实际问题 |
|------|------------|----------|
| 只保留最后一段 | 避免拼错 token | 丢掉前期推理、工具使用和探索 token；长程行为学不到 |
| 强行拼接所有 token | 数据量最大 | token/logprob/mask 错位，$r_t$ 全错，训练样本不可信且无声 |
| 每段当独立轨迹训练 | 实现直观 | 中间段没有可靠终态 reward；GRPO group 中长轨迹被重复计权（见 5.2） |
| 给每段伪造过程奖励 | credit 更细 | 需要额外 PRM 或启发式，容易引入 reward hacking |

Multi-Segment 选择的是更保守的中间路线：**不跨越不可信 boundary，但也不把切开的 segment 当作独立轨迹**。

### 2.4 fail closed：为什么宁可多切，不可错拼

切段判定时两类错误的成本极不对称：

| 错误 | 代价 |
|------|------|
| **错拼**（该切没切） | logprob/mask 错位 → 梯度被污染且**无任何报错**，训练悄悄劣化 |
| **多切**（不该切切了） | 多一个 segment sample；advantage 照常广播，prompt-equal 分母下权重几乎不变（见 4.7） |

错拼是"静默的错误数据"，多切只是"多一条正确但冗余的样本"。所以 boundary 检测的总原则是 **fail closed**：只要不能证明 token 连续，就切。这也是 `tools_changed`、路由渲染失败这类"可疑但未必真断"的情况一律切段、而不是尝试修复的原因。

### 2.5 它为什么重要

对长轨迹 Agent 来说，失败或成功往往不是最后一句话决定的。前面的文件定位、环境观察、工具调用、错误分析都携带训练价值。如果因为一次 context compaction 就丢弃前半条轨迹，训练会系统性忽略长程探索行为。

同时，RL 算法又不能让"切成 5 段的轨迹"比"没切段的轨迹"天然大 5 倍。Multi-Segment 的价值就在这里：既保住 token，又守住 reward 和梯度公平。

### 2.6 本章小结

```text
渲染管线（messages → template → tokens）决定了前缀连续性只是偶然；
logprob 的条件语义决定了前缀一断、其后 token 的训练信号全部失效；
代价不对称（错拼无声污染 vs 多切几乎无害）决定了切段策略必须 fail closed。
```

## 3. 核心对象：Segment、Lineage、Boundary、View、Anchor

**本章结论先行**：五个核心对象（外加 TITO 这个机制）各自回答一个问题，层级关系是**轨迹 ⊃ lineage ⊃ segment ⊃ step**：

| 对象 | 回答的问题 | 一句话定义 |
|------|------------|------------|
| TITO（机制） | 增量 token 怎么来 | 后缀差分：每步只记录"新增"的 token 及对齐数组 |
| Lineage | 哪些 step 共享同一条消息历史 | append-only 的血缘链 |
| Segment | 哪些 token 能合法拼在一起训练 | token 连续且数组对齐的训练单元 |
| Boundary | 在哪里切 | 两类标记：timeline 记"发生了什么"，lineage 记"还能不能拼" |
| View | 切好的段怎么组织 | lineage（紧凑增量，训练用）/ timeline（逐步快照，审计用） |
| Anchor | 一条轨迹谁来领 reward | segment_index 最大的段 |

一句话串起来：**TITO 产增量 → lineage 定范围 → boundary 定切点 → segment 是结果 → view 定组织方式 → anchor 定谁来领 reward。**

### 3.1 Segment 是什么

Segment 是一条轨迹中**token 序列内部连续且数组对齐的一段训练单元**。对应 [trajectory_store.py](../dressage/proxy/trajectory_store.py) 第 13-36 行的 `TrajectorySegment` dataclass，完整字段：

| 字段 | 含义 |
|------|------|
| `uid` | 段唯一 id |
| `trajectory_id` | 所属轨迹（即 session_id） |
| `turn_id` / `instance_id` | 所属 turn / prompt 实例 |
| `segment_index` / `segment_count` | 段在当前 view 中的位置 / 当前 view 总段数 |
| `messages` / `tools` | 该段末尾 step 的请求快照 |
| `tokens` / `full_logprobs` / `full_loss_mask` | **三等长数组（核心 invariant）** |
| `aligned_response_length` | 对齐后的 response 长度 |
| `full_versions` | 逐 token 权重版本（staleness 追踪用） |
| `routed_experts` / `*_chunks` / `*_parts` | MoE 路由回放数据（R3 用，见 [02-r3-moe-routing-replay.md](final/02-r3-moe-routing-replay.md)） |
| `extra_info.segment_view` | `lineage` 或 `timeline` |
| `extra_info.segment_reasons` | 触发切段的原因列表 |

完整定义（[trajectory_store.py](../dressage/proxy/trajectory_store.py) 第 13-36 行）：

```python
@dataclass
class TrajectorySegment:
    """One finalized trajectory segment stored by the proxy."""
    uid: str
    trajectory_id: str          # 即 session_id（@property 做了别名）
    turn_id: str
    instance_id: str
    segment_index: int
    segment_count: int
    messages: list[dict]        # 段末 step 的请求快照
    tools: list[dict[str, Any]] | None
    tokens: list[int]           # 三个核心数组，必须等长
    full_logprobs: list[float]
    full_loss_mask: list[int]
    aligned_response_length: int
    full_versions: list[str] | None = None                 # 逐 token 权重版本
    routed_experts: str | None = None                      # R3 路由数据（三种格式）
    routed_experts_chunks: list[dict[str, Any]] | None = None
    routed_experts_parts: list[dict[str, Any]] | None = None
    label: Any | None = None
    finish_reason: str = "stop"
    timestamp: float = field(default_factory=time.time)
    extra_info: dict = field(default_factory=dict)
```

注意 `session_id` 只是 `trajectory_id` 的 `@property` 别名——两个名字一个东西，读代码时别当成两个字段。

写入 store 时有两道硬性校验（`TrajectoryStore.write_many`，trajectory_store.py 第 265-288 行，完整代码与逐层解释见 4.4 节）：

1. 同批 records 必须属于同一条 trajectory（`trajectory_id` / `instance_id` 唯一）；
2. **每个 view 内 `segment_index` 必须恰好构成 `0..N-1` 完整集合**，且所有段的 `segment_count` 一致——缺一段、重一段都直接 `ValueError`，防止"半条轨迹"悄悄进入训练。

Segment 不是独立 episode。它只是同一条 episode 的一个 token 连续片段。

### 3.2 Lineage 是什么

Lineage 表示一条**可以 append-only 延续的消息历史链**。[session_manager.py](../dressage/proxy/session_manager.py) 第 36-40 行的结构：

```python
@dataclass
class Lineage:
    id: str
    index: int
    latest_step_id: str | None
    branch_from_step_id: str | None
```

字段语义：

| 字段 | 含义 |
|------|------|
| `id` | lineage 唯一标识 |
| `index` | 第几条 lineage（首条为 0，每次分叉 +1） |
| `latest_step_id` | 该链上最近的 step——**只有队尾 step 能被 append** |
| `branch_from_step_id` | 从哪个 step 分叉而来（branch 时记录血缘；create 时为 None） |

**Lineage 的生命周期由三种 route 驱动**（`_select_route`，server.py 第 705-789 行）：

| route | 触发条件 | lineage 变化 | token 后果 |
|-------|----------|--------------|------------|
| `append` | 请求能接在某 lineage **队尾** step 之后 | 不变，续当前链 | 差分增量，直接拼接（无 boundary 时） |
| `branch` | 前缀只能对上某历史**中间点**（该 lineage 队尾已被更晚 step 占据） | 创建新 lineage，`branch_from_step_id` 记下分叉点 | 新链首段；分叉点前缀作 context（不训练），新 response 才训练 |
| `create` | 与任何历史 step 都对不上前缀 | 创建新 lineage，无父 | 新链首段，全量 tokenize |

典型 branch 场景：Step 3 失败后 harness 回滚到 Step 2 的状态换工具重试——新请求与 Step 3 前缀不符、与 Step 2 相符，而 Step 2 已不是队尾 → 分叉出 lineage-1。典型 create 场景：轨迹首步（天然 create）、context compaction 后面目全非。

注意两个层次：**新 lineage 必然开始新 segment；但同一 lineage 内也可能因工具变化或 TITO tokenize 失败而额外切 segment**——lineage 是"消息历史链"，segment 是"token 连续段"，前者断裂必然导致后者断裂，反之不必。

### 3.3 Boundary 是什么

Boundary 是"当前 step 之前不能和上一段继续合并"的标记。Dressage 有两类 boundary，服务于两个不同的视角：

| Boundary | 字段 | 服务的 view | 回答的问题 |
|----------|------|-------------|------------|
| Timeline boundary | `segment_boundary_before` | snapshot / timeline | "这一步之前发生了什么断裂事件"（审计语义） |
| Lineage boundary | `lineage_segment_boundary_before` | TITO / lineage | "这一步的 `concat_*` 还能不能拼进当前段"（拼接语义） |

主要触发原因及判定逻辑：

| 原因 | 代码里的标记 | 判定逻辑 | 典型影响 |
|------|--------------|----------|----------|
| 历史重写 | `history_rewrite` | 同一 turn 下 messages 不再 append-only | 创建新 lineage（branch/create） |
| 消息前缀不匹配 | `message_prefix_mismatch` | TITO 模式下 route=create，但非显式 history rewrite | timeline 切段 |
| 工具变化 | `tools_changed` | route base step 的 tools 与当前 tools 不同（schema 参与渲染） | lineage 切段 |
| 路由渲染失败 | `tito_routing_render_failed` | 路由渲染或前缀比较抛异常 | **保守切段** |
| TITO tokenize 失败 | `tito_incremental_tokenization_failed` | 后缀差分前提不成立（2.2 节） | timeline 和 lineage 都切段 |

其中"保守切段"值得单独说：其他原因都是**确定断了**才切，而 `tito_routing_render_failed` 是**连判定本身都做不了**（渲染或比较过程抛异常）——无法证明连续，就当断开处理。这是 2.4 节 fail closed 原则的典型实例。

两类 boundary 的成分差异也由此解释：`history_rewrite` / `message_prefix_mismatch` 这两类消息层断裂已经通过"创建新 lineage"自然表达（新链在切段时天然就是新段），无需重复标记；而 `tools_changed` 等 token 层断裂发生在**同一 lineage 内**，必须显式标记才能被切开。所以 **timeline reasons ⊇ lineage reasons**。

### 3.4 Segment View 是什么

`TrajectoryStore` 支持两种 segment view：

| View | 构建方式 | 默认用途 |
|------|----------|----------|
| `lineage` | 按 lineage 和 lineage boundary 合并多个 step，拼接 TITO `concat_*` 数组 | TITO 模式默认训练视图：增量拼接、token 不重复，最紧凑 |
| `timeline` | 每个 step 一个 snapshot segment | snapshot 模式默认视图；也可用于调试审计（逐步完整现场） |

设计双视图的动机是分离"训练效率"与"可审计性"：lineage view 里一个 token 只出现一次，训练数据最小；timeline view 里每步是完整快照，排查问题时能逐帧回放。

在 `token_build_mode="tito"` 时，finalize 会同时写入两种 view。读取时若调用方不指定 view，`_default_segment_view()`（trajectory_store.py 第 101-108 行）按 `token_build_mode` 选择：含 `tito` → `lineage`；含 `snapshot` → `timeline`。

**不混用原则**：一次训练读取必须只选一个 view。两个 view 包含同样的 token（一份增量表示、一份快照表示），混用等于把同一 token 训练两遍。

### 3.5 Anchor 是什么

Anchor 是同一条轨迹中 `segment_index` 最大的 segment，也就是最后一段。Multi-Segment 规定：

```text
只有 anchor 保持 reward=None，让 reward_fn 真正执行
非 anchor 预设 reward=0.0
reward_post_process 只用 anchor 代表整条轨迹进入归一化
归一化后的 advantage 再广播给所有 sibling segments
```

Anchor 不是因为最后一段"更重要"，而是因为**终端 reward 通常只能在完整轨迹结束后评估**（测试是否通过、任务是否完成），只有最后一段携带终态。

代码层面的两道保障：

- 展开时非 anchor 段 `reward = 0.0`、anchor 段保持 `None` 以便 slime 执行 reward_fn（[multi_segment.py](../dressage/rollout/multi_segment.py) 第 188-189 行）；
- 若 segment_index 出现重复，anchor 选择在定义上就不可靠，`expand_segments_to_samples` 直接 `ValueError`（multi_segment.py 第 162-165 行）——又是 fail closed。

### 3.6 rollout_id 与层级模型

Multi-Segment 涉及四个层级，理解它们的包含关系是理解后文所有"分组/归一化/分母"的前提：

```text
Prompt（一道题 / 一个 instance，instance_id 标识）
  └─ Group（同一 prompt 的 G 次采样，GRPO 组内比较的单位，group_index 标识）
       └─ Trajectory（一次完整尝试，rollout_id 唯一标识）
            └─ Segment（轨迹的 token 连续片段，segment_index 标识；
                        同轨迹所有 segment 共享 rollout_id 与 parent_traj_id）
```

`rollout_id` 的取值是 `template_sample.index`（multi_segment.py 第 168 行），同轨迹所有 segment 共享。共享的目的：slime 的 `build_dp_schedule` 按 `rollout_id` 把同轨迹的 segment **排进同一个 training step**。这是优势广播的前提——anchor 的 advantage 广播给兄弟段之后，若兄弟段被拆到不同 step 训练，梯度口径就不一致了。

### 3.7 它不是什么

| 不是 | 说明 |
|------|------|
| 不是把长轨迹拆成多个 episode | 所有 segment 仍共享 `parent_traj_id`，reward 也只按轨迹算一次 |
| 不是过程奖励 | 中间 segment 没有单独 reward model |
| 不是 context window 扩展 | 它不让模型看到更多上下文，只保证训练数据表示正确 |
| 不是 Partial Rollout | Partial Rollout 处理**权重更新抢占**导致的续跑；Multi-Segment 处理**消息历史/token 前缀**断裂。两者可叠加（一条轨迹既被抢占又重写历史），但正交 |

### 3.8 本章小结

```text
TITO 产增量 → lineage 定范围 → boundary 定切点 → segment 是结果 → view 定组织方式 → anchor 定谁来领 reward。
层级不变式：轨迹 ⊃ lineage ⊃ segment ⊃ step；lineage 断则 segment 必断，反之不必。
```

## 4. 端到端流程：从 proxy 记录到训练归一化

**本章结论先行**：数据在八步中逐级变换形态——从"每次请求一条记录"，到"按轨迹组织的段"，再到"带公平分母的训练样本"。每一步只做一件事，且都留下可校验的不变式：

| 步 | 环节 | 数据形态变化 | 本步守护的不变式 |
|----|------|--------------|------------------|
| 4.1 | proxy 记录 | 一次请求 → 一条 StepRecord | 字段完整（route / lineage / concat_*） |
| 4.2 | boundary 检测 | StepRecord 打上两类 boundary 标记 | fail closed |
| 4.3 | finalize | steps → TrajectorySegment（双 view） | 段内三数组等长；view 内 0..N-1 完整 |
| 4.4 | 存储 | TrajectoryStore 双索引 | 一次读取只取一个 view |
| 4.5 | 展开 | 每段一个 Sample | 同轨迹共享 rollout_id / parent_traj_id |
| 4.6 | reward 后处理 | (raw_rewards, rewards) 两数组 | anchor 归一化、advantage 广播、raw 不广播 |
| 4.7 | 分母 | train_data["rollout_mask_sums"] | prompt-equal |
| 4.8 | 审计 | rollout/* 与 staleness/* 指标 | 失败样本不计入 |

### 4.1 第一步：proxy 记录 StepRecord

- **输入**：一次 `/v1/chat/completions` 请求（session + messages + tools）。
- **处理**：[server.py](../dressage/proxy/server.py) 做两件事——用 `_select_route` 决定这步接在哪段历史后面；TITO 模式下做后缀差分得到 `concat_*` 增量数组。
- **输出**：一条 `StepRecord` 写入 session。

route 判定就是 3.2 节的三种 route（append / branch / create），此处不再重复，只讲支撑判定的三个字段——它们解释了"前缀比较"到底是怎么做的：

| 概念 | 是什么 | 为什么需要 |
|------|--------|------------|
| `normalized_messages_snapshot` | 本步请求 messages 的规范化快照 | route 判定的第一层：用消息级前缀比较快速筛候选（snapshot 模式也靠它） |
| `prefix_tree` | session 内按 normalized messages 建的前缀树 | 让"哪些历史 step 是当前请求的前缀"从 O(N×消息数) 降到一次 trie 查询 |
| `snapshot_rendered` + `snapshot_tools_hash` | 候选 step 渲染文本的缓存，按 tools_hash 失效 | route 判定的第二层：渲染文本级前缀校验；tools 变了渲染就变了，缓存必须作废重渲 |

随后 proxy 把请求、响应、token、logprob、版本、route、lineage、boundary 原因等写入 `StepRecord`。和 Multi-Segment 密切相关的字段（session_manager.py 第 110-139 行）：

| 字段 | 用途 |
|------|------|
| `route_type` / `route_base_step_id` | 这步是 append / branch / create；拼接的锚点 step |
| `lineage_id` / `lineage_index` | 决定 lineage view 如何分组 |
| `segment_boundary_before` / `segment_reasons_before` | timeline 视图切段标记与原因 |
| `lineage_segment_boundary_before` / `lineage_segment_reasons_before` | lineage 视图切段标记与原因 |
| `concat_token_ids` | TITO 下可拼进 lineage segment 的增量 token |
| `concat_response_logprobs` / `concat_response_mask` | 与 `concat_token_ids` 对齐的 logprob / loss mask |
| `normalized_messages_snapshot` / `snapshot_rendered*` | 前缀比较与渲染审计材料 |

`StepRecord` 的完整字段分组（[session_manager.py](../dressage/proxy/session_manager.py) 第 92-143 行）——一个 step 同时携带全量和增量两套 token 表示，供两种 view 各取所需：

```python
@dataclass
class StepRecord:
    """One assistant generation step captured by the proxy."""
    # ── 身份与请求 ──
    turn_id: str
    request_messages: list[dict]
    normalized_request_messages: list[dict]
    raw_response_text: str

    # ── 全量 token 表示（snapshot / timeline view 用）──
    prompt_token_ids: list[int]
    snapshot_token_ids: list[int]
    all_token_ids: list[int]
    all_logprobs: list[float]
    response_token_ids: list[int]
    response_logprobs: list[float]

    # ── TITO 增量数组（lineage view 用，彼此等长）──
    concat_token_ids: list[int]
    concat_response_logprobs: list[float]
    concat_response_mask: list[int]
    concat_versions: list[str]
    concat_context_token_count: int
    concat_output_token_count: int
    concat_logprobs_invalid: bool
    concat_incremental_tokenization_failed: bool

    # ── 路由与血缘 ──
    step_id: str
    lineage_id: str
    lineage_index: int
    route_type: RouteType              # "append" / "branch" / "create"
    route_base_step_id: str | None

    # ── 两类 boundary 标记 ──
    segment_boundary_before: bool              # timeline 视角
    segment_reasons_before: list[str]
    lineage_segment_boundary_before: bool      # lineage 视角
    lineage_segment_reasons_before: list[str]

    # ── 渲染缓存与审计 ──
    snapshot_rendered: str                     # 渲染文本缓存
    snapshot_rendered_len: int
    snapshot_tools_hash: str | None            # tools 变则缓存失效
    prompt_versions / response_versions / all_versions   # 逐 token 权重版本

    # ── R3 路由数据 ──
    response_routed_experts: str | None
    response_routed_experts_chunks: list[dict]

    tools: list[dict] | None
    finish_reason: str = "stop"
    request_version: str | None
    response_version: str | None
    timestamp: float
```

这张"一个 step 两套表示"的结构正是双 view 存在的基础：finalize 时 lineage view 读 `concat_*` 列，timeline view 读 `all_token_ids` 列。

### 4.2 第二步：检测 boundary

- **输入**：当前请求的 messages/tools + session 历史 + 本步 route。
- **处理**：计算五个布尔判定量，每个回答一个具体问题。
- **输出**：更新本步的两类 boundary 标记。

五个判定量各自比较什么：

| 判定量 | 比较双方 | 判定为真的含义 |
|--------|----------|----------------|
| `rewrite_detected` | 同一 turn 的上一步 messages vs 当前 messages | 当前 messages 不是上一步的追加（append-only 被破坏）→ 显式历史重写 |
| `message_prefix_mismatch` | TITO 模式 + route=create + 非显式重写 | "看起来像新起点但不是明确重写"——消息层对不上，隐性断裂 |
| `tools_changed` | route base step 的 tools vs 当前 tools（规范化 JSON 比较，`None` 与 `[]` 按配置视为相等） | tools schema 变了 → system prompt 渲染变了 → 前缀 token 变了 |
| `routing_render_failed` | 路由渲染/前缀比较过程本身 | 判定过程抛异常——无法证明连续（保守切段的来源，见 3.3） |
| `tito_incremental_tokenization_failed` | `rendered_with` vs `rendered_without` | 后缀差分前提不成立（2.2 节的 `ValueError`） |

然后分别更新两类 boundary（为什么是"timeline ⊇ lineage"的成分差异，已在 3.3 节解释）：

```text
timeline boundary（记"发生了什么"）:
  history_rewrite
  message_prefix_mismatch
  tools_changed
  tito_routing_render_failed
  tito_incremental_tokenization_failed

lineage boundary（记"还能不能拼"）:
  tools_changed
  tito_routing_render_failed
  tito_incremental_tokenization_failed
  （新 lineage 自然形成新 segment，无需额外标记）
```

### 4.3 第三步：Session finalize 生成 segment records

- **输入**：session 中的全部 StepRecords。
- **处理**：按两种 view 分别切段、段内合并数组。
- **输出**：一组 `TrajectorySegment` records（带完整性校验）。

核心函数，每个都是"输入 → 输出"一个明确变换：

| 函数 | 输入 → 输出 | 作用 |
|------|-------------|------|
| `_split_session_into_lineage_segments()` | steps → 段划分 | 先按 lineage 分组，组内遇 `lineage_segment_boundary_before` 就切 |
| `_split_session_into_timeline_segments()` | steps → 段划分 | 每个 step 一个 timeline segment |
| `_build_tito_lineage_segment_record()` | 段内 steps → 一条 record | 合并所有 step 的 `concat_*` 数组 |
| `_build_step_snapshot_record()` | 单 step → 一条 record | 用该 step 的完整 snapshot token（`all_token_ids`，全量 prompt+response）构建 |

两个切分函数产出的中间结构是一个轻量 dict（不是 dataclass）：

```python
segment = {
    "steps": [StepRecord, ...],      # 该段包含的 step；lineage view 一段可多步，timeline 恒为单步
    "turn_ids": [...],               # 涉及的 turn 列表
    "segment_reason": str,           # 本段的切段原因（首个原因）
    "segment_reasons": list[str],    # 全部原因
    "lineage_id": ...,               # 仅 lineage view 携带
    "lineage_index": ...,
}
```

两种 view 的 record 构建差异（`_build_tito_lineage_segment_record` vs `_build_step_snapshot_record`，server.py 第 899-1123 行）：

| 维度 | lineage record（TITO） | timeline record（snapshot） |
|------|------------------------|------------------------------|
| `tokens` 来源 | 逐步 `extend(step.concat_token_ids)`（增量拼接） | `list(step.all_token_ids)`（单步全量） |
| 覆盖 step 数 | `num_steps ≥ 1` | 恒为 1 |
| `extra_info.segment_view` | `"lineage"` | `"timeline"` |
| `extra_info.alignment_method` | `"tito"` | `"snapshot_step"` |
| 对齐保障 | 三数组等长，否则 `RuntimeError`（含 `full_versions` 二次校验） | `_normalize_logprobs_to_length` 归一化，异常打 `snapshot_logprobs_invalid` 标记 |
| step 溯源 | `step_ids` 列表 + `branch_from_step_id` | 单 `step_id` + `route_type` |
| R3 路由数据 | `routed_experts_parts`（按 step 偏移组装） | `routed_experts` / `routed_experts_chunks`（直接透传） |

TITO lineage segment 的合并逻辑：

```text
tokens = []
response_logprobs = []
response_mask = []

for step in segment.steps:
    tokens += step.concat_token_ids
    response_logprobs += step.concat_response_logprobs
    response_mask += step.concat_response_mask

assert len(tokens) == len(response_logprobs) == len(response_mask)
```

这个 assert 是关键 invariant：只要写进 `TrajectoryStore`，segment 内部数组必须严格对齐（2.1 节：这是 loss 定义的前提）。随后 `write_many` 再做 3.1 节的 `0..N-1` 完整性校验。

真实代码中这不是 assert 而是显式抛错（server.py 第 1069-1083 行），fail closed：

```python
if not (len(tokens) == len(response_logprobs) == len(response_mask)):
    raise RuntimeError(
        "tito segment arrays are not aligned: "
        f"tokens={len(tokens)}, "
        f"full_logprobs={len(response_logprobs)}, "
        f"full_loss_mask={len(response_mask)}"
    )
if record_token_versions and len(full_versions) != len(tokens):
    raise RuntimeError(...)   # full_versions 也要等长
```

### 4.4 第四步：TrajectoryStore 存储和读取

- **输入**：finalize 产出的 `TrajectorySegment` records。
- **处理**：校验 + 双索引存储 + view 过滤。
- **输出**：`read_trajectory()` / `pop_trajectory()` 返回按 `(segment_index, timestamp)` 排序的 segment dicts。

`dressage/proxy/trajectory_store.py` 的 `TrajectoryStore` 做三件事：

| 职责 | 说明 | 为什么 |
|------|------|--------|
| schema 校验 | `tokens`、`full_logprobs`、`full_loss_mask` 必须等长；view 内段集合完整 | 坏数据在入库时就爆炸，而不是流到训练侧才发作 |
| 双索引存储 | 按 `instance_id` 和 `trajectory_id` 都能读 | rollout 按 prompt 实例组织，轨迹审计按轨迹组织，两种读法都要快 |
| view 过滤 | 根据 `extra_info.segment_view` 返回 lineage 或 timeline | 一次读取只取一个 view（3.4 节不混用原则） |

**入库校验的完整实现**（`write_many` → `_validate_finalized_batch`，trajectory_store.py 第 230-288 行）。输入是 finalize 产出的 segment dict 列表，输出是校验通过并原子写入的 `TrajectorySegment` 列表——任何一层校验不过，整批一条不写：

```python
def write_many(self, records):
    """Validate every record, then atomically publish the complete batch."""
    items = [self._item_from_dict(data) for data in records]  # dict → dataclass
    self._validate_finalized_batch(items)                     # 分层校验
    self._write_items(items)                                  # 全部通过才写入
    return items

@classmethod
def _validate_finalized_batch(cls, items):
    # 第 0 层：只对声明了"原子 finalize"的批次生效
    marked = any("finalization_id" in item.extra_info
                 or "finalization_complete" in item.extra_info for item in items)
    if not marked:
        return

    # 第 1 层：finalize 标记一致性——防半批提交
    finalization_ids = {item.extra_info.get("finalization_id") for item in items}
    if (len(finalization_ids) != 1
            or not isinstance(next(iter(finalization_ids)), str)
            or not next(iter(finalization_ids))
            or any(item.extra_info.get("finalization_complete") is not True
                   for item in items)):
        raise ValueError("finalized trajectory batch has inconsistent markers")

    # 第 2 层：轨迹身份唯一——防两条轨迹混进一批
    if len({item.trajectory_id for item in items}) != 1 or \
       len({item.instance_id for item in items}) != 1:
        raise ValueError("finalized trajectory batch mixes trajectory identities")

    # 第 3 层：每个 view 内 segment 集合完整——防半条轨迹
    by_view: dict[str, list[TrajectorySegment]] = {}
    for item in items:
        view = cls._item_segment_view(item)
        if view not in {"lineage", "timeline"}:
            raise ValueError("finalized trajectory batch has invalid segment_view")
        by_view.setdefault(view, []).append(item)
    for view, view_items in by_view.items():
        counts = {item.segment_count for item in view_items}
        indices = [item.segment_index for item in view_items]
        if (len(counts) != 1                                  # count 全组一致
                or isinstance(next(iter(counts)), bool)       # 且不能是 bool（见下）
                or next(iter(counts)) != len(view_items)      # count == 实际条数
                or any(isinstance(index, bool) for index in indices)
                or len(indices) != len(set(indices))          # index 无重复
                or set(indices) != set(range(len(view_items)))):  # 恰好覆盖 0..N-1
            raise ValueError(
                f"finalized trajectory {view} segments are not the complete 0..N-1 set"
            )
```

一个容易看漏的细节：**bool 检查**。Python 里 `True == 1`、`False == 0`，如果上游把布尔值混进 `segment_count`/`segment_index`，`set` 比较会被骗过（`{True}` 与 `{1}` 相等），所以用 `isinstance(x, bool)` 显式拒绝——这是防"类型混淆骗过集合比较"的防御性写法。

`pop_trajectory()` 是"读后即删"：rollout worker 在 finalize 后立即精确读走。没有这个语义，长时间运行的 fully async rollout 会把每条已完成轨迹永远留在 proxy 内存里。

### 4.5 第五步：rollout 展开成 Samples

- **输入**：`template_sample`（任务级模板，携带 label 等公共字段）+ 一条轨迹的 segment dicts。
- **处理**：[multi_segment.py](../dressage/rollout/multi_segment.py) 的 `expand_segments_to_samples()` 将 segment 展成多个 slime `Sample`。
- **输出**：长度 = 段数的 `list[Sample]`，按 `segment_index` 升序。

关键逻辑：

```python
sorted_segments = sorted(segments, key=lambda s: int(s.get("segment_index", 0)))
rollout_id = getattr(template_sample, "index", None)     # 轨迹级 id，全段共享

for i, segment in enumerate(sorted_segments):
    sample = copy.deepcopy(template_sample)              # 每段独立 Sample
    sample.rollout_id = rollout_id

    write_sample_from_segment(                           # 每段的 Sample 构造
        sample, args=args, segment=segment, all_segments=sorted_segments,
        session_id=sid, instance_id=iid, agent_response=agent_response,
    )

    sample.metadata["parent_traj_id"] = sid              # 奖励后处理的分组键
    sample.metadata["segment_index"] = int(segment.get("segment_index", 0))

    if i != last_idx:                                    # 非 anchor
        sample.reward = 0.0
```

三个设计点的"为什么"：

- **为什么 deepcopy 模板**：每段是独立 Sample（独立 tokens/loss_mask/logprobs），但任务级字段（label、instance 元数据）共享——deepcopy 模板是最安全的构造方式，避免段间字段串扰。
- **为什么非 anchor 是 `reward=0.0` 而 anchor 保持 `None`**：slime 只对 `reward is None` 的 sample 执行 reward_fn。非 anchor 预设 0.0 是占位（会被后续广播覆盖），anchor 的 None 是"请评分"的信号。
- **为什么防御重复 `segment_index`**：anchor 选择依赖 `segment_index` 取最大值，重复则"谁是 anchor"在定义上就不可靠，直接 `ValueError`（fail closed）。

`write_sample_from_segment()`（[samples.py](../dressage/rollout/artifacts/samples.py) 第 266-311 行）负责把一条 segment dict 转成 Sample 的训练字段：

- **输入**：segment 的 `tokens` / `full_loss_mask` / `full_logprobs` / `full_versions`。
- **处理**：定位 `response_start`（第一个 `loss_mask=1` 的位置——prompt 部分 mask 为 0，response 部分为 1）；`sample.loss_mask` 与 `rollout_log_probs` 只保留 response 段；超长时按 `token_cap` 截断保护；存在 `full_versions` 时可把非最新版本的 token mask 掉（staleness 控制）。
- **输出**：写好 `tokens` / `response_length` / `loss_mask` / `rollout_log_probs` 的 Sample，并校验 `len(loss_mask) == response_length`。

字段语义汇总：

| 字段 | 是否共享 | 作用 |
|------|----------|------|
| `rollout_id` | 同轨迹共享 | slime DP schedule 将同轨迹 segments 放进同一 training step（3.6 节） |
| `parent_traj_id` | 同轨迹共享 | reward 后处理和轨迹级统计用它识别 siblings |
| `segment_index` | 每段不同 | 最大 index 是 anchor |
| `segment_count` | 同 view 共享 | 记录这条轨迹在当前 view 下被切成几段 |
| `reward` | 每段不同 | 非 anchor 为 0，anchor 等 reward_fn 填 |

### 4.6 第六步：reward 后处理按轨迹归一化

- **输入**：本批全部 Samples（anchor 的 `reward` 已由 reward_fn 填好）。
- **处理**：[reward_post_process.py](../dressage/training/reward_post_process.py) 把 raw reward 转成训练用 advantage。
- **输出**：`(raw_rewards, rewards)` 两个等长数组——前者供统计，后者供训练。

算法分四步，每步的输入输出：

1. **提取**：`raw_rewards = [s.reward for s in samples]`（第 108-114 行；dict 型 reward 按 `reward_key` 取值）。
2. **建索引**（`_compute_parent_groups`，第 18-48 行）：输入 samples → 输出 `parent_groups: ptid → [sample_idx...]`（同轨迹所有段）和 `parent_anchor: ptid → anchor_idx`（segment_index 最大者）。`remove_sample=True` 的失败样本跳过。
3. **组内归一化**：按 `group_index` 分 GRPO 组，**只有 anchor 代表多段轨迹入组**（单段轨迹以自己入组，第 131-140 行）；`normalized = reward - group_mean`，可选再除以组内 std（std < 1e-6 跳过除法防爆，第 151-155 行）。
4. **广播**（`_broadcast_to_segments`，第 51-76 行）：输入 advantage 数组 + 两个索引 → 原地把 anchor 的 advantage 复制给所有兄弟段。

第 4 步的两个数组走向不同——这是"统计 vs 训练"解耦（1.2 节）的直接体现：

| 数组 | 是否广播 | 原因 |
|------|----------|------|
| `raw_rewards` | **不广播** | 保持终端 reward 稀疏（anchor 携带 R，其余 0.0），下游按 `parent_traj_id` 求和即可还原轨迹级 reward，供 wandb 轨迹级均值统计；广播会让 N 段轨迹被统计成 N 倍 |
| `rewards` / advantage | **广播** | 让每段的可训练 token 都获得轨迹级训练信号 |

### 4.7 第七步：prompt-equal loss 分母

**这一步在防什么**：即使 reward 只按 anchor 算、advantage 广播也做了，只要 loss 的分母按 segment 各自算，多段轨迹的总权重仍会随段数膨胀——分母是权重控制的最后一道闸。

分四层讲：reducer 机制 → 分母来源 → 三种分母的权重对比 → prompt-equal 公式推导。

**（1）reducer 机制**

slime 训练侧的 loss 归约函数是 [cp_utils.py](../slime/slime/backends/megatron_utils/cp_utils.py) 的 `sum_of_sample_mean`（第 73-81 行）：

$$
L = \sum_{i \in \text{batch}} \frac{\sum_{t} x_{i,t} \cdot m_{i,t}}{\mathrm{denom}_i}
$$

其中 $x_{i,t}$ 是样本 $i$ 第 $t$ 个 token 的 loss 贡献，$m_{i,t}$ 是 loss_mask。**$\mathrm{denom}_i$ 就是权重控制的阀门**。

**（2）分母来源**

$\mathrm{denom}_i$ 来自 rollout 侧预计算的 `train_data["rollout_mask_sums"]`（训练时由 [loss.py](../slime/slime/backends/megatron_utils/loss.py) 第 1256-1261 行传入 reducer）。若缺省，fallback 是**每个 sample 自己的** `loss_mask.sum()`（cp_utils.py 第 67-68 行）——即 sample 级均值。

[Dressage 的 convert_samples.py](../dressage/rollout/convert_samples.py) 替换了 slime 的 `_convert_samples_to_train_data`，提供两种分母：

| 分支 | 触发条件 | $\mathrm{denom}_i$ |
|------|----------|--------------------|
| trajectory 级（slime 默认） | 非 GRPO / Reinforce++ baseline | 所属 rollout 全部兄弟段的 mask 总和 $M_T$ |
| **prompt 级（prompt-equal）** | `--advantage-estimator grpo` 或 `reinforce_plus_plus_baseline` | $M_P \times N_P / \mathrm{gbs}$ |

**（3）三种分母的权重对比**

设轨迹 $T$ 有 $S_T$ 段，段 $s$ 的 mask 数为 $m_s$，轨迹级总和 $M_T = \sum_s m_s$，prompt 级总和 $M_P = \sum_{T \in P} M_T$，$\mathrm{gbs}$ 为 global batch size，$N_P$ 为 batch 中有效 prompt 数。用 $x_{i,t} = 1$ 作探针（即"这条轨迹/prompt 在 loss 里贡献了多少权重"）：

| 分母方案 | 单段贡献 | 轨迹 $T$ 贡献 | prompt $P$ 贡献 | 问题 |
|----------|----------|---------------|------------------|------|
| sample 级（fallback） | $m_s/m_s = 1$ | $S_T$（**段数膨胀**） | $\sum_T S_T$ | 切 3 段的轨迹权重 ×3 |
| trajectory 级（slime 默认） | $m_s/M_T$ | $1$（轨迹等权） | $G_P$ | prompt 内 token 不等权（短轨迹 token 权重被放大）；$G_P$ 不等时 prompt 间不公平 |
| **prompt 级（Dressage）** | $m_s/(M_P c)$ | $M_T/(M_P c)$ | $gbs/N_P$（**恒定**） | — |

其中 $c = N_P / \mathrm{gbs}$。

**（4）prompt-equal 公式推导**

`_prompt_equal_rollout_mask_sums()`（convert_samples.py 第 26-82 行）为每个 sample 给出：

$$
\mathrm{denom}_i = M_P \times \frac{N_P}{\mathrm{gbs}}
$$

验证 prompt $P$ 的总贡献（$x=1$ 探针，对该 prompt 所有 live sample 求和）：

$$
\sum_{T \in P}\sum_{s \in T} \frac{m_s}{M_P \cdot N_P/\mathrm{gbs}} = \frac{M_P \cdot \mathrm{gbs}}{M_P \cdot N_P} = \frac{\mathrm{gbs}}{N_P}
$$

**每个 prompt 在 loss 中的总权重恒为 $\mathrm{gbs}/N_P$**，与它被切成几段、有几条有效轨迹、长度多少全部无关。三个公平性推论：

1. **段数无关**：同一条轨迹切几段，分母都是同一个 $M_P \cdot c$——段数只是数据表示的偶然；
2. **prompt 内 token 等权**：同一 prompt 的所有 segment 共享分母，等价于 prompt 内的 token-mean——长轨迹按 token 数量自然贡献，与 GRPO"同一道题的 G 次尝试公平比较"的组内哲学一致（对比：trajectory 级分母会人为放大短轨迹每个 token 的权重，产生长度偏置）；
3. **prompt 间等权**：dynamic sampling 过滤后各 prompt 的有效轨迹数不等时，每个 prompt 的总权重仍恒定，防止某道题主导训练。

为什么只对 GRPO / Reinforce++ baseline 启用（`_PROMPT_EQUAL_ESTIMATORS`，convert_samples.py 第 23 行）：这两种估计器走"组内相对优势"路线，公平性必须在 prompt 级度量；GSPO 等走 slime 默认分母。

实现上的两条防御：

- live sample 必须同时有 `parent_traj_id` 和 `instance_id`（缺一直接 assert，第 51-58 行）；
- live sample 必须有 `group_index`——dead sample 用 `_NONE_GROUP = -1` 哨兵分组，若活样本混入该哨兵组会污染优势（第 60-65 行 assert）。

### 4.8 第八步：指标与审计

`compute_multi_segment_metrics()`（multi_segment.py 第 195-249 行）按 `parent_traj_id` 聚合输出：

| 指标 | 含义 | 怎么看 |
|------|------|--------|
| `rollout/num_segments` | 实际 segment 总数 | 与 `num_trajectories` 的比值即平均段数 |
| `rollout/num_trajectories` | 实际轨迹数 | 应接近 rollout_batch_size × n_samples |
| `rollout/segments_per_trajectory_mean/max/min` | 每条轨迹被切段的分布 | mean 异常升高说明 Agent 在频繁重写历史或换工具 |
| `staleness/version_span_mean/max/min` | 轨迹 token 权重版本跨度（合并 `full_versions`，滤除 `""/-1/none/unknown` 等非真实版本） | 跨度大说明 partial rollout 让轨迹跨了多次权重更新 |

`remove_sample=True` 的失败样本不计入。

### 4.9 本章小结

```text
proxy 层负责"记准"：StepRecord 字段完整 + boundary fail closed；
store 层负责"存全"：双 view + 0..N-1 完整性校验，坏数据入库即爆炸；
训练侧负责"算公平"：anchor 归一化（不多计）+ advantage 广播（不丢段）+ prompt-equal 分母（段数不进权重）。
```

## 5. 例子：一条轨迹如何被切段又作为整体训练

**本章结论先行**：五个例子分别验证五件事——切段怎么发生（5.1）、归一化不重复计权（5.2）、统计口径不被污染（5.3）、分母与段数无关（5.4）、端到端数字闭合（5.5）。

### 5.1 历史重写导致两个 lineage segments

假设一个 session 有 5 个 step：

```text
Step 1: route=create, lineage-001
Step 2: route=append, lineage-001
Step 3: route=append, lineage-001

Agent 压缩/重写历史

Step 4: route=create, lineage-002
Step 5: route=append, lineage-002
```

lineage view：

```text
segment-0: [Step1, Step2, Step3]
segment-1: [Step4, Step5]
```

训练展开：

```text
Sample 0:
  parent_traj_id = session_id
  segment_index = 0
  reward = 0.0          # 非 anchor 占位

Sample 1:
  parent_traj_id = session_id
  segment_index = 1
  reward = None         # anchor，等 reward_fn
```

reward_fn 只在 Sample 1 上执行。假设终端 reward 是 `R`，GRPO 归一化后 advantage 是 `A`：

```text
raw_rewards = [0.0, R]     # 稀疏，供统计
rewards     = [A,  A]      # 广播，供训练
```

### 5.2 多段轨迹不会在 GRPO 中重复计权

假设一个 GRPO group 中有两条轨迹：

```text
轨迹 A：3 segments，终端 reward = 1
轨迹 B：1 segment，终端 reward = 0
```

错误做法会把 group 看成 4 个样本：

```text
A.seg0, A.seg1, A.seg2, B
```

这样 A 在 group 均值里出现 3 次。Multi-Segment 的做法是只用 anchor：

```text
归一化成员：
  A.anchor reward=1
  B.anchor reward=0

mean = 0.5
A advantage = 0.5，广播给 A 三段
B advantage = -0.5
```

轨迹 A 只在 reward 归一化中出现一次，但它的三个 segment token 都能训练。

### 5.3 raw reward sparse 的统计意义

如果把 raw reward 也广播：

```text
轨迹 A 三段，每段 raw_reward=1
轨迹 B 一段，raw_reward=0
```

普通 `mean(raw_reward)` 会得到 `(1 + 1 + 1 + 0) / 4 = 0.75`。但真实轨迹级均值应该是 `(1 + 0) / 2 = 0.5`。

因此 raw reward 必须保持：

```text
A: [0.0, 0.0, 1.0]
B: [0.0]
```

需要轨迹级统计时，再按 `parent_traj_id` 求和恢复每条轨迹的 terminal reward。

### 5.4 prompt-equal 的直观例子

假设：

```text
轨迹 A：切成 3 段，总 trainable tokens = 900
轨迹 B：切成 1 段，总 trainable tokens = 900
```

如果每段各自作为 denominator（sample 级 fallback），A 的 3 段各自贡献 1 份权重，合计 3 份；B 只贡献 1 份——同样的 token 总量，权重差 3 倍。prompt-equal 则让 A 的三段共享同一个 prompt 级分母，A 的总贡献仍由其 900 个 token 决定，与 B 公平。

### 5.5 端到端完整数字演算

把前面所有机制放进一个 batch 里过一遍。设 batch 含 2 个 prompt、每个 prompt 2 条轨迹：

```text
Prompt 1:
  轨迹 A：3 段，每段 300 trainable tokens（M_A = 900）
  轨迹 B：1 段，300 trainable tokens（M_B = 300）
Prompt 2:
  轨迹 C：1 段，600 tokens
  轨迹 D：1 段，600 tokens

live samples = 3 + 1 + 1 + 1 = 6，gbs = 6，N_P = 2
M_P1 = 900 + 300 = 1200，M_P2 = 600 + 600 = 1200
```

**第一步：reward**。reward_fn 只在 4 个 anchor 上执行（A.seg2、B、C、D）；A.seg0/A.seg1 预设 0.0。

**第二步：anchor 归一化**。GRPO group 归一化只认 4 个 anchor：A.seg2 代表轨迹 A，而非 3 个样本。

**第三步：prompt-equal 分母**。

```text
denom(P1 的每个 sample) = M_P1 × N_P / gbs = 1200 × 2/6 = 400
denom(P2 的每个 sample) = M_P2 × N_P / gbs = 1200 × 2/6 = 400
```

**验证权重**（x=1 探针，即每个 trainable token 贡献 1）：

```text
A 的贡献 = 300/400 + 300/400 + 300/400 = 2.25
B 的贡献 = 300/400 = 0.75
P1 合计 = 3.0 = gbs/N_P ✓

C 的贡献 = 600/400 = 1.5
D 的贡献 = 600/400 = 1.5
P2 合计 = 3.0 = gbs/N_P ✓
```

三个要点全部兑现：

1. A 切成 3 段但总贡献 2.25 由 900 token 决定，**与段数无关**；
2. prompt 内 A（900 token）的贡献恰为 B（300 token）的 3 倍——**token 等权**；
3. P1 与 P2 总权重都是 3.0——**prompt 等权**。

**对比 trajectory 级分母（slime 默认）**：A 的 denom = 900、B 的 denom = 300，A 贡献 1、B 贡献 1——A 的 token 权重只有 B 的 1/3，短轨迹被放大；若 dynamic sampling 把 D 过滤掉，P2 只剩 C，P1 总权重 2、P2 总权重 1，prompt 间也不再公平。

### 5.6 本章小结

```text
五个例子里，"段数"从未进入任何一个权重公式：
归一化只认 anchor（5.2）、统计只认稀疏 raw reward（5.3）、分母只认 prompt 级 token 总数（5.4/5.5）。
段数只决定数据怎么摆，不决定数据多重。
```

## 6. 适用范围、误区与局限

**本章结论先行**：Multi-Segment 只解决"token 表示断裂 × 轨迹级归因"的交叉问题——它不改 reward、不扩上下文、不处理 staleness；最大的理论妥协是 uniform advantage，而它是终端奖励约束下不引入额外假设的最简选择，不是理论终点。

### 6.1 适合用在哪里

Multi-Segment 适合：

| 场景 | 原因 |
|------|------|
| 长上下文 Agent 训练 | 历史压缩和重写常见 |
| 工具调用型任务 | tools/schema 变化会影响 prompt |
| TITO 增量 tokenize | 需要在前缀不匹配时安全降级 |
| 终端 reward 任务 | reward 只能在整条轨迹结束后计算 |
| 多轮黑盒 Agent | 无法完全控制 Agent 是否重写 messages |

### 6.2 不适合解决什么

| 问题 | 为什么不属于它 |
|------|----------------|
| 需要精确 step-level credit assignment | Multi-Segment 只广播轨迹级 advantage |
| reward 本身不可靠 | 它不改 reward function |
| context window 不够 | 它处理训练数据，不扩展模型可见上下文 |
| 版本 staleness 过高 | 需要 Partial Rollout span limit 和 `rollout/staleness.py` |
| segment 内 logprob 缺失 | segment record 写入前必须保证数组对齐，否则应 fail closed |

### 6.3 常见误区

| 误区 | 正确认知 |
|------|----------|
| segment 越多，轨迹越重要 | 错。prompt-equal 和 anchor 归一化会避免段数偏置 |
| 非 anchor reward=0 表示前段是负样本 | 错。它只是占位，真实训练 advantage 会从 anchor 广播 |
| raw reward 和 advantage 应该一样广播 | 错。raw reward 为统计服务，advantage 为训练服务 |
| timeline view 和 lineage view 可以混着训练 | 不应混用。同一 token 会被训练两遍；一次训练读取应选择一个 segment view |
| 最后一段一定包含所有上下文 | 不一定。它只是终态评估 anchor，不代表它的 token 含有完整历史 |
| trajectory 级分母也能做到段数公平 | 部分对。它能做到轨迹等权，但做不到 prompt 内 token 等权和 prompt 间等权（5.5 节） |

### 6.4 当前局限

| 局限 | 说明 |
|------|------|
| Uniform advantage | 同一轨迹内所有 segment 获得同一个 advantage，无法区分哪个动作贡献更大 |
| 无过程奖励 | 中间步骤没有独立 reward |
| 无 value function | 不能做 per-step value-based credit assignment |
| 依赖 boundary 检测保守性 | 过于保守会增加 segment 数，过于激进会冒 token 错拼风险 |
| 训练统计更复杂 | raw reward、advantage、loss denominator 必须分别处理 |

其中 uniform advantage 值得展开。它的**理论地位**是：在只有终端奖励、没有任何 per-step 信号的约束下，把轨迹级 advantage 均匀赋给所有 token/segment，是不引入额外假设的最简选择——任何"给某段更大权重"的规则都需要额外信息支撑，否则就是注入人为先验。它的**已知偏差**是：一条长轨迹里早期关键决策（选对了探索方向）与后期普通执行获得相同 credit，因果链被抹平。

为什么不直接用更细的方案：

- **PPO value function**：critic 要对"含工具返回的超长前缀"估值，输入分布极不均匀、critic 自身训不稳；且 critic 与 actor 同尺寸，显存翻倍（见 [llm-rl-algorithms-zh.md](llm-rl-algorithms-zh.md) 9.5 节）。
- **过程奖励 PRM**：per-step 标注昂贵、泛化差，且容易被 hack（见 2.3 节坏方案表）。

因此 uniform advantage 是"终端奖励约束下的工程最优近似"，而不是理论终点。

### 6.5 可能的未来方向

| 方向 | 思路 | 风险 |
|------|------|------|
| Process Reward Model | 给中间 step 或 segment 评分 | 标注和泛化困难，容易被 hack |
| PPO + value function | 学状态价值做更细粒度 advantage | Agent 状态复杂，成本更高 |
| Segment 权重衰减 | 按位置或长度给广播 advantage 加权 | 权重规则缺少理论依据 |
| 成功/失败轨迹对比 | 从大量轨迹中学习 segment 级差异 | 实现复杂，信号噪声大 |

Dressage 的 segment 边界本身就是天然的 turn 切分点，数据结构（`parent_traj_id` / `segment_index` / 双 view）为以上任何更细粒度方案都留好了接口。

### 6.6 本章小结

```text
能用：长轨迹、工具调用、TITO、终端奖励、黑盒 Agent；
不管：reward 质量、上下文长度、staleness、per-step credit；
最大妥协：uniform advantage——无额外假设下的最简选择，不是理论终点。
```

## 7. 收束：设计契约

Multi-Segment 的本质价值是：**把"训练数据表示是否连续"与"轨迹 reward 如何归因"解耦**。

它最重要的设计契约（每条都对应一个"违反后果"）：

| # | 契约 | 违反后果 |
|---|------|----------|
| 1 | segment 内 token/logprob/loss_mask 严格等长 | 错位一个 token，之后所有 $r_t$ 失真且无声（2.1） |
| 2 | view 内 segment_index 构成 0..N-1 完整集合 | "半条轨迹"进训练，统计与归因同时错乱（3.1） |
| 3 | 同一轨迹共享 `parent_traj_id`，只由 anchor 代表轨迹计算 reward | GRPO group 中长轨迹被重复计权（5.2） |
| 4 | advantage 广播给 segment，raw reward 保持稀疏 | 广播 raw 会让 N 段轨迹统计成 N 倍（5.3） |
| 5 | loss 分母 prompt-equal | 段数/长度/组大小扭曲梯度权重（4.7、5.5） |
| 6 | 同轨迹 segment 共享 rollout_id、同 step 训练 | 广播后的梯度口径被拆乱（3.6） |
| 7 | boundary 判定 fail closed | 错拼的代价远大于多切（2.4） |

最需要记住的一点：

**切段是为了保证 token 正确，不是为了把一条轨迹变成多条轨迹。**
