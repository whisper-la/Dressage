# Partial Rollout 的 Staleness 控制设计说明

> 对应代码：`dressage/proxy/generation_controller.py`、`dressage/proxy/server.py`、`dressage/rollout/partial_async_rollout.py`、`dressage/rollout/fully_async_rollout.py`、`dressage/rollout/staleness.py`、`dressage/rollout/multi_segment.py`

## 1. 总览：异步 rollout 的版本质量闸

Partial Rollout 的 staleness 控制，解决的是异步训练里一个很现实的矛盾：**我们希望权重更新时不要浪费正在生成的长响应，但又不能让一条轨迹横跨太多代模型权重后仍进入训练**。

在普通 rollout 中，一次生成如果遇到权重更新，最保守的做法是中断并丢掉结果。这样最 on-policy，但长轨迹任务会浪费大量已经生成的 token。Partial Rollout 换了一个策略：权重更新时抢占正在运行的 SGLang 请求，保留已经生成的部分输出；新权重 resume 后，用“原始输入 + 已生成输出”继续生成，最后把多个分片拼成一个完整响应。

这带来了新的 staleness 问题：一条轨迹可能前半段来自 v1，后半段来自 v2，甚至继续跨到 v3/v4。Partial Rollout 的 staleness 控制不是禁止跨版本，而是设置上限：

```text
允许少量跨版本续跑以换吞吐
但限制单条轨迹的版本切换次数
超过 --max-partial-rollout-preempts 就拒绝该轨迹
```

一句话结论：

**Partial Rollout staleness 是轨迹级版本跨度限制。它允许一条轨迹被抢占并用新权重续跑，但通过 `--max-partial-rollout-preempts` 控制最多切换几次模型版本，超限时在 proxy 侧返回 `partial_rollout_staleness_exceeded`，训练侧再重试或丢弃。**

## 2. 问题来源：续跑提升吞吐，也会混合权重版本

### 2.1 为什么需要 Partial Rollout

Dressage 的目标场景包括 SWE-Gym、Claw、ALFWorld、HotpotQA 等长轨迹 Agent 训练。和短回答任务不同，这些任务常常有以下特征：

| 特征 | 具体表现 | 对 rollout 的影响 |
|------|----------|-------------------|
| 响应长 | 一次 assistant 输出可能包含大量推理和工具调用准备 | 单次生成耗时长 |
| 交互多 | 一个 episode 里有多轮 LLM 调用和环境反馈 | 一个样本占用 rollout worker 时间长 |
| 权重更新频繁 | 异步训练中 learner 持续推进新权重 | rollout 过程中容易遇到 pause/resume |
| 生成成本高 | 已生成 token 丢掉等于浪费算力和时间 | 简单重跑吞吐差 |

`GenerationController.generate_preemptible()` 的设计就是为了降低这种浪费。它在 proxy 层接管 SGLang 请求：

```text
pause 到来
→ abort active request_id
→ SGLang 返回已生成的部分 token
→ resume 后继续请求
→ 输入变成 original_input_ids + partial_output_ids
→ 多个 chunk 拼成一个 PreemptibleGenerateResult
```

这样，权重更新不再必然意味着“当前长响应作废”。对长轨迹任务来说，这是吞吐和资源利用率上的关键优化。

### 2.2 Partial Rollout 带来的新风险

Partial Rollout 的代价是：同一条轨迹的 token 可能来自多个权重版本。

例如：

```text
token 1-100:  weight v1
token 101-180: weight v2
token 181-260: weight v3
```

如果完全不限制，这条轨迹可能跨越很多次训练更新。它对当前策略的代表性会越来越差，也就是 off-policy 程度越来越高。训练这样的数据可能导致：

| 问题 | 说明 |
|------|------|
| on-policy 性下降 | GRPO/PPO 类训练默认假设 rollout 数据接近当前策略 |
| credit 归因变混 | 同一响应里不同行为来自不同版本，难以解释 reward 对哪个版本负责 |
| 质量不稳定 | 被频繁抢占的长响应可能混合多个策略阶段的风格和决策 |
| 训练指标失真 | 如果旧版本轨迹不断混入 batch，reward/advantage 反映的不是当前模型行为 |

因此 Partial Rollout 不能只解决“生成能续跑”，还必须回答“续跑到什么程度还可以训练”。

### 2.3 非 Partial 的零容忍为什么不适用

非 partial 模式的策略很严格：一条轨迹只要跨版本，就直接拒绝。`dressage/proxy/server.py` 里有几道相关检查：

| 检查 | 作用 | partial 模式下 |
|------|------|----------------|
| `_raise_if_cross_version_trajectory()` | session 历史版本和本次候选版本不一致就拒绝 | 关闭 |
| `_raise_if_stale_rollout_epoch()` | session 绑定 epoch 和当前 epoch 不一致就拒绝 | 关闭 |
| `GenerationStaleEpoch` | 生成中遇到 epoch 变化就拒绝 | partial 续跑路径替代 |

这些检查在 partial 模式下不能照搬，因为 partial 的核心目标就是允许中断后继续生成。于是系统引入了更细的规则：**非 partial 是零跨版，partial 是有限跨版**。

## 3. 关键概念：partial、版本跨度和 staleness

### 3.1 Partial Rollout 是什么

Partial Rollout 是一种可抢占生成机制。它允许一次逻辑上的 LLM 响应由多个实际 SGLang generate chunk 组成。

它的关键语义是：

```text
对调用方：仍然看到一次完整 chat/completions 响应
对 proxy：内部可能经历多次 abort/resume
对训练侧：样本 metadata 记录 token 级版本信息
```

`dressage/proxy/generation_controller.py` 里的 `PreemptibleGenerateResult` 会汇总：

| 字段 | 含义 |
|------|------|
| `output_ids` | 拼接后的完整输出 token |
| `output_token_logprobs` | 与输出 token 对齐的 logprob |
| `output_versions` | 每个输出 token 对应的权重版本 |
| `chunks` | 每次生成分片的信息 |
| `rollout_epoch` | 生成开始时观察到的 rollout epoch |

### 3.2 Staleness 在这里指什么

本文档中的 staleness 不是“时间过了多久”，而是“模型权重版本旧到什么程度”。

Partial Rollout 的 staleness 关注的是**轨迹内部的版本跨度**：

| 术语 | 定义 | 例子 |
|------|------|------|
| 权重版本 | SGLang 当前服务的模型版本标签 | `v1`、`v2`、`weight-42` |
| 版本跨度 `version_span` | 一条轨迹实际用过的不同真实版本数，按首次出现顺序去重 | `[v1, v1, v2] -> 2` |
| 版本切换数 `version_switches` | `max(0, version_span - 1)` | span=3 -> switches=2 |
| 最大抢占次数 `max_partial_rollout_preempts` | 单条轨迹允许的最大版本切换次数 | 1 表示最多跨到 2 个版本 |

无效版本标签会被忽略，例如 `""`、`"-1"`、`"unknown"`、`"none"`。

### 3.3 它不是什么

Partial Rollout staleness 容易和几个概念混淆：

| 容易混淆的概念 | 区别 |
|----------------|------|
| 不是组级 staleness 过滤 | 它在 proxy 生成过程中判断单条轨迹；`rollout/staleness.py` 是训练收集 batch 时过滤整个 prompt group |
| 不是 Multi-Segment | Multi-Segment 解决 token 历史重写导致的序列断裂；Partial staleness 解决权重版本跨越过多 |
| 不是 reward filtering | 它不看 reward 好坏，只看权重版本跨度 |
| 不是 wall-clock timeout | 它不按秒数判断，只按版本标签的变化判断 |

### 3.4 三层 staleness 治理

Dressage 实际上有三道闸：

| 闸口 | 粒度 | 位置 | 配置 | 解决的问题 |
|------|------|------|------|------------|
| 非 partial 零跨版 | 单条轨迹 | `proxy/server.py` | 自动生效 | 非 partial 样本必须来自同一代权重 |
| Partial 版本跨度限制 | 单条轨迹 | `proxy/server.py` | `--max-partial-rollout-preempts` | partial 轨迹不能跨太多代 |
| 组级新鲜度过滤 | prompt group | `rollout/staleness.py` | `dressage_staleness_keep_versions` | batch 收集时丢弃相对当前版本过旧的组 |

它们互补而不是替代关系。Partial 版本跨度限制防止“一条轨迹内部混太多版本”；组级过滤防止“已经完成但整体太旧的轨迹组继续进入训练”。

## 4. 工作机制：从抢占续跑到超限拒绝

### 4.1 抢占与续跑：GenerationController 做什么

`GenerationController` 是 Partial Rollout 的执行核心。它维护：

| 状态 | 作用 |
|------|------|
| `_active` | 当前正在 SGLang 侧运行的 request_id |
| `_paused` / `_resume_event` | 控制是否允许新 chunk 发往 SGLang |
| `_current_version` | resume 后当前权重版本 |
| `_rollout_epoch` | 权重更新的逻辑时钟 |

生成循环的核心思想是：

```text
while 还没生成够 max_new_tokens:
    等待 resume_event
    检查是否 stale epoch
    注册 active request
    调用 SGLang generate
    如果 pause 期间 abort 成功:
        收集 partial chunk
        generated_ids += partial output
        等待下一次 resume
    否则:
        收集 final chunk
        结束
```

对非 partial 来说，生成中断通常意味着失败。对 partial 来说，中断本身不是失败，只要能拿到部分 token，就可以继续。

### 4.2 版本从哪里来

版本信息主要有两个来源：

| 来源 | 说明 |
|------|------|
| SGLang 返回的 `weight_version` | 真实服务请求的权重版本 |
| 请求头 `X-Dressage-Expected-Version` | rollout 侧声明期望版本，用于版本钉住和审计 |

`generation_controller.py` 在每个 chunk 结束后，把该 chunk 的输出 token 都标上版本，形成 `output_versions`。之后 `server.py` 在写 `StepRecord` 时保存：

```text
response_version: 该响应的代表版本
response_versions: token 级或 chunk 级版本序列
```

再往训练侧，`dressage/rollout/artifacts/samples.py` 会从 `full_versions` 中提取：

| metadata | 含义 |
|----------|------|
| `full_versions` | token 级版本序列 |
| `version_spans` | 压缩后的连续版本段 |
| `dressage_start_token_version` | trainable output 的起始版本 |
| `dressage_end_token_version` | trainable output 的结束版本 |
| `dressage_partial_rollout` | token 级版本显示这条样本经历过 partial |

### 4.3 版本跨度怎么算

`server.py` 里的 `_ordered_real_versions()` 做两件事：

```text
1. 过滤无效版本：None、空字符串、-1、unknown、none
2. 去重但保持首次出现顺序
```

例子：

```text
输入: [v1, v1, unknown, v2, v1, v3]
输出: [v1, v2, v3]

version_span = 3
version_switches = 2
```

这里不关心每个版本出现了多少 token，也不关心版本标签的字典序，只关心“这条轨迹实际跨过多少个不同权重版本”。

### 4.4 超限判定

`server.py` 的 `_raise_if_partial_version_span_exceeded()` 是核心检查：

```text
如果不是 partial_rollout：返回
如果 max_partial_rollout_preempts is None：返回

versions = ordered_real_versions(session 历史 response_versions + 本次候选 output_versions)
version_span = len(versions)
version_switches = max(0, version_span - 1)

如果 version_switches <= max_partial_rollout_preempts:
    放行
否则:
    502 partial_rollout_staleness_exceeded
```

几个关键点：

| 细节 | 设计原因 |
|------|----------|
| 用 session 历史 + 本次候选版本一起算 | 一条 session 可能有多次 LLM 调用，不能只看当前 response |
| 在写 step 之前检查 | 超限时本次生成不污染 session |
| `None` 表示不限制 | 便于实验中关闭该闸口 |
| `max_preempts=0` 表示 partial 也不能跨版本 | 保留“允许抢占机制但不允许跨权重”的严格模式 |

### 4.5 拒绝后发生什么

超限时 proxy 返回 502，错误 detail 包含审计信息：

```text
error: partial_rollout_staleness_exceeded
versions: [v1, v2, v3]
version_span: 3
version_switches: 2
max_preempts: 1
max_version_span: 2
session_id / instance_id
```

这次生成发生在“记录 step 之前”，所以拒绝后不会写入 session。测试 `test_partial_rollout_rejects_staleness_exceeded` 覆盖了这个行为：轨迹被拒后 `session.steps` 仍为空。

### 4.6 Rollout 侧如何处理拒绝

502 回到训练侧后，由 `dressage/rollout/partial_async_rollout.py` 的失败分支处理。它复用 `fully_async_rollout.py` 里的 `_group_has_staleness_failure()` 识别错误对象或 sample metadata 中的 `partial_rollout_staleness_exceeded` 标记。

处理流程：

```text
completed group failed
    │
    ├─ 是否 staleness failure?
    │   └─ 是：累加 staleness_rejected_groups / samples
    │
    ├─ retry_count < DRESSAGE_ROLLOUT_MAX_RETRIES?
    │   ├─ 是：放回 data_buffer，用新 session 重跑
    │   └─ 否：计入 dropped_failed_groups
    │
    └─ dropped 太多则报错，避免无限等待 trainable batch
```

相关指标：

| 指标 | 含义 |
|------|------|
| `staleness/partial_rollout_rejected_groups` | 因 partial 版本跨度超限被拒的 group 数 |
| `staleness/partial_rollout_rejected_samples` | 被拒 group 中包含的 sample 数 |
| `staleness/version_span_mean/max/min` | 实际进入训练的轨迹版本跨度分布，由 `compute_multi_segment_metrics()` 输出 |

### 4.7 组级过滤如何补位

`dressage/rollout/staleness.py` 处理的是另一个层面：batch 收集过程中，已经完成的 group 可能在等待凑齐 batch 时变旧。

它维护一个按观察顺序增长的版本列表：

```text
versions = [v1, v2, v3, ...]
current_version_index = len(versions) - 1
cutoff_version_index = max(0, len(versions) - keep_versions)
```

一个 group 是否丢弃，取决于它里面任意轨迹的结束版本是否早于 cutoff。Multi-Segment 轨迹会按 `segment_index` 最大的 segment 的 `dressage_end_token_version` 作为轨迹版本，避免前段旧版本误导整条轨迹的新鲜度判断。

这就是为什么 Partial 版本跨度限制和组级过滤需要同时存在：

```text
Partial span limit:
  约束一条轨迹内部不要混太多版本

Group staleness filter:
  约束进入训练 batch 的 group 不要整体太旧
```

## 5. 典型场景与算例

### 5.1 正常放行：允许一次抢占

配置：

```text
--dressage-partial-rollout
--max-partial-rollout-preempts 1
```

生成过程：

| 阶段 | 事件 | token 版本 | 判定 |
|------|------|------------|------|
| chunk 1 | v1 生成 100 token 后 pause | `[v1]` | span=1, switches=0 |
| chunk 2 | resume 到 v2，生成完成 | `[v1, v2]` | span=2, switches=1 <= 1 |

结果：放行，step 写入 session，样本进入训练。metadata 中可看到 token 级版本跨度。

### 5.2 超限拒绝：跨到第三个版本

同样配置 `max_partial_rollout_preempts=1`。

| 阶段 | 事件 | 轨迹版本序列 | 判定 |
|------|------|--------------|------|
| chunk 1 | v1 生成后被抢占 | `[v1]` | switches=0 |
| chunk 2 | v2 续跑后又被抢占 | `[v1, v2]` | switches=1 |
| chunk 3 | v3 续跑完成 | `[v1, v2, v3]` | switches=2 > 1，拒绝 |

结果：

```text
HTTP 502
detail.error = partial_rollout_staleness_exceeded
detail.versions = [v1, v2, v3]
本次 step 不写入 session
rollout 侧重试或丢弃
```

### 5.3 `max_preempts=0` 的含义

`max_partial_rollout_preempts=0` 不等于关闭 partial。它表示仍然可以使用 partial 的控制路径，但单条轨迹不能跨权重版本。

这适合想保留 pause/resume 框架和审计字段，但训练策略要求非常严格 on-policy 的实验。

### 5.4 `max_preempts=None` 的含义

`None` 表示不做轨迹级跨度限制。系统仍可能通过组级 `dressage_staleness_keep_versions` 过滤太旧的 group，但不会因为一条轨迹内部跨了多个版本而在 proxy 阶段拒绝。

这适合早期吞吐实验，但不建议作为需要稳定 RL 训练质量的默认设置。

## 6. 配置、监控与使用边界

### 6.1 适合用在哪里

Partial Rollout staleness 控制适合以下场景：

| 场景 | 为什么适合 |
|------|------------|
| 长响应 Agent 训练 | 已生成 token 很贵，partial 可以减少浪费 |
| 异步 rollout + learner 持续更新 | 权重更新会频繁打断生成，需要控制跨版本成本 |
| GRPO / GSPO / PPO 类在线 RL | 需要兼顾吞吐和 on-policy 新鲜度 |
| 多 worker rollout | 不同 worker 完成速度不同，staleness 更容易出现 |

### 6.2 不适合依赖它解决的问题

| 问题 | 为什么不是它负责 |
|------|------------------|
| 对话历史重写导致 token 拼接断裂 | 这是 Multi-Segment / TITO boundary 的职责 |
| reward 质量差 | 它不检查 reward，只检查版本 |
| 某个 group 已经整体太旧 | 这是 `rollout/staleness.py` 的组级过滤职责 |
| SGLang 不返回可靠版本 | 需要 version routing / capability 检查保证版本来源可靠 |

### 6.3 常见误区

| 误区 | 正确认知 |
|------|----------|
| Partial Rollout 就是不管 staleness | 错。它允许有限跨版本，并用 `max_partial_rollout_preempts` 加边界 |
| `max_preempts=1` 表示最多用 1 个版本 | 错。它表示最多切换 1 次，也就是最多 2 个版本 |
| 被拒后可以继续沿用原 session | 错。拒绝发生在写 step 前，训练侧通常用新 session 重试 |
| 组级 keep_versions 可以替代 span limit | 不完全。组级过滤看 batch 新鲜度，不限制单轨迹内部混了多少版本 |
| 版本标签可以按大小排序 | 错。代码按观察顺序处理版本，把标签当 opaque string |

### 6.4 参数建议

经验上可以这样理解：

| 取值 | 语义 | 适用情况 |
|------|------|----------|
| `None` | 不限制单轨迹跨度 | 吞吐探索、调试 partial 行为 |
| `0` | partial 路径可用，但轨迹不能跨版本 | 极严格 on-policy 实验 |
| `1` | 允许一次抢占续跑 | 常见折中默认，最大跨度 2 |
| `2+` | 允许多次抢占 | 长响应极多、更新频率高，但要监控训练质量 |

同时建议观察：

```text
staleness/partial_rollout_rejected_groups
staleness/version_span_mean
staleness/version_span_max
staleness/version_gap_mean
```

如果拒绝率很高，说明权重更新频率、生成长度和 `max_preempts` 之间不匹配；如果 `version_span_max` 长期很高，则训练数据可能明显偏 off-policy。

### 6.5 与 Multi-Segment 的关系

Partial Rollout 和 Multi-Segment 可以同时出现在一条轨迹里，但它们解决的是正交问题：

| 机制 | 关注点 | 切分依据 |
|------|--------|----------|
| Partial staleness | 权重版本是否跨太多 | `output_versions` / `response_versions` |
| Multi-Segment | token 序列是否还能连续拼接 | history rewrite、tools change、TITO prefix mismatch |

一条轨迹可能因为 partial rollout 跨了 v1/v2，同时又因为 Agent 压缩历史被切成多个 segment。训练侧会用 Multi-Segment 保证 token 表示正确，用 staleness 控制保证版本新鲜度边界。

## 7. 收束：用有限跨版换可控吞吐

Partial Rollout 的本质是用“可抢占续跑”换取更高 rollout 吞吐；Partial Rollout staleness 控制的本质是给这个吞吐优化加一条质量边界。

最需要记住的是：

```text
非 partial：轨迹内零跨版
partial：允许有限跨版
组级 staleness：训练前过滤整体过旧的 group
```

`--max-partial-rollout-preempts` 控制的是**版本切换次数**，不是 token 数、时间，也不是 group 新鲜度。它让 Dressage 可以在长轨迹异步训练中保留已生成 token，同时避免把横跨过多权重版本的轨迹送进训练。
