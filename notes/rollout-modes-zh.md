# 从同步到部分异步：Dressage 三种 Rollout 模式的来龙去脉

> **本文目的**：把 Dressage 的三种 rollout 调度模式——**同步（sync）**、**完全异步（fully async）**、**部分异步（partial async）**——从"为什么会有它们"讲到"代码里到底怎么跑"，让你读完之后能自己判断"我这套训练该用哪一种，以及它背后省了什么、又多担了什么风险"。
>
> **适用读者**：正在做 Agentic RL 训练、对 rollout 与训练如何在时间和空间上"排班"感到困惑的工程师与研究者。
>
> **阅读建议**：第 0～2 节建立共同语言，第 3～6 节是三种模式的逐层递进，第 7 节补上"部分异步为什么必须配 pause/resume"这块拼图，第 8 节给选型结论。建议顺序阅读。

---

## 目录

- [0. 开篇：Rollout 调度为什么会成为 Agentic RL 的核心矛盾](#0-开篇rollout-调度为什么会成为-agentic-rl-的核心矛盾)
- [1. 一句话定位：三种模式各自在回答什么问题](#1-一句话定位三种模式各自在回答什么问题)
- [2. 共享地基：三种模式复用的同一套零件](#2-共享地基三种模式复用的同一套零件)
- [3. 同步模式：最朴素，也最稳](#3-同步模式最朴素也最稳)
- [4. 完全异步模式：让训练不再等 Agent](#4-完全异步模式让训练不再等-agent)
- [5. Staleness：异步换来的吞吐，代价是"新鲜度"](#5-staleness异步换来的吞吐代价是新鲜度)
- [6. 部分异步模式：只取够训练的那一份](#6-部分异步模式只取够训练的那一份)
- [7. 最后一块拼图：pause/resume 与 token 级续跑](#7-最后一块拼图pauseresume-与-token-级续跑)
- [8. 怎么选：一张决策图收尾](#8-怎么选一张决策图收尾)
- [9. 结语](#9-结语)

---

## 0. 开篇：Rollout 调度为什么会成为 Agentic RL 的核心矛盾

**结论先行：三种 rollout 模式，本质上是对同一个问题的三种回答——"负责采样的 GPU 和负责训练的 GPU，在时间上要不要重叠、在空间上要不要分开"。同步不重叠也不分开，完全异步既重叠又分开，部分异步则在"重叠"的基础上进一步只取够一步训练的量。**

先说清楚一件事：这里的 **rollout（轨迹采样）**，指的是让当前的模型策略去和环境交互、跑出一条完整轨迹的过程。在传统 RLHF 里，一次 rollout 就是"模型一口气生成一段回答"，快、齐、方差小。但在 **Agentic RL（智能体强化学习）** 里，一次 rollout 是这样的：

- 模型要和外部环境多轮交互（调用工具、读写文件、执行命令）；
- 工具执行要在**沙箱**（隔离的执行环境）里进行，可能跑个测试就要几十秒；
- 甚至整个 agent 循环是交给一个跑在沙箱里的**外部 HTTP agent**（黑盒模式）来完成的，Dressage 只负责把它拉起来、喂任务、等它跑完。

这就带来两个 RLHF 时代不太严重、但在 Agentic RL 里致命的特性：

1. **延迟高**：一条轨迹动辄几分钟；
2. **方差大**：同一批里，有的 prompt 三轮就结束，有的要几十轮，尾部拖得很长。

于是矛盾出现了。训练用的 GPU（跑 Megatron 反向传播）非常贵，如果它必须**傻等**整批轨迹全部采样完才能开工，那么在采样阶段，训练 GPU 就是纯空转。轨迹越长、方差越大，空转越严重。

三种 rollout 模式，就是围绕"怎么把训练 GPU 的空转时间榨干"这一件事，逐步演化出来的。理解了这条主线，后面所有的代码细节都能对号入座。

---

## 1. 一句话定位：三种模式各自在回答什么问题

**结论先行：sync 追求"简单确定"，fully async 追求"吞吐最大化"，partial async 在 fully async 的基础上追求"每一步训练都拿到够用且尽量新鲜的数据"。**

| 维度 | 同步 sync | 完全异步 fully async | 部分异步 partial async |
|---|---|---|---|
| GPU 拓扑 | 共置（rollout 与 train 共享同一批 GPU） | 分离（rollout 与 train 各有独立 GPU 池） | 分离 |
| 时间关系 | 采样与训练**串行**，交替占用 GPU | 采样与训练**重叠**，后台常驻采样 | 采样与训练**重叠** |
| 返回时机 | 整批全部跑完才返回 | 攒够 `rollout_batch_size` 个组才返回 | 攒够"一步训练所需"的组就早退 |
| 核心诉求 | 训练前能卸载 SGLang，腾出显存 | 隐藏 Agent 延迟，训练不空等 | 解决"采样量 ≫ 单步训练量"的错配 |
| 新鲜度问题 | 天然没有（整批同权重） | 有，靠 staleness 过滤 | 有，靠 staleness + pause/resume |
| slime 入口函数 | `generate_rollout_sync` | `generate_rollout_fully_async` | `generate_rollout_partial_async` |
| 典型场景 | 35B-A3B 共享 8×H100 调试 | 黑盒长轨迹的生产训练默认 | 大规模异步 + 连续训练 |

这张表现在看不懂没关系，它是全文的"地图"。下面我们从三种模式共用的那套零件讲起。

---

## 2. 共享地基：三种模式复用的同一套零件

**结论先行：三种模式在"如何采一条轨迹""失败怎么重试""什么样的批次算废批"这些问题上用的是完全相同的代码。它们的差异只在于"如何调度这些轨迹的采样和收集"。**

在深入三种模式之前，先把它们脚下这块共同的地基铺清楚，否则后面会反复被同一个概念绊住。

### 2.1 group 是调度的基本单位

Dressage 的调度不是以单个样本为单位，而是以 **group（组）** 为单位。一个 group 对应"同一个 prompt 的 `n_samples_per_prompt` 次采样"——这正是 GRPO 做"组内相对优势"所需要的一组兄弟样本。

所以你会看到两个尺寸反复出现：

- `rollout_batch_size`：一批有多少个 prompt（多少个 group）；
- `n_samples_per_prompt`：每个 prompt 采样几条；
- 二者相乘 = 这一批一共多少条样本。

记住"**调度按 group 走，训练按样本走**"，后面 partial async 的一切精妙都建立在这句话上。

### 2.2 采一条轨迹：`generate_and_rm_group`

三种模式真正"干活"的函数是同一个——slime 提供的 `generate_and_rm_group`。它接收一个 group，跑完 agent 交互、finalize 轨迹、算好 reward，返回带训练数据的样本。三种模式都只是在不同的时机、用不同的并发方式去调用它而已。

### 2.3 失败处理三件套：重试、废批保护、失败摘要

这三样东西定义在 `fully_async_rollout.py` 里，另外两个模式直接 import 复用：

> **源码位置**：重试计数 [`_retry_count` / `_increment_retry`](../dressage/rollout/fully_async_rollout.py#L393-L415) ｜ 可训练 token 判断 [`_group_has_trainable_tokens`](../dressage/rollout/fully_async_rollout.py#L191-L209) ｜ 失败摘要 [`_group_failure_summary`](../dressage/rollout/fully_async_rollout.py#L146-L188) ｜ 多段结果展平 [`_flatten_multi_segment_result`](../dressage/rollout/fully_async_rollout.py#L100-L109)

- **组级重试**：一个 group 失败（抛异常，或样本状态是 `ABORTED`），会在 `DRESSAGE_ROLLOUT_MAX_RETRIES`（默认 2）次内重试。重试前 `_increment_retry` 会清掉旧的 `session_id`/`parent_traj_id`，让它重新获得一个干净的沙箱会话。
- **废批保护**：如果整批里没有任何一个样本有可训练的 token（response 长度 > 0 且 loss_mask 有非零位），就直接抛错拒绝训练——因为拿一批全是失败占位符的样本去更新权重，等于往模型里灌噪声。除非你显式设 `DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH=1`。
- **失败摘要**：`_group_failure_summary` 把状态分布、`blackbox_error`、`session_id` 等信息拼成一行日志，方便排查为什么这批采废了。

### 2.4 多段展平：`_flatten_multi_segment_result`

一条长轨迹可能因为历史压缩、工具 schema 变化等原因被切成多个**段（segment）**，`generate_and_rm_group` 返回的可能是"列表套列表"。三种模式都会先用 `_flatten_multi_segment_result` 把它拍平成一维样本列表再处理。多段训练本身是另一个话题，这里只需知道：**它对三种调度模式是透明的**。

铺完地基，我们从最朴素的同步模式开始。

---

## 3. 同步模式：最朴素，也最稳

> **源码位置**：整体流程 [`_run_sync_rollout`](../dressage/rollout/sync_rollout.py#L70-L149) ｜ slime 入口 [`generate_rollout_sync`](../dressage/rollout/sync_rollout.py#L152-L166) ｜ 提交单组 [`_submit_group`](../dressage/rollout/sync_rollout.py#L49-L67)

### 3.1 它为什么存在：共置 GPU 必须"腾地方"

**结论先行：同步模式是为"共置（colocate）"架构服务的——当 actor（训练）和 SGLang（推理）挤在同一批 GPU 上时，必须等这一批 rollout 彻底跑完，框架才能把 SGLang 引擎卸载、把显存腾给 Megatron 训练。**

想象 qwen3.5-35B-A3B 跑在 8×H100 上，开了 `--colocate`。这时候训练和推理是**分时复用**同一批显卡的：采样阶段显卡跑 SGLang，训练阶段显卡跑 Megatron。二者不可能同时在场，因为显存装不下两套。

那么调度上唯一合理的选择就是：**这一批 rollout 从头跑到尾，一个不剩地交付，然后卸载 SGLang，再训练。** 这就是同步模式，也是 slime 的默认行为。它天生没有异步那些花活，但也天生没有异步的烦恼。

### 3.2 流程走读

同步模式虽然叫"同步"，但内部依然用 `asyncio` 做**并发**——注意"并发"不等于"异步流水线"。它是"一次性把整批提交出去并发地跑，然后同步地等它们全部完成"：

```text
get_samples(target)              # 一次性取够整批 group
    │
    ├─ 对每个 group: _submit_group  # 全部提交为 asyncio.Task
    ▼
while pendings:                   # 只要还有没跑完的
    asyncio.wait(FIRST_COMPLETED) # 谁先完成先处理谁
        ├─ 成功 → data.append
        └─ 失败 → 重试（< max_retries）或标记为不可训练
    ▼
按 index 排序 → 废批保护 → 返回整批
```

几个值得停下来看的细节：

**① 不 oversample。** 代码里 `get_samples(target)` 拿到的组数若少于 `target` 会直接抛错。注释写得很直白：同步模式"一次性提交整批，不做超采样"。这和异步模式形成鲜明对比——异步会多备一些以对冲失败和延迟，同步则要求"要多少给多少，一个不多一个不少"，保证批次组成是**确定的**。

**② 并发但同步返回。** `asyncio.wait(..., FIRST_COMPLETED)` 让整批 group 并发地跑，谁先完成先收谁的结果，但函数直到 `pendings` 清空才返回。也就是说，对外它是一个"阻塞到整批完成"的同步调用，对内它是并发的。这样既拿到了并发采样的速度，又保住了"整批同时交付"的语义。

**③ 失败就地重试。** 某个 group 失败且没超重试上限，就用 `_submit_group` 重新丢回 `pendings` 里继续跑；超了上限就 `_mark_no_grad_failed` 打成不参与梯度的占位样本。整个过程都在这一次调用内闭环，不存在"留到下一步"的概念。

### 3.3 优缺点

- ✅ **简单、确定、好调试**：批次组成固定，没有跨步骤的隐藏状态，出问题容易复现。
- ✅ **共置友好**：跑完即可卸载 SGLang，显存腾挪干净。
- ❌ **训练 GPU 空转**：整个采样阶段训练侧是闲着的。轨迹越长、方差越大，浪费越多。
- 🎯 **适用**：开发、调试、小规模实验，以及共置部署的大模型。

同步模式的痛点很明确——训练在等采样。那能不能让训练**别等**？这就引出了完全异步。

---



## 4. 完全异步模式：让训练不再等 Agent

> **源码位置**：后台 worker [`AsyncRolloutWorker`](../dressage/rollout/fully_async_rollout.py#L268-L369) ｜ 生产者主循环 [`continuous_worker_loop`](../dressage/rollout/fully_async_rollout.py#L302-L335) ｜ 消费者/收集 [`generate_rollout_async`](../dressage/rollout/fully_async_rollout.py#L418-L530) ｜ 全局单例 [`get_global_worker`](../dressage/rollout/fully_async_rollout.py#L376-L382) ｜ slime 入口 [`generate_rollout_fully_async`](../dressage/rollout/fully_async_rollout.py#L533-L548)

### 4.1 它为什么存在：把 Agent 延迟藏到训练背后

**结论先行：完全异步用一个"常驻后台采样 worker"把 rollout 和训练在时间上重叠起来。训练侧每次要数据时，不再触发采样，而是直接从一个已经攒好的队列里"抽干"够用的量。Agent 那几分钟的延迟，被藏到了上一步训练的时间里。**

前提是 **分离（disaggregated）架构**：rollout 有自己的 GPU 池，训练有自己的 GPU 池，两边可以同时在场。既然显存不再打架，那就没有理由让训练干等——让采样在后台一直跑就是了。

### 4.2 架构：一个生产者，一个消费者，一条有界队列

这是一个经典的**生产者-消费者**模型，但拆得很干净：

```text
        ┌─────────────────────── 后台线程（生产者）───────────────────────┐
        │  continuous_worker_loop（独立线程里跑一个 asyncio 事件循环）        │
        │                                                                  │
        │   收割已完成 task ──► output_queue（有界，默认 1000）              │
        │        ▲                    │                                    │
        │        │                    ▼                                    │
        │   prewarm 预热 ──► dispatch 新 group（受 max_active_groups 限流）  │
        └──────────────────────────────┬───────────────────────────────────┘
                                        │ 线程安全的 queue.Queue
        ┌───────────────────────────────▼──────────────────────────────────┐
        │  generate_rollout_async（消费者，被 slime 每步调用一次）            │
        │   反复 get_completed_groups() 抽干队列                             │
        │   攒够 rollout_batch_size 个可用组 → 排序 → 返回训练              │
        └───────────────────────────────────────────────────────────────────┘
```

为什么要用"线程 + 队列"而不是纯 asyncio？因为 slime 的训练循环是**同步**调用 rollout 函数的，而后台采样需要一个不受训练调用节奏影响、自己一直转的事件循环。于是 Dressage 把后台 worker 塞进一个 daemon 线程，线程里 `asyncio.run(continuous_worker_loop())` 自转，再用线程安全的 `queue.Queue` 把成品递给同步的消费者。`get_global_worker` 保证整个进程只有一个这样的 worker 单例。

### 4.3 生产者：`continuous_worker_loop` 的三个阶段

后台循环每一轮做三件事（对应源码里的三段）：

1. **收割（drain done tasks）**：把已经跑完的 `asyncio.Task` 结果打包成 `CompletedGroup` 塞进 `output_queue`，并立刻释放这个 group 没用上的预热沙箱。
2. **预热（prefetch）**：调用调度器给"接下来要跑的 group"提前拉起沙箱，把冷启动开销藏到当前 group 还在跑的时候。
3. **派发（dispatch）**：只要还没到限流上限，就从数据缓冲区取新 group、`create_task` 丢进去跑。

限流是这个循环的灵魂，由两个闸门共同把关：

```python
while (self.running
       and len(active) < self.max_active_groups          # 在跑的 group 数上限
       and self.output_queue.qsize() < self.high_watermark):  # 成品堆积高水位线
```

- `max_active_groups`（默认 = `rollout_batch_size`）：同时在跑的 group 不能太多，否则会把成百上千个沙箱会话一起拉起来打爆资源；
- `high_watermark`（队列容量的 80%）：如果成品已经堆到高水位，说明消费得慢，就先别采了，等训练把库存消化一些——这是一个天然的**背压（backpressure）**机制。

### 4.4 消费者：`generate_rollout_async` 只负责"收够就走"

消费者这一侧反而简单：循环 `get_completed_groups()` 把队列抽干，对每个完成的组做判定——

- **失败组**：够重试次数就打回数据缓冲区重来（`data_buffer.add_samples`），超了就丢弃并计数；丢太多（超过 `DRESSAGE_ASYNC_MAX_DROPPED_FAILED_GROUPS`）直接抛错，避免为了凑一个能训练的批次无限等下去。
- **成功组**：过一遍 staleness 过滤（下一节讲），留下来的进 `data`。
- 一旦 `len(data)` 攒够 `rollout_batch_size`，排序后返回。

注意这里生产者和消费者用的 `group_id` 是**各自独立**的两套编号，中间隔着队列，靠 `CompletedGroup` 这个数据结构把"原始组"和"结果"一起打包传递。

### 4.5 关机语义：不做无用功

异步 worker 是常驻的，什么时候停？看 `_should_drain_worker_on_rollout`——当跑到**最后一个 rollout**（`rollout_id + 1 >= num_rollout`）时，就停掉 worker。停的时候有个很务实的决定（见 `continuous_worker_loop` 的 `finally`）：

> 直接 `cancel` 掉所有还在跑的 task，而不是等它们跑完。

理由很直白：这些结果没人要了，继续等只是白烧算力。但取消 task 之后，每个 task 的 `finally` 会去释放它占用的沙箱，`drain_terminate_tasks()` 会等这些后台清理任务收尾——**算力可以不要，资源必须回收**。

### 4.6 优缺点

- ✅ **吞吐最大化**：训练几乎不空等，Agent 延迟被完全重叠掉。
- ✅ **对长尾方差友好**：慢轨迹在后台慢慢跑，不阻塞快轨迹交付。
- ❌ **复杂**：多了线程、队列、限流、关机语义，调试成本上升。
- ❌ **引入新鲜度问题**：后台一直在用"当时的权重"采样，等这些轨迹被消费时，权重可能已经更新过了——这批数据就"过时"了。
- 🎯 **适用**：黑盒长轨迹的生产训练，是黑盒运行的默认选择。

这个"❌ 新鲜度问题"是异步的原罪，必须专门处理。

---

## 5. Staleness：异步换来的吞吐，代价是"新鲜度"

> **源码位置**：版本追踪 [`StalenessTracker`](../dressage/rollout/staleness.py#L87-L132) ｜ 组过滤器 [`StalenessGroupFilter`](../dressage/rollout/staleness.py#L135-L215) ｜ 配置解析 [`config_from_args`](../dressage/rollout/staleness.py#L32-L41) ｜ 轨迹版本提取 [`trajectory_version_infos`](../dressage/rollout/staleness.py#L65-L84)

### 5.1 问题：off-policy 到什么程度可以接受

**结论先行：异步下一条轨迹可能是用好几个版本前的旧权重采的。用太旧的数据更新当前策略，等于用错误方向的梯度拖后腿。Staleness 机制的作用，就是设一个"最多容忍几个版本前的数据"的窗口，超窗的组直接丢。**

RL 是 on-policy 敏感的：理想情况下，采样用的策略要和被更新的策略尽量一致。异步打破了这一点——为了吞吐，我们容忍了一定程度的 off-policy。但容忍要有度，这个度就是 `dressage_staleness_keep_versions`。

### 5.2 机制：以"权重版本"为时间轴

`StalenessTracker` 维护一个**版本列表**，按观察到的先后顺序记录每个出现过的权重版本标签。于是：

- `current_version_index`：目前见过的最新版本的下标；
- `cutoff_version_index = max(0, len(versions) - keep_versions)`：容忍窗口的下界。

判定一个组是否该丢，就看它里面任一轨迹的版本下标是否 `< cutoff`。是就丢。`keep_versions=0`（或没配）等于关闭这个功能，容忍任意陈旧。

### 5.3 一个关键设计：按"轨迹"而非"样本"取版本

`trajectory_version_infos` 做了件很讲究的事：一条轨迹可能有多段（segment），每段可能横跨不同权重版本，它取的是**每条轨迹（按 `parent_traj_id` 分组）里段序号最大的那一段的版本**——也就是这条轨迹"最后落笔"时的权重版本。

为什么？因为一条轨迹的新鲜度应该由它**最新的那部分**代表，而不是它最早的那部分。这个细节保证了跨版本的长轨迹被公平对待。

### 5.4 消费时的两处过滤

`StalenessGroupFilter` 在消费者循环里出现在两个地方：

1. **`observe_completed`**：每收到一批完成组，先"登记"它们带来的新版本——异步下新版本正是通过完成组的元数据被发现的。
2. **`keep_group` / `filter_pending`**：既过滤新来的组，也回头**重新过滤已经攒在 `data` 里的组**——因为版本可能刚刚推进，之前还合格的组现在可能超窗了。

它还顺带产出 `staleness/dropped_groups`、`staleness/version_gap_*` 等监控指标，让你能看见"到底丢了多少、平均落后几个版本"。

理解了 staleness，我们才能真正讲清楚 partial async——因为它把新鲜度这件事做到了极致。

---

## 6. 部分异步模式：只取够训练的那一份

> **源码位置**：目标组数计算 [`_partial_target_groups`](../dressage/rollout/partial_async_rollout.py#L165-L202) ｜ 后台 worker [`PartialAsyncRolloutWorker`](../dressage/rollout/partial_async_rollout.py#L205-L352) ｜ 收集主逻辑 [`generate_rollout_partial_async_impl`](../dressage/rollout/partial_async_rollout.py#L394-L582) ｜ 剩余回填 [`return_completed_groups`](../dressage/rollout/partial_async_rollout.py#L344-L349) ｜ 分组标记 [`_annotate_submitted_group` / `_annotate_returned_group`](../dressage/rollout/partial_async_rollout.py#L149-L162)

### 6.1 它为什么存在：采样量 ≫ 单步训练量

**结论先行：部分异步解决的是一个很具体的错配——一次 rollout 采出来的样本数（`rollout_batch_size × n_samples_per_prompt`），往往远大于一步训练真正需要的样本数（`global_batch_size`）。既然如此，何必等全部采完？攒够一步训练用的量就先返回，剩下的留在后台继续跑，喂给下一步。**

举文件头注释里的例子：`rollout_batch_size=16`、`n_samples_per_prompt=8`，一次采 128 条。但如果 `global_batch_size=64`，一步训练只吃 64 条。完全异步会傻等 16 个组全齐；部分异步则在第 8 个组齐了的时候就返回，剩下 8 个组不作废，而是**滚动到下一步的库存里**。

这带来两个好处：一是**每步的等待更短**（只等一半的量），二是**没有浪费**——多采的那部分不是丢掉，而是提前为下一步备好了货，进一步把流水线填满。

### 6.2 核心：`_partial_target_groups` 到底返回几个组

这是 partial 相对 fully 最本质的差别——**目标数不再是 `rollout_batch_size`，而是"够一步训练"的组数**。计算按优先级层层回退：

1. `DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS`：直接指定组数，最高优先级；
2. `DRESSAGE_PARTIAL_ROLLOUT_TARGET_SAMPLES`：指定样本数，再换算成组数；
3. 都没配，就看 `global_batch_size`：如果它小于全量采样数，就用它当目标（这是最常见的自动路径）；
4. 兜底用 `rollout_batch_size`。

换算时用 `ceil(target_samples / n_samples_per_prompt)`，并且结果被 `min(rollout_batch_size, ...)` 夹住——**目标永远不会超过一次全量**。样本数不能被整除时会打日志提醒你实际返回的是向上取整后的组数。

### 6.3 早退之后：剩下的组怎么办

这是 partial async 里最容易被忽略、但设计最巧的一段。消费循环 `while len(data) < target_groups` 攒够就跳出，但此时队列里（`completed_by_id`）很可能还躺着一些**已经完成、但这一步用不上**的组。怎么处理？

```python
if completed_by_id:
    leftovers = list(completed_by_id.values())
    if drain_final_worker:
        # 最后一步：没有下一步了，丢弃
        ...
    else:
        worker.return_completed_groups(leftovers)   # 关键：塞回队列
```

`return_completed_groups` 把这些"多抽出来的成品"重新放回 `output_queue`，让**下一次** rollout 调用能直接消费它们，而不是白白丢掉已经烧过算力的黑盒轨迹。这就是"没有浪费"的落地实现。

### 6.4 分组标记：给每条样本盖上时间戳

partial async 特有的两个标注函数会给样本 metadata 盖章：

- `_annotate_submitted_group`：提交时记下 `dressage_start_rollout_id`（这条轨迹是在哪一步开始采的）、组号，并打上 `dressage_partial_rollout=True`；
- `_annotate_returned_group`：返回时记下 `dressage_return_rollout_id`（在哪一步被消费）。

有了"开始步"和"返回步"这两个戳，配合上一节的 staleness 版本追踪，训练侧就能精确知道每条样本跨了多远、是否需要对旧权重生成的 token 做掩码。partial async 也确实比 fully async 多产出一整套 `dressage/partial_rollout_*` 与 `staleness/partial_rollout_*` 指标。

### 6.5 收尾：最后一步要"排干"

`_should_drain_worker_on_rollout` 判断是不是最后一步。是的话，除了丢弃 leftover，还会 `stop_global_partial_worker()` 把 worker 停掉并**排干**队列里剩余的完成组，统计进 `drained_completed_groups`。非最后一步则让 worker 继续常驻——这正是 partial 的名字来源：**部分**返回，部分留驻。

### 6.6 和完全异步的差异，浓缩成一张表

| 维度 | 完全异步 fully async | 部分异步 partial async |
|---|---|---|
| 返回目标 | `rollout_batch_size`（全量组） | `_partial_target_groups`（够一步训练即可） |
| 多余的完成组 | 不存在（要么全收，要么在跑） | `return_completed_groups` 回填给下一步 |
| 样本标注 | 无 | start/return rollout_id + `partial_rollout` 戳 |
| 与权重更新的关系 | 靠 staleness 兜底 | **必须**配 pause/resume 保护在途生成 |
| 定位 | 吞吐优先 | 吞吐 + 新鲜度 + 连续训练 |

表格最后一行点出了 partial async 的最后一块拼图：为什么它**必须**配 pause/resume。

---



## 7. 最后一块拼图：pause/resume 与 token 级续跑

> **源码位置**：可抢占生成 [`generate_preemptible`](../dressage/proxy/generation_controller.py#L159-L452) ｜ 暂停/中止 [`pause`](../dressage/proxy/generation_controller.py#L454-L560) ｜ 恢复/推进 epoch [`resume`](../dressage/proxy/generation_controller.py#L562-L607) ｜ 训练侧包装 [`_safe_update_weights`](../dressage/training/train_async_with_rollout_pause.py#L95-L113) ｜ 训练主循环 [`train`](../dressage/training/train_async_with_rollout_pause.py#L116-L187)

### 7.1 问题：权重更新会"劈开"一条正在生成的轨迹

**结论先行：异步下后台一直在生成 token，而训练每隔几步就要更新一次权重。如果一条轨迹正生成到一半时权重被换掉，这条轨迹就横跨了两个权重版本——要么把它整条作废（浪费），要么想办法让它"接着旧的往下写、只是换了新权重"。partial rollout（部分轨迹续跑）选的是后者。**

注意这里的 "partial rollout" 和上一节的 "partial async rollout" 不是同一个东西，很容易混淆：

- **partial async rollout（第 6 节）**：是**组级**的调度策略——只返回够一步训练的组；
- **partial rollout（本节）**：是**token 级**的续跑能力——一条轨迹被权重更新打断后，从断点接着生成。

partial async 之所以**必须**配 partial rollout，是因为它的后台 worker 是常驻的，权重更新时一定有轨迹正在途中。没有 token 级续跑，这些在途轨迹就只能作废或被硬生生撕成脏数据。

### 7.2 生成侧：`GenerationController` 如何做到"可抢占"

Dressage 的 proxy 是所有 LLM 请求的唯一入口。它没有直接把请求透传给 SGLang，而是包了一层 `GenerationController`，让每一次生成都是**可抢占（preemptible）**的。核心是 `generate_preemptible` 里的一个 while 循环：

```text
while 还没生成够 max_new_tokens:
    await resume_event.wait()        # 若已暂停，卡在这里等恢复
    发一段 /generate 给 SGLang（带 request_id）
    if 被 abort 打断（preempted）:
        把已生成的部分 token 收进 chunk
        if 不支持 partial_rollout: 直接抛错
        else: 等 resume_event，然后带着 已有前缀 继续下一轮循环
    else:
        break                        # 正常生成完
```

关键在于：**abort 只是一个信号**。当 `pause` 去 abort 某个 request 时，SGLang 会结束当前这段 `/generate`、把已经生成的部分 token 通过原请求返回；controller 把这段 token 拼进 `generated_ids`，然后**挂起**（在 `resume_event` 上等待）。等 resume 到来，下一轮循环用 `input_ids + 已生成的前缀` 重新发起生成——对上层的黑盒 agent 来说，它那一次 HTTP 调用只是**响应慢了一点**，完全感知不到中间被劈开又缝合。

### 7.3 暂停侧：`pause` 精确地在 token 边界停下

`pause` 做的事：置暂停标志、清 `resume_event`（让所有生成循环卡住）、对当前所有 active 的 SGLang 请求按 `request_id` 逐个 abort，然后**等它们都进入 quiesced（静默）状态**才返回。

这里有个非常讲究的时序（源码注释专门强调）：controller 只有在"被抢占的那段 token 已经被追加进 `generated_ids` 和 chunks 之后"，才把该请求标记为 quiesced。这保证了 `pause` 返回时，"resume 将从哪里继续"的那个前缀是**已经落袋、可恢复**的。换句话说，pause 返回 = 模型侧真正安静了 + 断点已保存。

### 7.4 恢复侧：`resume` 顺手推进"版本时钟"

`resume` 先确认 SGLang 后端已经加载完新权重（`wait_until_ready`），再把 `resume_event` 放开让生成继续。它还做了一件和第 5 节呼应的事：

```python
if was_paused:
    self._rollout_epoch += 1     # 版本时钟 +1
```

每次真正的 pause→resume 都让 `rollout_epoch` 递增。这个 epoch 就是 staleness 机制的"版本时钟"来源——权重换了一版，时钟就走一格，后面采出来的轨迹自然带上新版本号，陈旧的轨迹据此被过滤。至此，第 5 节的 staleness 和本节的 pause/resume 严丝合缝地咬在一起。

### 7.5 训练侧：`train_async_with_rollout_pause` 把 pause/resume 夹在权重更新两侧

这是那个把一切串起来的训练入口。它是 slime 异步训练循环的镜像，唯一的加料就是在**每次权重更新前后**插入 proxy 的 pause/resume（见 `_safe_update_weights`）：

```text
weight_update 前  →  POST /rollout/pause   →  GenerationController 在 token 边界中止在途生成
                  →  actor_model.update_weights()（Megatron 更新 + SGLang 加载新权重）
weight_update 后  →  POST /rollout/resume  →  epoch+1，生成从断点带新权重续跑
```

它开头有一句斩钉截铁的断言：`assert not args.colocate`——**这个入口只服务于分离架构**，共置场景请走同步模式。这也再次印证了全文的主线：pause/resume 是异步（尤其 partial async）专属的机制，同步模式压根不需要，因为同步整批同权重、没有"在途轨迹被劈开"这回事。

还有一个务实细节：主循环里只在"**还有下一步 rollout 会消费这批新权重**"时才真正 pause+更新；如果是最后一步之后，更新了也没人用，就跳过（日志会说明"skipping actor weight update after final rollout"）。这和第 4、6 节"最后一步不做无用功"的思路一脉相承。

---

## 8. 怎么选：一张决策图收尾

**结论先行：先看你的 GPU 是共置还是分离——共置只能用 sync；分离再看采样量和单步训练量是否错配——错配且要连续训练就用 partial async，否则用 fully async。**

```text
                    ┌─────────────────────────────┐
                    │ actor 和 SGLang 共享 GPU 吗？ │
                    └──────────────┬──────────────┘
                   是（colocate）  │  否（disaggregated）
              ┌──────────────────┘   └───────────────────┐
              ▼                                           ▼
      ┌───────────────┐                   ┌─────────────────────────────┐
      │  同步 sync     │                   │ rollout 采样量 ≫ 单步训练量？ │
      │ 整批跑完再卸载  │                   │ (rbs×n ≫ global_batch_size) │
      │ SGLang 后训练  │                   └──────────────┬──────────────┘
      └───────────────┘                        是         │        否
                                        ┌─────────────────┘        └────────────┐
                                        ▼                                        ▼
                            ┌───────────────────────────┐          ┌─────────────────────┐
                            │  部分异步 partial async     │          │  完全异步 fully async │
                            │ 只取够一步的组 + 剩余回填   │          │ 攒够全量组再返回      │
                            │ 必须配 pause/resume 续跑    │          │ 靠 staleness 兜底新鲜度│
                            └───────────────────────────┘          └─────────────────────┘
```

配套要记住的开关：

- 异步两种模式的新鲜度：`dressage_staleness_keep_versions=N` 决定容忍几个版本前的数据；
- partial async 的返回量：不配则自动取 `min(rollout_batch_size, ceil(global_batch_size / n_samples_per_prompt))`，也可用 `DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS` 直接指定；
- pause/resume：由 `train_async_with_rollout_pause` 入口自动接管，`DRESSAGE_PROXY_PAUSE_AROUND_WEIGHT_UPDATE` 控制开关（默认开）。

---

## 9. 结语

回头看这三种模式，它们不是三个孤立的实现，而是同一条演化线上的三个刻度：

1. **同步**：不重叠、不分开，用"整批跑完再训练"换来了确定性和共置友好，代价是训练 GPU 空转；
2. **完全异步**：分开 GPU、重叠时间，用一个"生产者-消费者 + 有界队列 + 背压"的后台 worker 把 Agent 延迟藏进训练时间里，代价是引入了新鲜度（staleness）问题；
3. **部分异步**：在完全异步的骨架上，用"只取够一步训练的量 + 剩余回填"解决采样量错配，再用 token 级 partial rollout（pause/resume）保住权重更新时的在途轨迹，把新鲜度和吞吐同时做到位。

而把它们缝合在一起的，是三样贯穿全文的机制：**共享的失败处理与废批保护**（保证任何模式都不会拿脏数据训练）、**staleness 版本时钟**（量化并约束 off-policy 程度）、**pause/resume 抢占续跑**（让权重更新与连续采样和平共处）。

理解了"为什么"，代码里那些 `high_watermark`、`return_completed_groups`、`rollout_epoch += 1` 就不再是零散的技巧，而是同一个目标——**在不牺牲训练正确性的前提下，尽可能不让昂贵的 GPU 闲着**——的必然结果。

> 想继续往下读：轨迹如何切段与转样本，可看 [Rollout Hooks & Async Modes](../docs/rollout.md) 与 [Training Layer](../docs/training.md)；整体架构与概念，可看 [Agentic RL 训练完全指南](agentic-rl-training-zh.md)。
