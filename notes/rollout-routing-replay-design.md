# Rollout Routing Replay (R3) 设计文档

> 对应代码：`dressage/proxy/sglang_client.py`、`dressage/proxy/generation_controller.py`、`dressage/proxy/server.py`、`dressage/rollout/artifacts/samples.py`、`slime/slime/utils/routing_replay.py`、`slime/slime/backends/megatron_utils/actor.py`
>
> 文档结构：总—分—总。第一章总览建立全局认知；第二至六章分层展开（问题 → 方案 → 原理 → 示例 → 边界）；第七章收束为核心结论。每章内部同样"章首结论先行 → 逐节展开 → 本章小结"。

## 一、总览

**本章结论先行：R3（Rollout Routing Replay，路由回放）解决的是 MoE 模型 RL 训练中的"训练-推理路由不一致"问题——rollout 推理时记录每个 token 的专家路由决策，训练前向时回放这些决策而不是重新计算。它把训练-推理 logprob 绝对差异从 0.0195 降到 0.0077（降幅约 60%），额外开销接近零。**

### 1.1 三分钟版

- **是什么**：一种消除 MoE 模型"推理和训练两次前向选中不同专家"现象的技术。
- **解决什么问题**：RL 训练要求 rollout 采样的 logprob 与训练重算的 logprob 逐 token 对应；MoE 的 router 是离散选择，两个框架（SGLang vs Megatron）的数值差异会让大量 token 走不同的专家路径，logprob 产生系统性噪声。
- **为什么重要**：在 Qwen3.5-35B-A3B 上实测约 85% 的 token 至少在一层存在路由差异——不开 R3，绝大多数训练样本的 logprob 都带噪声，严重时训练不收敛。
- **怎么做**：把"训练时重算路由"换成"回放 rollout 时记录的路由"——`torch.topk`（重选）变 `scores.gather`（按预存索引取值）。
- **代价**：接近零。不增加额外计算 pass，只替换路由索引的来源；路由数据 offload 到 CPU 锁页内存。

### 1.2 全文地图

| 章节 | 回答的问题 |
|------|------------|
| 二、背景与问题 | 路由不一致为什么会发生、有多严重、不处理会怎样 |
| 三、核心方案 | R3 的整体思路、为什么有效、为什么便宜 |
| 四、原理拆解 | 数据怎么从 SGLang 流到 Megatron 的 router（四层链路） |
| 五、示例说明 | 一个 token / 一条跨版本轨迹 / 一个训练 step 的具体演算 |
| 六、应用与边界 | 怎么开启、适用边界、常见误区、代码索引 |
| 七、总结 | 本质价值与最需要记住的一点 |

## 二、背景与问题

**本章结论先行：MoE 的 router 是对分数做 top-k 的离散选择，而离散选择对数值扰动天然敏感；推理（SGLang）和训练（Megatron）两个框架的精度与 kernel 差异足以翻转 top-k 的成员构成，实测约 85% 的 token 受影响。RL 训练又恰恰要求"两次前向结果一致"（重要性采样比率逐 token 对应），于是路由不一致直接变成梯度噪声。**

### 2.1 前置知识：MoE 路由是什么

Mixture-of-Experts（MoE）模型中，每个 token 经过每个 MoE 层时，由 router 网络对该 token 的 hidden state 打出一组 expert 分数，然后选 **top-k** 个 expert 处理（如 Qwen3.5-35B-A3B：256 个 expert 选 8 个，共 40 层 MoE）。

关键在 top-k 这一步：**它是 argmax 式的离散选择**。分数是连续的，但"谁进前 8"是离散的——两个 expert 分数只差小数点后几位时，任何数值扰动都可能翻转它们的相对顺序，进而改变入选集合。

### 2.2 为什么不一致：两个框架，两次前向

RL 训练中同一条 token 序列要经历两次前向：

- **推理阶段（SGLang）**：rollout 采样生成 token，用推理优化的 kernel，FP8/BF16 精度。
- **训练阶段（Megatron）**：对这些 token 重新前向计算 logprob（算重要性采样比率和 loss），用训练 kernel，BF16 精度。

两个框架的 router 实现、精度、浮点求和顺序都不同。浮点运算不满足结合律（$(a+b)+c \neq a+(b+c)$ 在浮点下成立是奢望），expert 分数在末位比特上必然有差异。对稠密模型这只是 logprob 的小扰动；对 MoE，它通过 top-k 离散选择被**放大成路径差异**——同一个 token 在两次前向中走了不同的 expert。

### 2.3 量化影响：85% 的 token 受影响

在 Qwen3.5-35B-A3B（40 层 MoE，256 experts，top-8）上的实测数据：

| 路由不一致指标 | 均值 | 含义 |
| --- | --- | --- |
| `element_mismatch_rate` | 45.25% | 每个 (token, layer, slot) 三元组中，约 45% 的 expert 选择不同 |
| `token_mismatch_rate` | 85.16% | 约 85% 的 token 至少在某一层存在路由差异 |
| `set_token_mismatch_rate` | 59.64% | 即使忽略 top-k 内的排列顺序，仍有约 60% 的 token 选择了不同的 expert 集合 |

注意这不是 bug，而是浮点运算的固有特性——任何"两个不同框架/精度各做一次前向"的 MoE 系统都会遇到。

### 2.4 对 RL 训练的危害

策略梯度 RL 的 loss 逐 token 计算重要性采样比率（见 [llm-rl-algorithms-zh.md](llm-rl-algorithms-zh.md) 8.1 节）：

$$r_t(\theta) = \frac{\pi_\theta(y_t \mid y_{<t})}{\pi_{\theta_{old}}(y_t \mid y_{<t})}$$

分子来自训练前向，分母来自推理采样。路由不一致意味着**分子分母是用不同的专家路径算出来的**：

1. **logprob 偏差**：训练算出的 logprob 与推理时实际采样的 logprob 系统性不一致。
2. **梯度信号失真**：$r_t$ 的比值偏离 1 不再反映"策略更新了多少"，而混入"路由摇了多少"的噪声。
3. **训练不稳定**：clip 被噪声触发（clipfrac 异常）、梯度方差增大、收敛变慢甚至发散。

实验测量：**不开 R3 时训练-推理 logprob 绝对差异均值为 0.0195**——对逐 token 比率来说是不可忽视的偏差源。

### 2.5 替代方案为什么不够

| 替代方案 | 思路 | 为什么不够 |
| --- | --- | --- |
| 统一精度/框架 | 让两次前向数值完全一致 | 工程上不可行：推理引擎为吞吐优化（FP8、融合 kernel），训练引擎为精度优化，目标天然冲突 |
| GSPO 序列级比率 | 换粒度，用整序列比率抹平单 token 波动 | 缓解症状不消除病因：路由噪声仍然存在，只是被平均了（两者可叠加，见 6.3） |
| 增大 clip 容忍度 | 放宽比率裁剪区间 | 把真信号也一并容忍了，收敛变慢 |

### 2.6 本章小结

```text
top-k 是离散选择 → 数值扰动被放大为路径差异（85% token 受影响）；
RL 的 IS 比率要求两次前向逐 token 对应 → 路径差异直接变成梯度噪声；
统一精度不可行、换粒度只是缓解 → 需要一条"消除噪声源"的路。
```

## 三、核心方案

**本章结论先行：R3 的思路是"回放而非重算"——rollout 推理时把每个 token 在每层选中的 expert IDs 记下来、随轨迹存好，训练前向时跳过 router 的 top-k 选择、直接注入这些预存索引。冻结的是离散选择，连续分数仍由当前权重实时计算；因此一致性问题被消除，而训练语义不变、开销接近零。**

### 3.1 核心思想三步

1. **推理时捕获**：SGLang 生成每个 token 时，记录其在每个 MoE 层被路由到的 expert IDs（`return_routed_experts`）。
2. **随轨迹存储**：路由决策以 base64 编码的 int32 数组形式，随 trajectory segment 的元数据一起流转。
3. **训练时 replay**：Megatron 前向传播时，`compute_topk` 被替换为"弹出预存索引 + gather 取分"，每个 token 走与推理时完全相同的 expert 路径。

### 3.2 关键洞察：冻结离散选择，不冻结连续分数

这是理解 R3 最重要的一点。MoE 路由的输出有两部分：

- **离散部分**：选哪 top-k 个 expert（indices）——不一致的根源；
- **连续部分**：这些 expert 的 gating 分数（probs）——由 router 对当前 hidden state 打分得到。

R3 只回放前者：

```text
不用 R3：probs, indices = torch.topk(scores, k=8)     # 重算分数 + 重选专家
用 R3：  indices = 预存的 rollout 索引                 # 回放选择
        probs   = scores.gather(1, indices)          # 分数仍由当前权重算出
```

`scores` 来自训练侧的实时前向——所以 gating 分数反映当前策略、梯度可以正常流过 router，**模型行为没有被冻结**；被冻结的只有"选谁"这个离散决策，保证训练前向与推理前向走同一条 expert 路径。

### 3.3 为什么开销接近零

Routing replay 不增加额外的计算 pass，而是在已有前向传播中**替换**路由索引的来源：`gather` 比 `topk` 更轻（取值 vs 排序选择），计算量只减不增。存储侧，路由索引 offload 到 CPU 锁页内存，只在对应层前向时拷回 GPU，不占显存。调试用的对比前向（`fallthrough`/`record` 双跑）可以随时关闭。

### 3.4 效果

开启 R3 后，训练-推理 logprob 绝对差异从 **0.0195 降至 0.0077，降幅 60.5%**，且在整个训练过程中保持高度稳定。残余差异来自 attention/dense 部分的 kernel 数值差——路由这一噪声源被完整消除（详细实验数据见 [Dressage支持Rollout-Routing-Replay.md](Dressage支持Rollout-Routing-Replay.md)）。

### 3.5 本章小结

```text
不一致的根源是离散选择，不是连续分数；
所以只回放 indices（gather），不重算 scores；
于是两次前向走同一条 expert 路径，而 router 照常训练、开销不变。
```

## 四、原理拆解：四层数据链

**本章结论先行：R3 的实现是一条横跨四层的数据链——SGLang 捕获（推理时记录）→ Proxy 组装（chunk 级存储）→ Artifacts 提取（解码、切片、拼接）→ Megatron 回放（monkey-patch 替换 compute_topk）。每层只做一件事，层间用 base64 int32 数组 + token 偏移量作为契约。**

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        SGLang 推理引擎                               │
│  /generate (return_routed_experts=True)                             │
│  → 每个 output token 返回 routed_experts: [layer_0_experts,          │
│    layer_1_experts, ..., layer_N_experts]  (int32 数组)              │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ SGLangResponse.routed_experts (base64 str)
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Dressage Proxy                                   │
│  GenerationController.generate_preemptible()                        │
│  → 收集每个 chunk 的 routed_experts                                  │
│  → 编码为 base64 int32 payload                                      │
│  → 存入 trajectory segment 元数据                                    │
│                                                                     │
│  三种存储格式：                                                      │
│  · routed_experts        → 单次连续生成                              │
│  · routed_experts_chunks → partial/resumed 生成（多 chunk 拼接）     │
│  · routed_experts_parts  → TITO 多步 segment（每步可含 chunks）      │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ segment["routed_experts"] / ["routed_experts_chunks"] / ["routed_experts_parts"]
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Rollout Artifacts 层                                │
│  extract_routed_experts(segment, args)                               │
│  → base64 解码 → numpy int32 数组                                    │
│  → reshape 为 (num_tokens-1, num_layers, moe_router_topk)           │
│  → 写入 sample.rollout_routed_experts                                │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ Sample.rollout_routed_experts: torch.Tensor
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Slime Megatron 训练侧                                   │
│  MegatronTrainRayActor.fill_routing_replay()                        │
│  → 按层拆分 routed_experts                                           │
│  → RoutingReplay.record(layer_experts)  # offload 到 CPU pinned mem  │
│                                                                     │
│  训练前向传播（Megatron MoE router patch）：                          │
│  → compute_topk 被 monkey-patch 替换                                 │
│  → ROUTING_REPLAY_STAGE="replay_forward" 时 pop 预存 indices         │
│  → scores.gather(1, replayed_indices) 替代 topk(scores)             │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.1 推理侧：SGLang 路由捕获

- **输入**：Proxy 发来的 `/generate` 请求（带 `return_routed_experts: true`）。
- **处理**：SGLang 在生成过程中记录每个 token 在每个 MoE 层的 top-k expert IDs。
- **输出**：`meta_info.routed_experts`——base64 编码的 int32 数组，形状 `(num_tokens, num_layers, moe_router_topk)`。

R3 的起点是 SGLang 的 `return_routed_experts` 参数。当 Dressage Proxy 以 `--use-rollout-routing-replay` 启动时，`SGLangRouterClient`（[sglang_client.py](../dressage/proxy/sglang_client.py)）会在每次 `/generate` 请求中附带该参数：

```python
class SGLangRouterClient:
    def __init__(self, router_url, *, return_routed_experts=False):
        self._return_routed_experts = return_routed_experts

    async def generate(self, input_ids, sampling_params, *,
                       return_routed_experts=False, ...):
        payload = {...}
        if return_routed_experts or self._return_routed_experts:
            payload["return_routed_experts"] = True
```

`_coerce_response` 把 `meta_info` 中的 `routed_experts` 提取到 `SGLangResponse.routed_experts`（仅当它是 base64 字符串时保留，否则置 None）。

### 4.2 代理侧：Proxy 存储与 chunk 管理

- **输入**：每个 SGLang chunk 的 `routed_experts` payload。
- **处理**：`GenerationController.generate_preemptible()`（[generation_controller.py](../dressage/proxy/generation_controller.py) 第 353-361 行）按 chunk 收集，并按生成形态选择存储格式。
- **输出**：写入 segment 元数据的三种格式之一（最终落在 `TrajectorySegment` 的同名字段）。

**为什么要按 chunk 收集**：Partial Rollout 下生成可能被权重更新抢占，一条 response 分多次生成；每次生成是独立的前向，路由数据也必须分 chunk 记录，事后才能精确对齐。每个 chunk 的落库结构（[generation_controller.py](../dressage/proxy/generation_controller.py) 第 353-361 行）：

```python
routed_experts_chunks.append({
    "data": routed_experts,                     # base64 payload
    "prefix_token_count": request_token_count,  # 本 chunk 之前的 token 数
    "output_token_count": len(chunk.output_ids),
    "is_first_chunk": request_token_count == len(input_ids),
})
```

**为什么需要三种存储格式**——按生成的复杂程度递增：

| 格式 | 产生条件 | 存储结构 | 产生代码 |
| --- | --- | --- | --- |
| `routed_experts` | 只有一次生成且从序列开头开始（`len(chunks)==1 and is_first_chunk`） | 一个 base64 字符串 | [generation_controller.py](../dressage/proxy/generation_controller.py) 第 425-432 行 |
| `routed_experts_chunks` | 一条 response 被抢占多次（多 chunk） | chunk dict 列表，各带 `prefix_token_count`/`output_token_count`/`is_first_chunk` | 同上，第 353-361 行 |
| `routed_experts_parts` | TITO 多步 segment（一段由多个 step 拼成，每步还可能多 chunk） | part 列表；每 part 带 `prefix_token_count`（段内累积偏移）+ `concat_token_count`，内嵌该 step 的 `data` 或 `chunks` | [server.py](../dressage/proxy/server.py) 第 1105-1121 行 |

格式一其实是格式二的存储优化——单 chunk 且为首 chunk 时省去列表包装，直接存字符串（[generation_controller.py](../dressage/proxy/generation_controller.py) 第 425-432 行）：

```python
single_routed_experts = (
    routed_experts_chunks[0]["data"]
    if (len(routed_experts_chunks) == 1
        and routed_experts_chunks[0].get("is_first_chunk"))
    else None
)
```

格式三的组装代码（TITO 多步 segment，[server.py](../dressage/proxy/server.py) 第 1105-1121 行）：

```python
routed_experts_parts = []
accumulated_prefix_len = 0
for step_index, step in enumerate(steps):
    if step.response_routed_experts_chunks or step.response_routed_experts is not None:
        part = {
            "prefix_token_count": accumulated_prefix_len,   # 段内累积偏移
            "concat_token_count": len(step.concat_token_ids),
            "is_first_step": step_index == 0,
        }
        if step.response_routed_experts_chunks:
            part["chunks"] = step.response_routed_experts_chunks
        if step.response_routed_experts is not None:
            part["data"] = step.response_routed_experts
        routed_experts_parts.append(part)
    accumulated_prefix_len += len(step.concat_token_ids)
```

**与存储的关系**：三种格式不是三套存储系统，而是 `TrajectorySegment` 上三个平级字段（[trajectory_store.py](../dressage/proxy/trajectory_store.py) 第 30-32 行）——生成侧按形态选其一写入，消费侧按固定优先级读回。格式存在的唯一理由是"用扁平字段表达三种嵌套深度不同的生成历史"，读取时统一还原为一条连续路由数组：

| 生成形态 | 写入字段 | 消费路径 |
| --- | --- | --- |
| 单 chunk 连续生成 | `routed_experts`（省一层包装） | `decode()` 直接解码 |
| 多 chunk（partial rollout） | `routed_experts_chunks` | 逐 chunk `decode` → `slice_generated` → `concatenate` |
| 多 step（TITO segment） | `routed_experts_parts` | 逐 part（内层可能再走 chunks 路径）→ 切片 → 拼接 |

消费侧的优先级实现（`extract_routed_experts()`，[samples.py](../dressage/rollout/artifacts/samples.py) 第 431-452 行）：

```python
chunks_info = segment.get("routed_experts_chunks")
if chunks_info:
    return check_min_length(combine_chunks(chunks_info))   # 优先级 1：多 chunk

raw = segment.get("routed_experts")
if raw is not None and isinstance(raw, str):
    return check_min_length(decode(raw))                    # 优先级 2：单 chunk

parts_info = segment.get("routed_experts_parts")           # 优先级 3：TITO 多步
if not parts_info:
    return None
slices = []
for part in parts_info:
    if part.get("chunks"):
        step_array = combine_chunks(part["chunks"])        # 内层复用 chunks 路径
    else:
        step_array = decode(part["data"])
    slices.append(slice_generated(step_array, int(part["prefix_token_count"]),
                                  int(part["concat_token_count"]),
                                  bool(part.get("is_first_step"))))
```

注意优先级顺序是 chunks → 单字段 → parts：只要更复杂的格式存在就先用它；单字段只是"唯一 chunk 恰为首 chunk"时的存储优化，两种写法语义等价。

### 4.3 数据侧：轨迹提取与样本构建

- **输入**：segment 元数据中的三种格式之一 + `args.num_layers` / `args.moe_router_topk`。
- **处理**：`extract_routed_experts()`（[samples.py](../dressage/rollout/artifacts/samples.py)）解码 → reshape → 切片 → 拼接。
- **输出**：形状 `(num_tokens-1, num_layers, moe_router_topk)` 的 numpy 数组，写入 `sample.rollout_routed_experts`。

按优先级尝试三种格式：

| 优先级 | 字段 | 适用场景 | 处理方式 |
| --- | --- | --- | --- |
| 1 | `routed_experts_chunks` | Partial rollout 多 chunk | 逐 chunk 解码 + `slice_generated` 切片 + `np.concatenate` |
| 2 | `routed_experts` | 单次连续生成 | 直接 `decode` |
| 3 | `routed_experts_parts` | TITO 多步 segment | 逐 step 解码（可能含 chunks）+ 切片 + 拼接 |

**切片逻辑与"-1 偏移"**——这是全链路最容易看错的地方：

```python
def slice_generated(full_array, prefix_count, output_count, is_first):
    if is_first:
        return full_array[:prefix_count + output_count - 1]   # [0, prefix+output-1)
    else:
        start = prefix_count - 1
        return full_array[start:start + output_count]         # [prefix-1, prefix-1+output)
```

为什么是 N-1：路由发生在"处理 token t 的 hidden state"时，而 token t 的 hidden state 决定的是 **token t+1 的 logits**。一条 N 个 token 的序列，只有前 N-1 个位置的路由参与训练（最后一个 token 的路由只用于生成"下一个"token，而下一个 token 不存在）。所以：

- 首 chunk：保留 `[0, prefix+output-1)`——从序列起点到"能预测最后一个 output token"的位置；
- 后续 chunk：从 `prefix-1` 开始取——因为该 chunk 第一个 output token 的 logits 由**它之前那个 token**的路由决定。

提取后还有一道 fail-fast 校验：数组长度必须等于 `len(tokens) - 1`，不符则丢弃并告警；若开了 `use_rollout_routing_replay` 却拿不到路由数据，直接 `ValueError`——避免静默降级到"不重放"状态。

### 4.4 训练侧：Megatron 路由重放

- **输入**：训练 batch 中的 `rollout_routed_experts`（每条样本一个路由数组）。
- **处理**：按层拆分注入各 MoE 层的 `RoutingReplay` 实例；前向时 monkey-patch 的 `compute_topk` 弹出预存索引。
- **输出**：与 rollout 完全一致的 expert 路径。

**（1）补丁机制**：slime 为多个 Megatron 版本提供 patch（`slime/docker/patch/*/megatron.patch`），在两处注入：

1. `TopKRouter.__init__`：每个 MoE 层注册独立的 `RoutingReplay` 实例 + forward pre-hook（确保该层前向时全局 `ROUTING_REPLAY` 指向自己）；
2. `topk_routing_with_score_function`：替换 `compute_topk` 函数。

**（2）RoutingReplay 数据结构**（[routing_replay.py](../slime/slime/utils/routing_replay.py)）：

```python
class RoutingReplay:
    all_routing_replays = []  # 全局列表，按层顺序排列

    def record(self, top_indices):
        # offload 到 CPU pinned memory，避免占用 GPU 显存
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())
```

两个设计点：**CPU 锁页内存 offload**（路由索引不占 GPU 显存，用时才拷回）；**forward/backward 独立索引**（前向和反向各弹各的，保证两遍遍历顺序一致）。

**（3）compute_topk 替换与四个 stage**：`get_routing_replay_compute_topk` 按 `ROUTING_REPLAY_STAGE` 环境变量切换行为：

| Stage | 行为 | 用途 |
| --- | --- | --- |
| `record` | 计算 topk 并存储 indices | 记录"本框架自己算出的路由"（调试用） |
| `replay_forward` | 弹出预存 indices + `scores.gather` | 训练前向，使用 rollout 路由 |
| `replay_backward` | 弹出预存 indices + `scores.gather` | 训练反向，使用相同路由 |
| `fallthrough` | 走原始 topk | 调试对比前向（可关闭） |

核心替换就一行：`torch.topk(scores, k=topk)` → `scores.gather(1, replayed_indices)`（见 3.2 节：只冻结离散选择）。

**（4）fill_routing_replay 注入流程**（[actor.py](../slime/slime/backends/megatron_utils/actor.py)）：逐 micro-batch 从数据里取出 `rollout_routed_experts`，按 VP stage 和层偏移拆开，逐层 `record` 到对应的 `RoutingReplay` 实例；注完后从 `rollout_data` 删除该字段，避免后续重复处理。

### 4.5 与 Partial Rollout 的协作

Partial rollout 允许一条轨迹的生成横跨多次权重更新（pause/resume），不同 chunk 的 token 由不同版本权重生成，路由决策也不同。

R3 用 **chunk 级捕获 + prefix 偏移记录**解决：每个 chunk 独立携带 payload 和 `prefix_token_count`；`extract_routed_experts` 中的 `combine_chunks` 按顺序切片拼接：

```python
def combine_chunks(chunks_info):
    slices = [
        slice_generated(
            decode(chunk["data"]),
            int(chunk["prefix_token_count"]),
            int(chunk["output_token_count"]),
            bool(chunk.get("is_first_chunk")),
        )
        for chunk in chunks_info
    ]
    return np.concatenate(slices, axis=0) if slices else None
```

即使一条轨迹横跨 v1→v2→v3 三个权重版本，每个 token 在其**生成时刻**的路由决策都能被正确恢复。

### 4.6 与 TITO 的协作

TITO 模式下，一个 trajectory segment 由多个 step 的增量 token 拼接而成（见 [multi-segment-design.md](multi-segment-design.md)），每个 step 有自己的生成请求和路由数据。R3 用 `routed_experts_parts` 格式处理：每个 part 对应一个 step，携带 segment 内累积偏移（`prefix_token_count`）和该 step 的 token 数（`concat_token_count`），提取时逐 step 解码、按偏移切片、顺序拼接成 segment 级路由数组。

### 4.7 本章小结

```text
捕获（SGLang）→ 组装（Proxy 三种格式）→ 提取（decode/切片/拼接）→ 回放（gather 替代 topk）；
层间契约只有两个：base64 int32 数组 + token 偏移量；
最容易看错的点：N 个 token 只需 N-1 条路由（最后一个 token 的路由用于生成不存在的下一个 token）。
```

## 五、示例说明

**本章结论先行：三个例子分别看清三件事——路由不一致在单个 token 上怎么发生（5.1）、跨权重版本的轨迹怎么拼接路由（5.2）、开/不开 R3 的一次训练差多少（5.3）。**

### 5.1 一个 token 的路由翻转

设 token `x_t` 经过第 12 层 MoE（256 选 8）。router 对当前 hidden state 打分后，两次前向的分数：

```text
SGLang（推理，FP8 kernel）:        expert 88: 0.2049 │ expert 95: 0.2048
Megatron（训练，BF16 另一套求和）: expert 88: 0.2048 │ expert 95: 0.2049
```

两者仅在第 4 位小数上不同，但 88 和 95 恰好卡在第 8/9 名的边界上——**推理选了 88，训练选了 95**。这个 token 在两次前向中走了不同的 expert，hidden state 从此分叉，logprob 产生差异。注意：分数差异微小到任何"提高精度"的努力都无法根除，因为两个框架的求和顺序本来就不一样。

开 R3 后：训练侧不再做这次 top-k 选择，直接 `scores.gather(1, [..., 88, ...])`——用 Megatron 自己算的分数、取 SGLang 当时选的位置。expert 路径一致，logprob 差异中"路由"这一项归零。

### 5.2 一条横跨权重更新的轨迹怎么拼路由

设一条轨迹：prompt 1000 token，生成中被权重更新抢占一次：

```text
chunk 1（v1 权重）: prefix=1000, 生成 50 token（is_first_chunk=True）
chunk 2（v2 权重）: prefix=1050, 生成 80 token
总 token 数 N = 1000 + 50 + 80 = 1130
```

切片拼接：

```text
chunk 1 切片（首 chunk）: [0, 1000+50-1) = [0, 1049)      → 1049 条路由
chunk 2 切片（后续 chunk）: [1050-1, 1050-1+80) = [1049, 1129) → 80 条路由
拼接后总数 = 1049 + 80 = 1129 = N - 1 ✓（与 len(tokens)-1 校验一致）
```

位置 1049 是 chunk 1 生成的最后一个 token——它的路由决定 chunk 2 第一个 output token 的 logits，所以归 chunk 2 的切片起点。每条路由都对应"生成时刻"的权重版本（1049 之前是 v1、之后是 v2），与训练前向无关。

### 5.3 开/不开 R3 的一个训练 step

同一批 rollout 数据进训练：

| | 不开 R3 | 开 R3 |
| --- | --- | --- |
| 每个 MoE 层的路由 | Megatron 重新 top-k | 弹出 rollout 预存索引 |
| token 路径一致性 | ~85% token 至少一层不同 | 100% 一致 |
| 训练-推理 logprob 绝对差异 | 0.0195 | **0.0077**（-60.5%） |
| $r_t$ 偏离 1 的来源 | 策略更新 + 路由噪声 | 只剩策略更新 |

残余的 0.0077 来自 attention/dense kernel 的数值差——R3 只消除路由这一噪声源，不声称消除全部差异。

### 5.4 本章小结

```text
5.1：第 4 位小数的差异，经过 top-k 离散边界，放大成"换了一个 expert"；
5.2：chunk 切片靠 prefix 偏移对齐，-1 偏移来自"token t 的路由决定 token t+1"；
5.3：R3 把 logprob 差异压低 60.5%，剩下的与路由无关。
```

## 六、应用与边界

**本章结论先行：R3 适用于"MoE 模型 + RL 训练（存在两次前向）"的组合；它要求 proxy 与训练侧同时开启、SGLang 支持路由导出；对稠密模型无意义，与 MOPD 不兼容；它不消除全部训练-推理差异，只消除路由这一类。**

### 6.1 配置与使用

Proxy 侧（开启路由捕获）：

```bash
dressage-proxy \
  --sglang-router-url http://localhost:30000 \
  --tokenizer-path /path/to/Qwen3.5-35B \
  --use-rollout-routing-replay \
  --token-build-mode tito \
  --tito-model qwen3_5
```

训练侧（开启回放注入）：

```bash
python train_async_with_rollout_pause.py \
  --use-rollout-routing-replay \
  --num-layers 40 \
  --moe-router-topk 8 \
  --num-experts 256 \
  ...
```

环境变量（通常由训练脚本自动管理）：

```bash
export ENABLE_ROUTING_REPLAY=1
export ROUTING_REPLAY_STAGE=replay_forward  # 或 record / replay_backward / fallthrough
```

必需参数一览：

| 参数 | 位置 | 说明 |
| --- | --- | --- |
| `--use-rollout-routing-replay` | Proxy | 开启 SGLang 路由捕获（所有 `/generate` 带 `return_routed_experts: true`） |
| `--use-rollout-routing-replay` | slime args | 开启训练侧 replay 注入 |
| `--num-layers` | slime args | MoE 层数（用于 reshape） |
| `--moe-router-topk` | slime args | 每层选择的 expert 数（用于 reshape） |
| `--num-experts` | slime args | expert 总数（用于 padding） |
| `ENABLE_ROUTING_REPLAY=1` | 环境变量 | 激活 Megatron patch |
| `ROUTING_REPLAY_STAGE` | 环境变量 | 控制当前 replay 阶段 |

### 6.2 适用场景

| 场景 | 为什么适合 |
| --- | --- |
| MoE 模型的 RL 训练（GRPO/PPO 等） | 存在"推理采样 + 训练重算"两次前向，IS 比率对一致性敏感 |
| 长轨迹 agentic RL | 轨迹越长，路由不一致累积越大（85% token 受影响是每 token 每层独立摇骰子） |
| 异步 / partial rollout | 跨权重版本的轨迹本就有 staleness，R3 至少把"同版本内的路由噪声"清零 |

### 6.3 常见误区

| 误区 | 正确认知 |
| --- | --- |
| 开了 R3，训练和推理就完全一致 | 错。只消除路由这一噪声源；attention/dense kernel 的数值差仍在（残余 0.0077） |
| R3 冻结了 router 分数/行为 | 错。冻结的是 top-k **索引**（离散选择）；gating 分数仍由当前权重实时算出（gather），梯度正常流过 router |
| R3 可以替代 GSPO / TIS | 互补而非替代。R3 消除 MoE 特有的路由噪声；GSPO/TIS 处理更一般的比率方差与 staleness（见 [llm-rl-algorithms-zh.md](llm-rl-algorithms-zh.md) 8.7/8.8 节），可叠加使用 |
| 评估/纯推理也要开 R3 | 不需要。只有一次前向的场景不存在"两次前向不一致"的问题 |

### 6.4 约束与不兼容

1. **Proxy 与训练侧必须同时开启**：训练侧开了而 segment 里没有路由数据，直接 `ValueError`（fail-fast，避免静默降级为不回放）。
2. **MOPD 不兼容**：Dressage 的 MOPD（多教师路由）模式目前不支持 R3（`mopd_megatron_actor.py` 中有显式检查）。
3. **仅 MoE 模型有意义**：稠密模型没有路由决策。
4. **依赖 SGLang 的路由导出**：需要推理引擎支持 `return_routed_experts`；换引擎需重新适配捕获层。

### 6.5 关键代码索引

| 组件 | 文件 | 核心函数/类 |
| --- | --- | --- |
| SGLang 客户端 | [sglang_client.py](../dressage/proxy/sglang_client.py) | `SGLangRouterClient.generate()`、`_coerce_response()` |
| 生成控制器 | [generation_controller.py](../dressage/proxy/generation_controller.py) | `GenerationController.generate_preemptible()` |
| Proxy 服务器 | [server.py](../dressage/proxy/server.py) | segment record 构建（`routed_experts_parts` 组装） |
| 路由提取 | [samples.py](../dressage/rollout/artifacts/samples.py) | `extract_routed_experts()`、`write_sample_from_segment()` |
| 路由重放核心 | [routing_replay.py](../slime/slime/utils/routing_replay.py) | `RoutingReplay`、`get_routing_replay_compute_topk()` |
| 训练注入 | [actor.py](../slime/slime/backends/megatron_utils/actor.py) | `MegatronTrainRayActor.fill_routing_replay()` |
| Megatron 补丁 | `slime/docker/patch/*/megatron.patch` | `TopKRouter.__init__` + `topk_routing_with_score_function` |
| 实验报告 | [Dressage支持Rollout-Routing-Replay.md](Dressage支持Rollout-Routing-Replay.md) | R3 实验数据与效果分析 |
| 面试版讲解 | [02-r3-moe-routing-replay.md](final/02-r3-moe-routing-replay.md) | 八小节面试结构版 |

### 6.6 本章小结

```text
能用：MoE + RL + SGLang，两侧同开；
不管：稠密模型、MOPD、纯推理；
记住：两侧同开是硬约束（缺数据直接报错），静默降级不存在。
```

## 七、总结

**本质价值**：R3 把 MoE 路由从"两次前向各算各的"变成"一次决策、两处使用"——它消除的不是数值误差，而是数值误差经过 top-k 离散选择被放大成的**路径分歧**。这是训练-推理一致性议题里 MoE 特有的一块拼图，与 TITO（token 一致性）、staleness 控制（权重版本一致性）互补。

**最需要记住的一点**：

**回放的是索引（离散选择），不是分数（连续权重）——`scores.gather(1, replayed_indices)` 替代 `torch.topk(scores)`，一次替换，噪声归零，开销近零。**
