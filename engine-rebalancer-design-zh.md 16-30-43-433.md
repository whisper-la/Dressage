# Dressage Engine Rebalancer：缓存感知的 Turn 级 Engine 调度方案

## 1. 文档定位

本文描述 `Dressage_public` 分支
`engine-rebalancer-recovery-prediction-20260811` 当前实现的 Engine
Rebalancer，并将它作为 Dressage 后续 Engine 间负载均衡的主方案。

方案只改变一次 LLM Turn 被放到哪个 SGLang Engine 上执行，不改变
prompt 数量、样本选择、trajectory 内容或训练 batch 语义。Dressage 不直接导出、
传输或导入 KV tensor，而是始终向目标 Engine 发送完整 `input_ids`：未命中缓存时由
目标 Engine 自然重新 prefill；启用兼容的 SGLang HiCache/Mooncake 时，由 SGLang
自行命中本地缓存或从共享 L3 恢复缓存。

本文严格区分：

- **当前行为**：本分支已经实现并由测试覆盖的行为。
- **评测关注项**：需要通过真实同步 Blackbox shadow/压测确认的效果和风险。
- **后续增强项**：不属于当前实现，不能当作现有保证。

## 2. 问题定义

同步训练要求一个 rollout batch 中的 trajectory 全部结束后才进入下一阶段。不同
trajectory 的 Turn 数和每个 Turn 的输入/输出长度不同，固定粘在某个 Engine 上会形成
长尾：

1. 一部分 Engine 积累长上下文或多 Turn 请求，prefill 和 decode 队列逐渐变深；
2. 另一部分 Engine 较早完成分配给自己的 trajectory，后续利用率下降；
3. batch 完成时间由最慢 Engine 上的最后几个请求决定；
4. 单纯按 running/waiting 请求数调度无法区分短请求与长上下文请求，也无法衡量迁移后
   恢复上下文的成本。

优化目标不是让每个 Engine 的请求数绝对相等，而是最小化每个新 Turn 的预计完成时间，
最终降低同步 rollout batch 的 P95/P99 完成时间：

$$
e^* = \underset{e}{\operatorname{arg\,min}}\;
\widehat{T}(e, \mathrm{turn})
$$

公式中：

- $e$ 是一个候选 SGLang Engine；$e^*$ 是最终选择的 Engine；
- $\widehat T(e,\mathrm{turn})$ 是把当前 Turn 放到 $e$ 后的预计完成时间，单位为秒；
- `arg min` 返回使预计时间最小的 Engine，而不是返回最小时间本身。

$\widehat T$ 不是直接观测值。它会在第 11 节由第 8 节的预计输出长度、第 9 节的上下文
成本、第 10 节的 queue/decode 成本和预测风险共同组成。

其中 `e` 必须健康、负载指标新鲜、权重版本满足请求要求，并且与 session 所属兼容池
一致。

## 3. 范围与非目标

### 3.1 当前范围

- Dressage Proxy 侧的 SGLang Engine 初始放置和 Turn 边界重调度；
- 同步 Blackbox 作为首要效果验证场景；
- 新 session 的负载感知放置；
- 已有 session 在兼容 Engine 之间迁移；
- 本地 KV、共享 Mooncake L3 和完整 re-prefill 三种上下文准备路径的成本预测；
- 在线学习 queue、prefill、decode、缓存命中和恢复误差；
- reservation、防惊群、session 生命周期和诊断快照；
- 功能默认关闭，通过 `--enable-engine-rebalancing` 显式启用。

### 3.2 非目标

- 不做 prompt 过采样，不丢弃生成完成的样本；
- 不改变 rollout batch 的样本数和选择规则；
- 不在一个 Turn 的生成过程中切换 Engine；
- 不由 Dressage 操作 SGLang 的 KV block、FULL/SWA/MAMBA pool；
- 不承诺把 Engine 请求数或 token 数调整为完全一致；
- 不在本阶段把该实现迁移到 `Dressage_inner`。

## 4. 总体架构

```text
Blackbox Agent
    |
    | 一次模型调用，即一个 LLM Turn
    v
Dressage Proxy
    |
    | 1. 构造完整 input_ids 和本 Turn generation budget
    v
EngineRebalancer.acquire()
    |
    | 2. 读取兼容池、实时负载、session owner、在线模型
    | 3. 估计每个候选 Engine 的 queue/context/decode/risk
    | 4. 锁内选择目标并创建 reservation
    v
RoutingLease(worker_url, decision, reservations)
    |
    | 5. GenerationController 将本 Turn 固定定向到 worker_url
    v
SGLang Engine /generate
    |
    | 6. 完整 input_ids；SGLang 自行 local/L3 hit 或 full prefill
    v
SGLang response meta_info
    |
    +-- 成功 --> complete(): 提交 owner、释放 reservation、训练在线模型
    |
    +-- 失败 --> fail(): 释放 reservation、撤销 pending owner
```

核心模块：

| 模块 | 责任 |
| --- | --- |
| `rebalancing/scheduler.py` | Engine 发现、session owner、负载、成本估计、决策、reservation、在线观测 |
| `scheduler_state.py` | `OFF/BOOTSTRAP/ACTIVE/DEGRADED` 兼容池状态机和配置 |
| `cache_hit_estimator.py` | LOCAL/MOONCAKE 缓存命中率估计和 token prefix LCP |
| `context_recovery_model.py` | Queue、prefill、TPOT、context recovery 和预测误差历史模型 |
| `model_cache_profile.py` | 根据模型和部署配置估算不同上下文长度的 KV/state 字节数 |
| `transfer_calibrator.py` | 构造恢复路径校准计划，保存传输延迟/吞吐结果 |
| `ray_calibration.py` | 使用短生命周期 Ray GPU actor 测量 CUDA/Mooncake 路径 |
| `snapshot_store.py` | 原子保存 initial/periodic/final 校准与运行时修正快照 |
| `sglang_client.py` | worker 定向 generate/abort、负载、server info、权重版本接口 |
| `generation_controller.py` | 保证一个 Turn 的生成、pause、abort 和 resume 固定在同一 worker |

### 4.1 从请求到反馈的完整流程

整个方案不是一个只比较 Engine 请求数的路由器，而是一条“安全筛选、性能预测、容量预留、
固定执行、在线反馈”的闭环：

```text
阶段 1：请求进入
    Agent 发起一个新 LLM Turn
    Proxy 构造完整 input_ids、session_id 和本 Turn generation budget
        |
        v
阶段 2：候选筛选
    读取健康状态、负载新鲜度、weight version 和 compatibility fingerprint
    排除不健康、指标过期、版本不符或缓存布局不兼容的 Engine
        |
        v
阶段 3：工作量建模
    计算本 Turn 输入 token 数和与上次成功输入的 Token LCP
    用 request cap、step 历史和 group 剩余长度预测输出 token 数
        |
        v
阶段 4：逐候选成本预测
    projected load：请求、token 和 queue 三类容量压力
    queue cost：历史 queue P75 与实时 prefill backlog 的最大值
    context cost：LOCAL / MOONCAKE / NONE 的恢复或 prefill 成本
    decode cost：预计输出 token 数乘 TPOT P75
    risk：queue/context 历史绝对误差 P90
        |
        v
阶段 5：Stay/Move 决策
    比较 owner 与每个可迁移目标的预计完成时间
    应用 ACTIVE、hold turn、source work 和反向迁移抑制条件
        |
        v
阶段 6：锁内 reservation
    立即登记目标 Engine 的 request/token/prefill 预留量
    后续并发 Turn 能看到刚刚做出的分配
        |
        v
阶段 7：本 Turn 固定执行
    完整 input_ids 定向发送到唯一 worker_url
    generate / pause / resume / abort 均不发生 Turn 内迁移
        |
        v
阶段 8：完成与反馈
    释放 reservation；成功才提交 session owner
    从 queue_time、e2e_latency、decode_throughput、cached token 明细更新模型
```

这八个阶段分别解决四类问题：

| 问题 | 对应阶段 | 核心回答 |
| --- | --- | --- |
| 能不能迁移 | 候选筛选、缓存路径检查 | 目标是否健康、版本与缓存布局是否兼容 |
| 值不值得迁移 | 工作量与成本预测 | 迁移节省的 queue 是否大于恢复、decode 和风险成本 |
| 并发决策会不会惊群 | reservation | 其他尚未进入 SGLang 指标的请求是否已被计入 |
| 模型会不会长期失真 | 完成反馈 | 预测与实际差异是否持续进入在线历史 |

### 4.2 各模块在流程中的边界

`scheduler.py` 是编排中心，但并不自己完成所有统计。各模块的输入、输出和使用位置如下：

| 模块 | 主要输入 | 主要输出 | 在流程中的用途 |
| --- | --- | --- | --- |
| `scheduler.py` | session、候选 Engine、输入 token、负载和各模型估计 | `RoutingDecision`、`RoutingLease` | 组织候选比较、提交 reservation、完成反馈 |
| `scheduler_state.py` | 健康 Engine 数、指标新鲜度、模型 readiness | pool state | 决定已有 session 是否允许正常迁移 |
| `cache_hit_estimator.py` | LCP、attempted cache source、实际 tier token | 缓存覆盖比例 P25 | 估计理论 LCP 中有多少 token 能复用 |
| `context_recovery_model.py` | queue/context/decode 实测值和负载 bucket | queue、prefill、TPOT、risk 分位数 | 把不同资源压力换算成秒 |
| `model_cache_profile.py` | 模型结构、dtype、上下文长度 | KV/state payload bytes | 将需要恢复的 token 数换算成传输量 |
| `transfer_calibrator.py` | source/target 节点、链路、payload bytes | 恢复时间 P75 | 提供 Mooncake 冷启动恢复成本 |
| `sglang_client.py` | Router/worker HTTP 接口 | worker、loads、server/version 信息 | 提供候选和实时观测，并执行 direct route |
| `generation_controller.py` | `RoutingLease.worker_url` | 固定 worker 的 Turn 生命周期 | 防止 chunk、pause、resume 在 Turn 内换 Engine |

### 4.3 一次决策使用的指标总表

| 指标类别 | 代表指标 | 数据来源 | 统计口径 | 影响哪一步 |
| --- | --- | --- | --- | --- |
| 正确性 | health、weight version、fingerprint | Router、worker server/model info | 当前值 | 候选筛选 |
| 请求压力 | running、queued、request capacity | `/v1/loads` | DP rank 聚合 | projected load、queue bucket |
| Token 压力 | active/capacity/usage | `/v1/loads` | sum capacity/active，usage 取最拥挤 rank | projected load |
| Prefill backlog | waiting uncached、reserved prefill | SGLang + Proxy | token 求和 | live queue |
| 输入复用 | Token LCP | session committed tokens | 精确 token 前缀 | cache path/context |
| 输出工作量 | request cap、step P75、group remaining | 请求、rollout、在线历史 | 可用上界取最小值 | reservation/decode |
| 缓存覆盖 | device/host/storage cached tokens | response meta | attempted path 覆盖率 P25 | context cost |
| Prefill 性能 | full-prefill token/context time | response meta | 吞吐 P25 | context/live queue |
| Decode 性能 | decode throughput | response meta | TPOT P75 | 异构 Engine 总成本 |
| Queue 性能 | queue time | response meta | P75 | queue cost |
| 不确定性 | prediction absolute error | 预测值与响应事实 | P90 | move 风险裕量 |

这里的分位数各有不同目的：吞吐取 P25 表示保守地按较慢速度估计；耗时取 P75 表示按较慢
情况估计；误差取 P90 表示给迁移加入更保守的安全裕量。它们是工程分位数，不是统计置信
区间。

### 4.4 统一符号、下标和单位

后续公式统一使用以下约定：

| 符号 | 含义 | 单位 |
| --- | --- | --- |
| $e$ | 任意候选 Engine | 无 |
| $s$、$t$ | source owner 和 target Engine | 无 |
| $N_x$ | 名称为 $x$ 的 token 数或请求数；具体见公式说明 | token 或 request |
| $\mathcal R_x$ | Proxy reservation | request 或 token |
| $V_x$ | 吞吐量 | token/s |
| $K_x$ | 容量 | request 或 token |
| $C_x$ | 某个阶段的预计成本 | 秒 |
| $T_x$ | 实际或预计时间 | 秒 |
| $Q_x$ | queue 时间估计 | 秒 |
| $P_x$ | 无量纲压力分数或比例 | 无 |
| $\widehat x$ | 预测值，不是执行后的实际值 | 与 $x$ 相同 |
| $x^{\mathrm{actual}}$ | 响应后得到或反推的实际值 | 与 $x$ 相同 |
| $P_{25},P_{75},P_{90}$ | 历史样本的第 25、75、90 百分位 | 与样本相同 |

Engine 下标表示“这个量属于哪台 Engine”，例如 $Q_s$ 是 source queue 时间，$Q_t$ 是
target queue 时间。省略 Engine 下标时，表示当前正在估计的候选 Engine。

### 4.5 公式之间的计算依赖

```text
原始请求
  N_input、生成上限、session committed tokens
        |
        +--> Token LCP ------------------------------> N_base
        |
        +--> step/group 历史 ------------------------> N_output_est
                                                        |
SGLang loads + Proxy reservation                        |
  running/queued/active/capacity/usage -----------------+--> projected load P_e
                                                        |       |
                                                        |       +--> queue 历史 bucket
                                                        |       +--> TPOT 历史 bucket
                                                        |
N_base + cache source + cache coverage -----------------+--> N_cached_est
                                                        |       |
prefill throughput + restore time ----------------------+--> C_context
                                                        |
waiting uncached + reserved prefill + prefill throughput ---> Q_live
queue history ----------------------------------------------> Q_history
Q_history + Q_live -----------------------------------------> Q_e

N_output_est + TPOT ---------------------------------------> C_decode
预测误差历史 ----------------------------------------------> R_decision

Q_e + C_context + [C_decode] + [R_decision]
        |
        +--> T_stay / T_move
                  |
                  +--> 选中 Engine --> reservation --> RoutingLease
                                                |
SGLang response meta_info <--- Turn 固定执行 <---+
        |
        +--> 更新 queue/prefill/TPOT/cache/restore/risk 历史
```

方括号表示条件项：同质 Engine 比较时省略两边近似相同的 decode；风险只加在 move 一侧，
使迁移需要承担模型不确定性。

## 5. Turn 和 Session 语义

### 5.1 调度粒度

每次 `/v1/chat/completions` 对应一次 LLM Turn。Proxy 完成本 Turn 的完整
`input_ids` 构造后调用一次 `EngineRebalancer.acquire()`。返回的 `worker_url` 在该
Turn 内保持不变，包括 partial rollout 导致的：

- 原始 generate；
- request-level abort；
- pause 后继续生成；
- 已生成 prefix 与原始输入拼接后的后续 chunk。

下一次 Blackbox 模型调用才会重新做 Engine 选择。因此调度不会在一次 decode 过程中
搬迁请求，也不会要求跨 Engine 拼接某个正在执行的 SGLang request。

### 5.2 Session 路由状态

每个 `session_id` 保存：

- `owner_worker_url`：上一个成功 Turn 的 Engine；
- `pending_owner_worker_url`：当前 Turn 正在尝试的新 Engine；
- `previous_owner_worker_url`：上一次 owner，用于抑制立即回迁；
- `previous_committed_tokens`：上一个成功响应提交的完整 token 序列；
- `seen_engines`：该 session 曾成功运行过的 Engine；
- `owner_turns`：连续停留在当前 owner 的成功 Turn 数；
- `group_id/group_size/task_key`：rollout group 和任务统计上下文；
- `generated_tokens`：当前 trajectory 已生成 token 数；
- `default_step_max_tokens`：rollout 提供的单步输出上限。

当前输入与上次提交序列的最长公共前缀给出可复用基础 token 数：

$$
N_{\mathrm{base}}
=
\operatorname{LCP}\!\left(
\mathrm{tokens}_{\mathrm{committed}},
\mathrm{inputIds}_{\mathrm{current}}
\right)
$$

公式中：

- $N_{\mathrm{base}}$ 是理论可复用前缀长度，单位为 token；
- $\mathrm{tokens}_{\mathrm{committed}}$ 是该 session 上一次成功 Turn 提交的完整 token 序列；
- $\mathrm{inputIds}_{\mathrm{current}}$ 是本 Turn 的完整输入 token 序列；
- $\operatorname{LCP}$ 从第一个 token 开始逐个比较，返回连续相同前缀的长度。

$N_{\mathrm{base}}$ 进入第 9 节缓存覆盖与 context cost，也决定第 7.2 节在模型缺失时如何
估算 owner 上的 prefill reservation。它是理论上限，不等于实际缓存 token 数。

调度只写入 `pending_owner_worker_url`。只有目标 Engine 成功返回后，`complete()` 才提交
新的 `owner_worker_url`、`previous_committed_tokens` 和统计样本。失败时 `fail()` 释放
reservation 并清除 pending owner，不提前改变已提交 owner。

### 5.3 Session 生命周期

同步 Blackbox rollout 在 agent 调用前通过 Proxy API 注册 session context，在正常完成、
失败、取消或 prewarm session ID 变化时清理对应 context。最终完成时将 trajectory
总生成长度写入 group/task 历史，然后移除 session 路由状态。

这个生命周期是整体方案的一部分，不能只复制调度函数而忽略注册和清理，否则会产生
陈旧 owner、reservation 或 group length 样本。

## 6. Engine 发现与兼容池

### 6.1 控制面数据

Rebalancer 每 250 ms 默认轮询 Router 和健康 HTTP worker：

- Router `/workers`：worker URL、健康状态和连接模式；
- Worker `/v1/loads?include=core,queues`：DP rank 负载与队列；
- Worker `/server_info`：模型并行和 HiCache/Mooncake 配置；
- Worker `/model_info`，旧版本回退 `/get_weight_version`：权重版本。

负载指标默认 2 秒后过期，并且 `metrics_stale_ms` 不允许小于四倍轮询周期。单个 worker
控制面查询失败不会阻塞其他 worker，但该 worker 在指标过期后退出健康候选集合。

### 6.2 Compatibility fingerprint

兼容 fingerprint 包含：

- model ID 和 weight version；
- SGLang version；
- TP、PP、DP；
- KV cache dtype、state dtype 和 page size；
- sliding-window 配置；
- Mamba/线性注意力 backend 和 tracking interval；
- HiCache 开关、ratio、write policy、memory layout 和 storage backend；
- Mooncake protocol、device、GPUDirect 等配置。

已有 session 只在 fingerprint 相同的 Engine 之间重调度。这样避免权重、KV layout、
并行拓扑或恢复路径不兼容时错误迁移。当前实现排除低于 `v0.5.15.post1` 的 SGLang
worker，并在诊断接口的 `excluded_engines` 中给出原因。

请求携带 `X-Dressage-Expected-Version` 时，候选还必须匹配该权重版本。

## 7. 负载模型与 Reservation

### 7.1 实时负载

对每个 Engine 聚合所有 DP rank：

- running 和 queued request；
- active token、token capacity 和 token usage；
- request capacity；
- waiting uncached token；
- generation throughput；
- waiting、paused、retracted 和 grammar queue；
- Proxy 尚未反映到 SGLang 快照中的 request/token/prefill reservation。

各字段的具体口径如下：

| 字段 | 聚合方式 | 包含的工作 | 主要用途 |
| --- | --- | --- | --- |
| `running` | 所有 DP rank 求和 | 已经进入执行集合的请求 | request 压力、负载分桶 |
| `queued` | 所有 DP rank 求和 | 尚未开始执行的请求 | queue 压力 |
| `active_tokens` | `num_total_tokens`，缺失时用 `num_used_tokens`，跨 rank 求和 | 活跃请求占用的 prompt/KV 和已生成 token | 总 token 压力 |
| `token_capacity` | `max_total_num_tokens` 跨 rank 求和 | Engine 总 token 容量 | token 压力归一化 |
| `request_capacity` | `max_running_requests` 跨 rank 求和 | Engine 最大运行请求数 | request 压力归一化 |
| `token_usage` | 跨 rank 取最大值 | 最拥挤 rank 的实时 token 使用比例 | 防止求和指标掩盖 rank 热点 |
| `waiting_uncached_tokens` | 跨 rank 求和 | 排队请求尚需计算的 prefill token | live queue 秒数 |
| `gen_throughput` | Engine 报告值 | 当前 decode 产出速率 | 诊断与性能观测 |

其中 `active_tokens` 不是纯 prefill token。一个正在 decode 的请求，其完整上下文 KV 和已生成
token 也会占用 token capacity。`waiting_uncached_tokens` 才是专门用于估计队列前方尚有多少
prefill 工作的指标。

新请求放入 Engine 后的 projected load score 为：

$$
\begin{aligned}
P_e
={}&
\frac{
N_{\mathrm{running},e} + \mathcal R_{\mathrm{req},e} + 1
}{
K_{\mathrm{req},e}
}
\\[4pt]
&+
\max\!\left(
\frac{
N_{\mathrm{active},e}
+ \mathcal R_{\mathrm{tok},e}
+ N_{\mathrm{input}}
+ \widehat{N}_{\mathrm{output}}
}{
K_{\mathrm{tok},e}
},
U_{\mathrm{tok},e}
\right)
\\[4pt]
&+
\frac{
N_{\mathrm{queued},e}
}{
K_{\mathrm{queue},e}
}.
\end{aligned}
$$

为了看清三部分之间的关系，可以定义：

$$
P_{\mathrm{req},e}
=
\frac{N_{\mathrm{running},e}+\mathcal R_{\mathrm{req},e}+1}
{K_{\mathrm{req},e}}
$$

$$
P_{\mathrm{tok},e}
=
\max
\left(
\frac{
N_{\mathrm{active},e}+\mathcal R_{\mathrm{tok},e}
+N_{\mathrm{input}}+\widehat N_{\mathrm{output}}
}{K_{\mathrm{tok},e}},
U_{\mathrm{tok},e}
\right)
$$

$$
P_{\mathrm{queue},e}
=
\frac{N_{\mathrm{queued},e}}{K_{\mathrm{queue},e}}
$$

最终：

$$
P_e
=
P_{\mathrm{req},e}
+P_{\mathrm{tok},e}
+P_{\mathrm{queue},e}
$$

符号、来源和单位如下：

| 符号 | 代码字段 | 含义 | 单位/范围 |
| --- | --- | --- | --- |
| $P_e$ | projected load score | 当前请求也放入 $e$ 后的综合压力 | 无量纲，越小越空闲 |
| $N_{\mathrm{running},e}$ | `running` | SGLang 已运行请求数 | request |
| $\mathcal R_{\mathrm{req},e}$ | `reserved_requests` | Dressage 已选中 $e$、尚未 settle 的请求数 | request |
| $1$ | 当前候选请求 | 模拟把本请求也放入 $e$ | 1 request |
| $K_{\mathrm{req},e}$ | `request_capacity` | Engine 请求容量 | request |
| $N_{\mathrm{active},e}$ | `active_tokens` | SGLang 活跃请求已占 token | token |
| $\mathcal R_{\mathrm{tok},e}$ | `reserved_tokens` | Dressage 尚未 settle 请求的预计总 token | token |
| $N_{\mathrm{input}}$ | `len(input_ids)` | 当前 Turn 完整输入长度 | token |
| $\widehat N_{\mathrm{output}}$ | `estimated_step_output_tokens` | 第 8 节得到的预计输出长度；缺失时按 0 | token |
| $K_{\mathrm{tok},e}$ | `token_capacity` | Engine token 容量 | token |
| $U_{\mathrm{tok},e}$ | `token_usage` | SGLang 最拥挤 DP rank 的 token 使用比例 | $[0,1]$ 附近 |
| $N_{\mathrm{queued},e}$ | `queued` | SGLang 已排队请求数 | request |
| $K_{\mathrm{queue},e}$ | 派生 capacity | $\max(K_{\mathrm{queue,max}},K_{\mathrm{req},e})$ | request |

当 SGLang 未提供 capacity 时，使用兼容池可观察到的最大 capacity 作为回退。相同得分
使用 `session_id + engine_url` 的稳定 hash 打破平局，使放置可复现且不会永远偏向列表
中的第一个 Engine。

三个子公式分别表示：

1. **请求压力**：当前 running、Proxy 已预留请求以及本请求占 request capacity 的比例；
2. **Token 压力**：活跃 token、已预留 token、本请求输入和预计输出占 token capacity 的
   比例，并与 SGLang 报告的最拥挤 rank usage 取最大值；
3. **Queue 压力**：已经排队的请求相对 queue capacity 的比例。

Token 压力的分子并不都是 prefill：

- `activeTokens` 包含当前活跃上下文与 decode 产生的 token；
- `reservedTokens` 等于尚未 settle 请求的完整输入加预计输出；
- `N_input` 是本 Turn 完整上下文，主要贡献未来 KV 和 prefill 工作；
- `N_output` 是本 Turn 预计 decode token，同时也会逐步增加 KV 占用。

因此 $P_e$ 是用于排序和分桶的综合容量压力，不是时间，也不能直接解释为 Engine 利用率。
它一方面直接用于第 12 节冷启动或模型缺失时的新 session 放置，另一方面决定第 10 节查询
哪一个 queue/TPOT 历史 bucket。真正以秒为单位的完成时间由后续
queue/context/decode 模型计算。

### 7.2 Reservation

`acquire()` 在同一把锁内完成决策并立即登记：

- `reserved_requests += 1`；
- `reserved_tokens += input_tokens + expected_output_tokens`；
- `reserved_prefill_tokens += expected_prefill_tokens`。

后续并发 Turn 即使仍看到相同 SGLang 快照，也会把这些 reservation 纳入成本，从而避免
全部迁入同一个刚刚空闲的 Engine。

Prefill reservation 与负载 snapshot generation 绑定。新的 SGLang load generation
出现后，已经被真实负载吸收的 reservation 自动退休；请求完成或失败时也会释放。这样
减少 Proxy reservation 与 SGLang 已观察负载的重复计算。

设当前 `acquire()` 产生的 lease 为 $\ell$，目标 Engine 为 $e$。单个 lease 新增的三个
reservation 分别为：

$$
\Delta \mathcal R_{\mathrm{req}}^{(\ell)}=1
$$

$$
\Delta \mathcal R_{\mathrm{tok}}^{(\ell)}
=
N_{\mathrm{input}}+\widehat N_{\mathrm{output}}
$$

如果已经得到目标 Engine 的上下文估计：

$$
\Delta \mathcal R_{\mathrm{prefill}}^{(\ell)}
=
\widehat N_{\mathrm{prefill}}
$$

公式中：

- 上标 $(\ell)$ 表示“本 lease 新增的量”，不是 Engine 上所有 lease 的总量；
- $\Delta \mathcal R_{\mathrm{req}}^{(\ell)}$ 的单位是 request，当前 lease 固定增加 1；
- $\Delta \mathcal R_{\mathrm{tok}}^{(\ell)}$ 的单位是 token，表示完整输入和第 8 节预计输出之和；
- $\Delta \mathcal R_{\mathrm{prefill}}^{(\ell)}$ 的单位是 token，表示第 9 节上下文模型预计仍需执行
  prefill 的 token，而不是总输入 token；
- $N_{\mathrm{input}}$ 来自本 Turn 的 `len(input_ids)`；
- $\widehat N_{\mathrm{output}}$ 来自第 8 节；如果模型返回 `None`，reservation 中按 0；
- $\widehat N_{\mathrm{prefill}}=N_{\mathrm{input}}-\widehat N_{\mathrm{cached}}$，其中
  $\widehat N_{\mathrm{cached}}$ 由第 9.2 节的 LCP 和缓存覆盖率得到。

这三个量加入目标 Engine 当前 reservation 总量：

$$
\mathcal R_{x,e}^{\mathrm{afterAcquire}}
=
\mathcal R_{x,e}^{\mathrm{beforeAcquire}}
+\Delta \mathcal R_x^{(\ell)},
\qquad
x\in\{\mathrm{req},\mathrm{tok},\mathrm{prefill}\}
$$

这里 $\mathcal R_{x,e}$ 是 Engine $e$ 上仍有效的同类 reservation 总和。更新发生在调度锁内，所以
下一个并发 Turn 计算第 7.1 节 $P_e$ 和第 10.2 节 live queue 时，马上能看到本 lease。

如果上下文模型尚未 ready，则留在 owner 时回退为
$N_{\mathrm{input}}-N_{\mathrm{base}}$；新 session 或迁移目标则保守使用完整输入长度。即：

$$
\Delta \mathcal R_{\mathrm{prefill}}^{(\ell)}
=
\begin{cases}
\widehat N_{\mathrm{prefill}},
& \text{上下文模型可用} \\
\max(0,N_{\mathrm{input}}-N_{\mathrm{base}}),
& \text{模型不可用且留在 owner} \\
N_{\mathrm{input}},
& \text{模型不可用且为新 session 或迁移目标}
\end{cases}
$$

第二种情况假设 owner 能复用全部理论 LCP；第三种情况不假设新目标存在缓存，因此按完整
prefill 预留。这只是模型缺失时的 reservation 回退，不会改变 SGLang 最终实际命中行为。

请求成功或失败、lease settle 时，request/token reservation 按同一个 lease 的增量释放：

$$
\mathcal R_{\mathrm{req},e}^{\mathrm{afterSettle}}
=
\max
\left(
0,
\mathcal R_{\mathrm{req},e}^{\mathrm{beforeSettle}}-1
\right)
$$

$$
\mathcal R_{\mathrm{tok},e}^{\mathrm{afterSettle}}
=
\max
\left(
0,
\mathcal R_{\mathrm{tok},e}^{\mathrm{beforeSettle}}
-\Delta \mathcal R_{\mathrm{tok}}^{(\ell)}
\right)
$$

`max(0,...)` 对应实现中的下界保护，避免重复 settle 或状态刷新使计数变成负数。

`reserved_requests` 和 `reserved_tokens` 一直保留到请求成功或失败、lease settle。原因是
调度器要在请求整个生命周期里表示“这个 Engine 已经承担了该请求”。如果收到请求后立即
释放，而下一次 SGLang load snapshot 还未反映该请求，就会再次出现容量空窗和并发惊群。

`reserved_prefill_tokens` 更特殊：它只表示尚未被 SGLang 快照吸收的 prefill backlog，
因此带有 load generation。新一代 load snapshot 到达后可以提前退休；否则它会和
SGLang 的 `waiting_uncached_tokens` 重复表示同一份工作。

设分配时 Engine 当前 load generation 为 $g$，该 lease 的 prefill reservation 被登记到
$g+1$：

$$
\mathcal R_{\mathrm{prefill},e}
=
\sum_{k>g_{\mathrm{observed},e}}
\mathcal R_{\mathrm{prefill},e}^{(k)}
$$

其中 $\mathcal R_{\mathrm{prefill},e}^{(k)}$ 是等待第 $k$ 代或之后快照吸收的 prefill reservation，
$g_{\mathrm{observed},e}$ 是最新已处理的 load generation。当新 snapshot 令
$g_{\mathrm{observed},e}$ 前进时，已经到代的 bucket 从求和中移除；lease settle 时也会按
自己的 generation 和 token 数释放。它的输出直接进入第 10.2 节：

$$
Q_{\mathrm{live},e}
=
\frac{
N_{\mathrm{waitingUncached},e}+\mathcal R_{\mathrm{prefill},e}
}{V_{\mathrm{prefill},e}}
$$

当前 `reserved_tokens` 没有同样的 generation retirement。当请求已经进入
`active_tokens`、但尚未 settle 时，两者可能暂时重复计数。这是当前实现有意接受的保守
高估：它降低请求集中涌入的风险，但可能在请求执行期间暂时低估该 Engine 的剩余容量。

#### 7.2.1 Reservation 数值示例

假设 Engine B 当前的 Proxy reservation 为：

```text
reserved_requests       = 2 requests
reserved_tokens         = 12,000 tokens
reserved_prefill_tokens = 7,000 tokens
current load generation = 10
```

新的 lease $\ell$ 输入为 8,000 tokens，第 8 节预计输出为 1,000 tokens，第 9 节预计缓存
5,000 tokens。因此：

$$
\Delta \mathcal R_{\mathrm{req}}^{(\ell)}=1
$$

$$
\Delta \mathcal R_{\mathrm{tok}}^{(\ell)}
=8{,}000+1{,}000=9{,}000
$$

$$
\Delta \mathcal R_{\mathrm{prefill}}^{(\ell)}
=8{,}000-5{,}000=3{,}000
$$

所以 `acquire()` 返回时 Engine B 在 Proxy 中表现为：

```text
reserved_requests       = 2 + 1      = 3 requests
reserved_tokens         = 12,000 + 9,000 = 21,000 tokens
reserved_prefill_tokens = 7,000 + 3,000  = 10,000 tokens
```

其中新增的 3,000 prefill tokens 登记在 generation 11。下一个并发请求会立即使用这三个
更新后的总量计算 $P_B$ 和 $Q_{\mathrm{live},B}$。

如果 generation 11 的 SGLang snapshot 先到达，则这 3,000 prefill reservation 被认为已经
进入真实 load 并退休，但 request/token reservation 仍保留：

```text
reserved_requests       = 3 requests
reserved_tokens         = 21,000 tokens
reserved_prefill_tokens = 7,000 tokens
```

如果随后 lease $\ell$ 成功或失败并 settle，再释放它的 request 和 token 增量：

```text
reserved_requests = 3 - 1      = 2 requests
reserved_tokens   = 21,000 - 9,000 = 12,000 tokens
```

如果 lease 在 generation 11 snapshot 到达前就 settle，则它对应的 3,000 prefill token 也会
同时从 generation 11 bucket 中释放。两条释放路径最终都保证该 lease 不再影响后续调度。

## 8. Step 输出长度与 Decode 预算

调度器估计的是当前一次模型调用，而不是完整 trajectory。当前 Turn 的输出 token 上限
由下列正值取最小值：

- Agent request 的 `max_tokens`；
- Proxy/rollout 提供的默认单步上限；
- context window 剩余空间。

模型还维护：

- 同一 task 和 max-token bucket 的成功 step 输出长度 P75；
- 同一 group 的最终 trajectory 长度，以及样本足够时 task 级最终长度。

最终预计输出长度为所有可用约束的最小值：

$$
\widehat{N}_{\mathrm{output}}
=
\min\!\left(
N_{\mathrm{stepCap}},
P_{75}\!\left(N_{\mathrm{stepHistory}}\right),
\widehat{N}_{\mathrm{groupRemaining}}
\right)
$$

公式中：

- $\widehat N_{\mathrm{output}}$ 是本 Turn 的预计输出长度，单位为 token；
- $N_{\mathrm{stepCap}}$ 是本 Turn 的有效硬上限；
- $P_{75}(N_{\mathrm{stepHistory}})$ 是历史成功 Turn 输出长度的 P75；
- $\widehat N_{\mathrm{groupRemaining}}$ 是根据同组或同任务最终 trajectory 长度估计的剩余
  token；
- `min` 只对当前非 `None` 的项执行，不存在的估计不会当作 0 参与。

$\widehat N_{\mathrm{output}}$ 同时进入第 7.1 节 token pressure、第 7.2 节 token reservation
和第 10.3 节 decode 成本，因此这三个模型共享同一份输出工作量估计。

三种输入的计算方式分别是：

1. `stepCap`：请求 `max_tokens`、rollout 默认 step 上限和 context 剩余空间中所有有效正值
   的最小值。这是硬上限，不代表模型一定生成这么长；
2. `stepHistory`：相同 fingerprint、task 和 max-token bucket 的成功 Turn 输出长度。精确
   task 样本不足默认 16 时，回退到相同兼容池和 bucket 的样本，取 P75；
3. `groupRemaining`：同 group 至少有 2 个完成样本时，使用最终 trajectory 长度 P75 减去
   当前已生成 token；否则 task 级最终长度历史至少需要 32 个样本。

更完整地写：

$$
N_{\mathrm{stepCap}}
=
\min
\left(
N_{\mathrm{request/proxy}},
N_{\mathrm{rollout}},
N_{\mathrm{contextRemaining}}
\right)
$$

其中 $N_{\mathrm{request/proxy}}$ 是请求或 Proxy 传入的单步生成上限，
$N_{\mathrm{rollout}}$ 是 session 注册的 rollout 默认上限，
$N_{\mathrm{contextRemaining}}$ 是 context window 还能容纳的输出 token。三者单位均为 token，
也只纳入存在且大于 0 的项。

$$
\widehat N_{\mathrm{groupRemaining}}
=
\max
\left(
0,
P_{75}(N_{\mathrm{trajectoryFinal}})-N_{\mathrm{generated}}
\right)
$$

其中 $N_{\mathrm{trajectoryFinal}}$ 是历史已完成 trajectory 的最终累计生成长度，
$N_{\mathrm{generated}}$ 是当前 session 已经累计生成的 token。二者相减得到还可能剩余的
输出工作，`max(0,...)` 防止当前已生成长度超过历史 P75 时出现负数。

公式只纳入当前存在的项。如果三类信息都不可用，预计输出为 `None`：projected load 和
reservation 中按 0 处理，不编造输出长度；与此同时 decode 成本也不可用，因此异构
Engine 之间不会用一个虚假的 decode 数值触发正常迁移。

## 9. 上下文恢复模型

### 9.1 三种缓存来源

调度器只预测 SGLang 可能采用的上下文准备路径：

| 来源 | 含义 | Dressage 请求行为 |
| --- | --- | --- |
| `NONE` | 目标没有可复用缓存 | 发送完整 `input_ids`，完整 prefill |
| `LOCAL` | owner 或曾运行过该 session 的目标可能有 device/host cache | 发送完整 `input_ids`，SGLang 本地命中或 prefill |
| `MOONCAKE` | 兼容共享 L3 可能恢复 prefix | 发送完整 `input_ids`，SGLang 从 L3 恢复或 prefill |

Dressage 不向 SGLang传入“预计命中 token 数”，也不把预测结果当作事实。最终分类依据
响应中的 `cached_tokens` 和 `cached_tokens_details.device/host/storage`。

### 9.2 缓存覆盖率估计

缓存命中历史按照 fingerprint、Engine、cache source 和上下文 bucket 分组。精确序列
不足时回退到兼容池序列，样本达到阈值后使用缓存覆盖比例 P25：

$$
r_{\mathrm{coverage}}
=
P_{25}\!\left(
\frac{
N_{\mathrm{attemptedSourceCached}}
}{
N_{\mathrm{base}}
}
\right)
$$

公式中：

- $r_{\mathrm{coverage}}\in[0,1]$ 是该 Engine、缓存来源和 context bucket 的保守覆盖比例；
- $N_{\mathrm{attemptedSourceCached}}$ 是本次 attempted source 对应 tier 实际命中的 token；
- $N_{\mathrm{base}}$ 是第 5.2 节 Token LCP 给出的理论可复用上限；
- 每个历史请求先形成一个比值，再对有效历史比值取 P25，而不是先汇总 token 再相除。

$r_{\mathrm{coverage}}$ 与 $N_{\mathrm{base}}$ 相乘，产生后续 context cost 和
`reserved_prefill_tokens` 共用的预计缓存 token 数。

这里必须区分调度前后的两个概念：

- `attempted_source`：调度器在成本预测中依赖的路径；
- `actual_source`：请求完成后，依据 `cached_tokens_details` 判断实际发生的路径。

单次覆盖率的分子只统计 attempted source 对应的 tier：尝试 LOCAL 时使用
`device + host`；尝试 MOONCAKE 时只使用 `storage`。例如尝试 Mooncake，但最终只有本地
device 命中，Mooncake 此次覆盖率应记为 0，不能把本地 token 归功给共享 L3。

历史使用三级回退：

1. 先查 `(fingerprint, target_engine, source, context_bucket)`；
2. 精确历史少于默认 16 个样本时，查去掉 Engine 维度的兼容池历史；
3. 池历史仍少于 16 时，使用冷启动先验。

冷启动时：

- LOCAL owner 默认覆盖率为 1；
- MOONCAKE 使用 `cold_start_hit_probability`，当前默认覆盖率也是 1；
- NONE 始终为 0。

代码字段仍名为 `hit_probability`，但输入样本是 token 覆盖比例且最终取 P25，所以它不是
严格的“完整命中/完全未命中”二元概率，更准确的含义是保守缓存覆盖率：

$$
r_{\mathrm{coverage}}
\approx
\frac{\widehat N_{\mathrm{cached}}}{N_{\mathrm{base}}}
$$

这里 $\widehat N_{\mathrm{cached}}$ 是预计可复用 token，不是 SGLang 响应中的实际
`cached_tokens`。近似号强调 $r_{\mathrm{coverage}}$ 来自历史 P25，并非当前请求的确定事实。

MOONCAKE 的乐观冷启动先验是当前效果评测的重要关注项，应通过真实 cache coverage 和迁移
尾延迟判断是否需要收紧。

### 9.3 Prefill throughput

只使用观测到的完整 prefill 请求训练 prefill throughput，避免把本地或 L3 restore 时间
误当成 prefill。按 Engine 和 context bucket 保存样本，精确样本不足时使用兼容池样本，
达到阈值后取吞吐 P25：

$$
V_{\mathrm{prefill}}
=
P_{25}\!\left(
\frac{
N_{\mathrm{prefill}}
}{
T_{\mathrm{context}}
}
\right)
$$

公式中 $N_{\mathrm{prefill}}$ 是一次实际完整 prefill 的 token 数，单位为 token；
$T_{\mathrm{context}}$ 是排除 queue 和 decode 后的上下文计算时间，单位为秒；单次比值单位为
token/s。$V_{\mathrm{prefill}}$ 是相应 Engine/context bucket 或兼容池样本的 P25。

它被三个后续公式共同使用：完整 prefill cost、缓存未覆盖部分的 prefill cost，以及
waiting uncached token 到 live queue 秒数的换算。

使用 P25 是保守估计，即按较慢的 prefill 吞吐计算成本。

### 9.4 Context cost

完整 prefill：

$$
C_{\mathrm{none}}
=
\frac{N_{\mathrm{input}}}{V_{\mathrm{prefill}}}
$$

$C_{\mathrm{none}}$ 的单位为秒，表示没有可复用缓存时处理全部 $N_{\mathrm{input}}$ token
需要的上下文时间。它也是下面 $C_{\mathrm{miss}}$ 的同义成本。

存在可复用 prefix 时：

$$
C_{\mathrm{hit}}
=
C_{\mathrm{restore}}
+
\frac{
N_{\mathrm{input}} - N_{\mathrm{base}}
}{
V_{\mathrm{prefill}}
}
$$

其中 $C_{\mathrm{restore}}$ 是恢复理论 LCP 对应 KV 的秒数；
$N_{\mathrm{input}}-N_{\mathrm{base}}$ 是即使 LCP 全部命中也仍需 prefill 的新后缀 token。
两项相加得到“理论前缀完全可用”这一分支的 context 秒数。

$$
C_{\mathrm{miss}}
=
\frac{N_{\mathrm{input}}}{V_{\mathrm{prefill}}}
$$

$C_{\mathrm{miss}}$ 表示虽然结构上存在理论 LCP，但实际完全没有缓存覆盖时的完整 prefill
成本。它与 $C_{\mathrm{hit}}$ 构成当前加权模型的两个端点。

$$
C_{\mathrm{context}}
=
r_{\mathrm{coverage}} C_{\mathrm{hit}}
+
\left(1-r_{\mathrm{coverage}}\right) C_{\mathrm{miss}}
$$

$C_{\mathrm{context}}$ 是候选 Engine 的最终上下文准备时间，单位为秒；它进入第 11.2 节
stay/move 总成本。$r_{\mathrm{coverage}}$ 越高，权重越偏向 hit 分支；越低，越偏向完整
prefill 分支。

LOCAL 路径的 restore 可以由在线恢复残差学习；MOONCAKE 路径可以使用离线校准、精确
source-target 在线样本或同类拓扑池化样本。

将公式展开：

$$
C_{\mathrm{context}}
=
r_{\mathrm{coverage}}C_{\mathrm{restore}}(N_{\mathrm{base}})
+
\frac{
N_{\mathrm{input}}-r_{\mathrm{coverage}}N_{\mathrm{base}}
}{V_{\mathrm{prefill}}}
$$

因此 prefill 部分等价于先计算：

$$
\widehat N_{\mathrm{cached}}
=
\left\lfloor
r_{\mathrm{coverage}}N_{\mathrm{base}}
\right\rfloor
$$

$\widehat N_{\mathrm{cached}}$ 的单位为 token；向下取整是因为 reservation 和 token 计数必须
为整数。实现还会把它限制在 $[0,N_{\mathrm{base}}]$ 对应范围内。

$$
\widehat N_{\mathrm{prefill}}
=
N_{\mathrm{input}}-\widehat N_{\mathrm{cached}}
$$

$\widehat N_{\mathrm{prefill}}$ 是候选 Engine 预计仍需计算的输入 token，单位为 token。它既
用于解释 $C_{\mathrm{context}}$ 的 prefill 部分，也直接成为第 7.2 节正常路径的
prefill reservation。

然后只对预计未缓存 token 计算 prefill。加权写法的额外价值是没有把 KV 恢复当作免费。

但当前模型也有明确近似：$r_{\mathrm{coverage}}$ 是覆盖比例，不是严格二元概率。恢复项
$rC_{\mathrm{restore}}(N)$ 隐含恢复时间随 token 数近似线性；若 Mooncake 恢复包含显著固定
启动开销，则它不一定等于 $C_{\mathrm{restore}}(rN)$，部分命中时可能低估恢复成本。
更符合覆盖率语义的后续模型可以直接计算：

$$
C_{\mathrm{context}}^{\mathrm{alternative}}
=
C_{\mathrm{restore}}
\left(\widehat N_{\mathrm{cached}}\right)
+
\frac{\widehat N_{\mathrm{prefill}}}{V_{\mathrm{prefill}}}
$$

该式是后续可评估的模型调整，不是当前代码行为。

## 10. Queue 和 Decode 预测

### 10.1 历史 Queue

根据 fingerprint、Engine 和 projected load bucket 保存实际 `queue_time`。精确 Engine
样本不足时回退到兼容池，并取 P75：

$$
Q_{\mathrm{history}}
=
P_{75}\!\left(T_{\mathrm{queue}}\right)
$$

其中 $T_{\mathrm{queue}}$ 是历史响应 `queue_time` 的单个样本，单位为秒；
$Q_{\mathrm{history}}$ 是当前 Engine 和 projected load/running bucket 的历史排队时间估计，
单位也是秒。它进入本节最终 $Q_e$。

查询时优先使用 projected load score 的四分之一宽度 bucket，例如 `0.50-0.75`；同时保留
projected running 的指数范围 bucket。Engine 精确序列不足默认 16 个样本时，回退到相同
fingerprint 和 bucket 的兼容池历史。这样同一个 Engine 在低负载和高负载下不会共享一个
固定 queue 均值。

### 10.2 实时 Queue

SGLang 暴露的 waiting uncached token backlog 与 Proxy prefill reservation 共同表示尚未
完成的 prefill 工作：

$$
Q_{\mathrm{live}}
=
\frac{
N_{\mathrm{waitingUncached}}
+ N_{\mathrm{reservedPrefill}}
}{
V_{\mathrm{prefill}}
}
$$

公式中：

- $N_{\mathrm{waitingUncached}}$ 是 SGLang 当前等待队列尚需 prefill 的 token；
- $N_{\mathrm{reservedPrefill}}$ 是第 7.2 节尚未被新 load generation 吸收的 Proxy prefill
  reservation；
- $V_{\mathrm{prefill}}$ 是第 9.3 节的保守 prefill 吞吐，单位 token/s；
- $Q_{\mathrm{live}}$ 的单位为秒，表示把当前已知 prefill backlog 处理完所需的近似时间。

分子只包含 prefill backlog，不使用 `active_tokens` 或总 `reserved_tokens`。后两者包含 decode
和 KV 容量压力，适合 projected load，但不能直接除以 prefill throughput 得到 queue 秒数。

最终：

$$
Q_e
=
\max\!\left(Q_{\mathrm{history}}, Q_{\mathrm{live}}\right)
$$

$Q_e$ 是候选 Engine 最终用于决策的 queue 秒数。它与第 9 节 $C_{\mathrm{context},e}$ 相加，
形成一次请求在开始 decode 前的预计等待与上下文成本；异构比较时再加 decode。

如果 SGLang 没有对应字段、指标已过期或 prefill 模型未准备好，则不使用 live queue，
仅依赖历史值。

取最大值而不是平均值，是为了防止其中一个数据源暂时滞后：历史 P75 可能尚未反映突然
形成的深队列，而当前 snapshot 也可能恰好在请求进入前采样。该选择倾向于保守高估 queue。

### 10.3 Decode

按 Engine 和 load bucket 保存 `1/decode_throughput`，样本达到阈值后使用 TPOT P75：

$$
C_{\mathrm{decode},e}
=
\widehat{N}_{\mathrm{output}}
\cdot
P_{75}\!\left(\mathrm{TPOT}_e\right)
$$

其中 $\widehat N_{\mathrm{output}}$ 来自第 8 节，单位 token；
$P_{75}(\mathrm{TPOT}_e)$ 是 Engine $e$ 在相应负载 bucket 下每生成一个 token 的保守时间，
单位秒/token；乘积 $C_{\mathrm{decode},e}$ 的单位为秒。

当 source 和 target 的 TPOT 与 prefill throughput 都在 10% 内时，认为 Engine 同质，
迁移比较省略两边近似相同的 decode 项；异构时必须同时获得两边 decode 估计，否则不因
预测不完整触发正常迁移。

TPOT 样本只在 SGLang 返回有效 `decode_throughput` 时产生：

$$
\mathrm{TPOT}=\frac{1}{V_{\mathrm{decode}}}
$$

$V_{\mathrm{decode}}$ 是 SGLang 返回的 `decode_throughput`，单位 token/s；取倒数得到 TPOT，
单位秒/token。这个 TPOT 历史只用于 decode cost，不使用第 7.1 节仅供诊断的瞬时
`gen_throughput` 替代。

不会仅根据输出 token 数和无法可靠拆分的总耗时臆造 TPOT。历史按 fingerprint、Engine 和
负载 bucket 保存，精确样本不足时回退兼容池，使用 P75 表示较慢 decode 情况。

## 11. 风险模型和最终决策

### 11.1 预测风险

模型保存实际值与当时预测值之间的绝对误差：

- Queue prediction error P90；
- Context prediction error P90；
- 精确恢复路径或同拓扑恢复池的 prediction error P90。

最终风险项：

先分别定义单侧 queue 和 context 的绝对误差样本：

$$
E_{\mathrm{queue},e}
=
\left|
\widehat Q_e-Q_e^{\mathrm{actual}}
\right|
$$

$$
E_{\mathrm{context},e}
=
\left|
\widehat C_{\mathrm{context},e}
-C_{\mathrm{context},e}^{\mathrm{actual}}
\right|
$$

其中 $\widehat Q_e$ 和 $\widehat C_{\mathrm{context},e}$ 是请求发出前保存的预测，带
`actual` 上标的是响应后得到或反推的实际秒数，$E$ 的单位为秒。历史达到样本阈值后：

$$
R_{\mathrm{queue},e}=P_{90}(E_{\mathrm{queue},e})
$$

$$
R_{\mathrm{context},e}=P_{90}(E_{\mathrm{context},e})
$$

$$
\begin{aligned}
R_{\mathrm{decision}}
=
\max\!\big(&
R_{\min},
R_{\mathrm{queue},s}
+ R_{\mathrm{queue},t}
\\
&+ R_{\mathrm{context},s}
+ R_{\mathrm{context},t}
\big).
\end{aligned}
$$

最终公式中：

- $R_{\min}$ 是配置的最低风险，默认 10 ms；
- $R_{\mathrm{queue},s}$、$R_{\mathrm{queue},t}$ 分别是 source 和 target 的 queue 误差 P90；
- $R_{\mathrm{context},s}$、$R_{\mathrm{context},t}$ 分别是两边 context 误差 P90；
- $R_{\mathrm{decision}}$ 是只加到 move 一侧的总风险秒数。

如果 Mooncake 的路径误差样本已经按 non-decode 总成本统计并覆盖 queue，对应 queue risk
不会再次相加，避免同一误差重复惩罚。

在样本不足时，某些历史风险项会返回 0，但 $R_{\min}$ 仍保证 move 至少承担 10 ms 的基本
不确定性成本；Mooncake 路径是否覆盖 queue 则由对应恢复误差模型显式标记。

### 11.2 Stay/Move 成本

#### 11.2.1 先算压力，但不能用压力直接比较 Stay/Move

整体执行顺序可以概括为：

```text
对 source 和每一个 target 分别模拟“把当前 Turn 放进去”
        |
        +--> projected running
        +--> projected load score P_e
                    |
                    +--> 选择 queue 历史 bucket
                    +--> 选择 TPOT 历史 bucket
        |
        +--> 将历史与实时指标换算为 Q_e（秒）
        +--> 计算 C_context,e（秒）
        +--> 计算 C_decode,e（秒，可选）
        |
        v
source 组成 T_stay
每个 target 分别组成 T_move(target)
        |
        v
选择成本最低且满足迁移门槛的 target
```

所以“先计算每个 Engine 的压力，再计算 stay/move”这个理解基本正确，但要补充两点：

1. $P_e$ 是无量纲压力分数，不直接加进 $T_{\mathrm{stay}}$ 或
   $T_{\mathrm{move}}$；
2. $P_e$ 主要决定应该查询哪一组历史 queue/TPOT 样本，之后这些历史指标才被换算成秒。

对每个候选 Engine $e$，首先模拟当前 Turn 已经被放入后的状态：

$$
N_{\mathrm{projectedRunning},e}
=
N_{\mathrm{running},e}
+\mathcal R_{\mathrm{req},e}
+1
$$

公式中，$N_{\mathrm{running},e}$ 是 SGLang 当前运行请求数，
$\mathcal R_{\mathrm{req},e}$ 是 Proxy 已经分配但尚未 settle 的请求数，最后的 1 是当前正在
评估的 Turn。source 和 target 都要加这个 1，因为比较的是“当前 Turn 留在 source”与
“当前 Turn 去 target”两种投放后的结果。

随后使用第 7.1 节得到的 $P_e$ 和这里的
$N_{\mathrm{projectedRunning},e}$ 查询历史：

$$
Q_{\mathrm{history},e}
=
P_{75}
\left(
T_{\mathrm{queue}}
\mid fingerprint,e,bucket(P_e)
\right)
$$

$$
\mathrm{TPOT}_e
=
P_{75}
\left(
\mathrm{TPOT}
\mid fingerprint,e,bucket(P_e)
\right)
$$

竖线右侧表示筛选历史样本的条件，不是除法。精确 projected-load bucket 没有足够样本时，
实现再尝试 projected-running bucket 和兼容池回退。由此可见，压力影响的是“应参考该
Engine 在什么负载状态下的历史耗时”，而不是通过固定系数将压力直接转换成时间。

对任意候选 Engine $e$，最终构造一个 `EngineStepEstimate`：

$$
\widehat T_{\mathrm{preDecode},e}
=
Q_e+C_{\mathrm{context},e}
$$

$$
\widehat T_{\mathrm{full},e}
=
Q_e+C_{\mathrm{context},e}+C_{\mathrm{decode},e}
$$

其中：

- $Q_e$ 是第 10.2 节得到的最终 queue 时间；
- $C_{\mathrm{context},e}$ 是第 9.4 节得到的缓存恢复或 prefill 时间；
- $C_{\mathrm{decode},e}$ 是第 10.3 节根据预计输出长度得到的 decode 时间；
- 两个结果的单位都是秒。

source 使用 LOCAL 上下文路径构造 source estimate。每个 target 根据 session/Engine 关系
分别使用 LOCAL、MOONCAKE 或 NONE 构造 target estimate，因此即使两台 Engine 的
$P_e$ 相同，其 context cost 也可能完全不同。

#### 11.2.2 为什么存在同质与异质两套公式

代码对当前 source-target 这一对 Engine 判断是否同质：

$$
\delta_{\mathrm{prefill}}(s,t)
=
\frac{
\left|V_{\mathrm{prefill},s}-V_{\mathrm{prefill},t}\right|
}{
\max
\left(
\left|V_{\mathrm{prefill},s}\right|,
\left|V_{\mathrm{prefill},t}\right|,
10^{-9}
\right)
}
$$

$$
\delta_{\mathrm{TPOT}}(s,t)
=
\frac{
\left|\mathrm{TPOT}_s-\mathrm{TPOT}_t\right|
}{
\max
\left(
\left|\mathrm{TPOT}_s\right|,
\left|\mathrm{TPOT}_t\right|,
10^{-9}
\right)
}
$$

两项均不超过 0.10 时视为同质。若其中一个比较量缺失，当前 helper 将该项视为“没有证据
表明异质”，不会仅因缺失值把两台 Engine 判为异质。

同质 Engine 的 decode 成本近似相同。它即使同时加到 stay 和 move 两边，也不会改变两者
大小关系，所以当前实现比较 pre-decode 成本以减少对输出长度和 TPOT 预测的依赖。异质
Engine 的 decode 速度不同，这一项不能抵消，必须比较完整 Turn 成本。

#### 11.2.3 同质 Engine 的 Stay/Move

同质 Engine：

$$
T_{\mathrm{stay}}
=
Q_s + C_{\mathrm{context},s}
$$

这里 $Q_s$ 是 source 的最终 queue 秒数，$C_{\mathrm{context},s}$ 是在 source 通过 LOCAL
或实际可用路径准备上下文的预计秒数；二者之和是同质 Engine 下留在 owner 的比较成本。

$$
T_{\mathrm{move}}
=
Q_t + C_{\mathrm{context},t} + R_{\mathrm{decision}}
$$

$Q_t$ 和 $C_{\mathrm{context},t}$ 是 target 对应成本，$R_{\mathrm{decision}}$ 是迁移的不确定性
罚项。风险只加在 move 侧，意味着预测收益必须先覆盖不确定性才会迁移。

将前置公式完全代入后，同质比较实际是：

$$
T_{\mathrm{stay}}
=
\max
\left(
Q_{\mathrm{history},s},
Q_{\mathrm{live},s}
\right)
+C_{\mathrm{context},s}
$$

$$
T_{\mathrm{move}}(t)
=
\max
\left(
Q_{\mathrm{history},t},
Q_{\mathrm{live},t}
\right)
+C_{\mathrm{context},t}
+R_{\mathrm{decision}}(s,t)
$$

这里显式写成 $T_{\mathrm{move}}(t)$，因为每一个 target 都有自己的 queue、context 路径和
source-target 风险。$T_{\mathrm{stay}}$ 对当前 source 固定，而调度器需要分别计算所有
$T_{\mathrm{move}}(t)$。

#### 11.2.4 异质 Engine 的 Stay/Move

异构 Engine：

$$
T_{\mathrm{stay}}
=
Q_s + C_{\mathrm{context},s} + C_{\mathrm{decode},s}
$$

异构情况下增加 $C_{\mathrm{decode},s}$，因为 source 和 target 的 decode 性能不能视为相互
抵消。三个成本项单位都为秒。

$$
T_{\mathrm{move}}
=
Q_t
+ C_{\mathrm{context},t}
+ C_{\mathrm{decode},t}
+ R_{\mathrm{decision}}
$$

target 异构总成本由 queue、context、decode 和 decision risk 四项组成。调度器对所有合法
target 分别计算这个值，选择 $T_{\mathrm{move}}$ 最小且满足迁移门槛的候选；如果没有候选
满足，则使用 source。

将 decode 公式代入，异质 target 的完整公式是：

$$
\begin{aligned}
T_{\mathrm{move}}(t)
={}&
\max
\left(
Q_{\mathrm{history},t},
Q_{\mathrm{live},t}
\right)
+C_{\mathrm{context},t}
\\
&+
\widehat N_{\mathrm{output}}
\cdot P_{75}(\mathrm{TPOT}_t)
+R_{\mathrm{decision}}(s,t).
\end{aligned}
$$

source 的完整公式相同，但不包含迁移风险：

$$
T_{\mathrm{stay}}
=
\max
\left(
Q_{\mathrm{history},s},
Q_{\mathrm{live},s}
\right)
+C_{\mathrm{context},s}
+\widehat N_{\mathrm{output}}
\cdot P_{75}(\mathrm{TPOT}_s)
$$

#### 11.2.5 完整数值示例

假设当前 owner 是 A，候选目标是 B。本 Turn：

```text
N_input           = 10,000 tokens
N_base            = 8,000 tokens
N_output_est      = 1,000 tokens
V_prefill,A/B     = 5,000 token/s
```

压力计算后得到 $P_A=2.25$、$P_B=0.75$。这两个值本身不进入总时间，而是分别定位到历史
bucket。假设随后得到：

| 分项 | A：Stay/LOCAL | B：Move/MOONCAKE |
| --- | ---: | ---: |
| 历史 queue P75 | 2.8 s | 0.8 s |
| 实时 queue | 3.1 s | 0.6 s |
| 最终 $Q_e$ | 3.1 s | 0.8 s |
| 缓存覆盖率 | 0.90 | 0.75 |
| 完全命中分支成本 $C_{\mathrm{hit}}$ | 0.45 s | 1.20 s |
| 完全未命中成本 $C_{\mathrm{miss}}$ | 2.00 s | 2.00 s |

上下文成本分别为：

$$
C_{\mathrm{context},A}
=0.90\times0.45+0.10\times2.00
=0.605\ \mathrm{s}
$$

$$
C_{\mathrm{context},B}
=0.75\times1.20+0.25\times2.00
=1.40\ \mathrm{s}
$$

假设风险为 0.25 秒。若 A、B 同质：

$$
T_{\mathrm{stay}}=3.10+0.605=3.705\ \mathrm{s}
$$

$$
T_{\mathrm{move}}(B)=0.80+1.40+0.25=2.45\ \mathrm{s}
$$

迁移的预计净收益为：

$$
\Delta T(B)
=
T_{\mathrm{stay}}-T_{\mathrm{move}}(B)
=1.255\ \mathrm{s}
$$

因为 $\Delta T(B)>0$，再满足第 11.3 节的 ACTIVE、source 有其他工作和 hold 条件后，才会
迁移到 B。

但如果两台 Engine 异质，A 的 TPOT 为 4 ms/token，B 为 6 ms/token：

$$
C_{\mathrm{decode},A}=1{,}000\times0.004=4.0\ \mathrm{s}
$$

$$
C_{\mathrm{decode},B}=1{,}000\times0.006=6.0\ \mathrm{s}
$$

于是：

$$
T_{\mathrm{stay}}=3.705+4.0=7.705\ \mathrm{s}
$$

$$
T_{\mathrm{move}}(B)=2.45+6.0=8.45\ \mathrm{s}
$$

虽然 B 的压力和 queue 都更低，但它 decode 更慢，最终 $T_{\mathrm{move}}>T_{\mathrm{stay}}$，
所以不会迁移。这个例子说明 Rebalancer 不是简单选择压力最低的 Engine，而是选择预计完成
时间更短的 Engine。

### 11.3 当前迁移条件

已有健康 owner 的正常迁移需同时满足：

1. 兼容池状态为 `ACTIVE`；
2. source 和 target 的必要模型、恢复路径已经 ready；
3. target 与 source 不同；
4. `T_move < T_stay`；
5. source 上除当前 Turn 外还有 running、queued 或 reserved work；
6. 在当前 owner 上已经完成至少 `min_hold_turns`，默认 1；
7. 如果 target 是 `previous_owner_worker_url`，收益必须大于两倍风险。

当前实现没有固定 500 ms/20% 的硬收益阈值，也没有单 session 最大迁移次数限制。这两项
属于评测后决定是否增加的增强项，不能写成当前保证。

## 12. 冷启动与降级状态机

每个 compatibility pool 独立维护：

```text
              prerequisites ready
BOOTSTRAP --------------------------> ACTIVE
    ^                                    |
    |                                    | readiness lost
    |                                    v
    +-------------------------------- DEGRADED
                 readiness restored       |
                         ACTIVE <----------+

config disabled -> OFF
```

Pool readiness 要求：

- 至少两个健康 Engine；
- 所有候选负载指标新鲜；
- Queue 模型 ready；
- Prefill 模型 ready；
- 至少一条保守可用的候选恢复路径。

当前行为：

- `OFF`：所有请求继续通过原 Router；
- `BOOTSTRAP`：新 session 使用 projected-load 放置，已有 session 保持健康 owner；
- `ACTIVE`：允许基于成本收益迁移；
- `DEGRADED`：新 session 仍做 projected-load 放置，已有健康 owner 保持 sticky；
- owner 不健康时不受正常收益阈值限制，选择可用兼容候选完成 full-input failover。

默认至少 16 个有效 queue 和 full-prefill 样本后相应在线模型才 ready。

## 13. Mooncake 路径校准

### 13.1 为什么需要离线校准

对于共享 L3 路径，仅知道 KV 字节数不等于知道恢复时延。恢复可能经过：

- GPU Direct RDMA；
- device-to-host staging；
- Mooncake TCP/RDMA；
- host-to-device staging。

Rebalancer 需要测量完整路径 P75，而不是把固定 latency 和带宽传输时间再次相加，避免
重复计算。

### 13.2 部署配置

高级部署通过环境变量指定 JSON：

```bash
export DRESSAGE_ENGINE_REBALANCING_DEPLOYMENT_CONFIG=/path/to/deployment.json
```

JSON 描述 Ray 地址、节点/GPU、HiCache write policy、Mooncake protocol/device、
metadata server、GPUDirect 和模型部署信息。共享 L3 校准要求 `write_through`；GPUDirect
要求 RDMA。

### 13.3 校准生命周期

```text
DISABLED
WAITING_FOR_RAY
RUNNING
READY | DEGRADED
```

短生命周期 Ray actor 在指定节点占用校准 GPU，测量 8K/16K/32K/64K 上下文对应的
payload bucket。actor 退出且 Ray 资源重新可用后才发布 `READY`。校准失败只影响性能
预测：相应路径回退到完整 prefill 估计，不应改变 token 或 trajectory 语义。

在线请求继续学习精确 source-target 和同拓扑池的恢复残差。默认精确/池化模型需要 16
个样本；Mooncake 池达到 4 个样本后允许使用 provisional P75，并且不会比离线下界更
乐观。

## 14. 请求执行、完成与失败

### 14.1 成功路径

1. `acquire()` 返回 `RoutingLease`；
2. `GenerationController` 定向调用 `{worker_url}/generate`；
3. pause/abort 只访问该 worker；
4. resume 使用 `original_input_ids + partial_output_ids`，仍访问同一 worker；
5. 响应版本校验通过；
6. `complete()` 释放 reservation、提交 owner 和 committed tokens；
7. 使用 `meta_info` 更新在线模型；
8. 正常构造 trajectory step，样本内容不因调度而改变。

### 14.2 当前失败语义

必须明确，当前 public 分支的语义是：

- Engine discovery、负载新鲜度、版本或兼容候选不可用时，`acquire()` 抛出错误，Proxy
  返回 `503 engine_rebalancing_unavailable`；
- 已选目标 worker 的 `/generate` 失败时，调用 `fail()` 释放 lease 后继续传播原异常；
- 当前不会在同一个 Turn 内自动重试旧 owner，也不会自动退回 Router；
- owner 不健康但仍存在其他健康兼容 Engine 时，决策阶段支持 full-input failover。

因此“调度控制面不可用自动回退 Router”和“目标 worker 失败自动回退旧 owner”是后续
可靠性增强项，不是当前效果测试中的现有行为。首次 public 评测需要单独统计这些失败，
避免只观察成功 batch 的耗时而忽略 rollout 成功率。

## 15. 在线观测如何训练模型

成功请求读取：

- `queue_time`；
- `e2e_latency`；
- `decode_throughput`；
- `cached_tokens`；
- `cached_tokens_details.device/host/storage`。

一次成功响应不是作为一个整体样本写入所有模型，而是先拆成 queue、context、decode、缓存
覆盖和恢复路径等不同观测。只有相应字段存在且语义可靠，才更新对应模型。

估算：

$$
T_{\mathrm{decode}}
=
\frac{
\max\!\left(0, N_{\mathrm{output}}-1\right)
}{
V_{\mathrm{decode}}
}
$$

公式中 $N_{\mathrm{output}}$ 是本次响应实际生成 token 数，$V_{\mathrm{decode}}$ 是响应中的
实际 decode throughput。减 1 与当前 SGLang 统计口径一致；`max` 保证空输出或单 token
输出不会得到负时间。结果单位为秒，用于从 e2e 中拆出 context。

$$
T_{\mathrm{context}}
=
\max\!\left(
0,
T_{\mathrm{e2e}}-T_{\mathrm{queue}}-T_{\mathrm{decode}}
\right)
$$

$T_{\mathrm{e2e}}$、$T_{\mathrm{queue}}$、$T_{\mathrm{decode}}$ 均以秒计；三者相减得到
prefill/缓存恢复所在的 context 阶段。`max(0,...)` 吸收计时粒度和字段口径造成的小幅负值。
这个实际 context 时间进入第 9.3 节 prefill 吞吐样本和第 11.1 节 context prediction error。

缓存来源优先使用 tier breakdown：storage 为 MOONCAKE，device/host 为 LOCAL，没有命中
为 NONE。缺少 breakdown 时才结合结构路径和总 `cached_tokens` 推断。

具体分类为：

$$
\mathrm{actualSource}
=
\begin{cases}
\mathrm{MOONCAKE}, & N_{\mathrm{storage}}>0 \\
\mathrm{LOCAL}, & N_{\mathrm{device}}+N_{\mathrm{host}}>0 \\
\mathrm{NONE}, & \text{otherwise}
\end{cases}
$$

其中 $N_{\mathrm{device}}$、$N_{\mathrm{host}}$、$N_{\mathrm{storage}}$ 分别来自 SGLang
`cached_tokens_details`，单位均为 token。判断按顺序执行：storage 非零优先分类为
MOONCAKE，其次 device/host 非零分类为 LOCAL，全部为零分类为 NONE。

如果 device/host/storage 同时有 token，只要 storage 大于 0，单值枚举就分类为 MOONCAKE；
完整 tier 数量仍保存在 observation 中，因此混合命中的信息没有丢失。

实现会区分：

- 预测尝试的 cache source；
- 实际观测的 cache source；
- raw queue；
- 可用于 queue 模型训练的 queue；
- context recovery residual；
- non-decode prediction error。

Mooncake 的 non-decode 时间可能同时包含 queue 和 restore，因此不会把它直接污染普通
Queue/Prefill 模型。

### 15.1 attempted source 与 actual source 的分工

`attempted_source` 回答“调度器做决策时依赖哪条缓存路径”，`actual_source` 回答“SGLang
最终从哪一层取到了 KV”。二者可能不同：

| attempted | actual | 含义 | 缓存覆盖率如何记账 |
| --- | --- | --- | --- |
| LOCAL | LOCAL | 本地路径按预期命中 | 分子取 device + host |
| LOCAL | NONE | 本地 KV 已淘汰或前缀未命中 | LOCAL 覆盖率记 0 |
| MOONCAKE | MOONCAKE | 共享 storage 恢复发生 | 分子只取 storage |
| MOONCAKE | NONE | 有恢复路径，但此次未恢复到 token | MOONCAKE 覆盖率记 0 |
| MOONCAKE | LOCAL | 目标本地意外存在缓存 | Mooncake 覆盖率记 0，不冒领本地 token |

如果没有旧 owner 或 Token LCP 为 0，结构来源直接是 NONE。即使部署了 Mooncake，也不能
把这种请求标记为“尝试 Mooncake 但 miss”，因为不存在理论可恢复的前缀。

缓存覆盖历史按 attempted source 更新，用来衡量“调度器押注的路径是否可靠”；性能归因、
实际 context 路径和恢复吞吐按 actual source 更新。只有 attempted 与 actual 相同时，相应
路径的预测误差才可安全用于校准该路径。

### 15.2 各模型的观测来源

| 在线模型 | 使用的响应字段 | 产生样本的条件 | 统计结果 |
| --- | --- | --- | --- |
| Queue | `queue_time` | 非 storage 混合污染，且字段有效 | Engine/pool P75 |
| Queue risk | 预测 queue 与实际 queue | 两者均存在 | 绝对误差 P90 |
| Prefill throughput | context time、实际 prefill token | `actual_source=NONE` 的完整 prefill | 吞吐 P25 |
| TPOT | `decode_throughput` | 吞吐为正 | TPOT P75 |
| Cache coverage | LCP、tier cached tokens | attempted source 明确 | 覆盖比例 P25 |
| Context risk | 预计和实际 context time | 两者可可靠拆分 | 绝对误差 P90 |
| Restore runtime | cached token、context/non-decode、prefill throughput | 实际命中且路径 key 可构造 | 恢复秒数 P75 |

缺少 `queue_time` 时，无法从 e2e 中可靠拆出 context，因此不生成该 context 样本；缺少
`e2e_latency` 时可用 Proxy 墙钟 elapsed 回退为端到端时间；缺少 tier breakdown 时只能用
结构来源和总 `cached_tokens` 做退化分类。

### 15.3 实际 Prefill 和 Restore 的反推

实际需要 prefill 的 token 数为：

$$
N_{\mathrm{prefill}}^{\mathrm{actual}}
=
N_{\mathrm{input}}-N_{\mathrm{cached}}^{\mathrm{actual}}
$$

$N_{\mathrm{cached}}^{\mathrm{actual}}$ 是 SGLang 返回并限制在输入长度范围内的实际总缓存
token；从本 Turn 输入长度中扣除后得到实际需要计算的 prefill token。该值用于反推恢复
时间，不等同于调度前的 $\widehat N_{\mathrm{prefill}}$。

如果 prefill throughput 已就绪，并且可以获得不含 decode 的 recovery base time，则恢复
耗时近似为：

$$
T_{\mathrm{restore}}^{\mathrm{actual}}
=
\max
\left(
0,
T_{\mathrm{recoveryBase}}
-
\frac{N_{\mathrm{prefill}}^{\mathrm{actual}}}
{V_{\mathrm{prefill}}}
\right)
$$

公式中 $T_{\mathrm{recoveryBase}}$ 是扣除 decode、并在 Mooncake 情况下再扣除预计 queue 后
留给恢复与剩余 prefill 的秒数；后一项是按第 9.3 节吞吐估算的实际剩余 prefill 秒数。二者
相减得到 KV 恢复残差，`max(0,...)` 防止观测误差产生负恢复时间。

对于实际 Mooncake 路径：

$$
T_{\mathrm{recoveryBase}}
=
T_{\mathrm{e2e}}-T_{\mathrm{decode}}-\widehat T_{\mathrm{queue}}
$$

这里 $\widehat T_{\mathrm{queue}}$ 是请求发出前保存的 queue 预测，不一定等于响应中的
`queue_time`。Mooncake 的 non-decode 路径可能把 queue 与 storage restore 混在一起，所以
当前实现用预测 queue 做拆分，并将产生的偏差交给路径风险模型吸收。

对于不涉及 storage 的 LOCAL 路径，recovery base 使用已拆出的 context time。这里 Mooncake
使用的是预测 queue 而非完全可观测的真实分段，因此 queue 与 restore 误差可能耦合；后续
通过路径 prediction error P90 给迁移增加风险裕量。

### 15.4 Session 状态提交顺序

请求 settle 时先释放 reservation。成功响应才执行：

1. 将旧 owner 记入 `previous_owner_worker_url`；
2. 提交新 `owner_worker_url`；
3. 将目标加入 `seen_engines`；
4. 保存本次完整 committed token 序列；
5. 累计 trajectory 已生成 token；
6. 更新 step length、queue、context、decode、cache 和 restore 历史。

失败时只清理 pending owner 和 reservation，不改变最后一个成功 owner，也不把失败耗时
写入成功性能历史。

## 16. 可观测性

### 16.1 HTTP 接口

`GET /v1/engines/load` 返回：

- effective config；
- compatibility pool 状态与 readiness；
- 每个 Engine 的负载、reservation、指标年龄；
- deployment fingerprint 和 excluded Engine；
- source-target path readiness；
- model cache profile；
- queue/prefill/TPOT/cache-hit/step-length 模型样本数；
- recent routing decisions；
- recent context observations；
- active session 数。

`GET /v1/engines/calibration` 返回离线校准状态、路径结果和运行时恢复模型摘要。

### 16.2 文件快照

启用正常 CLI 后，Proxy 默认在：

```text
${LOG_DIR}/proxy/rebalancing/${DRESSAGE_RUN_NAME}/${UTC_START}-${PID}/
```

原子写入：

- `initial.json`；
- 每 128 个成功在线请求的 `request-XXXXXXXXX.json`；
- 优雅关闭时的 `final.json`。

快照应作为性能实验产物保留，用于离线检查预测来源、样本充足度和恢复误差。

### 16.3 训练侧效果指标

ON/OFF 对照至少记录：

- rollout batch wall time P50/P95/P99；
- trajectory wall time P50/P95/P99；
- 每 Engine running/queued/token usage 时间序列；
- Engine idle 时间和 batch 尾部利用率；
- 新 session 放置次数、正常迁移次数、不健康 owner failover 次数；
- LOCAL/MOONCAKE/NONE 实际命中次数和 token 数；
- `T_stay/T_move` 预测与实际误差；
- reservation 与 live backlog；
- Rebalancer 503、worker 请求失败和 rollout 成功率；
- batch 样本数、group 数和 trajectory 内容一致性。

## 17. Public 效果验证方案

### 17.1 单元与代理集成测试

当前专项测试覆盖：

- 配置、状态机和 SGLang 版本门槛；
- cache profile、LCP 和三种 cache source；
- queue/prefill/TPOT/step-length/group-length 模型；
- load 解析、DP rank 聚合和 live backlog；
- reservation 防惊群与 snapshot generation 退休；
- 新 session 放置、ACTIVE 迁移和 owner failover；
- Mooncake 校准、运行时恢复残差和 cache tier 分类；
- 定向 generate/abort；
- session 注册/清理、取消和 lease settle；
- benchmark 确定性与 Proxy 生命周期。

本分支 `tests/test_engine_rebalancing.py` 当前为 101 项通过。

### 17.2 双 Fake Engine 集成场景

构造两个相同 fingerprint 的 Engine：

1. 第一个 Turn 成功落在 A 并提交 owner；
2. 注入 A 的 deep queue 或 waiting uncached token backlog；
3. B 保持低负载；
4. 预热 queue 和 full-prefill 模型进入 ACTIVE；
5. 第二个 Turn 应计算出 B 的总完成时间更低并定向到 B；
6. 验证发送的是完整 `input_ids`；
7. 验证 owner 只在 B 成功后提交；
8. 验证 batch 内容、样本数、token/logprob 对齐不变。

### 17.3 真实 ON/OFF Benchmark

使用同一套：

- prompt 文件和 prompt hash；
- sampling seed；
- 模型 checkpoint；
- SGLang/HiCache/Mooncake 配置；
- rollout batch size 和并发度；
- 节点、GPU、网络和训练配置。

先运行 OFF，再运行 ON；benchmark 脚本应一次生成有效 prompt/seed 清单，并在两个阶段
复用，避免因输入差异制造虚假收益。

建议至少覆盖：

| 场景 | 目的 |
| --- | --- |
| 无 L3、同质 Engine | 验证完整 re-prefill 下迁移是否仍有净收益 |
| 单机 Mooncake | 验证 local storage restore 与 prefill 的边界 |
| 多机 TCP/RDMA | 验证网络恢复预测和 P99 |
| 高并发长短混合 trajectory | 验证同步 batch 长尾改善 |
| 均匀短请求 | 验证调度不会显著增加正常 workload 开销 |
| worker/control-plane 故障注入 | 量化当前失败语义对成功率的影响 |

### 17.4 首轮效果判定

首轮不预设必须达到某个百分比，但至少满足：

1. ON/OFF 的样本数、group 数和训练数据语义一致；
2. ON 不得出现 token/logprob/version 对齐回归；
3. 在目标长短混合 workload 上，rollout batch P95/P99 有稳定改善；
4. 迁移请求的预测误差随样本增加收敛，而不是持续系统性低估；
5. 任何吞吐收益必须与 Rebalancer 503 和 rollout 失败率一起解释；
6. 对均匀 workload，调度控制面开销不能抵消收益。

## 18. 当前配置

正常用户只需要一个开关：

```bash
dressage-proxy \
    --tokenizer-path /path/to/model \
    --sglang-router-url http://router:port \
    --enable-engine-rebalancing
```

当前 `EngineRebalancingConfig` 默认值：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `enabled` | `false` | 默认不改变现有 Router 行为 |
| `load_poll_interval_ms` | 250 | worker 负载轮询周期 |
| `metrics_stale_ms` | 2000 | 指标新鲜度上限 |
| `history_size` | 128 | 每条历史序列最大样本数 |
| `min_samples` | 16 | 在线模型 ready 阈值 |
| `min_hold_turns` | 1 | 正常迁移前在 owner 的最少成功 Turn |
| `min_risk_ms` | 10 | 决策最小风险裕量 |
| `cold_start_hit_probability` | 1.0 | 未观测 Mooncake 路径的命中先验 |

高级 Mooncake 校准通过
`DRESSAGE_ENGINE_REBALANCING_DEPLOYMENT_CONFIG` 提供，不增加普通 CLI 参数集合。

## 19. 已知限制与 Public 阶段重点决策

### 19.1 已知限制

- 控制面或所有候选不可用时当前返回 503，不自动回退 Router；
- 目标 worker generate 失败时当前不自动重试旧 owner；
- 正常迁移只有 prediction risk，没有 500 ms/20% 硬收益门槛；
- 没有单 session 最大迁移次数；
- 权重发布后依赖 fingerprint/version 过滤，没有显式清空全部在线性能历史；
- 冷启动 Mooncake hit probability 默认 1，可能在实际 L3 命中率较低时偏乐观；
- 核心 scheduler 体量较大，效果验证应包含 Proxy CPU/事件循环开销。

### 19.2 Public 效果确认后再决定的增强项

以下增强不在本阶段直接混入，以保证先评估当前 public 实现：

1. 调度控制面不可用时返回 `worker_url=None`，恢复原 Router sticky 路径；
2. 目标 Engine 请求失败后，以完整 `input_ids` 重试旧 owner，再回退 Router；
3. 增加绝对/相对硬收益门槛；
4. 增加单 session 最大迁移次数；
5. 权重发布、Proxy pause/resume 或拓扑变化时显式重置路由/模型；
6. 根据真实 shadow 数据调整 cold-start hit prior、min samples 和风险分位数。

## 20. 后续迁移到 Dressage_inner 的边界

如果 public 验证达到预期，inner 不再维护另一套独立成本模型，而是整体迁移以下链路：

1. `rebalancing/` 全部模型、状态机和校准模块；
2. SGLang client 的定向请求和控制面接口；
3. GenerationController 的 Turn 固定 worker；
4. Server 的 acquire/complete/fail 生命周期和诊断接口；
5. ProxyClient、Blackbox dispatch、prewarm 的 session context 注册/清理；
6. benchmark、配置和专项测试。

迁移前应以 public 实验数据决定第 19.2 节的可靠性增强是否作为 inner 上线前置条件。
整体迁移不包含 prompt 过采样，也不保留另一套简单 TurnScheduler 与 EngineRebalancer
同时做决策。

## 21. 代码与提交索引

关键提交：

- `78232b5`：加入 Proxy Engine Rebalancer 主体；
- `5b36b42`：加入 SGLang load metrics、live queue 和 prefill reservation；
- `d6fd26d`：完善 benchmark 确定性和 session 清理；
- `4affab2`：修复恢复成本和 queue prediction；
- `20699d6`：修复 cache source 分类和 benchmark load。

关键入口：

- `dressage/proxy/rebalancing/scheduler.py`；
- `dressage/proxy/server.py` 的 `/v1/chat/completions`、`/v1/engines/load` 和
  `/v1/engines/calibration`；
- `dressage/proxy/generation_controller.py`；
- `dressage/proxy/sglang_client.py`；
- `tests/test_engine_rebalancing.py`；
- `examples/scripts/benchmark_engine_rebalancing_qwen3.5_4b_sync_local_l3_hicache.sh`。
