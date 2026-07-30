# Dressage 项目基础细节问题清单

> **用途**：梳理理解 Dressage 代码库所需掌握的基础问题，作为 onboarding 自检、面试准备与代码导览索引。
>
> **使用方式**：按类别逐题自查——每道题都应能指向具体代码文件或文档给出回答；答不上来的题目即为需要深入阅读的模块。已解答的题目带 ✅ 标记，答案折叠在题目下方的 `<details>` 块中。

## 已解答的四个入门问题

以下四个问题已在初期探索中解答，作为理解本清单的前置知识：

1. **Proxy 的存储格式有哪些字段？**
   - 运行时逐步记录：[StepRecord](../dressage/proxy/session_manager.py)（约 40 个字段：标识 / 消息 / token 级 / 权重版本 / TITO 增量 / 分段标记 / MoE 路由）
   - finalize 后存储：[TrajectorySegment](../dressage/proxy/trajectory_store.py)（`tokens` / `full_logprobs` / `full_loss_mask` 等长校验，`trajectory_id` / `instance_id` / `segment_index` / `extra_info` 等）
2. **整个流程是怎么做的？**
   - Agent → Proxy（`/v1/chat/completions`，逐步记录）→ SGLang → `/session/finalize` 切段 → `/trajectory/read` 取段 → 展开为 Sample → reward → train_data → slime 训练
   - 详见 [docs/proxy.md](../docs/proxy.md) 与 [docs/rollout.md](../docs/rollout.md)
3. **送给训练的样本是什么样的？**
   - 中间表示：slime [Sample](../slime/slime/utils/types.py)（`tokens` 为完整序列，`loss_mask` / `rollout_log_probs` 只含 response 部分）
   - 最终形式：`train_data` dict，见 [convert_samples.py](../dressage/rollout/convert_samples.py)（含 prompt-equal 的 `rollout_mask_sums`）
4. **OpenAI 兼容格式是什么样的？**
   - `POST /v1/chat/completions` 完全 drop-in 替代，见 [server.py](../dressage/proxy/server.py)
   - 标准 OpenAI 请求/响应字段 + 扩展 header（`X-Session-Id` / `X-Instance-Id` / `X-Turn-Id` / `X-Dressage-Expected-Version`）
   - 自定义错误码：409（session finalized）/ 413（context_overflow）/ 502（版本与抢占类）/ 503（生命周期与上游）

---

## 扩展问题清单

### 一、Proxy 内部机制（存储与会话）

> 涉及代码：[session_manager.py](../dressage/proxy/session_manager.py)、[trajectory_store.py](../dressage/proxy/trajectory_store.py)

1. ✅ Session 的完整生命周期是什么？（创建 → 活跃 → finalize → 过期清理，`session_timeout` 怎么起作用？）

<details>
<summary><b>答案（点击展开）</b></summary>

**结论**：Proxy 的 Session 有 **4 个状态**——`不存在 → 活跃（Active）→ 已终结（Finalized）→ 彻底遗忘`，由 [SessionManager](../dressage/proxy/session_manager.py#L217) 管理，内部维护三张表：

| 表 | 类型 | 内容 |
|---|---|---|
| `_sessions` | `dict[str, Session]` | 活跃 session |
| `_finalized_session_ids` | `dict[str, float]` | 已终结 session_id → 终结时间戳 |
| `_finalization_results` | `dict[str, dict]` | 已终结 session 的缓存响应（幂等用） |

```text
                POST /v1/chat/completions                        POST /session/finalize
                (session_id 不存在)                               (write_many 成功后)
  ┌────────┐  ──────────────────────▶  ┌─────────┐  ─────────────────────────▶  ┌───────────┐
  │ 不存在  │                           │  Active  │                               │ Finalized │
  └────────┘  ◀──────────────────────  └─────────┘  ◀────────────────────────    └───────────┘
                空闲超时被清理                    │ 同 session_id 再请求 → 409         │
                (last_active 超期                │ 重复 finalize → 返回缓存结果        │
                 且无 request_lock)               ▼                                  ▼
                                          活跃期间：                        终结记录超期后
                                          记 step / 推进 turn              两张表条目被清除
                                          / 记录 rewrite                   （同名 session_id 可重新创建）
```

**`session_timeout`（默认 3200 秒，CLI `--session-timeout`）是"空闲超时 + 墓碑保留期"二合一参数**，且清理是**纯懒触发**——没有后台扫描线程，只在 4 个入口顺路执行。

**阶段一：创建**

代码入口：`SessionManager.get_or_create_session(session_id, messages, instance_id)`（[session_manager.py](../dressage/proxy/session_manager.py#L322-L345)）

- **输入**：`session_id`（可空）、`messages`（当前未使用，预留）、`instance_id`（可空）
- **输出**：`tuple[Session, bool]`——session 对象 + "是否新建"标志
- **调用方**：`POST /v1/chat/completions` 处理函数的开头（[server.py](../dressage/proxy/server.py#L1440)）

逻辑（按顺序）：

1. **先顺路清理**：调用 `_cleanup_expired_locked()`（懒清理触发点之一）
2. **墓碑检查**：`session_id` 在 `_finalized_session_ids` 中 → 抛 `SessionFinalizedError` → 端点转成 **HTTP 409**（已终结的 session 禁止复活）
3. **复用检查**：`session_id` 在 `_sessions` 中 → 刷新 `last_active`，若调用方给了新 `instance_id` 则覆盖，返回 `(session, False)`
4. **真正创建**：`sid = session_id or uuid4()`，`iid = instance_id or uuid4()`，存入 `_sessions`，返回 `(session, True)`

新建 Session 的初始状态（[Session dataclass](../dressage/proxy/session_manager.py#L149-L214)）：

| 字段 | 初始值 | 生命周期中的作用 |
|---|---|---|
| `session_id` / `instance_id` | 传入或 uuid4 | 身份；`instance_id` 是训练侧 prompt-equal 聚合键 |
| `steps` | `[]` | 逐步累积 `StepRecord`；finalize 时若为空直接 400 |
| `created_at` / `last_active` | `time.time()` | `last_active` 是空闲超时的判据 |
| `request_lock` | `asyncio.Lock` | 单 session 串行化 + 清理保护（见阶段四） |
| `turn_mode` | `None` | 首次 `resolve_turn_id` 时确定为 `"implicit"` 或 `"explicit"`，**不可逆** |
| `rollout_epoch` | `None` | 首个生成响应到达时锁定（staleness 校验基准） |
| `lineages` / `steps_by_id` / `prefix_tree` | 空 | route 判定与 TITO 的索引结构 |

**阶段二：活跃**

活跃期 = session 存在于 `_sessions` 中，接受 `/v1/chat/completions` 请求。每个请求在端点内的完整路径：

```text
get_or_create_session           → 拿到/创建 session（刷新 last_active）
async with session.request_lock → 串行化同一 session 的并发请求
  ensure_session_active         → 二次确认没被并发 finalize 掉（否则 409）
  resolve_turn_id               → 推进/校验 turn（刷新 last_active）
  ...路由判定、TITO 构建、SGLang 生成...
  record_step                   → 追加 StepRecord（刷新 last_active）
```

三个关键方法：

- `ensure_session_active(session_id, session)`（[L347-L354](../dressage/proxy/session_manager.py#L347-L354)）：再次清理后检查 ① 是否已被 finalize；② `_sessions[session_id] is session`（**身份比较**，防止"同名新 session 顶替旧对象"的竞态）。失败 → `SessionFinalizedError` → 409。
- `resolve_turn_id(session_id, requested_turn_id)`（[L360-L406](../dressage/proxy/session_manager.py#L360-L406)）：输入 `X-Turn-Id`（可空），输出生效的 turn_id。首次调用决定 turn 模式：不带 → **implicit**（生成 `implicit-<hex>` 作为整 session 唯一 turn）；带了 → **explicit**。模式确定后不可混用；explicit 只允许**单调前进**，回到旧 turn_id → 400；`implicit-` 是保留前缀。
- `record_step(...)`（[L408-L542](../dressage/proxy/session_manager.py#L408-L542)）：输入约 40 个字段，输出 `StepRecord | None`。分配 `step_id` → 按需创建 lineage → 追加 `session.steps`、登记 `steps_by_id`、插入 `prefix_tree`、更新 lineage 的 `latest_step_id`、刷新 `last_active`。**降级路径**：session 已不存在时不抛异常，返回 `None` 并打 WARNING（`"dropping trajectory step ... stale-cleanup race"`）——从"静默丢步"修复来的 fail-visible 行为，见 [test_session_manager_cleanup.py](../tests/test_session_manager_cleanup.py#L74-L102)。

活跃期还会发生 `mark_history_rewritten`（检测到历史改写时打标，进入 finalize 结果的 `history_rewritten` 字段）。

**阶段三：Finalize（终结）**

分两层：端点层做编排，manager 层做状态迁移。

端点层 `POST /session/finalize`（[server.py](../dressage/proxy/server.py#L1956-L2087)）：

- **输入**：`{"session_id": 必填, "instance_id": 可选, "label": 可选}`
- **输出**：`finalization_result` dict（`success`、`num_steps`、`num_turns`、`num_segments`、`history_rewritten`、token build 配置等）
- **逻辑顺序**：

```text
1. get_finalization_result(session_id)         ← 幂等短路：已 finalize 过就直接返回缓存
2. session = get_session(session_id)           ← 不存在 → 404
3. async with session.request_lock:            ← 等在途生成跑完；与 chat_completions 互斥
4.   再查一次缓存结果                            ← 双检查：等在锁上的并发重复 finalize 直接拿缓存
5.   再确认 get_session(session_id) is session  ← 防锁等待期间被替换
6.   if not session.steps → 400                ← 空 session 不可 finalize
7.   切段 + 构建 records：                      ← tito 模式写 lineage + timeline 两种 view；
        每条 record 的 extra_info 打上             snapshot 模式只写 timeline
        finalization_complete=True + finalization_id
8.   trajectory_store.write_many(records)      ← 先发布数据（全量校验 + 单锁原子提交）
9.   session_manager.finalize_session(...)     ← 后摘除 session（顺序不能反）
```

第 8、9 步的顺序是原子性保证的核心：**轨迹数据先完整落进 [TrajectoryStore](../dressage/proxy/trajectory_store.py)，活跃 Session 才被移除**，反之会存在"session 已终结但数据没存上"的窗口期。

Manager 层 `finalize_session(session_id, result)`（[L568-L580](../dressage/proxy/session_manager.py#L568-L580)）做三件事：`_sessions.pop(session_id)`、`_finalized_session_ids[session_id] = time.time()`、缓存 `result`。

Finalize 后的行为：

| 后续操作 | 结果 |
|---|---|
| 同名 session_id 再发 `/v1/chat/completions` | 409 `SessionFinalizedError`（墓碑拦截） |
| 重复 `/session/finalize` | **200 + 缓存的 finalization_result**（幂等，不重复写轨迹） |
| `/trajectory/read` | 正常读——数据已在 TrajectoryStore，与 Session 存活无关 |

**阶段四：过期清理（session_timeout 的完整机制）**

清理逻辑本体 `_cleanup_expired_locked()`（[L587-L609](../dressage/proxy/session_manager.py#L587-L609)）：无输入，持有 `self._lock` 调用，就地删除。两条独立路径：

```text
路径 A（活跃 session）：
  if now - session.last_active <= session_timeout: 保留
  elif session.request_lock.locked():              保留，且 last_active = now   ← 关键保护
  else:                                            删除

路径 B（finalized 墓碑）：
  if now - finalized_at > session_timeout:
    删除墓碑 + 删除 _finalization_results 里的缓存结果
```

懒清理的 4 个触发点（无后台线程，顺路执行）：

| 触发点 | 场景 |
|---|---|
| `get_or_create_session` | 每次 `/v1/chat/completions` |
| `ensure_session_active` | 每次 `/v1/chat/completions`（锁内二次确认时） |
| `get_finalization_result` | 每次 `/session/finalize` |
| `active_count` | 每次 `/health`（**健康检查也有清理副作用**） |

`request_lock` 保护的设计动机（[测试 docstring](../tests/test_session_manager_cleanup.py#L1-L9) 记录了这段历史）：早期版本纯按墙钟清理——慢 SGLang worker 或大批量生成超过 3200 秒后，**请求还在飞，session 却被清了**，随后 `record_step` 静默丢弃整个 step。修复后的语义（三个测试固化）：

1. 锁被持有 + 超时 → **不清理**，并把 `last_active` 刷成 now（锁释放后重获一个完整空闲窗口）
2. 超时 + 无锁 → 正常清理
3. finalized 墓碑**不适用**锁保护（它不携带锁），超期即删

两个容易混淆的超时参数：

| 参数 | 默认值 | 作用对象 |
|---|---|---|
| `--session-timeout` | **3200s** | SessionManager：活跃 session 空闲超时 + finalized 墓碑保留期 |
| `--group-timeout` | 300s | TrajectoryStore：`read_batch` 按 instance 成组时的兜底超时 |

推论：finalized 记录超期被清后，**同名 session_id 可重新创建新 session**（409 保护失效）——墓碑只是覆盖"finalize 后短期内防误用"的窗口，不无限占用内存。

**附：finalize 之后数据的去向**

Session 死了但数据活着——两层存储分离的意义。`TrajectoryStore` 侧有自己的生命周期：rollout worker 用 `pop_trajectory(drain=True)` 精确取走并删除某条轨迹的段（fully async 必须 drain，否则长跑 rollout 撑爆 proxy 内存）；`read_batch` 按 instance 成组排出（`min_group_size` 或 `group_timeout` 触发），排出即删。`finalization_id` + `finalization_complete` 标记让 `write_many` 能校验整批原子性。

**一句话总结**：Session 生命周期 = `get_or_create` 懒创建 → `request_lock` 串行的活跃记步 → finalize 时"先发布轨迹、后摘除 session"的原子终结 → 懒清理双路径（活跃看 `last_active`+锁保护，墓碑看终结时间戳），`session_timeout` 同时扮演空闲超时和墓碑保留期两个角色。

</details>

2. `StepRecord` 和 `TrajectorySegment` 为什么要分两层？各自在什么时刻产生？
3. Segment 边界（segment boundary）的触发条件有哪些？（history rewrite / tools 变化 / TITO 前缀不匹配 / TITO 失败）
4. Route 判定（`create` vs `append`）的逻辑是什么？`lineage_id` 和 `route_base_step_id` 起什么作用？
5. `lineage` 和 `timeline` 两种 segment_view 有什么区别？为什么 tito 模式 finalize 时两种都写？
6. `loss_mask` 具体怎么算出来的？哪些 token 可训练、哪些被 mask？（reasoning、tool 结果、prompt 部分分别怎么处理）
7. TITO 模式下 `concat_*` 系列字段是怎么逐步拼出来的？增量 tokenize 如何避免漂移？
8. Snapshot 模式和 TITO 模式在 token 构建上的本质差异是什么？各自适合什么场景？
9. `extra_info` 里实际会出现哪些 key？（`finalization_id`、`segment_view`、`token_build_mode`、`mask_nonlast_version_tokens` 等）
10. `full_versions` 记录的是什么？prompt token 的版本标记为什么和 response 不同（`_INPUT_TOKEN_VERSION`）？
11. prompt 部分的 logprob 从哪来？`logprob_start_len=-1`（tito）和 `0`（snapshot）的差别是什么？
12. `read_batch` 的成组逻辑是什么？`min_group_size` 和 `group_timeout` 怎么影响出批？
13. `pop_trajectory`（drain）和 `read_trajectory` 的区别？为什么 fully async 必须用 drain？

### 二、请求处理与上下文安全

> 涉及代码：[server.py](../dressage/proxy/server.py)、[generation_controller.py](../dressage/proxy/generation_controller.py)

14. `max_new_tokens` 的决策链是什么？（请求 `max_tokens` vs `--max-output-tokens` vs 动态 context 余量，三者怎么取 min）
15. `context_overflow` 在 input 阶段和 output 阶段的处理有何不同？（一个 413 不记步，一个记步后 413）
16. 权重版本校验有哪些层级？（`X-Dressage-Expected-Version`、跨版本轨迹拒绝、partial 模式的 version span 上限）
17. `GenerationController` 的 pause/resume 状态机怎么工作？被抢占的生成如何保留部分输出并续写？
18. tool call 解析和 reasoning 解析的三种后端（`local` / `sglang_api` / `hybrid`）各自怎么工作？
19. 流式响应为什么是"伪流式"？usage chunk 的顺序有什么讲究？

### 三、Rollout 调度与样本展开

> 涉及代码：[sync_rollout.py](../dressage/rollout/sync_rollout.py)、[fully_async_rollout.py](../dressage/rollout/fully_async_rollout.py)、[partial_async_rollout.py](../dressage/rollout/partial_async_rollout.py)、[multi_segment.py](../dressage/rollout/multi_segment.py)、[staleness.py](../dressage/rollout/staleness.py)、[prewarm/](../dressage/rollout/prewarm/)、[data_source.py](../dressage/rollout/data_source.py)

20. 三种调度模式（sync / fully async / partial async）在实现上的核心差异是什么？各自怎么和 slime 的训练循环对接？
21. generate 函数的输入输出契约是什么？（输入 `list[Sample]` + args，输出 `list[Sample]`？）
22. 一个 prompt 的 `n_samples_per_prompt` 是怎么展开成多条轨迹的？`group_index` 和 `instance_id` 怎么分配？
23. `expand_segments_to_samples` 具体做什么？anchor 段为什么 `reward=None` 而非 0？
24. 失败/中断的样本怎么处理？（`mark_aborted_no_grad`、`remove_sample=True`、`_NONE_GROUP` sentinel 的作用）
25. staleness（陈旧性）怎么定义？异步模式下 rollout 用了旧版本权重怎么办？
26. prewarm 沙箱预热机制怎么工作？`DRESSAGE_SANDBOX_PREWARM_AHEAD` 的单位为什么是 prompt group？
27. rollout buffer 和 `round_number` 是什么？什么场景下样本会跨训练步复用？
28. 数据源 JSONL 一行有哪些字段？`metadata` 里支持哪些 key？（`instance_id`、`reward_fn`、`blackbox_execute_cmds`…）
29. 日志指标有哪些？（`segments_per_trajectory_*`、`raw_reward_trajectory_mean` 等怎么算出来的）

### 四、Reward 与训练数据转换

> 涉及代码：[dressage/reward/](../dressage/reward/)、[reward_post_process.py](../dressage/training/reward_post_process.py)、[convert_samples.py](../dressage/rollout/convert_samples.py)、[mopd.py](../dressage/rollout/mopd.py)

30. reward 注册表（`@register_reward`）怎么工作？`sample.metadata["reward_fn"]` 怎么分发？
31. `reward_post_process` 做了什么？GRPO 归一化和兄弟段 advantage 广播的具体公式？
32. prompt-equal 分母（`M_P × N_P / gbs`）为什么要替代 slime 默认的 trajectory-equal 分母？多段场景下不分平会怎样？
33. `train_data` 进入 slime 之后怎么被消费？DP rank 间怎么分片、`rollout_id` 在 packing 时起什么作用？
34. `rollout_log_probs` 在训练侧怎么用？（off-policy 校正在 loss 里怎么体现）
35. MOPD 是什么？`teacher_log_probs` 和 `prompt` 槽位的 teacher ids 分别是什么机制？
36. R3 的 `routed_experts` base64 编解码格式是什么？（int32 → `[-1, num_layers, moe_router_topk]`，chunks/parts 三种形态怎么拼）

### 五、Blackbox 与沙箱

> 涉及代码：[blackbox_server/](../blackbox_server/)、[dressage/paddock/](../dressage/paddock/)、[dressage/sandbox/](../dressage/sandbox/)

37. BlackboxServer 是什么？它和 Proxy 的关系、各自职责边界在哪？
38. paddock 的五个生命周期操作（init / register_agent / execute_cmd / call_agent / terminate）各自做什么？
39. 黑盒模式下，agent 进程发出的 LLM 请求怎么被路由回 Dressage Proxy？（header 注入、进程内 LLM proxy 转发链）
40. `rollout_llm_proxy` 为什么要把 OpenAI Responses API 桥接成 Chat Completions？
41. 沙箱有哪几种实现（e2b / local_bwrap）？slot 池、租约（lease）怎么管理？
42. opencode / openclaw / claude_code / codex 四个适配器的差异和适配层结构？

### 六、训练侧与权重更新

> 涉及代码：[dressage/training/](../dressage/training/)、[slime/](../slime/)

43. 训练入口有哪些？（`slime.train` / `train_async` / `train_async_with_rollout_pause` 各自配哪种调度模式）
44. 权重更新怎么触发、rollout 侧怎么感知？（pause → 权重同步 → resume 的完整时序，KV cache flush）
45. 异步训练时 staleness 校验在哪几层做？（proxy 侧 epoch 校验、版本 span、mask 非最新版本 token）
46. token-level policy gradient loss 实际怎么算？`rollout_mask_sums` 在 reducer 里怎么当分母用？

### 七、工程与部署

> 涉及代码：[pyproject.toml](../pyproject.toml)、[docker/](../docker/)、[dressage/integrations/harbor/](../dressage/integrations/harbor/)

47. 项目的 CLI 入口有哪些？（`dressage-proxy` 等，pyproject 里注册了哪些 entry points）
48. 配置体系是怎样的？（环境变量 + Pydantic + Harbor YAML 三层各自的职责）
49. Docker 镜像怎么构建？`build.sh` / `run.sh` / `image_tag.sh` 的流程？
50. Harbor 集成是做什么的？`dressage/integrations/harbor/` 解决什么问题？
51. 错误处理体系怎么分层？（`errors.py` 的异常分类、HTTP 状态码语义、监控指标）

### 八、数据流全景（串起来的问题）

52. 一个 token 从 agent 的文本消息到训练张量的完整路径是什么？（messages → chat template 渲染 → tokenize → SGLang → 记录 → 段 → Sample → train_data → megatron batch）
53. 一个 rollout batch 的端到端时序是怎样的？（数据源取 prompt → 调度 → agent 执行 → finalize → reward → convert → 训练 → 权重更新 → 下一轮）
54. 多段轨迹在整个链路上的身份标识怎么传递？（`session_id` → `trajectory_id` → `parent_traj_id` → `rollout_id` 的对应关系）

---

## 相关文档索引

- 用户文档：[docs/proxy.md](../docs/proxy.md)、[docs/rollout.md](../docs/rollout.md)、[docs/training.md](../docs/training.md)、[docs/paddock.md](../docs/paddock.md)、[docs/blackbox-server.md](../docs/blackbox-server.md)、[docs/sandbox.md](../docs/sandbox.md)
- 设计稿：[multi-segment-design.md](./multi-segment-design.md)、[partial-rollout-staleness-design.md](./partial-rollout-staleness-design.md)、[async-rollout-staleness-design.md](./async-rollout-staleness-design.md)、[rollout-routing-replay-design.md](./rollout-routing-replay-design.md)、[blackbox-async-turn-design.md](./blackbox-async-turn-design.md)
- 导览：[GETTING_STARTED.md](./GETTING_STARTED.md)、[dressage-architecture-study-guide-zh.md](./dressage-architecture-study-guide-zh.md)
