# BlackboxServer 异步 Turn 模型设计文档

> 对应 commit: `77f7f6046657afa97d9b83f7835bf7bba66f51b3`
> 关联 issue: #14

## 1. 背景与目标

### 1.1 旧模型的问题

旧 `POST /v1/sessions/{id}/messages` 采用**请求绑定同步**模型：HTTP 请求持有 `session_lock`，阻塞等待后端 agent 返回。agent turn 可能持续数分钟，期间存在以下问题：

- HTTP 连接被占死，无法释放
- 无法取消正在执行的 turn
- 网络中断即丢失全部进度
- 超时后 turn 卡在 INFLIGHT，session 进入 DESYNCED

### 1.2 新模型的目标

将同步调用拆分为 **submit + long-poll** 两阶段：

- `POST /messages?mode=async` → 立即返回 202，后台异步执行
- `GET /turns/{turn_id}?wait=N` → 长轮询获取结果
- `POST /turns/{turn_id}/cancel` → 取消正在执行的 turn
- `mode=sync` 保持完全向后兼容

## 2. Turn 状态机

### 2.1 状态定义

代码位置：[`models.py` L35-L41](../blackbox_server/core/models.py#L35-L41)

```
旧状态机:  inflight → {committed | unknown}

新状态机:  queued → inflight → {committed | failed | cancelled | unknown}
```

| 状态 | 含义 | 是否终态 |
|------|------|----------|
| `queued` | 已接收，未开始执行 | 否（活跃态） |
| `inflight` | 后端调用进行中 | 否（活跃态） |
| `committed` | 成功完成 | 是 |
| `failed` | 确定性失败 | 是 |
| `cancelled` | 被取消 | 是 |
| `unknown` | 结果未知（超时/进程错误），session → DESYNCED | 是 |

活跃态与终态的分类定义在 [`server.py` L87-L96](../blackbox_server/core/server.py#L87-L96)：

```python
_ACTIVE_TURN_STATUSES = frozenset({TurnStatus.QUEUED, TurnStatus.INFLIGHT})
_TERMINAL_TURN_STATUSES = frozenset({TurnStatus.COMMITTED, TurnStatus.FAILED, TurnStatus.CANCELLED, TurnStatus.UNKNOWN})
_MAX_TURN_WAIT_SECONDS = 60.0
```

### 2.2 状态流转规则

- `submit_turn` 创建记录时状态为 `queued`
- `_run_turn` 后台任务启动后将 `queued → inflight`
- 后端调用成功 → `committed`
- 后端调用异常（超时/溢出/步数超限/通信失败/未预期错误）→ `unknown`，session → `desynced`
- 取消 queued turn → 直接 `cancelled`
- 取消 inflight turn → `task.cancel()` 触发 `CancelledError` → `cancelled`
- abort session → 所有活跃态 turn → `unknown`
- graceful shutdown → 所有活跃态 turn → `unknown`

## 3. 时序图

### 3.1 异步模式完整流程（mode=async）

```mermaid
sequenceDiagram
    participant C as Client<br/>(BlackboxServerClient)
    participant A as API Route<br/>(sessions.py)
    participant S as BlackboxServer<br/>(server.py)
    participant T as _run_turn<br/>(background task)
    participant AD as Adapter<br/>(BackendAdapter)

    Note over C,S: Phase 1: Submit（持有 session_lock）

    C->>A: POST /messages {mode: "async", turn_id, messages}
    A->>S: submit_turn(session_id, request)
    S->>S: 校验 session 状态 / turn_id 幂等 / 活跃 turn 约束
    S->>S: 创建 TurnRecord(status=QUEUED)
    S->>S: 创建 asyncio.Event
    S->>T: asyncio.create_task(_run_turn)
    S-->>A: TurnSubmission(status=QUEUED)
    A-->>C: 202 Accepted {turn_id, status}

    Note over T,AD: Phase 2: 后台执行（不持有 session_lock）

    T->>T: 短暂持锁: QUEUED → INFLIGHT
    T->>AD: adapter.send_message(session, turn_context, messages)
    AD-->>T: AdapterResponse
    T->>T: 短暂持锁: 写入 COMMITTED + response
    T->>T: event.set()（唤醒等待者）

    Note over C,S: Phase 3: 长轮询

    C->>A: GET /turns/{turn_id}?wait=30
    A->>S: get_turn(session_id, turn_id, wait=30)
    S->>S: wait_for(event, timeout=min(30, 60))
    S-->>A: TurnStatusResponse(committed, outputs, usage)
    A-->>C: 200 {status: "committed", outputs, usage, backend}
```

> 如果 turn 在 `wait` 窗口内未完成，服务端返回当前快照（`queued`/`inflight`），客户端继续轮询。服务端**不返回 504**。

### 3.2 同步模式（mode=sync，向后兼容 facade）

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API Route
    participant S as BlackboxServer
    participant T as _run_turn
    participant AD as Adapter

    C->>A: POST /messages {mode: "sync", messages}
    A->>S: send_message(session_id, request)
    S->>S: submit_turn() → 创建 QUEUED + Event + spawn _run_turn
    S->>S: event.wait()（阻塞直到 turn 完成）

    T->>T: QUEUED → INFLIGHT
    T->>AD: adapter.send_message()
    AD-->>T: AdapterResponse
    T->>T: commit → COMMITTED
    T->>T: event.set()（唤醒 send_message）

    S->>S: 读取 turn record
    S->>S: _build_committed_message_response()
    S-->>A: MessageResponse（与旧格式完全一致）
    A-->>C: 200 {outputs, usage, backend}
```

> `send_message` 是 `submit_turn` + `event.wait()` + 响应重建的组合。失败时通过 `_raise_from_turn_error()` 从 turn record 的 `error.http_status` 还原原始 HTTP 错误码（504/413/429/502/500），保证 sync 模式行为与旧实现完全一致。

### 3.3 取消流程

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API Route
    participant S as BlackboxServer
    participant T as _run_turn

    alt Turn 处于 QUEUED（同步取消）
        C->>A: POST /turns/{turn_id}/cancel
        A->>S: cancel_turn(session_id, turn_id)
        S->>S: 持锁: status → CANCELLED
        S->>S: event.set()
        S-->>A: TurnCancelResponse(status: "cancelled")
        A-->>C: 200 {status: "cancelled"}
    else Turn 处于 INFLIGHT（best-effort 取消）
        C->>A: POST /turns/{turn_id}/cancel
        A->>S: cancel_turn(session_id, turn_id)
        S->>S: 持锁: 设置 cancel_inflight 标志
        S->>T: task.cancel()
        S->>S: adapter.abort_session(session)
        S-->>A: TurnCancelResponse(status: "cancel_requested")
        A-->>C: 200 {status: "cancel_requested"}

        T->>T: 捕获 CancelledError
        T->>T: _finalize_interrupted_turn_locked()
        T->>T: status → CANCELLED + event.set()
    else Turn 已终态（幂等）
        C->>A: POST /turns/{turn_id}/cancel
        A->>S: cancel_turn(session_id, turn_id)
        S-->>A: TurnCancelResponse(status: 当前终态)
        A-->>C: 200 {status: "committed/failed/..."}
    end
```

### 3.4 异常处理流程

```mermaid
sequenceDiagram
    participant T as _run_turn
    participant AD as Adapter
    participant S as BlackboxServer
    participant C as Client

    T->>AD: adapter.send_message()
    AD-->>T: 抛出异常

    alt asyncio.CancelledError
        T->>S: 持锁: _finalize_interrupted_turn_locked()
        S->>S: status → CANCELLED
    else TimeoutError / Overflow / Steps / Transport / 未知异常
        T->>T: 打包 outcome = (error_type, http_status)
        T->>S: 持锁: _handle_unknown_turn()
        S->>S: status → UNKNOWN, session → DESYNCED
        S->>S: 记录 error.http_status (504/413/429/502/500)
    end

    T->>T: event.set()

    alt sync 模式（send_message 阻塞中）
        C->>S: event.wait() 被唤醒
        S->>S: _raise_from_turn_error(http_status)
        S-->>C: ApiError(对应 HTTP 状态码)
    else async 模式（client 轮询中）
        C->>S: GET /turns/{turn_id}
        S-->>C: TurnStatusResponse(status: "unknown", error)
        C->>C: _raise_turn_error(http_status)
        C->>C: 抛出 httpx.HTTPStatusError
    end
```

> 关键设计：所有异常的 `http_status` 被保存在 turn record 的 `error` 字典中，使 sync facade 和 polling client 都能还原完全一致的 HTTP 错误码。

### 3.5 Abort / Shutdown 联动

```mermaid
sequenceDiagram
    participant C as Client
    participant S as BlackboxServer
    participant T as _run_turn

    Note over C,S: Abort Session
    C->>S: POST /sessions/{id}/abort
    S->>T: active_turn_task.cancel()
    S->>S: 遍历 turn_ledger: 活跃态 → UNKNOWN
    S->>S: 对每个被 settle 的 turn: event.set()
    S-->>C: AbortResponse(state: "aborted")

    Note over S: Graceful Shutdown
    S->>S: _terminate_all_active_turns()
    S->>T: 对所有活跃 task: task.cancel()
    S->>S: asyncio.gather(等待所有 task 完成)
    S->>S: 每个 task 的 CancelledError → _finalize_interrupted_turn_locked
    S->>S: shutdown_started=True → status=UNKNOWN (非 CANCELLED)
```

## 4. 调用关系图

### 4.1 服务端内部调用关系

```mermaid
graph TB
    subgraph API["API Layer (sessions.py)"]
        R1["POST /messages"]
        R2["GET /turns/:id"]
        R3["POST /turns/:id/cancel"]
        R4["POST /sessions/:id/abort"]
    end

    subgraph Core["Server Core (server.py)"]
        ST["submit_turn()"]
        SM["send_message()"]
        RT["_run_turn()"]
        GT["get_turn()"]
        CT["cancel_turn()"]
        AS["abort_session()"]
        GS["graceful_shutdown()"]

        FIT["_finalize_interrupted_turn_locked()"]
        BCM["_build_committed_message_response()"]
        RFE["_raise_from_turn_error()"]
        BTS["_build_turn_status_response()"]
        HUT["_handle_unknown_turn()"]
        ATI["_active_turn_id()"]
        TAT["_terminate_all_active_turns()"]
    end

    subgraph Adapter["Backend Adapter"]
        AD_SM["send_message()"]
        AD_AS["abort_session()"]
    end

    R1 -->|mode=async| ST
    R1 -->|mode=sync| SM
    R2 --> GT
    R3 --> CT
    R4 --> AS

    SM --> ST
    SM --> BCM
    SM --> RFE
    ST --> ATI
    ST -->|create_task| RT
    RT --> AD_SM
    RT -->|on cancel| FIT
    RT -->|on error| HUT
    GT --> BTS
    CT -->|inflight: task.cancel| RT
    CT -->|inflight| AD_AS
    AS -->|cancel active turn| RT
    AS -->|settle turns| ATI
    GS --> TAT
    TAT -->|cancel all| RT

    RT -.->|event.set| SM
    RT -.->|event.set| GT
    CT -.->|event.set| GT
    AS -.->|event.set| SM
```

> 实线 `-->` 表示直接方法调用；虚线 `-.->` 表示通过 `asyncio.Event` 的异步唤醒（非直接调用）。

### 4.2 客户端调用关系

```mermaid
graph TB
    subgraph Client["Client (client.py)"]
        CA["call_agent()"]
        RM["_resolve_call_mode()"]
        PT["_poll_turn()"]
        CP["_committed_call_payload()"]
        RTE["_raise_turn_error()"]
        POST["_post_agent_with_retry()"]
        GET["_get_turn_with_retry()"]
    end

    subgraph Retry["HTTP Retry (http_retry.py)"]
        PJ["post_json_with_retry()"]
        GJ["get_json_with_retry()"]
    end

    CA --> RM
    CA --> POST
    CA -->|async mode| PT
    PT --> GET
    PT -->|on committed| CP
    PT -->|on error| RTE
    POST --> PJ
    GET --> GJ
```

### 4.3 完整数据流

```mermaid
graph LR
    subgraph Client["Client (client.py)"]
        CA["call_agent()"]
    end

    subgraph API["API (sessions.py)"]
        R1["POST /messages"]
        R2["GET /turns/:id"]
    end

    subgraph Server["Server (server.py)"]
        ST["submit_turn()"]
        RT["_run_turn()"]
        GT["get_turn()"]
        EV["asyncio.Event"]
    end

    subgraph Adapter["Adapter"]
        AD["adapter.send_message()"]
    end

    CA -->|1. POST mode=async| R1
    R1 --> ST
    ST -->|2. create_task| RT
    ST -->|3. 202| R1
    RT -->|4. call| AD
    RT -->|5. event.set| EV
    CA -->|6. GET wait=30| R2
    R2 --> GT
    GT -->|7. wait event| EV
    GT -->|8. return snapshot| R2
    R2 -->|9. 200| CA
```

## 5. 关键设计决策

### 5.1 锁纪律：只在准入和提交时短暂持锁

`session_lock` 的持有窗口被严格控制：

| 阶段 | 持锁 | 耗时 |
|------|------|------|
| `submit_turn` 准入 | 是 | 毫秒级（校验 + 创建记录 + spawn task） |
| `_run_turn` 状态流转 queued→inflight | 是 | 毫秒级 |
| `adapter.send_message` 后端调用 | **否** | 可能数分钟 |
| `_run_turn` 提交结果 | 是 | 毫秒级 |
| `get_turn` 读取快照 | 是 | 毫秒级 |

代码位置：
- `submit_turn` 持锁范围：[`server.py` L447-L531](../blackbox_server/core/server.py#L447-L531)
- `_run_turn` 持锁范围：[`server.py` L590-L595](../blackbox_server/core/server.py#L590-L595)（流转）和 [`server.py` L674-L702](../blackbox_server/core/server.py#L674-L702)（提交）
- 后端调用在锁外：[`server.py` L606-L609](../blackbox_server/core/server.py#L606-L609)

### 5.2 向后兼容：sync facade 重建旧错误语义

`send_message`（[`server.py` L533-L574](../blackbox_server/core/server.py#L533-L574)）复用 `submit_turn` + `event.wait()`，然后从 turn record 重建旧格式响应：

- 成功：`_build_committed_message_response()`（[`server.py` L1004-L1027](../blackbox_server/core/server.py#L1004-L1027)）返回 `MessageResponse`
- 失败：`_raise_from_turn_error()`（[`server.py` L1029-L1039](../blackbox_server/core/server.py#L1029-L1039)）从 `error["http_status"]` 还原 `ApiError`

这保证了 `mode=sync` 的 HTTP 状态码和响应体与旧实现完全一致。

### 5.3 幂等性升级：replay 代替 409

旧模型中 INFLIGHT turn 的重试直接返回 409。新模型（[`server.py` L481-L491](../blackbox_server/core/server.py#L481-L491)）：

```python
# Same turn_id + same fingerprint → idempotent replay
return TurnSubmission(
    turn_id=effective_turn_id,
    status=existing.status,      # committed / queued / inflight
    idempotent_replay=True,
    instance_id=self._response_instance_id(),
)
```

- `committed` → replay 缓存结果
- `queued` / `inflight` → 重新附着到同一执行，不启动第二次 agent 调用
- 不同 fingerprint → 仍然 409

客户端（[`client.py` L111-L112](../dressage/paddock/blackbox/client.py#L111-L112)）生成稳定 `turn_id`，跨重试复用，使 submit POST 本身也幂等。

### 5.4 Event 驱动的等待者唤醒

每个 turn 对应一个 `asyncio.Event`（[`server.py` L521](../blackbox_server/core/server.py#L521)），在以下场景被 `set()`：

| 触发者 | 代码位置 | 唤醒的等待者 |
|--------|----------|-------------|
| `_run_turn` finally 块 | [`server.py` L704-L706](../blackbox_server/core/server.py#L704-L706) | `send_message` + `get_turn` |
| `cancel_turn` (queued) | [`server.py` L786-L788](../blackbox_server/core/server.py#L786-L788) | `send_message` + `get_turn` |
| `abort_session` | [`server.py` L969-L971](../blackbox_server/core/server.py#L969-L971) | `send_message` + `get_turn` |

### 5.5 异常全捕获：turn 不卡 INFLIGHT

`_run_turn` 的异常处理（[`server.py` L604-L672](../blackbox_server/core/server.py#L604-L672)）覆盖所有路径：

| 异常类型 | outcome | http_status | session 状态 |
|----------|---------|-------------|-------------|
| `asyncio.CancelledError` | `_finalize_interrupted_turn_locked` | 499 | CANCELLED |
| `asyncio.TimeoutError` | unknown | 504 | DESYNCED |
| `BackendContextOverflowError` | unknown | 413 | DESYNCED |
| `BackendMaxStepsExceededError` | unknown | 429 | DESYNCED |
| `BackendTransportError` / `Protocol` / `Process` | unknown | 502 | DESYNCED |
| `Exception`（未预期） | unknown | 500 | DESYNCED |

### 5.6 单活跃 turn 约束

每个 session 最多一个活跃 turn（[`server.py` L493-L506](../blackbox_server/core/server.py#L493-L506)）：

- `submit_turn` 检查 `_active_turn_id(session)`，有活跃 turn 时返回 409
- `execute_cmd` 也检查活跃 turn（[`server.py` L856-L863](../blackbox_server/core/server.py#L856-L863)），有活跃 turn 时返回 409

## 6. 环境变量与配置

### 6.1 服务端

| 环境变量 | 说明 |
|----------|------|
| `BBS_BACKEND_TIMEOUT` | 后端调用超时（秒），是 turn 执行的最终兜底 |

### 6.2 客户端

代码位置：[`client.py` L29-L37](../dressage/paddock/blackbox/client.py#L29-L37)

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DRESSAGE_BLACKBOX_AGENT_CALL_MODE` | `async` | 调用模式：`async`（submit + poll）或 `sync`（请求绑定） |
| `DRESSAGE_BLACKBOX_AGENT_POLL_WAIT_SEC` | `30` | 单次长轮询 wait 秒数（服务端 clamp 到 60） |
| `DRESSAGE_BLACKBOX_AGENT_POLL_TOTAL_TIMEOUT_SEC` | `0` | 客户端总轮询预算（0 = 无限，由服务端 `backend_timeout` 兜底） |

### 6.3 新增 HTTP API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/sessions/{id}/messages` | `mode=async` → 202 + TurnSubmitResponse；`mode=sync` → 200 + MessageResponse |
| `GET` | `/v1/sessions/{id}/turns/{turn_id}` | 长轮询，`?wait=<seconds>`（clamp 60s），返回 TurnStatusResponse |
| `POST` | `/v1/sessions/{id}/turns/{turn_id}/cancel` | 取消 turn，返回 TurnCancelResponse |

## 7. 涉及文件清单

| 文件 | 变更行数 | 职责 |
|------|----------|------|
| [`blackbox_server/core/server.py`](../blackbox_server/core/server.py) | +591/-151 | 核心重构：submit_turn / _run_turn / send_message / get_turn / cancel_turn |
| [`blackbox_server/core/models.py`](../blackbox_server/core/models.py) | +49/-3 | TurnStatus 扩展 6 态 + 3 个新响应模型 + mode 字段 |
| [`blackbox_server/api/sessions.py`](../blackbox_server/api/sessions.py) | +59/-5 | 新增 GET /turns + POST /cancel 路由，messages 路由分支 |
| [`dressage/paddock/blackbox/client.py`](../dressage/paddock/blackbox/client.py) | +194/-15 | call_agent 默认 async + _poll_turn 轮询循环 |
| [`dressage/paddock/blackbox/common/http_retry.py`](../dressage/paddock/blackbox/common/http_retry.py) | +82 | 新增 get_json_with_retry |
| [`dressage/paddock/interface.py`](../dressage/paddock/interface.py) | +1 | call_agent 签名新增 turn_id 参数 |
| [`dressage/paddock/blackbox/paddock.py`](../dressage/paddock/blackbox/paddock.py) | +2 | call_agent 透传 turn_id |
| [`tests/blackbox_server/test_async_turns.py`](../tests/blackbox_server/test_async_turns.py) | +560 | 新建：async submit/poll/cancel 全流程测试 |
| [`tests/blackbox_server/test_server.py`](../tests/blackbox_server/test_server.py) | +59 | UnexpectedErrorAdapter：验证未预期异常不卡 INFLIGHT |
| [`tests/test_new_paddock_layers.py`](../tests/test_new_paddock_layers.py) | +35/-5 | mock 从单次 200 改为 202 + 轮询 200 |
| [`docs/blackbox-server.md`](../docs/blackbox-server.md) | +77/-5 | API 表格 + 流程图 + curl 示例 |
| [`docs/paddock.md`](../docs/paddock.md) | +5/-1 | call_agent 说明 + 环境变量表 |
