# 推理代理（Inference Proxy）

**面向 Agentic RL 的 Token 级轨迹记录**

> 本文是 [docs/proxy.md](../docs/proxy.md) 的中文译本。

[← 返回主 README](../README.md) · [概览](#-概览) · [核心特性](#-核心特性) · [核心模块](#-核心模块) · [会话模型](#-会话与步骤模型) · [构建模式](#-token-构建模式) · [端点](#-http-端点) · [路由回放](#-路由回放r3)

## 📖 概览

Dressage **Proxy** 是一个 OpenAI 兼容的 HTTP 服务，位于 Agent rollout 与 SGLang 推理路由器之间。它是轨迹记录的中枢神经系统——每一次 LLM 调用都要经过它，每一个 token、logprob 和 loss mask 都会被捕获下来用于训练。Proxy 是 Dressage 训练流水线得以成立的前提：没有它，就无法忠实地重建 Agent 在 rollout 期间产生的精确 token 序列、概率分布和决策边界。

> [!IMPORTANT]
> Agent 永远不会直接调用 SGLang。Proxy 在透明转发生成请求的同时，构建出内容丰富、可直接用于训练的轨迹数据。这一设计确保 token 级记录始终生效——无论 Agent 是 Python 白盒循环，还是像 `opencode` 这样的外部 HTTP 黑盒。

```text
Agent（白盒或黑盒）
        │  POST /v1/chat/completions
        │  header/body 中的标识：X-Session-Id, X-SMG-Routing-Key, X-Instance-Id, X-Turn-Id
        ▼
Dressage Proxy
        │  将生成请求转发给 SGLang
        │  逐步记录 token、logprob、loss mask
        │  跟踪权重版本、MoE 路由 ID
        ▼
SGLang Router  →  策略模型（Policy Model）
```

Proxy 以独立 FastAPI 服务运行（CLI：`dressage-proxy`），设计上支持来自多个 rollout worker 的并发会话。每个 session 代表一条完整的 Agent 轨迹，session 内每次调用 `/v1/chat/completions` 都会向轨迹记录追加一个新的 **step**。

## ✨ 核心特性

- **OpenAI 兼容 API** —— 对 `/v1/chat/completions` 的 drop-in 替代。Agent 不需要任何定制集成，只需把 `base_url` 指向 Proxy 即可。支持流式与非流式模式、工具调用，以及所有标准 OpenAI chat completion 参数。
- **逐步记录（Per-Step Recording）** —— 每次 Proxy 调用都会捕获完整的请求消息、prompt/response token ID、逐 token logprob、权重版本戳，以及计算出的 loss mask。这些逐步记录构成了训练数据构建的原材料。
- **TITO 支持** —— 当 `tito` token 构建模式生效时，Proxy 会把增量分词数据记录在 `concat_token_ids`、`concat_response_logprobs`、`concat_response_mask`、`concat_versions` 等字段中。这些字段会在 finalize 阶段被缝合起来，保证任意长度多轮轨迹上的前缀完全一致。详见下文 [TITO 深入剖析](#-tito-深入剖析)。
- **自动分段（Auto Segmentation）** —— Proxy 会自动检测 Agent 是否在轨迹中途重写了对话历史（压缩、摘要）或改变了可用的工具 schema。一旦发生，它会关闭当前 segment 并开启新的 segment，为训练保留干净的 token 边界。每个 segment 都成为一条独立的训练样本。
- **可抢占生成（Preemptible Generation）** —— `GenerationController` 能够响应权重更新信号，在任意 token 边界中止正在进行的 SGLang 生成。已生成的 partial output 会保留在 step 记录中；当 Proxy 以 `--dressage-partial-rollout` 启动时，生成会在 `/v1/rollout/resume` 之后继续。这使得 rollout 可以持续进行，而不必丢弃进行中的计算。
- **权重版本跟踪（Weight Version Tracking）** —— 每个生成的 token 都会被标记上产生它的模型权重版本。当一条轨迹跨越多次权重更新时（partial rollout），`--record-token-versions` 会存储逐 token 的版本，`--mask-nonlast-version-tokens` 则会标记来自旧版本的 token 以便进行选择性 loss masking。
- **路由回放（R3）** —— 对于 Mixture-of-Experts（MoE）模型，Proxy 可通过 `--use-rollout-routing-replay` 捕获每个生成 token 的 routed expert ID。该数据以 base64 编码的 chunk 形式存储，并转发给训练侧用于忠实的 MoE 路由回放。
- **可配置解析器（Configurable Parsers）** —— 工具调用与 reasoning 抽取后端可插拔（`local`、`sglang_api`、`hybrid`）。两类解析器后端均默认为 `sglang_api`；`local` 直接解析模型输出，`hybrid` 优先尝试 SGLang 并在失败时回退到 local。Reasoning 解析器负责为 Qwen3 这类模型抽取 `<think>` 块。
- **版本与上下文安全（Version and Context Safety）** —— 非 partial 轨迹若在中途发生模型权重版本或 rollout epoch 变化，会被拒绝（`trajectory_version_changed`）。Proxy 侧的上下文检查会返回稳定的 `context_overflow` 载荷，并把生成裁剪到剩余上下文预算的精确值。`--max-output-tokens` 是一个可选的、逐请求生效的额外硬上限。

## 🧱 核心模块

Proxy 代码库按职责单一原则拆分为若干聚焦的模块：

 | 模块 | 职责 |
 | :------- | :--------------- |
 | `server.py` | FastAPI 应用 —— chat completions 端点、session finalize、trajectory read。CLI 入口 `dressage-proxy`。负责请求校验、header 提取与响应格式化。 |
 | `session_manager.py` | 逐 session 的 step 管理、turn 跟踪与 history-rewrite 检测。为每个活跃 session 维护有序的 `StepRecord` 列表。当对话消息违反 append-only 契约时进行检测并触发 segment 边界。 |
 | `trajectory_store.py` | 线程安全的内存 segment 存储。finalize 后的 segment 写入此处，rollout 代码可通过 `/trajectory/read` 读回。支持按 session ID 清理。 |
 | `generation_controller.py` | 面向 partial rollout 的可抢占 SGLang 生成。为 SGLang 客户端调用包装 abort/resume 能力。管理生成状态机（idle → generating → paused → resumed）。 |
 | `sglang_client.py` | 低层 SGLang 路由器客户端，带权重版本跟踪。发送生成请求，接收含 token ID 与 logprob 的响应，并记录当时生效的权重版本。 |
 | `tool_call_parser.py` | 从 assistant 响应中抽取模型特定格式的工具调用。支持多种后端模式（`local` 直接解析、`sglang_api` 走 SGLang 原生、`hybrid` 回退链）。当前针对 Qwen3.5 工具调用格式做了优化。 |
 | `reasoning_parser.py` | 针对会产生结构化思考块的模型（如 Qwen3 的 `<think>...</think>` 格式）解析 reasoning 内容。把 reasoning token 与 action token 分离，以支持选择性 loss masking。 |
 | `proxy_client.py` | rollout 代码用于与 Proxy 交互的异步 HTTP 客户端。提供 `chat_completions`、`finalize_session`、`read_trajectory` 的类型化方法。 |
 | `tool_call_ids.py` | 确定性的工具调用 ID 生成。确保工具调用 ID 在重跑之间可复现，这对轨迹一致性很重要。 |
 | `last_step/prompt_assistant_mask.py` | Snapshot 对齐辅助模块。为完整的 step 快照构建 mask。 |

## 📋 会话与步骤模型

一个 **session**（`session_id`）代表一条完整的 Agent 轨迹——从第一条 user 消息直到最后一条 assistant 响应。session 内每次调用 `/v1/chat/completions` 都会向轨迹追加一个新的 **step**。step 是有序的，一旦记录便不可变。

### 每个 Step 记录了什么

每个 step 都会捕获一次 LLM 交互的完整快照：

- **请求消息（Request messages）** —— Agent 发来的完整对话历史
- **Prompt token ID** —— 分词后的输入，附带逐 token logprob（若可用）
- **Response token ID** —— 生成的输出 token，附带逐 token logprob
- **权重版本（Weight versions）** —— 每个 token 由哪个模型权重版本生成
- **Loss mask** —— 标识哪些 token 可训练的二值 mask
- **TITO 字段** —— tito 模式生效时的增量分词数据（`concat_token_ids`、`concat_response_logprobs`、`concat_response_mask` 等）
- **分段标记（Segment markers）** —— 此 step 是否触发了 segment 边界
- **MoE 路由数据** —— 逐 token 的 routed expert ID（启用 R3 时）

### 运行时标识符

每个 `/v1/chat/completions` 请求都必须提供以下标识符，以便正确归属轨迹。优先使用 header，但 `session_id`、`instance_id`、`turn_id` 这几个 body 字段也可作为回退。当 `X-Session-Id` 缺失时，`X-SMG-Routing-Key` 也会被接受作为 session 路由键。

 | Header | 用途 | 示例 |
 | :------- | :-------- | :-------- |
 | `X-Session-Id` | 轨迹键 —— 在训练样本中成为 `parent_traj_id` | `sess-abc123` |
 | `X-SMG-Routing-Key` | 粘性路由代理使用的备用 session 键 | `sess-abc123` |
 | `X-Instance-Id` | Prompt / 任务实例 —— 用于 prompt 等权梯度聚合 | `inst-xyz789` |
 | `X-Turn-Id` | 可选的显式 turn 标识符，用于幂等性跟踪 | `turn-001` |

> [!TIP]
> `X-Instance-Id` header 对 prompt 等权梯度缩放至关重要。来自同一 prompt 实例的所有样本共享同一个梯度分母，从而保证无论每条轨迹产生多少 segment，其贡献都是公平的。

## 🧬 Token 构建模式

Dressage 支持两种模式，把 Proxy 记录的 step 转换为可用于训练的 segment。构建模式的选择从根本上决定了 token 序列如何构造、多轮上下文如何处理。

### TITO 模式（默认）

面向长 Agentic 轨迹的默认且推荐模式。segment 通过在完整多轮上下文上拼接逐 step 的 TITO 片段来组装，保证前缀完全一致。

```text
第 1 轮 → TITO 片段₁（system + user₁）
第 2 轮 → TITO 片段₂（asst₁ + tool₁ + user₂）
第 3 轮 → TITO 片段₃（asst₂ + tool₂ + user₃）
                    ↓
         stitch(片段₁ + 片段₂ + 片段₃) → Segment
```

- 设置 `token_build_model=qwen3_5` 时，在 tito 模式下会推断出 `model_mask_type=qwen3_5`、`model_tool_call_type=qwen3_5`、`model_reasoning_type=qwen3` 与 `tito_model=qwen3_5`
- 最适合**长 Agentic 轨迹**（SWE 任务、编码 Agent、多步推理）
- 避免重分词漂移（retokenization drift）—— Agentic RL 训练中的头号正确性挑战
- 每个片段独立分词，然后拼接 ID（绝不整体重新分词）

### Snapshot 模式

一种更简单的模式：每个 segment 由最后一个 assistant step 的完整消息快照构建。整段对话在 finalize 时从零重新分词。

```text
第 1 轮 → （上下文，不直接使用）
第 2 轮 → （上下文，不直接使用）
第 3 轮 → 完整消息列表快照 → 分词 → Segment
```

- Loss mask 只把 assistant token 标记为可训练
- 最适合**较短的轨迹**，此时重分词漂移可忽略
- 模型支持更通用（无需模型特定的 TITO 模板）
- 由于存在前缀不一致风险，不推荐用于长多轮 rollout

<details>
<summary><b> 配置</b></summary>
<br>

```bash
dressage-proxy \
  --tokenizer-path /path/to/Qwen3.5-4B \
  --token-build-mode tito \
  --token-build-model qwen3_5 \
  --tito-model qwen3_5
```

</details>

## 🧬 TITO 深入剖析

TITO（Token-In-Token-Out）是 Proxy 对重分词漂移问题给出的答案。在标准多轮 LLM 推理中，每轮重新编码完整消息列表，可能对同一段前缀文本产生略有差异的 token ID —— 这会破坏 rollout 时记录的 logprob 与训练时使用的 token 序列之间的对齐关系。

### 问题所在

```text
第 1 轮:  tokenize("system: ... user: Hello")                          → [101, 202, 303]
第 2 轮:  tokenize("system: ... user: Hello assistant: Hi user: How?")  → [101, 202, 304, ...]
                                                                              ↑ 漂移！303 ≠ 304
```

### TITO 如何解决

```text
第 1 轮:  encode("system: ... user: Hello")           → 片段₁ = [101, 202, 303]
第 2 轮:  encode("assistant: Hi user: How?")           → 片段₂ = [405, 506]
         stitch(片段₁ + 片段₂)                          → [101, 202, 303, 405, 506]  ✅ 前缀完好
```

Proxy 把 TITO 数据存放在 `StepRecord` 的以下字段中：
- `concat_token_ids` —— 该 step 拼接后的上下文与响应 token ID
- `concat_response_logprobs` —— 逐 token logprob，上下文位置以 `0.0` 填充
- `concat_response_mask` —— loss mask，上下文位置置 `0`，生成的响应位置置 `1`
- `concat_versions` —— token 的权重版本标记
- `concat_context_token_count` / `concat_output_token_count` —— 上下文与生成 token 的计数
- `concat_logprobs_invalid` / `tito_incremental_tokenization_failed` —— TITO 组装的安全标志位

### Append-Only 契约

TITO 依赖对话历史上的 **append-only 契约**。如果 Agent 重写了历史、改变了已有消息前缀、变更了工具 schema，或者 TITO 分词失败，Proxy 就会触发一次 **segment 边界** —— 关闭当前 segment，并以重置后的 TITO 状态开启一个新的 segment。

> [!NOTE]
> 当 TITO 失败时（例如模板渲染错误），Proxy 会在该 step 上标记 `concat_incremental_tokenization_failed=True` 并开启新 segment。这是一种安全回退——不会丢失任何数据，只是被拆分到不同的 segment 中。

## ✂️ Segment 边界

当 Proxy 检测到会破坏 token 级一致性的事件时，会自动把一个 session 拆分成多个 segment。理解 segment 边界很重要，因为每个 segment 都会成为一条独立的训练样本。

 | 触发条件 | 检测方式 | 发生什么 |
 | :-------- | :---------- | :------------- |
 | **历史重写（History Rewrite）** | Agent 发来的消息并非对上一轮对话的延续 | 当前 segment 被 finalize；新 segment 以全新状态开始 |
 | **工具 Schema 变更** | 可用工具在两轮之间发生变化 | 产生 segment 边界；新的工具上下文从干净状态开始 |
 | **TITO 前缀不匹配** | tito 模式下已有的消息前缀发生变化 | 当前 segment 被 finalize；新 segment 以全新状态开始 |
 | **TITO 回退** | 增量分词失败（模板错误、编码不匹配） | 标记失败标志位；以重置的 TITO 状态开启新 segment |

> [!NOTE]
> 每个 segment 都成为一条独立的训练样本，但同一 session 产出的所有 segment 共享相同的 `parent_traj_id` 与 `rollout_id`，确保它们在训练时被归为一组。

`DRESSAGE_PROXY_MAX_STEPS_PER_SESSION` 是一道独立的守卫：一旦某个 Proxy session 已达到该步数，下一次生成请求会在生成前直接返回 HTTP 400。它不会自动 finalize 该 session。

## 🌐 HTTP 端点

Proxy 暴露以下端点，供 Agent 交互与 rollout 管理使用：

 | 端点 | 方法 | 用途 | 说明 |
 | :--------- | :------- | :-------- | :-------- |
 | `/v1/models` | `GET` | 模型列表 | OpenAI 兼容的模型列表透传。 |
 | `/v1/chat/completions` | `POST` | Agent 推理 | OpenAI 兼容。记录 step 数据。需要 session 相关 header。 |
 | `/session/finalize` | `POST` | 终结 session | 关闭所有未闭合的 segment，写入 trajectory store。 |
 | `/trajectory/read` | `POST` | 读取 segment | 按 session ID 或 trajectory ID 返回已 finalize 的 segment。 |
 | `/trajectory/stats` | `GET` | 存储统计 | 报告内存 trajectory store 的统计信息。 |
 | `/v1/rollout/pause` | `POST` | 暂停生成 | 通知 `GenerationController` 在下一个 token 边界中止。 |
 | `/v1/rollout/resume` | `POST` | 恢复生成 | 权重更新完成后重新启用生成。 |
 | `/v1/rollout/pause_state` | `GET` | 暂停状态 | 报告 `GenerationController` 的 pause/resume 状态。 |
 | `/health` | `GET` | 健康检查 | 返回活跃 session、trajectory store、rollout pause 与 Proxy 配置状态。 |

### 可抢占生成流程

`GenerationController` 使得在 partial rollout 期间可以为权重更新安全地中断正在进行的生成。这对于 rollout 与训练相互重叠的持续训练场景至关重要。

```text
1️⃣  权重更新信号到达
2️⃣  POST /v1/rollout/pause → GenerationController.abort()
3️⃣  活跃的 SGLang 请求在下一个 token 边界中止
4️⃣  partial output 保留在当前 StepRecord 中
5️⃣  权重更新完成
6️⃣  POST /v1/rollout/resume → GenerationController.resume()
7️⃣  下一次 chat_completions 调用从生成中断处接续
```

> [!TIP]
> pause/resume 机制是原子的 —— 不存在任何窗口会让 token 用陈旧权重被生成出来。`GenerationController` 状态机保证 `idle → generating → paused → resumed` 之间的干净转换。

## 🚀 使用方式

### 启动 Proxy

```bash
# 包含当前的启动与解析器控制项
dressage-proxy \
  --tokenizer-path /path/to/Qwen3.5-4B \
  --sglang-router-url http://<sglang-router-host>:<port> \
  --token-build-model qwen3_5 \
  --context-window 32768 \
  --rollout-temperature 1.0 \
  --record-token-versions \
  --mask-nonlast-version-tokens \
  --dressage-partial-rollout \
  --tool-call-parse-backend sglang_api \
  --reasoning-parse-backend sglang_api \
  --model-tool-call-type qwen3_5 \
  --model-reasoning-type qwen3
```

输出长度上限对每个请求独立解析：

- Agent 提供的 `max_tokens` 限制该次模型调用。
- `--max-output-tokens` 在显式配置时，是 Proxy 级别的硬上限。
- 启用动态上下文限制时，`context_window - prompt_tokens` 是物理预算。Proxy 会使用全部剩余量（含最后一个 token），并用 `min(...)` 与请求侧或 Proxy 侧的上限组合。
- 如果请求与 CLI 都未提供上限，则使用剩余上下文。若 `--context-window` 也缺失，Proxy 会省略 `max_new_tokens`，由后端选择其默认值（[SGLang 0.5.12 SamplingParams](https://github.com/sgl-project/sglang/blob/v0.5.12.post1/python/sglang/srt/sampling/sampling_params.py) 默认为 128）。

`--no-dynamic-max-tokens` 仅关闭由上下文推导出的裁剪，不会改变请求侧或可选的 CLI 上限。

### 使用 Proxy 客户端

```python
from dressage.proxy.proxy_client import ProxyClient

client = ProxyClient(proxy_url="http://localhost:8800")

# 发送一次 chat completion
response = await client.chat_completions(
    {"model": "proxy-model", "messages": [{"role": "user", "content": "Hello!"}]},
    session_id="sess-001",
    instance_id="inst-001",
    turn_id="turn-001",
)

# 终结该 session
await client.finalize_session("sess-001", instance_id="inst-001")

# 读取轨迹
payload = await client.read_trajectory(session_id="sess-001", drain=True)
segments = payload["data"]
```

## 🔀 路由回放（R3）

对于 **Mixture-of-Experts（MoE）** 模型，Proxy 可以捕获每个生成 token 的 **routed expert ID**，从而在训练期间实现忠实的路由回放。如果没有 R3，训练将使用随机的 expert 路由，可能与 rollout 时的行为发生偏离。

```text
Proxy (--use-rollout-routing-replay)
        │
        ├── 为每个生成的 token 向 SGLang 请求 routed expert ID
        ├── 将 expert ID 数组编码为 base64 chunk 以高效传输
        ├── 存入 trajectory segment 的元数据
        └── rollout.artifacts.samples.extract_routed_experts → 训练数据
```

### 数据格式

R3 以 base64 编码的 int32 载荷存储 routed expert ID。Dressage 支持三种记录形态：

| 字段 | 说明 |
| ----------------------- | ------------------------------------------------------------------------------------ |
| `routed_experts` | 单次不间断生成的直接载荷。 |
| `routed_experts_chunks` | partial 或续跑生成的分块载荷。 |
| `routed_experts_parts` | 面向 TITO segment 的多 step 包装；每个 part 可包含直接数据或 chunk。 |

通过在 Proxy 上设置 `--use-rollout-routing-replay` 启用 R3。

## 🔧 可配置解析器

Proxy 为工具调用与 reasoning 抽取支持可插拔后端，以适配不同的模型架构与 SGLang 配置：

 | 解析器类型 | 后端 | 说明 |
 | :------------ | :-------- | :------------ |
 | 工具调用 | `local` | 使用模型特定的正则/启发式规则直接解析模型输出 |
 | 工具调用 | `sglang_api` | 委托给 SGLang 内置的工具调用抽取 |
 | 工具调用 | `hybrid` | 先尝试 `sglang_api`，失败时回退到 `local` |
 | Reasoning | `local` | 从模型输出中解析 `<think>...</think>` 块 |
 | Reasoning | `sglang_api` | 把 reasoning 抽取委托给 SGLang |
 | Reasoning | `hybrid` | SGLang 优先，local 回退 |

`--tool-call-parse-backend` 与 `--reasoning-parse-backend` 均默认为 `sglang_api`。

```bash
dressage-proxy \
  --tokenizer-path /path/to/Qwen3.5-4B \
  --tool-call-parse-backend sglang_api \
  --reasoning-parse-backend sglang_api \
  --model-tool-call-type qwen3_5 \
  --model-reasoning-type qwen3
```

> [!TIP]
> 生产环境推荐 `hybrid` 后端。它在 SGLang 可用时利用其优化过的解析能力，并在 SGLang 不支持该模型格式时优雅回退到本地解析。

## 📊 数据流

从 Agent 请求到轨迹落库的完整数据流（Proxy 逐步记录 token ID + logprob、loss mask、权重版本、TITO 片段、MoE 路由 ID；finalize 后写入 Trajectory Store）：

```text
┌─────────────┐     ┌──────────────────────────┐     ┌──────────────┐
│    Agent    │────▶│          Proxy           │────▶│   SGLang     │
│             │     │                          │     │   Router     │
│  whitebox   │◀────│  records per-step:       │◀────│              │
│ or blackbox │     │  • token IDs + logprobs  │     │   policy     │
└─────────────┘     │  • loss masks            │     │   model      │
                    │  • weight versions       │     └──────────────┘
                    │  • TITO fragments        │
                    │  • MoE routing IDs       │
                    └──────────┬───────────────┘
                               │ finalize
                    ┌──────────▼───────────────┐
                    │    Trajectory Store      │
                    │                          │
                    │    segments[]            │
                    │    ├── tokens[]          │
                    │    ├── logprobs[]        │
                    │    ├── loss_mask[]       │
                    │    ├── weight_vers[]     │
                    │    └── experts[]         │  ← MoE 路由（可选）
                    └──────────────────────────┘
```

## 📁 包结构

```text
dressage/proxy/
├── server.py                     # FastAPI 应用，CLI 入口
├── session_manager.py            # 逐 session 的 step 跟踪
├── trajectory_store.py           # 内存 segment 存储
├── generation_controller.py      # 可抢占生成
├── sglang_client.py              # SGLang 路由器客户端
├── tool_call_parser.py           # 工具调用抽取
├── reasoning_parser.py           # Reasoning 内容解析
├── proxy_client.py               # 供 rollout 代码使用的异步客户端
├── tool_call_ids.py              # 确定性 ID 生成
├── last_step/                    # Snapshot 对齐辅助
│   └── prompt_assistant_mask.py  # Assistant loss mask 构建器
└── tito/                         # TITO 分词器
    ├── tito_tokenizer.py         # Qwen35TITOTokenizer
    ├── template_utils.py         # 固定模板渲染
    └── templates/
        └── qwen3_5_fixed.jinja   # 锁定的 chat template
```

## 🔗 集成点

 | 组件 | 关系 |
 | :---------- | :------------ |
 | [Paddock](../docs/paddock.md) | Paddock 协调 Proxy session —— 每次 rollout 通过 Proxy 客户端创建一个 session |
 | [Sandbox](../docs/sandbox.md) | BlackboxServer 的进程内 LLM 代理把所有 Agent 调用转发经由 Dressage Proxy |
 | [BlackboxServer](../docs/blackbox-server.md) | 在每次 LLM 调用上注入 session/turn header，经 Proxy 路由 |
 | [Rollout](../docs/rollout.md) | 生成钩子使用 `ProxyClient` 管理 session 并读取轨迹 |
 | [Training](../docs/training.md) | 训练层消费 Proxy 产出的 segment，用于 TITO 分词与多 segment 展开 |

---

[← 返回主 README](../README.md) · [英文原文：docs/proxy.md](../docs/proxy.md)
