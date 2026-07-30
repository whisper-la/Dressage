# Dressage 架构讲解 & 代码解析

> 这份文档干三件事：**讲架构**（系统由哪些组件构成、为什么这样切）、**讲思路**（每个设计决策背后想解决什么问题，不这样做会出什么事）、**逐行解析代码**（从一行 JSONL 数据走到一次梯度更新，沿途的每一段关键代码都读到）。
>
> **读者画像**：一名**只懂编程的本科生**（写过 Python、知道函数/类/异常、跑过 `pip install`），但**没接触过**大语言模型、深度学习训练、强化学习、agent 框架、分布式训练。你需要的所有背景知识，这份文档全部从零讲起。
>
> **配套类比**：每个新概念都尽量配一个程序员熟悉的类比（loss ≈ 单元测试失败数，gradient ≈ 多维曲面的最陡下降方向，`@register_reward` ≈ Flask 的 `@app.route`，proxy ≈ Nginx 反向代理，`asyncio.Semaphore(32)` ≈ 令牌池，等等）。
>
> **怎么读这份文档**：
> - 完全是新手、想要全景 → 从 **Part 0** 的"§0.0 十分钟全景图"开始，按顺序读到 §0.15，再进入 Part 1。
> - 已经懂大模型基本概念 → 跳到 §0.6（强化学习起步）。
> - 已经懂 PPO/GRPO → 跳到 **Part 1**（Dressage 是什么）。
> - 已经熟悉 RL，只想看 Dressage 怎么实现 → 跳到 **Part 4**（端到端代码细读）。
> - 只想了解架构、不读代码 → 读 **Part 1 + Part 2 + Part 3**，跳过 Part 4 的代码细节。
>
> **文档结构**：
> - **Part 0** — 全部背景知识，从零开始。包括：十分钟全景图、Python/网络/异步等软件工程前置课、大模型怎么运作、训练循环、强化学习、PPO/GRPO、logprob/loss_mask、agent 和 tool_call。**这是文档里最长的一节**。
> - **Part 1** — Dressage 是什么、要解决什么问题、和 slime（底层 RL 框架）的边界。
> - **Part 2** — 五个核心设计决策。每个决策附"如果不这样做会怎样"反例，让你看清"为什么必须这样"。
> - **Part 3** — 代码组织 walkthrough：5 个子包各自的职责、依赖方向、文件清单。
> - **Part 4** — **端到端代码细读**：从一行 JSONL 数据进入、到产生梯度更新、再到下一轮 rollout，每一步都点到具体代码（文件名:行号）。**这是文档的主体**。
> - **Part 5** — 不那么显然的设计细节、容易踩的坑、反直觉的地方。
> - **Part 6** — 上手做事：跑测试、加 reward、接 Paddock、接黑盒 agent、调试技巧。
> - **Part 7** — 扩展阅读、推荐代码阅读顺序、外部资料。

---

# 目录

- **[Part 0 — 基础知识：从程序员视角理解大模型、训练、RL、Agent](#part-0--基础知识从程序员视角理解大模型训练rlagent)**
  - [0.0 十分钟全景图](#00-十分钟全景图一个本科生能记住的版本)
    - [0.0.1 我们在解决什么问题](#001-我们在解决什么问题)
    - [0.0.2 三个独立进程的总览图](#002-三个独立进程的总览图)
    - [0.0.3 一条样本的端到端旅程（口述版）](#003-一条样本的端到端旅程口述版)
  - [0.0' 软件工程前置课](#00-软件工程前置课你需要知道的-python--网络--并发概念)
    - [A. 装饰器（decorator）](#a-装饰器decorator)
    - [B. 抽象基类（ABC）](#b-抽象基类abcabstract-base-class)
    - [C. dataclass](#c-dataclass)
    - [D. 异步（async / await / asyncio）](#d-异步async--await--asyncio)
    - [E. HTTP 服务 / FastAPI](#e-http-服务--fastapi)
    - [F. httpx 异步 HTTP 客户端](#f-httpx-异步-http-客户端)
    - [G. 反向代理（reverse proxy）](#g-反向代理reverse-proxy)
    - [H. 动态模块加载（importlib）](#h-动态模块加载importlib)
    - [I. git submodule](#i-git-submodule)
    - [J. 进程 vs 线程 vs 协程](#j-进程-vs-线程-vs-协程一句话区分)
    - [K. 环境变量](#k-环境变量)
  - [0.1 大模型本质：一个超大型的"接龙函数"](#01-大模型本质一个超大型的接龙函数)
  - [0.2 Tokenizer：文本 ↔ 数字 ID](#02-tokenizer文本--数字-id)
  - [0.3 模型怎么"生成"文本：autoregressive](#03-模型怎么生成文本autoregressive)
  - [0.4 模型怎么"学"：训练循环](#04-模型怎么学训练循环)
  - [0.5 三种训练范式](#05-三种训练范式)
  - [0.6 强化学习的核心概念](#06-强化学习的核心概念)
  - [0.7 策略梯度直觉](#07-策略梯度直觉怎么从-reward-反推到-token)
  - [0.8 PPO：让一份采样数据能多次复用](#08-ppo让一份采样数据能多次复用)
  - [0.9 GRPO：省掉 value model 的 PPO 变种](#09-grpo省掉-value-model-的-ppo-变种)
  - [0.10 logprob 详解](#010-logprob-详解)
  - [0.11 loss_mask 详解](#011-loss_mask-详解)
  - [0.12 Agent 是什么：从 chatbot 升级](#012-agent-是什么从-chatbot-升级)
  - [0.13 tool_call 的协议细节](#013-tool_call-的协议细节)
  - [0.14 Rollout / Trajectory](#014-rollout--trajectory)
  - [0.15 名词速查表（汇总）](#015-名词速查表汇总)
- **[Part 1 — Dressage 是什么，要解决什么问题](#part-1--dressage-是什么要解决什么问题)**
  - [1.1 一句话定位](#11-一句话定位)
  - [1.2 从 RLHF 到 Agentic RL 的演进](#12-从-rlhf-到-agentic-rl-的演进)
  - [1.3 slime 是什么，缺什么](#13-slime-是什么缺什么)
  - [1.4 Dressage 怎么补齐](#14-dressage-怎么补齐)
- **[Part 2 — 五个核心设计决策](#part-2--五个核心设计决策)**
  - [决策 1：用 slime 作 RL 底座，绝不修改](#决策-1用-slime-作-rl-底座绝不修改)
  - [决策 2：所有 LLM 调用收口到一个 proxy](#决策-2所有-llm-调用收口到一个-proxy)
  - [决策 3：traj_id 是全系统主键](#决策-3traj_id-是全系统主键)
  - [决策 4：环境/工具/黑盒 agent 全抽成 Paddock 接口](#决策-4环境工具黑盒-agent-全抽成-paddock-接口)
  - [决策 5：reward 是每个样本独立的，可注册可热插](#决策-5reward-是每个样本独立的可注册可热插)
- **[Part 3 — 代码组织 walkthrough](#part-3--代码组织-walkthrough)**
- **[Part 4 — 端到端代码细读](#part-4--端到端代码细读从一行-jsonl-走到一次梯度更新)**
  - [4.1 启动：proxy 先起，slime 再起](#41-启动proxy-先起slime-再起)
  - [4.2 数据加载：DressageDataSource](#42-数据加载dressagedatasource)
  - [4.3 rollout 入口：generate_rollout_fully_async](#43-rollout-入口generate_rollout_fully_async)
  - [4.4 后台 worker 并发生成](#44-后台-worker-并发生成)
  - [4.5 单条 trajectory 的生成钩子](#45-单条-trajectory-的生成钩子)
  - [4.6 白盒 agent 循环：WhiteboxAgent](#46-白盒-agent-循环whiteboxagent)
  - [4.7 进入 proxy：POST /v1/chat/completions](#47-进入-proxypost-v1chatcompletions)
  - [4.8 SGLang 调用：SGLangRouterClient.generate](#48-sglang-调用sglangrouterclientgenerate)
  - [4.9 SessionManager 记账：record_step](#49-sessionmanager-记账record_step)
  - [4.10 回到 whitebox loop，工具执行](#410-回到-whitebox-loop工具执行)
  - [4.11 Finalize：把 turns 拼成 Trajectory](#411-finalize把-turns-拼成-trajectory)
  - [4.12 读 trajectory：read_trajectory](#412-读-trajectoryread_trajectory)
  - [4.13 算 reward](#413-算-reward)
  - [4.14 Trajectory → Sample(s)](#414-trajectory--samples)
  - [4.15 清理环境](#415-清理环境)
  - [4.16 所有 trajectory 跑完，重新分组](#416-所有-trajectory-跑完重新分组)
  - [4.17 训练侧：convert_samples_to_train_data](#417-训练侧convert_samples_to_train_data)
  - [4.18 进入 slime 训练循环（黑盒）](#418-进入-slime-训练循环黑盒)
- **[Part 4B — 黑盒 agent 端到端：blackbox_dispatch 详解](#part-4b--黑盒-agent-端到端blackbox_dispatch-详解)**
  - [4B.0 入口：slime 的 --custom-generate-function-path](#4b0-入口slime-的---custom-generate-function-path)
  - [4B.1 元数据落桌：session_id / instance_id / blackbox_type](#4b1-元数据落桌session_id--instance_id--blackbox_type)
  - [4B.2 paddock.init：申请一个沙箱](#4b2-paddockinit申请一个沙箱)
  - [4B.3 paddock.register_agent：把 proxy URL 注入沙箱](#4b3-paddockregister_agent把-proxy-url-注入沙箱)
  - [4B.4 before_agent execute_cmds：环境探针](#4b4-before_agent-execute_cmds环境探针)
  - [4B.5 paddock.call_agent：把控制权交给黑盒 agent](#4b5-paddockcall_agent把控制权交给黑盒-agent)
  - [4B.6 黑盒 agent 内部如何串接 proxy](#4b6-黑盒-agent-内部如何串接-proxy)
  - [4B.7 after_agent execute_cmds：采集环境产物](#4b7-after_agent-execute_cmds采集环境产物)
  - [4B.8 Finalize + read_trajectory + 写回 Sample](#4b8-finalize--read_trajectory--写回-sample)
  - [4B.9 错误处理与 best-effort terminate](#4b9-错误处理与-best-effort-terminate)
- **[Part 4C — Paddock 与 Sandbox Provider 后端](#part-4c--paddock-与-sandbox-provider-后端)**
  - [4C.1 三种 sandbox provider 的对比](#4c1-三种-sandbox-provider-的对比)
  - [4C.2 BlackboxAgentPaddock 与 SandboxProvider](#4c2-blackboxagentpaddock-与-sandboxprovider)
  - [4C.3 local_bwrap provider 三层架构](#4c3-local_bwrap-provider-三层架构)
  - [4C.4 Local 沙箱隔离：bubblewrap vs direct](#4c4-local-沙箱隔离bubblewrap-vs-direct)
  - [4C.5 启动顺序：手动 vs auto-start](#4c5-启动顺序手动-vs-auto-start)
  - [4C.6 怎么选：决策清单](#4c6-怎么选决策清单)
- **[Part 4D — 新功能与进阶机制](#part-4d--新功能与进阶机制)**
  - [4D.1 配置模块：dressage/config/config.py](#4d1-配置模块dressageconfigconfigpy)
  - [4D.2 GenerationController：pause / resume / shutdown](#4d2-generationcontrollerpause--resume--shutdown)
  - [4D.3 Partial Rollout：部分样本先行](#4d3-partial-rollout部分样本先行)
  - [4D.4 Staleness 追踪](#4d4-staleness-追踪)
  - [4D.5 concat 轨迹构建模式 + TITO 分词器](#4d5-concat-轨迹构建模式--tito-分词器)
  - [4D.6 reasoning_parser.py](#4d6-reasoning_parserpy)
  - [4D.7 三种 Rollout 模式](#4d7-三种-rollout-模式)
  - [4D.8 train_async_with_rollout_pause.py](#4d8-train_async_with_rollout_pausepy)
- **[Part 5 — 不那么显然的设计细节](#part-5--不那么显然的设计细节)**
  - [5.1 多 turn loss_mask 构造](#51-多-turn-loss_mask-构造)
  - [5.2 记录时的权威 token 序列](#52-记录时的权威-token-序列不要事后重新-tokenize)
  - [5.3 异步 / 同步交界](#53-异步--同步交界)
  - [5.4 GRPO 组归一化 + parent 段广播](#54-grpo-组归一化--parent-段广播)
  - [5.5 黑盒 agent 怎么把 traj_id 透传](#55-黑盒-agent-怎么把-traj_id-透传)
  - [5.6 缺 logprob 时 remove_sample = True](#56-缺-logprob-时-remove_sample--true)
  - [5.7 slime 的 --custom-reward-post-process-path 坑](#57-slime-的---custom-reward-post-process-path-坑)
- **[Part 6 — 上手做事](#part-6--上手做事)**
  - [6.1 跑测试](#61-跑测试)
  - [6.2 起 proxy 走一遍假数据](#62-起-proxy-走一遍假数据)
  - [6.3 加一个新的 reward 函数](#63-加一个新的-reward-函数)
  - [6.4 实现一个真实 Paddock](#64-实现一个真实-paddock)
  - [6.5 接入一个黑盒 agent](#65-接入一个黑盒-agent)
  - [6.6 调试技巧](#66-调试技巧)
- **[Part 7 — 扩展阅读](#part-7--扩展阅读)**
- **[读完之后](#读完之后)**

---

# Part 0 — 基础知识：从程序员视角理解大模型、训练、RL、Agent

> **这一节最长**。读完之后，后面所有专有名词都不再陌生。如果你已经熟悉某一节内容，可以直接跳过。
>
> 这一节按"从最基本到稍复杂"的顺序组织：
>
> §0.0 十分钟全景图 → §0.0' 软件工程前置课（async/装饰器/ABC/插件加载/HTTP/进程） → 0.1 大模型本质 → 0.2 tokenizer → 0.3 文本生成机制 → 0.4 训练循环 → 0.5 三种训练范式 → 0.6 RL 的基本词汇 → 0.7 策略梯度的直觉 → 0.8 PPO → 0.9 GRPO → 0.10 logprob 深度展开 → 0.11 loss_mask 深度展开 → 0.12 Agent 是什么 → 0.13 tool_call 的协议细节 → 0.14 Rollout/Trajectory → 0.15 名词速查表。

## 0.0 十分钟全景图：一个本科生能记住的版本

在跳进细节之前，先用一节给你一张"全局地图"。读完这一节，你应该能用三句话向同学描述 Dressage：**"它训练一个会用工具的 AI agent。具体是：用强化学习——让 agent 反复尝试任务，给最终表现打分，回头调模型参数。整个系统由三个进程组成：训练进程改模型权重、推理进程做模型采样、proxy 进程在中间记账。"**

### 0.0.1 我们在解决什么问题

想象你想让一个 AI 助手帮你完成下面这个任务：

> *"帮我看看 `/data` 目录下的所有 CSV 文件，找出哪个文件行数最多，然后把这个文件的前 10 行打印出来。"*

一个纯文本聊天机器人（chatbot）做不到这件事——它没法真的去看文件。它只能像 ChatGPT 早期那样**猜**：

> "你可以用 `ls /data | xargs wc -l | sort -n` 来找出最大的文件……"

它给你建议，但执行得你自己来。

**agent** 不一样：agent 能输出"我要执行 `ls /data`"这种**结构化指令**，由外部环境真的去执行，然后把执行结果（"我看到了 a.csv b.csv c.csv"）塞回模型让它继续。整个过程像下面这样：

```
你:    "帮我看看 /data 目录下的所有 CSV 文件..."
模型:  "我先列出来" → 工具调用: bash("ls /data")
环境:  执行 → 返回 "a.csv  b.csv  c.csv"
模型:  "再统计行数" → 工具调用: bash("wc -l /data/*.csv")
环境:  执行 → 返回 "100 a.csv  50 b.csv  200 c.csv  350 total"
模型:  "c.csv 最大" → 工具调用: bash("head /data/c.csv")
环境:  执行 → 返回 "<前10行内容>"
模型:  "完成！下面是 c.csv 的前 10 行：..."
```

**这就是 agent**。它不是一次性输出答案，而是来回多 turn 地"思考 → 调工具 → 看结果 → 继续思考"，直到完成任务。

**Dressage 训练的就是这种 agent**。"训练"意思是：调整模型的参数（几十亿到几千亿个浮点数），让它面对类似任务时**更可能选对工具、按更优的步骤完成**。训练方法是**强化学习（RL）**——让它跑 N 次任务、给每次跑打分，根据分数高低反推每一步该不该那样做。

为什么用 RL 而不是传统的"监督学习"（给一堆"问题 → 标准答案"的对子）？因为 agent 任务**没有标准答案**：完成 "/data 找最大 CSV" 这个任务的"步骤序列"可以千奇百怪（先 `ls` 还是先 `wc -l`、用 `cat` 还是 `head`），只有"最终是否完成"是可衡量的。RL 的能力恰好是：**只要能给一条完整轨迹打分，就能训练**。

### 0.0.2 三个独立进程的总览图

Dressage 运行时由**三个独立的 OS 进程**构成（实际部署里 slime 进程会跨多机多卡，但概念上是一个）：

```
   ┌─────────────────────────────────┐
   │   进程 A: slime 训练进程         │
   │   ─ 拿模型权重做反传 + 梯度更新   │
   │   ─ 通过插件 hook 调用 Dressage   │
   │   ─ 跑在 GPU 上（Megatron 并行）  │
   │   ─ 由 examples/scripts/*.sh 启 │
   └────────┬────────────────────────┘
            │
            │ (1) 训练前: "给我一批 trajectory"
            │      ↓ 调插件函数 generate_rollout_fully_async(...)
            │
   ┌────────▼────────────────────────┐
   │   进程 A 里的 Dressage 代码      │
   │   （rollout/、reward/、training/）│
   │   ─ 跑 agent 主循环              │
   │   ─ 调 paddock 执行工具          │
   │   ─ 调 proxy 做模型推理          │
   └────────┬────────────────────────┘
            │
            │ (2) HTTP POST /v1/chat/completions
            │
   ┌────────▼────────────────────────┐
   │   进程 B: dressage-proxy        │
   │   ─ FastAPI HTTP 服务（端口 8800）│
   │   ─ 接收 "请帮我对这段对话采样"   │
   │   ─ 记录每个 traj_id 的 turn 历史 │
   │   ─ 强制 return_logprob=True     │
   │   ─ 由 dressage-proxy CLI 启     │
   └────────┬────────────────────────┘
            │
            │ (3) HTTP POST /generate
            │
   ┌────────▼────────────────────────┐
   │   进程 C: SGLang 推理服务        │
   │   ─ 真正"加载模型权重 + 跑前向"   │
   │   ─ HTTP 服务（端口 30000）       │
   │   ─ 由 slime 内部启动（无需手动）  │
   │   ─ 跑在 GPU 上                  │
   └─────────────────────────────────┘
```

注意箭头方向都是**从训练进程出发**——A 调 B、B 调 C。Dressage 没有 daemon 在那儿自己跑，它是在 slime 请求 rollout 时才被激活。

**为什么要有 proxy（进程 B）？** 在朴素方案里，进程 A 完全可以直接调进程 C。但那样会出几个问题：(i) 每个调用点都要记得开 `return_logprob=True`，漏一个就缺一段训练数据；(ii) agent 是多 turn 的，每个 turn 一次推理调用——零散的 turn 谁来汇总成一条 trajectory？(iii) 如果未来想接入"别人写的黑盒 agent"（比如 LangChain agent，我们改不了它的源码），怎么强制它把所有 LLM 调用绑到我们关心的 traj_id 上？答案就是在中间塞一层 proxy 兜底——所有 LLM 流量都过它，它做记账。这是 Dressage 最核心的架构决策之一（详见 §2 决策 2）。

### 0.0.3 一条样本的端到端旅程（口述版）

假设你的数据文件 `data.jsonl` 里有一行：

```json
{"prompt": "求 1+1 等于几", "label": "2", "agent_mode": "whitebox", "reward_fn": "exact_match"}
```

下面这一段是这条样本从"被加载"到"产生梯度更新"的完整旅程，**全部用口述**——细节会在 Part 4 里逐行解析：

1. **加载**：slime 决定开始下一轮 rollout，问 Dressage："给我 8 个 prompt"。Dressage 的 `DressageDataSource.get_samples` 读 JSONL 文件，把每一行变成一个 `Sample` 对象。同一个 prompt **被复制 4 份**（GRPO 算法要求每个 prompt 多采几条来对比），组成一个 4 元素的组。8 个 prompt × 4 = 32 个 sample。
2. **分发**：rollout 入口（如 `generate_rollout_fully_async`）把这些 prompt 组交给后台 worker 并发生成。每个 sample 拿到一个唯一的 `traj_id`/`session_id`（如 `bbs-...`）——这个 id 接下来串起整个流水线。
3. **agent 循环**：对每个 sample，根据 `agent_mode` 字段选择"白盒"（agent 循环在 Dressage 内部）或"黑盒"（委托外部 agent）。我们这条是 whitebox，进入 `whitebox_loop`：构造 messages → POST 给 proxy → 拿到模型回复 → 如果回复里有 `<tool_call>` 就用 Paddock 执行 → 把结果追加到 messages → 再 POST → 直到模型不再调工具或达到 20 turn 上限。
4. **proxy 记账**：每次 POST 到 proxy 的 `/v1/chat/completions`，proxy 做四件事：把 messages 用 chat template 拼成 token、调 SGLang 采样（强制 `return_logprob=True`）、解析返回里的 `<tool_call>` 标签、把这个 turn 的所有信息（输入 token 数、输出 token、每个输出 token 的 logprob）记到 `SessionManager` 里 `traj_id` 对应的 session 列表。
5. **结算**：whitebox 循环结束后，Dressage 调 `proxy.finalize_session(traj_id)`——proxy 把这个 session 里所有 turn 拼成一个完整的 `Trajectory`（连续 token 序列 + 对应的 logprobs + loss_mask）放进 `TrajectoryStore`。Dressage 再调 `proxy.read_trajectory(traj_id)` 把它取出来（取的同时删除）。
6. **打分**：根据 sample 的 `reward_fn` 字段（这里是 `"exact_match"`）从注册表里查出对应函数，调 `exact_match(sample, args=None)` 拿到一个 float。这就是这条样本的原始 reward。
7. **回收**：调 `paddock.terminate(traj_id)` 释放环境。32 个并发任务全部结束后，rollout 函数收齐 32 个 `Sample`（带上 reward、tokens、logprobs、loss_mask），返回给 slime。
8. **训练侧 1：GRPO 归一化**。slime 把 32 个 sample 交给 Dressage 的 `convert_samples_to_train_data`。第一步是把 reward 按组归一化——同一个 prompt 的 4 个采样里，比平均高的 reward 变正（"做对了"），比平均低的变负（"做错了"）。这一步把 reward 转成 GRPO advantage。
9. **训练侧 2：拼 batch**。把 32 个 sample 的 tokens、loss_masks、normalized rewards、采样时的 logprobs 拼成一个 dict 还给 slime。
10. **梯度更新**：slime 在 GPU 上把所有 tokens 重新过一遍模型（拿到新策略下的 logprob），算 PPO 损失（`exp(new − old) × advantage × loss_mask`，加 clip 和 KL 项），反传，Adam 更新参数。
11. **循环**：slime 把新权重同步到 SGLang，再次问 Dressage 要下一批 rollout。回到 step 1。

整个流程**核心数据流**用一张图概括：

```
   JSONL 一行
      │
      ▼
   ┌──────────────────────┐
   │ Sample(prompt, label, │
   │ metadata={agent_mode, │   ← 复制 N 份 (GRPO)
   │ reward_fn, ...})      │
   └──────────┬───────────┘
              │ traj_id 分配
              ▼
   ┌──────────────────────────────────────────────┐
   │ async whitebox_loop:                          │
   │   while turn < max_turns:                     │
   │     msg = proxy.chat_completions(traj_id,    │
   │                                  messages)    │
   │     if no tool_call: break                    │
   │     result = paddock.tool_call(traj_id, ...)  │
   │     messages.append(result)                   │
   └──────────┬───────────────────────────────────┘
              │  finalize + read
              ▼
   ┌──────────────────────────────────────────────┐
   │ Trajectory(tokens=[...], logprobs=[...],     │
   │            loss_mask=[...], messages=[...])   │
   └──────────┬───────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────┐
   │ reward_fn(trajectory, sample) → float        │
   └──────────┬───────────────────────────────────┘
              │
              ▼ (×32 个 sample 并发收齐)
   ┌──────────────────────────────────────────────┐
   │ convert_samples_to_train_data:               │
   │   ① GRPO 组归一化: r → advantage             │
   │   ② 拼 dict: tokens, loss_masks, rewards     │
   └──────────┬───────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────┐
   │ slime Megatron actor:                        │
   │   logprob_new = model(tokens)                │
   │   ratio = exp(logprob_new - logprob_old)     │
   │   loss = -mean(min(ratio·A, clip(ratio)·A))  │
   │   loss.backward(); optimizer.step()          │
   └──────────────────────────────────────────────┘
```

**核心抽象一句话总结**：
- **traj_id** = 一条 trajectory 的全局唯一 ID，串起 Paddock + proxy session + Trajectory + Sample。
- **Paddock** = 抽象基类，封装"非 LLM 的执行能力"（起环境、执行工具、调黑盒 agent、收环境）。
- **Proxy** = FastAPI 服务，所有 LLM 调用必须经过它（保证 logprob 不漏、session 不散）。
- **Trajectory** = 一条 agent 任务的完整记录，reward 函数看的就是它。
- **Sample** = slime 的训练数据单元；一条 Trajectory 通常变成 1 个 Sample。
- **rollout** = 一次"批量采样 + 算 reward"的过程，对应一次 rollout 入口函数（如 `generate_rollout_fully_async()`）调用。
- **GRPO** = 同 prompt 多采几条、组内归一化当 advantage 的 RL 算法。Dressage 默认用它。

如果上面这些名词暂时没全看懂，没关系——后面 §0.1 到 §0.15 会从头讲一遍。这一节只是给你一张地图，让你读后面细节时知道"这块拼图在整张图的哪里"。

## 0.0' 软件工程前置课：你需要知道的 Python / 网络 / 并发概念

> 这一节面向"只学过基础 Python，没系统接触过异步、装饰器、ABC、HTTP 服务、动态模块加载"的本科生。**已经熟悉这些的可以直接跳到 §0.1**。

Dressage 是一个**多进程异步系统**，它大量使用了下面这些 Python/网络概念。先把这些工具讲清楚，后面读代码不会被它们绊倒。

### A. 装饰器（decorator）

装饰器是"把函数当参数传给另一个函数，得到一个新函数"的语法糖。例如：

```python
def register_reward(name):
    def wrapper(fn):
        REGISTRY[name] = fn
        return fn        # 原样返回函数本身
    return wrapper

@register_reward("exact_match")
def exact_match(sample, *, args=None, **_):
    return ...
```

这段代码等价于：

```python
def exact_match(sample, *, args=None, **_):
    return ...
exact_match = register_reward("exact_match")(exact_match)
```

效果：**模块被 import 的瞬间，`exact_match` 这个函数就被登记到了全局表 `REGISTRY` 里**，键名是 `"exact_match"`。后面别处可以 `REGISTRY["exact_match"](sample, args=None)` 调到它，**不需要事先 import**。

> Dressage 用法：`@register_reward(...)` 在 `dressage/reward/registry.py`、`@app.post(...)` 在 `dressage/proxy/server.py`（FastAPI 自带的路由装饰器）。

### B. 抽象基类（ABC，Abstract Base Class）

Python 的 ABC 是"定义接口"的方式。子类必须实现所有 `@abstractmethod` 标记的方法，否则**实例化时**就报错：

```python
import abc

class Animal(abc.ABC):
    @abc.abstractmethod
    def speak(self) -> str: ...

class Dog(Animal):
    def speak(self) -> str:
        return "woof"

Dog()     # OK
Animal()  # TypeError: Can't instantiate abstract class
```

> 类比：Java 的 `interface` 继承体系或 C++ 的纯虚函数层级。
>
> Dressage 用法：`Paddock` 在 `dressage/paddock/interface.py` 定义了三层 ABC 结构（`Paddock` → `BlackboxPaddock` / `WhiteboxPaddock`），所有方法均为 `async def`。完整接口定义见 Part 2 决策 4。

### C. dataclass

`@dataclass` 装饰器自动给类生成 `__init__`/`__repr__`/`__eq__`：

```python
from dataclasses import dataclass

@dataclass
class TurnRecord:
    messages_snapshot: list
    output_tokens: list[int]
    output_logprobs: list[float] | None
    input_token_count: int

t = TurnRecord(messages_snapshot=[...], output_tokens=[1,2,3],
               output_logprobs=None, input_token_count=10)
print(t)  # TurnRecord(messages_snapshot=[...], ...)
```

> 类比：C 的 struct、TypeScript 的 interface（但 dataclass 真的会跑代码）。

> Dressage 用法：所有数据结构（`Trajectory`、`TrajectorySegment`、`TurnRecord`、`ChatCompletionResult`）都是 dataclass。轻量、零样板。

### D. 异步（async / await / asyncio）

普通函数：调用 → 跑完 → 返回。
**协程函数**（`async def`）：调用得到一个"协程对象"，必须由 event loop 调度才执行。`await` 是"暂停当前协程、把控制权还给 event loop、等被 await 的事情完成"。

```python
import asyncio

async def fetch_one(url):
    print(f"start {url}")
    await asyncio.sleep(1)        # 模拟 I/O
    print(f"done {url}")
    return f"<{url}>"

async def main():
    # 同时跑 3 个，总耗时 1 秒而不是 3 秒
    results = await asyncio.gather(
        fetch_one("a"),
        fetch_one("b"),
        fetch_one("c"),
    )

asyncio.run(main())
```

**为什么 Dressage 必须用 async**：rollout 阶段同时有几十条 trajectory 在跑，每条都在等 LLM 返回（几百毫秒到几秒）。如果用同步串行，32 条要 32 倍的时间。用 async 同时发起、各自等待，GPU 利用率就上去了。

**`asyncio.Semaphore(N)`** 是个"并发上限令牌池"：

```python
sem = asyncio.Semaphore(32)

async def run_one(sample):
    async with sem:          # 拿令牌，最多 32 个同时持有
        return await do_work(sample)
```

**`asyncio.to_thread(sync_fn, *args)`** 把同步函数扔进线程池跑，让 async 代码可以"无阻塞地"调阻塞 API：

```python
async def do_work():
    # paddock.tool_call 是同步函数，可能阻塞几秒
    # 用 to_thread 包一下，event loop 不会被卡
    tool_response = await paddock.tool_call(traj_id, paddock_tool_id, paddock_args)
```

> Dressage 用法：rollout 主循环（`fully_async_rollout.py` / `partial_async_rollout.py` / `sync_rollout.py`、`generate/whitebox_agent.py`）全用 async。Paddock 接口也是 `async def`（包括黑盒的 `register_agent` / `call_agent` / `execute_cmd` 和白盒的 `tool_call`），调用时直接 `await`。

### E. HTTP 服务 / FastAPI

FastAPI 是 Python 的现代 HTTP 服务框架。一个最小例子：

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class HelloRequest(BaseModel):
    name: str

@app.post("/hello")
async def hello(req: HelloRequest):
    return {"message": f"hi, {req.name}"}
```

启动后访问 `POST /hello` 带 JSON body `{"name": "alice"}` 就返回 `{"message": "hi, alice"}`。FastAPI 自动用 pydantic 校验请求 body 的类型——`req.name` 一定是 str，不需要手写校验。

> Dressage 用法：`dressage-proxy` 命令启动的就是一个 FastAPI app，对外开三个端点：`POST /v1/chat/completions`（采样）、`POST /session/finalize`（结算）、`POST /trajectory/read`（取数据）。代码在 `dressage/proxy/server.py`。

### F. httpx 异步 HTTP 客户端

`requests` 是同步的，不能用在 async 代码里。`httpx` 提供 async 接口：

```python
import httpx

async def call_proxy(payload):
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:8800/v1/chat/completions",
                                  json=payload)
        return resp.json()
```

> Dressage 用法：`ProxyClient` 用 httpx 调 proxy（`dressage/proxy/proxy_client.py`），`SGLangRouterClient` 用 httpx 调 SGLang Router（`dressage/proxy/sglang_client.py`）。

### G. 反向代理（reverse proxy）

"代理"在网络里有两种：
- **正向代理**：客户端通过代理访问外网（公司翻墙 VPN、`HTTP_PROXY=...`）。
- **反向代理**：服务端在真服务前加一层中间人（Nginx 把请求转给后面真正的应用服务器）。反向代理的好处是：在中间做日志、缓存、限流、协议转换，不用改后面应用的代码。

**Dressage 的 proxy 是一种反向代理**：

```
caller → (HTTP) → dressage-proxy → (HTTP) → SGLang
                  ↑
                  中间在这里偷偷
                  记 traj_id session、强开 logprob、解析 tool_call
```

caller 看到的是"OpenAI 兼容的端点"，不知道背后是 SGLang；proxy 看到的是 OpenAI 协议的请求；SGLang 看到的是它自己的原生协议。三层各自单职责。

### H. 动态模块加载（importlib）

`import foo.bar` 是静态的（写代码时就知道路径）。`importlib.import_module("foo.bar")` 是动态的（字符串里的路径运行时才确定）：

```python
import importlib

# dressage/paddock/factory.py 的 create_paddock_from_env() 内部：
# 默认按 DRESSAGE_PADDOCK_MODE 选内置实现；DRESSAGE_PADDOCK_CLASS 是高级覆盖，
# 设了它就动态加载指定的类路径：
class_path = os.environ.get("DRESSAGE_PADDOCK_CLASS")
if class_path:
    module_path, _, attr = class_path.rpartition(".")
    paddock_cls = getattr(importlib.import_module(module_path), attr)
```

> Dressage 用法：
> - `create_paddock_from_env()` 在 `dressage/paddock/factory.py` 中默认按 `DRESSAGE_PADDOCK_MODE`（`blackbox`/`whitebox`）创建内置 Paddock；设了 `DRESSAGE_PADDOCK_CLASS` 时则动态加载自定义类。
> - slime 加载 Dressage 插件函数（如 `dressage.rollout.fully_async_rollout.generate_rollout_fully_async`）也是这个套路——slime 在命令行收到一个字符串，运行时 `importlib` + `getattr` 拿到对应的函数对象。

### I. git submodule

git submodule 是"在一个 git 仓库里嵌另一个 git 仓库"。父仓库里只记录"应该 checkout 子仓库的哪个 commit"，**不复制子仓库的文件**。第一次 clone 完父仓库后，子仓库目录是**空的**，必须再执行 `git submodule update --init` 才会真的把子仓库下载下来。

```bash
git clone https://github.com/your-org/dressage.git    # slime/ 是空目录
ls slime/                                             # 空
git submodule update --init                            # 真的下载 slime
ls slime/                                             # 一堆文件
```

> Dressage 用法：`slime/` 目录就是 git submodule。文档和测试假设它**可能没下载**（"standalone fallback types" 分支），但跑训练前必须 `git submodule update --init`。

### J. 进程 vs 线程 vs 协程（一句话区分）

| 单位 | 隔离强度 | 切换开销 | 适合 |
|---|---|---|---|
| **进程（process）** | 独立内存 | 重 | 完全独立的服务（slime / proxy / SGLang 各一个进程） |
| **线程（thread）** | 共享内存 | 中 | 同进程内"调用阻塞库"（Paddock 实现） |
| **协程（coroutine, asyncio）** | 共享内存 + 单线程 | 极轻 | 同进程内大量 I/O 等待（rollout 32 并发） |

Dressage 同时用到两种并发：
- 三个独立**进程**（slime / proxy / SGLang）通过 HTTP 通信。
- proxy 和 rollout 用**协程**（asyncio）做高并发。Paddock 接口本身也是 `async def`，无需额外线程池。

### K. 环境变量

POSIX 系统每个进程有一组键值对的环境变量。`os.environ.get("FOO", "default")` 读取。Dressage 用环境变量做"启动时配置"——比传命令行参数更方便（不需要改启动脚本就能切实现）：

```bash
export DRESSAGE_PADDOCK_MODE=blackbox            # blackbox / whitebox
export DRESSAGE_SANDBOX_PROVIDER=local_bwrap     # local_bwrap / e2b
export DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS=64       # 后台并发的 group 数上限
bash examples/scripts/run_example_qwen3.5_4b_async_local.sh
```

> Dressage 的运行时配置环境变量分散在 `dressage/config/config.py` 和示例脚本头部；后面 §4C / §6 也会列常用项。

读完这些前置知识后，下面的 §0.1 - §0.15 就可以专心讲大模型和 RL 了。

## 0.1 大模型本质：一个超大型的"接龙函数"

把大语言模型（Large Language Model, LLM）想象成一个**超大型的 autocomplete 函数**：

```python
def llm(context: str) -> dict[str, float]:
    """
    输入：一段文本上下文，比如 "今天天气真"
    输出：词表里每个词作为"下一个词"的概率，比如
          {"好": 0.45, "棒": 0.20, "差": 0.10, "热": 0.08, ...}
    """
    ...
```

它"知道"的所有信息，都藏在它的**参数（weights / 权重 / θ）**里——通常是几十亿到几千亿个浮点数。训练就是不停地调整这些数字，让模型在"合理的上下文"里给"合理的下一个词"高概率。

**关键事实**：

- 模型本身**不"理解"语言**，它只是一个**概率分布生成器**。
- 这个函数极其大。GPT-3 有 1750 亿参数（约 700GB 内存，远超单卡 GPU 显存），需要分布式 GPU 才能跑得动。
- 训练好的模型，只要给它好的"上文"，配合采样策略，就能续写出连贯的"下文"。

> **程序员类比**：把模型想象成一个查表的纯函数 `(context) → probabilities`。这个表无限大、查不完，但模型用神经网络"算"出查表结果。训练就是修改神经网络的权重，使"查表结果"越来越像我们期望的分布。

## 0.2 Tokenizer：文本 ↔ 数字 ID

模型不直接处理字符串，要先把文本切成 **token**（小块单位）再映射成整数 ID：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

ids = tokenizer.encode("今天天气真好")
# → [109198, 110125, 99500, 108386]
#    每个数字是这个 token 在词表里的 ID（具体数字看模型）

text = tokenizer.decode(ids)
# → "今天天气真好"
```

几个关键名词：

- **词表大小（vocabulary size, V）**：通常 3 万到 20 万。
- token 不是字、不是词，是一个 **subword unit**——常见词是 1 个 token，生僻词被切成多个。
- 不同模型的 tokenizer **不通用**（Qwen 的 token 在 LLaMA 里没意义）。
- 模型最后一层输出长度 = 词表大小 V，每个位置对应一个候选 token 的"分数"。

**chat template（聊天模板）**：聊天模型规定了"多轮对话应该怎么拼成单条 prompt"。例如 Qwen 的模板：

```python
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮您？"},
    {"role": "user", "content": "1+1 等于几？"}
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
# → "<|im_start|>user\n你好<|im_end|>\n
#    <|im_start|>assistant\n你好，有什么可以帮您？<|im_end|>\n
#    <|im_start|>user\n1+1 等于几？<|im_end|>\n
#    <|im_start|>assistant\n"
```

最后留着 `<|im_start|>assistant\n` 当占位符，等模型续写。`<|im_start|>` 这些是**特殊 token**——它们告诉模型"现在角色切换了"。

> **为什么 Dressage 关心 chat template**：proxy 把 messages 拼成 prompt 必须走 chat template。如果你绕过它手动拼字符串，token 数对不上、特殊 token 漏了，模型会输出乱码或 SessionManager 会算错 turn 边界。

## 0.3 模型怎么"生成"文本：autoregressive

LLM 生成是 **autoregressive（自回归）**的——一次出一个 token，再把它接到 context 后面，循环。伪代码：

```python
def generate(prompt: str, max_tokens: int = 100) -> str:
    tokens = tokenizer.encode(prompt)
    for _ in range(max_tokens):
        logits = model(tokens)           # 前向：返回每个候选 token 的分数
        probs = softmax(logits[-1])      # 转成概率分布（只看最后一个位置）
        next_token = sample(probs)       # 从分布里抽一个 token
        tokens.append(next_token)
        if next_token == EOS_TOKEN:      # 模型自己决定"我说完了"
            break
    return tokenizer.decode(tokens)
```

涉及的术语：

- **logits**：模型最后一层的输出，每个候选 token 一个实数。**没归一化**（可能是负的、可能很大）。
- **softmax**：标准的归一化方法。`P_i = exp(z_i) / Σ_j exp(z_j)`。所有 P_i ≥ 0、求和 = 1。
- **采样（sampling）**：怎么从概率分布里抽 token。常见做法：
  - **贪心（greedy）**：永远选概率最大的（确定性，无多样性）。
  - **temperature T**：把 logits 除以 T，T<1 让分布更尖（保守）、T>1 更平（多样）。
  - **top-p（nucleus）**：只在"累计概率 ≤ p 的那些 token"里抽（截断长尾）。
  - **top-k**：只在"概率前 k 个 token"里抽。
- **EOS token**：end-of-sequence，模型词表里的特殊 token，模型一旦采到它就停。
- **max_tokens**：硬上限，防止模型一直生成不停。

> **关键观察**：生成一个 100 token 的回答，需要前向模型 100 次。每次前向都是 GB 级的矩阵乘法。这就是为什么 LLM 推理慢、为什么 RL 训练里"rollout 阶段"是瓶颈。

## 0.4 模型怎么"学"：训练循环

> **本节要解决的问题**：上一节我们说"模型是个有几百亿参数的概率分布生成器"。这些数字怎么变成"知识"的？答案是**训练**——一种最基本的、所有现代神经网络都用的循环。理解这个循环，你就理解了"训练"在程序员视角到底是什么操作。

训练 = **调整模型参数，使在训练数据上输出"我们期望的"分布**。一步训练的伪代码：

```python
for batch in dataloader:                  # 每次拿一小批数据
    inputs, targets = batch
    logits = model(inputs)                # ① 前向：算预测
    loss = loss_fn(logits, targets)       # ② 算"错得多离谱"
    loss.backward()                       # ③ 反向：算每个参数的梯度
    optimizer.step()                      # ④ 用梯度更新参数
    optimizer.zero_grad()                 # ⑤ 清空梯度，准备下一步
```

涉及的术语，配上程序员能 grok 的类比：

| 名词 | 一句话 | 类比 |
|---|---|---|
| **loss（损失）** | 一个标量数字，"现在错得多离谱"。越大越糟。 | 单元测试失败数 |
| **gradient（梯度）** | loss 对每个参数的偏导：参数变大会让 loss 变大还是变小？多快？ | 多维曲面的最陡下降方向 |
| **backpropagation** | 自动算每个参数的梯度。`loss.backward()` 一行搞定。 | 链式法则 + 拓扑排序 |
| **optimizer** | 怎么用梯度更新参数。最简单是 SGD：`θ ← θ - lr * gradient`。 | 参数更新调度策略 |
| **learning rate (lr)** | 每次更新的步长。 | 二分查找的步长选择 |
| **batch** | 一次更新用的样本数。越大越稳定，但越占显存。 | 多线程的 batch size |

最常见的 loss：**交叉熵（cross-entropy）**。如果"正确的下一个 token"是 `t`，loss 就是 `-log P(t | context)`：

- 模型给 `t` 高概率 → log 接近 0 → loss 小（接近 0）。
- 模型给 `t` 低概率 → log 很负 → loss 大。

> **关键点**：训练 loop 的"形状"和写普通深度学习一样。**RL 训练特殊在哪？特殊在 loss 怎么算**——不是"预测 vs 标签的交叉熵"，而是"采样 + reward + 策略梯度公式"。这是 §0.7 - §0.9 的主题。

## 0.5 三种训练范式

| 范式 | 数据形式 | loss |
|---|---|---|
| **预训练（pre-training）** | 海量纯文本（万亿 token） | 交叉熵 `-log P(下一个 token)`。无人工标注。 |
| **SFT（监督微调，Supervised Fine-Tuning）** | 高质量"问题→答案"对 | 还是交叉熵，但**只在"答案"部分算 loss**（prompt 不算）。 |
| **RL / RLHF** | "问题"+"事后评分函数"，**没标准答案** | 策略梯度，下一节开讲。 |

为什么需要 RL？因为很多任务**没有标准答案**：

- "写一段优雅的代码"——什么叫优雅？很难写出标准答案。
- "agent 完成了任务"——任务过程可能千奇百怪，没办法逐 token 监督。
- "回答让用户满意"——"满意"是事后整体判断的标量。

**RL 的能力**：只要能给一条完整回答打一个分（reward），就能训练模型。这一点对 agent 训练尤其关键——agent 任务的"过程"千变万化，只有"最终结果"能稳定打分。

## 0.6 强化学习的核心概念

经典 RL 用的名词（先记一下，下面要反复用）：

| 名词 | RL 通用含义 | 在 LLM 训练里 |
|---|---|---|
| **agent（智能体）** | 做决策的家伙 | 模型本身 |
| **environment（环境）** | agent 行动的世界 | 用户 + 工具 + 沙箱 |
| **state（状态）s** | 当前情况 | 当前上下文（已有的 token） |
| **action（动作）a** | agent 能做的事 | 输出下一个 token |
| **reward（奖励）r** | 动作好坏的标量反馈 | 整条回答跑完后给一个分 |
| **policy（策略）π** | state → action 概率分布的函数 | **模型本身**——给上下文，输出每个 token 的概率 |
| **trajectory（轨迹）τ** | 一次完整的 state-action-reward 序列 | 一次完整的对话/回答 |

**RL 的优化目标**：调整 policy 的参数 θ，让"期望累积 reward"最大化：

```
J(θ) = E_{trajectory ~ π_θ} [ R(trajectory) ]
```

我们想往"梯度上升"方向走，让 reward 越变越大。这就是**策略梯度（policy gradient）**算法的目标。

> **程序员视角**：把 RL 想象成"调一个函数的参数，让它的输出能拿到更高分"。难点是"分数"不是直接可导的——你不能 `score.backward()`——所以要用一个特殊的公式把它转化成可导的形式（下一节）。

## 0.7 策略梯度直觉：怎么从 reward 反推到 token

> **本节要解决的问题**：在 §0.4 我们看到训练靠"loss 反传 + 梯度下降"。但 RL 里没有"标准答案"，**只有一个事后给的整体分数**。怎么把这个分数变成"每个 token 该往哪边走"的梯度信号？这一节给你一个最朴素但够用的直觉版答案——后面 PPO/GRPO 都是在这个基础上做工程优化。
>
> 这一节的数学密度是整个文档里相对最高的（但只到求导链式法则）。如果只读两段：把上面一段口语化的总结记住，再读下面"中文翻译"那段。

这是 RL 最反直觉的一步：**reward 是回答级的（一个标量），但梯度要对每个 token 算**（梯度下降必须）。怎么办？

**朴素策略梯度（vanilla policy gradient）**的核心公式（不严格推导，只给直觉，数学叫 "log derivative trick"）：

```
∇_θ J(θ)  ≈  R · Σ_t  ∇_θ log π_θ(a_t | s_t)
                           ↑
                       每一步采样 token 的 logprob 对 θ 的梯度
```

中文翻译：

> **"采一条回答，记下每个 token 的 logprob（log 概率）。如果整条回答 reward 高，就把这些 logprob 全部往上推（推到接近 0，即提高概率）；reward 低，就往下推。"**

举例：
- prompt 是 "1+1=?"，模型采样出 "答案是 2"（reward=1.0）。
- 把 "答案"、"是"、"2" 这三个 token 的 logprob 各往 0 推一点。
- 反传几次后，模型对这个 prompt 给 "答案是 2" 的概率变高。
- 反之采样出 "答案是 11"（reward=0.0），把这串 token 的 logprob 往下推。

**这就是 RL 的整个核心**。后面 PPO/GRPO 都是在解决"朴素策略梯度的两个工程问题"。

### 问题 1：方差太大

同一个 prompt 采 10 次，可能 reward 一会儿 1.0 一会儿 0.0 一会儿 0.3，梯度跳来跳去，训练很难收敛。

**解决：用 baseline**。把 `R` 换成 `R − b`（叫 **advantage 优势**）：

- 比平均**好** → `R − b > 0` → 提高它的概率
- 比平均**差** → `R − b < 0` → 降低它的概率

最经典做法：再训一个 **value model `V(s)`** 来预测 baseline。但这成本不低（value model 通常和 policy model 一样大）。GRPO 会省掉这一步（§0.9）。

### 问题 2：每次梯度更新都要重新采样

朴素策略梯度是 **on-policy** 的：梯度公式里的"采样分布"必须是**当前正在被优化的策略 π_θ**。

但 LLM 训练里：
- **rollout（采样）是慢操作**：每条 trajectory 要跑很多 token + 工具调用，几秒到几分钟。
- **梯度更新是快操作**：GPU 矩阵乘法，几十毫秒一次。

如果每次小批量梯度更新都要重新采样，GPU 利用率会糟透。

**解决：用 PPO 的重要性采样**。下一节。

## 0.8 PPO：让一份采样数据能多次复用

**PPO（Proximal Policy Optimization，近端策略优化）**是目前 RL 训练的主流算法。它干三件事：

### A. 重要性采样比（importance sampling ratio）

我们想用"一段时间前用 `π_old` 采的数据"来训练 `π_new`。统计学要求乘一个修正因子：

```
ratio_t = π_new(a_t | s_t) / π_old(a_t | s_t)
        = exp( logprob_new(a_t | s_t) − logprob_old(a_t | s_t) )
                       ↑                          ↑
                  训练时重新算的            采样时记下来的
```

PPO 损失就用这个 ratio：

```
L_PPO = E[ ratio_t · A_t ]
```

含义：

- `A_t > 0` 且 `ratio > 1`（新策略更喜欢这个 token）→ loss 是负的，梯度推 ratio 继续上升 ✓
- `A_t > 0` 且 `ratio < 1`（新策略反而不喜欢）→ loss 是正的，梯度推 ratio 升回去 ✓
- `A_t < 0` 同理反过来。

**关键观察**：`logprob_new` 训练时算，`logprob_old` 采样时存。所以 Dressage 必须在 rollout 阶段把每个 token 的 logprob 存下来——这是 §0.10 的主题。

### B. Clip（裁剪）：防止步子迈太大

如果 `π_new` 和 `π_old` 偏离太多，重要性采样会失真——一个本来概率 0.001 的 token 在新策略里突然变成 0.5，ratio = 500，梯度爆炸。

PPO 加了一个简单的限制：

```
L_PPO_clip = E[ min( ratio_t · A_t , clip(ratio_t, 1-ε, 1+ε) · A_t ) ]
```

`ε` 通常是 0.2，意思是"ratio 被强制锁在 `[0.8, 1.2]` 范围内"。当 ratio 跑出去太远时，clip 把它卡住，梯度变成 0（因为 `min` 选了被裁过的那一支），相当于"这步太激进了，先不更新"。

Dressage 启动脚本里 `--eps-clip 0.2` 就是这个意思。

### C. KL 散度：限制偏离参考模型

光 clip 还不够。有些做法（包括 Dressage 默认）会再加一个 KL 散度惩罚项：

```
L = L_PPO_clip + β · KL(π_new ‖ π_ref)
```

`π_ref` 是参考模型（一般是 SFT 后、RL 开始前的版本快照）。这一项防止 RL 把模型训得离原始能力太远（典型现象：reward 上去了但人类觉得回答变怪了，叫 "reward hacking"）。

近似计算（Schulman 的"低方差"近似，Dressage 默认用这个）：

```
k = exp(log_ratio) - 1 - log_ratio,  log_ratio = logprob_policy - logprob_ref
KL ≈ E[k]
```

启动脚本里的 `--use-kl-loss --kl-loss-type low_var_kl` 就是这件事。

## 0.9 GRPO：省掉 value model 的 PPO 变种

**GRPO（Group Relative Policy Optimization）** 是 DeepSeek 团队提出的 PPO 变种，**Dressage 默认用这个**。

### 痛点：value model 太贵

PPO 的 advantage `A_t = R_t − V(s_t)` 需要一个 value model 估计 `V(s)`。在 LLM 场景里 value model 通常和 policy model 一样大（甚至更大），每步要算两次前向，显存和算力都翻倍。

### GRPO 的核心想法

> **同一个 prompt 采 N 条回答（一个"组"），用组内平均当 baseline。**

```
对 prompt p：
  采 N 条回答 → 拿到 N 个 reward: r_1, r_2, ..., r_N
  组内归一化:
    A_i = (r_i - mean({r_j})) / std({r_j})    ← 可选除标准差
  把 A_i 当成"这条回答里每个 token 的 advantage"
```

直觉：
- 一条回答**比组内平均好** → advantage > 0 → 推高它所有 token 的概率。
- **比平均差** → advantage < 0 → 推低。

**省掉了 value 模型！** 同组样本互相当 baseline。代价是要为每个 prompt 多采几条（典型 N = 4~16），rollout 量变大，但 LLM 推理本来就比 value 模型前向便宜，整体还是合算。

这就是 Dressage 启动脚本里 `--n-samples-per-prompt 4` 和 `--advantage-estimator grpo` 的含义。

### GRPO 完整损失

把 PPO + 组归一化 + KL 拼起来：

```
L_GRPO = - E[ min( ratio_t · A_grpo, clip(ratio_t, 1-ε, 1+ε) · A_grpo ) ]
         + β · KL(π_new ‖ π_ref)

         其中 A_grpo = (r - mean(r_group)) / std(r_group)
              ratio_t = exp(logprob_new(a_t) - logprob_old(a_t))
```

每一项你都见过了，拼起来就是。

> **符号约定**：公式开头的负号把"要最大化的 RL 目标"转换为"要最小化的训练 loss"——框架做梯度下降最小化 loss，等价于最大化原始目标。`kl_loss` 前面是正号（要 KL 小），`entropy_loss` 前面是负号（要熵大）。

### 一个完整的小例子

假设：
- prompt = "求 1+1 = ?"
- `n_samples_per_prompt = 4`
- reward 函数：回答包含正确答案给 1.0，否则 0.0

**Step 1：rollout 采 4 条**

| 样本 | response | reward |
|---|---|---|
| s0 | "答案是 2" | 1.0 ✓ |
| s1 | "答案是 3" | 0.0 ✗ |
| s2 | "应该是 2" | 1.0 ✓ |
| s3 | "答案是 11" | 0.0 ✗ |

**Step 2：组内归一化**

```
rewards = [1.0, 0.0, 1.0, 0.0]
mean    = 0.5
center  = [+0.5, -0.5, +0.5, -0.5]
std     = sqrt( mean([0.25, 0.25, 0.25, 0.25]) ) = 0.5
A_grpo  = [+1.0, -1.0, +1.0, -1.0]
```

**Step 3：训练**

- s0、s2 拿到 advantage = +1.0 → 推高它们 token 的概率。
- s1、s3 拿到 advantage = −1.0 → 推低它们 token 的概率。

下一轮 rollout 时，模型对这个 prompt 输出 "2" 的概率上升。

**Step 4：循环**

下一轮 rollout 用新权重采 4 条 → 归一化 → 训练 → 直到收敛。

## 0.10 logprob 详解

> **本节要解决的问题**：在 §0.8 我们看到 PPO 损失里有个 `ratio = exp(logprob_new − logprob_old)`，其中 `logprob_old` 是"采样时的概率"。这个数字**采样的瞬间就必须存下来**，否则事后没法重现（模型权重更新过了，再次跑同样 input 出来的概率已经不同了）。Dressage 的整个 proxy 设计有一半是为了**保证这个数字一定能被存下来**。所以这一节专门讲 logprob——你后面看 proxy 代码时会一直碰到它。

这一节专门讲 logprob，因为它是 Dressage 里**最核心的工程数据**。

**定义**：`logprob = log P(token | context)`，模型采样某 token 时它的"对数概率"。

**性质**：
- 概率 ≤ 1 → log 概率 ≤ 0，**永远是非正数**。
- 高概率 token（模型很有信心）→ logprob 接近 0（如 -0.1）。
- 低概率 token（模型其实不太想选）→ 很负（如 -8）。
- 后面会看到，**"logprob 越接近 0" 等价于 "模型越喜欢这个 token"**。

**怎么算（数学）**：

```
logit = [z_1, z_2, ..., z_V]              # 模型最后一层
P_i = exp(z_i) / Σ_j exp(z_j)             # softmax
logprob_i = log P_i = z_i - log Σ_j exp(z_j)
                              ↑
                          log-sum-exp（数值稳定的实现）
```

**为什么 RL 总用 log**：概率连乘容易下溢、求导不方便；取 log 把乘变成加、把幂变成乘，数值稳定且导数简单。

**举个数字例子**：假设词表只有 5 个 token，某一步 logits = `[2.0, 0.5, 1.0, -1.0, 0.0]`：

```
exp(z)        : [7.39, 1.65, 2.72, 0.37, 1.00]
Σ exp         : 13.13
P             : [0.563, 0.126, 0.207, 0.028, 0.076]
logprob       : [-0.575, -2.075, -1.575, -3.575, -2.575]
```

采样器抽到 token_3（概率 0.207）→ 这一步的 logprob 是 -1.575。整条回答有 T 个 token 就有 T 个 logprob。

**RL 训练为什么必须存它**：

- PPO ratio = `exp(logprob_new − logprob_old)`
- `logprob_old` 是**采样时的概率**，**事后无法重现**——模型权重更新过了，再次跑同样的 input，算出的概率已经不同。
- 所以**必须在采样的当下**存下来，跟着 token 一起带回训练侧。

**Dressage 怎么获取它**：

SGLang 在 `/generate` 调用里接受 `return_logprob=True` 参数，返回每个生成 token 的 logprob 数组。Dressage 的 SGLang 客户端（`dressage/proxy/sglang_client.py`）强制开启这个开关：

```python
payload = {
    "input_ids": input_ids,
    "sampling_params": sampling_params,
    "return_logprob": True,                # ★ 强制 True
}
if return_logprob:
    payload["logprob_start_len"] = 0      # 从第 0 个 token 开始捞
    payload["top_logprobs_num"] = 0        # 只要被采样 token 的 logprob

# 调 SGLang
resp = await client.post("/generate", json=payload)
data = resp.json()

# 抠出 logprob
meta = data.get("meta_info", {})
logprobs = meta.get("output_token_logprobs")    # 长度 == output_ids 长度
```

然后 SessionManager 把它和 token 一起记下来（`dressage/proxy/session_manager.py` 的 `record_step` 方法），最终落到 `TrajectorySegment.full_logprobs`。

**logprob 缺失的处理**：如果一条回答的 logprob 没存下来（比如黑盒 agent 用了别的模型，proxy 收不到该模型的 logprob），Dressage 会把这个样本打 `remove_sample=True`，让它的 loss_mask 全置 0——只让 reward 参与组归一化，不让它贡献策略梯度。代码在 `dressage/rollout/convert_samples.py` + `dressage/training/reward_post_process.py`。

## 0.11 loss_mask 详解

> **本节要解决的问题**：在一段多轮 agent 对话里，**最终的 token 序列由两类来源混合而成**——模型自己生成的（assistant turn）和环境塞进来的（tool turn / 用户中途追加的话）。前者要训（这是模型该学的输出分布），**后者绝对不能训**（如果你让模型"预测工具输出"，它会学着幻想出本来不存在的 tool 返回值——这叫 reward hacking 的一种形式）。loss_mask 就是回答"哪些 token 算 loss 哪些不算"的位标志。理解它你就理解了 Dressage 怎么把"多轮 agent 训练"和"单轮 SFT"做出根本区别。

loss_mask 是一个长度 == `response_length` 的 0/1 数组，告诉训练器"这个位置的 token 要不要算 loss"。

**为什么需要它**：多轮对话场景，response 区里既有"模型自己生成的 token"（assistant turn），也有"环境塞进来的 token"（tool turn / 用户中途追加的话）。

- 前者要训（学着生成更好的输出）→ mask = 1
- 后者**不能训**（否则等于教模型"预测工具输出"，会产生幻觉）→ mask = 0

举例，一条 2-turn 对话：

```
[prompt 区] 用户问 "1+1=?"
              ↑ 整个 prompt 区都不算 loss（SFT/RL 都不在 prompt 上训）

[response 区开始]

[turn 1 assistant] 模型生成 <tool_call>{"name": "bash", "args": "echo 2"}</tool_call>
                              ↑ 这些 token loss_mask = 1
[turn 1 tool] 工具返回 "2"
                ↑ 这些 token loss_mask = 0   ← 关键
[turn 2 assistant] 模型生成 "答案是 2"
                              ↑ 这些 token loss_mask = 1
```

**slime 的硬约束**：`assert len(loss_mask) == response_length`——loss_mask **不包含 prompt 区**，长度严格 = response 区的 token 数。

**Dressage 的实现**：loss_mask 由 `PromptAssistantMaskBuilder`（在 `dressage/proxy/last_step/prompt_assistant_mask.py`）在 proxy 构段时生成。它不手动切 turn 边界，而是用一份"mask-only chat template"把这段对话重放一遍，让 tokenizer 通过 `return_assistant_tokens_mask=True` 直接标出 assistant token（详见 §5.1）。概念上等价于：

```python
# 概念示意：只有 assistant 生成的 token 置 1，prompt / tool / user 注入的 token 置 0
response_mask = tokenizer.apply_chat_template(
    messages, chat_template=mask_only_template,
    return_dict=True, return_assistant_tokens_mask=True,
)["assistant_masks"]
# 若 mask 模板渲染与正式模板不一致，则退化为 output-only mask（prompt 全 0、response 全 1）
```

**loss_mask 怎么进入 loss 计算**（在 slime 那一侧）：

```
L_per_token = ratio_t · A_grpo · loss_mask_t
loss        = sum(L_per_token) / sum(loss_mask)
```

`loss_mask_t = 0` 的位置完全不贡献梯度。

## 0.12 Agent 是什么：从 chatbot 升级

普通 chatbot 交互：

```
用户: 你好
模型: 你好！有什么可以帮您？
用户: 再见
模型: 再见！
```

模型只输出文本，不能"做事"。

**agent** 不一样：

```
用户: 帮我把工作目录里所有 .py 文件的行数加起来
模型: <tool_call>{"name": "bash", "args": {"cmd": "ls *.py"}}</tool_call>
工具: a.py  b.py  c.py
模型: <tool_call>{"name": "bash", "args": {"cmd": "wc -l a.py b.py c.py"}}</tool_call>
工具: 10 a.py
      20 b.py
      30 c.py
      60 total
模型: 一共 60 行
```

模型输出了**结构化的"工具调用"指令**，外部执行后把结果塞回对话，模型再继续。这就是 agent。

**程序员视角：agent 就是一个 while 循环 + 函数调用 dispatch**：

```python
messages = [{"role": "user", "content": user_prompt}]
for turn in range(max_turns):
    response = llm.chat(messages)                # 调模型
    messages.append({"role": "assistant", **response})
    if not response.tool_calls:
        break                                     # 模型说完了，结束
    for tc in response.tool_calls:
        result = execute_tool(tc.name, tc.args)   # 调工具
        messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
```

Dressage 的 whitebox 模式就是上面这段伪代码的真实版本——`WhiteboxAgent` 类在 `dressage/rollout/generate/whitebox_agent.py`，子类实现 `rollout()` 方法驱动对话，框架自动管理 session id 和 trajectory 回收。

> **训练 agent 比训练 chatbot 难在哪？**
> 1. 多轮——loss_mask 必须正确区分 assistant token 和 tool token（§0.11）。
> 2. 工具执行——要有沙箱（不能让模型 `rm -rf /`），要并发可控。
> 3. 黑盒 agent——可能想训练别人写的 agent 框架（LangChain 等），它内部怎么调 LLM 不归我们管。
> 4. 多任务 reward——同一批数据可能混了数学题、代码题、写作题，每种打分不一样。

## 0.13 tool_call 的协议细节

模型输出 tool_call 的格式是**协议规定的**。常见的有两种：

- **OpenAI function calling**：模型在 JSON 响应里有专门的 `tool_calls` 字段。
- **Hermes 格式**：模型在文本里输出 `<tool_call>{"name": "bash", "arguments": {...}}</tool_call>`，由后处理解析。

Dressage 默认用 **Hermes 格式**。proxy 收到 SGLang 输出后用 `ProxyToolCallParser` 类解析（`dressage/proxy/tool_call_parser.py`），支持 `local`、`sglang_api`、`hybrid` 三种后端：

```python
# ProxyToolCallParser.parse() 的简化版逻辑：
def parse_tool_calls(text):
    tool_calls = []
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            name = parsed.get("name", parsed.get("function", ""))
            arguments = parsed.get("arguments", parsed.get("parameters", {}))
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                function=ToolCallFunction(name=str(name), arguments=str(arguments)),
            ))
        except (json.JSONDecodeError, KeyError):
            continue                              # 解析失败就跳过
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return cleaned_text, tool_calls
```

caller（whitebox_loop）拿到结构化的 `tool_calls` 之后，按 `function.name` 找到对应的 Paddock 工具执行——和 LLM 服务本身无关。

> **为什么 tool_call 解析放在 proxy 里？** 因为这样所有 caller（whitebox / blackbox / 未来任何新的 caller）都不用关心 SGLang 的原始输出格式。proxy 是单一职责的"OpenAI 协议适配层"。

## 0.14 Rollout / Trajectory

最后两个 RL 工程里的"批量调度单位"：

- **trajectory（轨迹）**：**一次完整的 agent 任务**，从 prompt 开始到 agent 说"做完了"或被截断。包含所有 turn、所有 token、所有 logprob、最终 reward。
- **rollout**：**一批 trajectory 并行跑**，跑完后把结果交给训练器做一次梯度更新。

伪代码（一次训练 step = 一次 rollout + 一次梯度更新）：

```python
def train_step():
    prompts = sample_prompts(batch_size)        # 拿一批 prompt
    trajectories = parallel_rollout(prompts)    # 并发跑 N 个 agent 任务
    rewards = [compute_reward(t) for t in trajectories]
    advantages = compute_grpo_advantage(rewards)
    loss = compute_ppo_loss(trajectories, advantages)
    loss.backward()
    optimizer.step()
```

Dressage 里：
- 1 个 rollout = 1 次 rollout 入口函数（如 `generate_rollout_fully_async`）调用。
- 1 条 trajectory = 1 个 `traj_id`（黑盒里就是 `session_id`）。

**rollout 是 RL 训练的吞吐瓶颈**：LLM 推理慢，要让 GPU 别空转，必须并发跑很多 trajectory。完全/部分异步模式下，后台 worker 用 `DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS`（默认 = `rollout_batch_size`）控制同时在跑的 prompt 组数。

## 0.15 名词速查表（汇总）

后面正文遇到这些名词如果忘了，回这里查：

### 大模型 / 训练

| 名词 | 一句话 |
|---|---|
| **token** | 模型的最小输入/输出单元，由 tokenizer 切分。 |
| **vocab size (V)** | 词表大小，模型最后一层输出维度。|
| **logits** | 模型最后一层输出，每个 token 一个未归一化实数。 |
| **softmax** | 把 logits 归一化为概率分布。 |
| **logprob** | `log P(token \| context)`，永远 ≤ 0。**RL 必存**。 |
| **chat template** | tokenizer 自带的"多轮 messages 拼成单条 prompt"的规则。 |
| **SFT** | 监督微调，交叉熵 loss，只在答案部分算。 |
| **gradient / backward** | loss 对每个参数的偏导，反向传播自动算。 |
| **optimizer (Adam/SGD)** | 怎么用梯度更新参数。 |
| **learning rate** | 每次更新的步长。 |
| **batch** | 一次更新用的样本数。 |

### 强化学习

| 名词 | 一句话 |
|---|---|
| **policy** | 模型本身，state → action 概率。 |
| **reward** | 一条回答跑完后给的标量打分。 |
| **advantage** | `reward − baseline`，作为梯度信号。 |
| **PPO** | 主流 RL 算法，核心是 logprob ratio + clip + KL。 |
| **GRPO** | PPO 变种，用组内平均当 baseline，省 value model。 |
| **KL 散度** | 衡量"新模型偏离参考模型多远"，作 loss 约束。 |
| **on-policy** | 采样和训练用同一个 policy（朴素 PG 这样）。 |
| **importance sampling** | 用 ratio 修正"采样分布 ≠ 训练分布"。 |
| **value model** | 估计"某个 state 的预期回报"，作为 baseline。GRPO 省掉了它。 |
| **rollout** | 批量跑 agent 任务收集数据的过程。 |

### Agent

| 名词 | 一句话 |
|---|---|
| **agent** | 会"决定下一步动作"的模型——能调工具、多轮交互。 |
| **tool call** | 模型输出"我要调用 X(args)"的结构化指令。 |
| **turn** | 模型生成一次叫一个 turn。 |
| **trajectory** | 一次完整的 agent 任务记录（一个 traj_id）。 |
| **segment** | trajectory 内部的训练数据单元（通常 1:1）。 |

### 训练框架

| 名词 | 一句话 |
|---|---|
| **slime** | 底层 RL 训练框架（git submodule），Dressage 不修改。 |
| **SGLang** | 高性能 LLM 推理引擎，Dressage 通过 proxy 调。 |
| **Megatron** | NVIDIA 的大模型分布式训练框架（TP/PP/CP/EP/DP 并行）。 |
| **Sample** | slime 的训练数据单元。 |
| **GRPO 组** | 同一 prompt 的 N 个采样副本（同 group_index）。 |
| **loss_mask** | response_length 长度的 0/1 数组，0 = 不算 loss。 |
| **TP/PP/DP** | Tensor / Pipeline / Data Parallel，Megatron 的三种并行方式。 |

### Dressage 自有

| 名词 | 一句话 |
|---|---|
| **Proxy** | FastAPI 服务，所有 LLM 调用经过它。 |
| **Paddock** | 抽象基类，统一环境/工具/黑盒 agent 的接口。 |
| **traj_id** | 全系统主键，串起 Paddock + proxy session + Trajectory + Sample。 |
| **白盒 agent** | agent 循环在 Dressage 里（`WhiteboxAgent` 类，`generate/whitebox_agent.py`）。 |
| **黑盒 agent** | 外部不透明的 agent，通过 `Paddock.call_agent` 调用。 |
| **reward 函数** | `fn(sample, *, args=None) → float`，用 `@register_reward` 注册。 |
| **Trajectory** | dataclass，reward 函数看的就是它。 |
| **TrajectorySegment** | dataclass，训练数据单元。 |

### 工程 / Python

| 名词 | 一句话 |
|---|---|
| **proxy（代理）** | 客户端和真服务之间的中间人。例：Nginx 反向代理、公司 VPN。 |
| **FastAPI** | Python 现代异步 HTTP 服务框架。 |
| **httpx / asyncio** | Python 异步 HTTP 库和异步运行时。 |
| **dataclass** | Python 轻量类，自动生成 `__init__`/`__repr__`/`__eq__`。 |
| **ABC（abstract base class）** | 抽象基类，子类必须实现 `@abstractmethod`。 |
| **git submodule** | git 项目里嵌套另一个 git 项目，`git submodule update --init` 才会下载。 |

---

# Part 1 — Dressage 是什么，要解决什么问题

## 1.1 一句话定位

> **Dressage 是一个"agentic RL 训练框架"，底座是 slime，在 slime 上加了一层 agent 能力，可以训练"会用工具、能多轮决策"的模型。**

"Agentic RL" 意思是"用强化学习训练能多轮决策、能用工具的智能体"——这是 2024-2025 年 LLM 训练的新方向。

## 1.2 从 RLHF 到 Agentic RL 的演进

传统 LLM 训练管线：

```
预训练           SFT              RLHF
─────           ─────            ──────
大量纯文本     问答对         "回答 A 比 B 好"
                            人/奖励模型打分
                                  │
                                  ▼
                          PPO 更新模型
```

RLHF 阶段，模型只要"输出一段话"就完事。但下一代 agent 训练要求：

```
prompt: "在这个 docker 里写个能通过测试的 fizzbuzz"
   │
   ▼
模型生成: <tool_call>{"name": "bash", "args": {"cmd": "ls"}}</tool_call>
   │
   ▼
环境返回: "test_fizzbuzz.py  README.md"
   │
   ▼
模型生成: <tool_call>{"name": "bash", "args": {"cmd": "cat test_fizzbuzz.py"}}</tool_call>
   │
   ▼
... (反复 N 次) ...
   │
   ▼
模型生成: <tool_call>{"name": "bash", "args": {"cmd": "pytest"}}</tool_call>
   │
   ▼
环境返回: "5 passed"
   │
   ▼
reward = 1.0
```

这就是 **agentic RL**。复杂的地方：

1. **多轮对话**——loss_mask 必须正确分配（§0.11）。
2. **要执行工具**（沙箱里跑 bash、Python 等），且要把工具结果安全追加回对话。
3. **可能要调度黑盒 agent**（让外部的 LangChain agent 或别的 LLM 来完成任务）。
4. **每条样本可能用不同的 reward**（数学题对答案、代码题跑测试、开放题用 LLM 评分）。
5. **每次 LLM 调用的 token 和 logprob 都要拿到**——这是 RL 算 ratio 的前提（§0.10）。

## 1.3 slime 是什么，缺什么

slime（[THUDM/slime](https://github.com/THUDM/slime)）是一个成熟的 RL 训练框架，作为 git submodule 挂在 Dressage 的 `slime/` 目录下（**当前 checkout 是空的，需要 `git submodule update --init` 才会下载**）。

slime 已经搞定的事：

- **Megatron 并行训练**（TP/PP/CP/EP/DP）
- **GRPO/GSPO/Reinforce++ baseline** advantage 估计
- **PPO clip / KL loss / entropy 正则**
- **SGLang 推理调度**（含 logprob 捕获）
- **Ray 编排**训练 worker 和 rollout worker
- **Checkpoint 转换**（HF ↔ Megatron `torch_dist`）
- **插件系统**：custom 模型、rollout 函数、数据源、loss、batch 转换

但 slime 默认假设"模型输出一段回答就完了"。上面 5 件 agent 化的事，slime 没有现成方案。

## 1.4 Dressage 怎么补齐

**完全通过 slime 的插件接口**，不修改 slime 一行代码：

```
┌──────────────────────────────┐
│   slime/  (git submodule)    │  ← 绝对不碰
│   - train.py 主循环           │
│   - Megatron 训练             │
│   - GRPO loss                │
│   - SGLang 推理调度           │
└────────────▲─────────────────┘
             │ slime 通过 --flag 加载
             │ dressage 的函数/类
┌────────────┴─────────────────┐
│   dressage/  (本仓库)         │  ← 我们写的
│   - rollout/   (rollout 入口) │
│   - proxy/     (LLM 入口)     │
│   - paddock/   (环境抽象)     │
│   - reward/    (奖励注册表)   │
│   - training/  (训练侧插件)   │
└──────────────────────────────┘
```

**集成机制**：`examples/scripts/run_example_*.sh` 启动 slime 时把插件路径作为 CLI flag：

```bash
ray job submit ... -- python3 train_async.py \
  --rollout-function-path     dressage.rollout.fully_async_rollout.generate_rollout_fully_async \
  --custom-generate-function-path \
                              dressage.rollout.generate.blackbox_dispatch.generate \
  --custom-rm-path            dressage.reward.custom_rm.custom_rm \
  --data-source-path          dressage.rollout.data_source.DressageDataSource \
  --custom-convert-samples-to-train-data-path \
                              dressage.rollout.convert_samples.convert_samples_to_train_data \
  --custom-rollout-log-function-path \
                              dressage.rollout.log_rollout.log_rollout_data \
  ...
```

slime 内部用 `importlib.import_module + getattr` 加载这些符号，在自己的训练循环里回调。我们只需要保证我们的函数签名符合 slime 的约定。

> **注意**：`convert_samples_to_train_data` 内部已经调用了 `reward_post_process`（GRPO 组归一化），因此**不需要**再单独注册 `--custom-reward-post-process-path`。重复注册会导致归一化被处理两次或跳过。

---

# Part 2 — 五个核心设计决策

理解 Dressage 的关键，是理解下面 5 个设计决策。每个决策**都不是**"这样设计很优雅"，而是"这样设计能解决某个具体问题"。

## 决策 1：用 slime 作 RL 底座，绝不修改

**问题**：从零写一个分布式 RL 训练框架要几千人月。Megatron 并行、checkpoint 管理、SGLang 推理调度都很复杂。

**决策**：把所有 RL/分布式的事委托给 slime。我们只通过 slime 的插件 hook 加 agent 能力。

**含义**：
- slime 的代码不动。`slime/` 是 git submodule，连改 import 路径都不允许。
- Dressage 写的所有 Python 函数都必须**符合 slime 期望的签名**（slime 用 `importlib` 加载，签名不对就崩）。
- 任何"想改 slime 行为"的需求，**优先找 slime 是否提供了 hook**；找不到再想别的办法（如 `train_async_with_rollout_pause.py` 中的 pause/resume 机制）。

> **类比**：像写一个 Django app——Django 框架不能改，你只能写满足 Django 约定的 view、model、middleware。Dressage 就是 slime 的"app"。

**如果不这样做会怎样**：你 fork slime、改它的代码加 agent 支持。一开始挺爽，能改就改。但 slime 在持续演进——每隔几周一个 PR 加新算法、修 bug。你想把上游更新合并进来，每次都得手动解 conflict，越拖越累，最后你的 fork 完全跟不上上游。这就是经典的"fork hell"。靠插件接口集成，slime 升级时**通常一行不用改**——这是用一点"接口适配"成本换长期可维护性。

## 决策 2：所有 LLM 调用收口到一个 proxy

**问题**：

1. RL 必须存 logprob（§0.10）——caller 各自调 SGLang 的话，每个 caller 都要记得设 `return_logprob=True`，容易漏。
2. 多轮 agent 要把零散的"每 turn LLM 调用"拼成完整 trajectory——谁来做这个簿记？
3. 黑盒 agent 我们控制不了它内部怎么调 SGLang——如果它绕过我们直接调，logprob 就丢了。
4. 模型输出的 `<tool_call>` 标签要解析成结构化字段——谁来做？

**决策**：起一个 FastAPI 服务（`dressage-proxy`，默认端口 8800），伪装成 **OpenAI 兼容端点**（`POST /v1/chat/completions`）。**所有 caller（whitebox loop、黑盒 agent、未来任何新组件）都必须通过它调 LLM**。

**含义**：

- proxy 是**唯一**和 SGLang 通信的组件。`dressage/proxy/sglang_client.py` 只在 proxy 内部用，其他地方不许 import。
- proxy 内部强制设 `return_logprob=True`，caller 不感知也漏不掉。
- proxy 内部按 `traj_id` 累积多 turn session（§4.9）。
- proxy 解析 `<tool_call>` 标签成结构化 `tool_calls` 字段——caller 看到的是 OpenAI 协议。
- 黑盒 agent 只要把 LLM endpoint 配成 `http://localhost:8800/v1/chat/completions`、并把 `traj_id` 带在每次请求里，就**自动被纳入训练数据采集**——它根本不知道自己被用来训练。

> **类比**：proxy 像一个反向代理（Nginx）。它的存在，让"采集"这件事和"调模型"这件事**解耦**——caller 只管发请求，proxy 偷偷把每次请求记账。

**如果不这样做会怎样**：
- 每个 caller 自己调 SGLang，**有一处忘了 `return_logprob=True`，那条 trajectory 的 logprob 就缺**，整条样本的 PPO ratio 算不出来，要么浪费要么报错。这种 bug 在测试里很难发现（采到的样本数和 token 数都对，只是 logprob 字段是空的）。
- 接入黑盒 agent 时——比如想用别人写的 React-Agent 框架——你**改不了它内部怎么调 LLM**。如果它直接调 SGLang，所有 turn 都散落在 SGLang 一侧、和你的训练系统毫无关联，整个黑盒训练根本做不成。有了 proxy 之后，你只要让它把 endpoint 配成 proxy 地址、带上 `traj_id`，就能透明地被采集——agent 框架本身一行不改。
- 多 turn 拼接（"把这次调用、上次调用、上上次调用的 token 接起来构成 trajectory"）是高度有状态的逻辑。如果分散在每个 caller 里，会被实现 N 遍，每个都微妙不同。集中到 proxy 一处，**只需实现+测试一次**。

## 决策 3：`traj_id` 是全系统主键

**问题**：黑盒 agent 在外部进程里跑，可能调几十次 LLM、调几次工具。怎么把这些零散调用绑定到"我们关心的某条 trajectory"上？

**决策**：分配一个字符串 `traj_id` 作为这条 trajectory 的全局 ID。它**同时**是：

```
sample (group_index=g, index=i)
        │
        ▼
traj_id = uuid4().hex   # 白盒: hex 32 位; 黑盒: "bbs-{uuid}"
        │
        ├──> paddock.init(traj_id, ...)          # Paddock 环境实例 ID
        │
        ├──> 每次 LLM 调用都带 traj_id            # proxy session ID
        │
        ├──> proxy.finalize_session(traj_id)              # Trajectory 的主键
        │
        ├──> proxy.read_trajectory(traj_id)       # 取数据时的 key
        │
        ├──> sample.metadata["parent_traj_id"]    # 落到训练数据的元数据
        │
        └──> paddock.terminate(traj_id)           # 释放环境
```

**代码位置**：`dressage/rollout/fully_async_rollout.py` 的 traj_id 分配逻辑。

**含义**：

- 如果你新加任何"会产生 LLM 调用的组件"，**不要发明新 session id**——挂到现有 `traj_id` 上。
- 这个 ID 在系统里一直流动，但**永不持久化**——rollout 结束后就丢掉（除了 `parent_traj_id` 元数据外）。
- 黑盒 agent 接入靠的就是它（§5.5）：把 `traj_id` 通过环境变量/HTTP header/请求体注入到 agent 进程，agent 调 LLM 时带上，proxy 一眼认出"这是 traj_X 的第 5 个 turn"。

**如果不这样做会怎样**：
- 每个组件发明自己的 session id（proxy 一个、Paddock 一个、reward 函数一个）。结算时要写 N 张"映射表"把这些 id 串起来，每加一种新组件就要扩展所有映射表。出 bug 时定位极困难——同一条 trajectory 在不同组件日志里 id 各不一样，grep 都找不全。
- 单一 traj_id 让 "**任何组件随时能问"现在我处理的这条样本到底是哪条 trajectory"**——只需要它当前手上有 traj_id。日志全用 traj_id 打头，整条 trajectory 跨组件的全部记录 `grep "r0_g0_i0_a1b2c3d4"` 就能拿到。

## 决策 4：环境/工具/黑盒 agent 全抽成 Paddock 接口

**问题**：训练时可能要用 docker 沙箱、k8s pod、远程 VM、本地 subprocess……每种执行环境的 API 都不一样。如果 rollout 代码直接对接这些 API，每加一种环境就要改一堆地方。

**决策**：定义一个三层 ABC 接口体系（`Paddock` → `BlackboxPaddock` / `WhiteboxPaddock`）管"和 agent/工具怎么交互"，再定义一个 `SandboxProvider` 协议管"沙箱放在哪、怎么起停"。两层分别可替换：`DRESSAGE_PADDOCK_MODE`（`blackbox`/`whitebox`）选 paddock，`DRESSAGE_SANDBOX_PROVIDER`（`local_bwrap`/`e2b`）选 sandbox；`DRESSAGE_PADDOCK_CLASS` 是加载自定义 paddock 类的高级覆盖。

```python
class Paddock(abc.ABC):
    async def init(self, traj_id, env_type=None, env_args=None, **kwargs)    # 起环境
    async def terminate(self, traj_id, env_args=None, **kwargs)              # 收环境

class BlackboxPaddock(Paddock):
    async def register_agent(self, state, *, instance_id, session_id, ...)   # 注入 proxy URL
    async def call_agent(self, state, *, session_id, messages, ...)          # 委托黑盒 agent
    async def execute_cmd(self, state, *, session_id, cmd, ...)              # 沙箱内执行命令
    async def write_files(self, state, *, files, dist_path="/data", ...)      # 上传文件到沙箱
    async def pause(self, traj_id=None, *, reason="weight_update", ...)     # 暂停生成
    async def resume(self, traj_id=None, *, version=None, ...)               # 恢复生成

class WhiteboxPaddock(Paddock):
    async def tool_call(self, traj_id, tool_id, tool_args)                    # 执行工具
```

**代码位置**：`dressage/paddock/interface.py`（接口）、`dressage/paddock/factory.py` 的 `create_paddock_from_env()`（选 paddock）、`dressage/sandbox/factory.py` 的 `create_sandbox_provider_from_env()`（选 sandbox provider）。

**含义**：

- 想接 docker？写 `DockerPaddock` 继承 `BlackboxPaddock` 或 `WhiteboxPaddock` 并实现异步方法（§6.4）。
- 想接黑盒 agent？在你的 `BlackboxPaddock.call_agent` 里启动它（§5.5）。
- Dressage 内部完全不关心环境细节——它只调接口方法。
- **Paddock 方法全是 `async def`**，rollout 主循环直接 `await` 调用，无需 `asyncio.to_thread` 包装。由于 asyncio 单线程模型，不需要额外的线程安全保护（但如果实现内部起了子线程/子进程，仍需自行同步）。

> **类比**：像 `unittest` 里的 setUp/tearDown——框架不管你具体怎么准备和清理资源，只要遵守接口契约就行。

**如果不这样做会怎样**：rollout 代码里硬编码 docker.from_env()/subprocess.run()/ssh 调用，每加一种执行环境就要改 rollout 主循环。想做 A/B 比较"同一个模型在 docker 沙箱 vs k8s pod 表现差异"得改两份 rollout——而且测试时根本没法用假 sandbox，每次跑测试都得真的起一个 docker daemon。抽象成接口之后，测试里注入一个 mock paddock / mock sandbox provider 就能一秒钟跑完。

## 决策 5：reward 是每个样本独立的，可注册可热插

**问题**：同一次训练里可能混合多种任务——数学题对答案、代码题跑测试、写作题用 LLM 评分。如果"reward 函数"是全局唯一的，混任务就做不到。

**决策**：

- reward 函数是普通 Python 函数 `fn(trajectory, sample) -> float`
- 用装饰器 `@register_reward("name")` 注册到全局表
- JSONL 数据里每条样本用 `"reward_fn": "name"` 指定用哪个
- 每条 trajectory 跑完**立即**调用对应的 reward 函数（不等同组其他样本）

**代码位置**：`dressage/reward/registry.py`（注册表）+ `dressage/reward/helpers.py`（内置函数）。

**含义**：

- 加新 reward 函数：在你自己的包里 `@register_reward("my_score")`，设环境变量 `DRESSAGE_REWARD_MODULES=your.pkg` 让 Dressage 启动时加载它。
- reward 可以是慢操作（调 judge LLM、跑测试套件）——**不会阻塞同组其他样本**，因为每条 trajectory 是独立 async task。
- GRPO 组归一化在所有 reward 都算完之后才发生（在 `convert_samples_to_train_data` 里）。

> **类比**：像 pytest 的 `@pytest.fixture` 或 Flask 的 `@app.route`——装饰器把函数注册到全局表，框架按名字找它。

**如果不这样做会怎样**：把 reward 写成一个全局唯一的 `reward(trajectory) -> float` 函数。混合任务时只能在函数内部 `if-else` 派发——每加一种新任务都要改这个核心函数、跑全套回归测试。注册表 + 元数据派发把"我加一种任务"和"框架核心代码"完全解耦——你的 reward 函数住在你自己的仓库，Dressage 不用知道它的存在。

---

# Part 3 — 代码组织 walkthrough

```
dressage/
├── __init__.py           # 仅 __version__
│
├── config/               # ★ 共享运行时默认值（环境变量、端口、模型默认）
│   └── config.py              # trajectory_build_defaults()、paddock_mode()、sandbox_provider()、proxy_url() 等
│
├── rollout/              # ★ rollout 主循环、agent 模式调度、数据源
│   ├── fully_async_rollout.py  # 完全异步入口：后台 worker + drain（generate_rollout_fully_async）
│   ├── partial_async_rollout.py # 部分异步入口：可提前返回部分组
│   ├── sync_rollout.py          # 同步入口：colocate 模式，跑完即返回
│   ├── data_source.py           # JSONL → list[list[Sample]]
│   ├── convert_samples.py       # ★ list[Sample] → batch dict（slime 的 convert 钩子；开头调 reward_post_process）
│   ├── multi_segment.py         # 多段 trajectory 展开为 Sample（expand_segments_to_samples）
│   ├── staleness.py             # StalenessTracker / StalenessGroupFilter 过期过滤
│   ├── log_rollout.py           # rollout 日志辅助（log_rollout_data）
│   ├── generate/                # ★ agent 执行层
│   │   ├── whitebox_agent.py    # WhiteboxAgent / PaddockWhiteboxAgent 类 + make_generate()
│   │   ├── blackbox_dispatch.py # 黑盒 agent 委托入口（generate）
│   │   └── runtime.py           # get_paddock_from_env / get_proxy_client
│   ├── prewarm/                 # 沙箱预热（PrewarmScheduler / claim_prewarm / store）
│   └── artifacts/               # trajectory / sample 产物落盘（samples.py / writer.py）
│
├── proxy/                # ★ 单一 LLM 入口、session 跟踪
│   ├── server.py              # FastAPI 服务（create_app / parse_args / main）
│   ├── session_manager.py     # 多 turn session + record_step（StepRecord）
│   ├── trajectory_store.py    # 内存 buffer（read_trajectory / pop_trajectory）
│   ├── generation_controller.py  # pause/resume/shutdown + partial rollout
│   ├── sglang_client.py       # 调 SGLang Router（SGLangRouterClient，仅 proxy 内部用）
│   ├── proxy_client.py        # 调 proxy 的客户端（rollout 用）
│   ├── tool_call_parser.py    # ProxyToolCallParser 类
│   ├── reasoning_parser.py    # ProxyReasoningParser 类
│   ├── tool_call_ids.py       # tool_call id 规范化
│   ├── last_step/             # loss_mask 构造
│   │   └── prompt_assistant_mask.py  # PromptAssistantMaskBuilder
│   └── tito/                  # TITO 分词器（concat 模式）
│
├── paddock/              # ★ agent 交互抽象层（不管沙箱放置）
│   ├── interface.py           # Paddock / BlackboxPaddock / WhiteboxPaddock ABC
│   ├── factory.py             # create_paddock_from_env()（按 DRESSAGE_PADDOCK_MODE 选 blackbox/whitebox）
│   ├── lifecycle.py           # terminate_paddock_best_effort / schedule_terminate_paddock 等
│   ├── blackbox/              # BlackboxAgentPaddock（paddock.py）+ client / execute_hooks / failures / common
│   └── whitebox/              # WhiteboxToolPaddock（paddock.py）+ tools
│
├── sandbox/              # ★ 沙箱放置抽象层（新增，与 paddock 分离）
│   ├── provider.py            # SandboxProvider 协议（create/terminate/run_command/... 全 async）
│   ├── factory.py             # create_sandbox_provider_from_env()（按 DRESSAGE_SANDBOX_PROVIDER 选）
│   ├── types.py               # SandboxSpec / SandboxLease / SandboxEndpoint / CommandResult
│   ├── local/bwrap/           # LocalBwrapSandboxProvider（Ray 管理的 bwrap 槽位池）
│   ├── remote/e2b/            # E2BSandboxProvider
│   ├── remote/harness/        # HarnessSandboxProvider
│   └── scripts/               # dressage-local-bwrap-{start,status,stop} CLI 入口
│
├── reward/               # ★ 奖励注册表 + 内置函数
│   ├── registry.py            # @register_reward、get_reward_fn、call_reward_fn、load_reward_modules
│   ├── helpers.py             # 内置 reward（exact_match / contains_label / constant / metadata_score / accio_claw / omni_grader / default）
│   └── custom_rm.py          # slime --custom-rm-path 入口（custom_rm）
│
└── training/             # ★ slime 训练侧的插件
    ├── train_async_with_rollout_pause.py  # 异步训练 + pause/resume（可选入口）
    ├── reward_post_process.py             # GRPO 组归一化（reward_post_process）
    └── log_helpers.py                     # 训练日志辅助
```

> **注意：paddock 和 sandbox 是两层**。`paddock/` 只负责"和 agent/工具怎么交互"（register/call/tool_call）；`sandbox/` 只负责"沙箱放在哪、怎么起停"（本地 bwrap 池 / E2B / Harness router）。`BlackboxAgentPaddock` 内部持有一个 `SandboxProvider`，两者分别由 `DRESSAGE_PADDOCK_MODE` 和 `DRESSAGE_SANDBOX_PROVIDER` 选择。详见 §4C。

**子包的"边界"**：

| 子包 | 输入 | 输出 | 不能依赖谁 |
|---|---|---|---|
| `config/` | 环境变量 | 默认值 | 不依赖任何子包 |
| `sandbox/` | SandboxSpec | SandboxLease、命令/文件结果 | 不依赖 proxy/rollout；不含 agent 概念 |
| `paddock/` | Paddock 接口方法的入参 | 工具响应、agent 响应 | 依赖 sandbox；不依赖 proxy/rollout |
| `proxy/` | HTTP 请求 | HTTP 响应 + 内存里的 Trajectory | 不依赖 rollout（rollout 调它） |
| `reward/` | Sample | float | 纯函数，不依赖 paddock/proxy 运行时 |
| `rollout/` | slime 传入的 args + data_buffer | list[list[Sample]] | 依赖所有其他子包 |
| `training/` | list[Sample] | dict（slime 训练 batch） | 不依赖 paddock/proxy/rollout 运行时 |

`rollout/` 是"司机"，把其他子包串起来。其他子包之间尽量不互相调用。

---

# Part 4 — 端到端代码细读：从一行 JSONL 走到一次梯度更新

**这是文档的主体**。我们跟踪一条样本从 JSONL 进入、到产生梯度，每一步都点到具体代码。

**怎么读这一部分**：每一节都遵循"先一段口语化的导语 → 真实代码片段（带行号引用）→ 逐行/逐块解释 → 关键点/坑总结"的结构。如果你只想拿一个大致印象，**只读每节的导语**也能拼出 80% 的图景；想真的读懂源码，就对着代码本人（用 IDE 跳到 `file:line` 处）和文档对照读。

**Part 4 全程跟踪的样本**：假设你的 JSONL 数据集 `data.jsonl` 里有这一行：

```json
{
  "prompt": "求 1+1 等于几",
  "label": "2",
  "agent_mode": "whitebox",
  "reward_fn": "exact_match"
}
```

**这一节会带这条样本经过的阶段**：

```
§4.1  启动 (proxy + slime 两个进程都起来)
  │
  ▼
§4.2  数据加载 (JSONL 一行 → Sample × 4 副本)
  │
  ▼
§4.3 - 4.4  rollout 入口 + 并发调度
  │
  ▼
§4.5  单条 trajectory 主控
  │
  ▼
§4.6  whitebox agent 循环                  (loop body ↓)
  │                                          │
  ├─ §4.7  POST /v1/chat/completions    ────┤
  │   ▼                                      │
  ├─ §4.8  SGLang /generate              ───┤
  │   ▼                                      │
  ├─ §4.9  SessionManager.record_step    ───┤
  │   ▼                                      │
  └─ §4.10 工具执行 + messages 追加      ───┘
                  │ (loop until no tool_call or max_turns)
                  ▼
§4.11 - 4.12  Finalize → Trajectory → read 取回
  │
  ▼
§4.13  算 reward
  │
  ▼
§4.14 - 4.15  Trajectory → Sample(s) → 释放环境
  │
  ▼
§4.16  重新分组 → 返回给 slime
  │
  ▼
§4.17  训练侧：GRPO 归一化 + 拼 batch
  │
  ▼
§4.18  slime 训练循环（黑盒：算 logprob_new、PPO loss、反传）
```

这张图你也可以当作"Part 4 自己的目录"——读到具体小节时回头看一下，知道自己处在流水线的哪一段。

## 4.1 启动：proxy 先起，slime 再起

`examples/scripts/run_example_*.sh` 做两件事：

```bash
# Step 1: 先起 proxy
dressage-proxy \
  --sglang-router-url http://localhost:8000 \
  --tokenizer-path "$HF_CHECKPOINT" \
  --port 8800 &

# 等 /health 通了
for i in $(seq 1 60); do
  curl -sf "${DRESSAGE_PROXY_URL}/health" && break
  sleep 1
done

# Step 2: ray job submit 启动 slime 训练（train_async.py），告诉它用 Dressage 的插件
ray job submit ... -- python3 train_async.py \
  --rollout-function-path dressage.rollout.fully_async_rollout.generate_rollout_fully_async \
  --custom-generate-function-path dressage.rollout.generate.blackbox_dispatch.generate \
  --custom-rm-path        dressage.reward.custom_rm.custom_rm \
  --data-source-path      dressage.rollout.data_source.DressageDataSource \
  --custom-reward-post-process-path dressage.training.reward_post_process.reward_post_process \
  --custom-convert-samples-to-train-data-path \
                          dressage.rollout.convert_samples.convert_samples_to_train_data \
  --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data \
  --prompt-data $PROMPT_DATA \
  --n-samples-per-prompt 4 \
  --rollout-batch-size 8 \
  --advantage-estimator grpo \
  ...
```

> 白盒示例（如 `run_alfworld_whitebox_agent_*.sh`）把 `--custom-generate-function-path` 换成自己的 `make_generate(MyAgent)` 生成的 `generate`；黑盒示例统一走 `blackbox_dispatch.generate`。`--rollout-function-path` 在 sync/partial 模式下分别换成 `generate_rollout_sync` / `generate_rollout_partial_async`。

**proxy 启动后的状态**（`dressage/proxy/server.py` 的 `create_app()` 函数，L537）：

```python
app = create_app(
    sglang_router_url=...,
    tokenizer_path=...,
    trajectory_store=TrajectoryStore(...),
    session_manager=SessionManager(...),
    trajectory_build_mode="concat",   # 或 "last_step"
    trajectory_build_model="qwen3_5",
    ...
)
```

`create_app` 内部创建多个关键对象：`SessionManager`、`TrajectoryStore`、`SGLangRouterClient`、`GenerationController`、`PromptAssistantMaskBuilder`、`ProxyToolCallParser`、`ProxyReasoningParser`。这些是后面所有请求的依赖。

**slime 启动后**：内部用 `importlib` 加载 Dressage 的 4 个插件符号，并在自己的训练循环里调用它们。第一次调 `generate_rollout` 时触发 Dressage 端的一次性初始化（§4.3）。

## 4.2 数据加载：DressageDataSource

`dressage/rollout/data_source.py` 的 `DressageDataSource` 继承 slime 的 `RolloutDataSourceWithBuffer`，主要重写 `__init__` 和 `get_samples`。

**`__init__` 干的事**（`data_source.py:103-185`）：

```python
def __init__(self, args):
    self.args = args
    self.buffer = []                                  # 重试样本的临时存放
    prompt_data = args.prompt_data
    self._use_text_first = args.multimodal_keys is None

    if self._use_text_first and prompt_data:
        self._samples = self._load_text_first(...)   # 走这条
```

`_load_text_first`（`data_source.py:186-282`）读 JSONL，每行变成一个 `Sample` 对象：

```python
def _load_text_first(self, path, prompt_key, label_key, metadata_key):
    samples = []
    for data in _read_jsonl(path):
        prompt = data.get(prompt_key, "")            # "求 1+1 等于几"
        label = data.get(label_key)                  # "2"

        meta = data.get(metadata_key) or {}
        for key in ("agent_mode", "env_type", "env_args",
                    "tool_set", "agent_id", "reward_fn"):
            if key in data and key not in meta:
                meta[key] = data[key]                # 元数据透传到 metadata
        # 其他不认识的字段也透传进 metadata

        samples.append(Sample(
            prompt=prompt,
            label=str(label) if label else None,
            metadata=meta,                           # 关键：所有调度信息都在这里
        ))
    return samples
```

**关键点**：JSONL 里的 `agent_mode / reward_fn / tool_set` 等字段，被 `_load_text_first` 透传到 `Sample.metadata` 里。后面 rollout 函数会读这些字段做"per-sample dispatch"——这就是"一个 batch 里混多种 agent 模式 / 多种 reward"能成立的基础。

**`get_samples` 干的事**（`data_source.py:283-355`）：

```python
def get_samples(self, num_samples):
    # 先从 buffer 拿（重试样本）
    buffer_samples = self._get_samples_from_buffer(num_samples)
    num_samples -= len(buffer_samples)

    n_per = self.args.n_samples_per_prompt           # =4
    groups = []

    for _ in range(num_samples):
        base_sample = self._samples[self.sample_offset]
        self.sample_offset += 1

        # 同一 prompt 复制 N 次，组成一个 GRPO 组
        group = []
        for _ in range(n_per):
            s = copy.deepcopy(base_sample)
            s.group_index = self.sample_group_index  # 组号
            s.index = self.sample_index              # 全局唯一索引
            self.sample_index += 1
            group.append(s)
        self.sample_group_index += 1
        groups.append(group)

    return buffer_samples + groups
```

**关键点**：返回的是 `list[list[Sample]]`——外层是"组"，内层是"组里的 N 个副本"。这就是 GRPO 训练的核心数据结构。

对我们这条 JSONL 行：

```
原始 1 条 Sample
        ↓ get_samples(rollout_batch_size=8) + n_samples_per_prompt=4
组 0: [Sample(group_index=0, index=0, prompt="求 1+1...", reward_fn="exact_match"),
       Sample(group_index=0, index=1, prompt="求 1+1...", reward_fn="exact_match"),
       Sample(group_index=0, index=2, prompt="求 1+1...", reward_fn="exact_match"),
       Sample(group_index=0, index=3, prompt="求 1+1...", reward_fn="exact_match")]
组 1: [...]  组 2: [...]  ... 组 7: [...]
```

## 4.3 rollout 入口：generate_rollout_fully_async

slime 每轮训练调 `--rollout-function-path` 指向的函数。完全异步模式是 `generate_rollout_fully_async`（[`fully_async_rollout.py:533`](../dressage/rollout/fully_async_rollout.py#L533)）：

```python
def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("Dressage fully async rollout does not support evaluation mode")
    # 后台常驻 worker 持续从 data_buffer 拉 prompt 组并生成；
    # 本次调用只 drain 已完成、且够一个 batch 的组。
    data, staleness_metrics = run(generate_rollout_async(args, rollout_id, data_buffer))
    metrics = compute_multi_segment_metrics([s for group in data for s in group])
    metrics.update(staleness_metrics)
    return RolloutFnTrainOutput(samples=data, metrics=metrics)
```

**关键转变**：Dressage **不再自己管 trajectory 级并发**。它复用 slime 的 `slime.rollout.sglang_rollout.generate_and_rm_group` 来跑一个 prompt 组（组内每个 sample 一次生成 + reward）。Dressage 只做三件 slime 没有的事：

1. **后台常驻 worker**（`AsyncRolloutWorker`，[`fully_async_rollout.py:268`](../dressage/rollout/fully_async_rollout.py#L268)）：持续拉组、`asyncio.create_task(generate_and_rm_group(...))`，完成的塞进 `output_queue`。
2. **组级重试**：失败组按 `DRESSAGE_ROLLOUT_MAX_RETRIES`（默认 2）退回 `data_buffer` 重跑；耗尽后 `_mark_no_grad_failed`（[`:228`](../dressage/rollout/fully_async_rollout.py#L228)）打成零梯度占位样本。
3. **staleness 过滤**：用旧权重生成的过期组被 `StalenessGroupFilter` 丢弃（§4D.4）。

同步模式 `generate_rollout_sync`（`sync_rollout.py`）和部分异步 `generate_rollout_partial_async`（`partial_async_rollout.py`）是同一套骨架的不同 drain 策略。

## 4.4 后台 worker 并发生成

`AsyncRolloutWorker.continuous_worker_loop`（[`fully_async_rollout.py:302`](../dressage/rollout/fully_async_rollout.py#L302)）的核心：

```python
# 同时在跑的 prompt 组数上限
self.max_active_groups = int(
    os.environ.get("DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS",
                   str(getattr(args, "rollout_batch_size", 1)))
)
...
while len(active) < self.max_active_groups and ...:
    group_id, group = self._scheduler.pop_next_group(self.data_buffer)
    task = asyncio.create_task(self._run_group(group, sampling_params))
    active[task] = (group_id, group)
```

**关键点**：并发单位是 **prompt 组**，上限由 `DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS`（默认 = `rollout_batch_size`）控制，不是旧版的 `DRESSAGE_MAX_CONCURRENT`。`_run_group` 直接 `await generate_and_rm_group(...)`，把 SGLang 的调度交给 slime。开启 sandbox 预热（§4D，harness 默认开）时，worker 还会提前给未来的组 `paddock.init` 沙箱。

## 4.5 单条 trajectory 的生成钩子

slime 的 `generate_and_rm_group` 对组里每个 sample 调 `--custom-generate-function-path` 指向的 `generate`。黑盒样本走 `blackbox_dispatch.generate`（[`blackbox_dispatch.py:78`](../dressage/rollout/generate/blackbox_dispatch.py#L78)），白盒走 `make_generate(MyAgent)` 生成的 `generate`（§4.6）。两者的签名都是 slime 约定的：

```python
async def generate(args, sample, sampling_params, evaluation=False) -> Any:
    ...
```

黑盒 `generate` 的骨架（详见 §4B 逐步拆解）：

```python
session_id  = ensure_blackbox_session_id(sample)   # "bbs-..."，即 traj_id
instance_id = _instance_id(sample)                 # GRPO 组主键
try:
    paddock = get_paddock_from_env(allow_whitebox_mode=False)
    state   = await paddock.init(session_id, env_type, env_args)   # 申请沙箱
    await paddock.register_agent(state, ...)                       # 注入 proxy URL
    await execute_blackbox_cmds_for_stage(..., stage="before_agent")
    call_payload = await paddock.call_agent(state, session_id=session_id, messages=...)
    await execute_blackbox_cmds_for_stage(..., stage="after_agent")
    await proxy_client.finalize_session(session_id, instance_id=instance_id, ...)
    payload = await proxy_client.read_trajectory(trajectory_id=session_id,
                                                 instance_id=instance_id, drain=True)
    return multi_segment.expand_segments_to_samples(sample, payload["data"], ...)
except Exception:
    ...  # 落盘错误、标 ABORTED、返回 sample
finally:
    schedule_terminate_paddock(paddock, session_id=session_id, env_args=env_args)  # best-effort 释放
```

注意：reward **不在这里算**。slime 在生成后通过 `--custom-rm-path`（`dressage.reward.custom_rm.custom_rm`）调 reward 函数，再由 `--custom-convert-samples-to-train-data-path` 做 GRPO 归一化（§4.13、§4.17）。

我们的样本是 whitebox 模式，下面跟到 `WhiteboxAgent.rollout()`。

## 4.6 白盒 agent 循环：`WhiteboxAgent.rollout()`

`dressage/rollout/generate/whitebox_agent.py`：

用户继承 `WhiteboxAgent`（[`whitebox_agent.py:97`](../dressage/rollout/generate/whitebox_agent.py#L97)），实现 `rollout()` 方法。框架负责 session 管理、proxy drain、segment 转换。用户只写对话逻辑：

```python
class MyAgent(WhiteboxAgent):
    name = "my_agent"

    async def rollout(self, sample, sampling_params):
    # 把 prompt 包成 messages
    messages = [{"role": "user", "content": sample.prompt}]
    # → [{"role": "user", "content": "求 1+1 等于几"}]

    # 如果配置了工具，加 system 提示（示例省略）
    # （实际使用时可根据模型需要在 messages 前插入 system 消息）
        messages.insert(0, {"role": "system",
            "content": "You have access to tools. Use <tool_call>...</tool_call>"})

    for turn in range(self.args.dressage_max_turns):
        # ★ 关键：所有 LLM 调用都走 proxy（通过 self.chat）
        body = {
            "messages": messages,
            **sampling_params,            # max_tokens / temperature / top_p
        }
        result = await self.chat(body)   # → proxy.chat_completions

        msg = result["choices"][0]["message"]
        assistant_msg = {"role": "assistant", "content": msg.get("content", "")}
        if msg.get("tool_calls"):
            assistant_msg["tool_calls"] = msg["tool_calls"]
        messages.append(assistant_msg)

        # 输出被截断 → 不要执行残废的 tool_call
        if result["choices"][0]["finish_reason"] == "length":
            return assistant_msg.get("content", "")  # 提前返回

        # 没工具调用 → 模型说完了，结束
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return assistant_msg.get("content", "")

        # 有工具调用 → 通过 Paddock 执行
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_args = json.loads(tc["function"]["arguments"])

            # ★ paddock.tool_call 现在是 async def，直接 await
            tool_response = await self.paddock.tool_call(
                self.session_id, tool_name, tool_args)

            messages.append({
                "role": "tool",
                "content": str(tool_response),
                "tool_call_id": tc["id"],
            })
    return messages[-1].get("content", "")  # 超出 max_turns
```

**几个细节**：

注册方式：在训练脚本里设
```bash
--custom-generate-function-path dressage.rollout.generate.my_agent.generate
```
其中 `generate = make_generate(MyAgent)`（[`whitebox_agent.py:322`](../dressage/rollout/generate/whitebox_agent.py#L322)）。

- `messages` 是这个循环的"状态"——每 turn 都在它后面 append（assistant turn / tool turn）。
- `self.chat(body)` 内部调 `proxy.chat_completions(body, session_id=..., instance_id=..., turn_id=...)`，proxy 按 `session_id` 记账。
- `paddock.tool_call` 是 `async def`（`WhiteboxPaddock` 接口），直接 `await`，不再需要 `asyncio.to_thread`。
- `finish_reason == "length"` 时**直接退**——SGLang 截断输出可能包含半截 `<tool_call>` 文本，硬执行会乱码。
- 需要沙箱生命周期的子类继承 `PaddockWhiteboxAgent`（[`whitebox_agent.py:195`](../dressage/rollout/generate/whitebox_agent.py#L195)），它自动在 `setup()` / `teardown()` 里调 `paddock.init()` / `paddock.terminate()`。

## 4.7 进入 proxy：`POST /v1/chat/completions`

> **这一节做的事**：从"调一个 Python 函数"切换到"发一个 HTTP 请求到另一个进程"。理解这一节，你就理解了进程 A（rollout）和进程 B（proxy）之间的边界。
>
> **请求路径**：caller 在 rollout 进程里调 `ProxyClient.chat_completions(...)` → httpx 发 HTTP POST → proxy 进程的 FastAPI handler 接收 → 用 tokenizer 把 messages 编码 → 调 SGLang → 解析输出 → 在 SessionManager 记账 → 返回 OpenAI 兼容的 JSON。

`WhiteboxAgent.chat()` 每次 `proxy.chat_completions(...)` 调用，背后是 `ProxyClient` 发 HTTP 请求（[`proxy_client.py:10-44`](../dressage/proxy/proxy_client.py#L10-L44)）：

```python
async def chat_completions(
    self, body, *, session_id, instance_id=None, turn_id=None
):
    headers = {"X-Session-Id": session_id}
    if instance_id is not None:
        headers["X-Instance-Id"] = instance_id
    if turn_id is not None:
        headers["X-Turn-Id"] = turn_id
    resp = await self._client.post(
        f"{self._proxy_url}/v1/chat/completions", json=body, headers=headers
    )
    resp.raise_for_status()
    return resp.json()
```

新签名：`body` 是完整的 OpenAI 请求体（messages + max_tokens + temperature 等），`session_id` / `instance_id` / `turn_id` 通过 HTTP header 传递。proxy 通过 header 提取 session_id 来记账。

proxy 那一头（[`server.py:1424`](../dressage/proxy/server.py#L1424)）：

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    session_mgr = _state["session_manager"]
    body = await request.json()
    messages = body.get("messages", [])

    # 从 header / body 提取 session_id（见 _runtime_ids_from_request, L392）
    session_id = _runtime_ids_from_request(request, body)["session_id"]

    # ① 用 chat template 把 messages 变成 token id 序列
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_token_count = len(input_ids)   # ★ 后面 SessionManager 需要这个

    # ② 把采样参数转成 SGLang 格式
    sampling_params = {
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
    }

    # ③ 调 SGLang
    sglang_resp = await sglang.generate(
        input_ids=input_ids,
        sampling_params=sampling_params,
        return_logprob=True,   # ★ 强制 True！
    )

    # ④ 把 token 解码成文本
    output_text = tokenizer.decode(sglang_resp.output_ids, skip_special_tokens=True)

    # ⑤ 解析工具调用（使用 ProxyToolCallParser, L174）
    cleaned_text, tool_calls = proxy_tool_call_parser.parse(output_text, messages)

    # ⑥ 在 SessionManager 里记账（record_step, L408）
    if session_id:
        session_mgr.record_step(
            session_id=session_id,
            request_messages=messages,
            prompt_token_ids=input_ids,
            response_token_ids=sglang_resp.output_ids,
            response_logprobs=sglang_resp.output_token_logprobs,
            # ... 其他字段
        )

    # ⑦ 拼 OpenAI 格式响应
    response_msg = ResponseMessage(
        role="assistant",
        content=(cleaned_text or None) if tool_calls else output_text,
        tool_calls=tool_calls if tool_calls else None,
    )
    finish_reason = "tool_calls" if tool_calls else "stop"
    if len(sglang_resp.output_ids) >= request.max_tokens:
        finish_reason = "length"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        choices=[ChatChoice(message=response_msg, finish_reason=finish_reason)],
    )
```

**几个关键细节**：

- `apply_chat_template(messages, add_generation_prompt=True)` 是 tokenizer 自带功能，把消息列表按模型规定的 chat 格式拼成 token 序列。`add_generation_prompt=True` 在末尾加上"现在轮到 assistant 说话"的占位符。
- `input_token_count` 是**权威 token 数**——SessionManager 后续构造 segment 时不会再 tokenize，而是用这个数字反推 turn 边界（见 §5.2）。
- `ProxyToolCallParser`（[`tool_call_parser.py:174`](../dressage/proxy/tool_call_parser.py#L174)）解析模型输出中的工具调用块，支持多种模型的 tool_call 格式。

**这段代码每一步在干什么（人话版）**：

| 编号 | 步骤 | 一句话 |
|---|---|---|
| ① | apply_chat_template | 把 `[{"role":"user","content":"..."},...]` 变成一长串 token ID |
| ② | 转 sampling params | OpenAI 字段名 → SGLang 字段名（`max_tokens` → `max_new_tokens`） |
| ③ | sglang.generate | 真正发起一次推理（**这是整个调用里最慢的一步**，可能几百毫秒） |
| ④ | decode | 把 SGLang 返回的 token ID 数组变回字符串 |
| ⑤ | ProxyToolCallParser | 解析模型输出中的工具调用块（支持 local/sglang_api/hybrid 后端） |
| ⑥ | record_step | 在 SessionManager 里记账（按 session_id append 一条 `StepRecord`） |
| ⑦ | 拼响应 | 按 OpenAI 协议返回（caller 看到的就是 OpenAI 标准格式） |

注意 ⑥ 之所以**放在 ⑤ 之后**：proxy 必须在记账时同时存"原始输出"和"解析后的 tool_calls"。如果先记账再解析，万一解析过程出 bug，记账数据可能不一致。

**caller 视角看不到的事**：每次 caller 调一次 `chat_completions`，proxy 这边偷偷给 SessionManager 加了一条记录。**proxy 是有状态的**——你可以连续调 10 次 `chat_completions`（同一个 session_id），它累积 10 条 step；调 `finalize_session` 才把它们拼成 Trajectory。

## 4.8 SGLang 调用：`SGLangRouterClient.generate`

`dressage/proxy/sglang_client.py` 的 `SGLangRouterClient.generate`（[`sglang_client.py:176`](../dressage/proxy/sglang_client.py#L176)）：

```python
async def generate(self, input_ids, sampling_params, return_logprob=True):
    client = await self._get_client()

    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": return_logprob,
    }
    if return_logprob:
        payload["logprob_start_len"] = 0      # 从第 0 个 token 开始捞 logprob
        payload["top_logprobs_num"] = 0        # 只要被采样 token 的 logprob，不要 top-k

    for attempt in range(max_retries):
        try:
            resp = await client.post("/generate", json=payload)
            data = resp.json()

            output_ids = data.get("output_ids", data.get("token_ids", []))
            # 处理 SGLang 返回可能是 [[...]] 嵌套的情况
            if isinstance(output_ids, list) and output_ids and isinstance(output_ids[0], list):
                output_ids = output_ids[0]

            logprobs = None
            if return_logprob:
                meta = data.get("meta_info", {})
                logprobs = meta.get("output_token_logprobs")

            return SGLangResponse(
                output_ids=output_ids,
                output_token_logprobs=logprobs,
                meta_info=data.get("meta_info", {}),
            )
        except httpx.ConnectError:
            # exponential backoff
            await asyncio.sleep(2 ** attempt)
```

**关键点**：`return_logprob: True` 让 SGLang 返回每个生成 token 的 logprob。这是后面 PPO 算 ratio 的输入（§0.10）。

## 4.9 SessionManager 记账：`record_step`

`dressage/proxy/session_manager.py` 的 `record_step`（[`session_manager.py:408`](../dressage/proxy/session_manager.py#L408)，实际参数比下面多，只保留核心）：

```python
def record_step(self, *, session_id, turn_id, request_messages,
                prompt_token_ids, response_token_ids, response_logprobs,
                all_token_ids, all_logprobs, messages, raw_response_text,
                finish_reason="stop", **more):   # 实际还有 concat_*/versions/tools 等大量字段
    with self._lock:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.steps.append(StepRecord(
            turn_id=turn_id,
            request_messages=request_messages,
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            response_logprobs=response_logprobs,
            all_token_ids=all_token_ids,
            all_logprobs=all_logprobs,
            messages_snapshot=messages,
            raw_response_text=raw_response_text,
            finish_reason=finish_reason,
            ...
        ))
        session.last_active = time.time()
```

核心：**按 session_id 找到 `Session`，往它的 `steps` 列表 append 一条 `StepRecord`**。`_sessions: dict[str, Session]`，每个 `Session` 持有 `.steps: list[StepRecord]`（`TurnRecord` 是 `StepRecord` 的别名）。

每条 `StepRecord` 关键字段（[`session_manager.py:93`](../dressage/proxy/session_manager.py#L93)，实际字段远不止这些）：
- `request_messages` / `messages_snapshot`：调用时的完整对话历史
- `response_token_ids`：这一 step 模型生成的 token id 列表
- `response_logprobs`：对应的 logprob 列表
- `prompt_token_ids` / `all_token_ids`：这一 step 的输入 token 序列 / 输入+输出全序列（用 chat template 算的权威值）
- `concat_*` 字段：concat 轨迹构建模式（§4D.5）用的增量 token / mask / logprob

## 4.10 回到 whitebox loop，工具执行

假设模型输出了 `<tool_call>{"name": "bash", "arguments": {"cmd": "echo 2"}}</tool_call>`。proxy 解析后返回，whitebox loop 收到：

```python
msg.tool_calls = [{"id": "call_a1b2c3d4",
                   "function": {"name": "bash", "arguments": '{"cmd": "echo 2"}'}}]
```

进入工具执行分支（`generate/whitebox_agent.py`）：

```python
for tc in tool_calls:
    tool_name = "bash"
    tool_args = {"cmd": "echo 2"}

    # 把模型可见的 tool 名映射到 Paddock 的 tool_id（可能相同）
    paddock_tool_id, paddock_args = tool_mapper.resolve(tool_name, tool_args)

    # ★ 调 Paddock 执行（async def，直接 await）
    tool_response = await paddock.tool_call(traj_id, paddock_tool_id, paddock_args)
        paddock.tool_call, traj_id, paddock_tool_id, paddock_args)
    # tool_response = "2"

    # 把工具结果作为 tool turn 加到对话
    messages.append({
        "role": "tool",
        "content": "2",
        "tool_call_id": "call_a1b2c3d4",
    })
```

接下来 for 循环继续，下一个 turn 模型看到这个 tool 结果，可能输出"答案是 2"，没有 tool_call，循环退出。

到这里，proxy 的 `SessionManager._sessions["r0_g0_i0_a1b2c3d4"]` 里累积了 2 条 `StepRecord`：一条是包含 tool_call 的输出，一条是"答案是 2"的最终输出。

## 4.11 Finalize：把 turns 拼成 Trajectory

> **这一节做的事**：whitebox loop 跑完后，proxy 的 SessionManager 里囤了 N 条零散的 `TurnRecord`（一个 turn 一条）。但训练侧要的是**一段连续的 token 序列 + 对应的 logprob 数组 + 一个 loss_mask**。`finalize_session` 干的就是这件"拼接"工作。它是整个 Dressage 里最复杂的一段代码（涉及 prompt/response 边界、turn 之间的 tool/user 插曲、loss_mask 角色划分）。
>
> **拼接成果的形状**：
>
> ```
> all_tokens   = [<prompt 区>, <assistant turn1>, <tool 插曲>, <assistant turn2>, ...]
>                  ↑ 长度 = response_start                                       ↑ 长度 = response_length
>
> all_logprobs = [<对应于 all_tokens 中 response 区每个 token 的 logprob>]
>
> loss_mask    = [0, 0, ..., 0,    1, 1, 1, ...,    0, 0, ...,   1, 1, 1, ...]
>                 ↑ tool 插曲部分     ↑ assistant     ↑ tool       ↑ assistant
>                                       turn1           插曲          turn2
> ```

黑盒/白盒 dispatch 跑完后：

```python
await proxy_client.finalize_session(session_id, instance_id=instance_id, label=...)
payload = await proxy_client.read_trajectory(trajectory_id=session_id,
                                             instance_id=instance_id, drain=True)
segments = payload["data"]   # list[segment dict]
```

`finalize` 在 proxy 端走到 `/session/finalize`（[`server.py:1956`](../dressage/proxy/server.py#L1956)）：

```python
@app.post("/session/finalize")
async def finalize_session(request: Request):
    body = await request.json()
    session_id = body["session_id"]
    session = session_manager.finalize_session(session_id)   # pop 出整个 Session
    if session is None or not session.steps:
        raise HTTPException(...)                              # 404 / "Session has no turns"
    # 把 session 的 steps 切成 1..N 段（rewrite/tools 变化处断段），逐段写入 store
    segments = _split_session_into_segments(session)
    build = (_build_concat_segment_record if trajectory_build_mode == "concat"
             else _build_segment_record)
    for i, segment in enumerate(segments):
        trajectory_store.write_dict(build(session=session, segment=segment,
                                          segment_index=i, ...))
    return {"success": True, "num_segments": len(segments), ...}
```

**关键点**：
- `SessionManager.finalize_session`（[`session_manager.py:568`](../dressage/proxy/session_manager.py#L568)）本身只做一件事——把 `Session` 从 `_sessions` 里 pop 出来。**真正把零散 step 拼成训练段的逻辑在 server.py 里**（`_split_session_into_segments` + `_build_segment_record`/`_build_concat_segment_record`）。
- 一次 finalize 可能产出 **多个 segment**（黑盒 agent 做 history rewrite/compaction 时会断段），每段是一个独立的训练单元，写进 `TrajectoryStore`。append-only 的普通对话通常只有 1 段。
- **loss_mask 不再靠手动追踪 turn 边界**，而是由 `PromptAssistantMaskBuilder.build_segment_alignment`（[`prompt_assistant_mask.py:245`](../dressage/proxy/last_step/prompt_assistant_mask.py#L245)）用"mask-only chat template"重放一遍对话，让 tokenizer 直接标出哪些 token 属于 assistant（详见 §5.1）。每段的产出形状是 `tokens` / `full_loss_mask` / `full_logprobs`（长度一致）。

## 4.12 读 trajectory：`read_trajectory`

dispatch 继续：

```python
payload = await proxy_client.read_trajectory(
    trajectory_id=session_id, instance_id=instance_id, drain=True)
segments = payload["data"]   # list[segment dict]
```

proxy 端 handler 是 `trajectory_read`（[`server.py:2089`](../dressage/proxy/server.py#L2089)）：

```python
@app.post("/trajectory/read")
async def trajectory_read(request: Request):
    body = await request.json()
    trajectory_id = _trajectory_id_from_body(body)   # trajectory_id 或 session_id
    instance_id = body.get("instance_id")
    drain = bool(body.get("drain", False))
    data = (trajectory_store.pop_trajectory(trajectory_id, instance_id=instance_id)
            if drain
            else trajectory_store.read_trajectory(trajectory_id, instance_id=instance_id))
    return {"success": bool(data), "mode": "trajectory", "data": data, "drained": drain}
```

`TrajectoryStore.pop_trajectory`（[`trajectory_store.py:307`](../dressage/proxy/trajectory_store.py#L307)）返回该 trajectory 的所有 segment dict（按 `segment_index` 排序），并从 `_by_trajectory` / `_by_instance` 两个索引里删除：

```python
def pop_trajectory(self, trajectory_id, instance_id=None) -> list[dict]:
    with self._lock:
        ...  # 匹配 trajectory_id（+ 可选 instance_id），移除并返回 [seg.to_dict() ...]
```

**关键点**：`drain=True` 时是 pop 不是 read——读完即销毁，避免长跑的异步 rollout 把已完成 segment 一直堆在内存里。

## 4.13 算 reward

reward **不在 dispatch 里算**。slime 在生成完成后通过 `--custom-rm-path`（`dressage.reward.custom_rm.custom_rm`）回调 reward：`custom_rm` 读 `sample.metadata["reward_fn"]`（默认 `"default"`），从注册表取函数并执行（[`custom_rm.py`](../dressage/reward/custom_rm.py)、[`registry.py`](../dressage/reward/registry.py)）：

```python
# custom_rm 内部（简化）
name = sample.metadata.get("reward_fn") or "default"
reward = await call_reward_fn(name, sample, args=args)   # get_reward_fn(name)(sample, args=args)
```

`exact_match` 在 [`dressage/reward/helpers.py:26`](../dressage/reward/helpers.py#L26)：

```python
@register_reward("exact_match")
def exact_match(sample: Any, *, args: Any | None = None, **_: Any) -> float:
    """Return 1.0 when the response exactly matches the label."""
    del args
    label = _label(sample)
    if not label:
        return 0.0
    return 1.0 if _response(sample).strip() == label else 0.0
```

`_label(sample)` 先取 `sample.label`，再回退到 `sample.metadata["label"]`（strip 后返回）；`_response(sample)` 返回 `sample.response`。我们的例子里 response 是 "答案是 2"，label 是 "2"，`"答案是 2".strip() != "2"` 所以 reward = 0.0。如果 response 恰好是 "2"，则 reward = 1.0。这只是示例，实际 reward 函数会更鲁棒。

## 4.14 Segment → Sample(s)

黑盒/白盒 dispatch 拿到 `segments` 后，用 `multi_segment.expand_segments_to_samples`（[`multi_segment.py:118`](../dressage/rollout/multi_segment.py#L118)）把每个 segment 变成一个训练 `Sample`：

```python
def expand_segments_to_samples(template_sample, segments, *, args,
                               agent_response="", session_id=None, instance_id=None):
    out = []
    sorted_segments = sorted(segments, key=lambda s: int(s.get("segment_index", 0)))
    rollout_id = getattr(template_sample, "index", None)   # 同一 trajectory 的段共享
    for i, segment in enumerate(sorted_segments):
        sample = copy.deepcopy(template_sample)
        sample.rollout_id = rollout_id
        write_sample_from_segment(sample, args=args, segment=segment, ...)  # 填 tokens/loss_mask/logprobs/status
        sample.metadata["parent_traj_id"] = session_id
        sample.metadata["segment_index"] = int(segment.get("segment_index", 0))
        if i != len(sorted_segments) - 1:
            sample.reward = 0.0            # 只有最后一段（anchor）跑 reward_fn
        out.append(sample)
    return out
```

**关键点**：
- 一条 trajectory → N 个 segment → N 个 `Sample`，全部共享 `rollout_id`（= 原 sample 的 `index`），slime 的 `build_dp_schedule` 靠它把同一 trajectory 的段放进同一个训练步。
- 只有**最后一段**（anchor，`segment_index` 最大）保留 `reward=None` 让 slime 跑 reward_fn；其余段先置 `reward=0.0`，由 `reward_post_process` 把 anchor 的 advantage 广播回所有兄弟段（§4.17、§5.4）。
- append-only 的普通对话就是 N=1，退化成"1 trajectory → 1 Sample"。

## 4.15 清理环境

dispatch 的 `finally` 里调 `schedule_terminate_paddock`（[`lifecycle.py:74`](../dressage/paddock/lifecycle.py#L74)）：

```python
finally:
    if initialized and paddock is not None:
        schedule_terminate_paddock(paddock, session_id=session_id, env_args=env_args)
```

它是 **best-effort 异步释放**：主流程立刻返回，不等沙箱真正回收；`terminate` RPC 超过 `DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC`（默认 30s）后转入后台 `asyncio.shield` 继续跑。进程退出前 `drain_terminate_tasks()`（[`lifecycle.py:91`](../dressage/paddock/lifecycle.py#L91)）会等所有后台 terminate 跑完，避免漏清沙箱。哪怕中间任何步骤抛异常，环境一定会被回收。

## 4.16 组级重试与 drain

回到 `generate_rollout_async`（[`fully_async_rollout.py:476`](../dressage/rollout/fully_async_rollout.py#L476)）：后台 worker 把完成的组放进队列，主循环 drain 出来，失败组按 `DRESSAGE_ROLLOUT_MAX_RETRIES`（默认 2）退回 `data_buffer` 重试；重试耗尽的组累计到阈值 `DRESSAGE_ASYNC_MAX_DROPPED_FAILED_GROUPS` 才报错：

```python
for group_id in list(completed_by_id.keys()):
    completed = completed_by_id.pop(group_id)
    if completed.is_failed:
        if _retry_count(completed.original_group) < max_retries:
            _increment_retry(completed.original_group)      # 清 session_id、+1 retry
            data_buffer.add_samples([completed.original_group])
            continue
        dropped_failed_groups += 1                            # 耗尽：丢弃并计数
        continue
    if not staleness_filter.keep_group(group_id, completed.result, logger):
        continue                                             # 过期组丢弃
    data.append(PendingGroup(group_id=group_id, samples=completed.result))
```

凑够 `rollout_batch_size` 个组就返回，其余继续在后台跑供下一步用。

到这里 rollout 阶段结束。

## 4.17 训练侧：`convert_samples_to_train_data`

> **这一节做的事**：rollout 阶段产出的是"32 个 Sample，每个带 prompt、tokens、reward、logprobs、loss_mask"。但 slime 的训练 loop 想要的是"一个 dict，里面键是 `tokens`/`rewards`/`loss_masks`/... 值是 list"。这一步就是这两种 shape 之间的转换。**核心是顺便在 ① 步做了 GRPO 组归一化**——把原始 reward 减去组均值（再可选除标准差），变成 PPO 公式里要用的 advantage。

slime 拿到 rollout 输出后，会展平成 `list[Sample]`（不再按组），然后调 Dressage 的 `convert_samples_to_train_data`（`dressage/rollout/convert_samples.py`）：

```python
def convert_samples_to_train_data(args, samples):
    if not samples:
        return {... empty dict ...}

    # ① reward 后处理（GRPO 归一化）
    raw_rewards, rewards = reward_post_process(args, samples)

    # ② 构造 loss_masks 列表，做断言
    loss_masks = []
    for sample in samples:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length

        assert len(sample.loss_mask) == sample.response_length, ...

        if getattr(sample, "remove_sample", False):
            sample.loss_mask = [0] * sample.response_length

        loss_masks.append(sample.loss_mask)

    # ③ 拼训练 batch dict
    train_data = {
        "tokens":           [s.tokens for s in samples],
        "response_lengths": [s.response_length for s in samples],
        "rewards":          rewards,            # 归一化后的 advantage
        "raw_reward":       raw_rewards,        # 归一化前
        "truncated":        [1 if s.status == TRUNCATED else 0 for s in samples],
        "sample_indices":   [s.index for s in samples],
        "loss_masks":       loss_masks,
    }

    # ④ 可选字段（如果有 rollout_log_probs 才加）
    if any(s.rollout_log_probs is not None for s in samples):
        train_data["rollout_log_probs"] = [s.rollout_log_probs for s in samples]
    ...

    return train_data
```

注意这里**没有 `total_lengths` 字段**——slime 的 Megatron actor 内部会从 `tokens` 自动派生。

`reward_post_process`（`dressage/training/reward_post_process.py:77`）做 GRPO 归一化：

```python
def reward_post_process(args, samples):
    raw_rewards = [float(s.reward) for s in samples]

    if args.advantage_estimator not in ("grpo", "gspo", ...) or not args.rewards_normalization:
        return raw_rewards, list(raw_rewards)

    # 找出多段 trajectory 的 parent_traj_id 分组
    parent_groups = defaultdict(list)
    for i, s in enumerate(samples):
        ptid = s.metadata.get("parent_traj_id") if s.metadata else None
        if ptid:
            parent_groups[ptid].append(i)

    # 按 group_index 分组算归一化
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        gi = s.group_index if s.group_index is not None else -1
        # 多段 trajectory 只用代表段参与归一化（避免一条 trajectory 算多次）
        if i in parent_representative:
            groups[gi].append((i, raw_rewards[i]))
        elif not (s.metadata and s.metadata.get("parent_traj_id")):
            groups[gi].append((i, raw_rewards[i]))

    rewards = list(raw_rewards)
    for gi, members in groups.items():
        values = [m[1] for m in members]
        mean_val = sum(values) / len(values)
        normalized = [v - mean_val for v in values]
        if grpo_std:
            std = (sum(v**2 for v in normalized) / len(normalized)) ** 0.5
            if std > 1e-6:
                normalized = [v / std for v in normalized]
        for idx, norm in zip([m[0] for m in members], normalized):
            rewards[idx] = norm

    # 多段 trajectory：把代表段的 advantage 广播给所有兄弟段
    for ptid, seg_indices in parent_groups.items():
        rep_reward = rewards[seg_indices[0]]
        for idx in seg_indices:
            rewards[idx] = rep_reward

    return raw_rewards, rewards
```

对我们这个例子（假设这组 4 个样本 reward 是 `[1.0, 0.0, 1.0, 0.0]`）：

```
mean = 0.5
normalized = [+0.5, -0.5, +0.5, -0.5]
(若 grpo_std)
std = 0.5
final = [+1.0, -1.0, +1.0, -1.0]
```

这就是 GRPO advantage（§0.9）。

## 4.18 进入 slime 训练循环（黑盒）

`convert_samples_to_train_data` 返回的 dict 进入 slime 的 Megatron actor。slime 内部：

1. 把 `tokens` 转成 tensor，做 padding/packing。
2. 把所有样本前向一次模型（即时计算 `logprob_new`）。
3. 算 ratio = `exp(logprob_new − rollout_log_probs)`。
4. 算 PPO clipped loss + KL + entropy。
5. 反传 + Adam step。

这部分**完全是 slime 的代码**，我们不动。具体公式见 §0.8 - §0.9。

到这里，一次完整的"rollout + 训练步"结束。slime 进入下一轮 rollout，再次调 `generate_rollout`，循环。

---

# Part 4B — 黑盒 agent 端到端：blackbox_dispatch 详解

> Part 4 跟着 whitebox 样本走完了一条 trajectory。本节跟着 **`agent_mode = "blackbox"`** 的样本走一遍——它走的是另一条插件链，流量方向、时序、错误模型都和白盒不同。
>
> 强烈建议先看完 §1.2、决策 2、决策 4，然后回来看这一节。

## 4B.0 入口：slime 的 --custom-generate-function-path

白盒走的是 slime 自己的 `slime.rollout.sglang_rollout.generate_and_rm_group`：rollout 进程**主动**给 SGLang 发请求，logprob/turn 由 proxy 顺手记账。

黑盒不一样：

- 训练脚本里多了一个 flag：

  ```bash
  --custom-generate-function-path dressage.rollout.generate.blackbox_dispatch.generate
  ```

- slime 的 rollout function（`fully_async_rollout` / `sync_rollout` / `partial_async_rollout`）拿到 sample 之后，看到这个 flag 就把样本交给 dressage 自己写的 `generate(args, sample, sampling_params, evaluation=False)`。
- 进入 dressage 之后，**LLM 流量方向反过来**——dressage 只负责申请沙箱、把 proxy URL 告诉沙箱、然后等沙箱里的 agent 自己跑；agent 跑的时候反向把 chat 请求打到 dressage proxy 上，proxy 替它调 SGLang，顺手记账。

```
  whitebox：rollout ──► proxy ──► sglang
                       (caller is rollout)

  blackbox：rollout ──► paddock.init ──► sandbox 起来
           rollout ──► paddock.register_agent (告诉 sandbox proxy 在哪)
           rollout ──► paddock.call_agent (一次 RPC 等 agent 跑完)
                                      └► sandbox 里 agent ──► proxy ──► sglang  (反向回打)
           rollout ──► proxy_client.finalize / read_trajectory
```

两件事要立刻明白：

1. **dressage rollout 进程不再发 LLM 请求**——它发的是给沙箱的 HTTP 调用。
2. **proxy 仍然是单点收口**。沙箱里的 agent 用的 OpenAI / SGLang URL 是从 `register_agent` 注入的 proxy 公网 URL，所有 LLM turn 都进同一个 `SessionManager` session，session 主键就是 `session_id`（Dressage 视角下 `session_id` ≡ `traj_id`）。

代码主入口：[`blackbox_dispatch.generate`](../dressage/rollout/generate/blackbox_dispatch.py#L69)，约 230 行走完整个生命周期。下面分九步剖开。

> **注意（重构后的文件布局）**：`blackbox_dispatch.py` 现在只留调度骨架，具体逻辑拆到了几个模块：session_id 生成和沙箱预热在 [`rollout/prewarm/store.py`](../dressage/rollout/prewarm/store.py)（`ensure_blackbox_session_id` / `claim_prewarm`）；execute_cmds 钩子在 [`paddock/blackbox/execute_hooks.py`](../dressage/paddock/blackbox/execute_hooks.py)（`parse_blackbox_execute_cmds` / `execute_blackbox_cmds_for_stage`）；agent 失败归一化在 [`paddock/blackbox/failures.py`](../dressage/paddock/blackbox/failures.py)（`failure_from_call_agent_exception` / `failure_from_payload_state` / `expected_abort_from_call_agent_exception`）；segment→sample 在 [`rollout/multi_segment.py`](../dressage/rollout/multi_segment.py)。下面的行号是骨架里对应调用点的近似位置。

## 4B.1 元数据落桌：session_id / instance_id / blackbox_type

[`blackbox_dispatch.py:90`](../dressage/rollout/generate/blackbox_dispatch.py#L90)：

```python
metadata = sample.metadata
metadata.pop("blackbox_error", None)             # 清上一次重试遗留的错误
metadata.pop("blackbox_error_log_path", None)
metadata["execute_cmds"] = []                    # 本轮探针记录器
session_id  = ensure_blackbox_session_id(sample) # "bbs-" + (metadata.session_id 或 uuid)
instance_id = _instance_id(sample)               # GRPO 组主键（同 prompt 多 sample 共享）
metadata["session_id"]  = session_id
metadata["instance_id"] = instance_id
blackbox_type = normalize_blackbox_type(
    metadata.get("blackbox_type") or DEFAULT_BLACKBOX_TYPE
)                                                # "opencode" / "openclaw" / ...
```

要点：

- **`session_id` 是这一条样本（rollout 视角）的唯一 id**，proxy 内部就是用它做 session key。后续 `paddock.init(session_id, ...)` 把 sandbox 也绑定到这个 id 上——所以 "1 sample = 1 sandbox = 1 proxy session" 的三位一体不变（决策 3）。
- **`instance_id` 是 GRPO 组 id**：同一个 prompt 复制 N 份得到 N 个样本，它们的 `instance_id` 相同、`session_id` 不同。`reward_post_process` 按 `instance_id` 做组内归一化。
- **`blackbox_type`** 控制 paddock 用哪套默认值（沙箱镜像、启动 cmd、backend_options），见 [`paddock/blackbox/common/defaults.py`](../dressage/paddock/blackbox/common/defaults.py)。

紧接着从 metadata 里挑出允许覆写沙箱构造的字段组成 `env_args`（`paddock_env_args_from_metadata`，[`runtime.py:77`](../dressage/rollout/generate/runtime.py#L77)）：

```python
# 只透传这几个 key：sandbox_timeout_sec / sandbox_image / sandbox_cmd /
# sandbox_extra_params / inject_files（外加 blackbox_type）
env_args = paddock_env_args_from_metadata(metadata, extra_env_args={"blackbox_type": blackbox_type})
```

这一步决定了「这条样本想用什么镜像 / 后端 / 入口命令」——JSONL 里写什么就生效什么。

## 4B.2 paddock.init：申请一个沙箱

[`blackbox_dispatch.py:118`](../dressage/rollout/generate/blackbox_dispatch.py#L118)：

```python
proxy_client = get_proxy_client()               # DRESSAGE_PROXY_URL 上的 ProxyClient
handle = await claim_prewarm(session_id)         # 先看有没有预热好的沙箱
if handle is not None:
    paddock, state, env_args = handle.paddock, handle.state, handle.env_args
else:
    paddock = get_paddock_from_env(allow_whitebox_mode=False)   # 按 MODE/CLASS 建 paddock
    state = await maybe_await(
        paddock.init(session_id, metadata.get("env_type"), env_args)
    )
initialized = True
```

这一行一返回，就有了一个**专属于这条样本**的沙箱 HTTP 端点 `state.sandbox_url`。具体怎么拿到的因 sandbox provider 不同：

- `local_bwrap` provider：调本地 Ray manager actor 的 `acquire(...)`，从节点级 supervisor 的槽位池里拿一个空闲槽。
- `harness` provider：发 `POST /v1/sandboxes/register` 给 harness router，轮询 `ready=True`。
- `e2b` provider：调 E2B SDK 从预构建模板起云沙箱。

三种 provider 起完沙箱后，`BlackboxAgentPaddock.init` 都返回同形态的 `SandboxState(trajectory_id, sandbox_url, sandbox_id, ecs_ip, raw_register_response)`（[`paddock/blackbox/common/state.py`](../dressage/paddock/blackbox/common/state.py)）。后续业务代码不需要关心是哪一种。详见 §4C。

> **注意**：`paddock.init` 在所有实现里都是 `async def`。`maybe_await()` 是一个兼容包装器——如果传入的是协程就 await，如果是普通值就直接返回。

## 4B.3 paddock.register_agent：把 proxy URL 注入沙箱

[`blackbox_dispatch.py:146`](../dressage/rollout/generate/blackbox_dispatch.py#L146)：

```python
await maybe_await(
    paddock.register_agent(
        state,
        instance_id=instance_id,
        session_id=session_id,
        router_url=proxy_url(),          # dressage proxy 的可达 URL（config.proxy_url()）
        blackbox_type=blackbox_type,
        backend_options=backend_options,
    )
)
```

这一步是黑盒链路最关键的一环——**dressage 把自己的 proxy URL 通过 HTTP 推到沙箱里，告诉沙箱里的 agent："你之后调 LLM 都打到这个地址"**。

沙箱内部收到 `POST /v1/rollout/register` 之后做的事（dressage 仓库不管，住在 `opencode` / `openclaw` 之类的 blackbox 服务里）：

1. 启动一个 agent 工作进程（agent runtime，比如 OpenCode 的 React loop）。
2. 把环境变量 `OPENAI_BASE_URL` / `LLM_BASE_URL` 设成 `router_url`（即 dressage proxy 的公网 URL）。
3. 把 `bound_session_id`（== `session_id` == `traj_id`）记下来，后续每次发给 proxy 的请求都带这个 id（body 字段或者 `x-traj-id` header，见 §5.5）。

> **`router_url` 必须是"沙箱能访问到的"地址**，不能是 `127.0.0.1:8800` ——`_validate_public_proxy_url` 会兜底拦截 loopback 写法（见 [`paddock/blackbox/common/utils.py`](../dressage/paddock/blackbox/common/utils.py)）。这就是训练脚本要分别设 `DRESSAGE_PROXY_URL`（本地访问）和 `DRESSAGE_PROXY_PUBLIC_URL`（沙箱访问）的原因。

`backend_options` 是给具体 agent runtime 的运行时配置（compaction 策略、tool 列表、模型名等），由 [`_backend_options_for_register`](../dressage/rollout/generate/blackbox_dispatch.py) 从 `args` + `metadata` 里合并出来。

## 4B.4 before_agent execute_cmds：环境探针

[`blackbox_dispatch.py:156`](../dressage/rollout/generate/blackbox_dispatch.py#L156)：

```python
await execute_blackbox_cmds_for_stage(
    paddock, state, metadata,
    schedule=execute_cmd_schedule,
    session_id=session_id,
    stage="before_agent",
)
```

命令来自数据集 metadata 里的 `blackbox_execute_cmds`（由 `parse_blackbox_execute_cmds` 解析），例如：

```python
{"stage": "before_agent", "name": "env_check",
 "cmd": "python -V && ls -la", "timeout": 30, "required": False},
```

实际行为：通过 `paddock.execute_cmd(state, session_id=..., cmd=...)` 在沙箱里跑一个 shell 命令，把 stdout/stderr/return_code 记到 `metadata["execute_cmds"]` 里。**这是给训练 reward 准备的环境快照**——可以拿来检查 agent 起来之前文件树长什么样、Python 版本对不对，等等。

`before_agent` / `after_agent` 命令由 JSONL 每条样本自带。实现见 [`execute_hooks.py`](../dressage/paddock/blackbox/execute_hooks.py) 的 `execute_blackbox_cmds_for_stage`。

## 4B.5 paddock.call_agent：把控制权交给黑盒 agent

[`blackbox_dispatch.py:167`](../dressage/rollout/generate/blackbox_dispatch.py#L167)：

```python
try:
    call_payload = await maybe_await(
        paddock.call_agent(
            state,
            session_id=session_id,
            messages=_chat_messages_from_prompt(sample.prompt),
            metadata={"source": "dressage", **metadata},
        )
    )
    call_succeeded = True
except Exception as exc:
    if agent_failure := failure_from_call_agent_exception(exc):
        record_agent_failure_metadata(metadata, agent_failure)
        # 可提前收集的失败（如 max_steps/context_overflow）标 early stop 继续，否则 raise
        ...
    else:
        raise
```

`call_agent` 是**一次同步阻塞的 HTTP 调用**——dressage 给沙箱发一个 `POST /v1/sessions/{session_id}/messages`，body 是用户 prompt。沙箱内部 agent 拿到这个 prompt 之后开始它自己的 ReAct / planning / tool-calling 循环——这一段 dressage 完全黑盒，可能跑几十秒到几十分钟，期间 agent 会反复打 dressage proxy 拉模型推理（这就是 §4B.0 那张图里的反向箭头）。

agent 自认为跑完之后，HTTP 响应回来，body 长这样：

```json
{
  "agent_response": "final answer text",
  "state": "COMPLETED",   // or FAILED / TIMEOUT / ABORTED
  "trace": { ... }         // 可选：agent 内部 step 信息
}
```

**dressage 要做两件防御**：

1. **HTTP 异常 → 业务异常**：[`failure_from_call_agent_exception`](../dressage/paddock/blackbox/failures.py) 把 httpx 抛的连接错 / 超时映射成结构化的 `BlackboxAgentFailure`，写进 metadata 后再决定 raise 还是当 early stop 收集。
2. **响应 state 检查**：`failure_from_payload_state` 验证 agent 自报的 `state == COMPLETED`，否则按失败处理（[`blackbox_dispatch.py:194`](../dressage/rollout/generate/blackbox_dispatch.py#L194)）。

失败的 sample 会走到 §4B.9 的统一错误分支，最终被打 `Status.ABORTED`，由 rollout 框架决定要不要重试。

## 4B.6 黑盒 agent 内部如何串接 proxy

虽然这部分代码不在 dressage 仓库（OpenCode / OpenClaw / 任何用户自带 agent），但**契约**是 dressage 这一侧定的。一个**最小合规黑盒 agent**长这样（伪代码）：

```python
# blackbox server 收到 register 调用之后初始化：
llm_base_url = payload["router"]                # = DRESSAGE_PROXY_PUBLIC_URL
llm_session_id = payload["bound_session_id"]    # = dressage session_id
openai_client = OpenAI(base_url=llm_base_url, api_key="any")

# 之后每次发推理：
resp = openai_client.chat.completions.create(
    model="default",
    messages=messages,
    extra_headers={"x-traj-id": llm_session_id},   # ★ 关键
    # 或者 extra_body={"traj_id": llm_session_id}
)
```

proxy 收到带 `x-traj-id` header 的请求时（[`server.py:_runtime_ids_from_request`](../dressage/proxy/server.py#L366)）会把它当做 session key，把 turn 记进同一个 `SessionManager` session。这就是为什么 dressage 在 §4B.8 能用 `session_id` 把所有零散 turn 取回来。

**三种透传方式**（任选其一，沙箱 agent 适配哪种用哪种）：

1. body：`{"session_id": "..."}` 或 `{"trajectory_id": "..."}`
2. header：`x-session-id` / `x-traj-id` / `X-SMG-Routing-Key`
3. extra_body：`{"extra_body": {"traj_id": "..."}}`

详见 §5.5 + [`server.py:_trajectory_id_from_body`](../dressage/proxy/server.py#L412)。

## 4B.7 after_agent execute_cmds：采集环境产物

[`blackbox_dispatch.py:201`](../dressage/rollout/generate/blackbox_dispatch.py#L201)：

```python
await execute_blackbox_cmds_for_stage(
    paddock, state, metadata,
    schedule=execute_cmd_schedule,
    session_id=session_id,
    stage="after_agent",
)
```

数据集里为这条样本配的 `after_agent` 命令会在这里跑，例如：

```python
{"stage": "after_agent", "name": "inspect_files",
 "cmd": "find . -maxdepth 2 -type f", "timeout": 30, "required": False},
```

用途：agent 跑完之后，看一下沙箱里多了哪些文件——通常是给 reward 函数算 "任务有没有完成" 用的（比如：要求 agent 生成 `solution.py`，after_agent 命令就 `cat solution.py` 把内容读出来塞进 metadata）。

reward 函数一会儿能从 `sample.metadata["execute_cmds"]` 拿到所有探针的 stdout/stderr/return_code（每条 stdio 有 4096 bytes 上限）。

## 4B.8 Finalize + read_trajectory + 写回 Sample

这一步和白盒**完全一样**——黑盒和白盒在 proxy 内的记账模型是统一的，所以最后取数据的接口是同一个：

```python
await proxy_client.finalize_session(
    session_id, instance_id=instance_id, label=getattr(sample, "label", None)
)
trajectory_payload = await proxy_client.read_trajectory(
    trajectory_id=session_id,
    instance_id=instance_id,
    drain=True,                                 # 取走的同时删除
)
segments = trajectory_payload.get("data") or []
result = multi_segment.expand_segments_to_samples(
    sample, segments, args=args,
    agent_response=agent_response,
    session_id=session_id, instance_id=instance_id,
)
```

要点：

- `finalize_session` 让 proxy 把这个 session 里的所有 step 切成 1..N 个训练段（§4.11、§5.1）——黑盒 agent 调了多少次 LLM、proxy 就有多少条 step。
- `read_trajectory(drain=True)` 取出来同时从 `TrajectoryStore` 删除——避免内存泄漏。
- `expand_segments_to_samples`（§4.14）把每段变成一个 `Sample`：多段共享 `rollout_id`，只有最后一段（anchor）跑 reward_fn，其余段 `reward=0.0` 由 `reward_post_process` 广播。黑盒 agent 做 history compaction（rewrite）时就会产生多段。详见 §4.14 / §5.4。
- 与此同时把 trajectory payload 异步落盘到 `DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR`（按 `instance_id/session_id/` 分目录，方便事后审计）。

之后 sample 进入 §4.16 的重新分组、§4.17 的 `convert_samples_to_train_data`、§4.18 的 slime 训练 step——和白盒一条流。

## 4B.9 错误处理与 best-effort terminate

```python
except Exception as exc:
    expected_abort = expected_abort_from_call_agent_exception(exc)
    if expected_abort is None:                                     # 真失败
        logger.warning("blackbox rollout failed for session_id=%s: %s", session_id, ...)
        error_log_path = await _ARTIFACT_WRITER.write_error(exc, ...)  # 落盘错误现场
        metadata["blackbox_error_log_path"] = str(error_log_path)
        record_blackbox_abort_for_retry(metadata, session_id, exc)
    else:                                                          # 预期中断（权重更新抢占等）
        metadata["blackbox_expected_abort"] = expected_abort
    multi_segment.mark_aborted_no_grad(sample, session_id=session_id, instance_id=instance_id)
    _set_status(sample, "ABORTED")
    return sample
finally:
    if initialized and paddock is not None:
        schedule_terminate_paddock(paddock, session_id=session_id, env_args=env_args)
```

两件事值得记：

1. **错误现场会落盘**到 `DRESSAGE_TRAJECTORY_ERROR_LOG_DIR/<instance_id>/<session_id>/` ——里面有完整的异常类型、stack、最后一个 HTTP 响应（headers 脱敏 + body 截断）、当时的 `metadata` / `env_args` / `sandbox_url`。这是事后调试唯一的现场，所以即使写日志失败也只是 warning，不影响主流程。
2. **terminate 是 best-effort 异步的**（[`schedule_terminate_paddock`](../dressage/paddock/lifecycle.py#L74)）。
   - 主线程立刻返回，不等沙箱真正释放——以免 sandbox 清理慢拖死 rollout。
   - 调 `paddock.terminate(session_id, env_args)`，超时 `DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC`（默认 30s）。
   - 超时之后把任务 `asyncio.shield` 起来后台继续跑——"调用方放手" ≠ "槽位空闲"。后者由 cluster manager 在背地里把 lease 推到 `READY` 才算（详见 §4C.3 的状态机）。
   - 进程退出前有一个 [`drain_terminate_tasks`](../dressage/paddock/lifecycle.py#L91) 等所有 background terminate 跑完，否则 ray cluster 可能漏掉沙箱清理。

失败样本走 `Status.ABORTED`，由 rollout 层（`fully_async_rollout._mark_no_grad_failed` / 重试预算）决定要不要重新做一次。重试预算耗尽的样本最终会带着空 trajectory 和零 loss_mask 进训练——`remove_sample = True`，参与 GRPO 组归一化但不贡献梯度（§5.6）。

---

# Part 4C — Paddock 与 Sandbox Provider 后端

> §4B 走完了 dressage 这一侧的黑盒流程，但故意把 `paddock.init` / `register_agent` / `call_agent` 当黑盒讲。本节把这层拆开。**关键：现在只有一个黑盒 paddock 类 `BlackboxAgentPaddock`，它把"沙箱放在哪"委托给一个可插拔的 `SandboxProvider`**。换后端 = 换 `DRESSAGE_SANDBOX_PROVIDER`，业务代码一行不动。

## 4C.1 三种 sandbox provider 的对比

`BlackboxAgentPaddock`（[`paddock/blackbox/paddock.py`](../dressage/paddock/blackbox/paddock.py)）负责 agent 交互（register/call/execute_cmd/pause/resume），沙箱的申请/释放交给 `SandboxProvider`（[`sandbox/provider.py`](../dressage/sandbox/provider.py)）。当前有三个 provider：

| 维度 | `local_bwrap` | `harness` | `e2b` |
|---|---|---|---|
| 实现 | [`sandbox/local/bwrap/provider.py`](../dressage/sandbox/local/bwrap/provider.py) | [`sandbox/remote/harness/provider.py`](../dressage/sandbox/remote/harness/provider.py) | [`sandbox/remote/e2b/provider.py`](../dressage/sandbox/remote/e2b/provider.py) |
| `create` 实现 | 调本地 detached Ray actor `LocalBwrapClusterManager.acquire(...)` 拿槽位租约 | `POST /v1/sandboxes/register` 给 harness router，轮询 `ready=True` | 调 E2B SDK 从预构建模板起云沙箱 |
| 沙箱寿命 | 长驻进程池，跑完 reset 复用 | 远程容器，按需起按需销毁 | E2B 云沙箱，按 timeout 存活 |
| 沙箱隔离 | bubblewrap (`bwrap`) 或 `direct`（debug） | 由 harness 服务负责 | 由 E2B 负责 |
| 启动依赖 | `ray start` + `dressage-local-bwrap-start` | `DRESSAGE_HARNESS_ROUTER_URL` 可达 | `DRESSAGE_E2B_API_KEY` + 预构建模板 |
| 资源调度 | dressage 自己做（slot pool + lease + 健康检查） | 由 harness 服务做 | 由 E2B 服务做 |
| 例子脚本 | `run_example_*_local.sh` | `run_example_*_remote.sh`（`DRESSAGE_SANDBOX_PROVIDER=harness`） | 数据集 metadata 配 E2B 模板 |

三个 provider 都实现同一份 `SandboxProvider` 协议：`create` / `terminate` / `get_public_url` / `run_command` / `read_file` / `write_file` / `inject_files`，返回同形状的 `SandboxLease`（[`sandbox/types.py`](../dressage/sandbox/types.py)）。所以 §4B 的 9 步在三种后端下行为一致，只是底下 Ray RPC / HTTP / E2B SDK 不同。

## 4C.2 BlackboxAgentPaddock 与 SandboxProvider

`BlackboxAgentPaddock.init`（[`paddock/blackbox/paddock.py:61`](../dressage/paddock/blackbox/paddock.py#L61)）构造一个 `SandboxSpec`，交给 provider 起沙箱，再取出 blackbox 服务端点、做健康检查：

```python
async def init(self, traj_id, env_type=None, env_args=None, **kwargs):
    spec = SandboxSpec(
        trajectory_id=traj_id,
        env_type=env_type,
        env_args={**(env_args or {}), **kwargs},
        services=(SandboxServiceSpec(name="blackbox",
                                     port=self._blackbox_port, health_path="/health"),),
        metadata={"paddock_mode": "blackbox"},
    )
    lease = await self._provider.create(spec)               # 各 provider 各显神通
    endpoint = lease.endpoints.get("blackbox") or await self._provider.get_public_url(...)
    if self._wait_health:
        await self._client.health(endpoint)                 # GET /health
    return SandboxState(trajectory_id=traj_id, sandbox_url=endpoint.url, ...)  # 向后兼容
```

之后的 `register_agent` / `call_agent` / `execute_cmd` 都通过 `BlackboxServerClient`（[`paddock/blackbox/client.py`](../dressage/paddock/blackbox/client.py)）打到 `endpoint`——沙箱内部跑的是同一套 blackbox agent server（opencode/openclaw/...），所以三种 provider 共享一份沙箱内协议：

```
rollout ──► paddock.init      ──► provider.create ──► SandboxLease（含 blackbox endpoint）
rollout ──► paddock.register_agent (POST /v1/rollout/register，注入 proxy 公网 URL)
rollout ──► paddock.call_agent     (POST /v1/sessions/<sid>/messages，等 agent 跑完)
                                    └► 沙箱内 agent ──► proxy ──► sglang（反向回打）
rollout ──► proxy_client.finalize / read_trajectory
```

**关键环境变量**：

| 变量 | 默认 | 作用 |
|---|---|---|
| `DRESSAGE_PADDOCK_MODE` | `blackbox` | 选 `BlackboxAgentPaddock` / `WhiteboxToolPaddock` |
| `DRESSAGE_SANDBOX_PROVIDER` | `local_bwrap` | 选 `local_bwrap` / `harness` / `e2b` |
| `DRESSAGE_PROXY_PUBLIC_URL`（或 `DRESSAGE_PROXY_URL`） | (必填) | 沙箱回打 proxy 的 URL；不能是 loopback |
| `DRESSAGE_BLACKBOX_PORT` | `31000` | 沙箱内 blackbox 服务端口 |
| `DRESSAGE_HARNESS_ROUTER_URL`（回退 `DRESSAGE_SANDBOX_ROUTER_URL`） | `http://accio-agentic-rl-router.alibaba-inc.com` | harness router endpoint |
| `DRESSAGE_HARNESS_REGISTER_MAX_ATTEMPTS` / `_POLL_INTERVAL` | `60` / `5`s | harness 注册轮询 |
| `DRESSAGE_SANDBOX_DEFAULT_IMAGE` | (defaults) | 远程 provider 的默认镜像/模板，被 metadata.sandbox_image 覆盖 |
| `DRESSAGE_E2B_API_KEY` / `DRESSAGE_E2B_TIMEOUT_SEC` | — / `3600` | E2B provider 配置 |

## 4C.3 local_bwrap provider 三层架构

代码：[`sandbox/local/bwrap/`](../dressage/sandbox/local/bwrap/)。整个目录就一个目的——**在不依赖远程沙箱平台的本地多 GPU 节点上提供等价的沙箱池**。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ rollout 进程 (任意节点)                                                     │
│   LocalBwrapSandboxProvider (provider.py)                                  │
│      │ ray.remote(manager.acquire / release / status)                       │
│      ▼                                                                      │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Detached Ray actor: LocalBwrapClusterManager  (manager.py)              │ │
│ │   - 全局节点表 / 槽位表 / 租约表                                          │ │
│ │   - 心跳检测 / 健康过滤                                                   │ │
│ │   - 后台任务：lease lifecycle、health refresh、cleanup                    │ │
│ └────────────┬────────────────────────────────────────────────────────────┘ │
│              │ 每个节点一份                                                  │
│              ▼                                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Ray actor per node: node supervisor  (supervisor.py)                    │ │
│ │   - 维护 N 个 slot 的状态机：READY -> LEASED -> RESETTING -> RESTARTING │ │
│ │   - 每个 slot 一个 LocalSandboxRunner 子进程                              │ │
│ │   - 处理 lease / release / reset / archive                               │ │
│ └────────────┬────────────────────────────────────────────────────────────┘ │
│              ▼                                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ subprocess per slot: LocalSandboxRunner  (runner.py)                    │ │
│ │   - bubblewrap (默认) / direct (debug)                                   │ │
│ │   - 跑实际的 blackbox agent server (opencode / openclaw / ...)           │ │
│ │   - 暴露 /v1/rollout/register 等 HTTP 端点                                │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Core / Actor 模式**：manager/supervisor 的纯异步逻辑（`*Core`）与套了 `@ray.remote` 壳的 actor 代理分离，这样 dressage 测试可以直接跑 Core 不起 ray（见 [`tests/test_ray_blackbox_scheduler.py`](../tests/test_ray_blackbox_scheduler.py)、[`tests/test_blackbox_node_supervisor.py`](../tests/test_blackbox_node_supervisor.py)）。

**租约状态机**（[`docs/local-blackbox-sandbox.md`](../docs/local-blackbox-sandbox.md)）：

```text
READY ─acquire──► LEASED ─release──► RESETTING ──► RESTARTING ──► READY
                              \─manager 视角同时进入 RELEASING，直到看到 READY 才算释放完
```

关键不变量：**"调用方放手" ≠ "槽位可复用"**。dressage rollout 调 `terminate` 之后立刻继续，但 cluster manager 在底下 still 把 lease 标 `RELEASING`，要等到该 slot 的 `LocalSandboxRunner` 子进程被停掉、archive 文件夹归档（可选）、新的 runner 起来上报 `READY` 才会从 lease 表里删——下一条 trajectory 不会撞到上一条留下的端口、文件、僵尸进程。这套机制由每个 slot 的**重置锁 + generation 号**保护幂等（重复 release 安全无副作用）。

**`create` 调用链**（[`local/bwrap/provider.py:49`](../dressage/sandbox/local/bwrap/provider.py#L49)）：

```python
async def create(self, spec: SandboxSpec) -> SandboxLease:
    # 校验连接到的池 pool_mode 与 paddock_mode 匹配（blackbox / command_only）
    payload = await _remote_call(self._manager, "acquire",
                                 trajectory_id=spec.trajectory_id,
                                 env_type=spec.env_type, env_args=spec.env_args)
    lease = SandboxLease(trajectory_id=spec.trajectory_id, provider="local_bwrap",
                         sandbox_id=payload.get("lease_id"), ...)
    lease.endpoints["blackbox"] = SandboxEndpoint(url=payload["sandbox_url"], headers={})
    return lease
```

`acquire` 在 manager 内部：找出有空闲 slot 的健康节点 → 通知 supervisor 标 LEASED → 返回该 slot 的 sandbox_url（`http://node_ip:slot_port`）。`BlackboxAgentPaddock` 拿到 lease 后用它做 `register_agent` / `call_agent`——**沙箱内部协议和 harness/e2b 完全一致**。

**关键环境变量**（最常调）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `DRESSAGE_LOCAL_BWRAP_RAY_NAMESPACE` | `dressage` | Ray namespace |
| `DRESSAGE_LOCAL_BWRAP_MANAGER_NAME` | `dressage_local_bwrap_manager` | detached actor 名 |
| `DRESSAGE_LOCAL_BWRAP_POOL_MODE` | 由 `DRESSAGE_PADDOCK_MODE` 推导 | `blackbox` / `command_only`（whitebox 用后者） |
| `DRESSAGE_LOCAL_BWRAP_TOTAL_SERVERS` | `512` | 集群总槽数 |
| `DRESSAGE_LOCAL_BWRAP_BASE_PORT` | `31000` | slot HTTP 端口段起始 |
| `DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC` | `1800` | 等空槽的最大时间 |
| `DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC` | `30` | rollout 等 release RPC 的最大时间（不影响实际清理） |
| `DRESSAGE_BLACKBOX_RUNNER_MODE` | `bwrap` | `bwrap` / `bubblewrap` / `direct` |

完整变量矩阵看 [`docs/local-blackbox-sandbox.md`](../docs/local-blackbox-sandbox.md)、[`docs/paddock-sandbox-provider-implementation.md`](../docs/paddock-sandbox-provider-implementation.md) 和示例脚本头部。

## 4C.4 Local 沙箱隔离：bubblewrap vs direct

`LocalSandboxRunner`（[`runner.py`](../dressage/sandbox/local/bwrap/runner.py)）只支持两种模式：

| 模式 | 值 | 用途 |
|---|---|---|
| **bubblewrap** | `bwrap` / `bubblewrap` | **默认**。轻量级 in-container 沙箱：只读绑定 `/usr` `/bin` `/lib*` `/etc` + 当前 Python 运行时根 + `PYTHONPATH` 路径 + agent 安装目录；可写区是 slot 自己的 `home/work/runtime/tmp` 目录 |
| **direct** | `direct` | **debug only**。直接在宿主机路径上跑 blackbox server，不做隔离。任何沙箱里的进程能看到宿主全部文件——只在排查 "是不是沙箱本身坏了" 时用 |

几个工程实践决定（已在 [`docs/local-blackbox-sandbox.md`](../docs/local-blackbox-sandbox.md) 解释，搬到这里方便参考）：

- **网络命名空间默认不隔离** (`DRESSAGE_BLACKBOX_BWRAP_UNSHARE_NET=0`)：因为外层 supervisor 必须通过 HTTP 访问槽位的 `/health` / lease / release。
- **PID 命名空间默认不隔离**：因为 OpenClaw runtime monitor 需要看到子进程。
- **`--disable-userns` 默认关**：很多容器宿主把 `/proc/sys/user` 设了只读，这个选项会在 Python 起来之前 fail。
- **`/proc` 默认 ro-bind 而非 `--proc /proc`**：避免 bubblewrap 在容器里挂不上 procfs。

nsjail 模式已被移除（曾经存在过，2026 年这一版只剩 `bwrap` 和 `direct`）。

## 4C.5 启动顺序：手动 vs auto-start

**手动**（推荐用来调试）：

```bash
# 1. 起 ray head（带 blackbox_slots resource，否则 manager 调度不到节点）
ray start --head --resources='{"blackbox_slots":8,"blackbox_node":1}'

# 2. 起 cluster manager + 预热槽位
dressage-local-bwrap-start          # = python -m dressage.sandbox.scripts.start_local_bwrap

# 3. 看状态
dressage-local-bwrap-status

# 4. 起 dressage proxy + 训练
dressage-proxy --sglang-router-url ... --tokenizer-path ... &
# ... slime train_async ...

# 5. 收尾
dressage-local-bwrap-stop
```

> `dressage-local-blackbox-{start,status,stop}` 和 `dressage-blackbox-{start,status,stop}` 是兼容别名，等价于强制 `pool_mode=blackbox` 的 `dressage-local-bwrap-*`。

**Auto-start**（生产默认）：示例脚本 `run_example_*_local.sh` 里把 `DRESSAGE_LOCAL_BWRAP_AUTO_START=1` 打开，启动序列变成：

1. 脚本起 dressage proxy。
2. 脚本起 ray head + 远端 ray worker（通过 `HOSTFILE` ssh）。
3. 脚本调 `python -m dressage.sandbox.scripts.start_local_bwrap` 起 manager + 预热槽位（见 [`run_example_qwen3.5_4b_async_local.sh:294`](../examples/scripts/run_example_qwen3.5_4b_async_local.sh#L294)）。
4. 脚本 `ray job submit` 启动 slime `train_async.py`，rollout 进程内的 `LocalBwrapSandboxProvider` 用 `ray.get_actor(manager_name, namespace=...)` 拿到刚起好的 manager。
5. 脚本 `trap cleanup EXIT` 兜底：训练退出时先停 local_bwrap 池、再 `ray stop --force`、再 kill proxy（见 [`run_example_qwen3.5_4b_async_local.sh:253`](../examples/scripts/run_example_qwen3.5_4b_async_local.sh#L253)）。

**Remote provider（harness/e2b）没有这一套**——rollout 里只要把 `DRESSAGE_SANDBOX_PROVIDER=harness`（或 `e2b`）换上就够，无需起 ray cluster manager。

## 4C.6 怎么选：决策清单

| 你的场景 | 选哪种 |
|---|---|
| 你已经有一套远程沙箱平台（公司内网 harness router） | **`harness`**，零额外基础设施 |
| 你想在本地多 GPU 机器上自包含跑 RL 训练（无外部沙箱依赖） | **`local_bwrap`** |
| 你想用 E2B 云沙箱（有预构建模板） | **`e2b`** |
| 你在做 dressage 自身开发 / 调试黑盒链路 | **`local_bwrap` + `DRESSAGE_BLACKBOX_RUNNER_MODE=direct`**（关掉 bubblewrap，方便挂 debugger） |
| 你需要在 CI 里跑端到端测试 | 都不要——测试里注入 mock provider / mock paddock |
| 你需要不同样本用不同后端 | 不支持，`DRESSAGE_SANDBOX_PROVIDER` 是进程级单例 |

切换后端只改环境变量：

```bash
# Harness（远程）
export DRESSAGE_PADDOCK_MODE=blackbox
export DRESSAGE_SANDBOX_PROVIDER=harness
export DRESSAGE_HARNESS_ROUTER_URL=http://your-router
export DRESSAGE_PROXY_PUBLIC_URL=http://your-rollout-host:8800

# 本地 bwrap
export DRESSAGE_PADDOCK_MODE=blackbox
export DRESSAGE_SANDBOX_PROVIDER=local_bwrap
export DRESSAGE_LOCAL_BWRAP_AUTO_START=1
export DRESSAGE_PROXY_PUBLIC_URL=http://your-rollout-host:8800
```

业务代码（`blackbox_dispatch.generate`、reward 函数、训练 flag）都不变。

--

# Part 4D — 新功能与进阶机制

> 本节介绍 Dressage 后续迭代引入的进阶机制。这些功能不影响核心 rollout → train 流程，但对于生产环境的大规模训练很重要。

## 4D.1 配置模块：`dressage/config/config.py`

[`dressage/config/config.py`](../dressage/config/config.py)（178 行）是集中化的配置模块，通过环境变量提供运行时参数：

- `token_build_defaults()`（L142-L160）：根据 `DEFAULT_TOKEN_BUILD_MODEL = "qwen3_5"`（L21）自动推导 `model_mask_type`、`model_tool_call_type`、`model_reasoning_type`、`tito_model`（返回 `TokenBuildDefaults` dataclass；注意字段带 `model_` 前缀，且 `model_reasoning_type` 是 `"qwen3"`）。
- `paddock_mode()`（L90）：读取 `DRESSAGE_PADDOCK_MODE` 环境变量（`blackbox` / `whitebox`，默认 `blackbox`）。
- `sandbox_provider()`（L94）：读取 `DRESSAGE_SANDBOX_PROVIDER`（`local_bwrap` / `e2b`，默认 `local_bwrap`）。
- `sglang_router_url()`（L81）：读取 `SGLANG_ROUTER_URL`（或 `SGLANG_ROUTER_HOST` + `SGLANG_ROUTER_PORT` 拼接）。
- `proxy_url()`（L69）：读取 `DRESSAGE_PROXY_URL`（缺省用 `PROXY_PUBLIC_HOST`/`master_addr()` + `PROXY_PORT`）。

使用方式：在启动脚本里设环境变量，config 模块自动推导其余参数。无需在代码里硬编码。

## 4D.2 GenerationController：pause / resume / shutdown

[`dressage/proxy/generation_controller.py`](../dressage/proxy/generation_controller.py)（851 行）管理推理生命周期的抢占式调度：

- `generate_preemptible()`（L159）：可被中断的推理调用。在 partial rollout 场景下，训练步需要更新权重时，proxy 会先 pause 所有正在进行的推理，等权重更新完再 resume。
- `pause()`（L454）：暂停所有正在进行的推理请求，让 SGLang 释放 GPU 显存。
- `resume()`（L562）：恢复被暂停的推理。
- `shutdown()`（L637）：优雅关闭。
- `state()`（L664）：查询当前状态（`running` / `paused` / `shutting_down`）。

Proxy 端点：
- `POST /v1/rollout/pause`（[server.py:2139](../dressage/proxy/server.py#L2139)）
- `POST /v1/rollout/resume`（[server.py:2151](../dressage/proxy/server.py#L2151)）
- `GET /v1/rollout/pause_state`（[server.py:2163](../dressage/proxy/server.py#L2163)）

客户端封装：`ProxyClient.pause_rollout()`（[proxy_client.py:116](../dressage/proxy/proxy_client.py#L116)）、`ProxyClient.resume_rollout()`（[proxy_client.py:133](../dressage/proxy/proxy_client.py#L133)）。

## 4D.3 Partial Rollout：部分样本先行

当 `rollout_batch_size * n_samples_per_prompt` 远大于 `global_batch_size` 时，fully async rollout 会等所有样本完成才返回。Partial Rollout（[`dressage/rollout/partial_async_rollout.py`](../dressage/rollout/partial_async_rollout.py)）解决这个问题：

- 后台 worker 持续拉取 prompt 分组并生成，但 rollout 调用**只要凑够 `global_batch_size` 个完成的分组就返回**。
- 其余分组继续在后台跑，供下一个训练步使用。
- 权重更新期间必须暂停推理（§4D.2 的 pause/resume），避免新旧权重混用。

使用 `--dressage-partial-rollout` 参数启用。配合 `train_async_with_rollout_pause.py`（§4D.7）使用。

## 4D.4 Staleness 追踪

[`dressage/rollout/staleness.py`](../dressage/rollout/staleness.py) 提供 partial rollout 的过期过滤：

- `StalenessTracker`（L87）：跟踪每个 pending group 的"年龄"（从提交到现在经过的训练步数）。
- `StalenessGroupFilter`（L135）：过滤掉超过 staleness 阈值的分组——这些分组是用旧权重生成的，对当前训练步没有价值。
- `config_from_args()`：从 CLI 参数推导 staleness 配置。

过期分组的样本会被标记为 `partial_rollout_staleness_exceeded`，走 `mark_no_grad_failed` 路径，不贡献梯度。

## 4D.5 concat 轨迹构建模式 + TITO 分词器

Proxy 在 `finalize_session` 时支持两种轨迹构建模式：

- **默认模式**：每个 step 独立成段（segment）。
- **concat 模式**（`trajectory_build_mode = "concat"`）：把所有 step 的 token 拼成一段连续序列。适用于多 turn 对话需要整体训练的场景。

TITO（Token-In-Token-Out）分词器：某些模型（如 Qwen3.5）需要特殊的 mask-only chat template 来正确区分 prompt 和 response token。`PromptAssistantMaskBuilder`（[`last_step/prompt_assistant_mask.py:38`](../dressage/proxy/last_step/prompt_assistant_mask.py#L35)）会加载模型对应的 mask template（如 `qwen3_5_mask_only_chat_template.jinja`），用独立的 chat template 来计算哪些 token 属于 assistant。

配置通过 `trajectory_build_defaults()` 自动推导，默认 `DEFAULT_TRAJECTORY_BUILD_MODEL = "qwen3_5"`。

## 4D.6 reasoning_parser.py

[`dressage/proxy/reasoning_parser.py`](../dressage/proxy/reasoning_parser.py)（182 行）：`ProxyReasoningParser` 类解析模型输出中的 reasoning 块（如`）。支持的模型通过 `ModelReasoningParserRegistry` 注册。proxy 在 `create_app` 时创建 `ProxyReasoningParser` 实例，在 `chat_completions` handler 中调用它把 reasoning content 从 final content 中分离。

## 4D.7 三种 Rollout 模式

Dressage 提供三种 rollout 入口（slime 的 `--custom-generate-function-path` 指向其中之一）：

| 模式 | 文件 | 特点 |
|---|---|---|
| **Fully Async** | [`fully_async_rollout.py`](../dressage/rollout/fully_async_rollout.py) | 后台 worker 持续拉取并生成，rollout 调用只 drain 完成的分组。适合大规模并发。 |
| **Partial Async** | [`partial_async_rollout.py`](../dressage/rollout/partial_async_rollout.py) | 类似 fully async，但 rollout 调用只要够 `global_batch_size` 就返回。配合 staleness 过滤。 |
| **Sync** | [`sync_rollout.py`](../dressage/rollout/sync_rollout.py) | 同步等待所有样本完成。适合调试。 |

白盒 agent 最终都落到 slime 自带的 `slime.rollout.sglang_rollout.generate_and_rm_group`；黑盒 agent 走 `blackbox_dispatch.generate`。

## 4D.8 train_async_with_rollout_pause.py

[`dressage/training/train_async_with_rollout_pause.py`](../dressage/training/train_async_with_rollout_pause.py)（191 行）：异步训练脚本，在权重更新前后自动调 proxy 的 `pause_rollout()` / `resume_rollout()`。

核心流程：
1. Rollout 阶段：proxy 正常处理推理请求。
2. 权重更新前：调 `proxy_client.pause_rollout()` → SGLang 释放显存。
3. 权重更新：trainer 加载新权重到 SGLang。
4. 权重更新后：调 `proxy_client.resume_rollout()` → SGLang 恢复推理。

这样确保权重更新期间不会有推理请求用旧权重生成 token，避免新旧权重混用导致训练不稳定。

---

# Part 5 — 不那么显然的设计细节

## 5.1 多 turn loss_mask 构造

`PromptAssistantMaskBuilder.build_segment_alignment`（[`last_step/prompt_assistant_mask.py:245`](../dressage/proxy/last_step/prompt_assistant_mask.py#L245)）负责给一段的 token 生成 dense response mask。它**不靠手动追踪 turn 边界**，而是用一份"mask-only chat template"把这段的规范化 messages 重放一遍，让 tokenizer 直接标出哪些 token 属于 assistant。

**为什么用 mask-only chat template 而不是手动切 turn 边界？** 因为 chat template 拼接时插入的特殊 token 很难手动对齐；让 tokenizer 用同一套模板算 assistant mask，天然保证 mask 与真实 token 序列逐位对齐。

## 5.2 记录时的权威 token 序列，不要事后重新 tokenize

proxy 在 `record_step` 时就把每一 step 的**权威 token 序列**存进了 `StepRecord`：`prompt_token_ids`（这一 step 的输入）、`response_token_ids`（模型生成）、`all_token_ids`（输入+输出全序列）。这些都是当时用 `tokenizer.apply_chat_template(完整对话)` 真正喂给模型的 token。

**为什么不在 finalize/构段时重新 tokenize 那些 tool/user 消息再拿长度？**

因为 chat template 把消息拼到一起时会插入特殊 token（`<|im_start|>` 等），单独 tokenize 一条 tool 消息和"把这条 tool 消息嵌入对话再 tokenize"得到的 token 数**不一样**。要保证段的 token 序列和"真正传给模型的 prompt"逐位一致，唯一可靠的来源就是 proxy 在 `record_step` 当下记下的 `all_token_ids`/`prompt_token_ids`。

这是为什么 `record_step` 必须保留这些权威 token 序列，也是为什么 proxy 这一层不可省（只有 proxy 同时看到所有 turn 调用，才能记下这些权威数字）。mask 对齐（§5.1）也依赖 `prompt_token_ids` 与 mask template 结果的逐位比对。

## 5.3 Paddock 实现的并发安全

Paddock 接口是 `async def`（§0.0'D），调用时直接 `await`。但 Paddock 实现内部如果用了阻塞 API（如 `subprocess.run`），应自行用 `asyncio.to_thread` 包装，避免阻塞事件循环。同时，几十个 traj_id 会并发调同一 Paddock 实例的方法，共享状态需用 lock 保护。

## 5.4 GRPO 组归一化 + parent 段广播

`reward_post_process` 同时支持两种轨迹模式：**append-only**（1 trajectory → 1 segment，`parent_traj_id` 不存在）和 **rewrite-aware**（1 trajectory → N segments，共享 `parent_traj_id`）。关键设计：每个 parent 只有代表段进入归一化输入（避免一条 trajectory 被算 N 次），归一化后把 advantage 广播给所有兄弟段。详见 §4.17 的代码解析。

## 5.5 黑盒 agent 怎么把 traj_id 透传

黑盒 agent 是别人的代码，Dressage 不能改它。怎么让它把 `traj_id` 带在所有 LLM 调用上？答案是在 `Paddock.call_agent` 启动 agent 时，把 `traj_id` 注入到它的环境变量里（设计原理见 Part 2 决策 3）：

```python
class MyPaddock(Paddock):
    async def call_agent(self, state, *, session_id, messages, metadata=None, turn_id=None):
        env = os.environ.copy()
        env["LLM_BASE_URL"] = "http://localhost:8800/v1"
        env["X_TRAJ_ID"] = session_id    # agent 把它放到每次请求的 header

        proc = await asyncio.create_subprocess_exec(
            *["python", "my_agent.py", "--prompt", json.dumps(messages)],
            env=env, capture_output=True)
        stdout, _ = await proc.communicate()
        return {"content": stdout.decode()}, {}
```

agent 自己的代码里：

```python
# my_agent.py
client = OpenAI(base_url=os.environ["LLM_BASE_URL"])
response = client.chat.completions.create(
    model="default",
    messages=[...],
    extra_headers={"x-traj-id": os.environ["X_TRAJ_ID"]},
)
```

Dressage proxy 收到带 `x-traj-id` header 的请求时，会自动把它当成 `traj_id` 处理（见 `server.py:366`）：

```python
traj_id = request.traj_id
if not traj_id:
    traj_id = raw_request.headers.get("x-traj-id")
if not traj_id:
    traj_id = request.extra_body.get("traj_id")
```

三种传递方式（body / header / extra_body）都支持，挑你的 agent 能用的。

## 5.6 缺 logprob 时 `remove_sample = True`

如果黑盒 agent 用了**别的模型**（不是被训练的那个），SGLang 返回的 logprob 没意义（因为不是这个模型采样的）。

`fully_async_rollout.py` 检查：

```python
if trajectory.segments and trajectory.segments[0].tokens == []:
    for s in result_samples:
        s.remove_sample = True
```

`rollout/convert_samples.py` 处理：

```python
if getattr(sample, "remove_sample", False):
    sample.loss_mask = [0] * sample.response_length
```

**效果**：loss_mask 全 0 → 这条样本不贡献策略梯度。但它的 reward 仍然参与 GRPO 组归一化（影响别的样本的 advantage）。

## 5.7 reward 归一化放在哪：`reward_post_process` 的位置

reward 后处理（GRPO 组归一化 + 多段广播）集中在 [`dressage/training/reward_post_process.py`](../dressage/training/reward_post_process.py) 的 `reward_post_process(args, samples) -> (raw_rewards, rewards)`。

关键点：**这段逻辑被 `convert_samples_to_train_data` 在第一步直接调用**（[`convert_samples.py:87`](../dressage/rollout/convert_samples.py#L87)）：

```python
def convert_samples_to_train_data(args, samples):
    from dressage.training.reward_post_process import reward_post_process
    raw_rewards, rewards = reward_post_process(args, samples)   # ← 第一步
    ...
```

这样即使 slime 用 `--custom-convert-samples-to-train-data-path` 替换掉自己的 `_convert_samples_to_train_data`（连带绕过它内部的 reward 后处理），归一化也一定会跑——因为它就写在我们这个替换函数的开头。

> **注意**：不要再单独注册 `--custom-reward-post-process-path`——`convert_samples_to_train_data` 内部已经调用了 `reward_post_process`，重复注册会导致归一化被处理两次。

---

# Part 6 — 上手做事

## 6.1 跑测试

```bash
# 安装（包括测试依赖）
pip install -e ".[test]"

# 全套测试
pytest

# 单个文件
pytest tests/test_proxy.py

# 单个测试
pytest tests/test_proxy.py::TestSessionManager::test_finalize_multi_turn -v
```

测试**不依赖 slime**——`fully_async_rollout.py` 和 `data_source.py` 都有"slime 不可用时退回独立类型"的分支（见 `except ImportError` 里定义的 standalone `Sample` / `RolloutDataSourceWithBuffer`）。

测试**不依赖真的 SGLang / 真沙箱**——用 mock tokenizer、mock proxy client、mock sandbox provider（见 `tests/` 下各测试）。

## 6.2 起 proxy 走一遍假数据

```bash
# 起 proxy（指向一个真的 SGLang Router）
dressage-proxy \
  --sglang-router-url http://localhost:8000 \
  --tokenizer-path /path/to/tokenizer \
  --port 8800

# 在另一个终端测试 health
curl http://localhost:8800/health
# → {"status":"ok"}

# 发一个假 chat 请求（如果有 SGLang 真服务）；session 通过 header 或 body.session_id 传
curl -X POST http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: test_sess_1" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'

# Finalize（body 用 session_id）
curl -X POST http://localhost:8800/session/finalize \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_sess_1"}'

# Read trajectory（body 用 trajectory_id 或 session_id；drain=true 读完即删）
curl -X POST http://localhost:8800/trajectory/read \
  -H "Content-Type: application/json" \
  -d '{"trajectory_id": "test_sess_1", "drain": true}'
```

## 6.3 加一个新的 reward 函数

在你自己的包里（不必在 Dressage 仓库内）：

```python
# my_pkg/my_rewards.py
from dressage.reward import register_reward

@register_reward("contains_keyword")
def contains_keyword(sample, *, args=None, **_) -> float:
    keyword = (args or {}).get("keyword", "") or (sample.metadata or {}).get("keyword", "")
    if not keyword:
        return 0.0
    response = getattr(sample, "response", "") or getattr(sample, "agent_response", "")
    return 1.0 if keyword in response else 0.0
```

JSONL 数据里指定：

```json
{"prompt": "...", "keyword": "tensor", "reward_fn": "contains_keyword"}
```

启动训练时让 Dressage 加载你的模块：

```bash
export DRESSAGE_REWARD_MODULES=my_pkg.my_rewards
bash examples/scripts/run_example_qwen3.5_4b_async_local.sh
```

启动时 `dressage.reward.custom_rm`（slime `--custom-rm-path` 的入口）会调 `load_reward_modules()` 读 `DRESSAGE_REWARD_MODULES` 并 import 你的模块，触发 `@register_reward` 装饰器把函数登记到注册表。

## 6.4 实现一个真实 Paddock

```python
# my_pkg/docker_paddock.py
import asyncio
import docker
from dressage.paddock.interface import WhiteboxPaddock

class DockerWhiteboxPaddock(WhiteboxPaddock):
    """白盒 Paddock：用 Docker 容器做沙箱，支持 tool_call。"""

    def __init__(self):
        self._client = docker.from_env()
        self._containers = {}

    async def init(self, traj_id, env_type, env_args, **kwargs):
        # Docker SDK 是同步的，用 asyncio.to_thread 避免阻塞事件循环
        c = await asyncio.to_thread(
            self._client.containers.run,
            image=env_args.get("image", "ubuntu:latest"),
            command="sleep infinity",
            detach=True,
            name=traj_id,
        )
        self._containers[traj_id] = c

    async def terminate(self, traj_id, env_args=None, **kwargs):
        c = self._containers.pop(traj_id, None)
        if c:
            await asyncio.to_thread(c.kill)
            await asyncio.to_thread(c.remove)

    async def tool_call(self, traj_id, tool_id, tool_args):
        # WhiteboxPaddock.tool_call 返回 (tool_response, metadata) 元组
        c = self._containers.get(traj_id)
        if not c:
            return "[error] container not found", {}
        if tool_id == "bash":
            # exec_run 是同步的，放线程池
            result = await asyncio.to_thread(c.exec_run, tool_args["cmd"])
            return result.output.decode()[:8000], {}   # 截断防爆
        return f"[unknown tool: {tool_id}]", {}
```

如果要接黑盒 agent，继承 `BlackboxPaddock` 并实现 `register_agent` / `call_agent` / `execute_cmd`：

```python
# my_pkg/docker_blackbox_paddock.py
import asyncio
import httpx
from dressage.paddock.interface import BlackboxPaddock

class DockerBlackboxPaddock(BlackboxPaddock):
    """黑盒 Paddock：Docker 容器里跑 agent server。"""

    async def init(self, traj_id, env_type, env_args, **kwargs):
        # 起容器，返回 SandboxState
        ...

    async def register_agent(self, state, *, instance_id, session_id,
                              router_url, blackbox_type, backend_options):
        # 把 proxy URL 注入容器（通过环境变量）
        ...

    async def call_agent(self, state, *, session_id, messages, metadata):
        # POST /v1/sessions/<sid>/messages 给容器里的 agent
        ...

    async def execute_cmd(self, state, *, session_id, cmd, timeout=None):
        # 在容器里跑 shell 命令
        ...

    async def terminate(self, traj_id, env_args=None, **kwargs):
        ...
```

启动时（`DRESSAGE_PADDOCK_CLASS` 是加载自定义 paddock 类的高级覆盖；类路径指向你上面写的类）：

```bash
export DRESSAGE_PADDOCK_CLASS=my_pkg.docker_paddock.DockerWhiteboxPaddock
bash examples/scripts/run_example_qwen3.5_4b_async_local.sh
```

## 6.5 接入一个黑盒 agent

核心是让你的 `Paddock.call_agent` 把 `traj_id` 注入到 agent 进程，并让 agent 把这个 id 带在每次对 proxy 的请求上（三种透传方式见 §5.5）。

端到端的代码走查（`dressage/rollout/generate/blackbox_dispatch.py` 怎样把 sample 喂给一个 Paddock 实例并取回 trajectory）见 **§4B**；如果你还在纠结沙箱后端选 `local_bwrap`、`harness` 还是 `e2b`，看 **§4C**（含决策清单）。

## 6.6 调试技巧

**看某条 trajectory 跑了什么**：临时**不**调 `proxy_client.read_trajectory`，让 trajectory 留在 proxy 的 `TrajectoryStore` 里。然后写个脚本调 `/trajectory/read` 拿出来 dump 成 JSON。

**看 proxy 收到了哪些请求**：开 DEBUG 日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**看模型生成了什么但没解析成 tool_call**：在 `dressage/proxy/tool_call_parser.py` 的 `ProxyToolCallParser.parse` 里加日志。常见原因：模型输出的 JSON 格式不规范（`<tool_call>` 包了多个 JSON 对象、字段名漏了引号）。

**看 GRPO 归一化的具体数值**：在 `reward_post_process.py` 加 log，打印每个组的 `raw_rewards` 和 `normalized`。

**怀疑 loss_mask 错了**：检查 `PromptAssistantMaskBuilder.build` 的 `turn_boundaries` 输入是不是符合预期。可以临时让 `finalize` 把 `(tokens, boundaries, mask)` 一起 dump 出来人肉对比。

---

# Part 7 — 扩展阅读

## 仓库内文件

- **`README.md`** — 仓库简介。
- **`agentic-rl-training-zh.md`** — Agentic RL 训练完全指南（原理 + 架构 + 实践，和本文互补）。
- **`proxy-architecture.md`** / **`proxy-create-app-components.md`** / **`proxy-tool-call-parser.md`** — proxy 的架构、`create_app` 组件、tool_call 解析器详解。
- **`dressage错误码总览.md`** / **`dressage框架错误快速排查.md`** — 错误码契约与快速排障。
- **`docs/paddock-sandbox-provider-implementation.md`** — paddock（mode）× sandbox（provider）两层拆分的设计与运行矩阵（§3 / §4C 反复引用）。
- **`docs/local-blackbox-sandbox.md`** — 本地 bwrap 沙箱集群（Ray + bubblewrap）的部署与运维（§4C.3 / §4C.5 反复引用）。
- **`docs/paddock-multi-segment-changes.md`** — 多段 trajectory 处理的设计。
- **`docs/blackbox-server*.md`** / **`docs/blackbox-context-window-overflow-function.md`** — blackbox server 侧的 adapter、dispatch、context 溢出等设计文档。
- **`docs/whitebox-agent-quickstart.md`** / **`docs/quickstart.md`** / **`docs/recipes.md`** — 上手与配方。

## 推荐代码阅读顺序

按下面的顺序看，30 分钟内能从 "接口" 到 "两条端到端链路" 过完：

1. `dressage/paddock/interface.py` — 三层 ABC 异步接口（`Paddock` / `BlackboxPaddock` / `WhiteboxPaddock`），104 行。
2. `dressage/sandbox/{provider.py,types.py,factory.py}` — 沙箱 provider 协议、`SandboxSpec`/`SandboxLease` 数据类型、provider 选择。
3. `dressage/proxy/trajectory_store.py` + `dressage/proxy/session_manager.py` — 数据结构（`StepRecord` / `TrajectorySegment`）和多 turn 记录 + 构段。
4. `dressage/reward/registry.py` + `dressage/reward/helpers.py` — 注册表 + 内置 reward 函数。
5. `dressage/rollout/data_source.py` — JSONL → Sample。
6. `dressage/proxy/server.py` — FastAPI proxy 端点（`/v1/chat/completions`、`/trajectory/*`、`/session/*`、`/v1/rollout/*`）。
7. **rollout 入口**：`dressage/rollout/sync_rollout.py` → `dressage/rollout/fully_async_rollout.py` → `dressage/rollout/partial_async_rollout.py` —— 三种 rollout 入口；都复用 slime 自带的 `slime.rollout.sglang_rollout.generate_and_rm_group`。对应文档 §4。
8. **黑盒链路**：`dressage/rollout/generate/blackbox_dispatch.py` + `dressage/paddock/blackbox/{paddock,client,execute_hooks,failures}.py` —— 对照 §4B 阅读。
9. **Sandbox provider**：
   - `dressage/sandbox/local/bwrap/{provider,manager,supervisor,runner,slot}.py` —— 本地 Ray 三层架构（§4C.3）；
   - `dressage/sandbox/remote/harness/provider.py`、`dressage/sandbox/remote/e2b/provider.py` —— 远程 provider（§4C.1）。
10. `dressage/rollout/convert_samples.py` + `dressage/training/reward_post_process.py` — 训练侧。
11. `dressage/sandbox/scripts/{start_local_bwrap,local_bwrap_status,stop_local_bwrap}.py` — 本地集群运维入口（§4C.5）。

## 外部资料

- **[slime README](https://github.com/THUDM/slime)** — 底层框架
- **[OpenAI Spinning Up "Policy Gradient"](https://spinningup.openai.com/)** — RL 入门
- **[PPO 原论文 (Schulman 2017)](https://arxiv.org/abs/1707.06347)** — clip surrogate
- **[GRPO 原论文 (DeepSeek-Math 2024)](https://arxiv.org/abs/2402.03300)** — 省 value model 的动机
- **[SGLang 文档](https://sgl-project.github.io/)** — 推理后端
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** — 大模型分布式训练
- **[Ray 文档](https://docs.ray.io/)** — 分布式 Python 编排

---

# 读完之后

读完这份文档之后，建议的下一步：

1. **建立直觉**：把 `dressage/paddock/interface.py` 和 `dressage/proxy/trajectory_store.py` 完整看一遍（合计 < 100 行）——建立对核心抽象的直觉。
2. **跑测试**：`pytest` 看测试覆盖了哪些行为。挑两个测试加断点跟一遍。
3. **跟读 §4**：对照 §4 的每一节，把 `dressage/rollout/fully_async_rollout.py / partial_async_rollout.py / sync_rollout.py` 完整看一遍。
4. **加 reward**：试着加一个最简单的 reward 函数（§6.3）。
5. **接 Paddock**：想训真实任务时，参考 §6.4 实现一个真的 Paddock。
6. **遇到不懂的术语**：回 Part 0 的 §0.15 速查表，或回相应小节细看。
