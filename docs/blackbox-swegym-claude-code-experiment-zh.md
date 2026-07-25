# 基于 Claude Code 的 Blackbox SWE-Gym 实验

本实验验证了 **Dressage 能够通过 blackbox RL 有效训练 SWE 类 agentic 任务，并显著提升模型的软件工程能力**。我们以 Qwen3.5-4B 为基础模型，在 SWE-Gym 数据集上使用 Claude Code 作为 coding agent 进行训练。实验结果表明，这套流程能够稳定支持由模型服务、代理服务和隔离沙箱组成的多服务执行环境，并在不同代码仓库和任务环境中取得明确收益。

## 训练曲线

![SWE-Gym training metrics](../assets/swegym-training-metrics.png)

上图展示了训练前 80 个 steps 的三个关键指标，浅色折线为原始记录，深色曲线为 13-step 居中滑动平均。`raw_reward_trajectory_mean` 的平滑值从训练初期约 0.27 上升到后期 **0.6 以上**，并基本收敛，表明模型解决 SWE-Gym 任务的成功率总体提高，且训练稳定性良好；`train_rollout_logprob_diff` 始终在约 0.009–0.011 的窄幅区间内波动，没有随训练持续扩大；`grad_norm` 虽然存在少量短时尖峰，但整体保持在可控范围内，末段平滑值回落到约 0.3，未出现持续性的梯度爆炸。

## SWE-bench Verified：准确率达到 37.8%

在难度显著更高的 **SWE-bench Verified** 上，经过 SWE-Gym 训练的模型同样取得了明确提升：

| 模型与 Agent | 准确率 |
| --- | ---: |
| Qwen3.5-4B + Claude Code baseline | 32.6% |
| **Dressage 训练后的 Qwen3.5-4B + Claude Code** | **37.8%** |
| **绝对提升** | **+5.2 个百分点** |

评测时，我们将上下文预算提高到 256K tokens，并将最大交互步数提高到 160，为 agent 留出更充分的问题分析和代码修改空间。**SWE-bench Verified 上 5.2 个百分点的绝对提升**表明，SWE-Gym 训练带来的软件工程能力增益能够迁移到训练 benchmark 之外。

## 下载并转换 SWE-Gym 数据

本实验使用 Hugging Face 上的 `NovaSky-AI/SkyRL-v0-293-data` 数据集。当前转换脚本支持 `train` 和 `validation` 两个 split，分别包含 293 和 23 个任务。每条任务给出待修复仓库、问题描述、基线提交、测试补丁以及官方 `FAIL_TO_PASS` / `PASS_TO_PASS` 用例。模型只会看到问题描述和任务仓库，不会得到 gold patch。

数据转换依赖 SWE-Gym 官方 harness。为了让生成的评测脚本和参考实验保持一致，本实验将 `SWE-Bench-Package` 固定在以下提交：

```text
16dd480cce9b27bf111a362d280881c6def5d2a7
```

先安装数据读取依赖和固定版本的 harness：

```bash
python3 -m pip install pyarrow huggingface_hub
python3 -m pip install \
  'swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7'
```

数据只需要执行一次转换。如果使用 E2B template，请先按下一节下载源 Parquet、枚举并构建 task template，并生成 `data/e2b-template-map.json`。之后只需一次命令即可同时完成 SWE-Gym 格式转换、官方评测配置生成、E2B image 映射和 Claude Code backend 配置：

```bash
python3 examples/data/swegym/prepare_swegym_data.py \
  data/swegym-train-claude-code-e2b.jsonl \
  --input data/swegym-source/train.parquet \
  --split train \
  --provider e2b \
  --sandbox-image-map data/e2b-template-map.json \
  --blackbox-type claude_code \
  --max-turns 80 \
  --permission-mode acceptEdits
```

转换器会先校验 split 的完整行数，再逐条调用固定版本 harness 的 `make_test_spec()` 生成 repository-specific eval script；调试时可以追加 `--limit 1`，验证集则使用 `--split validation`。

对于能够直接启动 Docker task image 的 custom provider，也只需调用同一个脚本。此时不需要 image map，原始 task image 会直接写入 `metadata.sandbox_image`；可通过成对的 `--binary-url` 和 `--runtime-url` 注入每次创建沙箱时执行的 Claude Code/BBS bootstrap：

```bash
python3 examples/data/swegym/prepare_swegym_data.py \
  data/swegym-train-claude-code-custom.jsonl \
  --download \
  --download-dir data/swegym-source \
  --split train \
  --provider custom \
  --blackbox-type claude_code \
  --binary-url https://artifact-host.example/claude-2.1.207-linux-x64 \
  --runtime-url https://artifact-host.example/claude-code-runtime-python-3.10.20-bbs-<version>.tar.gz \
  --max-turns 80 \
  --permission-mode acceptEdits
```

生成的最终 Dressage JSONL 主要字段如下：

| 字段 | 作用 |
| --- | --- |
| `prompt` | 由 `problem_statement` 构造的用户消息，并追加完整性约束：不得修改测试、pytest 配置或评测 harness，也不得从远程仓库或镜像获取现成修复。 |
| `metadata.agent_mode` | 固定为 `blackbox`。 |
| `metadata.blackbox_type` | 固定为 `claude_code`，使 Blackbox Server 注册 Claude Code backend。 |
| `metadata.instance_id` | SWE-Gym 实例 ID，用于镜像选择、日志记录和结果追踪。 |
| `metadata.repo` / `repo_name` | 待修复仓库及其短名称。 |
| `metadata.base_commit` | 任务声明的基线提交；agent 运行前清理仓库和 fresh-sandbox 评测都会以它为准。 |
| `metadata.workdir` | 仓库在沙箱中的工作目录，默认 `/testbed`。 |
| `metadata.backend_options.working_directory` | 将 Claude Code 进程的 cwd 指向任务仓库（默认 `/testbed`）；若 public BBS 未设置此项，Claude Code 会使用隔离的 runtime workspace。 |
| `metadata.dataset` / `dataset_split` | 数据集名称和 split。 |
| `metadata.sandbox_image` | custom 路径使用 SWE-Bench/SWE-Gym task image；E2B 路径通过 `--sandbox-image-map` 写入对应 template name。 |
| `metadata.sandbox_cmd` | E2B 预构建路径不设置；custom 路径可写入下载 Claude Code、解压 portable runtime 并启动 BBS 的命令。 |
| `metadata.FAIL_TO_PASS` | 修复前失败、修复后必须通过的测试。 |
| `metadata.PASS_TO_PASS` | 修复前通过、修复后不得回归的测试。 |
| `metadata.reward_fn` | 固定为 `swegym_harness_marker`，从官方评测输出中读取二值 resolved reward。 |
| `metadata.blackbox_execute_cmds.before_agent` | 必须成功的 Git 清理命令。 |
| `metadata.blackbox_execute_cmds.after_agent` | 在 fresh evaluation sandbox 中执行的官方评测命令。 |

before-agent Git sanitizer 会先确认 `/testbed` 是 Git worktree 且 `base_commit` 存在，然后执行 hard reset、清理未跟踪文件并切到 detached HEAD。它还会删除所有 remote、命名 ref 和 reflog，再 prune 不可达对象，避免镜像中的未来提交、分支、tag 或 stash 泄漏答案。这里使用 `git clean -ffd` 而不是 `git clean -ffdx`，因此镜像中被忽略的依赖和构建缓存可以保留。

after-agent 命令不依赖沙箱预装 `swegym` 包。转换器会把官方 eval script、固定版本 `log_parsers.py` 和期望测试列表压缩后直接写入 JSONL；运行时在沙箱内解压并执行。为了兼容部分仍使用 Python 3.7/3.8 的旧 task image，转换器会延迟类型注解求值，并将 `TestStatus` 定义内联到 parser 中。评测结束后输出统一的 `DRESSAGE_SWEGYM_REWARD_JSON=` marker，其中记录 resolved 状态、两类测试的通过数、评测返回码和失败原因。

正常情况下最终训练 JSONL 为 293 行：

```bash
wc -l data/swegym-train-claude-code-e2b.jsonl
```

## 准备 Sandbox

开源 recipe 支持两种沙箱路径：使用 E2B template，或者在自有基础设施中实现自定义 `SandboxProvider`。两种实现都必须按样本启动 `metadata.sandbox_image` 指定的 task image，并在其中提供 Claude Code 和 Blackbox Server。

Claude Code 和 Blackbox Server 通常需要比 task image 更新的 Node.js 或 Python，但不能因此修改 task image 原有的 Python 环境。Blackbox runtime 应使用独立目录和解释器启动；任务中的 `python`、`pip`、pytest 及项目依赖仍使用 Docker image 自带的版本，不要把 Blackbox Server 的依赖写入全局 `PYTHONPATH`。

### E2B Template

Dressage 已内置 `E2BSandboxProvider`。它把 `metadata.sandbox_image` 作为 template name 传给 `AsyncSandbox.create()`，并通过 E2B 暴露的 31000 端口访问 Blackbox Server。因此 E2B template 必须以原始 task image 为基础，预先安装 Claude Code 和 Blackbox Server，并在快照中启动 Blackbox Server。

SWE-Gym 任务使用多个 task image，需要为每个不同的 Docker image 构建一个对应的 E2B template。E2B 当前只支持 Debian 及其衍生发行版；不兼容的 task image 应改用 custom provider，这点需要注意。

E2B template 的准备分为三步：

1. 收集数据集中的全部 task image，并为每个唯一 image 构建一个 template；
2. 记录原始 Docker image 到 E2B template name 的 JSON 映射；
3. 在数据转换时通过 `--sandbox-image-map` 写入 template name。

下载原始 Parquet 并直接收集全部唯一 task image，不需要先生成一份中间 Dressage JSONL：

```bash
python3 examples/data/swegym/prepare_swegym_e2b.py list-images \
  --download \
  --download-dir data/swegym-source \
  --split train \
  --output data/swegym-images.txt
```

每个 template 都要安装 Claude Code 和 Blackbox Server。先从当前公开仓库的
实现构建 BBS wheel，再运行仓库脚本生成 Claude Code binary 和独立的 BBS
runtime：

```bash
python3 -m pip install build uv
python3 -m build --wheel --outdir dist blackbox_server/

export CLAUDE_CODE_BBS_VERSION=1.1.0
export CLAUDE_CODE_BBS_WHEEL_URL="$PWD/dist/dressage_blackbox_server-1.1.0-py3-none-any.whl"
export CLAUDE_CODE_ARTIFACT_DIR=/shared/path/claude-code-artifacts

bash dressage/recipes/swegym/prepare_claude_code_sandbox_artifacts.sh
```

脚本默认生成 Claude Code 2.1.207 的 Linux x86-64 binary，以及包含 portable Python 3.10.20、BBS 和相关依赖的 runtime 压缩包：

```text
/shared/path/claude-code-artifacts/
├── claude-2.1.207-linux-x64
└── claude-code-runtime-python-3.10.20-bbs-<version>.tar.gz
```

构建 template 时，仓库内的 builder 会将 Claude Code 安装到 `/usr/local/bin/claude`，将 BBS runtime 解压到 `/opt/cc-runtime`，并使用 portable Python 启动 Blackbox Server。先设置 `E2B_API_KEY=e2b_...`，从 `data/swegym-images.txt` 选择一个 image，为其分配唯一 template name，然后执行：

```bash
export E2B_API_KEY=e2b_...
export TASK_IMAGE=xingyaoww/sweb.eval.x86_64.example:latest
export TEMPLATE_NAME=e2b-swegym-example

python3 examples/data/swegym/prepare_swegym_e2b.py build
```

builder 可以直接使用公开 registry 中的 task image；私有 registry 需要按 E2B SDK 要求提供 registry credential。它会在 template build 末尾启动 BBS，等待 `/health` 返回成功后制作快照；以后从该 template 创建 sandbox 时，快照中的 BBS 已经处于运行状态。默认的 4 CPUs 和 8192 MiB 只是参考，可通过 `--cpu-count` 和 `--memory-mb` 按各仓库测试负载调整。E2B template 的定义、构建及 start/ready command 语义以 [E2B Template 文档](https://e2b.dev/docs/template/defining-template) 和 [Start & ready commands](https://e2b.dev/docs/template/start-ready-command) 为准。

每个 template 构建成功后，把映射记录到 `data/e2b-template-map.json`，格式为 `{docker_image: template_name}`。随后执行上一章唯一一次 `prepare_swegym_data.py` 调用；转换器会完成 image 映射、保留 `metadata.docker_image`，并直接写出最终 Claude Code 训练数据。E2B 路径不要传 artifact URL，template 中已经启动的 BBS 应是唯一服务进程。启动训练时选择 E2B：

```bash
export DRESSAGE_SANDBOX_PROVIDER=e2b
export DRESSAGE_E2B_API_KEY=e2b_...
export DRESSAGE_E2B_BLACKBOX_PORT=31000
export DRESSAGE_PROXY_URL=https://proxy.example.com
export PROMPT_DATA=data/swegym-train-claude-code-e2b.jsonl
```

`DRESSAGE_PROXY_URL` 必须是 E2B 能够访问的 HTTP(S) 地址。请通过集群
ingress 或安全隧道暴露 Dressage proxy，不要依赖 `hostname -I` 推导出的节点
私网地址。

在批量构建和训练前，先对一个 template 做 smoke test：

```bash
python3 examples/data/swegym/prepare_swegym_e2b.py smoke \
  --template-name e2b-swegym-example
```

### Custom `SandboxProvider`

如果已有能够按样本启动 Docker image 的自建容器平台或 Kubernetes 集群，可以实现 Dressage 的 `SandboxProvider`，不需要为每个 task image 构建 E2B template。

Custom 路径依赖 `metadata.sandbox_cmd`。上一章的 `prepare_swegym_data.py --provider custom` 会将原始 task image 写入 `metadata.sandbox_image`；同时传入 `--binary-url` 和 `--runtime-url` 时，转换器还会生成 `sandbox_cmd`，用于在新沙箱中安装 Claude Code、解压独立的 BBS runtime 并启动 Blackbox Server。Custom provider 的 `create()` 必须读取并执行该字段，不能忽略它。

`create()` 需要完成以下工作：

1. 从 `spec.env_args["sandbox_image"]` 读取 task image，并用该 image 启动一个全新的沙箱；
2. 保留 image 中的 `/testbed`、Python 环境和测试依赖；
3. 将 `spec.env_args["sandbox_cmd"]` 作为容器 bootstrap 执行，并确保其中的 artifact URL 可从沙箱内部访问；
4. `sandbox_cmd` 最终会以前台进程启动 Blackbox Server，因此不要等待命令自然退出，而应等待 `/health` 检查通过，再暴露 31000 端口；
5. 返回包含 `endpoints["blackbox"]` 的 `SandboxLease`，并在创建失败时清理已经分配的资源。

Provider 还需要实现 `terminate()` 和 `get_public_url()`；如需完整支持 `SandboxProvider` 的命令和文件能力，还应实现 `run_command()`、`read_file()` 和 `write_file()`。这些方法只负责沙箱生命周期和基础能力，不负责 Claude Code session、agent turn 或 `register_agent`，后者仍由 Blackbox Server 和 `BlackboxAgentPaddock` 处理。

当前公开 factory 没有注册名为 `custom` 的 provider。接入时通过 `DRESSAGE_PADDOCK_CLASS` 注入一个构造 `BlackboxAgentPaddock(provider=<your provider>)` 的自定义 paddock，即可复用 Dressage 的 agent 注册、调用、execute hook 和清理流程。

SWE-Gym 使用 fresh-sandbox 评测。模型执行结束后，Dressage 会从 source sandbox 提取相对 `base_commit` 的 patch，再创建一个相同 task image 的干净 evaluation sandbox，只应用模型 patch 并运行官方评测。因此 Custom provider 必须保证每次 `create()` 返回全新环境，并对 source sandbox 和 evaluation sandbox 执行相同的 `sandbox_cmd`。

正式训练前至少用一个样本确认：`/usr/local/bin/claude --version` 正常，Blackbox Server 的 31000 端口可访问，Claude Code 能在 `/testbed` 生成 patch，fresh evaluation sandbox 能输出 `DRESSAGE_SWEGYM_REWARD_JSON=` marker。

## SWE 类任务反作弊

Agent 可能利用任务环境中残留的信息绕过正常的软件修复过程。一个常见原因是 task image 没有完整清理 Git 历史、remote 或其他可用于定位现成答案的信息。为了让 reward 尽可能反映真实的代码修复能力，本实验采用了以下措施：

1. 在 before-agent command 中清理 Git 信息。Agent 启动前先将仓库重置到任务指定的 `base_commit`，切换到 detached HEAD，并删除 remote、额外 refs、reflog 及不可达对象，避免通过历史提交、分支、tag 或 stash 找到未来版本中的修复。
2. 检查从远程仓库获取答案的行为。Rollout 完成后，Dressage 会扫描完整的工具调用轨迹，识别通过 `git clone/fetch/pull`、`curl`、`wget`、GitHub CLI 或 Web 工具访问当前任务仓库的行为；如需同时覆盖相关 fork，可通过 `DRESSAGE_SWEGYM_BLOCKED_REPOS` 或 `DRESSAGE_SWEGYM_BLOCKED_REPOS_FILE` 补充仓库列表。命中后即使官方 harness 通过，也会将该 trajectory 的 reward 置为 0。
3. 保护测试与评测逻辑。Prompt 中明确禁止修改测试、pytest 配置和评测 harness；运行时同时扫描写文件工具和 shell 命令对 `tests`、`testing`、`r2e_tests`、`conftest.py`、`pytest.ini`、`tox.ini`、`setup.cfg`、`pyproject.toml` 等路径的写入，fresh-sandbox 评测前还会检查最终 patch。任何一层发现违规都会将 reward 置为 0。

## 实验设置

完整的实验参数见 `examples/scripts/run_dressage_swegym_qwen3.5_4b_claude_code_sync_4_node.sh`。本实验以 Qwen3.5-4B 为基础模型；考虑到 SWE-Gym 任务的整体复杂度在SWE任务中并不太高，将模型上下文长度设为 64K tokens，单条 rollout 最多保留 16K response tokens，并将 Claude Code 的最大交互轮数设为 80。训练采用同步的方式进行。

任务 reward 完全由 fresh sandbox 中的官方 SWE-Gym harness 决定：只有 `FAIL_TO_PASS` 和 `PASS_TO_PASS` 中的全部目标测试都通过时，trajectory 才获得 1 分，其他情况均为 0 分，不提供按测试通过比例计算的部分奖励。空 patch、patch 应用失败、评测执行失败或 reward marker 缺失同样记为 0 分；上一节提到的反作弊检查也只会将违规 trajectory 置为 0 分，而不会额外给予负分。

每个 prompt 默认采样 16 条 trajectory，并在同组内对二值 reward 做 GRPO 均值中心化，使成功样本相对于同题失败样本获得正 advantage。训练同时启用 Slime 默认的 vanilla token-level TIS，并将重要性权重裁剪到 `[0, 2]`；此外使用系数为 `0.001` 的 low-variance KL loss，以限制策略相对初始模型偏移过快。这两项属于训练校正与正则化，不改变任务的原始二值 reward。本实验没有额外设置响应长度、工具调用次数或交互轮数惩罚，也没有启用 custom MIS/RS。
