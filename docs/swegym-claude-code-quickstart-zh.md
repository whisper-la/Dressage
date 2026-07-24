# SWE-Gym + Claude Code 快速开始

[English](swegym-claude-code-quickstart-en.md) ·
[完整实验说明](blackbox-swegym-claude-code-experiment-zh.md)

本文说明如何使用公开版 Dressage 准备 SWE-Gym 数据，在 E2B task template
中运行 Claude Code，对每个 patch 使用全新沙箱评测，并启动 Qwen3.5-4B
同步 GRPO recipe。

## 1. 安装仓库和数据依赖

```bash
git submodule update --init --recursive

python3 -m pip install -e .
python3 -m pip install --no-build-isolation blackbox_server/
python3 -m pip install pyarrow huggingface_hub
python3 -m pip install \
  'swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7'
```

固定版本的 SWE-Gym package 只在数据准备阶段使用。转换器会把官方的
repository-specific 测试脚本和 log parser 写入生成数据，因此 task sandbox
不需要安装 `swegym` Python package。

## 2. 准备 E2B template

每条 SWE-Gym 数据需要一个全新的 task image，并满足：

- 仓库及原始测试环境位于 `/testbed`；
- 已安装 Claude Code；
- 已安装当前公开仓库的 Blackbox Server；
- Blackbox Server 监听 `31000` 端口。

先从公开 Blackbox Server 构建 wheel，并打包独立运行时：

```bash
python3 -m pip install build uv
python3 -m build --wheel --outdir dist blackbox_server/

export CLAUDE_CODE_BBS_VERSION=1.1.0
export CLAUDE_CODE_BBS_WHEEL_URL="$PWD/dist/dressage_blackbox_server-1.1.0-py3-none-any.whl"
export CLAUDE_CODE_ARTIFACT_DIR="$PWD/data/claude-code-artifacts"

bash examples/scripts/prepare_claude_code_sandbox_artifacts.sh
```

为每个不同的 SWE-Gym task image 构建一个 E2B template，并准备从原始
Docker image 到 E2B template name 的 JSON 映射：

```json
{
  "xingyaoww/sweb.eval.x86_64.example:latest": "e2b-swegym-example"
}
```

如何枚举 image 和构建 template，见
[完整实验说明](blackbox-swegym-claude-code-experiment-zh.md)。

## 3. 转换 SWE-Gym 数据

下载并转换 293 条训练数据：

```bash
python3 examples/data/swegym/prepare_swegym_data.py \
  data/swegym-train-claude-code-e2b.jsonl \
  --download \
  --download-dir data/swegym-source \
  --split train \
  --provider e2b \
  --sandbox-image-map data/e2b-template-map.json \
  --blackbox-type claude_code \
  --max-turns 80 \
  --permission-mode acceptEdits
```

单条转换 smoke 可追加 `--limit 1`；转换器会先校验完整 split，再应用
limit。最终 JSONL 不含 gold patch，并包含：

- 强制执行的 before-agent Git sanitizer；
- 固定版本的官方 SWE-Gym evaluator 和 log parser；
- task-specific `FAIL_TO_PASS` 与 `PASS_TO_PASS`；
- Claude Code backend options，包括 `working_directory=/testbed`；
- 每条任务对应的 E2B template。

## 4. Smoke-test template

占用训练 GPU 前，先确认 template 能恢复 Blackbox Server 并暴露 31000：

```python
import asyncio
from e2b import AsyncSandbox


async def main():
    sandbox = await AsyncSandbox.create(template="e2b-swegym-example")
    try:
        print(sandbox.get_host(31000))
        result = await sandbox.commands.run(
            "curl -sf http://127.0.0.1:31000/health"
        )
        print(result.stdout)
    finally:
        sandbox.kill()


asyncio.run(main())
```

还应至少完整验证一条真实任务：Claude Code 能在 `/testbed` 生成 patch，
全新 evaluation sandbox 能输出 `DRESSAGE_SWEGYM_REWARD_JSON=` marker。

## 5. 启动同步 GRPO

在同一模型目录准备 Qwen3.5-4B Hugging Face 和 Megatron distributed
checkpoint，然后执行：

```bash
export MODEL_ROOT=/path/to/models
export PROMPT_DATA="$PWD/data/swegym-train-claude-code-e2b.jsonl"
export DRESSAGE_SANDBOX_PROVIDER=e2b
export DRESSAGE_E2B_API_KEY=e2b_...
export DRESSAGE_E2B_BLACKBOX_PORT=31000
export DRESSAGE_PROXY_URL=https://proxy.example.com

bash examples/scripts/run_swegym_claude_code_grpo_sync.sh
```

`DRESSAGE_PROXY_URL` 必须是 E2B 沙箱能够访问的 HTTP(S) 地址。

launcher 会显式选择
`dressage.rollout.generate.blackbox_dispatch_swegym.generate`。通用 blackbox
dispatch 不包含 SWE-Gym 判断；fresh evaluation 和 trajectory integrity
检查都由专用 dispatch 负责。

参考默认值保留已评审实验配置：TP2/CP4、8 prompts × 16 samples、global
batch size 128、500 次 rollout update、normalized GRPO advantage、vanilla
token-level TIS，以及系数为 `0.001` 的 low-variance KL loss。基础设施和
拓扑可通过环境变量覆盖。
