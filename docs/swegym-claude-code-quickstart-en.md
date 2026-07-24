# SWE-Gym + Claude Code Quickstart

[中文版](swegym-claude-code-quickstart-zh.md)

This quickstart prepares the public Dressage SWE-Gym recipe, runs Claude Code
inside E2B task templates, evaluates every patch in a fresh sandbox, and starts
the synchronous Qwen3.5-4B GRPO recipe.

## 1. Install the repository and data dependencies

```bash
git submodule update --init --recursive

python3 -m pip install -e .
python3 -m pip install --no-build-isolation blackbox_server/
python3 -m pip install pyarrow huggingface_hub
python3 -m pip install \
  'swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7'
```

The pinned SWE-Gym package is used only while preparing data. The generated
evaluation command embeds the official repository-specific test script and log
parser, so task sandboxes do not need the `swegym` Python package.

## 2. Prepare E2B templates

Each SWE-Gym row needs a fresh task image containing:

- the repository and its original test environment under `/testbed`;
- Claude Code;
- this repository's Blackbox Server;
- Blackbox Server listening on port `31000`.

Build a wheel from the public Blackbox Server and package an isolated runtime:

```bash
python3 -m pip install build uv
python3 -m build --wheel --outdir dist blackbox_server/

export CLAUDE_CODE_BBS_VERSION=1.1.0
export CLAUDE_CODE_BBS_WHEEL_URL="$PWD/dist/dressage_blackbox_server-1.1.0-py3-none-any.whl"
export CLAUDE_CODE_ARTIFACT_DIR="$PWD/data/claude-code-artifacts"

bash examples/scripts/prepare_claude_code_sandbox_artifacts.sh
```

Build one E2B template per distinct SWE-Gym task image and write a JSON object
mapping each Docker image to its E2B template name:

```json
{
  "xingyaoww/sweb.eval.x86_64.example:latest": "e2b-swegym-example"
}
```

See the repository's [sandbox documentation](sandbox.md) for the public E2B
template contract.

## 3. Convert SWE-Gym data

Download and convert the 293-row training split:

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

For a one-row conversion smoke test, add `--limit 1`. The converter verifies
the complete split before applying the limit. The final JSONL contains no gold
patch. It includes:

- a mandatory before-agent Git sanitizer;
- the fixed official SWE-Gym evaluator and log parser;
- task-specific `FAIL_TO_PASS` and `PASS_TO_PASS` tests;
- Claude Code backend options, including `working_directory=/testbed`;
- the E2B template mapped for each task.

## 4. Smoke-test a template

Before allocating training GPUs, verify that a template resumes Blackbox Server
and exposes port `31000`:

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

Also verify one real task end to end: Claude Code must produce a patch under
`/testbed`, and the fresh evaluation sandbox must emit a
`DRESSAGE_SWEGYM_REWARD_JSON=` marker.

## 5. Start synchronous GRPO

Prepare Qwen3.5-4B Hugging Face and Megatron distributed checkpoints under the
same model root, then run:

```bash
export MODEL_ROOT=/path/to/models
export PROMPT_DATA="$PWD/data/swegym-train-claude-code-e2b.jsonl"
export DRESSAGE_SANDBOX_PROVIDER=e2b
export DRESSAGE_E2B_API_KEY=e2b_...
export DRESSAGE_E2B_BLACKBOX_PORT=31000
export DRESSAGE_PROXY_URL=https://proxy.example.com

bash examples/scripts/run_swegym_claude_code_grpo_sync.sh
```

`DRESSAGE_PROXY_URL` must be an HTTP(S) endpoint that the E2B sandboxes can
reach.

The launcher explicitly selects
`dressage.rollout.generate.blackbox_dispatch_swegym.generate`. The generic
blackbox dispatcher remains recipe-agnostic; the SWE-Gym dispatcher owns fresh
evaluation and trajectory-integrity checks.

Reference defaults reproduce the reviewed experiment shape: TP2/CP4,
8 prompts × 16 samples, global batch size 128, 500 rollout updates, normalized
GRPO advantages, vanilla token-level TIS, and low-variance KL loss coefficient
`0.001`. Override infrastructure and topology through environment variables.
