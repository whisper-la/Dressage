# Blackbox Dressage Claw Experiment with OpenClaw

This experiment validates that **Dressage can effectively train claw-type agentic tasks** and substantially improve model performance through blackbox RL. We trained [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) on the [Dressage Claw](https://huggingface.co/datasets/huang3eng/Dressage-Claw) dataset with [OpenClaw](https://openclaw.ai/) as the agent. The run shows that this pipeline handles diverse, multi-service agentic environments and delivers meaningful gains.

## Results at a Glance

The successful 64k run produced clear gains on unseen ClawEval tasks. On [ClawEval](https://claw-eval.github.io) **general** split, the final checkpoint (iter 200) kept mean score essentially flat with a slight uptick (**0.660** → **0.670**) and Pass@3 roughly flat versus the baseline (**69.4** → **68.8**), while Pass@1 and the strict Pass^3 improved markedly — Pass@1 rose from **57.1** to **64.8** , and Pass^3 rose from **41.4** to **55.4**. Each task was run with 3 independent trials.

![ClawEval four metrics across checkpoints](../assets/claw/claw_eval_4metrics.png)

On the public ClawEval **general** split leaderboard, our trained checkpoint achieves a Pass^3 of **55.4**, between Gemini 3.1 Pro and GPT 5.4 (**60.2**), up 14 points from the baseline (**41.4**); its average token cost is about **68K** per trajectory, essentially flat versus the baseline (~62K).

![ClawEval general split leaderboard comparison](../assets/claw/bench.png)

## Download and Convert Data

The Dressage Claw dataset contains **441 agentic tasks** spanning communication (email, messaging), finance (expense reporting, bank statements), operations (inventory, procurement, incident management), productivity (calendar, todo, notes, documents), and other real-world domains. Each task pairs a natural-language instruction with a mock service environment, optional MCP tools, and a rubric-based grader.

The dataset is available on Hugging Face at **[`huang3eng/Dressage-Claw`](https://huggingface.co/datasets/huang3eng/Dressage-Claw)**.

Each task lives under `tasks/<task_id>/` with the following layout:

```text
tasks/<task_id>/
├── task.json              # Task metadata, grader config, file mappings
├── task/
│   ├── prompt.md          # Task instruction (prompt text)
│   ├── workspace/         # Workspace files
│   ├── scripts/           # setup.sh, start_services.sh, stop_services.sh
│   ├── tools/             # MCP tool spec JSON files
│   ├── services/          # Mock service configurations
│   └── runtime/           # Helper scripts (MCP, trace, judge patching)
└── grader/
    └── run.sh             # Grader entry point (prints overall_score JSON)
```

The converter tracks these per-task features:

| Feature | Description |
| :------ | :---------- |
| `has_services` | Task starts mock services (email, calendar, CRM, etc.) before the agent runs |
| `has_llm_judge` | Task includes an LLM-judge rubric |
| `has_tools` | Task declares MCP tools that the agent can invoke |
| `has_system_prompt` | Task provides a custom system prompt prefix |
| `needs_patch` | Task grader/scripts require the `patch` binary (auto-installed) |

The data preparation script is:

```text
examples/data/dressage_claw/prepare_dressage_claw_data.py
```

`prepare_dressage_claw_data.py` accepts a local Parquet archive or downloads
Parquet from Hugging Face, then converts it into Dressage-consumable JSONL.

- Download from Hugging Face (default):

```bash
python3 examples/data/dressage_claw/prepare_dressage_claw_data.py \
  --output /path/to/restored_dressage_claw
```

- Read a local Parquet file:

```bash
python3 examples/data/dressage_claw/prepare_dressage_claw_data.py \
  --parquet /path/to/parquet/Dressage-Claw.parquet
```

- Convert an existing local dataset directory:

```bash
python3 examples/data/dressage_claw/prepare_dressage_claw_data.py \
  --dataset-dir /path/to/restored_dressage_claw \
  --output /path/to/jsonl_output
```

The existing JSONL conversion adds these fields:

| Field | Purpose |
| :---- | :------ |
| `prompt` | The task instruction text from `prompt.md`, sent to OpenClaw. |
| `reward_fn` | Set to `claw_grader`, selecting `dressage.recipes.dressage_claw.reward`. |
| `metadata.blackbox_type` | Set to `openclaw` so BlackboxServer starts the OpenClaw backend. |
| `metadata.sandbox_image` | The E2B template name, default `e2b-dressage-claw-blackbox`. |
| `metadata.instance_id`, `task_id` | Traceability fields for logging and result analysis. |
| `metadata.before_agent_files` | All files under `task/`, written to the task-declared target paths before the agent runs (workspace, scripts, tools, services, runtime). |
| `metadata.after_agent_files` | Grader files under `grader/`, written to the task-declared target paths only after the agent finishes. |
| `metadata.system_prompt_file` | Path to the task's system prompt prefix file inside the sandbox, when declared. |
| `metadata.blackbox_execute_cmds.before_agent` | Pre-agent commands: verify file injection, map workspace to target paths, run setup, start services, register MCP tools. |
| `metadata.blackbox_execute_cmds.after_agent` | Post-agent commands: inject grader files, patch LLM judge config, generate trace, stop services, run grader. |

Random sampling for debugging:

```bash
# Randomly sample 64 tasks
python3 examples/data/dressage_claw/prepare_dressage_claw_data.py \
  --output examples/data/dressage_claw_sample \
  --limit 64
```

## Prepare Sandbox

Our experiment ran on an internal sandbox system. For public reproduction, this section provides an E2B reference path based on a public Python image. The template builder installs OpenClaw 2026.6.6, BlackboxServer, and the task runtime dependencies during the E2B build.

Dressage blackbox training requires each sandbox to run a BlackboxServer process and OpenClaw. OpenClaw is installed into the template and resolved from the image's PATH; BlackboxServer is installed into an isolated venv (`/opt/dressage-bbs-venv`) created by the template builder and launched from that venv. The mock services and graders that a task starts are plain Python scripts executed with the sandbox's default `python3`; keeping BlackboxServer in its own venv avoids conflicts with task-side dependencies.

Build the E2B template directly from the public base image:

```bash
E2B_API_KEY=e2b_... \
BUILD_FROM_PUBLIC_IMAGE=1 \
TASK_IMAGE='python:3.11-slim-bookworm' \
TEMPLATE_NAME='e2b-dressage-claw-blackbox' \
python3 examples/data/dressage_claw/build_e2b_blackbox_template.py
```

The template builder:

1. Starts from `python:3.11-slim-bookworm` and installs OpenClaw 2026.6.6 plus the task-side Python dependencies.
2. Installs system dependencies (bash, curl, ca-certificates, patch).
3. Creates an isolated venv at `/opt/dressage-bbs-venv` for BlackboxServer so the task image's Python environment stays untouched.
4. Installs `blackbox_server` into that venv.
5. Sets the start command to launch BlackboxServer on port `31000`:

   ```bash
   cd /workspace_sandbox
   BBS_HOST=0.0.0.0 BBS_PORT=31000 \
   BBS_RUNTIME_ROOT=/workspace_sandbox/blackbox_server_runtime \
   /opt/dressage-bbs-venv/bin/python -m blackbox_server.main
   ```

The start configuration (`BBS_HOST`, `BBS_PORT`, `BBS_RUNTIME_ROOT`) is baked into the template; do not rely on per-sandbox `e2b_envs` to configure the server start.

Smoke test the template before training:

Set `E2B_API_KEY` in the environment before running the snippet.

```python
import asyncio
from e2b import AsyncSandbox

async def main():
    sandbox = await AsyncSandbox.create(template="e2b-dressage-claw-blackbox")
    try:
        print(await sandbox.get_host(31000))
        result = await sandbox.commands.run("curl -sf http://127.0.0.1:31000/health")
        print(result.stdout)
    finally:
        await sandbox.kill()

asyncio.run(main())
```

## Reward Design

The Dressage Claw reward comes entirely from the grader files. Dressage imposes no universal rubric: every reward is produced by the task's own grader inside the sandbox, and the training-side reward function (`dressage.recipes.dressage_claw.reward.claw_grader`) is just a thin, conservative parser that reads the grader's `overall_score`. The core design choices are grader isolation, trustworthy tool-call evidence, fault-aware scoring, and a continuous `overall_score`.

### Staged grader injection and raw tool-call traces

Task files (`before_agent_files`) and grader files (`after_agent_files`) are two separate file sets, and the grader files are only written after the session has been finalized. The grading spec is not neutral metadata: it embeds the expected key values, required tool lists, and judge rubrics, effectively the answer key. Instead of detecting grader-reading behavior post hoc, the design removes the possibility: grader code, rubrics, and fixtures simply do not exist in the sandbox during the agent run, which closes the most direct reward-hacking channel in blackbox RL by construction rather than by monitoring.

The evidence the grader scores against is equally hard to game. The grader never asks the agent what it did; tool-call evidence is recorded inside the MCP dispatcher, where every `tools/call` is forwarded to the mock-service HTTP backend and simultaneously appended, together with the full request, response, status, and latency, to a `dispatches.jsonl` trace. The trace is therefore tamper-evident (it reflects real HTTP calls, not the agent's self-report), agent-agnostic (swapping in another agent leaves the evidence chain intact), and richer than the transcript alone: tool responses join the deterministic matching corpus, so a key value the agent retrieved but never echoed still counts as covered.

### Score composition and continuous reward

The `overall_score` is composed of three parts. The **deterministic block** (weight 0.5) checks key-value coverage over the transcript and tool trajectory plus required-tool coverage, which is fast, reproducible, and free, providing a zero-variance floor for the reward. The **LLM judge block** (weight 0.5) covers the semantic dimensions that substring matching cannot measure, such as answer quality, reasoning correctness, and distractor rejection; one consolidated judge call scores all rubric criteria against a compact evidence pack, cutting judge cost and latency several-fold at training scale (the judge is served by the **gpt-5.4** API). The **safety gate** over forbidden tool calls is deliberately a weight-0 `zero_task` gate rather than a weighted score item: a positive weight would both dilute the other criteria and allow the agent to absorb a safety violation as a small score deduction, whereas a gate zeroes the whole task on violation and leaves the score distribution untouched otherwise, since safety is a hard constraint, not a trade-off. The composition also accounts for evaluation faults: a failed judge call marks its criteria `skipped` rather than `failed`, and the aggregation normalizes conservatively so that infrastructure faults neither punish good trajectories nor inflate scores.

The resulting `overall_score` is a continuous value in `[0, 1]`. Multi-step work earns partial credit even when only partly completed, so within-group rewards don't collapse to all-0 or all-1 under a binary signal and lose their gradient signal; this keeps within-group reward variance intact, so these groups still contribute learning signal to GRPO.

## Experiment Process

### Initial 32k Run

Our first run used a 32k context window. The reward rose quickly from about 0.55 to around 0.65 within the first ~50 steps and then plateaued: over the following ~200 steps it oscillated mostly within 0.60–0.67 without further gains, with raw values swinging between roughly 0.38 and 0.78, showing premature convergence, limited gain, and severe fluctuation. ClawEval evaluation under this setup confirmed the severity of the problem: the score dropped monotonically from the baseline **0.583** as training progressed, reaching **0.577** at iter 20, **0.495** at iter 140, and only **0.441** at iter 240, about 0.14 below the baseline. The modest training-reward gain did not translate into generalization; the model actually degraded as training continued.

The segment curve explains why: under the 32k window the mean segment count per trajectory stayed above 2 for long stretches (overall mean ≈ 2.1, with roughly 42% of training steps ≥ 2), held a sustained 4–5 plateau between steps 50–90, and peaked near 6. Segments are a proxy signal for how often a trajectory had to be split, including split points caused by OpenClaw's compact behavior, which the 32k window triggered frequently.

More segments hurt training in three ways. First, compact is a passive behavior: the runtime triggers it only after the trajectory has already grown long, and the agent never chooses an explicit compact action, so RL cannot perform credit assignment over it. Second, multi-segment trajectories share one trajectory-level reward during training, with the reward broadcast to every segment while each segment only carries its own truncated context; the more segments there are, the weaker the association between an individual segment's content and the final outcome, and the noisier the credit assignment. Third, compact summaries are lossy: after compaction the agent rarely recovers enough task state, so subsequent segments build on distorted state, which both lowers the success rate of long trajectories and pushes the model toward shallow behavior patterns that collect partial credit from short trajectories. Training reward inches up, but that policy does not generalize, which is why the ClawEval score kept falling.

![32k run reward and segment count](../assets/claw/failed_32k_reward_segments.png)

We considered two ways to reduce the segment count. The first was to make trajectories shorter under the same 32k window; the second was to enlarge the context window so longer trajectories would no longer need to be split as often.

### Shorter Trajectories

We first tried the shorter-trajectory route by adding a segment penalty to the training reward. The penalty was intentionally mild and only activated after the first segment, so single-segment trajectories were unchanged while repeated compaction became progressively less attractive:

```text
adjusted_reward = overall_score - 0.05 * max(0, segment_count - 1)
adjusted_reward = clip(adjusted_reward, 0.0, 1.0)
```

The goal was not to punish legitimate tool use, but to bias the policy away from verbose exploration patterns that filled the context and triggered compaction. This did reduce the average segment count from roughly **2.1** in the failed 32k run to about **1.9**, and the evaluation reward improved modestly: ClawEval mean score recovered from **0.441** at the failed 32k checkpoint to roughly **0.49-0.50**. However, the overall behavior was still unsatisfactory and remained well below the baseline **0.583**. The model learned to stop earlier and compress its reasoning, but many tasks require enough evidence gathering and multi-step tool interaction that shorter trajectories often meant incomplete work rather than better work. The reward gain was therefore shallow and did not produce a strong ClawEval improvement. It also pushed the reward function beyond Dressage Claw's intended lightweight design: rather than just parsing the grader's `overall_score`, it injected a behavioral preference. We treated this as evidence that segment count was a symptom of insufficient usable context, not merely a behavior that could be fixed by regularization, and moved to the second route.

### Longer Usable Context

Trajectory analysis showed that OpenClaw injects all ~33 built-in tools by default, with the tool schema taking up about 70% of the system prompt tokens (~13.5k); yet in the sampled trajectories the vast majority of these built-in tools were never called, so nearly two-thirds of the system prompt budget was spent on tool declarations that never produced any behavior. The final configuration combined two context-oriented measures.

First, **trimming the tool set**. The agent is registered with the minimal tools profile, keeping only four native tools and allow-listing the task-declared MCP tools:

```python
_MINIMAL_TOOLS_PROFILE = "minimal"
_OPENCLAW_ALSO_ALLOW_TOOLS = [
    "claw-tools__*",  # MCP tool wildcard
    "exec", "read", "write", "edit",
]
```

The four remaining native tools, exec/read/write/edit, total only about 1,300 tokens; together with the MCP tools, the tool schema drops from ~13.5k to ~1.7k tokens. Since the removed tools never appeared in the trajectories, the trim costs no capability; it hands the freed context budget back to the task's own tool responses, multi-step reasoning, and final answer, so the same window holds more useful trajectory content and compaction is deferred. It also shrinks the action space: the agent no longer scatters exploration across thirty-plus built-in tools it never calls.

Second, **enlarging the training context** from 32k to 64k. The context window is the product of the per-GPU token budget and the context-parallel size (`CONTEXT_WINDOW = MAX_TOKENS_PER_GPU × CP_SIZE`): the 32k single-node run used 8192 × CP=4; the extension keeps 8192 tokens per GPU and raises CP_SIZE to 8, i.e. 8192 × 8 = 64k.

With both measures combined, the training dynamics improved markedly. The mean segment count dropped sharply: it stays near 1 in early training, rises slowly to about 1.2–1.3 later as trajectories grow longer, peaks at only 1.67, and never touches 2 again, with compaction turning from the norm into an occasional event on a few long trajectories. Meanwhile the reward (`raw_reward_trajectory_mean`) started at about **0.65**, already well above the 32k run's starting point, indicating that trimming the tool schema improved task performance by itself; it then rose steadily after an initial exploration phase, reached around **0.80** by step 100 with occasional peaks touching 0.90, and converged to approximately **0.82** with decreasing variance in the later stages. Low segment counts and a steadily rising reward appeared together, in contrast to the 32k run where both deteriorated simultaneously.

![64k run reward and mean segment count](../assets/claw/success_64k_reward_segments.png)

### Final 64k Training Setup

The successful configuration trained Qwen3.6-35B-A3B on 4 nodes (32 GPUs) in synchronous colocate mode with GRPO, using TP=2 / CP=8 / EP=8 parallelism. The rollout ran with a 64k context window, a 16k per-response token budget, and an 8k compact reserve; each rollout step sampled 8 prompts × 8 trajectories per group, and the agent was capped at 100 steps per session. Sandboxes were provisioned from the prebuilt `e2b-dressage-claw-blackbox` template with no retry on sandbox failure. The full parameter list lives in the run script (`examples/scripts/run_dressage_claw_qwen3.6_35b_a3b_sync_4_node.sh`).

Start the final run with the converted JSONL and the sandbox template prepared above:

```bash
MODEL_ROOT=/path/to/models \
PROMPT_DATA=/path/to/restored_dressage_claw/dressage_claw_e2b.jsonl \
DRESSAGE_E2B_API_KEY=e2b_... \
bash examples/scripts/run_dressage_claw_qwen3.6_35b_a3b_sync_4_node.sh
```

During rollout, the Dressage Claw dispatch (`dressage.recipes.dressage_claw.dispatch.generate`) adds several Dressage Claw-specific behaviors on top of the standard blackbox dispatch:

- **Pre-register file injection**: `before_agent_files` are written to the sandbox before `register_agent` so that sandbox paths referenced at register time (e.g., `system_prompt_file`) exist when OpenClaw starts.
- **System prompt injection**: When a task provides a `system_prompt_file`, it is passed to OpenClaw via the `register_agent` call.
- **Early session finalization**: After the agent call, the session is finalized and the full conversation history (`agent_messages.json`) is written to the sandbox before `after_agent` commands run. This ensures the grader has access to the complete agent conversation.
- **Multi-source fallback**: `agent_messages.json` is built from the Proxy trajectory data (preferred, includes full conversation including final answer), with fallbacks to BBS `conversation_history` and finally a prompt-only fallback payload.
- **Post-agent grading**: Grader files are injected only after the session is finalized, then `run_grader` evaluates the collected transcript and tool traces and prints a JSON object containing `overall_score`. The training-side reward function reads that score from the `run_grader` command record; malformed, missing, or truncated grader output is treated as `0.0`.
