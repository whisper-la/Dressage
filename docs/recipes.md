# Recipes

**Ready-to-Run Example Agents for Agentic RL**

[← Back to Main README](../README.md) · [Overview](#-overview) · [ALFWorld](#-alfworld) · [HotpotQA](#-hotpotqa) · [Build Your Own](#-building-your-own-recipe) · [Training Scripts](#-training-scripts)

## 📖 Overview

Dressage includes **example recipes** — complete whitebox agent implementations that demonstrate how to build, train, and evaluate agents using the framework. Each recipe is a self-contained agent with its own tools, reward functions, and training scripts. They serve as both working examples and starting points for your own agents.

Recipes are designed to showcase the framework's key capabilities:
- **Whitebox agent pattern** — subclass `WhiteboxAgent`, implement `rollout()`, export via `make_generate`
- **Custom tools** — define tools as Python functions, register them with the agent
- **Reward functions** — implement sample-oriented reward functions, wire via the registry
- **Training scripts** — complete bash scripts that wire all slime hooks together

```text
dressage/recipes/
├── alfworld/        # TextWorld navigation agent
│   ├── agent_whitebox.py  # ALFWorld WhiteboxAgent implementation
│   ├── tools.py           # Prompt, parsing, and TextWorld helpers
│   └── reward.py          # Task completion reward function
└── hotpotqa/        # Multi-hop retrieval agent
    ├── agent_whitebox.py  # HotpotQA WhiteboxAgent implementation
    ├── tools.py           # Prompt, parsing, and local search helpers
    └── reward.py          # Exact-match answer reward
```

## 🏠 ALFWorld

**TextWorld Navigation Agent for Household Tasks**

ALFWorld is a text-based interactive environment for household task completion, built on the TextWorld engine. The Dressage ALFWorld recipe implements a whitebox agent that navigates TextWorld environments using an `env_step` tool — sending text commands and receiving environment observations.

### Task Description

The agent receives natural language instructions for household tasks and must navigate a simulated home environment using text commands:

```text
Task: "Put a clean cup on the counter"

Agent thinks: I need to find a cup, clean it, and put it on the counter.

Agent actions:
  1. "go to kitchen"       → You see a kitchen with cabinets and a counter.
  2. "open cabinet 1"      → You see a dirty cup.
  3. "take cup from cabinet 1" → You pick up the cup.
  4. "go to sink"          → You see a sink.
  5. "clean cup with sink" → You clean the cup.
  6. "go to counter"       → You see a counter.
  7. "put cup on counter"  → ✅ Task complete!
```

### How It Works

The ALFWorld agent follows a simple but effective loop: ask the LLM for the next action, execute it in the environment, feed the observation back, repeat.

```text
ALFWorld Agent (WhiteboxAgent subclass)
        │
        ├── self.chat(messages)           → LLM generates next action
        │   └── System prompt describes available commands
        │   └── User messages include task + observation history
        │
        ├── env_step(action)              → Execute in TextWorld
        │   └── Agent sends text command to environment
        │   └── Environment returns observation + done signal
        │   └── Observation appended to conversation as tool result
        │
        └── Loop until:
            ├── Task complete (environment signals done)
            ├── Max steps reached
            └── Error occurs
```

### Architecture Details

 | Component | Implementation | Description |
 | :---------- | :--------------- | :------------ |
 | **Agent** | `WhiteboxAgent` subclass | Multi-turn agent with TextWorld interaction loop. Manages conversation history, parses LLM actions, handles tool responses. |
 | **Tool** | `env_step` function | Sends text commands to the ALFWorld environment. Returns environment observation and done signal. Commands include navigation (go to X), interaction (take, put, open, close), and cleaning (clean X with Y). |
 | **Reward** | Task completion binary | Returns `1.0` on successful task completion, `0.0` otherwise. The environment provides the completion signal. Simple but effective for RL training. |
 | **Evaluation** | Success rate | Measured as success rate across ALFWorld's six task categories (pick, clean, heat, cool, examine, pick two). |

### Running ALFWorld

```bash
# Sync rollout — simpler, good for debugging
bash examples/scripts/run_alfworld_whitebox_agent_qwen3.5_4b.sh

# Async rollout — better GPU utilization for training
bash examples/scripts/run_alfworld_whitebox_agent_qwen3.5_4b_async.sh
```

### Key Configuration

The ALFWorld scripts set these Dressage-specific variables:

```bash
DRESSAGE_PADDOCK_MODE=whitebox
DRESSAGE_SANDBOX_PROVIDER=local_bwrap
DRESSAGE_LOCAL_BWRAP_POOL_MODE=command_only

# Generate function points to the ALFWorld agent
--custom-generate-function-path dressage.recipes.alfworld.agent_whitebox.generate
--custom-rm-path dressage.reward.custom_rm.custom_rm
DRESSAGE_REWARD_MODULES=dressage.recipes.alfworld.reward

# Environment limits
ALFWORLD_MAX_STEPS=30
ALFWORLD_MAX_EPISODE_STEPS=30
```

## 🔍 HotpotQA

**Multi-Hop Retrieval Agent for Complex Question Answering**

HotpotQA is a question answering dataset requiring multi-hop reasoning over multiple Wikipedia paragraphs. The Dressage HotpotQA recipe implements a whitebox agent with a local FAISS+BGE retrieval tool — the agent searches for relevant passages, reasons over them, and produces an answer.

### Task Description

The agent answers complex questions that require finding and combining information from multiple documents:

```text
Question: "What is the birthplace of the director of Inception?"

Agent reasoning:
  1. Search "director of Inception"
     → "Inception is a 2010 film directed by Christopher Nolan"
  2. Search "Christopher Nolan birthplace"
     → "Christopher Nolan was born in London, England"
  3. Answer: "London, England"  ✅
```

### How It Works

The HotpotQA agent uses a retrieve-reason-retrieve loop: search for relevant information, reason about what's needed next, search again if necessary, then produce a final answer.

```text
HotpotQA Agent (WhiteboxAgent subclass)
        │
        ├── self.chat(messages)               → LLM generates query or answer
        │   └── System prompt describes retrieval tool
        │   └── User messages include question + retrieved passages
        │
        ├── retrieval_tool(query)             → Search FAISS index
        │   ├── Encode query using BGE embeddings (sentence-transformers)
        │   ├── Search FAISS index for top-k nearest neighbors
        │   └── Return top-k passage texts with scores
        │
        └── Loop: retrieve → reason → retrieve → answer
            ├── Agent decides whether to search or answer
            ├── Multiple retrieval rounds for multi-hop questions
            └── Final answer extracted from last assistant response
```

### Architecture Details

 | Component | Implementation | Description |
 | :---------- | :--------------- | :------------ |
 | **Agent** | `WhiteboxAgent` subclass | Multi-hop reasoning agent. Decides when to retrieve vs. answer. Manages retrieval context in conversation. |
 | **Retrieval** | FAISS + BGE | Local FAISS index from `HOTPOTQA_CORPUS_DIR/index.bin` plus passages from `HOTPOTQA_CORPUS_DIR/hpqa_corpus.jsonl`, using the configured BGE embedding model. |
 | **Reward** | Exact Match | Checks the last `<answer>...</answer>` block against gold answers after normalization. No F1, format bonus, or partial credit. |
 | **Evaluation** | Exact-match reward | `1.0` only when the normalized answer exactly matches a normalized target; otherwise `0.0`. |

### Additional Dependencies

The HotpotQA recipe requires additional packages for the retrieval component:

```bash
pip install faiss-cpu sentence-transformers numpy
```

### Retrieval Defaults

```bash
HOTPOTQA_CORPUS_DIR=/path/to/hotpotqa/corpus  # contains index.bin and hpqa_corpus.jsonl
HOTPOTQA_EMBEDDING_DEVICE=cpu
HOTPOTQA_TOPK=5
```

### Running HotpotQA

```bash
# Sync rollout — simpler, good for debugging
bash examples/scripts/run_hotpotqa_whitebox_agent_qwen3.5_4b.sh

# Async rollout — better GPU utilization for training
bash examples/scripts/run_hotpotqa_whitebox_agent_qwen3.5_4b_async.sh
```

### Key Configuration

```bash
DRESSAGE_PADDOCK_MODE=whitebox

# Generate function points to the HotpotQA agent
--custom-generate-function-path dressage.recipes.hotpotqa.agent_whitebox.generate
--custom-rm-path dressage.reward.custom_rm.custom_rm
DRESSAGE_REWARD_MODULES=dressage.recipes.hotpotqa.reward
```

## 🔮 Building Your Own Recipe

Creating a custom agent is straightforward: subclass `WhiteboxAgent`, implement `rollout()`, and export a `generate` function. Here's a complete guide.

### Step 1: Create Your Agent

```python
from dressage.rollout.generate.whitebox_agent import WhiteboxAgent, make_generate

class MyAgent(WhiteboxAgent):
    name = "my_agent"

    async def rollout(self, sample, sampling_params) -> str:
        """Run one complete agent trajectory.

        Args:
            sample: The prompt sample with task description
            sampling_params: LLM sampling configuration

        Returns:
            str: The agent's final answer or output
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": sample.prompt},
        ]

        response = await self.chat({
            "messages": messages,
            "model": "qwen3.5-4b",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for information",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
        })

        # Multi-turn tool use loop
        while has_tool_calls(response):
            tool_result = execute_my_tool(response)
            messages.append({"role": "tool", "content": tool_result})
            response = await self.chat({
                "messages": messages,
                "model": "qwen3.5-4b",
            })

        return extract_final_answer(response)

# Export the generate function — this is what slime calls
generate = make_generate(MyAgent)
```

### Step 2 (Optional): Add Sandbox Access

If your agent needs to execute code, read/write files, or run shell commands in an isolated environment, use `PaddockWhiteboxAgent`:

```python
from dressage.rollout.generate.whitebox_agent import PaddockWhiteboxAgent, make_generate

class MySandboxAgent(PaddockWhiteboxAgent):
    name = "my_sandbox_agent"

    async def rollout(self, sample, sampling_params) -> str:
        response = await self.chat({
            "messages": [{"role": "user", "content": sample.prompt}],
            "model": "qwen3.5-4b",
        })

        # Execute code in an isolated sandbox
        shell_text, shell_metadata = await self.paddock.tool_call(
            self.session_id, "shell.exec", {"cmd": "python solve.py"}
        )

        # Read output files from sandbox
        output, output_metadata = await self.paddock.tool_call(
            self.session_id, "file.read", {"path": "/workspace/result.txt"}
        )

        return output

generate = make_generate(MySandboxAgent)
```

### Step 3: Define Your Reward Function

```python
from dressage.reward.registry import register_reward

@register_reward("my_reward")
def my_reward(sample, *, args=None):
    """Compute reward for a completed trajectory.

    The sample contains the full trajectory data including
    the agent's final output, all tool interactions, and metadata.
    """
    metadata = sample.metadata or {}
    agent_output = sample.response or ""
    gold_answer = sample.label or metadata.get("gold_answer", "")
    proxy_extra = metadata.get("proxy_extra_info", {})

    # Your reward logic here
    if is_correct(agent_output, gold_answer):
        return 1.0
    elif is_partially_correct(agent_output, gold_answer):
        return 0.5
    else:
        return 0.0
```

### Step 4: Wire Into Slime

```bash
python3 -m slime.train \
  --custom-generate-function-path my_package.my_agent.generate \
  --rollout-function-path \
    dressage.rollout.sync_rollout.generate_rollout_sync \
  --custom-rm-path dressage.reward.custom_rm.custom_rm \
  --custom-convert-samples-to-train-data-path \
    dressage.rollout.convert_samples.convert_samples_to_train_data \
  --custom-reward-post-process-path \
    dressage.training.reward_post_process.reward_post_process \
  --data-source-path \
    dressage.rollout.data_source.DressageDataSource \
  --advantage-estimator grpo
```

```bash
export DRESSAGE_REWARD_MODULES=my_package.reward
```

Custom datasets select the reward through sample metadata:

```json
{"prompt": "Solve the task", "label": "expected answer", "metadata": {"instance_id": "custom-001", "reward_fn": "my_reward"}}
```

> [!TIP]
> Check out the ALFWorld and HotpotQA implementations in `dressage/recipes/` for complete working examples. They demonstrate the full pattern including edge case handling, error recovery, and training script configuration.

### Recipe Checklist

When building a new recipe, make sure you have:

- [ ]  Agent class extending `WhiteboxAgent` or `PaddockWhiteboxAgent`
- [ ]  Tool implementations (Python functions or paddock `tool_call`)
- [ ]  Reward function registered via `@register_reward`
- [ ]  Training script with all slime hook paths
- [ ]  Prompt dataset in JSONL format
- [ ]  Basic tests for agent logic and reward computation

## 📊 Training Scripts

All example training scripts are in `examples/scripts/`. They demonstrate different combinations of agent modes, sandbox providers, and async scheduling:

### Recipe Scripts

 | Script | Agent | Mode | Description |
 | :------- | :------ | :----- | :------------ |
 | `run_alfworld_whitebox_agent_qwen3.5_4b.sh` | ALFWorld | Sync | Whitebox, synchronous rollout |
 | `run_alfworld_whitebox_agent_qwen3.5_4b_async.sh` | ALFWorld | Async | Whitebox, fully async rollout |
 | `run_hotpotqa_whitebox_agent_qwen3.5_4b.sh` | HotpotQA | Sync | Whitebox, synchronous rollout |
 | `run_hotpotqa_whitebox_agent_qwen3.5_4b_async.sh` | HotpotQA | Async | Whitebox, fully async rollout |

### Blackbox Scripts

 | Script | Model | Sandbox | Mode |
 | :------- | :------ | :-------- | :----- |
 | `run_blackbox_qwen3.5_4b_async_local.sh` | Qwen3.5-4B | Local bwrap | Fully async |
 | `run_blackbox_qwen3.5_4b_async_remote.sh` | Qwen3.5-4B | E2B remote | Fully async |
 | `run_blackbox_qwen3.5_4b_partial_rollout_async_local.sh` | Qwen3.5-4B | Local bwrap | Partial async |
 | `run_blackbox_qwen3.5_4b_partial_rollout_async_remote.sh` | Qwen3.5-4B | E2B remote | Partial async |
 | `run_blackbox_qwen3.5_35b_a3b_sync_local.sh` | Qwen3.5-35B-A3B | Local bwrap | Sync |
 | `run_blackbox_qwen3.5_35b_a3b_sync_remote.sh` | Qwen3.5-35B-A3B | E2B remote | Sync |

## 📂 Sample Data

Prompt datasets for training runs are in `examples/data/`:

 | File | Description | Format |
 | :----- | :------------ | :------- |
 | `dressage_dapo_prompts.jsonl` | DAPO-style coding prompts with SWE-bench-like task descriptions | JSONL with `prompt`, `metadata` fields |

---

[← Training](./training.md) · [Back to Main README](../README.md) · [Next: Quick Start →](./quickstart.md)
