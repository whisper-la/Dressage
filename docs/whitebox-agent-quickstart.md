# Whitebox Agent Quick Start

This guide prepares the two bundled whitebox agent recipes:

- ALFWorld: TextWorld environment interaction through the `env_step` tool.
- HotpotQA: multi-hop retrieval with local FAISS+BGE search.

The scripts use these defaults:

- repo checkout: this repository root
- data root: `./examples/data`
- model root: `/root`

If you already have these model directories, skip directly to the recipe you
want to run:

- `/root/Qwen3.5-4B`
- `/root/Qwen3.5-4B_torch_dist`
- `/root/Qwen3.5-4B_slime`

Jump to [ALFWorld](#alfworld) or [HotpotQA](#hotpotqa).

## Common Qwen Setup

Install the common helpers:

```bash
python3 -m pip install -U huggingface_hub
```

This installs the `hf` CLI.

Download the Qwen HF checkpoint into `/root/Qwen3.5-4B`. Use the public HF repo
or your internal mirror path:

```bash
hf download <QWEN3_5_4B_HF_REPO> \
  --local-dir /root/Qwen3.5-4B
```

Convert the HF checkpoint to the torch-dist format used by Slime:

```bash
source slime/scripts/models/qwen3.5-4B.sh

PYTHONPATH=/root/Megatron-LM:$PWD/slime:$PWD \
torchrun --nproc_per_node 8 slime/tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /root/Qwen3.5-4B \
  --save /root/Qwen3.5-4B_torch_dist \
  "${MODEL_ARGS[@]}"
```

This script converts the HF checkpoint into the Slime torch-dist checkpoint
format used by training.

Initialize the trainable checkpoint directory:

```bash
rm -rf /root/Qwen3.5-4B_slime
cp -a /root/Qwen3.5-4B_torch_dist /root/Qwen3.5-4B_slime
```

Expected model layout:

```text
/root/
  Qwen3.5-4B/
  Qwen3.5-4B_torch_dist/
  Qwen3.5-4B_slime/
```

## ALFWorld

Install ALFWorld dependencies:

```bash
python -m pip install -U pip setuptools wheel
python3 -m pip install alfworld textworld gymnasium pyyaml
```

Download ALFWorld game data into the repo-local data directory:

```bash
mkdir -p examples/data/alfworld/alfworld_data
ALFWORLD_DATA="$(pwd)/examples/data/alfworld/alfworld_data" alfworld-download
```

Build `train.jsonl`:

```bash
python3 examples/data/alfworld/prepare_alfworld.py \
  --alfworld-data examples/data/alfworld/alfworld_data \
  --output-dir examples/data/alfworld \
  --split train
```

This script will:

- scan the ALFWorld train split under `alfworld_data`
- write `examples/data/alfworld/train.jsonl`
- store repo-relative `game_file` paths for rollout-time environment loading

Run ALFWorld:

```bash
bash examples/scripts/run_alfworld_whitebox_agent_qwen3.5_4b.sh
```

Run ALFWorld async:

```bash
bash examples/scripts/run_alfworld_whitebox_agent_qwen3.5_4b_async.sh
```

## HotpotQA

Install data and retrieval dependencies:

```bash
python3 -m pip install -U datasets sentence-transformers faiss-cpu huggingface_hub
```

Download the BGE embedding model for local retrieval:

```bash
hf download BAAI/bge-large-en-v1.5 \
  --local-dir /root/bge-large-en-v1.5
```

If `/root/bge-large-en-v1.5` already exists, skip this download step.

If you use a Hugging Face mirror, set `HF_ENDPOINT` before the following
commands.

Build HotpotQA train/dev JSONL from the `hotpot_qa` dataset, `distractor`
config:

```bash
python3 examples/data/hotpotqa/prepare_hotpotqa.py \
  --out-dir examples/data/hotpotqa
```

This script will:

- download the HotpotQA distractor train/dev splits
- write `examples/data/hotpotqa/train.jsonl` and `dev.jsonl`
- store answers in the reward-compatible JSON-string `label` field

Build the retrieval corpus and FAISS index:

```bash
python3 examples/data/hotpotqa/build_corpus.py \
  --out-dir examples/data/hotpotqa/corpus

python3 examples/data/hotpotqa/build_index.py \
  --corpus examples/data/hotpotqa/corpus/hpqa_corpus.jsonl \
  --out-dir examples/data/hotpotqa/corpus \
  --embedding-model /root/bge-large-en-v1.5 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --batch-size 2048
```

Use `--devices cpu` if GPU memory is unavailable.

These scripts will:

- extract HotpotQA context paragraphs into `hpqa_corpus.jsonl`
- embed passages with BGE
- save `hpqa_corpus.npy` and `index.bin` for local search

Run HotpotQA:

```bash
bash examples/scripts/run_hotpotqa_whitebox_agent_qwen3.5_4b.sh
```

Run HotpotQA async:

```bash
bash examples/scripts/run_hotpotqa_whitebox_agent_qwen3.5_4b_async.sh
```

## Overrides

The scripts default to `./examples/data` and `/root`, but the common overrides
remain available:

```bash
DATA_ROOT=/some/other/data/root \
BASE_FOLDER=/root \
HOTPOTQA_EMBEDDING_DEVICE=cuda:0 \
bash examples/scripts/run_hotpotqa_whitebox_agent_qwen3.5_4b.sh
```
