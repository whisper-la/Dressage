"""Prepare HotpotQA distractor split as Dressage-compatible JSONL.

Output schema (one JSON object per line):
    prompt:   str          - the raw question (DressageDataSource input_key="prompt")
    label:    json-string  - {"ground_truth": {"target": [answer]}} for search_em compat
    metadata: dict         - {instance_id, level, type, supporting_facts, reward_fn}

Run on the training host with HF_ENDPOINT either default or set to a mirror.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_NAMES = ("hotpotqa/hotpot_qa", "hotpot_qa")


def to_jsonl_record(ex: dict, *, reward_fn: str) -> dict:
    answer = ex["answer"]
    label = {"ground_truth": {"target": [answer]}}
    metadata = {
        "instance_id": ex["id"],
        "level": ex.get("level"),
        "type": ex.get("type"),
        "supporting_facts": ex.get("supporting_facts"),
        "reward_fn": reward_fn,
    }
    return {
        "prompt": ex["question"],
        "label": json.dumps(label, ensure_ascii=False),
        "metadata": metadata,
    }


def dump_split(split: str, out_path: Path, reward_fn: str, limit: int | None) -> int:
    last_error: Exception | None = None
    for dataset_name in DATASET_NAMES:
        try:
            ds = load_dataset(dataset_name, "distractor", split=split)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError("failed to load HotpotQA distractor split") from last_error

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            rec = to_jsonl_record(ex, reward_fn=reward_fn)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPT_DIR),
    )
    parser.add_argument("--reward-fn", default="hotpotqa")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"

    n_train = dump_split("train", train_path, args.reward_fn, args.train_limit)
    print(f"wrote {n_train} train examples -> {train_path}")
    n_dev = dump_split("validation", dev_path, args.reward_fn, args.dev_limit)
    print(f"wrote {n_dev} dev examples   -> {dev_path}")


if __name__ == "__main__":
    main()
