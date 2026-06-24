#!/usr/bin/env python3
"""Prepare ALFWorld training data for Dressage.

Scans the ALFWorld game files (after `alfworld-download`) and produces a
JSONL dataset compatible with DressageDataSource.

Each line:
{
    "prompt": "<task description from the game>",
    "label": "",
        "metadata": {
            "instance_id": "pick_and_place-Knife-None-SideTable-...",
            "reward_fn": "alfworld",
            "task_type": "pick_and_place",
            "game_file": "examples/data/alfworld/alfworld_data/.../game.tw-pddl"
        }
}

Usage:
    # After: pip install alfworld && alfworld-download
    python prepare_alfworld.py [--alfworld-data PATH] [--output-dir PATH] [--split train]

Default ALFWORLD_DATA is read from alfworld's config or env var.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_ALFWORLD_DATA = SCRIPT_DIR / "alfworld_data"


def looks_like_alfworld_data(path: str | Path) -> bool:
    root = Path(path)
    return any((root / "json_2.1.1").glob("*")) or any(root.glob("**/game.tw-pddl"))


def find_alfworld_data(override: str | None = None) -> str:
    """Locate ALFWorld data directory."""
    if override:
        return override

    env_path = os.environ.get("ALFWORLD_DATA")
    if env_path and os.path.isdir(env_path):
        return env_path

    if DEFAULT_ALFWORLD_DATA.is_dir() and looks_like_alfworld_data(DEFAULT_ALFWORLD_DATA):
        return str(DEFAULT_ALFWORLD_DATA)

    # Default alfworld-download location
    default = os.path.expanduser("~/.cache/alfworld/data")
    if os.path.isdir(default) and looks_like_alfworld_data(default):
        return default

    # Try alfworld's own config
    try:
        import alfworld.agents
        cfg_path = os.path.join(os.path.dirname(alfworld.agents.__file__), "config.yaml")
        if os.path.exists(cfg_path):
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            data_path = cfg.get("general", {}).get("data_path", "")
            if data_path and os.path.isdir(data_path) and looks_like_alfworld_data(data_path):
                return data_path
    except Exception:
        pass

    print("ERROR: Cannot find ALFWorld data. Set --alfworld-data or ALFWORLD_DATA env var.")
    sys.exit(1)


def get_task_type(game_file: str) -> str:
    """Infer task type from game file path."""
    parts = game_file.split("/")
    for part in parts:
        for task in ["pick_and_place", "clean", "heat", "cool", "pick_two"]:
            if task in part:
                return task
        if "look_at_obj_in_light" in part or "examine" in part:
            return "examine"
    return "unknown"


def extract_task_desc(game_file: str) -> str:
    """Extract task description from the game's json or tw-pddl file."""
    json_path = os.path.join(os.path.dirname(game_file), "traj_data.json")
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                traj = json.load(f)
            task_desc = traj.get("turk_annotations", {}).get("anns", [{}])[0].get("task_desc", "")
            if task_desc:
                return task_desc.strip()
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

    # Fallback: derive from folder name
    folder_name = os.path.basename(os.path.dirname(game_file))
    return folder_name.replace("-", " ").replace("_", " ")


def collect_games(data_dir: str, split: str = "train") -> list[dict]:
    """Collect all game files for the given split."""
    samples = []

    # ALFWorld stores games under json_2.1.1/train/ or similar
    patterns = [
        os.path.join(data_dir, "json_2.1.1", split, "**", "game.tw-pddl"),
        os.path.join(data_dir, "json_2.1.1", split, "**", "*.tw-pddl"),
    ]

    game_files = set()
    for pattern in patterns:
        game_files.update(glob.glob(pattern, recursive=True))

    if not game_files:
        # Try alternative structure
        alt_patterns = [
            os.path.join(data_dir, split, "**", "game.tw-pddl"),
            os.path.join(data_dir, "**", split, "**", "*.tw-pddl"),
        ]
        for pattern in alt_patterns:
            game_files.update(glob.glob(pattern, recursive=True))

    for game_file in sorted(game_files):
        game_file = os.path.abspath(game_file)
        try:
            stored_game_file = Path(game_file).resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            stored_game_file = game_file
        task_type = get_task_type(game_file)
        task_desc = extract_task_desc(game_file)
        instance_id = os.path.basename(os.path.dirname(game_file))

        samples.append({
            "prompt": task_desc,
            "label": "",
            "metadata": {
                "instance_id": instance_id,
                "reward_fn": "alfworld",
                "task_type": task_type,
                "game_file": stored_game_file,
            },
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare ALFWorld data for Dressage")
    parser.add_argument("--alfworld-data", type=str, default=None,
                        help="Path to ALFWorld data root")
    parser.add_argument("--output-dir", type=str,
                        default=str(SCRIPT_DIR),
                        help="Output directory for JSONL files")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "eval_out_of_distribution", "eval_in_distribution"],
                        help="Which split to process")
    args = parser.parse_args()

    data_dir = find_alfworld_data(args.alfworld_data)
    print(f"ALFWorld data directory: {data_dir}")

    samples = collect_games(data_dir, args.split)
    if not samples:
        print(f"ERROR: No game files found for split '{args.split}' in {data_dir}")
        print("Make sure you ran: alfworld-download")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.split}.jsonl")

    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {output_file}")

    # Print task type distribution
    from collections import Counter
    dist = Counter(s["metadata"]["task_type"] for s in samples)
    print("Task type distribution:")
    for task_type, count in dist.most_common():
        print(f"  {task_type}: {count}")


if __name__ == "__main__":
    main()
