#!/usr/bin/env python3
"""Build hpqa_corpus.jsonl from HotpotQA distractor split context paragraphs.

Each HotpotQA example contains 10 Wikipedia context paragraphs (title + sentences).
This script extracts and deduplicates them into a single corpus file suitable for
FAISS indexing.

Usage:
    python build_corpus.py [--out-dir ./examples/data/hotpotqa/corpus]

Output:
    <out-dir>/hpqa_corpus.jsonl  — one JSON object per line: {"title": "...", "text": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_NAMES = ("hotpotqa/hotpot_qa", "hotpot_qa")


def load_hotpotqa_split(split: str):
    last_error: Exception | None = None
    for dataset_name in DATASET_NAMES:
        try:
            return load_dataset(dataset_name, "distractor", split=split)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to load HotpotQA distractor split: {split}") from last_error


def build_corpus(out_dir: Path, limit: int | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "hpqa_corpus.jsonl"

    seen: set[tuple[str, str]] = set()
    count = 0

    print("[build_corpus] Loading HotpotQA distractor split (train + validation)...")

    for split in ("train", "validation"):
        print(f"[build_corpus] Processing split: {split}")
        ds = load_hotpotqa_split(split)

        for ex in ds:
            context = ex.get("context", {})
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])

            for title, sentences in zip(titles, sentences_list):
                title = str(title).strip()
                text = " ".join(str(s).strip() for s in sentences if str(s).strip())
                if not text:
                    continue
                key = (title, text)
                if key in seen:
                    continue
                seen.add(key)
                count += 1

            if limit and count >= limit:
                break
        if limit and count >= limit:
            break

    print(f"[build_corpus] Writing {count:,} unique paragraphs to {corpus_path}")
    with corpus_path.open("w", encoding="utf-8") as f:
        for title, text in seen:
            f.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")

    print(f"[build_corpus] Done: {count:,} paragraphs -> {corpus_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HotpotQA retrieval corpus from context paragraphs.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(SCRIPT_DIR / "corpus"),
        help="Output directory for hpqa_corpus.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max paragraphs to extract (for testing)",
    )
    args = parser.parse_args()
    build_corpus(Path(args.out_dir), limit=args.limit)


if __name__ == "__main__":
    main()
