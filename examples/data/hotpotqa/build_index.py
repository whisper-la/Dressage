#!/usr/bin/env python3
"""Build FAISS index from hpqa_corpus.jsonl using BGE embeddings.

Usage:
    python build_index.py \
        --corpus ./examples/data/hotpotqa/corpus/hpqa_corpus.jsonl \
        --out-dir ./examples/data/hotpotqa/corpus \
        --embedding-model /root/bge-large-en-v1.5 \
        --devices cuda:0,cuda:1 \
        --batch-size 1024

Output:
    <out-dir>/index.bin         — FAISS flat inner-product index
    <out-dir>/hpqa_corpus.npy   — embedding cache (optional, for reuse)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


SCRIPT_DIR = Path(__file__).resolve().parent


def load_corpus_texts(corpus_path: Path) -> list[str]:
    corpus: list[str] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            title = rec.get("title", "")
            text = rec.get("text", "")
            corpus.append(f"{title} {text}".strip())
    return corpus


def parse_devices(devices: str) -> list[str]:
    return [device.strip() for device in devices.split(",") if device.strip()]


def encode_corpus(
    corpus: list[str],
    *,
    embedding_model: str,
    devices: list[str],
    batch_size: int,
) -> np.ndarray:
    if len(devices) <= 1:
        device = devices[0] if devices else "cpu"
        print(f"[build_index] Loading model: {embedding_model}, device={device}")
        model = SentenceTransformer(embedding_model, device=device)
        print(f"[build_index] Encoding corpus (batch_size={batch_size})...")
        return np.asarray(
            model.encode(
                corpus,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

    print(f"[build_index] Loading model: {embedding_model}, devices={','.join(devices)}")
    model = SentenceTransformer(embedding_model, device="cpu")
    pool = model.start_multi_process_pool(target_devices=devices)
    try:
        print(f"[build_index] Encoding corpus on {len(devices)} devices (batch_size={batch_size})...")
        vectors = model.encode_multi_process(
            corpus,
            pool,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    finally:
        model.stop_multi_process_pool(pool)
    return np.asarray(vectors, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for HotpotQA retrieval.")
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(SCRIPT_DIR / "corpus" / "hpqa_corpus.jsonl"),
        help="Path to hpqa_corpus.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(SCRIPT_DIR / "corpus"),
        help="Output directory for index.bin",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="/root/bge-large-en-v1.5",
        help="BGE model path or HuggingFace hub id",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="Embedding device(s). Examples: cuda:0, cuda:0,cuda:1, cpu. Empty = CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse existing hpqa_corpus.npy if present",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "hpqa_corpus.npy"
    index_path = out_dir / "index.bin"

    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    vectors: np.ndarray

    if args.reuse_embeddings and emb_path.exists():
        print(f"[build_index] Reusing existing embeddings: {emb_path}")
        vectors = np.load(str(emb_path)).astype(np.float32)
    else:
        corpus = load_corpus_texts(corpus_path)
        print(f"[build_index] Corpus size: {len(corpus):,} passages")

        vectors = encode_corpus(
            corpus,
            embedding_model=args.embedding_model,
            devices=parse_devices(args.devices),
            batch_size=args.batch_size,
        )
        np.save(str(emb_path), vectors)
        print(f"[build_index] Saved embeddings: {emb_path} ({vectors.shape})")

    dim = vectors.shape[-1]
    print(f"[build_index] Building FAISS index: dim={dim}, n={vectors.shape[0]:,}")
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    print(f"[build_index] Done: {index_path} (ntotal={index.ntotal:,})")


if __name__ == "__main__":
    main()
