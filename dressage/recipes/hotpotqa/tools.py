"""HotpotQA prompt, parsing, and local search helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DEFAULT_CORPUS_DIR = os.environ.get(
    "HOTPOTQA_CORPUS_DIR",
    str(Path(__file__).resolve().parents[3] / "examples" / "data" / "hotpotqa" / "corpus"),
)
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "HOTPOTQA_EMBEDDING_MODEL", "/root/bge-large-en-v1.5"
)
DEFAULT_EMBEDDING_DEVICE = os.environ.get("HOTPOTQA_EMBEDDING_DEVICE", "cpu")
DEFAULT_TOPK = int(os.environ.get("HOTPOTQA_TOPK", "5"))


HOTPOTQA_SYSTEM_PROMPT = (
    "You are a research agent. Your goal is to answer the User Query using "
    "Wikipedia search evidence.\n\n"
    "Each turn, briefly reason inside <analysis>...</analysis>, then either "
    "(a) issue one or more `search` tool calls in parallel for new evidence, "
    "or (b) emit the final short answer inside <answer>...</answer> tags. "
    "Never repeat a query that already appears in the conversation. Once you "
    "can answer from accumulated passages, stop searching and emit <answer>."
)

HOTPOTQA_INITIAL_USER_TEMPLATE = """### User Query
{user_query}

### Seed Evidence
{seed_evidence}

### Instructions
- Reason briefly inside `<analysis>...</analysis>`.
- Call the `search` tool one or more times in parallel when new evidence is needed.
- When you can answer from prior passages, output `<answer>short answer</answer>` (no extra prose, no further tool calls).

### Output Format
<analysis>
[Your analysis...]
</analysis>
<tool_call>
{{"name": "search", "arguments": {{"query": "..."}}}}
</tool_call>
"""

HOTPOTQA_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search Wikipedia for passages relevant to the user question. "
                "Use natural-language or keyword queries; must differ from "
                "any prior query in the conversation when possible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A single search query (natural language or "
                            "keywords). Must differ from prior queries when "
                            "seeking new evidence."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]

_QUESTION_MARKER_RE = re.compile(r"Question:\s*(.+)\Z", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SEARCH_TOOL: HotpotQALocalSearch | None = None
_SEARCH_TOOL_LOCK = threading.Lock()


def load_config() -> dict[str, Any]:
    return {
        "max_steps": int(os.environ.get("HOTPOTQA_MAX_STEPS", "5")),
        "max_parallel_calls": int(os.environ.get("HOTPOTQA_MAX_PARALLEL_CALLS", "4")),
        "force_first_search": os.environ.get("HOTPOTQA_FORCE_FIRST_SEARCH", "1").lower()
        in {"1", "true", "yes"},
        "passage_max_chars": int(os.environ.get("HOTPOTQA_PASSAGE_MAX_CHARS", "1200")),
    }


def resolve_device(requested: str) -> str:
    import torch

    dev = (requested or "cpu").strip().lower()
    if dev == "cpu":
        return "cpu"
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "HOTPOTQA_EMBEDDING_DEVICE=%r but CUDA unavailable; falling back to cpu",
                requested,
            )
            return "cpu"
        if ":" in dev:
            try:
                idx = int(dev.split(":")[-1])
                if idx >= torch.cuda.device_count():
                    logger.warning(
                        "cuda:%d invalid (device_count=%d); using cpu",
                        idx,
                        torch.cuda.device_count(),
                    )
                    return "cpu"
            except ValueError:
                pass
    return dev


class HotpotQALocalSearch:
    _lock = threading.RLock()
    _shared_key: str | None = None
    _shared_index: Any = None
    _shared_corpus: list[str] | None = None
    _shared_model: Any = None

    def __init__(
        self,
        corpus_dir: str | None = None,
        embedding_model: str | None = None,
        embedding_device: str | None = None,
        topk: int | None = None,
    ) -> None:
        self.corpus_dir = Path(corpus_dir or DEFAULT_CORPUS_DIR)
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.embedding_device = resolve_device(embedding_device or DEFAULT_EMBEDDING_DEVICE)
        self.topk = topk if topk is not None else DEFAULT_TOPK
        self._index: Any = None
        self._corpus: list[str] = []
        self._model: Any = None
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        cache_key = f"{self.corpus_dir}|{self.embedding_device}|{self.embedding_model}"
        with self.__class__._lock:
            if (
                self.__class__._shared_key != cache_key
                or self.__class__._shared_index is None
                or self.__class__._shared_corpus is None
                or self.__class__._shared_model is None
            ):
                import faiss
                from sentence_transformers import SentenceTransformer

                index_path = self.corpus_dir / "index.bin"
                corpus_path = self.corpus_dir / "hpqa_corpus.jsonl"
                if not index_path.exists():
                    raise FileNotFoundError(f"FAISS index not found: {index_path}")
                if not corpus_path.exists():
                    raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

                logger.info("Loading FAISS index from %s", index_path)
                index = faiss.read_index(str(index_path))
                logger.info("Loading corpus from %s", corpus_path)
                corpus: list[str] = []
                with corpus_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        title = str(rec.get("title", ""))
                        text = str(rec.get("text", ""))
                        corpus.append(f"{title}\n{text}".strip())

                logger.info(
                    "Loading BGE model=%s device=%s",
                    self.embedding_model,
                    self.embedding_device,
                )
                model = SentenceTransformer(self.embedding_model, device=self.embedding_device)
                self.__class__._shared_key = cache_key
                self.__class__._shared_index = index
                self.__class__._shared_corpus = corpus
                self.__class__._shared_model = model

            self._index = self.__class__._shared_index
            self._corpus = self.__class__._shared_corpus or []
            self._model = self.__class__._shared_model

    def _encode_queries(self, queries: list[str]) -> Any:
        import numpy as np

        prefixed = [QUERY_INSTRUCTION + q for q in queries]
        with self.__class__._lock:
            out = self._model.encode(prefixed, normalize_embeddings=True)
        arr = np.asarray(out, dtype=np.float32)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr

    def _format_results(self, ids: Any) -> str:
        parts: list[str] = []
        for rank, idx in enumerate(ids):
            idx = int(idx)
            if idx < 0 or idx >= len(self._corpus):
                continue
            entry = self._corpus[idx]
            lines = entry.split("\n", 1)
            title = lines[0] if lines else ""
            text = lines[1] if len(lines) > 1 else ""
            parts.append(f"Doc {rank + 1}(Title: {title}) {text}")
        return "\n".join(parts)

    def execute(self, query: str) -> str:
        try:
            embeddings = self._encode_queries([query])
            _, ids = self._index.search(embeddings, self.topk)
            return self._format_results(ids[0])
        except Exception as e:
            logger.warning("Local search failed for query=%r: %s", query[:50], e)
            return ""


def get_search_tool() -> HotpotQALocalSearch:
    global _SEARCH_TOOL
    if _SEARCH_TOOL is None:
        with _SEARCH_TOOL_LOCK:
            if _SEARCH_TOOL is None:
                _SEARCH_TOOL = HotpotQALocalSearch()
    return _SEARCH_TOOL


async def do_search(query: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_search_tool().execute, query)


def extract_user_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return "\n".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return str(prompt) if prompt is not None else ""


def extract_question(prompt_text: str) -> str:
    text = prompt_text.strip()
    match = _QUESTION_MARKER_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def truncate_passage(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def parse_json_loose(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    if len(raw) > 8192 or "__" in raw:
        return None
    try:
        import ast

        return ast.literal_eval(raw)
    except Exception:
        return None


def extract_structured_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    tcs = message.get("tool_calls")
    if not isinstance(tcs, list):
        return []
    return [tc for tc in tcs if isinstance(tc, dict)]


def extract_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if choices:
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            return msg
    return {"role": "assistant", "content": ""}


def extract_query_from_call(name: Any, arguments: Any) -> tuple[str | None, str | None]:
    if name != "search":
        return None, f"unknown tool {name!r}"
    if isinstance(arguments, str):
        arguments = parse_json_loose(arguments)
    if not isinstance(arguments, dict):
        return None, "arguments not a JSON object"
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return None, "missing query"
    return query.strip(), None


def collect_calls(
    content: str,
    structured: list[dict[str, Any]],
    *,
    max_parallel: int,
) -> list[tuple[str | None, str | None, str]]:
    out: list[tuple[str | None, str | None, str]] = []
    for tc in structured:
        if len(out) >= max_parallel:
            break
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        query, err = extract_query_from_call(fn.get("name"), fn.get("arguments"))
        out.append((tc.get("id"), query, err or ""))
    if out:
        return out
    for raw in _TOOL_CALL_BLOCK_RE.findall(content or ""):
        if len(out) >= max_parallel:
            break
        obj = parse_json_loose(raw)
        if not isinstance(obj, dict):
            out.append((None, None, "could not parse <tool_call> JSON"))
            continue
        query, err = extract_query_from_call(obj.get("name"), obj.get("arguments"))
        out.append((None, query, err or ""))
    return out


def extract_answer(content: str) -> str | None:
    matches = _ANSWER_BLOCK_RE.findall(content or "")
    if not matches:
        return None
    return matches[-1].strip() or None
