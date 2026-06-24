"""HotpotQA pure-EM reward function for whitebox agents.

Scoring contract (mirrors StepPO ``recipe/hotpotqa/reward_fn.py``):
- ``sample.response`` is the concatenated assistant text (all segments).
- The answer is whatever text is between the **last** ``<answer>...</answer>``
  block — the prompt protocol used by
  ``dressage.recipes.hotpotqa.agent_whitebox``.
- Ground truth is read from ``sample.label``; accepts either a JSON-encoded
  ``{"ground_truth": {"target": [...]}}`` envelope (data prep default) or a
  bare answer string.

Returned reward = 1.0 if the normalized answer exactly matches any normalized
target, else 0.0. No F1 partial credit and no format bonus — those allowed the
policy to collapse onto a "skip search, just guess" shortcut while still
collecting ~0.1–0.3 per sample.
"""

from __future__ import annotations

import json
import re
from typing import Any

import string

from dressage.reward import register_reward


def normalize_answer(s: str) -> str:
    """Lower-case, strip articles/punctuation/whitespace (StepPO convention)."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer(text: str) -> str | None:
    matches = _ANSWER_BLOCK_RE.findall(text or "")
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer or None


def _coerce_targets(label: Any) -> list[str]:
    if label is None:
        return []
    if isinstance(label, str):
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            return [label]
    if isinstance(label, dict):
        gt = label.get("ground_truth", label)
        if isinstance(gt, dict) and "target" in gt:
            target = gt["target"]
            if isinstance(target, list):
                return [str(t) for t in target]
            return [str(target)]
        if isinstance(gt, str):
            return [gt]
    return [str(label)]


@register_reward("hotpotqa")
def hotpotqa(sample: Any, *, args: Any = None, **kwargs: Any) -> float:
    targets = _coerce_targets(getattr(sample, "label", None))
    if not targets:
        return 0.0

    response = getattr(sample, "response", "") or ""
    answer = _extract_answer(response)
    if answer is None:
        return 0.0

    norm_pred = normalize_answer(answer)
    norm_targets = {normalize_answer(t) for t in targets}
    return 1.0 if norm_pred in norm_targets else 0.0
