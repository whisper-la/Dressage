"""claw_grader reward: parse overall_score from the run_grader execute_cmds record."""

from __future__ import annotations

import json
import logging
from typing import Any

from dressage.reward import register_reward

logger = logging.getLogger(__name__)


def _metadata(sample: Any) -> dict:
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _parse_grader_overall_score(stdout: str) -> float:
    """Parse ``overall_score`` from grader stdout; 0.0 on contract violation."""
    if not stdout or not stdout.strip():
        return 0.0
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        logger.warning("grader stdout is not valid JSON; returning 0.0")
        return 0.0
    if not isinstance(data, dict) or "overall_score" not in data:
        logger.warning("grader stdout missing 'overall_score' field; returning 0.0")
        return 0.0
    try:
        return float(data["overall_score"])
    except (TypeError, ValueError):
        return 0.0


@register_reward("claw_grader")
def claw_grader(sample: Any, *, args: Any | None = None, **_: Any) -> float:
    """Extract the grader's overall_score from the run_grader execute_cmds record."""
    del args
    for cmd_record in _metadata(sample).get("execute_cmds", []):
        if cmd_record.get("name") != "run_grader":
            continue
        result = cmd_record.get("cmd_result")
        if not isinstance(result, dict):
            return 0.0
        if result.get("stdout_truncated"):
            logger.warning("run_grader stdout was truncated; returning 0.0")
            return 0.0
        return _parse_grader_overall_score(result.get("stdout", ""))
    return 0.0
