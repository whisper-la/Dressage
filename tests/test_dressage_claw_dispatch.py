"""Tests for the dressage_claw blackbox dispatch helpers and reward."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any

import pytest

from dressage.recipes.dressage_claw.dispatch import (
    _build_agent_messages_json,
    _build_fallback_messages_b64,
)
from dressage.recipes.dressage_claw.reward import _parse_grader_overall_score


class TestParseGraderOverallScore:
    def test_valid_stdout(self):
        assert _parse_grader_overall_score('{"overall_score": 0.75}') == 0.75

    def test_int_score(self):
        assert _parse_grader_overall_score('{"overall_score": 1}') == 1.0

    def test_empty_stdout_returns_zero(self):
        assert _parse_grader_overall_score("") == 0.0

    def test_blank_stdout_returns_zero(self):
        assert _parse_grader_overall_score("   \n  ") == 0.0

    def test_invalid_json_returns_zero(self):
        assert _parse_grader_overall_score("not json") == 0.0

    def test_missing_field_returns_zero(self):
        assert _parse_grader_overall_score('{"score": 0.5}') == 0.0

    def test_non_dict_returns_zero(self):
        assert _parse_grader_overall_score("[1, 2, 3]") == 0.0

    def test_non_numeric_score_returns_zero(self):
        assert _parse_grader_overall_score('{"overall_score": "bad"}') == 0.0


class TestBuildAgentMessagesJson:
    def test_uses_last_segment_messages(self):
        trajectory = {
            "data": [
                {"messages": [{"role": "user", "content": "old"}]},
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "final answer"},
                    ]
                },
            ]
        }
        json_str, response = _build_agent_messages_json(
            trajectory, prompt="hi", response=""
        )
        payload = json.loads(json_str)
        assert payload["prompt"] == "hi"
        assert payload["response"] == "final answer"
        assert response == "final answer"
        assert payload["messages"][-1]["content"] == "final answer"

    def test_keeps_existing_response(self):
        trajectory = {
            "data": [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "from trace"},
                    ]
                }
            ]
        }
        _, response = _build_agent_messages_json(
            trajectory, prompt="hi", response="original"
        )
        assert response == "original"

    def test_skips_tool_call_messages_for_response(self):
        trajectory = {
            "data": [
                {
                    "messages": [
                        {"role": "assistant", "content": "real answer"},
                        {
                            "role": "assistant",
                            "content": "tool call msg",
                            "tool_calls": [{"id": "1"}],
                        },
                    ]
                }
            ]
        }
        _, response = _build_agent_messages_json(trajectory, prompt="p", response="")
        assert response == "real answer"

    def test_no_segments_raises(self):
        with pytest.raises(ValueError, match="no segments"):
            _build_agent_messages_json({"data": []}, prompt="p", response="")

    def test_no_messages_raises(self):
        with pytest.raises(ValueError, match="no messages"):
            _build_agent_messages_json({"data": [{"messages": []}]}, prompt="p", response="")


class TestBuildFallbackMessagesB64:
    def test_roundtrip(self):
        encoded = _build_fallback_messages_b64("hello", "world")
        payload = json.loads(base64.b64decode(encoded).decode("utf-8"))
        assert payload["prompt"] == "hello"
        assert payload["response"] == "world"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]


class TestClawGraderReward:
    @pytest.fixture(autouse=True)
    def _load_reward(self):
        from dressage.reward.registry import _REWARD_REGISTRY

        saved = dict(_REWARD_REGISTRY)
        import dressage.recipes.dressage_claw.reward  # noqa: F401

        yield
        _REWARD_REGISTRY.clear()
        _REWARD_REGISTRY.update(saved)

    def _sample(self, metadata: dict[str, Any]) -> Any:
        return SimpleNamespace(metadata=metadata)

    def test_registered(self):
        from dressage.reward.registry import get_reward_fn

        assert get_reward_fn("claw_grader") is not None

    def test_reads_score_from_run_grader_record(self):
        from dressage.recipes.dressage_claw.reward import claw_grader

        sample = self._sample({
            "execute_cmds": [
                {"name": "stop_services", "cmd_result": {"stdout": ""}},
                {"name": "run_grader", "cmd_result": {"stdout": '{"overall_score": 0.8}'}},
            ]
        })
        assert claw_grader(sample) == 0.8

    def test_missing_run_grader_returns_zero(self):
        from dressage.recipes.dressage_claw.reward import claw_grader

        assert claw_grader(self._sample({"execute_cmds": []})) == 0.0

    def test_missing_execute_cmds_returns_zero(self):
        from dressage.recipes.dressage_claw.reward import claw_grader

        assert claw_grader(self._sample({})) == 0.0

    def test_non_dict_metadata_returns_zero(self):
        from dressage.recipes.dressage_claw.reward import claw_grader

        assert claw_grader(SimpleNamespace(metadata=None)) == 0.0

    def test_bad_stdout_returns_zero(self):
        from dressage.recipes.dressage_claw.reward import claw_grader

        sample = self._sample({
            "execute_cmds": [
                {"name": "run_grader", "cmd_result": {"stdout": "boom"}},
            ]
        })
        assert claw_grader(sample) == 0.0

    def test_missing_cmd_result_returns_zero(self):
        from dressage.recipes.dressage_claw.reward import claw_grader

        sample = self._sample({"execute_cmds": [{"name": "run_grader"}]})
        assert claw_grader(sample) == 0.0

    def test_truncated_stdout_returns_zero(self):
        from dressage.recipes.dressage_claw.reward import claw_grader

        sample = self._sample({
            "execute_cmds": [
                {
                    "name": "run_grader",
                    "cmd_result": {"stdout": '{"overall_sc', "stdout_truncated": True},
                },
            ]
        })
        assert claw_grader(sample) == 0.0
