"""ALFWorld whitebox agent implemented with the shared WhiteboxAgent base."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from dressage.rollout.generate.whitebox_agent import (
    WhiteboxAgent,
    extract_assistant_content,
    extract_finish_reason,
    make_generate,
)

from .tools import (
    ALFWORLD_INITIAL_USER_TEMPLATE,
    ALFWORLD_SYSTEM_PROMPT,
    ALFWORLD_TOOL_SCHEMAS,
    extract_assistant_message,
    extract_prompt_fallback,
    extract_structured_tool_calls,
    extract_task_text,
    first_command,
    format_admissible,
    format_invalid_response,
    format_tool_response,
    load_config,
    make_env,
)

class ALFWorldWhiteboxAgent(WhiteboxAgent):
    name = "alfworld_whitebox_agent"
    session_prefix = "alf"

    async def rollout(self, sample: Any, sampling_params: dict[str, Any]) -> str:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata

        config = load_config()
        game_file = metadata.get("game_file")
        if not game_file:
            metadata["task_success"] = False
            metadata["num_steps"] = 0
            metadata["invalid_action_count"] = 0
            raise ValueError("alfworld sample.metadata missing game_file")
        game_path = Path(str(game_file))
        if not game_path.is_absolute():
            game_path = Path(__file__).resolve().parents[3] / game_path

        temperature = sampling_params.get("temperature", 1.0)
        max_tokens = int(
            sampling_params.get("max_new_tokens")
            or getattr(self.args, "rollout_max_response_len", 1024)
        )

        env = make_env(str(game_path), config["max_episode_steps"])
        full_response_parts: list[str] = []
        task_success = False
        invalid_action_count = 0
        executed_steps = 0

        try:
            obs, infos = env.reset()
            task_text = extract_task_text(obs, extract_prompt_fallback(sample.prompt))
            admissible = list(infos.get("admissible_commands", []) or [])

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ALFWORLD_INITIAL_USER_TEMPLATE.format(
                        task_text=task_text,
                        observation=(obs or "").strip()[:2000],
                        admissible_commands=format_admissible(admissible),
                    ),
                },
            ]

            for step_idx in range(config["max_steps"]):
                response = await self.chat(
                    {
                        "model": "proxy-model",
                        "messages": messages,
                        "tools": ALFWORLD_TOOL_SCHEMAS,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    },
                    turn_id=f"alf-step{step_idx}",
                )

                assistant_msg = extract_assistant_message(response)
                assistant_content = extract_assistant_content(response)
                full_response_parts.append(assistant_content)
                messages.append(assistant_msg)

                finish = extract_finish_reason(response)
                tc_id, command, reason = first_command(
                    assistant_content,
                    extract_structured_tool_calls(response),
                )
                tool_call_id = tc_id or f"call_{uuid.uuid4().hex[:12]}"

                if command is None:
                    invalid_action_count += 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": format_invalid_response(
                                obs, admissible, reason or "unknown"
                            ),
                        }
                    )
                    if finish == "length":
                        break
                    continue

                if command not in admissible:
                    invalid_action_count += 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": format_invalid_response(
                                obs,
                                admissible,
                                f"command {command!r} is not in admissible commands",
                            ),
                        }
                    )
                    if finish == "length":
                        break
                    continue

                obs, _reward, done, infos = env.step(command)
                executed_steps += 1
                admissible = list(infos.get("admissible_commands", []) or [])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": format_tool_response(obs, admissible),
                    }
                )

                if infos.get("won", False):
                    task_success = True
                    break
                if done:
                    break
                if finish == "length":
                    break
        except Exception as exc:
            metadata["alfworld_error"] = str(exc)
            raise
        finally:
            try:
                env.close()
            except Exception:
                pass

        metadata["task_success"] = task_success
        metadata["num_steps"] = executed_steps
        metadata["invalid_action_count"] = invalid_action_count
        return "".join(full_response_parts)


generate = make_generate(ALFWorldWhiteboxAgent)
