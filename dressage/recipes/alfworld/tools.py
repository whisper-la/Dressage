"""ALFWorld prompt, parsing, and TextWorld helpers."""

from __future__ import annotations

import json
import os
import re
from typing import Any

ALFWORLD_SYSTEM_PROMPT = (
    "You are acting in ALFWorld TextWorld. Each turn, choose exactly one "
    "command from the admissible commands provided in the latest tool "
    "response. Reason briefly inside <think></think> if useful, then call "
    "the `env_step` tool with that exact command. Follow ALFWorld TextWorld "
    "command style such as `go to dresser 1`, `take mug 1 from cabinet 3`, "
    "`use desklamp 1`. Do not output a final natural-language answer."
)

ALFWORLD_INITIAL_USER_TEMPLATE = """### Task
{task_text}

### Initial Observation
{observation}

### Admissible Commands
{admissible_commands}

### Output Format
<think>
[Your brief reasoning about the next command.]
</think>
<tool_call>
{{"name": "env_step", "arguments": {{"command": "[one admissible command]"}}}}
</tool_call>
"""

ALFWORLD_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "env_step",
            "description": (
                "Execute one ALFWorld TextWorld command and return the next "
                "official observation plus the new admissible commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "A single ALFWorld TextWorld command such as "
                            "`go to dresser 1`, `open cabinet 3`, "
                            "`take mug 1 from cabinet 3`, `use desklamp 1`. "
                            "Must exactly match one currently admissible command."
                        ),
                    }
                },
                "required": ["command"],
            },
        },
    }
]

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def load_config() -> dict[str, Any]:
    return {
        "max_steps": int(os.environ.get("ALFWORLD_MAX_STEPS", "50")),
        "max_episode_steps": int(os.environ.get("ALFWORLD_MAX_EPISODE_STEPS", "50")),
    }


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


def first_command(
    content: str,
    structured: list[dict[str, Any]],
) -> tuple[str | None, str | None, str | None]:
    for tc in structured:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name")
        if name != "env_step":
            return tc.get("id"), None, f"expected env_step, got {name!r}"
        args = fn.get("arguments")
        if isinstance(args, str):
            args = parse_json_loose(args)
        if not isinstance(args, dict):
            return tc.get("id"), None, "arguments not a JSON object"
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return tc.get("id"), None, "missing command argument"
        return tc.get("id"), command.strip(), None

    blocks = _TOOL_CALL_BLOCK_RE.findall(content or "")
    if not blocks:
        return None, None, "missing <tool_call> block"
    for raw in blocks:
        obj = parse_json_loose(raw)
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if name != "env_step":
            return None, None, f"expected env_step, got {name!r}"
        args = obj.get("arguments")
        if isinstance(args, str):
            args = parse_json_loose(args)
        if not isinstance(args, dict):
            return None, None, "arguments not a JSON object"
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return None, None, "missing command argument"
        return None, command.strip(), None
    return None, None, "could not parse <tool_call> JSON"


def make_env(game_file: str, max_episode_steps: int):
    import textworld
    import textworld.gym

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        description=True,
        inventory=True,
    )
    env_id = textworld.gym.register_game(
        game_file,
        request_infos=request_infos,
        max_episode_steps=max_episode_steps,
    )
    return textworld.gym.make(env_id)


def format_admissible(commands: list[str]) -> str:
    if not commands:
        return "None"
    return "\n".join(f"- {cmd}" for cmd in commands if cmd != "help")


def extract_task_text(observation: str, fallback: str) -> str:
    marker = "Your task is to:"
    if observation and marker in observation:
        task = observation.split(marker, 1)[1].strip()
        task = task.split("\n", 1)[0].strip()
        return f"{marker} {task}"
    fallback = (fallback or "").strip()
    if fallback:
        return f"{marker} {fallback}"
    return f"{marker} Unknown."


def extract_prompt_fallback(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
    return ""


def format_tool_response(observation: str, admissible: list[str]) -> str:
    obs = (observation or "").strip()[:2000]
    return (
        f"### Observation\n{obs}\n\n"
        f"### Admissible Commands\n{format_admissible(admissible)}"
    )


def format_invalid_response(
    previous_obs: str,
    admissible: list[str],
    reason: str,
) -> str:
    return (
        "Invalid tool call. Call `env_step` with JSON arguments like "
        '{"command": "<one admissible command>"}. '
        f"Reason: {reason}\n\n"
        "Environment state did not change.\n\n"
        f"{format_tool_response(previous_obs, admissible)}"
    )
