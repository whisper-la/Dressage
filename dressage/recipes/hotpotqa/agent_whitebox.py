"""HotpotQA whitebox agent implemented with the shared WhiteboxAgent base."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from dressage.rollout.generate.whitebox_agent import (
    WhiteboxAgent,
    extract_assistant_content,
    extract_finish_reason,
    make_generate,
)

from .tools import (
    HOTPOTQA_INITIAL_USER_TEMPLATE,
    HOTPOTQA_SYSTEM_PROMPT,
    HOTPOTQA_TOOL_SCHEMAS,
    collect_calls,
    do_search,
    extract_answer,
    extract_question,
    extract_structured_tool_calls,
    extract_user_prompt,
    extract_assistant_message,
    load_config,
    truncate_passage,
)

class HotpotQAWhiteboxAgent(WhiteboxAgent):
    name = "hotpotqa_whitebox_agent"
    session_prefix = "hpqa"

    async def rollout(self, sample: Any, sampling_params: dict[str, Any]) -> str:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata

        config = load_config()
        base_user_query = extract_question(extract_user_prompt(sample.prompt))

        seed_evidence = "None"
        if config["force_first_search"]:
            seed_text = await do_search(base_user_query)
            seed_evidence = (
                f"[seed query: {base_user_query}]\n"
                f"{truncate_passage(seed_text, config['passage_max_chars'])}"
                if seed_text
                else "None (initial search returned no passages)"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": HOTPOTQA_INITIAL_USER_TEMPLATE.format(
                    user_query=base_user_query,
                    seed_evidence=seed_evidence,
                ),
            },
        ]

        full_response_parts: list[str] = []
        valid_search_count = 0
        temperature = sampling_params.get("temperature", 1.0)
        max_tokens = int(
            sampling_params.get("max_new_tokens")
            or getattr(self.args, "rollout_max_response_len", 1024)
        )

        try:
            for step_idx in range(config["max_steps"]):
                response = await self.chat(
                    {
                        "model": "proxy-model",
                        "messages": messages,
                        "tools": HOTPOTQA_TOOL_SCHEMAS,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    },
                    turn_id=f"hpqa-step{step_idx}",
                )

                assistant_msg = extract_assistant_message(response)
                assistant_content = extract_assistant_content(response)
                full_response_parts.append(assistant_content)
                messages.append(assistant_msg)

                if extract_answer(assistant_content) is not None:
                    break
                if extract_finish_reason(response) == "length":
                    break

                calls = collect_calls(
                    assistant_content,
                    extract_structured_tool_calls(response),
                    max_parallel=config["max_parallel_calls"],
                )
                if not calls:
                    break

                tasks = [
                    do_search(query) if query is not None else asyncio.sleep(0, result="")
                    for _, query, _ in calls
                ]
                results = await asyncio.gather(*tasks)

                for (tc_id, query, err), text in zip(calls, results):
                    if query is None:
                        tool_content = (
                            f"Tool call could not be parsed: {err}. "
                            'Use {"name": "search", "arguments": {"query": "..."}}.'
                        )
                    else:
                        truncated = truncate_passage(text, config["passage_max_chars"])
                        tool_content = truncated or "(no passages returned)"
                        if text:
                            valid_search_count += 1

                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_content,
                            "tool_call_id": tc_id or f"call_{uuid.uuid4().hex[:12]}",
                        }
                    )
        except Exception as exc:
            metadata["hotpotqa_error"] = str(exc)
            raise

        metadata["valid_search_count"] = valid_search_count
        return "".join(full_response_parts)


generate = make_generate(HotpotQAWhiteboxAgent)
