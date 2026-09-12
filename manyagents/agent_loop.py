"""Provider-agnostic agentic tool-loop.

Send messages and tools, execute the model's tool calls, append results, and
repeat until the model stops calling tools or max_steps is reached. Adapters
must expose chat() with OpenAI-style tool calls. Callers supply Tool definitions;
tool bodies can delegate computation to downstream libraries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from manyagents.tools import Tool, to_openai_schemas

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Final answer, transcript, turn count, stop reason, and executed tool calls."""
    answer: str
    messages: list[dict[str, Any]]   # full transcript (the reasoning/tool trace)
    steps: int
    stopped: str                     # "end_turn" | "max_steps"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


async def run_agent_loop(
    prompt: str,
    *,
    agent: str = "ollama",
    model: Optional[str] = None,
    tools: Optional[list[Tool]] = None,
    system_prompt: Optional[str] = None,
    history: Optional[list[dict[str, Any]]] = None,
    max_steps: int = 8,
    on_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> AgentResult:
    """Run the agentic loop until the model stops calling tools.

    Args:
        prompt: the user's request.
        agent: registry key for an adapter exposing ``chat()`` (ollama/openai/...).
        model: model override (e.g. "qwen3:30b"); adapter default otherwise.
        tools: the tools the model may call. None/empty => plain chat, one turn.
        system_prompt: prepended once when ``history`` is empty.
        history: prior transcript to continue (a previous ``AgentResult.messages``).
            Enables multi-turn REPLs — pass it back each turn; ``result.messages``
            is the updated transcript to feed into the next call.
        max_steps: safety rail — local models loop poorly; bound the turns.
        on_event: optional callback(kind, payload) for UI ("assistant"/"tool_call"/"tool_result").
    """
    from manyagents.adapters import ADAPTER_REGISTRY

    if agent not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown agent {agent!r}. Choose from: {sorted(ADAPTER_REGISTRY)}")
    adapter = ADAPTER_REGISTRY[agent]()
    if not hasattr(adapter, "chat"):
        raise TypeError(f"Adapter {agent!r} does not support the agentic loop (no chat())")

    tools = list(tools or [])
    registry = {t.name: t for t in tools}
    schemas = to_openai_schemas(tools) if tools else None

    messages: list[dict[str, Any]] = list(history) if history else []
    if system_prompt and not messages:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    all_tool_calls: list[dict[str, Any]] = []

    def _emit(kind: str, payload: dict[str, Any]) -> None:
        if on_event:
            on_event(kind, payload)

    for step in range(1, max_steps + 1):
        turn = await adapter.chat(messages, tools=schemas, model=model)
        messages.append(turn["message"])

        if turn["content"]:
            _emit("assistant", {"text": turn["content"]})

        calls = turn["tool_calls"]
        if not calls:
            return AgentResult(
                answer=turn["content"], messages=messages, steps=step,
                stopped="end_turn", tool_calls=all_tool_calls,
            )

        for call in calls:
            all_tool_calls.append(call)
            try:
                args = json.loads(call["arguments"]) if call["arguments"] else {}
            except json.JSONDecodeError as e:
                args, parse_err = {}, str(e)
            else:
                parse_err = None

            _emit("tool_call", {"name": call["name"], "arguments": args})

            tool = registry.get(call["name"])
            if tool is None:
                result = f"Error: unknown tool {call['name']!r}"
            elif parse_err is not None:
                result = f"Error: could not parse arguments ({parse_err})"
            else:
                try:
                    result = await tool.invoke(args)
                except Exception as e:  # tool failures are fed back, not raised
                    log.exception("tool %s failed", call["name"])
                    result = f"Error: tool {call['name']!r} raised: {e}"

            _emit("tool_result", {"name": call["name"], "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

    return AgentResult(
        answer="(stopped: reached max_steps without a final answer)",
        messages=messages, steps=max_steps, stopped="max_steps",
        tool_calls=all_tool_calls,
    )
