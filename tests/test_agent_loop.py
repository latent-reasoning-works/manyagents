"""Agentic tool-loop — CI-safe (adapter.chat mocked, no model/network)."""

import pytest

from manyagents.agent_loop import run_agent_loop
from manyagents.tools import MOCK_TOOL, Tool, to_openai_schemas


def test_tool_to_openai_schema():
    s = MOCK_TOOL.to_openai_schema()
    assert s["type"] == "function"
    assert s["function"]["name"] == "echo"
    assert "text" in s["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_loop_executes_tool_then_answers(monkeypatch):
    """Turn 1 emits a tool call; turn 2 (after the tool result) answers."""
    turns = [
        # turn 1: model calls echo
        {
            "message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "echo", "arguments": '{"text": "hi"}'}}
            ]},
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": '{"text": "hi"}'}],
            "content": "",
        },
        # turn 2: model answers, no more tool calls
        {"message": {"role": "assistant", "content": "Tool said: echo: hi"},
         "tool_calls": [], "content": "Tool said: echo: hi"},
    ]

    class FakeAdapter:
        async def chat(self, messages, *, tools=None, model=None, **kw):
            return turns.pop(0)

    monkeypatch.setattr(
        "manyagents.adapters.ADAPTER_REGISTRY", {"fake": FakeAdapter}, raising=False,
    )

    result = await run_agent_loop("echo hi", agent="fake", tools=[MOCK_TOOL], max_steps=5)
    assert result.stopped == "end_turn"
    assert result.steps == 2
    assert [c["name"] for c in result.tool_calls] == ["echo"]
    assert "echo: hi" in result.answer
    # transcript carries the tool result back to the model
    assert any(m.get("role") == "tool" and "echo: hi" in m["content"] for m in result.messages)


@pytest.mark.asyncio
async def test_loop_respects_max_steps(monkeypatch):
    """A model that never stops calling tools is bounded by max_steps."""
    looping_turn = {
        "message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function",
             "function": {"name": "echo", "arguments": '{"text": "x"}'}}
        ]},
        "tool_calls": [{"id": "c", "name": "echo", "arguments": '{"text": "x"}'}],
        "content": "",
    }

    class LoopAdapter:
        async def chat(self, messages, *, tools=None, model=None, **kw):
            return looping_turn

    monkeypatch.setattr(
        "manyagents.adapters.ADAPTER_REGISTRY", {"loop": LoopAdapter}, raising=False,
    )

    result = await run_agent_loop("go", agent="loop", tools=[MOCK_TOOL], max_steps=3)
    assert result.stopped == "max_steps"
    assert result.steps == 3


@pytest.mark.asyncio
async def test_unknown_tool_is_fed_back_not_raised(monkeypatch):
    turns = [
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "nonexistent", "arguments": "{}"}}]},
         "tool_calls": [{"id": "c1", "name": "nonexistent", "arguments": "{}"}],
         "content": ""},
        {"message": {"role": "assistant", "content": "done"},
         "tool_calls": [], "content": "done"},
    ]

    class FakeAdapter:
        async def chat(self, messages, *, tools=None, model=None, **kw):
            return turns.pop(0)

    monkeypatch.setattr(
        "manyagents.adapters.ADAPTER_REGISTRY", {"fake": FakeAdapter}, raising=False,
    )

    result = await run_agent_loop("x", agent="fake", tools=[MOCK_TOOL])
    assert result.stopped == "end_turn"
    assert any("unknown tool" in m.get("content", "") for m in result.messages if m.get("role") == "tool")


def test_to_openai_schemas_list():
    extra = Tool("noop", "noop", {"type": "object", "properties": {}}, run=lambda: "ok")
    schemas = to_openai_schemas([MOCK_TOOL, extra])
    assert [s["function"]["name"] for s in schemas] == ["echo", "noop"]
