"""ClaudeAdapter.chat() — OpenAI<->Anthropic translation. CI-safe (client mocked)."""

import json
from types import SimpleNamespace

import pytest

from manyagents.adapters.claude_adapter import ClaudeAdapter


def test_to_anthropic_tools():
    openai_tools = [{
        "type": "function",
        "function": {"name": "run_dr", "description": "do DR",
                     "parameters": {"type": "object", "properties": {"p": {"type": "string"}}}},
    }]
    out = ClaudeAdapter._to_anthropic_tools(openai_tools)
    assert out[0]["name"] == "run_dr"
    assert out[0]["input_schema"]["properties"]["p"]["type"] == "string"
    assert "description" in out[0]


def test_to_anthropic_messages_translates_roles():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "echo", "arguments": '{"text": "x"}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "echo: x"},
    ]
    system, out = ClaudeAdapter._to_anthropic_messages(msgs)
    assert system == "SYS"
    # user, assistant(tool_use), user(tool_result)
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["role"] == "assistant"
    assert out[1]["content"][0]["type"] == "tool_use"
    assert out[1]["content"][0]["input"] == {"text": "x"}
    assert out[2]["role"] == "user"
    assert out[2]["content"][0]["type"] == "tool_result"
    assert out[2]["content"][0]["tool_use_id"] == "t1"


def test_api_key_injection():
    a = ClaudeAdapter(api_key="sk-test")
    assert a.api_key == "sk-test"


@pytest.mark.asyncio
async def test_chat_maps_tool_use_to_normalized_calls(monkeypatch):
    """A Claude response with a tool_use block surfaces as the loop's tool_calls."""
    fake_response = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="let me run that"),
        SimpleNamespace(type="tool_use", id="tu_1", name="run_dr",
                        input={"preset": "umap_2d", "dataset": "swissroll"}),
    ])

    class FakeMessages:
        async def create(self, **kw):
            # tools were translated to Anthropic shape
            assert kw["tools"][0]["name"] == "run_dr"
            assert "temperature" not in kw  # Opus 4.7+ reject it
            return fake_response

    adapter = ClaudeAdapter(api_key="sk-test")
    adapter.client = SimpleNamespace(messages=FakeMessages())

    turn = await adapter.chat(
        [{"role": "user", "content": "embed swissroll"}],
        tools=[{"type": "function", "function": {
            "name": "run_dr", "description": "DR",
            "parameters": {"type": "object", "properties": {}}}}],
    )
    assert turn["content"] == "let me run that"
    assert turn["tool_calls"][0]["name"] == "run_dr"
    assert json.loads(turn["tool_calls"][0]["arguments"]) == {"preset": "umap_2d", "dataset": "swissroll"}
    # assistant message round-trips back to Anthropic tool_use on the next turn
    assert turn["message"]["tool_calls"][0]["function"]["name"] == "run_dr"


@pytest.mark.asyncio
async def test_chat_raises_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = ClaudeAdapter()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await adapter.chat([{"role": "user", "content": "hi"}])
