"""OpenAI-compatible base_url support + the ollama local adapter.

CI-safe: AsyncOpenAI is mocked, so no network or running server is needed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from manyagents.adapters import ADAPTER_REGISTRY
from manyagents.adapters.openai_adapter import OpenAIAdapter, OllamaAdapter


def test_ollama_registered():
    assert ADAPTER_REGISTRY["ollama"] is OllamaAdapter


def test_openai_adapter_base_url_passthrough():
    with patch("manyagents.adapters.openai_adapter.AsyncOpenAI") as mock_client:
        OpenAIAdapter(base_url="http://localhost:11434/v1", api_key="x")
    _, kwargs = mock_client.call_args
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["api_key"] == "x"


def test_openai_adapter_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example:1234/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("manyagents.adapters.openai_adapter.AsyncOpenAI") as mock_client:
        OpenAIAdapter()
    _, kwargs = mock_client.call_args
    assert kwargs["base_url"] == "http://example:1234/v1"
    # dummy key filled in when a base_url is set without an explicit key
    assert kwargs["api_key"] == "local"


def test_ollama_adapter_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    with patch("manyagents.adapters.openai_adapter.AsyncOpenAI") as mock_client:
        adapter = OllamaAdapter()
    assert adapter.name == "ollama"
    assert adapter.base_url == "http://localhost:11434/v1"
    _, kwargs = mock_client.call_args
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["api_key"] == "ollama"


def test_ollama_adapter_respects_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    with patch("manyagents.adapters.openai_adapter.AsyncOpenAI"):
        adapter = OllamaAdapter()
    assert adapter.base_url == "http://gpu-box:11434/v1"


def _recording_adapter(recorded):
    """An OllamaAdapter whose client records the create() kwargs and answers "ok"."""
    class FakeCompletions:
        async def create(self, **kw):
            recorded.append(kw)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None))])

    with patch("manyagents.adapters.openai_adapter.AsyncOpenAI"):
        adapter = OllamaAdapter()
    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    return adapter


@pytest.mark.asyncio
async def test_chat_forwards_reasoning_effort():
    """A thinking model is told not to think, or it answers with empty content."""
    recorded = []
    adapter = _recording_adapter(recorded)

    turn = await adapter.chat(
        [{"role": "user", "content": "should I trust this?"}],
        model="qwen3:8b",
        reasoning_effort="none",
    )
    assert turn["content"] == "ok"
    assert recorded[0]["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_chat_omits_reasoning_effort_when_unset():
    """Default None sends the same request body callers sent before the parameter."""
    recorded = []
    adapter = _recording_adapter(recorded)

    await adapter.chat([{"role": "user", "content": "hi"}], model="qwen3:8b")
    assert "reasoning_effort" not in recorded[0]
