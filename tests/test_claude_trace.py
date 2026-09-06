# tests/test_claude_trace.py
"""Tests for ClaudeAdapter trace building."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from manyagents.adapters.claude_adapter import ClaudeAdapter
from manyagents.schemas.reasoning import ModelBackend, StepKind


@pytest.fixture
def adapter():
    a = ClaudeAdapter()
    a.api_key = "test-key"
    return a


@pytest.mark.asyncio
async def test_claude_adapter_build_trace(adapter):
    """ClaudeAdapter builds ReasoningTrace when build_trace=True."""
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(type="thinking", thinking="Let me think..."),
        MagicMock(type="text", text="The answer is 4."),
    ]
    mock_response.usage = MagicMock(input_tokens=50, output_tokens=20)
    mock_response.stop_reason = "end_turn"

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    adapter.client = mock_client

    result = await adapter.run(
        task_config={
            "prompt": "What is 2+2?",
            "build_trace": True,
            "dataset": "gsm8k",
            "task_id": "test_001",
        },
        input_files={},
    )

    assert result["success"]
    assert "trace" in result.get("output_files", {})
    trace_path = result["output_files"]["trace"]
    from pathlib import Path
    from manyagents.schemas.reasoning import ReasoningTrace
    trace = ReasoningTrace.from_json(Path(trace_path).read_text())
    assert trace.model.backend == ModelBackend.ANTHROPIC
    assert len(trace.steps) == 2
    assert trace.steps[0].kind == StepKind.THINKING
    assert trace.steps[1].kind == StepKind.OUTPUT
    assert trace.input_tokens == 50
    assert trace.has_tensors is False


@pytest.mark.asyncio
async def test_claude_adapter_no_trace_by_default(adapter):
    """ClaudeAdapter does NOT build trace when build_trace is not set."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Answer")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_response.stop_reason = "end_turn"

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    adapter.client = mock_client

    result = await adapter.run(
        task_config={"prompt": "Hello"},
        input_files={},
    )

    assert result["success"]
    assert "trace" not in result.get("output_files", {})
