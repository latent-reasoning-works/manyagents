# tests/schemas/test_reasoning.py
"""Tests for ReasoningTrace schema and TraceStore."""

import json
import tempfile
from pathlib import Path

from manyagents.schemas.reasoning import (
    ModelBackend,
    ModelInfo,
    StepKind,
    TaskInfo,
    ReasoningStep,
    ReasoningTrace,
    TraceStore,
    steps_from_anthropic_response,
    trace_from_anthropic,
)


def _make_trace(**overrides) -> ReasoningTrace:
    """Helper: build a minimal valid trace."""
    defaults = dict(
        model=ModelInfo("olmo-7b", ModelBackend.LOCAL, path="/network/weights/olmo/OLMo-7B-Twin-2T"),
        task=TaskInfo("gsm8k", "train_001", "What is 2+2?", expected_answer="4", domain="math"),
        steps=[
            ReasoningStep(0, "I need to add 2 and 2.", StepKind.THINKING, token_count=8),
            ReasoningStep(1, "The answer is 4.", StepKind.OUTPUT, token_count=5),
        ],
        response_text="The answer is 4.",
        success=True,
        judge="exact_match",
        answer_extracted="4",
        input_tokens=10,
        output_tokens=13,
        total_tokens=23,
        duration_ms=450,
    )
    defaults.update(overrides)
    return ReasoningTrace(**defaults)


def test_trace_json_roundtrip():
    """ReasoningTrace round-trips through JSON."""
    trace = _make_trace()
    j = trace.to_json()
    loaded = ReasoningTrace.from_json(j)

    assert loaded.trace_id == trace.trace_id
    assert loaded.model.name == "olmo-7b"
    assert loaded.model.backend == ModelBackend.LOCAL
    assert loaded.model.path == "/network/weights/olmo/OLMo-7B-Twin-2T"
    assert loaded.steps[0].kind == StepKind.THINKING
    assert loaded.steps[1].kind == StepKind.OUTPUT
    assert loaded.success is True
    assert loaded.judge == "exact_match"
    assert loaded.task.domain == "math"
    assert loaded.duration_ms == 450
    assert loaded.total_tokens == 23


def test_trace_dict_roundtrip():
    """ReasoningTrace round-trips through dict."""
    trace = _make_trace()
    d = trace.to_dict()

    # Dict should have string enum values
    assert d["model"]["backend"] == "local"
    assert d["steps"][0]["kind"] == "thinking"
    # Outcome fields are flat
    assert d["success"] is True
    assert d["judge"] == "exact_match"

    loaded = ReasoningTrace.from_dict(d)
    assert loaded.model.backend == ModelBackend.LOCAL
    assert loaded.steps[0].kind == StepKind.THINKING


def test_trace_defaults():
    """ReasoningTrace has sensible defaults."""
    trace = ReasoningTrace()
    assert trace.model.name == "unknown"
    assert trace.model.backend == ModelBackend.LOCAL
    assert trace.steps == []
    assert trace.success is None
    assert trace.judge == "none"
    assert trace.has_tensors is False
    assert trace.input_tokens == 0
    assert len(trace.trace_id) == 12


def test_anthropic_response_parsing():
    """steps_from_anthropic_response parses thinking + text blocks."""
    response = {
        "content": [
            {"type": "thinking", "thinking": "Let me work through this step by step."},
            {"type": "thinking", "thinking": "2 + 2 = 4."},
            {"type": "text", "text": "The answer is 4."},
        ]
    }
    steps = steps_from_anthropic_response(response)

    assert len(steps) == 3
    assert steps[0].kind == StepKind.THINKING
    assert steps[0].index == 0
    assert steps[1].kind == StepKind.THINKING
    assert steps[2].kind == StepKind.OUTPUT
    assert steps[2].text == "The answer is 4."


def test_anthropic_tool_use_parsing():
    """steps_from_anthropic_response handles tool_use blocks."""
    response = {
        "content": [
            {"type": "thinking", "thinking": "I should use a calculator."},
            {"type": "tool_use", "name": "calculator", "input": {"expression": "2+2"}},
            {"type": "text", "text": "The answer is 4."},
        ]
    }
    steps = steps_from_anthropic_response(response)

    assert len(steps) == 3
    assert steps[1].kind == StepKind.TOOL_CALL
    parsed = json.loads(steps[1].text)
    assert parsed["tool"] == "calculator"


def test_trace_from_anthropic():
    """trace_from_anthropic builds a complete trace from API response."""
    response = {
        "content": [
            {"type": "thinking", "thinking": "Working through it..."},
            {"type": "text", "text": "The answer is 4."},
        ],
        "usage": {"input_tokens": 50, "output_tokens": 20},
    }
    task = TaskInfo("gsm8k", "test_001", "What is 2+2?", domain="math")
    trace = trace_from_anthropic(response, task, duration_ms=1200)

    assert trace.model.backend == ModelBackend.ANTHROPIC
    assert len(trace.steps) == 2
    assert trace.response_text == "The answer is 4."
    assert trace.input_tokens == 50
    assert trace.total_tokens == 70
    assert trace.duration_ms == 1200


def test_trace_store_write_read():
    """TraceStore writes JSONL and reads back."""
    with tempfile.TemporaryDirectory() as d:
        with TraceStore(Path(d) / "run_001") as store:
            for i in range(3):
                store.append(_make_trace(
                    task=TaskInfo("gsm8k", f"task_{i}", f"Question {i}"),
                ))

        store = TraceStore(Path(d) / "run_001", mode="r")
        traces = list(store)
        assert len(traces) == 3
        assert traces[1].task.task_id == "task_1"


def test_trace_store_summary():
    """TraceStore.summary() computes correct stats."""
    with tempfile.TemporaryDirectory() as d:
        with TraceStore(Path(d) / "run") as store:
            store.append(_make_trace(success=True, judge="exact_match"))
            store.append(_make_trace(success=False, judge="exact_match"))
            store.append(_make_trace(success=None))  # unjudged

        store = TraceStore(Path(d) / "run", mode="r")
        s = store.summary()
        assert s["total_traces"] == 3
        assert s["success"] == 1
        assert s["failure"] == 1
        assert s["unjudged"] == 1
        assert s["backends"] == {"local": 3}
        assert s["datasets"] == {"gsm8k": 3}


def test_trace_store_with_tensors():
    """TraceStore saves companion .npz files for hidden states."""
    numpy = __import__("numpy")

    with tempfile.TemporaryDirectory() as d:
        trace = _make_trace()
        hs = numpy.random.randn(2, 32, 128).astype(numpy.float16)

        with TraceStore(Path(d) / "run_hs") as store:
            store.append(trace, hidden_states={"steps": hs})

        store = TraceStore(Path(d) / "run_hs", mode="r")
        for loaded in store:
            assert loaded.has_tensors
            tensors = store.load_tensors(loaded.trace_id)
            assert tensors is not None
            assert tensors["steps"].shape == (2, 32, 128)


def test_trace_store_no_tensors_returns_none():
    """TraceStore.load_tensors returns None when no .npz exists."""
    with tempfile.TemporaryDirectory() as d:
        with TraceStore(Path(d) / "run") as store:
            store.append(_make_trace())

        store = TraceStore(Path(d) / "run", mode="r")
        for trace in store:
            assert store.load_tensors(trace.trace_id) is None
            assert not trace.has_tensors


def test_multiple_backends_in_store():
    """TraceStore handles traces from different backends."""
    with tempfile.TemporaryDirectory() as d:
        with TraceStore(Path(d) / "mixed") as store:
            store.append(_make_trace(
                model=ModelInfo("olmo-7b", ModelBackend.LOCAL),
            ))
            store.append(_make_trace(
                model=ModelInfo("claude-sonnet-4-5-20250929", ModelBackend.ANTHROPIC),
            ))

        store = TraceStore(Path(d) / "mixed", mode="r")
        s = store.summary()
        assert s["backends"] == {"local": 1, "anthropic": 1}
