# tests/test_inference_trace.py
"""Tests for inference.py trace building functions."""

import numpy as np

from manyagents.inference import build_reasoning_trace
from manyagents.schemas.reasoning import (
    ModelBackend, StepKind, TaskInfo, ReasoningTrace,
)


def test_build_reasoning_trace_basic():
    """build_reasoning_trace produces a valid ReasoningTrace with correct steps."""
    task = TaskInfo("gsm8k", "train_0", "What is 2+2?", expected_answer="4", domain="math")
    step_defs = [
        {"text": "First, 2+2=4", "token_start": 0, "token_end": 5},
        {"text": "The answer is 4", "token_start": 5, "token_end": 10},
    ]
    gen_metadata = {
        "input_length": 20,
        "n_new_tokens": 10,
        "generation_time_ms": 500,
        "layers_captured": [32],
    }

    trace = build_reasoning_trace(
        text="First, 2+2=4\nThe answer is 4",
        gen_metadata=gen_metadata,
        model_name="olmo-7b",
        model_path="/network/weights/olmo/OLMo-7B-Twin-2T",
        task=task,
        step_defs=step_defs,
        generation_config={"max_new_tokens": 512, "temperature": 0.7},
    )

    assert isinstance(trace, ReasoningTrace)
    assert trace.model.name == "olmo-7b"
    assert trace.model.backend == ModelBackend.LOCAL
    assert trace.model.path == "/network/weights/olmo/OLMo-7B-Twin-2T"
    assert len(trace.steps) == 2
    assert trace.steps[0].kind == StepKind.THINKING
    assert trace.steps[1].kind == StepKind.OUTPUT
    assert trace.steps[0].token_count == 5
    assert trace.steps[1].layers_captured == [32]
    assert trace.input_tokens == 20
    assert trace.output_tokens == 10
    assert trace.total_tokens == 30
    assert trace.duration_ms == 500
    assert trace.task.dataset == "gsm8k"


def test_build_reasoning_trace_single_step():
    """Single-step trace marks the only step as OUTPUT."""
    task = TaskInfo("gsm8k", "train_0", "Q?")
    step_defs = [{"text": "Answer", "token_start": 0, "token_end": 3}]
    gen_metadata = {"input_length": 5, "n_new_tokens": 3, "generation_time_ms": 100, "layers_captured": []}

    trace = build_reasoning_trace(
        text="Answer",
        gen_metadata=gen_metadata,
        model_name="test",
        model_path="/test",
        task=task,
        step_defs=step_defs,
        generation_config={},
    )

    assert len(trace.steps) == 1
    assert trace.steps[0].kind == StepKind.OUTPUT
