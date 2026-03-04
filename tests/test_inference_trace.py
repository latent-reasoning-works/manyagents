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


# ---------------------------------------------------------------------------
# extract_trace tests
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock
from manyagents.inference import extract_trace


def _mock_generate_with_hidden_states(model, tokenizer, prompt, **kwargs):
    """Mock that returns realistic generation output."""
    n_tokens = 10
    n_layers = 1
    d_model = 64
    return {
        "text": "Step one.\nStep two.\nThe answer is 4.",
        "token_hidden_states": np.random.randn(n_tokens, n_layers, d_model).astype(np.float32),
        "input_length": 20,
        "n_new_tokens": n_tokens,
        "generation_time_ms": 500,
        "layers_captured": [32],
    }


def _mock_tokenizer():
    """Mock tokenizer with encode method."""
    tok = MagicMock()
    tok.encode = lambda text, add_special_tokens=False: list(range(len(text) // 5 + 1))
    tok.apply_chat_template = MagicMock(return_value="<formatted prompt>")
    return tok


def test_extract_trace_returns_trace_and_hidden_states():
    """extract_trace returns (ReasoningTrace, hidden_states_dict)."""
    model = MagicMock()
    tokenizer = _mock_tokenizer()
    task = TaskInfo("gsm8k", "train_0", "What is 2+2?", expected_answer="4")

    with patch("manyagents.inference.generate_with_hidden_states", _mock_generate_with_hidden_states):
        trace, hs = extract_trace(
            model, tokenizer,
            prompt="What is 2+2?",
            task=task,
            model_name="olmo-7b",
            model_path="/weights/olmo",
            system_prompt="Solve step by step.",
            max_new_tokens=512,
            temperature=0.7,
            layers=[-1],
        )

    assert isinstance(trace, ReasoningTrace)
    assert trace.model.name == "olmo-7b"
    assert len(trace.steps) >= 1
    assert trace.output_tokens == 10
    assert trace.duration_ms == 500

    assert "pooled_steps" in hs
    assert "token_level" in hs
    assert hs["pooled_steps"].dtype == np.float16
    assert hs["token_level"].dtype == np.float16
    assert hs["token_level"].shape[0] == 10
    assert hs["pooled_steps"].shape[0] == len(trace.steps)


def test_extract_trace_fallback_single_step():
    """extract_trace falls back to single step when no delimiter matches."""
    model = MagicMock()
    tokenizer = _mock_tokenizer()
    task = TaskInfo("gsm8k", "train_0", "Q?")

    def _gen_no_newlines(model, tokenizer, prompt, **kwargs):
        return {
            "text": "The answer is simply four",
            "token_hidden_states": np.random.randn(5, 1, 64).astype(np.float32),
            "input_length": 10,
            "n_new_tokens": 5,
            "generation_time_ms": 200,
            "layers_captured": [32],
        }

    with patch("manyagents.inference.generate_with_hidden_states", _gen_no_newlines):
        trace, hs = extract_trace(
            model, tokenizer, prompt="Q?", task=task,
            model_name="test", model_path="/test",
        )

    assert len(trace.steps) >= 1
    assert hs["pooled_steps"].shape[0] == len(trace.steps)


# ---------------------------------------------------------------------------
# Model cache tests
# ---------------------------------------------------------------------------

from manyagents.inference import get_model, clear_model_cache, _model_cache


def test_model_cache_stores_and_retrieves(monkeypatch):
    """get_model caches by resolved path; second call returns same objects."""
    call_count = 0
    sentinel_model = object()
    sentinel_tok = object()
    sentinel_hf = object()

    def _fake_load(path, **kwargs):
        nonlocal call_count
        call_count += 1
        return sentinel_model, sentinel_tok, sentinel_hf

    monkeypatch.setattr("manyagents.inference.load_model", _fake_load)
    monkeypatch.setattr("manyagents.inference.resolve_model_path", lambda m: f"/fake/{m}")
    clear_model_cache()

    m1, t1, h1 = get_model("test-model")
    m2, t2, h2 = get_model("test-model")

    assert m1 is sentinel_model
    assert m1 is m2
    assert call_count == 1

    clear_model_cache()
    assert len(_model_cache) == 0


def test_clear_model_cache_empties():
    """clear_model_cache empties the cache dict."""
    from manyagents.inference import _model_cache
    _model_cache["dummy"] = ("a", "b", "c")
    clear_model_cache()
    assert "dummy" not in _model_cache
