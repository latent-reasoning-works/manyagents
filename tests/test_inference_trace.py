# tests/test_inference_trace.py
"""Tests for inference.py trace building and segmentation functions."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from manyagents.inference import (
    extract_trace,
    get_model,
    clear_model_cache,
    _model_cache,
    build_reasoning_trace,
    segment_by_delimiter,
    segment_by_tags,
    segment_by_velocity,
    segment_hybrid,
    segment,
    split_into_steps,
)
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
        model_path="/models/olmo-7b",
        task=task,
        step_defs=step_defs,
        generation_config={"max_new_tokens": 512, "temperature": 0.7},
    )

    assert isinstance(trace, ReasoningTrace)
    assert trace.model.name == "olmo-7b"
    assert trace.model.backend == ModelBackend.LOCAL
    assert trace.model.path == "/models/olmo-7b"
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


def test_extract_trace_forwards_repetition_penalty():
    """extract_trace threads repetition_penalty into HF gen and records it (>1 only)."""
    model = MagicMock()
    tokenizer = _mock_tokenizer()
    task = TaskInfo("gsm8k", "train_0", "Q?")
    seen = {}

    def _recording_gen(model, tokenizer, prompt, **kwargs):
        seen.update(kwargs)
        return _mock_generate_with_hidden_states(model, tokenizer, prompt, **kwargs)

    with patch("manyagents.inference.generate_with_hidden_states", _recording_gen):
        trace, _ = extract_trace(
            model, tokenizer, prompt="Q?", task=task,
            model_name="t", model_path="/t", repetition_penalty=1.3,
        )
    assert seen["repetition_penalty"] == 1.3
    # provenance: a non-default penalty is recorded on the trace
    assert trace.model.generation_config["repetition_penalty"] == 1.3

    # default (1.0) is the committed/no-ablation condition: not recorded as a knob
    seen.clear()
    with patch("manyagents.inference.generate_with_hidden_states", _recording_gen):
        trace, _ = extract_trace(
            model, tokenizer, prompt="Q?", task=task,
            model_name="t", model_path="/t",
        )
    assert seen["repetition_penalty"] == 1.0
    assert "repetition_penalty" not in trace.model.generation_config


def test_generate_with_hidden_states_passes_repetition_penalty():
    """generate_with_hidden_states forwards repetition_penalty to model.generate."""
    import torch
    from manyagents.inference import generate_with_hidden_states

    model = MagicMock()
    model.config.num_hidden_layers = 1
    model.config.hidden_size = 4
    model.device = "cpu"

    tok = MagicMock()
    tok.return_value = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
    tok.eos_token_id = 0
    tok.decode = lambda ids, skip_special_tokens=True: "hello world"

    # 2 generation steps, each carrying (n_layers + 1) = 2 hidden tensors
    step = (torch.randn(1, 3, 4), torch.randn(1, 3, 4))
    out = MagicMock()
    out.hidden_states = (step, step)
    out.sequences = torch.zeros(1, 5, dtype=torch.long)
    model.generate.return_value = out

    generate_with_hidden_states(
        model, tok, "prompt", max_new_tokens=2, temperature=0.7,
        repetition_penalty=1.3,
    )
    assert model.generate.call_args.kwargs["repetition_penalty"] == 1.3


def test_extract_traces_batch_generates_all_in_one_call():
    """extract_traces_batch batches every prompt through a single vllm_generate."""
    from manyagents.inference import extract_traces_batch

    model = MagicMock()
    tokenizer = _mock_tokenizer()
    prompts = ["Q1?", "Q2?", "Q3?"]
    tasks = [TaskInfo("gsm8k", f"t{i}", p) for i, p in enumerate(prompts)]

    calls = {}

    def _fake_vllm_generate(formatted, **kwargs):
        calls["formatted"] = formatted
        calls["overrides"] = kwargs.get("sampling_overrides")
        return [{
            "text": f"Step.\nThe answer is {i}.",
            "prompt_token_ids": [0, 1], "completion_token_ids": [2, 3, 4],
            "input_length": 2, "n_new_tokens": 3, "generation_time_ms": 10,
            "finish_reason": "stop",
        } for i in range(len(formatted))]

    def _fake_forward(model, full_ids, *, input_length, layers=None):
        return {
            "token_hidden_states": np.random.randn(3, 1, 8).astype(np.float32),
            "input_length": input_length, "n_new_tokens": 3,
            "generation_time_ms": 5, "layers_captured": [-1],
        }

    with patch("manyagents.inference.vllm_generate", _fake_vllm_generate), \
         patch("manyagents.inference.forward_hidden_states", _fake_forward):
        out = extract_traces_batch(
            model, tokenizer, prompts, tasks,
            vllm_engine=MagicMock(), model_name="m", model_path="/m",
            repetition_penalty=1.3, layers=[-1],
        )

    assert len(calls["formatted"]) == 3            # one batched call, all prompts
    assert calls["overrides"] == {"repetition_penalty": 1.3}
    assert len(out) == 3                           # aligned with inputs
    for trace, hs in out:
        assert isinstance(trace, ReasoningTrace)
        assert hs["token_level"].dtype == np.float16
        assert trace.model.generation_config["repetition_penalty"] == 1.3


def test_extract_traces_batch_guards():
    """extract_traces_batch validates its inputs and short-circuits empties."""
    from manyagents.inference import extract_traces_batch

    tok = _mock_tokenizer()
    import pytest
    with pytest.raises(ValueError, match="requires `vllm_engine`"):
        extract_traces_batch(MagicMock(), tok, ["q"], [TaskInfo("d", "t", "q")],
                             vllm_engine=None, model_name="m", model_path="/m")
    with pytest.raises(ValueError, match="same length"):
        extract_traces_batch(MagicMock(), tok, ["q1", "q2"], [TaskInfo("d", "t", "q1")],
                             vllm_engine=MagicMock(), model_name="m", model_path="/m")
    # empty input is a no-op, not an engine call
    assert extract_traces_batch(MagicMock(), tok, [], [], vllm_engine=MagicMock(),
                                model_name="m", model_path="/m") == []


# ---------------------------------------------------------------------------
# Segmentation tests
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer mock for segmentation tests."""

    def encode(self, text, add_special_tokens=False):
        # ~1 token per 4 chars, deterministic
        return list(range(len(text) // 4 + 1))

    def decode(self, token_ids, skip_special_tokens=True):
        return f"decoded_{len(token_ids)}_tokens"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<formatted>"


def test_segment_by_delimiter_basic():
    """segment_by_delimiter splits on newlines and adds kind field."""
    tok = _FakeTokenizer()
    steps = segment_by_delimiter("Step one\nStep two\nThe answer", tok)

    assert len(steps) == 3
    assert steps[0]["kind"] == "thinking"
    assert steps[1]["kind"] == "thinking"
    assert steps[2]["kind"] == "output"  # last step
    assert steps[0]["text"] == "Step one"
    assert all("token_start" in s and "token_end" in s for s in steps)


def test_segment_by_delimiter_single_line():
    """Single line gets kind=output."""
    tok = _FakeTokenizer()
    steps = segment_by_delimiter("Just the answer", tok)

    assert len(steps) == 1
    assert steps[0]["kind"] == "output"


def test_split_into_steps_is_alias():
    """split_into_steps is a backward-compatible alias for segment_by_delimiter."""
    assert split_into_steps is segment_by_delimiter


def test_segment_by_tags_with_think_block():
    """segment_by_tags parses <think>...</think> and marks output correctly."""
    tok = _FakeTokenizer()
    text = "<think>First I add 2+2. Then I get 4.</think>The answer is 4."
    steps = segment_by_tags(text, tok)

    # Should have thinking steps + output
    thinking_steps = [s for s in steps if s["kind"] == "thinking"]
    output_steps = [s for s in steps if s["kind"] == "output"]
    assert len(thinking_steps) >= 1
    assert len(output_steps) == 1
    assert output_steps[0]["text"] == "The answer is 4."


def test_segment_by_tags_no_tags_fallback():
    """segment_by_tags falls back to single output when no tags present."""
    tok = _FakeTokenizer()
    steps = segment_by_tags("No tags here at all", tok)

    assert len(steps) == 1
    assert steps[0]["kind"] == "output"


def test_segment_by_tags_multiline_think():
    """segment_by_tags splits think content on newlines and sentence boundaries."""
    tok = _FakeTokenizer()
    text = "<think>Line one.\nLine two.\nLine three.</think>Answer."
    steps = segment_by_tags(text, tok)

    thinking = [s for s in steps if s["kind"] == "thinking"]
    assert len(thinking) == 3
    assert steps[-1]["kind"] == "output"


def test_segment_by_tags_empty_after_think():
    """segment_by_tags handles case with no text after </think>."""
    tok = _FakeTokenizer()
    text = "<think>Just thinking here.</think>"
    steps = segment_by_tags(text, tok)

    assert len(steps) >= 1
    assert all(s["kind"] == "thinking" for s in steps)


try:
    import scipy  # noqa: F401
    _has_scipy = True
except ImportError:
    _has_scipy = False

_scipy_required = pytest.mark.skipif(not _has_scipy, reason="scipy required")


@pytest.mark.requires_manylatents
@_scipy_required
def test_segment_by_velocity_basic():
    """segment_by_velocity finds the cosine-distance spike at a sharp transition.

    Two internally-smooth regions meet at an orthogonal jump at token 10: region A
    points along dim 0, region B along the orthogonal dim 1, each with a tiny drift
    so within-region velocity stays ~0. That makes the 9->10 boundary the one
    prominent velocity peak, so the sequence splits into two steps there.

    (A previous version filled both regions with i.i.d. random states; that gives
    *every* step a high, peak-less velocity, so the boundary never stood out and
    find_peaks returned nothing — the data had a level shift, not an isolated spike.)
    """
    tok = _FakeTokenizer()
    n_tokens, d_model = 20, 32
    hs = np.zeros((n_tokens, 2, d_model), dtype=np.float32)
    for i in range(n_tokens):
        if i < 10:
            hs[i, -1, 0] = 1.0
            hs[i, -1, 2] = 0.01 * i           # smooth drift within region A
        else:
            hs[i, -1, 1] = 1.0
            hs[i, -1, 3] = 0.01 * (i - 10)    # smooth drift within region B

    text = "x" * (n_tokens * 4)  # enough chars for ~n_tokens tokens
    steps = segment_by_velocity(text, tok, hs, min_segment_tokens=3, prominence_factor=0.5)

    assert len(steps) >= 2
    assert any(s["token_start"] == 10 for s in steps)  # boundary at the injected transition
    assert steps[-1]["kind"] == "output"
    assert all(s["kind"] == "thinking" for s in steps[:-1])


@_scipy_required
def test_segment_by_velocity_short_text():
    """segment_by_velocity handles very short input gracefully."""
    tok = _FakeTokenizer()
    hs = np.random.randn(1, 2, 32).astype(np.float32)
    steps = segment_by_velocity("hi", tok, hs)

    assert len(steps) == 1
    assert steps[0]["kind"] == "output"


@pytest.mark.requires_manylatents
@_scipy_required
def test_segment_hybrid_with_tags():
    """segment_hybrid uses tags for structure + velocity within thinking."""
    tok = _FakeTokenizer()
    think_text = "x" * 80  # long enough thinking block
    text = f"<think>{think_text}</think>Final answer."

    full_tokens = tok.encode(text, add_special_tokens=False)
    n_tokens = len(full_tokens)
    hs = np.random.randn(n_tokens, 2, 32).astype(np.float32)

    steps = segment_hybrid(text, tok, hs, min_segment_tokens=2, prominence_factor=0.5)

    assert len(steps) >= 1
    output_steps = [s for s in steps if s["kind"] == "output"]
    assert len(output_steps) >= 1


@pytest.mark.requires_manylatents
@_scipy_required
def test_segment_hybrid_no_tags_falls_back_to_velocity():
    """segment_hybrid falls back to velocity when no tags present."""
    tok = _FakeTokenizer()
    n_tokens = 20
    hs = np.random.randn(n_tokens, 2, 32).astype(np.float32)
    text = "x" * (n_tokens * 4)

    steps = segment_hybrid(text, tok, hs, min_segment_tokens=3)
    assert len(steps) >= 1


def test_segment_dispatcher_delimiter():
    """segment() routes 'delimiter' to segment_by_delimiter."""
    tok = _FakeTokenizer()
    steps = segment("Line one\nLine two", tok, "delimiter")
    assert len(steps) == 2


def test_segment_dispatcher_tags():
    """segment() routes 'tags' to segment_by_tags."""
    tok = _FakeTokenizer()
    steps = segment("<think>Reasoning.</think>Answer.", tok, "tags")
    output = [s for s in steps if s["kind"] == "output"]
    assert len(output) >= 1


def test_segment_dispatcher_velocity_requires_hidden_states():
    """segment() raises ValueError for velocity without hidden states."""
    tok = _FakeTokenizer()
    with pytest.raises(ValueError, match="token_hidden_states"):
        segment("text", tok, "velocity")


def test_segment_dispatcher_hybrid_requires_hidden_states():
    """segment() raises ValueError for hybrid without hidden states."""
    tok = _FakeTokenizer()
    with pytest.raises(ValueError, match="token_hidden_states"):
        segment("text", tok, "hybrid")


def test_segment_dispatcher_unknown_raises():
    """segment() raises ValueError for unknown strategy."""
    tok = _FakeTokenizer()
    with pytest.raises(ValueError, match="Unknown segmentation"):
        segment("text", tok, "nonexistent")


def test_build_reasoning_trace_uses_kind_from_step_defs():
    """build_reasoning_trace uses kind from step_defs when present."""
    task = TaskInfo("gsm8k", "train_0", "Q?")
    step_defs = [
        {"text": "Think", "token_start": 0, "token_end": 3, "kind": "thinking"},
        {"text": "More think", "token_start": 3, "token_end": 6, "kind": "thinking"},
        {"text": "Answer", "token_start": 6, "token_end": 10, "kind": "output"},
    ]
    gen_metadata = {"input_length": 5, "n_new_tokens": 10, "generation_time_ms": 100, "layers_captured": []}

    trace = build_reasoning_trace(
        text="Think\nMore think\nAnswer",
        gen_metadata=gen_metadata,
        model_name="test",
        model_path="/test",
        task=task,
        step_defs=step_defs,
        generation_config={},
    )

    assert trace.steps[0].kind == StepKind.THINKING
    assert trace.steps[1].kind == StepKind.THINKING
    assert trace.steps[2].kind == StepKind.OUTPUT


def test_build_reasoning_trace_falls_back_without_kind():
    """build_reasoning_trace falls back to heuristic when kind absent."""
    task = TaskInfo("gsm8k", "train_0", "Q?")
    step_defs = [
        {"text": "Step 1", "token_start": 0, "token_end": 3},
        {"text": "Step 2", "token_start": 3, "token_end": 6},
    ]
    gen_metadata = {"input_length": 5, "n_new_tokens": 6, "generation_time_ms": 100, "layers_captured": []}

    trace = build_reasoning_trace(
        text="Step 1\nStep 2",
        gen_metadata=gen_metadata,
        model_name="test",
        model_path="/test",
        task=task,
        step_defs=step_defs,
        generation_config={},
    )

    assert trace.steps[0].kind == StepKind.THINKING
    assert trace.steps[1].kind == StepKind.OUTPUT


# ---------------------------------------------------------------------------
# Model cache tests
# ---------------------------------------------------------------------------


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
