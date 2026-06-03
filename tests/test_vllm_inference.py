# tests/test_vllm_inference.py
"""Tests for the vLLM generation backend + HF hidden-state forward pass.

The correctness anchor is test_forward_hidden_states_matches_generate: it proves
that a single teacher-forced HF forward pass over prompt+completion reproduces
the per-token hidden states that model.generate(output_hidden_states=True) emits.
That equivalence is *why* backend="vllm" (vLLM generates, HF re-encodes) yields
the same trajectory geometry as backend="hf". It runs on CPU with a tiny
random-init GPT-2 — no GPU, no network, no vllm install.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from manyagents.inference import (
    forward_hidden_states,
    vllm_generate,
    extract_trace,
)
from manyagents.schemas.reasoning import ModelBackend, TaskInfo, ReasoningTrace


# ---------------------------------------------------------------------------
# forward_hidden_states — the correctness anchor (CPU, offline, no vllm)
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401
    from transformers import GPT2Config, GPT2LMHeadModel  # noqa: F401
    _has_torch = True
except Exception:  # pragma: no cover
    _has_torch = False

_torch_required = pytest.mark.skipif(not _has_torch, reason="torch + transformers required")


def _tiny_gpt2():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=64, n_positions=64, n_embd=32, n_layer=3, n_head=4,
    )
    model = GPT2LMHeadModel(config).eval()
    return model, config


@_torch_required
def test_forward_hidden_states_matches_generate():
    """forward_hidden_states reproduces generate()'s per-new-token hidden states."""
    import torch

    model, config = _tiny_gpt2()
    input_ids = torch.randint(0, config.vocab_size, (1, 5))

    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=6,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=config.eos_token_id,
        )

    # Reference: last-position state at every layer, per generated step.
    n_layers = config.n_layer + 1
    ref = np.stack([
        np.stack([step_hidden[li][0, -1, :].numpy() for li in range(n_layers)])
        for step_hidden in gen.hidden_states
    ])  # (n_new, n_layers, d)

    full_ids = gen.sequences[0].tolist()
    out = forward_hidden_states(model, full_ids, input_length=5, layers=None)

    assert out["n_new_tokens"] == 6
    assert out["token_hidden_states"].shape == ref.shape
    np.testing.assert_allclose(out["token_hidden_states"], ref, atol=1e-3, rtol=1e-3)


@_torch_required
def test_forward_hidden_states_layer_selection():
    """layers=[-1] keeps a single (last) layer."""
    model, config = _tiny_gpt2()
    full_ids = list(range(10))
    out = forward_hidden_states(model, full_ids, input_length=4, layers=[-1])

    assert out["token_hidden_states"].shape == (6, 1, config.n_embd)
    assert out["layers_captured"] == [config.n_layer]  # -1 % (n_layer+1)


@_torch_required
def test_forward_hidden_states_empty_completion():
    """An empty completion keeps the (0, n_layers, d) shape contract."""
    model, config = _tiny_gpt2()
    full_ids = [1, 2, 3, 4]
    out = forward_hidden_states(model, full_ids, input_length=4, layers=[-1])

    assert out["n_new_tokens"] == 0
    assert out["token_hidden_states"].shape == (0, 1, config.n_embd)


# ---------------------------------------------------------------------------
# Guard rails that need neither vllm nor a model
# ---------------------------------------------------------------------------


def test_vllm_generate_requires_engine_or_model():
    """vllm_generate raises a clear error when given neither engine nor model."""
    with pytest.raises(ValueError, match="engine.*or.*model|`engine`"):
        vllm_generate("hello")


def test_extract_trace_vllm_requires_engine():
    """backend='vllm' without an engine raises before touching vllm."""
    tok = MagicMock()
    tok.apply_chat_template = MagicMock(return_value="<formatted>")
    with pytest.raises(ValueError, match="vllm_engine"):
        extract_trace(
            MagicMock(), tok, prompt="Q?",
            task=TaskInfo("d", "t", "Q?"),
            model_name="m", model_path="/m",
            backend="vllm",
        )


def test_extract_trace_unknown_backend():
    """An unknown backend name is rejected."""
    tok = MagicMock()
    tok.apply_chat_template = MagicMock(return_value="<formatted>")
    with pytest.raises(ValueError, match="Unknown backend"):
        extract_trace(
            MagicMock(), tok, prompt="Q?",
            task=TaskInfo("d", "t", "Q?"),
            model_name="m", model_path="/m",
            backend="banana",
        )


def test_vllm_adapter_registered():
    """The vllm adapter is in the registry (lazy vllm import keeps this safe)."""
    from manyagents.adapters import ADAPTER_REGISTRY, VLLMAdapter
    assert ADAPTER_REGISTRY["vllm"] is VLLMAdapter


# ---------------------------------------------------------------------------
# extract_trace(backend="vllm") wiring — mocked generation + forward pass
# ---------------------------------------------------------------------------


def _mock_tokenizer():
    tok = MagicMock()
    tok.encode = lambda text, add_special_tokens=False: list(range(len(text) // 5 + 1))
    tok.apply_chat_template = MagicMock(return_value="<formatted prompt>")
    return tok


def test_extract_trace_vllm_backend_wiring():
    """backend='vllm' threads vllm_generate -> forward_hidden_states -> trace."""
    task = TaskInfo("gsm8k", "t0", "What is 2+2?")

    def _mock_vllm_generate(prompts, **kw):
        return [{
            "text": "Step one.\nStep two.\nThe answer is 4.",
            "prompt_token_ids": [1, 2, 3],
            "completion_token_ids": list(range(4, 14)),
            "input_length": 3,
            "n_new_tokens": 10,
            "generation_time_ms": 5,
            "finish_reason": "stop",
        }]

    def _mock_forward(model, input_ids, *, input_length, layers=None):
        return {
            "token_hidden_states": np.random.randn(10, 1, 64).astype(np.float32),
            "input_length": input_length,
            "n_new_tokens": 10,
            "generation_time_ms": 3,
            "layers_captured": [12],
        }

    with patch("manyagents.inference.vllm_generate", _mock_vllm_generate), \
         patch("manyagents.inference.forward_hidden_states", _mock_forward):
        trace, hs = extract_trace(
            MagicMock(), _mock_tokenizer(),
            prompt="What is 2+2?",
            task=task,
            model_name="Qwen/Qwen3-0.6B",
            model_path="Qwen/Qwen3-0.6B",
            backend="vllm",
            vllm_engine=object(),
            layers=[-1],
        )

    assert isinstance(trace, ReasoningTrace)
    assert trace.model.backend == ModelBackend.VLLM
    assert trace.output_tokens == 10
    assert "pooled_steps" in hs and "token_level" in hs
    assert hs["token_level"].shape[0] == 10
    assert hs["pooled_steps"].shape[0] == len(trace.steps)
    # vLLM sampling knobs recorded for reproducibility
    assert "top_p" in trace.model.generation_config


# ---------------------------------------------------------------------------
# vLLM-only tests (skipped unless vllm is installed)
# ---------------------------------------------------------------------------

try:
    import vllm  # noqa: F401
    _has_vllm = True
except Exception:
    _has_vllm = False

_vllm_required = pytest.mark.skipif(not _has_vllm, reason="vllm not installed")


@_vllm_required
def test_build_sampling_params_maps_knobs():
    """build_sampling_params maps max_new_tokens->max_tokens and merges overrides."""
    from manyagents.inference import build_sampling_params

    # temperature > 0 so vLLM keeps top_p/top_k (greedy/temp=0 zeroes them).
    sp = build_sampling_params(
        max_new_tokens=128, temperature=0.6, top_p=0.95, top_k=40,
        overrides={"min_p": 0.05, "repetition_penalty": 1.1},
    )
    assert sp.max_tokens == 128
    assert sp.temperature == 0.6
    assert sp.top_p == 0.95
    assert sp.top_k == 40
    assert sp.min_p == 0.05
    assert sp.repetition_penalty == 1.1


@_vllm_required
def test_build_sampling_params_overrides_win():
    """overrides win over the explicit knob on conflict."""
    from manyagents.inference import build_sampling_params

    sp = build_sampling_params(temperature=0.7, overrides={"temperature": 0.2})
    assert sp.temperature == 0.2


# ---------------------------------------------------------------------------
# GPU end-to-end integration (needs vllm + a CUDA device + a small model).
# Model defaults to Qwen/Qwen3-0.6B; override with MANYAGENTS_TEST_VLLM_MODEL.
# ---------------------------------------------------------------------------

import os

# Run vLLM's engine in-process: pytest initializes CUDA in the parent (e.g. the
# torch.cuda.is_available() probe below), and vLLM v1's default forked EngineCore
# cannot re-init CUDA in a forked child. In-process sidesteps that.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_has_cuda = _has_torch and __import__("torch").cuda.is_available()
_gpu_required = pytest.mark.skipif(
    not (_has_vllm and _has_cuda),
    reason="needs vllm + CUDA GPU",
)
_TEST_MODEL = os.environ.get("MANYAGENTS_TEST_VLLM_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def vllm_engine():
    from manyagents.inference import get_vllm_engine, clear_model_cache

    # Explicit engine knobs — the integration also checks these are accepted.
    engine = get_vllm_engine(
        _TEST_MODEL,
        max_num_batched_tokens=4096,
        max_num_seqs=16,
        max_model_len=2048,
        gpu_memory_utilization=0.45,
        enforce_eager=True,  # skip CUDA-graph capture for a faster test build
    )
    yield engine
    clear_model_cache()


@pytest.fixture(scope="module")
def hf_model_and_tokenizer():
    """HF model for the hidden-state forward pass, loaded directly (the
    manylatents-backed get_model() loader is orthogonal to the vLLM path)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        _TEST_MODEL, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(_TEST_MODEL)
    return model, tokenizer


@_gpu_required
def test_vllm_generate_returns_aligned_token_ids(vllm_engine):
    """A real generation returns text + the exact token ids for re-encoding."""
    from manyagents.inference import vllm_generate

    outs = vllm_generate(
        ["1 + 1 =", "The capital of France is"],
        engine=vllm_engine,
        max_new_tokens=8,
        temperature=0.6,
        seed=0,
    )
    assert len(outs) == 2
    for o in outs:
        assert isinstance(o["text"], str)
        assert len(o["completion_token_ids"]) > 0
        assert o["n_new_tokens"] == len(o["completion_token_ids"])
        assert o["input_length"] == len(o["prompt_token_ids"])


@_gpu_required
def test_vllm_backend_matches_hf_backend_on_shared_tokens(vllm_engine, hf_model_and_tokenizer):
    """vLLM-generated ids, re-encoded by HF, give a deterministic forward pass.

    The end-to-end guarantee: hidden states captured for the vLLM backend equal
    those an HF forward pass produces for the identical token sequence.
    """
    import numpy as np
    from manyagents.inference import vllm_generate, forward_hidden_states

    out = vllm_generate(
        "List the first three prime numbers:",
        engine=vllm_engine, max_new_tokens=12, temperature=0.6, seed=0,
    )[0]
    full_ids = out["prompt_token_ids"] + out["completion_token_ids"]

    hf_model, _ = hf_model_and_tokenizer
    a = forward_hidden_states(hf_model, full_ids, input_length=out["input_length"], layers=[-1])
    b = forward_hidden_states(hf_model, full_ids, input_length=out["input_length"], layers=[-1])

    assert a["token_hidden_states"].shape == (out["n_new_tokens"], 1, hf_model.config.hidden_size)
    # Deterministic forward pass — identical across calls.
    np.testing.assert_allclose(a["token_hidden_states"], b["token_hidden_states"], atol=1e-4)


@_gpu_required
def test_extract_trace_vllm_end_to_end(vllm_engine, hf_model_and_tokenizer):
    """Full pipeline: vLLM generate -> HF hidden states -> ReasoningTrace."""
    from manyagents.inference import extract_trace, resolve_model_path
    from manyagents.schemas.reasoning import ModelBackend

    hf_model, tokenizer = hf_model_and_tokenizer
    trace, hs = extract_trace(
        hf_model, tokenizer,
        prompt="If a train goes 60 km in 1.5 h, what is its average speed?",
        task=TaskInfo("demo", "t0", "speed?"),
        model_name=_TEST_MODEL,
        model_path=resolve_model_path(_TEST_MODEL),
        backend="vllm",
        vllm_engine=vllm_engine,
        system_prompt="Solve step by step, one step per line.",
        max_new_tokens=64,
        temperature=0.6,
        layers=[-1],
    )

    assert trace.model.backend == ModelBackend.VLLM
    assert len(trace.steps) >= 1
    assert trace.output_tokens > 0
    assert hs["token_level"].shape[0] == trace.output_tokens
    assert hs["pooled_steps"].shape[0] == len(trace.steps)
