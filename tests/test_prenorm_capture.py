"""Tests for forward_hidden_states(capture_prenorm=True).

The final RMSNorm is a readout transform; HF's output_hidden_states[-1] is
post-norm, so the pre-norm residual stream is only reachable via a hook. These
tests pin the contract: the hook fires, shapes are right, and norm(prenorm)
reproduces the post-norm last hidden state.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from manyagents.inference import _final_norm_module, forward_hidden_states  # noqa: E402


def _tiny_model():
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg).eval(), cfg


def test_final_norm_module_located():
    model, _ = _tiny_model()
    assert _final_norm_module(model) is model.model.norm


def test_prenorm_shape_and_norm_identity():
    model, cfg = _tiny_model()
    ids = [1, 2, 3, 4, 5, 6]
    input_length = 2
    n_new = len(ids) - input_length

    out = forward_hidden_states(
        model, ids, input_length=input_length, capture_prenorm=True)

    assert out["prenorm_hidden_states"].shape == (n_new, cfg.hidden_size)
    assert out["token_hidden_states"].shape[0] == n_new

    # norm(prenorm) must equal the post-norm last hidden state over the same slice
    norm = _final_norm_module(model)
    with torch.no_grad():
        post = norm(torch.tensor(out["prenorm_hidden_states"])).numpy()
        full = model(torch.tensor([ids]), output_hidden_states=True, use_cache=False)
    post_ref = full.hidden_states[-1][0, input_length - 1:len(ids) - 1, :].numpy()

    assert np.allclose(post, post_ref, atol=1e-4)
    # and the norm actually changes the state (pre != post)
    assert not np.allclose(out["prenorm_hidden_states"], post_ref, atol=1e-3)


def test_prenorm_absent_when_not_requested():
    model, _ = _tiny_model()
    out = forward_hidden_states(model, [1, 2, 3, 4], input_length=1)
    assert "prenorm_hidden_states" not in out


def test_final_norm_module_families():
    """Locator handles base/attr names beyond Llama: GPT-2, GPT-NeoX, MPT."""
    import torch.nn as nn

    def _wrap(base_attr, norm_attr):
        norm = nn.LayerNorm(8)
        base = nn.Module()
        setattr(base, norm_attr, norm)
        top = nn.Module()
        setattr(top, base_attr, base)
        return top, norm

    for base_attr, norm_attr in [
        ("transformer", "ln_f"),            # GPT-2
        ("gpt_neox", "final_layer_norm"),   # GPT-NeoX / Pythia
        ("transformer", "norm_f"),          # MPT
        ("model", "norm"),                  # Llama / Qwen / Gemma
    ]:
        top, norm = _wrap(base_attr, norm_attr)
        assert _final_norm_module(top) is norm


def test_final_norm_module_not_found():
    import torch.nn as nn

    top = nn.Module()
    top.head = nn.Linear(4, 4)  # base falls through to `top`; no known norm attr
    with pytest.raises(ValueError, match="final norm"):
        _final_norm_module(top)


def test_input_length_out_of_range_raises():
    """input_length must be in [1, total]; 0 made token/prenorm slices disagree."""
    model, _ = _tiny_model()
    ids = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="input_length"):
        forward_hidden_states(model, ids, input_length=0, capture_prenorm=True)
    with pytest.raises(ValueError, match="input_length"):
        forward_hidden_states(model, ids, input_length=len(ids) + 1)


def test_empty_completion_shapes():
    """input_length == total (no completion) → zero-row, shape-correct outputs."""
    model, cfg = _tiny_model()
    ids = [1, 2, 3]
    out = forward_hidden_states(
        model, ids, input_length=len(ids), capture_prenorm=True)
    assert out["n_new_tokens"] == 0
    assert out["token_hidden_states"].shape[0] == 0
    assert out["prenorm_hidden_states"].shape == (0, cfg.hidden_size)
