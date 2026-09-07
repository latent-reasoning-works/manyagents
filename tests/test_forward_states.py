"""Tests for forward_recurrent_states and forward_hidden_states_batched."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from manyagents.inference import (
    forward_hidden_states,
    forward_hidden_states_batched,
    forward_recurrent_states,
)


# ---------------------------------------------------------------------------
# forward_recurrent_states — synthetic weight-tied recurrent model
# ---------------------------------------------------------------------------

class _TinyRecurrent(nn.Module):
    """Minimal recurrent-depth causal LM stand-in: embed, apply one shared core
    block ``num_steps`` times, project. The core's successive firings are the
    recurrence axis a hook should capture."""

    def __init__(self, vocab=50, d=16, default_steps=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.core = nn.Linear(d, d)
        self.head = nn.Linear(d, vocab)
        self.default_steps = default_steps

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, input_ids, use_cache=False, num_steps=None, **_):
        h = self.embed(input_ids)
        for _step in range(num_steps or self.default_steps):
            h = torch.tanh(self.core(h))
        return self.head(h)


def test_recurrent_states_capture_each_firing():
    model = _TinyRecurrent(default_steps=4)
    ids = [1, 2, 3, 4, 5, 6]
    out = forward_recurrent_states(model, ids, input_length=3, step_module="core")
    assert out["n_steps"] == 4
    assert out["step_hidden_states"].shape == (3, 4, 16)  # (n_new, n_steps, d)
    assert out["n_new_tokens"] == 3
    # successive recurrence steps must differ (the core transforms the state)
    s = out["step_hidden_states"]
    assert not np.allclose(s[:, 0, :], s[:, 1, :])


def test_recurrent_states_threads_forward_kwargs():
    model = _TinyRecurrent(default_steps=4)
    ids = [1, 2, 3, 4]
    out = forward_recurrent_states(
        model, ids, input_length=1, step_module="core", forward_kwargs={"num_steps": 7}
    )
    assert out["n_steps"] == 7
    assert out["step_hidden_states"].shape == (3, 7, 16)


def test_recurrent_states_single_firing_on_nonrecurrent_module():
    model = _TinyRecurrent(default_steps=1)
    out = forward_recurrent_states(model, [1, 2, 3], input_length=1, step_module="core")
    assert out["n_steps"] == 1


def test_recurrent_states_validates_input_length():
    model = _TinyRecurrent()
    with pytest.raises(ValueError):
        forward_recurrent_states(model, [1, 2, 3], input_length=0, step_module="core")


# ---------------------------------------------------------------------------
# forward_hidden_states_batched — equivalence with the single-sequence path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_gpt2():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(
        vocab_size=100, n_positions=64, n_embd=32, n_layer=2, n_head=2
    )
    model = transformers.GPT2LMHeadModel(cfg)
    model.eval()
    return model


def test_batched_matches_single(tiny_gpt2):
    torch.manual_seed(0)
    seqs = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15, 16]]
    lens = [3, 2, 4]
    batched = forward_hidden_states_batched(tiny_gpt2, seqs, input_lengths=lens)
    assert len(batched) == 3
    for seq, il, res in zip(seqs, lens, batched):
        single = forward_hidden_states(tiny_gpt2, seq, input_length=il)
        assert res["token_hidden_states"].shape == single["token_hidden_states"].shape
        np.testing.assert_allclose(
            res["token_hidden_states"],
            single["token_hidden_states"],
            rtol=1e-4,
            atol=1e-5,
        )
        assert res["layers_captured"] == single["layers_captured"]
        assert res["n_new_tokens"] == single["n_new_tokens"]


def test_batched_layer_selection(tiny_gpt2):
    seqs = [[1, 2, 3, 4], [5, 6, 7]]
    res = forward_hidden_states_batched(
        tiny_gpt2, seqs, input_lengths=[2, 1], layers=[-1]
    )
    assert res[0]["token_hidden_states"].shape[1] == 1
    assert res[1]["token_hidden_states"].shape == (2, 1, 32)


def test_batched_validates_lengths(tiny_gpt2):
    with pytest.raises(ValueError):
        forward_hidden_states_batched(tiny_gpt2, [[1, 2]], input_lengths=[0])
    with pytest.raises(ValueError):
        forward_hidden_states_batched(tiny_gpt2, [[1, 2]], input_lengths=[1, 1])
