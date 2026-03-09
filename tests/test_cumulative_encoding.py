"""Tests for encode_cumulative_trajectory.

This function implements Algorithm 1 from Zhou et al. (2026) —
forward-pass-only cumulative encoding of pre-written reasoning steps.
"""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fake model + tokenizer for testing without GPU
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Mock tokenizer that splits on whitespace."""

    def __init__(self):
        self.vocab = {}
        self._next_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = text.split()
        ids = []
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = self._next_id
                self._next_id += 1
            ids.append(self.vocab[t])
        return ids

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        result = MagicMock()
        result.input_ids = torch.tensor([ids], dtype=torch.long)
        result.attention_mask = torch.ones(1, len(ids), dtype=torch.long)
        result.to = lambda device: result
        return result


class FakeModel:
    """Mock HF model returning position-dependent hidden states.

    Hidden state at position i, layer L = i + L*1000.
    This makes it possible to verify that the correct token range was extracted.
    """

    def __init__(self, d_model: int = 64, n_layers: int = 2):
        self.d_model = d_model
        self.n_layers = n_layers
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=True, **kwargs):
        batch, seq_len = input_ids.shape
        hidden_states = []
        for layer in range(self.n_layers + 1):  # +1 because HF includes embedding layer
            h = torch.zeros(batch, seq_len, self.d_model)
            for pos in range(seq_len):
                h[0, pos, :] = pos + layer * 1000
            hidden_states.append(h)
        result = MagicMock()
        result.hidden_states = tuple(hidden_states)
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCumulativeEncoding:

    def test_output_shape(self):
        """Output is (n_steps, d_model)."""
        from manyagents.inference import encode_cumulative_trajectory

        model = FakeModel(d_model=64, n_layers=2)
        tokenizer = FakeTokenizer()

        result = encode_cumulative_trajectory(
            model, tokenizer,
            prompt="Solve this problem",
            steps=["Step one here", "Step two here", "Step three here"],
            layer=-1,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 64)
        assert result.dtype == np.float32

    def test_growing_prefix(self):
        """Each step should see a longer input than the previous."""
        from manyagents.inference import encode_cumulative_trajectory

        call_lengths = []

        class TrackingModel(FakeModel):
            def __call__(self, input_ids=None, **kwargs):
                call_lengths.append(input_ids.shape[1])
                return super().__call__(input_ids=input_ids, **kwargs)

        model = TrackingModel(d_model=32)
        tokenizer = FakeTokenizer()

        encode_cumulative_trajectory(
            model, tokenizer,
            prompt="Problem",
            steps=["Alpha bravo", "Charlie delta", "Echo foxtrot"],
            layer=-1,
        )

        # Each call should have strictly more tokens than the previous
        assert len(call_lengths) == 3
        for i in range(1, len(call_lengths)):
            assert call_lengths[i] > call_lengths[i - 1], (
                f"Call {i} ({call_lengths[i]} tokens) should be longer "
                f"than call {i-1} ({call_lengths[i-1]} tokens)"
            )

    def test_extracts_step_tokens_not_prefix(self):
        """Hidden states should be from step t's tokens only, not the full prefix."""
        from manyagents.inference import encode_cumulative_trajectory

        model = FakeModel(d_model=16, n_layers=1)
        tokenizer = FakeTokenizer()

        result = encode_cumulative_trajectory(
            model, tokenizer,
            prompt="Problem",
            steps=["Alpha", "Bravo"],
            layer=-1,  # last layer = index 1 in FakeModel (0=embed, 1=layer1)
        )

        # With FakeModel, hidden state at position p, last layer = p + 1*1000
        # Step 0 ("Alpha") is token at position 1 (after "Problem")
        # Its mean-pooled value should be 1 + 1000 = 1001, broadcast across d_model
        # Step 1 ("Bravo") is token at position 2
        # Its mean-pooled value should be 2 + 1000 = 1002

        # The key check: step 1's embedding should NOT be the mean of positions 0,1,2
        # (that would be (0+1+2)/3 + 1000 = 1001). It should be just position 2.
        assert result[0, 0] == pytest.approx(1001.0, abs=1.0)
        assert result[1, 0] == pytest.approx(1002.0, abs=1.0)

    def test_empty_steps_raises(self):
        """Should raise ValueError on empty steps list."""
        from manyagents.inference import encode_cumulative_trajectory

        model = FakeModel()
        tokenizer = FakeTokenizer()

        with pytest.raises(ValueError, match="steps"):
            encode_cumulative_trajectory(model, tokenizer, prompt="P", steps=[])

    def test_single_step(self):
        """Works with a single reasoning step."""
        from manyagents.inference import encode_cumulative_trajectory

        model = FakeModel(d_model=32)
        tokenizer = FakeTokenizer()

        result = encode_cumulative_trajectory(
            model, tokenizer,
            prompt="Problem",
            steps=["Only step"],
            layer=-1,
        )

        assert result.shape == (1, 32)
