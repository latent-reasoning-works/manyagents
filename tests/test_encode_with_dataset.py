"""Integration test: dataset loader + cumulative encoding."""
import numpy as np
import pytest
from tests.test_cumulative_encoding import FakeModel, FakeTokenizer


def test_dataset_entry_through_cumulative_encoding():
    """Load one Zhou entry and encode it with a mock model."""
    from manyagents.datasets.zhou_reasoning_flow import load_zhou_reasoning_flow
    from manyagents.inference import encode_cumulative_trajectory

    try:
        entries = load_zhou_reasoning_flow(n_samples=1)
    except Exception:
        pytest.skip("Dataset not accessible")

    if not entries:
        pytest.skip("No entries loaded")

    entry = entries[0]
    steps = entry["steps"]
    # Use logic type as prompt for symbolic entries
    prompt = f"Logic structure {entry['logic_type']}"

    model = FakeModel(d_model=32)
    tokenizer = FakeTokenizer()

    result = encode_cumulative_trajectory(
        model, tokenizer,
        prompt=prompt,
        steps=steps,
        layer=-1,
    )

    assert result.shape == (len(steps), 32)
    assert result.dtype == np.float32
    # Each step should have a different embedding (growing prefix → different position values)
    if len(steps) > 1:
        assert not np.allclose(result[0], result[1])
