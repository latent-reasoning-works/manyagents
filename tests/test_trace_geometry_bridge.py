"""Release contracts for chained arrays and the documented TraceStore bridge."""

from pathlib import Path

import numpy as np
import pytest

from manyagents.config_utils import validate_manylatents_config


def test_chained_data_satisfies_validation_without_dummy_dataset():
    config = {"algorithms": {"latent": {}}}
    assert validate_manylatents_config(config, input_data=np.ones((4, 3))) is config
    with pytest.raises(ValueError, match="data.*pipeline"):
        validate_manylatents_config(config)
    with pytest.raises(ValueError, match="algorithms.*pipeline"):
        validate_manylatents_config({}, input_data=np.ones((4, 3)))


@pytest.mark.requires_manylatents
async def test_adapter_pca_uses_supplied_array_without_dummy_dataset(tmp_path, monkeypatch):
    from manyagents.adapters.manylatents_adapter import ManyLatentsAdapter
    from sklearn.decomposition import PCA

    monkeypatch.chdir(tmp_path)
    data = np.random.default_rng(13).normal(size=(12, 5)).astype(np.float32)
    result = await ManyLatentsAdapter().run({"algorithm": "pca", "n_components": 2}, {}, input_data=data)
    assert result["success"], result["summary"]
    embeddings = result["embeddings"]["embeddings"]
    assert embeddings.shape == (12, 2)
    # Pairwise distances avoid arbitrary PCA sign choices and verify data identity.
    expected = PCA(n_components=2).fit_transform(data)
    np.testing.assert_allclose(
        np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=-1),
        np.linalg.norm(expected[:, None] - expected[None, :], axis=-1), atol=1e-5,
    )


@pytest.mark.requires_manylatents
@pytest.mark.parametrize("has_usable_traces", [True, False])
def test_readme_geometry_snippet_runs_on_trace_store(tmp_path, capsys, has_usable_traces):
    from manyagents.schemas.reasoning import TraceStore, ReasoningTrace, ReasoningStep
    from manylatents.metrics.trajectory_geometry import compute_cosine_velocity, compute_menger_curvature

    arrays = []
    with TraceStore(tmp_path / "traces") as store:
        # Distinct paths ensure concatenation would create spurious boundary motion.
        for i, length in enumerate([4, 5] if has_usable_traces else [1, 2]):
            data = np.random.default_rng(i).normal(size=(length, 2, 6)).astype(np.float16)
            trace = ReasoningTrace(trace_id=f"trace_{i}", steps=[
                ReasoningStep(index=j, text=f"Step {j}", layers_captured=[4, 8])
                for j in range(length)
            ])
            store.append(trace, hidden_states={"pooled_steps": data})
            arrays.append(data[:, 1, :].astype(np.float32))
        store.append(ReasoningTrace(trace_id="text_only"))

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    section = readme.split("## From traces to geometry", 1)[1]
    snippet = section.split("```python\n", 1)[1].split("```", 1)[0]
    snippet = snippet.replace("<output_dir>", str(tmp_path))
    namespace = {}
    if not has_usable_traces:
        with pytest.raises(SystemExit, match="No traces with at least three"):
            exec(compile(snippet, "README.md", "exec"), namespace)
        return
    exec(compile(snippet, "README.md", "exec"), namespace)
    # Grouped reductions exclude transitions between independent traces.
    expected_velocity = np.mean([np.mean(compute_cosine_velocity(a)) for a in arrays])
    expected_curvature = np.mean([np.mean(compute_menger_curvature(a)) for a in arrays])
    compute = namespace["compute_metric"]
    for name, expected in [("trajectory_velocity", expected_velocity), ("trajectory_curvature", expected_curvature)]:
        assert compute(name, namespace["X"], dataset=namespace["_Grouped"]()) == pytest.approx(expected)
    assert namespace["r"]["embeddings"].shape == (9, 2)
    assert "trace_0" in capsys.readouterr().out
