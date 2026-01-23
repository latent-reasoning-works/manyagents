# tests/workflows/test_sequence.py
"""E2E tests for Phase A2: execute_sequence()."""
import numpy as np
import tempfile
from pathlib import Path


def test_execute_sequence_single_step():
    """Single-step workflow returns 2 G-vectors (raw + result)."""
    from manyagents.workflows.sequence import execute_sequence

    result = execute_sequence(
        workflow=[{"algorithm": "PCA", "params": {"n_components": 10}}],
        dataset="swissroll",
        metrics=["beta_0", "participation_ratio"],
    )

    assert len(result.g_vectors) == 2  # raw + after PCA
    assert result.g_vectors[0].beta_0 is not None
    assert result.g_vectors[1].beta_0 is not None
    assert result.dataset_name == "swissroll"
    assert len(result.workflow) == 1


def test_execute_sequence_multi_step():
    """Multi-step workflow returns N+1 G-vectors."""
    from manyagents.workflows.sequence import execute_sequence

    result = execute_sequence(
        workflow=[
            {"algorithm": "PCA", "params": {"n_components": 50}},
            {"algorithm": "UMAP", "params": {"n_components": 2}},
        ],
        dataset="swissroll",
        metrics=["beta_0", "beta_1", "participation_ratio", "local_intrinsic_dim"],
    )

    assert len(result.g_vectors) == 3  # raw + PCA + UMAP
    assert len(result.per_step_times) == 3
    assert result.total_time_seconds > 0

    # Metrics should change across steps
    assert result.g_vectors[0].participation_ratio != result.g_vectors[2].participation_ratio


def test_execute_sequence_saves_embeddings():
    """Embeddings saved when requested."""
    from manyagents.workflows.sequence import execute_sequence

    with tempfile.TemporaryDirectory() as tmpdir:
        result = execute_sequence(
            workflow=[{"algorithm": "PCA", "params": {"n_components": 5}}],
            dataset="swissroll",
            metrics=["beta_0"],
            save_embeddings=True,
            embedding_dir=Path(tmpdir),
        )

        assert result.embedding_paths is not None
        assert len(result.embedding_paths) == 2  # raw + PCA
        for path in result.embedding_paths:
            assert path.exists()
            emb = np.load(path)
            assert emb.ndim == 2
