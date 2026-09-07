# tests/workflows/test_sequence.py
"""E2E tests for Phase A2: execute_sequence()."""
import numpy as np
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def small_data():
    """Small synthetic dataset for fast testing."""
    # 300 points in 10D - small enough for fast persistent homology
    np.random.seed(42)
    return np.random.randn(300, 10)


@pytest.mark.requires_manylatents
def test_execute_sequence_single_step(small_data):
    """Single-step workflow returns 2 G-vectors (raw + result)."""
    from manyagents.workflows.sequence import execute_sequence

    result = execute_sequence(
        workflow=[{"algorithm": "PCA", "params": {"n_components": 5}}],
        dataset=small_data,
        metrics=["beta_0", "participation_ratio"],
    )

    assert len(result.g_vectors) == 2  # raw + after PCA
    assert result.g_vectors[0].beta_0 is not None
    assert result.g_vectors[1].beta_0 is not None
    assert result.dataset_name == "array"
    assert len(result.workflow) == 1


@pytest.mark.requires_manylatents
def test_execute_sequence_multi_step(small_data):
    """Multi-step workflow returns N+1 G-vectors."""
    from manyagents.workflows.sequence import execute_sequence

    result = execute_sequence(
        workflow=[
            {"algorithm": "PCA", "params": {"n_components": 5}},
            {"algorithm": "UMAP", "params": {"n_components": 2}},
        ],
        dataset=small_data,
        metrics=["beta_0", "beta_1", "participation_ratio", "local_intrinsic_dim"],
    )

    assert len(result.g_vectors) == 3  # raw + PCA + UMAP
    assert len(result.per_step_times) == 3
    assert result.total_time_seconds > 0

    # Metrics should change across steps (dimensionality changes)
    assert result.g_vectors[0].participation_ratio != result.g_vectors[2].participation_ratio


@pytest.mark.requires_manylatents
def test_execute_sequence_saves_embeddings(small_data):
    """Embeddings saved when requested."""
    from manyagents.workflows.sequence import execute_sequence

    with tempfile.TemporaryDirectory() as tmpdir:
        result = execute_sequence(
            workflow=[{"algorithm": "PCA", "params": {"n_components": 5}}],
            dataset=small_data,
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


@pytest.mark.requires_manylatents
def test_execute_sequence_with_named_dataset():
    """Can load dataset by name (uses smaller subset for speed)."""
    from manyagents.workflows.sequence import execute_sequence
    from manylatents.data import get_dataset

    # Get swissroll and subsample for speed
    sr = get_dataset("swissroll")
    small_sr = sr.data[:300]

    result = execute_sequence(
        workflow=[{"algorithm": "PCA", "params": {"n_components": 2}}],
        dataset=small_sr,
        metrics=["participation_ratio"],  # Skip beta for speed
    )

    assert len(result.g_vectors) == 2
    assert result.g_vectors[0].participation_ratio > 0
