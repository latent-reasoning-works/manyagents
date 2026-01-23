# tests/schemas/test_trajectory.py
"""E2E tests for Phase A1: GVector and TransformationTrajectory."""
from datetime import datetime


def test_gvector_to_array():
    """GVector converts to numpy array."""
    from manyagents.schemas.gvector import GVector

    g = GVector(beta_0=5, beta_1=1, participation_ratio=0.5, local_intrinsic_dim=3.2)
    arr = g.to_array()

    assert arr.shape == (4,)
    assert arr[0] == 5  # beta_0


def test_trajectory_computes_deltas():
    """Trajectory computes G-vector deltas between steps."""
    from manyagents.schemas.trajectory import TransformationTrajectory
    from manyagents.schemas.gvector import GVector

    traj = TransformationTrajectory(
        id="test-123",
        workflow=[{"algorithm": "PCA"}, {"algorithm": "UMAP"}],
        dataset_name="test",
        g_vectors=[
            GVector(beta_0=10, beta_1=2, participation_ratio=0.1, local_intrinsic_dim=50),
            GVector(beta_0=5, beta_1=1, participation_ratio=0.3, local_intrinsic_dim=20),
            GVector(beta_0=1, beta_1=0, participation_ratio=0.6, local_intrinsic_dim=2),
        ],
        executed_at=datetime.now(),
        total_time_seconds=5.0,
        per_step_times=[0.0, 2.0, 3.0],
    )

    deltas = traj.deltas

    assert len(deltas) == 2  # Two transitions
    assert deltas[0]["beta_0"] == -5  # 5 - 10
    assert deltas[1]["beta_0"] == -4  # 1 - 5


def test_trajectory_serialization():
    """Trajectory round-trips through JSON."""
    from manyagents.schemas.trajectory import TransformationTrajectory
    from manyagents.schemas.gvector import GVector

    traj = TransformationTrajectory(
        id="test-456",
        workflow=[{"algorithm": "PCA", "params": {"n_components": 10}}],
        dataset_name="swissroll",
        g_vectors=[
            GVector(beta_0=10, beta_1=0, participation_ratio=0.2, local_intrinsic_dim=10),
            GVector(beta_0=1, beta_1=0, participation_ratio=0.5, local_intrinsic_dim=3),
        ],
        executed_at=datetime.now(),
        total_time_seconds=1.5,
        per_step_times=[0.0, 1.5],
    )

    # Round-trip
    json_str = traj.to_json()
    loaded = TransformationTrajectory.from_json(json_str)

    assert loaded.id == traj.id
    assert len(loaded.g_vectors) == 2
    assert loaded.g_vectors[0].beta_0 == 10
