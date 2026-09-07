"""Undefined GVector measurements must remain explicit and serializable."""

import json
import sys
from datetime import datetime
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

from manyagents.schemas import GVector, TransformationTrajectory
from manyagents.workflows.sequence import compute_gvector


@pytest.fixture
def metric_result(monkeypatch):
    module = ModuleType("manylatents.metrics")
    module.compute_metric = Mock()
    monkeypatch.setitem(sys.modules, "manylatents.metrics", module)
    return module.compute_metric


@pytest.mark.parametrize("value", [
    np.array([]), np.nan, np.inf, -np.inf, np.array([1.0, np.nan]),
    (np.nan, np.ones(3)),
], ids=["empty-array", "nan", "inf", "negative-inf", "partial-nan-array", "nan-tuple"])
def test_compute_gvector_rejects_undefined_measurement(value, metric_result):
    metric_result.return_value = value
    with pytest.raises(ValueError, match="participation_ratio"):
        compute_gvector(np.ones((3, 2)), ["participation_ratio"])


def test_omitted_gvector_metrics_roundtrip_as_null(metric_result):
    metric_result.return_value = 0.0
    g = compute_gvector(np.ones((3, 2)), ["participation_ratio"])
    expected = {"beta_0": None, "beta_1": None, "participation_ratio": 0.0, "local_intrinsic_dim": None}
    assert g.to_dict() == expected
    assert json.loads(g.to_json()) == expected
    assert GVector.from_json(g.to_json()).to_dict() == expected
    with pytest.raises(ValueError, match="undefined"):
        g.to_array()


def test_partial_gvector_trajectory_preserves_undefined_deltas(metric_result):
    metric_result.side_effect = [1.0, 2.0]
    vectors = [compute_gvector(np.ones((3, 2)), ["participation_ratio"]) for _ in range(2)]
    trajectory = TransformationTrajectory(
        id="partial", workflow=[{"algorithm": "PCA"}], dataset_name="array",
        g_vectors=vectors, executed_at=datetime.now(), total_time_seconds=1.0,
        per_step_times=[0.0, 1.0],
    )
    expected = {"beta_0": None, "beta_1": None, "participation_ratio": 1.0, "local_intrinsic_dim": None}
    assert trajectory.deltas == [expected]
    assert TransformationTrajectory.from_json(trajectory.to_json()).deltas == [expected]


@pytest.mark.parametrize("field", ["beta_0", "beta_1", "participation_ratio", "local_intrinsic_dim"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_gvector_rejects_nonfinite_fields(field, value):
    values = {"beta_0": 1, "beta_1": 0, "participation_ratio": 2.0, "local_intrinsic_dim": 3.0}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        GVector(**values)


def test_gvector_serialization_rejects_nonfinite_mutation():
    g = GVector(1, 0, 2.0, 3.0)
    g.participation_ratio = np.nan
    with pytest.raises(ValueError):
        g.to_json()


@pytest.mark.parametrize("value, expected", [
    (2.0, 2.0), (np.array([1.0, 3.0]), 2.0), ((2.0, np.array([1.0, 3.0])), 2.0),
])
def test_finite_gvector_measurements_still_supported(value, expected, metric_result):
    metric_result.return_value = value
    g = compute_gvector(np.ones((3, 2)), ["participation_ratio"])
    assert g.participation_ratio == expected
