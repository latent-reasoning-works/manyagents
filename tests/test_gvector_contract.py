"""Measurements retain named outcomes; numeric padding is never a result."""

import json
import sys
from datetime import datetime
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

from manyagents.schemas import GVector, TransformationTrajectory
from manyagents.schemas.gvector import CORE_METRICS
from manyagents.workflows.sequence import compute_gvector


@pytest.fixture
def metric_result(monkeypatch):
    module = ModuleType("manylatents.metrics")
    module.compute_metric = Mock()
    monkeypatch.setitem(sys.modules, "manylatents.metrics", module)
    return module.compute_metric


def assert_failed_roundtrip(g, name):
    """Failures remain serializable and cannot contribute to numeric arrays."""
    for record in (g, GVector.from_json(g.to_json())):
        assert record.measurements[name] == {
            "status": "failed", "reason": "ValueError: Metric returned a non-finite value",
        }
        with pytest.raises(ValueError, match=f"{name}.*failed.*non-finite"):
            record.metric_value(name)
        with pytest.raises(ValueError, match=f"{name}.*failed"):
            record.to_array()
    # The outcome record, including its padding, is valid strict JSON.
    json.dumps(g.to_dict(), allow_nan=False)


@pytest.mark.parametrize("value", [
    np.array([]), np.nan, np.inf, -np.inf, np.array([1.0, np.nan]),
    (np.nan, np.ones(3)),
], ids=["empty-array", "nan", "inf", "negative-inf", "partial-nan-array", "nan-tuple"])
def test_compute_gvector_records_undefined_measurement(value, metric_result):
    metric_result.return_value = value
    g = compute_gvector(np.ones((3, 2)), ["participation_ratio"])
    assert_failed_roundtrip(g, "participation_ratio")


def test_omitted_gvector_metrics_roundtrip_as_not_requested(metric_result):
    metric_result.return_value = 0.0
    g = compute_gvector(np.ones((3, 2)), ["participation_ratio"])
    expected = {name: {"status": "not_requested"} for name in CORE_METRICS}
    expected["participation_ratio"] = {"status": "measured", "value": 0.0}
    assert json.loads(g.to_json())["measurements"] == expected
    for record in (g, GVector.from_json(g.to_json())):
        assert record.measurements == expected
        assert record.metric_value("participation_ratio") == 0.0
        assert all(isinstance(getattr(record, name), (int, float)) for name in CORE_METRICS)
        with pytest.raises(ValueError, match="beta_0.*not requested"):
            record.metric_value("beta_0")
        with pytest.raises(ValueError, match="not requested"):
            record.to_array()


def test_partial_gvector_trajectory_rejects_unavailable_deltas(metric_result):
    metric_result.side_effect = [1.0, 2.0]
    vectors = [compute_gvector(np.ones((3, 2)), ["participation_ratio"]) for _ in range(2)]
    trajectory = TransformationTrajectory(
        id="partial", workflow=[{"algorithm": "PCA"}], dataset_name="array",
        g_vectors=vectors, executed_at=datetime.now(), total_time_seconds=1.0,
        per_step_times=[0.0, 1.0],
    )
    for record in (trajectory, TransformationTrajectory.from_json(trajectory.to_json())):
        assert [g.metric_value("participation_ratio") for g in record.g_vectors] == [1.0, 2.0]
        with pytest.raises(ValueError, match="beta_0.*not requested"):
            _ = record.deltas


@pytest.mark.parametrize("field", CORE_METRICS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_measurements_are_recorded_for_each_field(field, value, metric_result):
    metric_result.side_effect = lambda name, *args, **kwargs: value if name == field else 0.0
    g = compute_gvector(np.ones((3, 2)), list(CORE_METRICS))
    assert_failed_roundtrip(g, field)
    for name in set(CORE_METRICS) - {field}:
        assert g.metric_value(name) == 0.0


def test_failure_reason_survives_strict_json(metric_result):
    metric_result.side_effect = RuntimeError("measurement unavailable")
    g = compute_gvector(np.ones((3, 2)), ["participation_ratio"])
    record = GVector.from_dict(json.loads(json.dumps(g.to_dict(), allow_nan=False)))
    assert record.measurements["participation_ratio"] == {
        "status": "failed", "reason": "RuntimeError: measurement unavailable",
    }
    with pytest.raises(ValueError, match="participation_ratio.*measurement unavailable"):
        record.to_array()


@pytest.mark.parametrize("value, expected", [
    (2.0, 2.0), (np.array([1.0, 3.0]), 2.0), ((2.0, np.array([1.0, 3.0])), 2.0),
])
def test_finite_gvector_measurements_still_supported(value, expected, metric_result):
    metric_result.return_value = value
    g = compute_gvector(np.ones((3, 2)), ["participation_ratio"])
    assert g.metric_value("participation_ratio") == expected
    assert GVector.from_json(g.to_json()).metric_value("participation_ratio") == expected


def test_inconsistent_deserialization_is_rejected():
    record = GVector(0, 0, 0.0, 0.0).to_dict()
    record["measurements"]["beta_0"]["value"] = 10
    with pytest.raises(ValueError, match="beta_0.*inconsistent"):
        GVector.from_dict(record)


@pytest.mark.parametrize("mutation", ["attribute", "metadata"])
@pytest.mark.parametrize("reader", ["metric", "array", "json", "deltas"])
def test_mutated_values_cannot_disagree(mutation, reader):
    previous = GVector(0, 0, 0.0, 0.0)
    current = GVector(0, 0, 0.0, 0.0)
    if mutation == "attribute":
        current.beta_0 = 10
    else:
        current.measurements["beta_0"]["value"] = 10
    trajectory = TransformationTrajectory(
        id="mutated", workflow=[{}], dataset_name="array", g_vectors=[previous, current],
        executed_at=datetime.now(), total_time_seconds=1, per_step_times=[0, 1],
    )
    with pytest.raises(ValueError, match="beta_0.*inconsistent"):
        if reader == "metric":
            current.metric_value("beta_0")
        elif reader == "array":
            current.to_array()
        elif reader == "json":
            current.to_json()
        else:
            _ = trajectory.deltas


@pytest.mark.parametrize("metadata", [None, "absent"])
def test_legacy_provenance_stays_unknown_through_roundtrip(metadata):
    record = {name: 0 for name in CORE_METRICS}
    if metadata is None:
        record["measurements"] = None
    legacy = GVector.from_dict(record)
    for g in [legacy, GVector.from_json(legacy.to_json())]:
        assert all(g.measurements[name] == {"status": "unknown"} for name in CORE_METRICS)
        with pytest.raises(ValueError, match="unknown"):
            g.to_array()
