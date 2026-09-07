"""Measurement failures must survive workflow recording and serialization."""
from unittest.mock import Mock

import numpy as np
import pytest

from manyagents.schemas import GVector, TransformationTrajectory
from manyagents.workflows.sequence import compute_gvector, execute_sequence


CORE_METRICS = ["beta_0", "beta_1", "participation_ratio", "local_intrinsic_dim"]


@pytest.mark.parametrize("failed_metric", CORE_METRICS + ["trustworthiness"])
def test_metric_failure_is_recorded_beside_successes(monkeypatch, failed_metric):
    """A failed measurement has a name and reason, never a measured zero."""
    def measure(name, *args, **kwargs):
        if name == failed_metric:
            raise RuntimeError("injected metric failure")
        return 0.0

    monkeypatch.setattr("manylatents.metrics.compute_metric", measure)
    result = execute_sequence([], np.zeros((3, 2)), metrics=CORE_METRICS + [failed_metric])

    for record in (result, TransformationTrajectory.from_json(result.to_json())):
        g = record.g_vectors[0]
        assert g.measurements[failed_metric] == {
            "status": "failed", "reason": "RuntimeError: injected metric failure",
        }
        with pytest.raises(ValueError, match=f"{failed_metric}.*injected metric failure"):
            g.metric_value(failed_metric)
        with pytest.raises(ValueError, match="injected metric failure"):
            g.to_array()
        for name in set(CORE_METRICS) - {failed_metric}:
            assert g.metric_value(name) == 0.0


@pytest.mark.parametrize("zero", [0.0, (0.0, np.zeros(3)), np.zeros(3)])
def test_real_zero_stays_numeric_after_round_trip(monkeypatch, zero):
    """Scalar, tuple, and per-sample zero results retain their existing meaning."""
    monkeypatch.setattr("manylatents.metrics.compute_metric", Mock(return_value=zero))
    g = compute_gvector(np.zeros((3, 2)), CORE_METRICS)

    for record in (g, GVector.from_json(g.to_json())):
        np.testing.assert_array_equal(record.to_array(), np.zeros(4))
        assert record.to_array().dtype == np.float64
        for name in CORE_METRICS:
            assert record.measurements[name] == {"status": "measured", "value": 0.0}
            assert record.metric_value(name) == 0.0


def test_unrequested_failure_and_zero_are_distinct(monkeypatch):
    """Requested failure, measured zero, and no request are distinct in JSON."""
    measure = Mock(side_effect=[RuntimeError("no measurement"), 0.0])
    monkeypatch.setattr("manylatents.metrics.compute_metric", measure)
    records = [
        execute_sequence([], np.zeros((3, 2)), metrics=metrics)
        for metrics in (["beta_0"], ["beta_0"], [])
    ]
    assert measure.call_count == 2
    for trajectories in (records, [TransformationTrajectory.from_json(r.to_json()) for r in records]):
        failed, zero, unrequested = [r.g_vectors[0] for r in trajectories]
        assert failed != zero != unrequested != failed
        assert failed.measurements["beta_0"]["status"] == "failed"
        assert zero.metric_value("beta_0") == 0
        assert unrequested.measurements["beta_0"] == {"status": "not_requested"}
        with pytest.raises(ValueError, match="beta_0.*not requested"):
            unrequested.metric_value("beta_0")
        with pytest.raises(ValueError, match="not requested"):
            unrequested.to_array()


def test_failure_after_transformation_blocks_deltas(monkeypatch):
    """Post-transform failures are retained and cannot become numeric deltas."""
    measure = Mock(side_effect=[1.0] * 4 + [RuntimeError("step failed")] + [0.0] * 3)
    monkeypatch.setattr("manylatents.metrics.compute_metric", measure)
    monkeypatch.setattr(
        "manyagents.workflows.sequence._instantiate_algorithm",
        lambda *args: Mock(fit_transform=lambda data: data),
    )
    result = execute_sequence([{"algorithm": "PCA"}], np.zeros((3, 2)))

    for record in (result, TransformationTrajectory.from_json(result.to_json())):
        assert record.g_vectors[0].metric_value("beta_0") == 1
        assert record.g_vectors[1].measurements["beta_0"]["status"] == "failed"
        with pytest.raises(ValueError, match="beta_0.*step failed"):
            _ = record.deltas


def test_unrequested_metrics_cannot_become_deltas(monkeypatch):
    """Numeric padding for unrequested metrics cannot enter arithmetic."""
    monkeypatch.setattr("manylatents.metrics.compute_metric", Mock(return_value=0.0))
    monkeypatch.setattr(
        "manyagents.workflows.sequence._instantiate_algorithm",
        lambda *args: Mock(fit_transform=lambda data: data),
    )
    result = execute_sequence([{"algorithm": "PCA"}], np.zeros((3, 2)), metrics=["beta_0"])
    with pytest.raises(ValueError, match="beta_1.*not requested"):
        _ = result.deltas


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "invalid"])
def test_invalid_metric_results_are_named_failures(monkeypatch, value):
    """Conversion errors and non-finite results cannot escape as numeric signals."""
    monkeypatch.setattr("manylatents.metrics.compute_metric", Mock(return_value=value))
    g = compute_gvector(np.zeros((3, 2)), ["beta_0"])
    loaded = GVector.from_json(g.to_json())
    assert loaded.measurements["beta_0"]["status"] == "failed"
    assert loaded.measurements["beta_0"]["reason"].startswith("ValueError:")
    assert "value" not in loaded.measurements["beta_0"]
    with pytest.raises(ValueError, match="beta_0.*failed"):
        loaded.to_array()


def test_measured_zero_deltas_survive_serialization(monkeypatch):
    """Complete measured zeros still support trajectory arithmetic."""
    monkeypatch.setattr("manylatents.metrics.compute_metric", Mock(return_value=0.0))
    monkeypatch.setattr(
        "manyagents.workflows.sequence._instantiate_algorithm",
        lambda *args: Mock(fit_transform=lambda data: data),
    )
    result = execute_sequence([{"algorithm": "PCA"}], np.zeros((3, 2)))
    loaded = TransformationTrajectory.from_json(result.to_json())
    assert loaded.deltas == [{name: 0.0 for name in CORE_METRICS}]


def test_legacy_numeric_vectors_remain_readable():
    """Existing numeric-only records retain the original constructor contract."""
    g = GVector.from_dict({name: 0 for name in CORE_METRICS})
    np.testing.assert_array_equal(g.to_array(), np.zeros(4))
    assert GVector.from_array(g.to_array()) == g
