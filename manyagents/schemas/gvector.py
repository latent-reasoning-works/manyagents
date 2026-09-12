"""GVector: Geometric summary of a dataset/embedding state."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np


CORE_METRICS = ("beta_0", "beta_1", "participation_ratio", "local_intrinsic_dim")


@dataclass
class GVector:
    """Geometric vector summarizing dataset properties at a point in the workflow.

    Captures topological (Betti numbers) and geometric (participation ratio,
    local intrinsic dimension) properties of the data.

    Attributes:
        beta_0: Betti-0 number (connected components).
        beta_1: Betti-1 number (loops/holes).
        participation_ratio: Measure of effective dimensionality.
        local_intrinsic_dim: Local intrinsic dimension estimate.
        measurements: Outcomes keyed by metric name: measured (with value),
            failed (with reason), not_requested, or unknown. Numeric fields for unavailable
            metrics are padding only; use metric_value() to read measurements.
            Direct construction without metadata asserts that supplied values are
            measured. Legacy deserialization without metadata marks them unknown.
            Measured fields must agree with metadata at construction and every
            measurement read/serialization; update both together or build a new vector.
    """

    beta_0: int
    beta_1: int
    participation_ratio: float
    local_intrinsic_dim: float
    measurements: Optional[dict[str, dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if self.measurements is None:
            self.measurements = {
                name: {"status": "measured", "value": getattr(self, name)}
                for name in CORE_METRICS
            }
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        """Reject disagreement, including edits to the nested outcome mapping."""
        if not isinstance(self.measurements, dict):
            raise ValueError("measurements must be an outcome mapping")
        for name, outcome in self.measurements.items():
            if outcome.get("status") == "measured":
                value = outcome.get("value")
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"Metric '{name}' must have a finite measured value")
                if name in CORE_METRICS and getattr(self, name) != value:
                    raise ValueError(f"Metric '{name}' has inconsistent numeric field and measurement")

    def metric_value(self, name: str) -> float:
        """Read a measured value, raising with the reason if unavailable."""
        self._validate_consistency()
        outcome = self.measurements.get(name, {"status": "not_requested"})
        if outcome["status"] == "failed":
            raise ValueError(f"Metric '{name}' failed: {outcome['reason']}")
        if outcome["status"] == "not_requested":
            raise ValueError(f"Metric '{name}' was not requested")
        if outcome["status"] == "unknown":
            raise ValueError(f"Metric '{name}' has unknown legacy provenance")
        if outcome["status"] != "measured":
            raise ValueError(f"Invalid measurement status for '{name}': {outcome['status']}")
        return outcome["value"]

    def require_complete(self) -> None:
        """Reject failed requests and absent core metrics before arithmetic."""
        for name, outcome in self.measurements.items():
            if outcome["status"] == "failed":
                self.metric_value(name)
        for name in CORE_METRICS:
            self.metric_value(name)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in canonical order.

        Order: [beta_0, beta_1, participation_ratio, local_intrinsic_dim]

        Returns:
            1D numpy array of shape (4,).

        Raises:
            ValueError: A requested metric failed or a core metric is unavailable.
        """
        self.require_complete()
        return np.array([self.metric_value(name) for name in CORE_METRICS], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> GVector:
        """Create GVector from numpy array.

        Args:
            arr: 1D array with 4 elements in canonical order.

        Returns:
            GVector instance.
        """
        if arr.shape != (4,):
            raise ValueError(f"Expected array of shape (4,), got {arr.shape}")
        return cls(
            beta_0=int(arr[0]),
            beta_1=int(arr[1]),
            participation_ratio=float(arr[2]),
            local_intrinsic_dim=float(arr[3]),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        self._validate_consistency()
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GVector:
        """Load a record; missing/null metadata means unknown legacy provenance.

        Numeric fields remain available for inspection. To use them as measurements,
        the caller must verify their provenance and supply explicit outcomes.
        """
        return cls(
            beta_0=int(d["beta_0"]),
            beta_1=int(d["beta_1"]),
            participation_ratio=float(d["participation_ratio"]),
            local_intrinsic_dim=float(d["local_intrinsic_dim"]),
            measurements=(
                d["measurements"] if d.get("measurements") is not None
                else {name: {"status": "unknown"} for name in CORE_METRICS}
            ),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> GVector:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))
