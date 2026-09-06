"""GVector: Geometric summary of a dataset/embedding state."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from numbers import Real

import numpy as np


@dataclass
class GVector:
    """Geometric vector summarizing dataset properties at a point in the workflow.

    Captures topological (Betti numbers) and geometric (participation ratio,
    local intrinsic dimension) properties of the data.
    Unmeasured fields are None; measured fields must be finite.

    Attributes:
        beta_0: Betti-0 number (connected components).
        beta_1: Betti-1 number (loops/holes).
        participation_ratio: Measure of effective dimensionality.
        local_intrinsic_dim: Local intrinsic dimension estimate.
    """

    beta_0: int | None
    beta_1: int | None
    participation_ratio: float | None
    local_intrinsic_dim: float | None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (not isinstance(value, Real) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite or None, got {value!r}")

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in canonical order.

        Order: [beta_0, beta_1, participation_ratio, local_intrinsic_dim]

        Returns:
            1D numpy array of shape (4,).
        """
        self._validate()
        if any(value is None for value in asdict(self).values()):
            raise ValueError("Cannot convert undefined GVector measurements to a numeric array")
        return np.array([
            self.beta_0,
            self.beta_1,
            self.participation_ratio,
            self.local_intrinsic_dim,
        ], dtype=np.float64)

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
        self._validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GVector:
        """Create GVector from dictionary."""
        return cls(
            beta_0=int(d["beta_0"]) if d["beta_0"] is not None else None,
            beta_1=int(d["beta_1"]) if d["beta_1"] is not None else None,
            participation_ratio=float(d["participation_ratio"]) if d["participation_ratio"] is not None else None,
            local_intrinsic_dim=float(d["local_intrinsic_dim"]) if d["local_intrinsic_dim"] is not None else None,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_json(cls, s: str) -> GVector:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))
