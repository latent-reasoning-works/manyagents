"""GVector: Geometric summary of a dataset/embedding state."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np


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
    """

    beta_0: int
    beta_1: int
    participation_ratio: float
    local_intrinsic_dim: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in canonical order.

        Order: [beta_0, beta_1, participation_ratio, local_intrinsic_dim]

        Returns:
            1D numpy array of shape (4,).
        """
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
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GVector:
        """Create GVector from dictionary."""
        return cls(
            beta_0=int(d["beta_0"]),
            beta_1=int(d["beta_1"]),
            participation_ratio=float(d["participation_ratio"]),
            local_intrinsic_dim=float(d["local_intrinsic_dim"]),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> GVector:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))
