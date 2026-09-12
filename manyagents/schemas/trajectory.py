"""TransformationTrajectory: Sequence of G-vectors across a workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from manyagents.schemas.gvector import CORE_METRICS, GVector


@dataclass
class TransformationTrajectory:
    """Record of a workflow execution with G-vectors at each step.

    Captures the geometric transformation trajectory as algorithms are
    applied sequentially. Each G-vector represents the state after
    applying the corresponding step (step 0 = raw data).

    Attributes:
        id: Unique identifier for this trajectory.
        workflow: List of algorithm specs (dicts with 'algorithm' and 'params').
        dataset_name: Name of the dataset used.
        g_vectors: G-vectors at each step. Length = len(workflow) + 1
            (includes raw data at index 0). Each vector retains named measurement
            outcomes, including failure reasons and unrequested metrics.
        executed_at: When the workflow was executed.
        total_time_seconds: Total execution time.
        per_step_times: Time taken for each step. Length = len(workflow) + 1
            (first is 0.0 for raw data).
        embedding_paths: Optional paths to saved embeddings at each step.
    """

    id: str
    workflow: list[dict[str, Any]]
    dataset_name: str
    g_vectors: list[GVector]
    executed_at: datetime
    total_time_seconds: float
    per_step_times: list[float]
    embedding_paths: Optional[list[Path]] = None

    @property
    def deltas(self) -> list[dict[str, float]]:
        """Compute G-vector deltas between consecutive steps.

        Returns:
            List of dicts with metric deltas. Length = len(g_vectors) - 1.
            Each dict has keys: beta_0, beta_1, participation_ratio, local_intrinsic_dim.

        Raises:
            ValueError: Either step has failed or unavailable measurements.
        """
        result = []
        for i in range(1, len(self.g_vectors)):
            prev = self.g_vectors[i - 1]
            curr = self.g_vectors[i]
            prev.require_complete()
            curr.require_complete()
            result.append({
                name: curr.metric_value(name) - prev.metric_value(name)
                for name in CORE_METRICS
            })
        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "workflow": self.workflow,
            "dataset_name": self.dataset_name,
            "g_vectors": [g.to_dict() for g in self.g_vectors],
            "executed_at": self.executed_at.isoformat(),
            "total_time_seconds": self.total_time_seconds,
            "per_step_times": self.per_step_times,
            "embedding_paths": (
                [str(p) for p in self.embedding_paths]
                if self.embedding_paths
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TransformationTrajectory:
        """Create from dictionary."""
        embedding_paths = None
        if d.get("embedding_paths"):
            embedding_paths = [Path(p) for p in d["embedding_paths"]]

        return cls(
            id=d["id"],
            workflow=d["workflow"],
            dataset_name=d["dataset_name"],
            g_vectors=[GVector.from_dict(g) for g in d["g_vectors"]],
            executed_at=datetime.fromisoformat(d["executed_at"]),
            total_time_seconds=d["total_time_seconds"],
            per_step_times=d["per_step_times"],
            embedding_paths=embedding_paths,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> TransformationTrajectory:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))
