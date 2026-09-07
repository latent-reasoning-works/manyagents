"""Execute DR workflow sequences with G-vector tracking."""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from manyagents.schemas import GVector, TransformationTrajectory
from manyagents.schemas.gvector import CORE_METRICS

logger = logging.getLogger(__name__)

# Metric name mapping: user-friendly names -> registry names
METRIC_ALIASES = {
    "beta_0": "beta_0",
    "beta_1": "beta_1",
    "participation_ratio": "participation_ratio",
    "local_intrinsic_dim": "local_intrinsic_dim",
    "trustworthiness": "trustworthiness",
}


def _load_dataset(dataset: Union[str, np.ndarray]) -> tuple[np.ndarray, Any]:
    """Load dataset by name or pass through numpy array.

    Returns:
        Tuple of (data_array, dataset_object).
        dataset_object is needed for metrics that require original data (e.g., trustworthiness).
    """
    if isinstance(dataset, np.ndarray):
        return dataset, None

    from manylatents.data import get_dataset

    dataset_obj = get_dataset(dataset)

    # Dataset objects have a .data attribute (numpy array)
    data = dataset_obj.data

    # Convert torch tensor to numpy if needed
    if hasattr(data, "numpy"):
        data = data.numpy()
    elif hasattr(data, "detach"):
        data = data.detach().cpu().numpy()

    return data, dataset_obj


def _get_algorithm_class(algorithm: str):
    """Get algorithm class from manyLatents registry."""
    # Use the adapter's registry for algorithm lookup
    from manyagents.adapters.metric_registry import get_metric_registry

    registry = get_metric_registry()
    return registry.get_algorithm_class(algorithm.upper())


def _instantiate_algorithm(algorithm: str, params: Dict[str, Any]):
    """Instantiate a DR algorithm with given parameters."""
    algo_class = _get_algorithm_class(algorithm)

    # Get defaults from registry
    from manyagents.adapters.metric_registry import get_metric_registry

    registry = get_metric_registry()
    defaults = registry.get_algorithm_defaults(algorithm.upper())

    # Merge defaults with provided params
    final_params = {**defaults, **params}

    logger.debug(f"Instantiating {algorithm} with params: {final_params}")
    return algo_class(**final_params)


def compute_gvector(
    embedding: np.ndarray,
    metrics: List[str],
    dataset_obj: Optional[Any] = None,
) -> GVector:
    """Compute G-vector from embedding using requested metrics.

    Args:
        embedding: The embedding array (n_samples, n_features).
        metrics: List of metric names to compute.
        dataset_obj: Optional dataset object for metrics requiring original data.

    Returns:
        GVector with explicit measurement outcomes. Failed requests carry their
        reasons in measurements; unrequested core metrics are marked separately.
        Unavailable numeric fields contain padding, not measured zeros.
    """
    from manylatents.metrics import compute_metric

    values: Dict[str, Any] = {}
    measurements = {name: {"status": "not_requested"} for name in CORE_METRICS}

    for metric_name in metrics:
        # Map to registry name if needed
        registry_name = METRIC_ALIASES.get(metric_name, metric_name)

        try:
            result = compute_metric(registry_name, embedding, dataset=dataset_obj)

            # Handle different return types
            if isinstance(result, tuple):
                # (scalar, per_sample_array) -> take scalar
                value = float(result[0])
            elif isinstance(result, np.ndarray):
                # Per-sample array -> take mean
                value = float(np.mean(result))
            else:
                value = float(result)

            if not np.isfinite(value):
                raise ValueError("Metric returned a non-finite value")
            if metric_name in ("beta_0", "beta_1"):
                value = int(value)

            values[metric_name] = value
            measurements[metric_name] = {"status": "measured", "value": value}

            logger.debug(f"  {metric_name} = {values[metric_name]:.4f}")

        except Exception as e:
            logger.warning(f"Failed to compute metric '{metric_name}': {e}")
            values.pop(metric_name, None)
            measurements[metric_name] = {
                "status": "failed", "reason": f"{type(e).__name__}: {e}",
            }

    # The fixed numeric contract needs padding; outcomes determine validity.
    return GVector(
        beta_0=int(values.get("beta_0", 0)),
        beta_1=int(values.get("beta_1", 0)),
        participation_ratio=float(values.get("participation_ratio", 0.0)),
        local_intrinsic_dim=float(values.get("local_intrinsic_dim", 0.0)),
        measurements=measurements,
    )


def execute_sequence(
    workflow: List[Dict[str, Any]],
    dataset: Union[str, np.ndarray],
    metrics: Optional[List[str]] = None,
    save_embeddings: bool = False,
    embedding_dir: Optional[Path] = None,
) -> TransformationTrajectory:
    """Execute a DR workflow sequence, capturing G-vectors at each step.

    Args:
        workflow: List of algorithm specs, e.g.:
            [{"algorithm": "PCA", "params": {"n_components": 50}},
             {"algorithm": "UMAP", "params": {"n_components": 2}}]
        dataset: Dataset name (e.g., "swissroll") or numpy array.
        metrics: Metrics to compute for G-vectors. Defaults to all core metrics.
        save_embeddings: Whether to save embeddings at each step.
        embedding_dir: Directory to save embeddings (required if save_embeddings=True).

    Returns:
        TransformationTrajectory with G-vectors at each step.
    """
    if metrics is None:
        metrics = ["beta_0", "beta_1", "participation_ratio", "local_intrinsic_dim"]

    if save_embeddings and embedding_dir is None:
        raise ValueError("embedding_dir required when save_embeddings=True")

    # Generate unique ID for this trajectory
    traj_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"Starting sequence execution: {len(workflow)} steps, metrics={metrics}")

    # Load dataset
    data, dataset_obj = _load_dataset(dataset)
    dataset_name = dataset if isinstance(dataset, str) else "array"

    logger.info(f"Loaded dataset '{dataset_name}': shape={data.shape}")

    # Track G-vectors, embeddings, and times
    g_vectors: List[GVector] = []
    embedding_paths: Optional[List[Path]] = [] if save_embeddings else None
    per_step_times: List[float] = []

    # Current embedding starts as raw data
    current_embedding = data

    # Step 0: Compute G-vector on raw data
    step_start = time.time()
    logger.info("Step 0: Computing G-vector on raw data")
    g0 = compute_gvector(current_embedding, metrics, dataset_obj)
    g_vectors.append(g0)
    per_step_times.append(0.0)  # Raw data has no computation time

    if save_embeddings:
        path = embedding_dir / f"{traj_id}_step0_raw.npy"
        np.save(path, current_embedding)
        embedding_paths.append(path)
        logger.debug(f"Saved raw embedding to {path}")

    # Execute each workflow step
    for i, step in enumerate(workflow, start=1):
        algorithm = step.get("algorithm")
        params = step.get("params", {})

        logger.info(f"Step {i}: {algorithm} with params={params}")
        step_start = time.time()

        # Instantiate and run algorithm
        algo = _instantiate_algorithm(algorithm, params)

        # Convert to torch tensor for manyLatents algorithms
        import torch

        if isinstance(current_embedding, np.ndarray):
            input_tensor = torch.from_numpy(current_embedding).float()
        else:
            input_tensor = current_embedding

        current_embedding = algo.fit_transform(input_tensor)

        # Convert to numpy if needed
        if hasattr(current_embedding, "numpy"):
            current_embedding = current_embedding.numpy()
        elif hasattr(current_embedding, "detach"):
            current_embedding = current_embedding.detach().cpu().numpy()

        step_time = time.time() - step_start
        per_step_times.append(step_time)

        logger.info(f"  Output shape: {current_embedding.shape}, time: {step_time:.2f}s")

        # Compute G-vector
        g = compute_gvector(current_embedding, metrics, dataset_obj)
        g_vectors.append(g)

        if save_embeddings:
            algo_name = algorithm.lower()
            path = embedding_dir / f"{traj_id}_step{i}_{algo_name}.npy"
            np.save(path, current_embedding)
            embedding_paths.append(path)
            logger.debug(f"Saved embedding to {path}")

    total_time = time.time() - start_time
    logger.info(f"Sequence complete: {len(g_vectors)} G-vectors, total time: {total_time:.2f}s")

    return TransformationTrajectory(
        id=traj_id,
        workflow=workflow,
        dataset_name=dataset_name,
        g_vectors=g_vectors,
        executed_at=datetime.now(),
        total_time_seconds=total_time,
        per_step_times=per_step_times,
        embedding_paths=embedding_paths,
    )
