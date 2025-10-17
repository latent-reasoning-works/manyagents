"""Core metrics computation feature tightly integrated with manylatents.metrics.api."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from manylatents.metrics.api import compute_metrics, list_available_metrics

log = logging.getLogger(__name__)


class MetricComputer:
    """
    Core manyAgents feature for computing geometric metrics on embeddings.

    IMPORTANT: This is ONLY for workflows that do NOT use manylatents as the
    embedding algorithm. When using manylatents (via ManyLatentsAdapter), metrics
    are automatically computed within the manylatents process itself.

    Use this for:
    - Custom dimensionality reduction algorithms
    - External tools (CellForge, BioDiscoveryAgent outputs)
    - Non-manylatents workflows that need geometric auditing

    Tightly integrated with manylatents.metrics.api for high-performance,
    in-memory metric calculation.
    """

    def __init__(self):
        self.available_metrics = list_available_metrics()
        log.debug(f"MetricComputer initialized with {len(self.available_metrics)} metrics")

    def compute(
        self,
        x: np.ndarray,
        embeddings: Dict[str, np.ndarray],
        metric_names: list[str],
        metric_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for multiple embeddings.

        Args:
            x: High-dimensional original data (n_samples, n_features_high)
            embeddings: Dictionary mapping run IDs to embedding arrays
            metric_names: List of metric names to compute (e.g., ['Trustworthiness'])
            metric_params: Optional parameters for metrics (e.g., {'n_neighbors': 10})

        Returns:
            Dictionary mapping run IDs to their metric scores

        Example:
            >>> computer = MetricComputer()
            >>> scores = computer.compute(
            ...     x=original_data,
            ...     embeddings={'baseline': z1, 'variant': z2},
            ...     metric_names=['Trustworthiness', 'Continuity'],
            ...     metric_params={'n_neighbors': 10}
            ... )
            >>> scores
            {'baseline': {'Trustworthiness': 0.85, 'Continuity': 0.82},
             'variant': {'Trustworthiness': 0.88, 'Continuity': 0.84}}
        """
        all_scores = {}

        for run_id, z in embeddings.items():
            log.info(f"Computing metrics for run '{run_id}'")
            scores = compute_metrics(
                x=x,
                z=z,
                metric_names=metric_names,
                metric_params=metric_params,
            )
            all_scores[run_id] = scores

        return all_scores

    def compute_reward(
        self,
        scores: Dict[str, Dict[str, float]],
        reward_config: Dict[str, Any],
    ) -> float:
        """
        Calculate a reward signal from metric scores.

        Args:
            scores: Dictionary of {run_id: {metric_name: score}}
            reward_config: Reward function configuration

        Returns:
            Scalar reward value
        """
        reward_type = reward_config.get("type")

        if reward_type == "trustworthiness_diff":
            baseline_id = reward_config["baseline_id"]
            variant_id = reward_config["variant_id"]

            baseline = scores[baseline_id]["Trustworthiness"]
            variant = scores[variant_id]["Trustworthiness"]

            return variant - baseline

        elif reward_type == "weighted_sum":
            run_id = reward_config["run_id"]
            weights = reward_config["weights"]

            return sum(
                scores[run_id][metric] * weight
                for metric, weight in weights.items()
            )

        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
