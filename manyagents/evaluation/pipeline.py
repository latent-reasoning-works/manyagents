"""Evaluation pipeline for comparing baseline vs variant embeddings."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from manyagents.metrics import MetricComputer

log = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Pipeline for evaluating and comparing embedding methods.

    This orchestrates the comparison between:
    1. Baseline workflow (usually manylatents with reference config)
    2. Variant workflow (any tool/config to compare against baseline)

    Computes geometric metrics and reward signals for RL training.
    """

    def __init__(self, metric_names: List[str], metric_params: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluation pipeline.

        Args:
            metric_names: List of geometric metrics to compute (e.g., ['Trustworthiness', 'Continuity'])
            metric_params: Optional parameters for metrics (e.g., {'n_neighbors': 10})
        """
        self.metric_names = metric_names
        self.metric_params = metric_params or {}
        self.metric_computer = MetricComputer()
        log.info(f"Initialized EvaluationPipeline with metrics: {metric_names}")

    def evaluate_embeddings(
        self,
        x_original: np.ndarray,
        baseline_embedding: np.ndarray,
        variant_embedding: np.ndarray,
        baseline_name: str = "baseline",
        variant_name: str = "variant",
    ) -> Dict[str, Any]:
        """
        Evaluate and compare two embeddings.

        Args:
            x_original: Original high-dimensional data
            baseline_embedding: Reference embedding (e.g., from manylatents PCA)
            variant_embedding: Comparison embedding (e.g., from external tool)
            baseline_name: Label for baseline
            variant_name: Label for variant

        Returns:
            Dictionary containing:
            - baseline_metrics: Metric scores for baseline
            - variant_metrics: Metric scores for variant
            - metric_deltas: Difference (variant - baseline) for each metric
            - reward: Scalar reward signal
            - summary: Human-readable summary
        """
        log.info(f"Evaluating embeddings: {baseline_name} vs {variant_name}")

        # Compute metrics for both embeddings
        embeddings = {
            baseline_name: baseline_embedding,
            variant_name: variant_embedding,
        }

        all_scores = self.metric_computer.compute(
            x=x_original,
            embeddings=embeddings,
            metric_names=self.metric_names,
            metric_params=self.metric_params,
        )

        baseline_metrics = all_scores[baseline_name]
        variant_metrics = all_scores[variant_name]

        # Compute deltas (improvement)
        metric_deltas = {
            metric: variant_metrics[metric] - baseline_metrics[metric]
            for metric in self.metric_names
        }

        # Simple reward: average improvement across all metrics
        reward = np.mean(list(metric_deltas.values()))

        # Build summary
        summary_lines = [f"Evaluation: {baseline_name} vs {variant_name}"]
        summary_lines.append(f"\n{baseline_name} metrics:")
        for metric, score in baseline_metrics.items():
            summary_lines.append(f"  {metric}: {score:.4f}")

        summary_lines.append(f"\n{variant_name} metrics:")
        for metric, score in variant_metrics.items():
            delta = metric_deltas[metric]
            sign = "+" if delta >= 0 else ""
            summary_lines.append(f"  {metric}: {score:.4f} ({sign}{delta:.4f})")

        summary_lines.append(f"\nReward (avg improvement): {reward:+.4f}")

        summary = "\n".join(summary_lines)
        log.info(summary)

        return {
            "baseline_metrics": baseline_metrics,
            "variant_metrics": variant_metrics,
            "metric_deltas": metric_deltas,
            "reward": reward,
            "summary": summary,
        }

    def evaluate_from_results(
        self,
        x_original: np.ndarray,
        baseline_result: Dict[str, Any],
        variant_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate embeddings from adapter results.

        Args:
            x_original: Original high-dimensional data
            baseline_result: Result dict from baseline adapter (must contain 'embeddings')
            variant_result: Result dict from variant adapter (must contain 'embeddings')

        Returns:
            Evaluation results dictionary
        """
        # Extract embeddings from adapter results
        # Handle both manylatents format (EmbeddingOutputs) and direct arrays
        baseline_emb = self._extract_embedding(baseline_result)
        variant_emb = self._extract_embedding(variant_result)

        return self.evaluate_embeddings(
            x_original=x_original,
            baseline_embedding=baseline_emb,
            variant_embedding=variant_emb,
        )

    def _extract_embedding(self, result: Dict[str, Any]) -> np.ndarray:
        """Extract embedding array from adapter result."""
        # Try output_files first (direct numpy array)
        if "output_files" in result and "embeddings" in result["output_files"]:
            emb = result["output_files"]["embeddings"]
            if isinstance(emb, np.ndarray):
                return emb

        # Try embeddings key (EmbeddingOutputs format)
        if "embeddings" in result:
            emb_data = result["embeddings"]
            if isinstance(emb_data, dict) and "embeddings" in emb_data:
                return emb_data["embeddings"]
            elif isinstance(emb_data, np.ndarray):
                return emb_data

        raise ValueError(
            "Could not extract embeddings from result. "
            "Result must contain 'output_files.embeddings' or 'embeddings' key."
        )

    def compute_custom_reward(
        self,
        baseline_metrics: Dict[str, float],
        variant_metrics: Dict[str, float],
        reward_config: Dict[str, Any],
    ) -> float:
        """
        Compute custom reward function.

        Args:
            baseline_metrics: Metric scores for baseline
            variant_metrics: Metric scores for variant
            reward_config: Reward function configuration

        Returns:
            Scalar reward value

        Example reward configs:
            # Weighted combination
            {'type': 'weighted', 'weights': {'Trustworthiness': 0.7, 'Continuity': 0.3}}

            # Single metric improvement
            {'type': 'single_metric', 'metric': 'Trustworthiness'}

            # Threshold-based
            {'type': 'threshold', 'metric': 'Trustworthiness', 'threshold': 0.85}
        """
        reward_type = reward_config.get("type", "weighted")

        if reward_type == "weighted":
            weights = reward_config.get("weights", {})
            # Default to equal weights if not specified
            if not weights:
                weights = {m: 1.0 / len(self.metric_names) for m in self.metric_names}

            reward = sum(
                (variant_metrics[metric] - baseline_metrics[metric]) * weight
                for metric, weight in weights.items()
            )
            return reward

        elif reward_type == "single_metric":
            metric = reward_config["metric"]
            return variant_metrics[metric] - baseline_metrics[metric]

        elif reward_type == "threshold":
            metric = reward_config["metric"]
            threshold = reward_config["threshold"]
            score = variant_metrics[metric]
            return 1.0 if score >= threshold else 0.0

        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
