"""Test script for metrics API and MetricComputer integration."""

import numpy as np
from manylatents.metrics.api import list_available_metrics, compute_metrics
from manyagents.metrics import MetricComputer


def test_metrics_api():
    """Test the manylatents.metrics.api module."""
    print("=== Testing manylatents.metrics.api ===\n")

    # List available metrics
    metrics = list_available_metrics()
    print(f"Found {len(metrics)} available metrics:")
    for metric in metrics[:10]:  # Show first 10
        print(f"  - {metric}")
    if len(metrics) > 10:
        print(f"  ... and {len(metrics) - 10} more")
    print()

    # Create test data
    np.random.seed(42)
    x = np.random.randn(100, 50)  # 100 samples, 50 dimensions
    z = np.random.randn(100, 2)   # 100 samples, 2 dimensions

    # Compute some basic metrics
    print("Computing metrics on test data (100 samples, 50->2 dims)...")
    scores = compute_metrics(
        x=x,
        z=z,
        metric_names=['Trustworthiness', 'Continuity'],
        metric_params={'n_neighbors': 10}
    )

    print("Results:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")
    print()


def test_metric_computer():
    """Test the manyagents.metrics.MetricComputer."""
    print("=== Testing manyagents.metrics.MetricComputer ===\n")

    computer = MetricComputer()

    # Create test data with two different embeddings
    # Simulating embeddings from non-manylatents tools
    np.random.seed(42)
    x = np.random.randn(100, 50)
    z_baseline = np.random.randn(100, 2)
    z_variant = np.random.randn(100, 2) * 0.8  # Different embedding

    embeddings = {
        'baseline': z_baseline,
        'variant': z_variant
    }

    print("Computing metrics for multiple embeddings (from non-manylatents tools)...")
    scores = computer.compute(
        x=x,
        embeddings=embeddings,
        metric_names=['Trustworthiness', 'Continuity'],
        metric_params={'n_neighbors': 10}
    )

    print("Results:")
    for run_id, run_scores in scores.items():
        print(f"  {run_id}:")
        for metric, score in run_scores.items():
            print(f"    {metric}: {score:.4f}")
    print()

    # Test reward computation
    print("Computing reward (trustworthiness_diff)...")
    reward = computer.compute_reward(
        scores=scores,
        reward_config={
            'type': 'trustworthiness_diff',
            'baseline_id': 'baseline',
            'variant_id': 'variant'
        }
    )
    print(f"  Reward: {reward:.4f}")
    print()


if __name__ == "__main__":
    test_metrics_api()
    test_metric_computer()
    print("✓ All tests passed!")
