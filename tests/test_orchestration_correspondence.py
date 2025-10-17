"""Test correspondence between manylatents direct API and manyagents orchestration.

This test suite implements Test 1 of the 3-way golden test architecture:
validating that manyagents acts as a transparent orchestrator without
corrupting or modifying manylatents outputs.

Test strategy:
1. Call manylatents.api.run() directly (ground truth)
2. Call same workflow through manyagents orchestration
3. Verify outputs are identical (within numerical precision)
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from omegaconf import OmegaConf

from manyagents.main import execute_workflow_chain
import manylatents.api


def extract_metrics_from_manylatents_result(result: Dict[str, Any]) -> Tuple[np.ndarray, list]:
    """Extract geometric metrics vector from manylatents API result.

    Args:
        result: Result dictionary from manylatents.api.run()

    Returns:
        Tuple of (metrics_vector, metric_names) where metrics_vector is a
        numpy array and metric_names is the sorted list of metric names
    """
    # manylatents.api.run() returns metrics in result["metrics"]
    metrics_dict = result.get("metrics", {})
    assert metrics_dict, "No metrics found in manylatents result"

    # Sort keys for consistent ordering
    metric_names = sorted(metrics_dict.keys())
    metrics_vector = np.array([metrics_dict[name] for name in metric_names])

    return metrics_vector, metric_names


def extract_metrics_from_manyagents_result(workflow_result: Dict[str, Any]) -> Tuple[np.ndarray, list]:
    """Extract geometric metrics vector from manyagents workflow result.

    Args:
        workflow_result: Result dictionary from execute_workflow_chain()

    Returns:
        Tuple of (metrics_vector, metric_names)
    """
    assert workflow_result["success"], "Workflow execution failed"
    assert len(workflow_result["steps"]) > 0, "No steps executed"

    # Get the first step result (typically the only manylatents step)
    step_result = workflow_result["steps"][0]["result"]
    metadata = step_result.get("metadata", {})
    metrics_dict = metadata.get("metrics", {})

    assert metrics_dict, "No metrics found in manyagents result"

    # Sort keys for consistent ordering
    metric_names = sorted(metrics_dict.keys())
    metrics_vector = np.array([metrics_dict[name] for name in metric_names])

    return metrics_vector, metric_names


def test_manylatents_manyagents_correspondence(
    test_config_name: str,
    test_config: Dict[str, Any],
    tolerance_config: Dict[str, float],
    temp_output_dir: Path
):
    """Test correspondence between manylatents API and manyagents orchestration.

    This is the PRIMARY test validating that manyagents doesn't corrupt outputs.

    Workflow:
    1. Extract manylatents config from test config
    2. Call manylatents.api.run() directly (reference/ground truth)
    3. Call through manyagents orchestration
    4. Verify outputs are identical

    Args:
        test_config_name: Name of the test (auto-injected by pytest)
        test_config: Loaded test configuration (auto-injected)
        tolerance_config: Test tolerance settings (auto-injected)
        temp_output_dir: Temporary output directory (auto-injected)
    """
    # Extract manylatents config from workflow definition
    workflow_step = test_config["workflow"]["steps"][0]
    assert workflow_step["agent"] == "manylatents", "Test must be for manylatents agent"

    manylatents_config = workflow_step["config"]

    print(f"\n{'='*70}")
    print(f"Testing orchestration correspondence: {test_config_name}")
    print(f"{'='*70}")

    # ========================================================================
    # Level 1: Direct manylatents API call (ground truth)
    # ========================================================================
    print(f"\n[Level 1] manylatents.api.run() - Direct API call")
    print(f"  Config: {manylatents_config}")

    result_direct = manylatents.api.run(**manylatents_config)
    metrics_direct, names_direct = extract_metrics_from_manylatents_result(result_direct)

    print(f"  ✓ Execution complete")
    print(f"    Metrics: {names_direct}")
    print(f"    G_vector: {metrics_direct}")

    # ========================================================================
    # Level 2: manyagents orchestration
    # ========================================================================
    print(f"\n[Level 2] manyagents.orchestration() - Through adapter layer")

    config_dict = test_config.copy()
    config_dict["output_dir"] = str(temp_output_dir)
    cfg = OmegaConf.create(config_dict)

    result_orchestrated = asyncio.run(execute_workflow_chain(cfg, temp_output_dir))
    metrics_orchestrated, names_orchestrated = extract_metrics_from_manyagents_result(result_orchestrated)

    print(f"  ✓ Execution complete")
    print(f"    Metrics: {names_orchestrated}")
    print(f"    G_vector: {metrics_orchestrated}")

    # ========================================================================
    # Verify correspondence
    # ========================================================================
    print(f"\n[Correspondence Check]")

    # Check metric names match
    assert names_direct == names_orchestrated, (
        f"Metric names differ!\n"
        f"  Direct:       {names_direct}\n"
        f"  Orchestrated: {names_orchestrated}"
    )

    # Check values match within tolerance
    max_diff = np.max(np.abs(metrics_direct - metrics_orchestrated))
    print(f"  Max absolute difference: {max_diff}")
    print(f"  Tolerance: rtol={tolerance_config['rtol']}, atol={tolerance_config['atol']}")

    np.testing.assert_allclose(
        metrics_direct,
        metrics_orchestrated,
        rtol=tolerance_config['rtol'],
        atol=tolerance_config['atol'],
        err_msg=(
            f"Orchestration correspondence test '{test_config_name}' FAILED\n"
            f"manyagents output differs from manylatents ground truth\n"
            f"Max difference: {max_diff}"
        )
    )

    print(f"\n{'='*70}")
    print(f"✅ PASSED: {test_config_name}")
    print(f"   manylatents.api.run() ≡ manyagents.orchestration()")
    print(f"   Correspondence verified within tolerance")
    print(f"{'='*70}")
