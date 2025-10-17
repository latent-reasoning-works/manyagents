"""Pytest fixtures and configuration for manyagents golden tests."""

import pytest
from pathlib import Path
import numpy as np
from typing import Dict, Any
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

# Test directories
TEST_DIR = Path(__file__).parent
CONFIGS_DIR = TEST_DIR / "configs"
REFERENCE_DIR = TEST_DIR / "reference_vectors"


def discover_test_configs():
    """Auto-discover test configs in tests/configs/.

    Returns:
        List[str]: Test names extracted from test_*.yaml files
    """
    if not CONFIGS_DIR.exists():
        return []

    configs = []
    for config_file in sorted(CONFIGS_DIR.glob("test_*.yaml")):
        # Extract test name (e.g., test_pca_50d.yaml -> pca_50d)
        test_name = config_file.stem.replace("test_", "")
        configs.append(test_name)

    return configs


def pytest_generate_tests(metafunc):
    """Auto-parametrize tests based on discovered configs.

    This hook is called during test collection. Any test function
    with 'test_config_name' in its parameters will be automatically
    parametrized with all discovered test configs.
    """
    if "test_config_name" in metafunc.fixturenames:
        configs = discover_test_configs()
        if not configs:
            pytest.skip("No test configs found in tests/configs/")
        metafunc.parametrize("test_config_name", configs)


@pytest.fixture
def test_config(test_config_name: str) -> Dict[str, Any]:
    """Load test config by name.

    Args:
        test_config_name: Name of the test (without 'test_' prefix)

    Returns:
        Dict containing the full test configuration
    """
    config_path = CONFIGS_DIR / f"test_{test_config_name}.yaml"

    if not config_path.exists():
        pytest.fail(f"Test config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


@pytest.fixture
def reference_vector(test_config_name: str) -> np.ndarray:
    """Load reference vector for given test config.

    Args:
        test_config_name: Name of the test (without 'test_' prefix)

    Returns:
        numpy array containing the golden reference output
    """
    ref_path = REFERENCE_DIR / f"{test_config_name}.npy"

    if not ref_path.exists():
        pytest.skip(f"No reference vector found: {ref_path}")

    return np.load(ref_path)


@pytest.fixture
def tolerance_config(test_config: Dict[str, Any]) -> Dict[str, float]:
    """Extract tolerance settings from test config.

    Args:
        test_config: Full test configuration

    Returns:
        Dict with 'rtol' and 'atol' tolerance values
    """
    test_tolerance = test_config.get('test_tolerance', {})
    return {
        'rtol': test_tolerance.get('rtol', 1e-5),
        'atol': test_tolerance.get('atol', 1e-6),
    }


@pytest.fixture(autouse=True)
def cleanup_hydra():
    """Ensure Hydra is clean between tests.

    This fixture runs automatically for every test to prevent
    Hydra GlobalHydra conflicts when running multiple tests.
    """
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary output directory for test runs.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Path to a temporary outputs directory
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
