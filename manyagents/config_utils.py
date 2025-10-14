"""
Configuration utilities for ManyAgents adapters.

This module provides helpers for loading, merging, and building configurations
when adapters need to translate workflow task configs into agent-specific formats.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ============================================================================
# EXPERIMENT LOADING (for manylatents)
# ============================================================================

def load_manylatents_experiment(
    experiment_name: str,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load a manylatents experiment config by name and apply overrides.

    This uses Hydra's compose API to load experiment configs from manylatents,
    then merges any additional overrides from the workflow task config.

    Args:
        experiment_name: Name of experiment (e.g., "hgdp_pca")
        overrides: Additional config overrides to merge

    Returns:
        Merged configuration dict ready for manylatents.api.run()

    Example:
        # Workflow task config:
        task_config = {
            "experiment": "hgdp_pca",
            "algorithms": {"latent": {"n_components": 10}}  # Override one param
        }

        # Load and merge:
        config = load_manylatents_experiment(
            "hgdp_pca",
            overrides={"algorithms": {"latent": {"n_components": 10}}}
        )
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Get manylatents config directory
    try:
        import manylatents
        config_dir = Path(manylatents.__file__).parent / "configs"
        config_dir = str(config_dir.resolve())
    except ImportError:
        raise ImportError(
            "manylatents not found. Install with: uv add manylatents"
        )

    # Clear Hydra state if needed
    if GlobalHydra.instance().is_initialized():
        logger.debug("Clearing GlobalHydra before loading manylatents experiment")
        GlobalHydra.instance().clear()

    # Load experiment config
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"experiment={experiment_name}"]
        )

    # Convert to plain dict (resolve=True to handle interpolations)
    # throw_on_missing=False allows missing interpolations
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

    # Ensure it's a plain dict, not a DictConfig
    if not isinstance(config_dict, dict):
        config_dict = dict(config_dict)

    # Merge additional overrides if provided
    if overrides:
        config_dict = deep_merge(config_dict, overrides)

    logger.info(f"Loaded manylatents experiment '{experiment_name}' with overrides")
    return config_dict


# ============================================================================
# CONFIG MERGING UTILITIES
# ============================================================================

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Unlike {**base, **override}, this recursively merges nested dicts
    instead of replacing them entirely.

    Handles both plain dicts and OmegaConf DictConfigs, converting
    all values to plain dicts for consistency.

    Args:
        base: Base configuration dictionary
        override: Override configuration (takes precedence)

    Returns:
        Merged dictionary (plain dict, no DictConfigs)

    Example:
        base = {"algorithms": {"latent": {"n_components": 2, "seed": 42}}}
        override = {"algorithms": {"latent": {"n_components": 10}}}
        merged = deep_merge(base, override)
        # Result: {"algorithms": {"latent": {"n_components": 10, "seed": 42}}}
        # Note: seed=42 is preserved!
    """
    result = base.copy()

    for key, value in override.items():
        # Convert DictConfig to plain dict
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True, throw_on_missing=False)

        # Check if we need to recursively merge
        is_result_dict = isinstance(result.get(key), dict)
        is_value_dict = isinstance(value, dict)

        if key in result and is_result_dict and is_value_dict:
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override takes precedence
            result[key] = value

    return result


# ============================================================================
# CONFIG BUILDING UTILITIES
# ============================================================================

def build_hydra_overrides(config_dict: Dict[str, Any], prefix: str = "") -> list[str]:
    """
    Convert a nested dict into Hydra override strings.

    Useful when you need to call Hydra's compose() with overrides
    derived from a Python dict.

    Args:
        config_dict: Nested configuration dictionary
        prefix: Key prefix for recursive calls (leave empty for top-level)

    Returns:
        List of Hydra override strings

    Example:
        config = {"algorithms": {"latent": {"n_components": 10}}, "seed": 42}
        overrides = build_hydra_overrides(config)
        # Returns: ["algorithms.latent.n_components=10", "seed=42"]

    This helps avoid string mismatch errors by programmatically generating
    override strings from the dict structure.
    """
    overrides = []

    for key, value in config_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively handle nested dicts
            overrides.extend(build_hydra_overrides(value, prefix=full_key))
        elif isinstance(value, list):
            # Lists need special handling - convert to string representation
            overrides.append(f"{full_key}={value}")
        elif isinstance(value, str):
            # Strings need quoting if they contain spaces
            if " " in value:
                overrides.append(f'{full_key}="{value}"')
            else:
                overrides.append(f"{full_key}={value}")
        else:
            # Numbers, bools, etc.
            overrides.append(f"{full_key}={value}")

    return overrides


# ============================================================================
# CONFIG VALIDATION
# ============================================================================

def validate_manylatents_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that a config dict has the minimum structure for manylatents.

    This catches common mistakes before passing to manylatents.api.run().

    Args:
        config: Configuration dictionary

    Returns:
        The validated config

    Raises:
        ValueError: If required fields are missing or malformed

    Example:
        config = {"data": "swissroll"}  # Missing algorithms!
        validate_manylatents_config(config)  # Raises ValueError
    """
    # Check for data OR pipeline
    has_data = "data" in config
    has_pipeline = "pipeline" in config and config["pipeline"]

    if not (has_data or has_pipeline):
        raise ValueError(
            "manylatents config must have either 'data' or 'pipeline' field"
        )

    # Check for algorithms OR pipeline
    has_algorithms = "algorithms" in config
    if not (has_algorithms or has_pipeline):
        raise ValueError(
            "manylatents config must have either 'algorithms' or 'pipeline' field"
        )

    # Validate algorithms structure if present
    if has_algorithms:
        from omegaconf import DictConfig
        if not isinstance(config["algorithms"], (dict, DictConfig)):
            raise ValueError(
                f"'algorithms' must be a dict or DictConfig, got {type(config['algorithms'])}"
            )

    logger.debug("manylatents config validation passed")
    return config


# ============================================================================
# HYDRA-ZEN PROTOTYPE (for experimentation)
# ============================================================================

def build_manylatents_config_with_hydra_zen(
    algorithm: str,
    data: str,
    **kwargs
) -> Dict[str, Any]:
    """
    EXPERIMENTAL: Build manylatents config using hydra-zen.

    This shows what hydra-zen would look like for config generation.
    Currently not used - kept as reference for potential future use.

    Args:
        algorithm: Algorithm name (e.g., "pca")
        data: Dataset name (e.g., "swissroll")
        **kwargs: Additional parameters

    Returns:
        Configuration dict

    Example:
        config = build_manylatents_config_with_hydra_zen(
            algorithm="pca",
            data="swissroll",
            n_components=10,
            seed=42
        )

    Why not use this?
        - Requires hydra-zen dependency
        - More complex than plain dicts
        - Doesn't add much value for our flexible schema approach
        - Keep this as reference if we need it later
    """
    try:
        from hydra_zen import builds, make_config
    except ImportError:
        raise ImportError(
            "hydra-zen not installed. This is an experimental feature. "
            "Install with: uv add hydra-zen"
        )

    # Build algorithm config
    algo_config = builds(
        dict,  # Placeholder - would be actual algorithm class
        _target_=f"manylatents.algorithms.latent.{algorithm}.{algorithm.upper()}Module",
        populate_full_signature=False,
        **{k: v for k, v in kwargs.items() if k not in ["data", "seed"]}
    )

    # Build full config
    Config = make_config(
        data=data,
        algorithms={"latent": algo_config},
        seed=kwargs.get("seed", 42),
        project="manyagents_workflow"
    )

    # Instantiate and convert to dict
    cfg = Config()
    return OmegaConf.to_container(cfg, resolve=True)
