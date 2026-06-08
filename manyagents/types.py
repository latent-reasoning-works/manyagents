"""
Type definitions and validation for ManyAgents orchestration.

This module defines flexible, validated interfaces inspired by manylatents'
EmbeddingOutputs pattern. We use runtime validation instead of rigid dataclasses
to support algorithm diversity without schema explosion.

See docs/design_decisions.md Decision 003 for the rationale behind this approach.
"""

import logging
from typing import Any, Optional, TypedDict
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# TASK CONFIGURATION TYPES
# ============================================================================

TaskConfig = dict[str, Any]
"""
Universal task configuration for agent adapters.

This flexible dict allows any agent to define its own parameters without
requiring per-algorithm dataclasses. Adapters validate what they need.

Examples:
    # ManyLatents with experiment reference
    {"experiment": "hgdp_pca", "seed": 42}

    # ManyLatents with direct config
    {
        "data": "swissroll",
        "algorithms": {"latent": {"_target_": "...PCAModule", "n_components": 2}},
        "seed": 42
    }

    # BioDiscovery agent (hypothetical)
    {
        "task": "literature_review",
        "topic": "population genetics",
        "geometric_features": [...numpy arrays...]
    }
"""


# ============================================================================
# ADAPTER RESULT TYPES
# ============================================================================

class AdapterResult(TypedDict, total=False):
    """
    Standard result format from agent adapters.

    Uses TypedDict for IDE autocomplete while allowing flexible keys.
    The 'total=False' makes all fields optional, enabling extensibility.

    Required fields (validated at runtime):
        success: Whether execution succeeded
        summary: Human-readable description

    Standard optional fields:
        embeddings: Geometric outputs (EmbeddingOutputs format from manylatents)
        output_files: Dict mapping output types to file paths or data
        metadata: Execution metadata (config, timing, etc.)

    Custom fields: Agents can add domain-specific outputs freely.
    """
    # Required fields
    success: bool
    summary: str

    # Standard optional fields
    embeddings: Optional[dict[str, Any]]  # EmbeddingOutputs from manylatents
    output_files: Optional[dict[str, Any]]  # File paths or in-memory data
    metadata: Optional[dict[str, Any]]  # Execution metadata


# ============================================================================
# EMBEDDING OUTPUTS (mirrors manylatents pattern)
# ============================================================================

EmbeddingOutputs = dict[str, Any]
"""
Geometric data interchange format (mirrors manylatents.callbacks.embedding.base).

Required:
    embeddings (np.ndarray): Primary reduced dimensionality output

Standard optional keys:
    label: Labels/targets for data samples
    metadata: Algorithm parameters and info
    scores: Evaluation metrics (including injected geometric properties)

Custom keys are encouraged for algorithm-specific outputs.

The 'scores' field is particularly important for multi-agent workflows:
    - Contains evaluation metrics (trustworthiness, continuity, etc.)
    - Can include injected geometric properties for downstream agents
    - Enables RL reward computation from geometric quality
"""


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_task_config(config: TaskConfig, adapter_name: str) -> TaskConfig:
    """
    Minimal validation for task configuration.

    Args:
        config: The task configuration dict
        adapter_name: Name of the adapter (for logging)

    Returns:
        The validated config

    Raises:
        ValueError: If config is not a dict
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"{adapter_name}: TaskConfig must be a dictionary, got {type(config)}"
        )

    logger.debug(f"{adapter_name}: Task config validated with {len(config)} keys")
    return config


def validate_adapter_result(result: dict[str, Any], adapter_name: str) -> AdapterResult:
    """
    Validates adapter result has required fields and expected structure.

    Args:
        result: Result dictionary from adapter
        adapter_name: Name of the adapter (for logging)

    Returns:
        The validated result (as AdapterResult type)

    Raises:
        ValueError: If required fields are missing or invalid
    """
    if not isinstance(result, dict):
        raise ValueError(
            f"{adapter_name}: Result must be a dictionary, got {type(result)}"
        )

    # Check required fields
    if "success" not in result:
        raise ValueError(f"{adapter_name}: Result missing required 'success' field")

    if "summary" not in result:
        raise ValueError(f"{adapter_name}: Result missing required 'summary' field")

    if not isinstance(result["success"], bool):
        raise ValueError(
            f"{adapter_name}: 'success' must be bool, got {type(result['success'])}"
        )

    if not isinstance(result["summary"], str):
        raise ValueError(
            f"{adapter_name}: 'summary' must be str, got {type(result['summary'])}"
        )

    # Log optional fields if present
    optional_fields = [k for k in result.keys() if k not in ["success", "summary"]]
    if optional_fields:
        logger.debug(
            f"{adapter_name}: Result includes optional fields: {optional_fields}"
        )

    return result  # type: ignore - we've validated it matches AdapterResult


def validate_embedding_outputs(
    outputs: EmbeddingOutputs,
    source: str = "unknown"
) -> EmbeddingOutputs:
    """
    Validates EmbeddingOutputs structure (mirrors manylatents validation).

    Args:
        outputs: Dictionary containing embedding outputs
        source: Source identifier for logging

    Returns:
        The validated outputs dictionary

    Raises:
        ValueError: If not a dictionary or missing 'embeddings' key
    """
    if not isinstance(outputs, dict):
        raise ValueError(
            f"{source}: EmbeddingOutputs must be a dictionary, got {type(outputs)}"
        )

    if "embeddings" not in outputs:
        raise ValueError(
            f"{source}: EmbeddingOutputs must contain an 'embeddings' key"
        )

    # Validate embeddings is array-like
    embeddings = outputs["embeddings"]
    if not isinstance(embeddings, (np.ndarray, list)):
        if not hasattr(embeddings, "shape"):  # Could be torch.Tensor
            raise ValueError(
                f"{source}: 'embeddings' must be array-like, got {type(embeddings)}"
            )

    # Log standard optional fields if present
    standard_fields = {"label", "metadata", "scores"}
    present_fields = standard_fields.intersection(outputs.keys())
    if present_fields:
        logger.debug(f"{source}: EmbeddingOutputs includes: {present_fields}")

    # Log custom fields
    custom_fields = set(outputs.keys()) - standard_fields - {"embeddings"}
    if custom_fields:
        logger.info(
            f"{source}: EmbeddingOutputs includes custom fields: {custom_fields}"
        )

    return outputs


# ============================================================================
# PASSTHROUGH MODE CONFIGURATION
# ============================================================================

class PassthroughMode:
    """
    Controls how data is passed between workflow steps.

    Some agents are locked into file-based I/O (CLI tools), while others
    can handle in-memory data passing (Python APIs). This class defines
    the modes and provides validation.
    """

    MEMORY = "memory"  # Keep data in-memory (numpy arrays, dicts)
    FILE = "file"      # Save to files (CSV, pickle, etc.)
    AUTO = "auto"      # Adapter decides based on data size/downstream agent

    VALID_MODES = {MEMORY, FILE, AUTO}

    @classmethod
    def validate(cls, mode: str) -> str:
        """Validate passthrough mode string."""
        if mode not in cls.VALID_MODES:
            raise ValueError(
                f"Invalid passthrough mode '{mode}'. "
                f"Valid modes: {cls.VALID_MODES}"
            )
        return mode


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def merge_embedding_outputs(
    base: EmbeddingOutputs,
    additional: dict[str, Any],
    source: str = "merge"
) -> EmbeddingOutputs:
    """
    Merge additional data into EmbeddingOutputs.

    Useful for injecting geometric properties from one step into another.

    Args:
        base: Base EmbeddingOutputs dict
        additional: Additional data to merge (won't overwrite 'embeddings')
        source: Source identifier for logging

    Returns:
        Merged EmbeddingOutputs

    Example:
        # Inject PCA scores into UMAP output for downstream agents
        pca_result = {"embeddings": pca_emb, "scores": {"trustworthiness": 0.95}}
        umap_result = {"embeddings": umap_emb}
        merged = merge_embedding_outputs(umap_result, pca_result["scores"])
    """
    base = validate_embedding_outputs(base, source)

    merged = base.copy()
    for key, value in additional.items():
        if key == "embeddings":
            logger.warning(
                f"{source}: Skipping 'embeddings' key from additional data "
                "to prevent overwriting base embeddings"
            )
            continue

        if key in merged:
            logger.info(
                f"{source}: Overwriting existing key '{key}' in EmbeddingOutputs"
            )

        merged[key] = value

    return merged


def extract_geometric_features(
    embedding_outputs: EmbeddingOutputs,
    include_scores: bool = True,
    include_metadata: bool = False,
) -> dict[str, Any]:
    """
    Extract geometric features for downstream agent consumption.

    This is particularly useful for multi-agent workflows where geometric
    properties (embeddings, metrics) need to be concatenated with other
    features for downstream agents.

    Args:
        embedding_outputs: The EmbeddingOutputs to extract from
        include_scores: Include evaluation metrics (trustworthiness, etc.)
        include_metadata: Include algorithm metadata

    Returns:
        Dictionary of geometric features ready for concatenation with agent inputs

    Example:
        # Extract features for BioDiscoveryAgent
        pca_output = run_manylatents(...)
        features = extract_geometric_features(pca_output, include_scores=True)
        # features = {
        #   "embeddings": array(...),
        #   "geometric_metrics": {"trustworthiness": 0.95, ...}
        # }
        biodiscovery_result = run_biodiscovery(
            base_data=genomic_data,
            geometric_features=features
        )
    """
    embedding_outputs = validate_embedding_outputs(embedding_outputs, "extract")

    features = {
        "embeddings": embedding_outputs["embeddings"],
    }

    if include_scores and "scores" in embedding_outputs:
        features["geometric_metrics"] = embedding_outputs["scores"]

    if include_metadata and "metadata" in embedding_outputs:
        features["geometric_metadata"] = embedding_outputs["metadata"]

    # Include any custom fields
    custom_keys = set(embedding_outputs.keys()) - {"embeddings", "label", "metadata", "scores"}
    for key in custom_keys:
        features[key] = embedding_outputs[key]

    return features
