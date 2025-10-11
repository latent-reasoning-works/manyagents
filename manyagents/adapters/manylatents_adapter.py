"""Adapter for ManyLatents agent using direct API calls."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from .base import AgentAdapter
from manyagents.types import (
    validate_task_config,
    validate_adapter_result,
    validate_embedding_outputs,
)
from manyagents.config_utils import (
    load_manylatents_experiment,
    validate_manylatents_config,
)

log = logging.getLogger(__name__)


class ManyLatentsAdapter(AgentAdapter):
    """
    Adapter for ManyLatents using direct Python API (no subprocess).

    This new implementation calls manylatents.api.run() directly, enabling:
    - In-memory data passing between workflow steps
    - No subprocess overhead
    - Direct access to embeddings and metrics
    """

    def __init__(self):
        super().__init__("manylatents")

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
        input_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute ManyLatents using direct API call.

        Args:
            task_config: Dictionary with ManyLatents configuration:
                Option 1 - Experiment reference:
                    - experiment: Experiment name (e.g., "hgdp_pca")
                    - Any additional overrides
                Option 2 - Direct config:
                    - algorithm: Algorithm type (e.g., "pca", "phate", "umap")
                    - data: Dataset name (e.g., "swissroll")
                    - pipeline: Optional list of pipeline steps
                    - n_components: Number of components (default: 2)
                    - Other Hydra config overrides
            input_files: Input data files (reserved for future use)
            input_data: Optional numpy array from previous step (for chaining)

        Returns:
            Standardized result dictionary with embeddings and metrics

        Example:
            # Using experiment reference
            result = await adapter.run(
                {"experiment": "hgdp_pca", "algorithms": {"latent": {"n_components": 10}}},
                {}
            )

            # Using direct config
            result = await adapter.run(
                {"algorithm": "pca", "data": "swissroll", "n_components": 2},
                {}
            )
        """
        try:
            # Validate task config
            task_config = validate_task_config(task_config, self.name)

            # Import manylatents API
            from manylatents.api import run
            from hydra.core.global_hydra import GlobalHydra

            log.info(f"ManyLatents adapter executing with task config: {task_config}")

            # Clear Hydra's global state before calling manylatents API
            # This is necessary because manyAgents already initialized Hydra
            # and manylatents.api.run() also initializes it
            if GlobalHydra.instance().is_initialized():
                log.debug("Clearing GlobalHydra instance before calling manylatents API")
                GlobalHydra.instance().clear()

            # Build configuration based on whether experiment name is provided
            if 'experiment' in task_config:
                # Load experiment and merge overrides
                experiment_name = task_config.pop('experiment')
                log.info(f"Loading manylatents experiment: {experiment_name}")

                overrides = load_manylatents_experiment(
                    experiment_name,
                    overrides=task_config  # Remaining keys are overrides
                )
            else:
                # Build config from scratch
                overrides = {
                    'project': 'manyagents_workflow',
                }

                # Handle dataset
                if 'data' in task_config:
                    overrides['data'] = task_config['data']

                # Handle pipeline vs single algorithm
                if 'pipeline' in task_config:
                    # Pipeline mode
                    overrides['pipeline'] = task_config['pipeline']
                elif 'algorithm' in task_config:
                    # Single algorithm mode
                    algo_name = task_config['algorithm'].lower()
                    n_components = task_config.get('n_components', 2)

                    overrides['algorithms'] = {
                        'latent': {
                            '_target_': f'manylatents.algorithms.latent.{algo_name}.{algo_name.upper()}Module',
                            'n_components': n_components
                        }
                    }

                # Add any additional overrides from task_config
                for key, value in task_config.items():
                    if key not in ['algorithm', 'data', 'pipeline', 'n_components']:
                        overrides[key] = value

            # Validate manylatents config structure
            validate_manylatents_config(overrides)

            log.info(f"Calling manylatents.api.run() with validated config")

            # Run in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            manylatents_result = await loop.run_in_executor(
                None,
                lambda: run(input_data=input_data, **overrides)
            )

            # Validate that result is EmbeddingOutputs format
            embedding_outputs = validate_embedding_outputs(
                manylatents_result,
                source=f"{self.name}_result"
            )

            # Extract key components
            embeddings = embedding_outputs['embeddings']
            scores = embedding_outputs.get('scores', {})
            metadata = embedding_outputs.get('metadata', {})

            # Build summary
            summary_parts = [
                f"ManyLatents successfully executed",
                f"Output shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}",
            ]

            if scores:
                metric_summary = ', '.join(f'{k}={v:.3f}' for k, v in list(scores.items())[:3])
                summary_parts.append(f"Metrics: {metric_summary}")

            # Build standardized adapter result
            adapter_result = {
                'summary': ' | '.join(summary_parts),
                'success': True,
                'embeddings': embedding_outputs,  # Full EmbeddingOutputs dict
                'output_files': {
                    'embeddings': embeddings,  # numpy array for convenience
                    'scores': scores,
                    'metadata': metadata
                },
                'metadata': metadata
            }

            # Validate result structure
            return validate_adapter_result(adapter_result, self.name)

        except Exception as e:
            log.error(f"ManyLatents API call failed: {e}", exc_info=True)
            error_result = {
                'summary': f"ManyLatents failed: {str(e)}",
                'success': False,
                'output_files': {},
                'metadata': {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'task_config': task_config
                }
            }
            return validate_adapter_result(error_result, self.name)
