"""Adapter for ManyLatents agent using direct API calls."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from .base import AgentAdapter

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
            adapter = ManyLatentsAdapter()
            result = await adapter.run(
                {"algorithm": "pca", "data": "swissroll", "n_components": 2},
                {}
            )
            embeddings = result['output_files']['embeddings']
        """
        try:
            # Import manylatents API
            from manylatents.api import run
            from hydra.core.global_hydra import GlobalHydra

            log.info(f"ManyLatents API executing with config: {task_config}")

            # Clear Hydra's global state before calling manylatents API
            # This is necessary because manyAgents already initialized Hydra
            # and manylatents.api.run() also initializes it
            if GlobalHydra.instance().is_initialized():
                log.debug("Clearing GlobalHydra instance before calling manylatents API")
                GlobalHydra.instance().clear()

            # Build configuration
            overrides = {
                'debug': True,  # Disable wandb for agent workflows
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

            log.info(f"Calling manylatents.api.run() with overrides: {overrides}")

            # Run in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: run(input_data=input_data, **overrides)
            )

            # Extract results
            embeddings = result.get('embeddings')
            scores = result.get('scores', {})
            metadata = result.get('metadata', {})

            # Build summary
            summary_parts = [
                f"ManyLatents successfully executed",
                f"Output shape: {embeddings.shape if embeddings is not None else 'N/A'}",
            ]

            if scores:
                metric_summary = ', '.join(f'{k}={v:.3f}' for k, v in list(scores.items())[:3])
                summary_parts.append(f"Metrics: {metric_summary}")

            return {
                'summary': ' | '.join(summary_parts),
                'output_files': {
                    'embeddings': embeddings,  # numpy array
                    'scores': scores,
                    'metadata': metadata
                },
                'success': True,
                'metadata': metadata
            }

        except Exception as e:
            log.error(f"ManyLatents API call failed: {e}", exc_info=True)
            return {
                'summary': f"ManyLatents failed: {str(e)}",
                'output_files': {},
                'success': False,
                'metadata': {
                    'error': str(e),
                    'task_config': task_config
                }
            }
