"""Adapter for ManyLatents agent using direct API calls."""

import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

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
    - Cached metric execution for fast RL loops
    """

    def __init__(self):
        super().__init__("manylatents")

        # Lazy-load metric registry (only when needed for cached mode)
        self._metric_registry = None
        self._metric_cache = {}  # {name: {'object': metric_obj, 'group': str}}
        self._cached_mode = False

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
            log.info(f"🔍 Task config has 'metrics' key: {'metrics' in task_config}")

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
                    if key not in ['algorithm', 'data', 'pipeline', 'n_components', 'metrics']:
                        overrides[key] = value

            # Handle metrics specifically - load from manyLatents default configs
            # This runs for BOTH experiment and direct config paths
            if 'metrics' in task_config:
                print("DEBUG: Metrics transformation triggered!")  # DEBUG
                log.info("🔄 Starting metric transformation")
                from omegaconf import OmegaConf
                from pathlib import Path
                import os

                # Find manyLatents config directory
                manylatents_config_dir = Path(os.path.dirname(__import__('manylatents').__file__)) / 'configs'

                log.info(f"Loading metric configs from manyLatents: {manylatents_config_dir}")

                transformed_metrics = {}

                for group, group_metrics in task_config['metrics'].items():
                    transformed_metrics[group] = {}

                    for metric_name, metric_params in group_metrics.items():
                        try:
                            # Load the default config from manyLatents using OmegaConf
                            config_path = manylatents_config_dir / 'metrics' / group / f'{metric_name}.yaml'

                            if config_path.exists():
                                # Load YAML directly with OmegaConf
                                metric_cfg = OmegaConf.load(config_path)

                                # Extract the metric config (structure: {metric_name: {_target_: ...}})
                                if metric_name in metric_cfg:
                                    metric_config = OmegaConf.to_container(metric_cfg[metric_name], resolve=True)

                                    # Apply user overrides
                                    if metric_params:
                                        metric_config.update(metric_params)

                                    transformed_metrics[group][metric_name] = metric_config
                                    log.info(f"  ✓ Loaded '{metric_name}' from {config_path.name}")
                                else:
                                    log.warning(
                                        f"Metric '{metric_name}' not found in {config_path}, "
                                        f"passing through as-is"
                                    )
                                    transformed_metrics[group][metric_name] = metric_params
                            else:
                                log.warning(
                                    f"Metric config not found: {config_path}, "
                                    f"passing through as-is"
                                )
                                transformed_metrics[group][metric_name] = metric_params

                        except Exception as e:
                            log.warning(
                                f"Failed to load metric '{metric_name}' from manyLatents: {e}, "
                                f"passing through as-is"
                            )
                            transformed_metrics[group][metric_name] = metric_params

                overrides['metrics'] = transformed_metrics
                log.info(f"✅ Composed {len(transformed_metrics)} metric groups from manyLatents configs")

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
                # Handle metric values: float, tuple[float, array], or dict
                def extract_scalar(v):
                    """Extract displayable scalar from metric value."""
                    if isinstance(v, tuple) and len(v) == 2:
                        return v[0]  # (scalar, per_sample) -> scalar
                    elif isinstance(v, dict):
                        # For structured metrics, try to find a scalar or use length
                        return v.get('scalar', len(v))
                    else:
                        return float(v)  # Already a scalar

                metric_summary = ', '.join(
                    f'{k}={extract_scalar(v):.3f}'
                    for k, v in list(scores.items())[:3]
                )
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

    # ========== Cached Metrics Mode (for RL Training) ==========

    def _parse_metric_spec(self, spec: Union[str, Dict]) -> tuple[str, Dict[str, Any]]:
        """
        Parse metric specification into name and parameter overrides.

        Args:
            spec: Either a simple name string or dict with name -> params

        Returns:
            Tuple of (metric_name, params_dict)

        Examples:
            >>> self._parse_metric_spec('participation_ratio')
            ('participation_ratio', {})

            >>> self._parse_metric_spec({'lid': {'k': 30}})
            ('lid', {'k': 30})
        """
        if isinstance(spec, str):
            return spec, {}
        elif isinstance(spec, dict):
            if len(spec) != 1:
                raise ValueError(
                    f"Metric spec dict must have exactly one key, got: {spec}"
                )
            name = list(spec.keys())[0]
            params = spec[name]
            if not isinstance(params, dict):
                raise ValueError(
                    f"Metric params must be a dict, got {type(params)} for '{name}'"
                )
            return name, params
        else:
            raise ValueError(
                f"Invalid metric spec type: {type(spec)}. "
                f"Must be str or dict, got: {spec}"
            )

    def setup_metrics(
        self,
        metric_names: List[Union[str, Dict]],
        **global_overrides
    ):
        """
        Pre-instantiate metrics for fast cached execution (RL mode).

        This method instantiates metric objects once and caches them for
        reuse across thousands of RL episodes. This avoids Hydra overhead
        on every metric computation.

        Args:
            metric_names: List of metric specifications, either:
                - Simple names: ['participation_ratio', 'lid']
                - With overrides: [{'lid': {'k': 30}}, 'participation_ratio']
            **global_overrides: Parameters applied to ALL metrics

        Examples:
            >>> adapter = ManyLatentsAdapter()
            >>> adapter.setup_metrics(['participation_ratio', 'lid'])

            >>> # With parameter overrides
            >>> adapter.setup_metrics(
            ...     [{'lid': {'k': 30}}, 'participation_ratio'],
            ...     return_per_sample=True  # Applied to all
            ... )

        Raises:
            KeyError: If a metric name is not found in the registry
            ImportError: If a metric class cannot be imported
        """
        # Lazy-load metric registry
        if self._metric_registry is None:
            from .metric_registry import MetricRegistry
            self._metric_registry = MetricRegistry()
            log.info(
                f"Loaded metric registry with {len(self._metric_registry)} metrics"
            )

        log.info(f"Setting up {len(metric_names)} metrics for cached execution")

        for spec in metric_names:
            # Parse specification
            name, spec_params = self._parse_metric_spec(spec)

            # Get metric information from registry
            metric_info = self._metric_registry.get_metric_info(name)
            metric_class = self._metric_registry.get_metric_class(name)

            # Merge parameters (defaults < global < spec-specific)
            final_params = {
                **metric_info['defaults'],
                **global_overrides,
                **spec_params
            }

            log.debug(
                f"Creating partial metric '{name}' "
                f"(class: {metric_info['class']}, group: {metric_info['group']}) "
                f"with params: {final_params}"
            )

            # Create partial function with config parameters pre-bound
            # manyLatents metrics are functions with signature:
            # metric(embeddings, dataset=None, module=None, **config_params)
            # We pre-bind the config_params to create a cached partial
            if metric_info.get('partial', True):
                metric_obj = partial(metric_class, **final_params)
            else:
                # For non-partial metrics, instantiate normally
                metric_obj = metric_class(**final_params)

            # Cache the partial metric
            self._metric_cache[name] = {
                'object': metric_obj,
                'group': metric_info['group'],
                'class': metric_info['class'],
                'params': final_params
            }

            log.debug(f"Cached partial metric '{name}' successfully")

        self._cached_mode = True
        log.info(
            f"✅ Cached metrics setup complete. "
            f"{len(self._metric_cache)} metrics ready for fast execution."
        )

    # ========== Fast Execution Mode (for RL Training) ==========

    async def execute_cached(
        self,
        algorithm: str,
        params: Dict[str, Any],
        data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fast execution using cached metrics (no Hydra overhead).

        This method directly instantiates algorithms and uses pre-cached
        metrics for extremely fast execution suitable for RL training loops.

        Args:
            algorithm: Algorithm name (e.g., 'PCA', 'UMAP')
            params: Algorithm parameters (e.g., {'n_components': 50})
            data: Input data array (n_samples, n_features)

        Returns:
            EmbeddingOutputs dict:
            {
                'embeddings': np.ndarray,
                'scores': {metric_name: score},
                'metadata': {...},
                'success': True
            }

        Raises:
            RuntimeError: If setup_metrics() wasn't called first
            KeyError: If algorithm not found in registry
            ImportError: If algorithm class cannot be imported

        Example:
            >>> adapter = ManyLatentsAdapter()
            >>> adapter.setup_metrics(['participation_ratio', 'lid'])
            >>> result = await adapter.execute_cached(
            ...     algorithm='PCA',
            ...     params={'n_components': 2},
            ...     data=np.random.randn(1000, 50)
            ... )
        """
        if not self._cached_mode:
            raise RuntimeError(
                "Must call setup_metrics() before execute_cached(). "
                "Cached mode is required for fast execution."
            )

        log.debug(
            f"Fast execution: {algorithm} with {len(self._metric_cache)} cached metrics"
        )

        # 1. Get algorithm class from registry
        algo_class = self._metric_registry.get_algorithm_class(algorithm)
        algo_defaults = self._metric_registry.get_algorithm_defaults(algorithm)

        # Merge defaults with provided params
        final_params = {**algo_defaults, **params}

        log.debug(f"Instantiating {algorithm} with params: {final_params}")

        # 2. Instantiate algorithm
        algo_instance = algo_class(**final_params)

        # 3. Create lightweight data wrapper
        from manylatents.data.precomputed_datamodule import PrecomputedDataModule
        datamodule = PrecomputedDataModule(data=data)
        datamodule.setup()

        # 4. Execute algorithm
        log.debug(f"Running {algorithm}.fit_transform on data shape {data.shape}")
        embeddings = algo_instance.fit_transform(datamodule.train_dataset.data)

        # Convert to numpy if needed (manyLatents may return torch.Tensor)
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.numpy()
        elif hasattr(embeddings, 'detach'):
            embeddings = embeddings.detach().cpu().numpy()

        # 5. Compute metrics using CACHED objects
        scores = {}
        for metric_name, metric_cache in self._metric_cache.items():
            metric_obj = metric_cache['object']
            group = metric_cache['group']

            log.debug(f"Computing metric '{metric_name}' (group: {group})")

            # Call metric with appropriate signature for its group
            try:
                if group == 'embedding':
                    score = metric_obj(
                        embeddings=embeddings,
                        dataset=datamodule.train_dataset
                    )
                elif group == 'module':
                    score = metric_obj(
                        module=algo_instance,
                        embeddings=embeddings,
                        dataset=datamodule.train_dataset
                    )
                elif group == 'dataset':
                    score = metric_obj(
                        dataset=datamodule.train_dataset
                    )
                else:
                    log.warning(f"Unknown metric group '{group}' for metric '{metric_name}'")
                    continue

                scores[metric_name] = score
                log.debug(f"  {metric_name} = {score}")

            except Exception as e:
                log.warning(f"Failed to compute metric '{metric_name}': {e}")
                scores[metric_name] = None

        # 6. Build EmbeddingOutputs result
        result = {
            'embeddings': embeddings,
            'scores': scores,
            'metadata': {
                'algorithm': algorithm,
                'params': final_params,
                'data_shape': data.shape,
                'embedding_shape': embeddings.shape,
                'cached_execution': True,
                'metrics_computed': len(scores)
            },
            'success': True
        }

        log.info(
            f"✅ Fast execution complete: {algorithm} → {embeddings.shape}, "
            f"{len(scores)} metrics computed"
        )

        return result
