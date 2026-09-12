"""Adapter for ManyLatents agent using direct API calls."""

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

    PRODUCES_TEXT_RESPONSE = False

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
        input_data: Optional[np.ndarray] = None,
        logging_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute ManyLatents using direct API call with configurable logging.

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
            input_data: Optional 2-D (samples, features) numpy array from a previous
                step; replaces the need for a data name when chaining. Stored
                3-D trace tensors require layer selection first. Trajectory
                velocity/curvature use manylatents.metrics directly; see the
                README "From traces to geometry" example.
            logging_config: Optional logging configuration from the caller:
                - logging_mode: 'collect_only' | 'immediate' | 'disabled'
                - save_metrics: bool
                - save_visualizations: bool
                - wandb_run_id: Optional[str]
                - step_idx: int

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

            # Extract logging configuration from the caller
            logging_config = logging_config or {}
            logging_mode = logging_config.get('logging_mode', 'immediate')

            log.info(f"ManyLatents adapter executing with task config: {task_config}")
            log.info(f"Logging mode: {logging_mode}")

            # Note: GlobalHydra clearing is now handled inside manylatents.api.run()

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
                # Project name can be overridden by task_config
                default_project = 'manyagents_workflow'
                overrides = {
                    'project': task_config.get('project', default_project),
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

                    # Extract algorithm parameters
                    algo_config = {
                        '_target_': f'manylatents.algorithms.latent.{algo_name}.{algo_name.upper()}Module',
                    }

                    # Known algorithm parameters (add to algorithm config, not root)
                    algo_params = ['n_components', 'n_neighbors', 'min_dist', 'metric',
                                   'n_epochs', 'learning_rate', 'random_state']

                    for param in algo_params:
                        if param in task_config:
                            algo_config[param] = task_config[param]

                    # Default n_components if not provided
                    if 'n_components' not in algo_config:
                        algo_config['n_components'] = 2

                    overrides['algorithms'] = {
                        'latent': algo_config
                    }

                # Add any additional overrides from task_config
                # Exclude keys that are handled specially (including algorithm params)
                algo_params_set = {'n_components', 'n_neighbors', 'min_dist', 'metric',
                                   'n_epochs', 'learning_rate', 'random_state'}
                for key, value in task_config.items():
                    if key not in ['algorithm', 'data', 'pipeline', 'metrics', 'step_idx', 'step_name'] and key not in algo_params_set:
                        overrides[key] = value

            # Handle callbacks for step-aware visualization
            # Extract step info from task_config (don't pass to manylatents as overrides)
            step_idx = task_config.pop('step_idx', None)
            step_name = task_config.pop('step_name', None)
            # Pop to strip from overrides passed to manylatents (value unused here)
            task_config.pop('visualize_steps', None)

            # Check logging mode to determine if we should create visualizations
            # collect_only: NO visualizations (The caller will create later)
            # immediate: YES visualizations (log in real-time)
            # disabled: NO visualizations
            should_create_viz = (
                logging_mode == 'immediate' and
                logging_config.get('save_visualizations', False)
            )

            if step_idx is not None and step_name is not None and should_create_viz:
                # Only add visualization callbacks for standalone manyAgents runs
                # (not when orchestrated by the caller)

                # Configure PlotEmbeddings callback with step-aware logging key
                if 'callbacks' not in overrides:
                    overrides['callbacks'] = {}
                if 'embedding' not in overrides['callbacks']:
                    overrides['callbacks']['embedding'] = {}

                # Add PlotEmbeddings callback with custom log_key
                # Use a concrete save_dir since Hydra runtime may not be available
                import tempfile
                save_dir = tempfile.mkdtemp(prefix=f"step_{step_idx}_{step_name}_")

                overrides['callbacks']['embedding']['plot_embeddings'] = {
                    '_target_': 'manylatents.callbacks.embedding.plot_embeddings.PlotEmbeddings',
                    '_partial_': False,  # PlotEmbeddings is not a partial function
                    'save_dir': save_dir,
                    'experiment_name': f'step_{step_idx}_{step_name}',
                    # Note: log_key removed - not supported in installed manyLatents version
                    'title': f'Step {step_idx}: {step_name}',
                    'figsize': [8, 6],
                    'legend': False
                }
                log.info(f"📊 Configured step-aware PlotEmbeddings for step {step_idx} ({step_name})")
                log.info(f"📊 Plots will be saved to: {save_dir}")
            else:
                log.info(f"🎯 Logging mode '{logging_mode}' - skipping manyLatents visualizations")

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

            # Handle callbacks specifically - load from manyLatents default configs
            # Same pattern as metrics - allows the caller to control manyLatents callbacks
            if 'callbacks' in task_config:
                log.info("🔄 Starting callback transformation")
                from omegaconf import OmegaConf
                from pathlib import Path
                import os
                import tempfile

                # Find manyLatents config directory
                manylatents_config_dir = Path(os.path.dirname(__import__('manylatents').__file__)) / 'configs'
                log.info(f"Loading callback configs from manyLatents: {manylatents_config_dir}")

                transformed_callbacks = {}

                for group, group_callbacks in task_config['callbacks'].items():
                    transformed_callbacks[group] = {}

                    for callback_name, callback_params in group_callbacks.items():
                        try:
                            # Load the default config from manyLatents using OmegaConf
                            config_path = manylatents_config_dir / 'callbacks' / group / f'{callback_name}.yaml'

                            if config_path.exists():
                                # Load YAML directly with OmegaConf
                                callback_cfg = OmegaConf.load(config_path)

                                # Extract the callback config (structure: {callback_name: {_target_: ...}})
                                if callback_name in callback_cfg:
                                    # Get the config as OmegaConf object first (don't resolve yet)
                                    callback_config_obj = callback_cfg[callback_name]

                                    # For PlotEmbeddings, override interpolated fields BEFORE resolving
                                    # This avoids Hydra interpolation issues
                                    if callback_name == 'plot_embeddings':
                                        # Use step info for unique directories and names
                                        if step_idx is not None and step_name is not None:
                                            save_dir = tempfile.mkdtemp(prefix=f"geomancer_step_{step_idx}_{step_name}_")
                                            exp_name = f"step_{step_idx}_{step_name}"
                                        else:
                                            save_dir = tempfile.mkdtemp(prefix="geomancer_callbacks_")
                                            exp_name = "callback_plot"

                                        # Override interpolated fields in OmegaConf object before resolution
                                        if 'save_dir' in callback_config_obj:
                                            OmegaConf.update(callback_config_obj, "save_dir", save_dir, merge=False)
                                        if 'experiment_name' in callback_config_obj:
                                            OmegaConf.update(callback_config_obj, "experiment_name", exp_name, merge=False)

                                        log.info(f"  📁 Set save_dir: {save_dir}")
                                        log.info(f"  📝 Set experiment_name: {exp_name}")

                                    # Now resolve to container (save_dir already set if needed)
                                    callback_config = OmegaConf.to_container(callback_config_obj, resolve=True)

                                    # Apply user overrides
                                    if callback_params:
                                        callback_config.update(callback_params)

                                    transformed_callbacks[group][callback_name] = callback_config
                                    log.info(f"  ✓ Loaded '{callback_name}' from {config_path.name}")
                                else:
                                    log.warning(
                                        f"Callback '{callback_name}' not found in {config_path}, "
                                        f"passing through as-is"
                                    )
                                    transformed_callbacks[group][callback_name] = callback_params
                            else:
                                log.warning(
                                    f"Callback config not found: {config_path}, "
                                    f"passing through as-is"
                                )
                                transformed_callbacks[group][callback_name] = callback_params

                        except Exception as e:
                            log.warning(
                                f"Failed to load callback '{callback_name}' from manyLatents: {e}, "
                                f"passing through as-is"
                            )
                            transformed_callbacks[group][callback_name] = callback_params

                overrides['callbacks'] = transformed_callbacks
                log.info(f"✅ Composed {len(transformed_callbacks)} callback groups from manyLatents configs")

            # CRITICAL: Run manyLatents in offline mode when collecting for aggregation
            # This prevents WandB conflicts and lets us aggregate metrics later
            logging_mode = logging_config.get('logging_mode', 'immediate')

            # Force offline mode for callbacks in orchestrated execution
            if logging_mode == 'collect_only' and 'callbacks' in overrides:
                for group in overrides['callbacks']:
                    for callback_name in overrides['callbacks'][group]:
                        if callback_name == 'plot_embeddings':
                            overrides['callbacks'][group][callback_name]['enable_wandb_upload'] = False
                            log.info(f"  🔒 Forced offline mode for {callback_name}")

            if logging_mode == 'collect_only':
                # Expert workflow: Disable WandB in manyLatents (callbacks already set to offline mode above)
                log.info("🔇 DISABLING all WandB in manyLatents (The caller will handle logging)")
                # Set debug=True to trigger wandb mode='disabled' in manyLatents
                overrides['debug'] = True
                overrides['logger'] = None
                # NOTE: Don't override callbacks here - they're already configured above with offline mode
            else:
                # RL or immediate mode: Use existing WandB run
                log.info("🔇 Setting logger: null to use existing WandB run (if any)")
                overrides['logger'] = None

            # Validate manylatents config structure
            validate_manylatents_config(overrides, input_data=input_data)

            log.info("Calling manylatents.api.run() with validated config")

            # DEBUG: Log the complete callbacks configuration
            if 'callbacks' in overrides:
                import json
                log.info("🔍 DEBUG: Full callbacks config being passed to manyLatents:")
                log.info(f"🔍 DEBUG: {json.dumps(overrides['callbacks'], indent=2)}")
            else:
                log.info("🔍 DEBUG: No callbacks in overrides")

            # Call manyLatents API
            # With WANDB_MODE='disabled', logger=None, and callbacks={},
            # manyLatents will run WITHOUT any WandB operations
            log.info("🚀 Calling manylatents.api.run() - WandB operations disabled")

            manylatents_result = run(input_data=input_data, **overrides)
            log.info("✅ manyLatents execution completed successfully")

            # Validate that result is EmbeddingOutputs format
            embedding_outputs = validate_embedding_outputs(
                manylatents_result,
                source=f"{self.name}_result"
            )

            # Note: With logging_mode='collect_only', manyLatents runs with NO WandB
            # The caller will create visualizations from the returned embeddings

            # Extract key components
            embeddings = embedding_outputs['embeddings']
            scores = embedding_outputs.get('scores', {})
            metadata = embedding_outputs.get('metadata', {})

            # Build summary
            summary_parts = [
                "ManyLatents successfully executed",
                f"Output shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}",
            ]

            if scores:
                # Handle metric values: float, tuple[float, array], or dict
                def extract_scalar(v):
                    """Extract displayable scalar from metric value."""
                    if isinstance(v, tuple) and len(v) == 2:
                        return v[0]  # (scalar, per_sample) -> scalar
                    elif isinstance(v, dict):
                        if v.get('status', 'measured') != 'measured':
                            return None
                        return v.get('scalar')
                    elif hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
                        # Array-like (numpy array, list, etc.) - take mean
                        import numpy as np
                        return float(np.mean(v))
                    else:
                        return float(v)  # Already a scalar

                from manyagents.metrics.llm import format_metric

                metric_summary = ', '.join(
                    f'{k}={format_metric(extract_scalar(v), ".3f")}'
                    for k, v in list(scores.items())[:3]
                )
                summary_parts.append(f"Metrics: {metric_summary}")

            # Extract callback outputs if any
            callback_outputs = embedding_outputs.get('callback_outputs', {})

            # Build standardized adapter result
            adapter_result = {
                'summary': ' | '.join(summary_parts),
                'success': True,
                'embeddings': embedding_outputs,  # Full EmbeddingOutputs dict
                'output_files': {
                    'embeddings': embeddings,  # numpy array for convenience
                    'scores': scores,
                    'metadata': metadata,
                    'callback_outputs': callback_outputs  # NEW: Pass through callback outputs
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

    def _log_step_visualizations_to_wandb(self, save_dir: str, step_idx: int, step_name: str) -> None:
        """
        Log generated visualization plots to WandB.

        This is needed because we hide the WandB run from manyLatents to prevent
        it from calling wandb.finish(), but that also prevents callbacks from
        logging images. So we log them ourselves after execution.
        """
        import glob
        import wandb
        from pathlib import Path

        if wandb.run is None:
            log.info("No WandB run available for visualization logging")
            return

        # Check if run is still active
        try:
            # Try a simple log to test if run is active
            wandb.log({"_test": 1}, step=step_idx)
            log.info(f"✅ WandB run {wandb.run.id} is active and ready for logging")
        except Exception as e:
            log.warning(f"WandB run appears to be finished or inactive: {e}")
            return

        # Find all PNG files in the save directory
        plot_files = glob.glob(f"{save_dir}/*.png")

        if not plot_files:
            log.info(f"No plot files found in {save_dir}")
            return

        log.info(f"📊 Logging {len(plot_files)} visualization(s) to WandB for step {step_idx}")

        for plot_file in plot_files:
            try:
                plot_path = Path(plot_file)

                # Create a descriptive WandB key for this step's visualization
                wandb_key = f"step_{step_idx}/{step_name}/embedding_plot"

                # Log the image to WandB
                wandb.log({
                    wandb_key: wandb.Image(
                        plot_file,
                        caption=f"Step {step_idx}: {step_name} Embedding Visualization"
                    )
                }, step=step_idx)

                log.info(f"📈 Logged plot to WandB: {wandb_key} -> {plot_path.name}")

            except Exception as e:
                log.warning(f"Failed to log plot {plot_file} to WandB: {e}")

        log.info(f"✅ Step {step_idx} visualizations logged to WandB")

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

    def _find_sweep_params(self, params: Dict[str, Any]) -> tuple:
        """Find parameters with list values that need to be swept."""
        excluded_keys = {'_target_', '_partial_'}
        sweep_keys = []
        sweep_vals = []

        for key, value in params.items():
            if key not in excluded_keys and isinstance(value, (list, tuple)):
                sweep_keys.append(key)
                sweep_vals.append(list(value))

        return sweep_keys, sweep_vals

    def _coerce_param_value(self, value: Any) -> Any:
        """Coerce parameter value to native Python type for naming."""
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def _generate_metric_variants(
        self,
        name: str,
        base_params: Dict[str, Any]
    ) -> List[tuple]:
        """
        Generate metric variants by expanding list-valued parameters.

        For parameters with list values (e.g., n_neighbors=[15,25,50]),
        creates a Cartesian product of all combinations, each with a
        unique suffixed name (e.g., trustworthiness__n_neighbors_15).
        """
        sweep_keys, sweep_vals = self._find_sweep_params(base_params)

        if not sweep_keys:
            return [(name, base_params)]

        from itertools import product

        variants = []
        for combo in product(*sweep_vals):
            variant_params = dict(base_params)
            suffix_parts = []

            for key, val in zip(sweep_keys, combo):
                val = self._coerce_param_value(val)
                variant_params[key] = val
                suffix_parts.append(f"{key}_{val}")

            variant_name = f"{name}__{'_'.join(suffix_parts)}"
            variants.append((variant_name, variant_params))

        log.info(
            f"Expanded '{name}' into {len(variants)} variants "
            f"(sweep keys: {sweep_keys})"
        )

        return variants

    def _instantiate_metric(
        self,
        metric_class: type,
        metric_info: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Any:
        """Instantiate a metric as partial or full instance."""
        if metric_info.get('partial', True):
            return partial(metric_class, **params)
        return metric_class(**params)

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
            base_params = {
                **metric_info['defaults'],
                **global_overrides,
                **spec_params
            }

            # Generate variants for list-valued params (flatten_and_unroll pattern)
            variants = self._generate_metric_variants(name, base_params)

            # Create and cache each variant
            for variant_name, final_params in variants:
                log.debug(
                    f"Creating metric '{variant_name}' "
                    f"(class: {metric_info['class']}, group: {metric_info['group']})"
                )

                metric_obj = self._instantiate_metric(
                    metric_class, metric_info, final_params
                )

                self._metric_cache[variant_name] = {
                    'object': metric_obj,
                    'group': metric_info['group'],
                    'class': metric_info['class'],
                    'params': final_params
                }

                log.debug(f"Cached metric '{variant_name}' successfully")

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
