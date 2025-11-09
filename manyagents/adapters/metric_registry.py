"""Unified registry for fast metric and algorithm lookup and instantiation.

The Registry provides a simple interface to query metric and algorithm information
and instantiate metric classes without Hydra overhead.

This is essential for RL training loops where metrics need to be instantiated
quickly thousands of times.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import importlib

logger = logging.getLogger(__name__)


class MetricRegistry:
    """
    Registry for looking up manyLatents metric information.

    The registry maps simple metric names (e.g., 'participation_ratio')
    to their class paths, default parameters, and group information.

    Usage:
        >>> registry = MetricRegistry()
        >>> info = registry.get_metric_info('participation_ratio')
        >>> print(info['class'])
        'manylatents.metrics.participation_ratio.ParticipationRatio'
        >>> print(info['defaults'])
        {'n_neighbors': 25, 'return_per_sample': True}

        >>> # Get metric class for instantiation
        >>> metric_class = registry.get_metric_class('participation_ratio')
        >>> metric = metric_class(n_neighbors=30)
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize metric registry.

        Args:
            registry_path: Path to metric_registry.json.
                          If None, uses default location in package data.

        Raises:
            FileNotFoundError: If registry file doesn't exist.
                              Run generation first: `python -m manyagents.adapters._generate_metric_registry`
        """
        if registry_path is None:
            # Default to package data directory
            registry_path = Path(__file__).parent / 'data' / 'metric_registry.json'

        if not registry_path.exists():
            raise FileNotFoundError(
                f"Metric registry not found at {registry_path}. "
                f"The registry should be auto-generated during package installation. "
                f"To generate manually, run: "
                f"python -m manyagents.adapters._generate_metric_registry"
            )

        self.registry_path = registry_path
        self._registry = self._load_registry()
        self._class_cache: Dict[str, Type] = {}

        logger.info(
            f"Loaded metric registry with {self.metadata['metrics_scanned']} metrics "
            f"(manylatents v{self.metadata['manylatents_version']})"
        )

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from JSON file."""
        with open(self.registry_path) as f:
            return json.load(f)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get registry metadata (version, generation time, etc.)"""
        return self._registry.get('_metadata', {})

    @property
    def metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics data."""
        return self._registry.get('metrics', {})

    def list_metrics(self, group: Optional[str] = None) -> List[str]:
        """
        List available metric names.

        Args:
            group: Filter by group ('embedding', 'dataset', 'module').
                  If None, returns all metrics.

        Returns:
            List of metric names
        """
        if group is None:
            return list(self.metrics.keys())

        return [
            name for name, info in self.metrics.items()
            if info['group'] == group
        ]

    def get_metric_info(self, name: str) -> Dict[str, Any]:
        """
        Get metric information.

        Args:
            name: Metric name (e.g., 'participation_ratio', 'lid')

        Returns:
            Metric info dict with keys:
            - class: Full class path
            - group: 'embedding', 'dataset', or 'module'
            - defaults: Default parameter values
            - partial: Whether metric uses _partial_
            - source_file: Original config file path

        Raises:
            KeyError: If metric name not found
        """
        if name not in self.metrics:
            available = ', '.join(sorted(self.metrics.keys())[:10])
            raise KeyError(
                f"Metric '{name}' not found in registry. "
                f"Available metrics: {available}... "
                f"(total: {len(self.metrics)})"
            )

        return self.metrics[name]

    def get_metric_class(self, name: str) -> Type:
        """
        Get metric class for instantiation.

        Args:
            name: Metric name

        Returns:
            Metric class (uninstantiated)

        Raises:
            KeyError: If metric not found
            ImportError: If metric class cannot be imported

        Example:
            >>> registry = MetricRegistry()
            >>> PR = registry.get_metric_class('participation_ratio')
            >>> metric = PR(n_neighbors=30, return_per_sample=True)
        """
        # Check cache first
        if name in self._class_cache:
            return self._class_cache[name]

        # Get metric info
        info = self.get_metric_info(name)
        class_path = info['class']

        # Parse class path: 'manylatents.metrics.module.ClassName'
        module_path, class_name = class_path.rsplit('.', 1)

        try:
            # Import module
            module = importlib.import_module(module_path)

            # Get class
            metric_class = getattr(module, class_name)

            # Cache for future use
            self._class_cache[name] = metric_class

            return metric_class

        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import metric class '{class_path}': {e}. "
                f"Is manylatents installed correctly?"
            ) from e

    def get_defaults(self, name: str) -> Dict[str, Any]:
        """
        Get default parameters for a metric.

        Args:
            name: Metric name

        Returns:
            Dict of default parameter values
        """
        info = self.get_metric_info(name)
        return info.get('defaults', {})

    def get_group(self, name: str) -> str:
        """
        Get metric group.

        Args:
            name: Metric name

        Returns:
            Group name: 'embedding', 'dataset', or 'module'
        """
        info = self.get_metric_info(name)
        return info['group']

    def __contains__(self, name: str) -> bool:
        """Check if metric exists in registry."""
        return name in self.metrics

    def __len__(self) -> int:
        """Number of metrics in registry."""
        return len(self.metrics)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MetricRegistry(metrics={len(self)}, "
            f"version={self.metadata.get('manylatents_version', 'unknown')})"
        )


    # ========== Algorithm Methods ==========

    @property
    def algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Get all algorithms data."""
        return self._registry.get('algorithms', {})

    def list_algorithms(self) -> List[str]:
        """
        List available algorithm names.

        Returns:
            List of algorithm names (e.g., ['PCA', 'UMAP', 'PHATE'])
        """
        return list(self.algorithms.keys())

    def get_algorithm_info(self, name: str) -> Dict[str, Any]:
        """
        Get algorithm information.

        Args:
            name: Algorithm name (e.g., 'PCA', 'UMAP')

        Returns:
            Algorithm info dict with keys:
            - class: Full class path
            - defaults: Default parameter values
            - source: Source package name
            - source_file: Original config file path

        Raises:
            KeyError: If algorithm name not found
        """
        if name not in self.algorithms:
            available = ', '.join(sorted(self.algorithms.keys()))
            raise KeyError(
                f"Algorithm '{name}' not found in registry. "
                f"Available algorithms: {available}"
            )

        return self.algorithms[name]

    def get_algorithm_class(self, name: str) -> Type:
        """
        Get algorithm class for instantiation.

        Args:
            name: Algorithm name (e.g., 'PCA', 'UMAP')

        Returns:
            Algorithm class (uninstantiated)

        Raises:
            KeyError: If algorithm not found
            ImportError: If algorithm class cannot be imported

        Example:
            >>> registry = MetricRegistry()
            >>> PCA = registry.get_algorithm_class('PCA')
            >>> algo = PCA(n_components=50)
        """
        # Check cache first
        cache_key = f'algo_{name}'
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]

        # Get algorithm info
        info = self.get_algorithm_info(name)
        class_path = info['class']

        # Parse class path
        module_path, class_name = class_path.rsplit('.', 1)

        try:
            # Import module
            module = importlib.import_module(module_path)

            # Get class
            algo_class = getattr(module, class_name)

            # Cache for future use
            self._class_cache[cache_key] = algo_class

            return algo_class

        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import algorithm class '{class_path}': {e}. "
                f"Is manylatents installed correctly?"
            ) from e

    def get_algorithm_defaults(self, name: str) -> Dict[str, Any]:
        """
        Get default parameters for an algorithm.

        Args:
            name: Algorithm name

        Returns:
            Dict of default parameter values
        """
        info = self.get_algorithm_info(name)
        return info.get('defaults', {})


# Singleton instance for convenience
_registry_instance: Optional[MetricRegistry] = None


def get_metric_registry() -> MetricRegistry:
    """
    Get singleton MetricRegistry instance.

    Returns:
        Shared MetricRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        _registry_instance = MetricRegistry()

    return _registry_instance
