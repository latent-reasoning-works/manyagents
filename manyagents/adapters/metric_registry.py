"""Unified registry for fast metric and algorithm lookup and instantiation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import importlib

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Registry for looking up manyLatents metric and algorithm information."""

    def __init__(self, registry_path: Optional[Path] = None, auto_regenerate: bool = True):
        # No implicit cache in site-packages: the default registry lives in memory.
        self.registry_path = Path(registry_path) if registry_path is not None else None
        if self.registry_path is None:
            from ._generate_metric_registry import generate_metric_registry
            self._registry = generate_metric_registry()
        elif auto_regenerate:
            try:
                from ._generate_metric_registry import generate_metric_registry
                self._registry = generate_metric_registry(self.registry_path, force=False)
            except Exception:
                if not self.registry_path.is_file():
                    raise
                logger.warning("Auto-regeneration failed; loading configured registry", exc_info=True)
                self._registry = json.loads(self.registry_path.read_text())
        else:
            self._registry = json.loads(self.registry_path.read_text())

        self._class_cache: Dict[str, Type] = {}

        logger.info(
            f"Loaded metric registry with {self.metadata['metrics_scanned']} metrics "
            f"(manylatents v{self.metadata['manylatents_version']})"
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return generation metadata, including the installed manyLatents version."""
        return self._registry.get('_metadata', {})

    @property
    def metrics(self) -> Dict[str, Dict[str, Any]]:
        """Return metric names mapped to class paths, groups, defaults, and sources."""
        return self._registry.get('metrics', {})

    @property
    def algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Return algorithm names mapped to class paths, defaults, and sources."""
        return self._registry.get('algorithms', {})

    def list_metrics(self, group: Optional[str] = None) -> List[str]:
        """List metric names, optionally filtered by group."""
        if group is None:
            return list(self.metrics.keys())
        return [name for name, info in self.metrics.items() if info['group'] == group]

    def list_algorithms(self) -> List[str]:
        """Return the names of discovered YAML-backed algorithms."""
        return list(self.algorithms.keys())

    def _get_info(self, name: str, registry: Dict, entity_type: str) -> Dict[str, Any]:
        """Generic lookup for metrics or algorithms."""
        if name not in registry:
            available = ', '.join(sorted(registry.keys())[:10])
            raise KeyError(f"{entity_type} '{name}' not found. Available: {available}...")
        return registry[name]

    def get_metric_info(self, name: str) -> Dict[str, Any]:
        """Return the metric entry; raise KeyError for an undiscovered name."""
        return self._get_info(name, self.metrics, "Metric")

    def get_algorithm_info(self, name: str) -> Dict[str, Any]:
        """Return the algorithm entry; raise KeyError for an undiscovered name."""
        return self._get_info(name, self.algorithms, "Algorithm")

    def _get_class(self, name: str, cache_key: str, get_info_fn) -> Type:
        """Generic class loading with caching."""
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]

        info = get_info_fn(name)
        class_path = info['class']
        module_path, class_name = class_path.rsplit('.', 1)

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._class_cache[cache_key] = cls
            return cls
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import '{class_path}': {e}") from e

    def get_metric_class(self, name: str) -> Type:
        """Import and cache a metric class; raise KeyError or ImportError on failure."""
        return self._get_class(name, name, self.get_metric_info)

    def get_algorithm_class(self, name: str) -> Type:
        """Import and cache an algorithm class; raise KeyError or ImportError on failure."""
        return self._get_class(name, f'algo_{name}', self.get_algorithm_info)

    def get_defaults(self, name: str) -> Dict[str, Any]:
        """Return a metric's default parameters, or an empty dict when absent."""
        return self.get_metric_info(name).get('defaults', {})

    def get_algorithm_defaults(self, name: str) -> Dict[str, Any]:
        """Return an algorithm's default parameters, or an empty dict when absent."""
        return self.get_algorithm_info(name).get('defaults', {})

    def get_group(self, name: str) -> str:
        """Return the metric call group: embedding, dataset, or module."""
        return self.get_metric_info(name)['group']

    def __contains__(self, name: str) -> bool:
        return name in self.metrics

    def __len__(self) -> int:
        return len(self.metrics)

    def __repr__(self) -> str:
        return f"MetricRegistry(metrics={len(self)}, version={self.metadata.get('manylatents_version', 'unknown')})"


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
