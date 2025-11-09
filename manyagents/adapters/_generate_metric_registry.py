"""Generate metric registry from manyLatents configs.

This module scans manyLatents metric configuration files and builds a registry
mapping metric names to their class paths, groups, and default parameters.

The registry is used by the ManyLatentsAdapter to instantiate metrics efficiently
during RL training loops.

Extension Support:
    The generator automatically discovers manyLatents extensions (packages with
    names starting with 'manylatents-') and scans their metric configs too.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import importlib.metadata
import importlib.util

logger = logging.getLogger(__name__)


def discover_manylatents_extensions() -> List[Tuple[str, Path]]:
    """
    Discover installed manyLatents extension packages.

    Returns:
        List of (package_name, metrics_dir_path) tuples for extensions
        that have metric configs.
    """
    extensions = []

    try:
        # Get all installed distributions
        for dist in importlib.metadata.distributions():
            package_name = dist.metadata['Name']

            # Look for manylatents extensions (packages starting with 'manylatents-')
            if package_name.startswith('manylatents-') or package_name.startswith('manylatents_'):
                try:
                    # Try to import the package to find its location
                    # Convert package name to module name (replace - with _)
                    module_name = package_name.replace('-', '_')

                    # Find the module spec
                    spec = importlib.util.find_spec(module_name)
                    if spec is None or spec.origin is None:
                        continue

                    # Get package root
                    package_root = Path(spec.origin).parent

                    # Look for metrics configs (try both package root and parent)
                    potential_metrics_dirs = [
                        package_root / 'configs' / 'metrics',  # In package
                        package_root.parent / 'configs' / 'metrics',  # In repo root
                    ]

                    for metrics_dir in potential_metrics_dirs:
                        if metrics_dir.exists() and metrics_dir.is_dir():
                            # Check if it has any metric group subdirs
                            has_metrics = any(
                                (metrics_dir / group).exists()
                                for group in ['embedding', 'dataset', 'module']
                            )

                            if has_metrics:
                                extensions.append((package_name, metrics_dir))
                                logger.info(f"Discovered extension: {package_name} at {metrics_dir}")
                                break

                except Exception as e:
                    logger.debug(f"Could not inspect extension {package_name}: {e}")
                    continue

    except Exception as e:
        logger.warning(f"Failed to discover extensions: {e}")

    return extensions


def scan_metric_configs(metrics_dir: Path, source_label: str = "core") -> Dict[str, Any]:
    """
    Scan manyLatents metrics directory and build registry.

    Args:
        metrics_dir: Path to manylatents/configs/metrics/

    Returns:
        Registry dict with structure:
        {
            "metric_name": {
                "class": "manylatents.metrics.module.ClassName",
                "group": "embedding" | "dataset" | "module",
                "defaults": {"param1": value1, ...},
                "partial": True | False
            },
            ...
        }
    """
    registry = {}

    # Scan each group directory
    groups = ['embedding', 'dataset', 'module']

    for group in groups:
        group_dir = metrics_dir / group
        if not group_dir.exists():
            logger.warning(f"Metrics group directory not found: {group_dir}")
            continue

        # Scan all YAML files in group
        for config_file in group_dir.glob('*.yaml'):
            # Skip test files and __init__
            if config_file.stem.startswith('test_') or config_file.stem == '__init__':
                continue

            try:
                with open(config_file) as f:
                    config_data = yaml.safe_load(f)

                if not config_data:
                    continue

                # Extract metric info from config
                # Config format: {metric_name: {_target_: ..., _partial_: ..., param: value, ...}}
                for metric_name, metric_config in config_data.items():
                    if not isinstance(metric_config, dict):
                        continue

                    # Extract _target_ (class path)
                    target = metric_config.get('_target_')
                    if not target:
                        logger.warning(f"No _target_ found for {metric_name} in {config_file}")
                        continue

                    # Extract _partial_ (default True for metrics)
                    partial = metric_config.get('_partial_', True)

                    # Extract default parameters (everything except _target_ and _partial_)
                    defaults = {
                        k: v for k, v in metric_config.items()
                        if not k.startswith('_')
                    }

                    # Add to registry
                    registry[metric_name] = {
                        'class': target,
                        'group': group,
                        'defaults': defaults,
                        'partial': partial,
                        'source': source_label,
                        'source_file': str(config_file.relative_to(metrics_dir.parent.parent))
                    }

                    logger.debug(f"Registered metric: {metric_name} ({group}) from {source_label}/{config_file.name}")

            except Exception as e:
                logger.error(f"Failed to parse {config_file}: {e}")
                continue

    return registry


def scan_algorithm_configs(algorithms_dir: Path, source_label: str = "manylatents") -> Dict[str, Any]:
    """
    Scan manyLatents algorithms directory and build registry.

    Args:
        algorithms_dir: Path to manylatents/configs/algorithms/
        source_label: Source package name

    Returns:
        Registry dict with structure:
        {
            "PCA": {
                "class": "manylatents.algorithms.latent.pca.PCAModule",
                "defaults": {"n_components": 2},
                "source": "manylatents"
            },
            ...
        }
    """
    registry = {}

    # Scan latent algorithms directory
    latent_dir = algorithms_dir / 'latent'
    if not latent_dir.exists():
        logger.warning(f"Latent algorithms directory not found: {latent_dir}")
        return registry

    # Scan all YAML files in latent directory
    for config_file in latent_dir.glob('*.yaml'):
        # Skip test files and __init__
        if config_file.stem.startswith('test_') or config_file.stem == '__init__':
            continue

        try:
            with open(config_file) as f:
                config_data = yaml.safe_load(f)

            if not config_data or not isinstance(config_data, dict):
                continue

            # Extract algorithm info from config
            # Config format: {_target_: ..., param: value, ...}
            target = config_data.get('_target_')
            if not target:
                logger.warning(f"No _target_ found in {config_file}")
                continue

            # Algorithm name is the file stem, uppercase
            algo_name = config_file.stem.upper()

            # Extract default parameters (everything except _target_)
            defaults = {
                k: v for k, v in config_data.items()
                if not k.startswith('_') and '${' not in str(v)  # Skip Hydra variables
            }

            # Add to registry
            registry[algo_name] = {
                'class': target,
                'defaults': defaults,
                'source': source_label,
                'source_file': str(config_file.relative_to(algorithms_dir.parent.parent))
            }

            logger.debug(f"Registered algorithm: {algo_name} from {source_label}/{config_file.name}")

        except Exception as e:
            logger.error(f"Failed to parse {config_file}: {e}")
            continue

    return registry


def generate_metric_registry(output_path: Path, force: bool = False) -> Dict[str, Any]:
    """
    Generate unified manyLatents registry (metrics + algorithms) JSON file.

    Args:
        output_path: Path where registry JSON should be written
        force: If True, regenerate even if version matches

    Returns:
        Registry data (also written to output_path)
    """
    import manylatents
    from importlib.metadata import version

    try:
        current_version = version('manylatents')
    except Exception:
        # Fallback if version can't be determined
        current_version = "unknown"
        logger.warning("Could not determine manylatents version, using 'unknown'")

    # Check if regeneration is needed
    if output_path.exists() and not force:
        try:
            with open(output_path) as f:
                existing_registry = json.load(f)

            stored_version = existing_registry.get('_metadata', {}).get('manylatents_version')

            if stored_version == current_version:
                logger.info(
                    f"Metric registry up-to-date (manylatents v{current_version}). "
                    f"Skipping regeneration."
                )
                return existing_registry
        except Exception as e:
            logger.warning(f"Could not read existing registry: {e}. Regenerating.")

    # Find manylatents metrics directory
    manylatents_path = Path(manylatents.__file__).parent
    metrics_dir = manylatents_path / 'configs' / 'metrics'

    if not metrics_dir.exists():
        raise FileNotFoundError(
            f"manyLatents metrics directory not found at {metrics_dir}. "
            f"Is manylatents installed correctly?"
        )

    logger.info(f"Scanning manyLatents core metrics from: {metrics_dir}")

    # Scan core manyLatents metrics
    metrics_registry = scan_metric_configs(metrics_dir, source_label="manylatents")

    # Scan core manyLatents algorithms
    algorithms_dir = manylatents_path / 'configs' / 'algorithms'
    if algorithms_dir.exists():
        logger.info(f"Scanning manyLatents core algorithms from: {algorithms_dir}")
        algorithms_registry = scan_algorithm_configs(algorithms_dir, source_label="manylatents")
    else:
        logger.warning(f"Algorithms directory not found at {algorithms_dir}")
        algorithms_registry = {}

    # Discover and scan extensions
    extensions = discover_manylatents_extensions()
    extension_names = []

    for ext_name, ext_metrics_dir in extensions:
        logger.info(f"Scanning extension '{ext_name}' metrics from: {ext_metrics_dir}")
        ext_registry = scan_metric_configs(ext_metrics_dir, source_label=ext_name)

        # Merge extension metrics into registry
        # Note: Extensions can override core metrics if they use the same name
        for metric_name, metric_info in ext_registry.items():
            if metric_name in metrics_registry:
                logger.warning(
                    f"Extension '{ext_name}' overrides metric '{metric_name}' "
                    f"from '{metrics_registry[metric_name]['source']}'"
                )
            metrics_registry[metric_name] = metric_info

        extension_names.append(ext_name)

    # Build metadata
    metric_sources = {}
    for metric_info in metrics_registry.values():
        source = metric_info['source']
        metric_sources[source] = metric_sources.get(source, 0) + 1

    algo_sources = {}
    for algo_info in algorithms_registry.values():
        source = algo_info['source']
        algo_sources[source] = algo_sources.get(source, 0) + 1

    full_registry = {
        '_metadata': {
            'generated_at': str(Path(__file__).stat().st_mtime),
            'manylatents_version': current_version,
            'manylatents_path': str(manylatents_path),
            'metrics_scanned': len(metrics_registry),
            'algorithms_scanned': len(algorithms_registry),
            'extensions_found': len(extensions),
            'extensions': extension_names,
            'metric_sources': metric_sources,
            'algorithm_sources': algo_sources,
            'metric_groups': {
                'embedding': len([m for m in metrics_registry.values() if m['group'] == 'embedding']),
                'dataset': len([m for m in metrics_registry.values() if m['group'] == 'dataset']),
                'module': len([m for m in metrics_registry.values() if m['group'] == 'module']),
            }
        },
        'metrics': metrics_registry,
        'algorithms': algorithms_registry
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write registry
    with open(output_path, 'w') as f:
        json.dump(full_registry, f, indent=2, sort_keys=True)

    logger.info(
        f"Generated manyLatents registry: {len(metrics_registry)} metrics, "
        f"{len(algorithms_registry)} algorithms (v{current_version}) -> {output_path}"
    )

    return full_registry


def main():
    """CLI entry point for manual registry generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate manyAgents metric registry')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent / 'data' / 'metric_registry.json',
        help='Output path for registry JSON'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if version matches'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    try:
        registry = generate_metric_registry(args.output, force=args.force)
        print(f"✅ Generated registry with {registry['_metadata']['metrics_scanned']} metrics")
        print(f"   Output: {args.output}")
        print(f"   Metrics: {registry['_metadata']['metrics_scanned']}, Algorithms: {registry['_metadata']['algorithms_scanned']}")
        print(f"   Metric groups: {registry['_metadata']['metric_groups']}")
    except Exception as e:
        print(f"❌ Failed to generate registry: {e}")
        raise


if __name__ == '__main__':
    main()
