"""Hatch build hook for generating metric registry during package installation.

This build hook automatically generates the metric registry from manyLatents
configs whenever the package is installed or synced (e.g., via `uv sync` or
`pip install -e .`).

The registry is regenerated only when the manyLatents version changes.
"""

import logging
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger(__name__)


class CustomBuildHook(BuildHookInterface):
    """Build hook to generate metric registry before package build/install."""

    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        """
        Called before build/install.

        This generates the metric registry from manyLatents configs
        and places it in the package data directory.

        Args:
            version: Package version
            build_data: Build configuration data
        """
        logger.info("Initializing metric registry build hook")

        try:
            # Import generator (after manylatents is available)
            from manyagents.adapters._generate_metric_registry import generate_metric_registry

            # Determine output path (in package data)
            package_root = Path(__file__).parent / 'manyagents' / 'adapters' / 'data'
            output_path = package_root / 'metric_registry.json'

            logger.info(f"Generating metric registry at {output_path}")

            # Generate registry (with version-based diff detection)
            registry = generate_metric_registry(output_path, force=False)

            metrics_count = registry['_metadata']['metrics_scanned']
            manylatents_version = registry['_metadata']['manylatents_version']

            logger.info(
                f"✅ Metric registry ready: {metrics_count} metrics "
                f"(manylatents v{manylatents_version})"
            )

            # Ensure the generated file is included in the build
            # (hatchling will include it automatically if it's in the package)

        except Exception as e:
            logger.error(f"Failed to generate metric registry: {e}")
            # Don't fail the build, but log the error
            logger.warning(
                "Continuing build without metric registry. "
                "Registry can be generated manually later with: "
                "python -m manyagents.adapters._generate_metric_registry"
            )
