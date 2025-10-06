import yaml
from pathlib import Path
from typing import Dict, Any, Tuple


class ConfigBuilder:
    """Builds ManyLatents configuration strings from high-level aliases using a schema file."""

    def __init__(self, schema_path: Path):
        """Initialize with path to the schema YAML file."""
        with open(schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)

    def build(self, dataset_alias: str, algorithm_alias: str) -> str:
        """
        Builds a ManyLatents config string from high-level aliases.

        Args:
            dataset_alias: Human-readable dataset name (e.g., 'swissroll')
            algorithm_alias: Human-readable algorithm name (e.g., 'pca')

        Returns:
            String of space-separated Hydra overrides for ManyLatents

        Raises:
            ValueError: If aliases are not found in schema
        """
        try:
            dataset = self.schema['datasets'][dataset_alias.upper()]
            algorithm = self.schema['algorithms'][algorithm_alias.upper()]
            template = self.schema['template']['base']
        except KeyError as e:
            available_datasets = list(self.schema['datasets'].keys())
            available_algorithms = list(self.schema['algorithms'].keys())
            raise ValueError(
                f"Invalid alias provided: {e}. "
                f"Available datasets: {available_datasets}. "
                f"Available algorithms: {available_algorithms}"
            ) from e

        # Construct the final command-line override string
        config_parts = [
            template,
            f"data={dataset}",
            algorithm['base'],
            algorithm['path']
        ]
        return " ".join(config_parts)

    def parse_objective(self, objective: str) -> Tuple[str, str]:
        """
        Parse a high-level objective string like 'PCA on swissroll' into components.

        Args:
            objective: String in format 'ALGORITHM on DATASET'

        Returns:
            Tuple of (algorithm_alias, dataset_alias)

        Raises:
            ValueError: If objective format is invalid
        """
        parts = objective.strip().split(" on ")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid objective format: '{objective}'. "
                "Expected format: 'ALGORITHM on DATASET' (e.g., 'PCA on swissroll')"
            )

        algorithm_alias = parts[0].strip()
        dataset_alias = parts[1].strip()

        return algorithm_alias, dataset_alias

    def build_from_objective(self, objective: str) -> str:
        """
        Build ManyLatents config from a complete objective string.

        Args:
            objective: String like 'PCA on swissroll'

        Returns:
            ManyLatents configuration string
        """
        algorithm_alias, dataset_alias = self.parse_objective(objective)
        return self.build(dataset_alias, algorithm_alias)