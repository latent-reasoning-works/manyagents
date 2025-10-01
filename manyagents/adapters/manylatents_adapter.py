"""Adapter for ManyLatents agent."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

from .base import AgentAdapter
from ..executor import ManyLatentsExecutor

log = logging.getLogger(__name__)


class ManyLatentsAdapter(AgentAdapter):
    """Adapter for ManyLatents dimensionality reduction and analysis."""

    def __init__(self, timeout_s: int = 300, dry_run: bool = False):
        super().__init__("manylatents")
        self.executor = ManyLatentsExecutor(timeout_s=timeout_s, dry_run=dry_run)

    async def run(self, task_config: Dict[str, Any], input_files: Dict[str, Path]) -> Dict[str, Any]:
        """
        Execute ManyLatents analysis with structured configuration.

        Args:
            task_config: Dictionary with ManyLatents-specific parameters:
                - workflow: ManyLatents workflow name (default: "single_algorithm")
                - algorithm: Algorithm type (e.g., "pca", "umap", "tsne")
                - data: Dataset name (e.g., "swissroll", "GaussianBlobs")
                - metrics: Metrics configuration (optional)
                - overrides: Additional Hydra overrides (optional)
            input_files: Input data files (currently not used by ManyLatents)

        Returns:
            Standardized result dictionary
        """
        log.info(f"ManyLatents executing with config: {task_config}")

        # Extract configuration with defaults
        workflow_name = task_config.get("workflow", "single_algorithm")
        overrides = self._build_overrides(task_config)

        # Create output directory for this execution
        output_dir = Path("outputs") / f"manylatents_{workflow_name}"

        # Execute in thread pool to avoid blocking async loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.executor.execute_workflow,
            workflow_name,
            overrides,
            output_dir
        )

        # Convert executor result to standardized format
        success = result.get("success", False)

        if success:
            summary = f"ManyLatents successfully executed {workflow_name} workflow"
            output_files = self._gather_output_files(output_dir)
        else:
            summary = f"ManyLatents failed: {result.get('stderr', 'Unknown error')}"
            output_files = {}

        return {
            "summary": summary,
            "output_files": output_files,
            "success": success,
            "metadata": {
                "workflow": workflow_name,
                "overrides": overrides,
                "command": result.get("cmd", []),
                "returncode": result.get("returncode"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")
            }
        }

    def _build_overrides(self, task_config: Dict[str, Any]) -> List[str]:
        """Build Hydra overrides from structured task configuration."""
        overrides = []

        # Algorithm configuration
        if "algorithm" in task_config:
            overrides.append("algorithm=default")
            overrides.append(f"algorithm.latent={task_config['algorithm']}")

        # Data configuration
        if "data" in task_config:
            overrides.append(f"data={task_config['data']}")

        # Metrics configuration
        if "metrics" in task_config:
            overrides.append(f"metrics={task_config['metrics']}")

        # Additional custom overrides
        if "overrides" in task_config:
            overrides.extend(task_config["overrides"])

        log.info(f"Built overrides: {overrides}")
        return overrides

    def _gather_output_files(self, output_dir: Path) -> Dict[str, List[Path]]:
        """Gather output files produced by ManyLatents execution."""
        output_files = {}

        if not output_dir.exists():
            return output_files

        # Look for common ManyLatents output patterns
        for pattern, file_type in [
            ("*.png", "plots"),
            ("*.pdf", "plots"),
            ("*.csv", "embeddings"),
            ("*.json", "metrics"),
            ("*.yaml", "config"),
            ("*.pt", "model"),
            ("*.ckpt", "checkpoint")
        ]:
            files = list(output_dir.glob(f"**/{pattern}"))
            if files:
                output_files[file_type] = files

        return output_files