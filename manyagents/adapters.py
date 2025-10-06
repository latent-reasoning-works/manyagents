import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np

from manyagents.models import ExecutionResult
from manyagents.config_builder import ConfigBuilder

logger = logging.getLogger(__name__)


class AgentAdapter(ABC):
    """Abstract base class for all agent adapters."""

    @abstractmethod
    async def run(self, objective: str, input_files: Dict[str, Any]) -> ExecutionResult:
        """Executes the agent and returns a standardized result."""
        pass


class ManyLatentsAdapter(AgentAdapter):
    """
    Adapter for executing ManyLatents using the direct Python API (no subprocess).

    This new implementation calls manylatents.api.run() directly, enabling:
    - In-memory data passing between workflow steps
    - No subprocess overhead
    - Direct access to embeddings and metrics
    """

    def __init__(self, schema_path: Path = None):
        """Initialize with path to the schema YAML file."""
        if schema_path is None:
            # Default to config_map.yaml in project root
            schema_path = Path(__file__).parent.parent / "config_map.yaml"
        self.config_builder = ConfigBuilder(schema_path)

    async def run(
        self,
        objective: str,
        input_files: Dict[str, Any],
        input_data: Optional[np.ndarray] = None
    ) -> ExecutionResult:
        """
        Execute ManyLatents using direct API call.

        Args:
            objective: High-level request in format 'ALGORITHM on DATASET'
            input_files: Additional input files (reserved for future use)
            input_data: Optional numpy array from previous step (for chaining)

        Returns:
            ExecutionResult with embeddings, metrics, and metadata

        Example:
            adapter = ManyLatentsAdapter()
            result = await adapter.run("PCA on swissroll", {})
            embeddings = result.output_files['embeddings']
        """
        try:
            # Import manylatents API
            from manylatents.api import run

            # Parse objective into algorithm and dataset
            algorithm_alias, dataset_alias = self.config_builder.parse_objective(objective)
            logger.info(f"Parsed objective: {algorithm_alias} on {dataset_alias}")

            # Build configuration from schema
            schema = self.config_builder.schema
            overrides = {
                'debug': True,  # Disable wandb for agent workflows
                'project': 'manyagents_workflow',
            }

            # Map dataset alias
            if dataset_alias.upper() in schema['datasets']:
                overrides['data'] = schema['datasets'][dataset_alias.upper()]

            # Map algorithm alias to full config
            if algorithm_alias.upper() in schema['algorithms']:
                algo_config = schema['algorithms'][algorithm_alias.upper()]
                # Extract algorithm name from path (e.g., 'algorithms/latent=pca' -> 'pca')
                algo_name = algo_config['path'].split('=')[1]

                overrides['algorithms'] = {
                    'latent': {
                        '_target_': f'manylatents.algorithms.latent.{algo_name}.{algo_name.upper()}Module',
                        'n_components': 2  # Default
                    }
                }

            logger.info(f"Calling manylatents.api.run() with overrides: {overrides}")

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
                f"Successfully executed: {objective}",
                f"Output shape: {embeddings.shape if embeddings is not None else 'N/A'}",
            ]

            if scores:
                metric_summary = ', '.join(f'{k}={v:.3f}' for k, v in list(scores.items())[:3])
                summary_parts.append(f"Metrics: {metric_summary}")

            return ExecutionResult(
                agent_name='manylatents',
                status='success',
                summary=' | '.join(summary_parts),
                output_files={
                    'embeddings': embeddings,  # numpy array
                    'scores': scores,
                    'metadata': metadata
                }
            )

        except Exception as e:
            logger.error(f"ManyLatents API call failed: {e}", exc_info=True)
            return ExecutionResult(
                agent_name='manylatents',
                status='failure',
                summary=f"Failed to execute ManyLatents for objective: {objective}",
                error_message=str(e)
            )

class BioDiscoveryAgentAdapter(AgentAdapter):
    """A mock adapter for the BioDiscoveryAgent to demonstrate modularity."""

    async def run(self, objective: str, input_files: Dict[str, Any]) -> ExecutionResult:
        print(f"SIMULATING: BioDiscoveryAgent with objective: {objective}")
        # In a real scenario, this would use subprocess to call research_assistant.py
        await asyncio.sleep(2) # Simulate async work

        # This agent produces a gene list file
        gene_list_path = "outputs/biodiscovery_genes.txt"
        with open(gene_list_path, "w") as f:
            f.write("GENE1, GENE2, GENE3")

        return ExecutionResult(
            agent_name='biodiscovery',
            status='success',
            summary="Proposed 3 new genes for experimental testing.",
            output_files={'gene_list': gene_list_path}
        )