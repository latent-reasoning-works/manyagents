"""
Simplified workflow entry point for direct execution.

This is a convenience script that demonstrates how to run workflows
programmatically without Hydra CLI. For production use, prefer:

    uv run manyagents experiment=single_algorithm

This script is useful for:
- Quick testing during development
- Integration into other Python scripts
- Understanding the workflow API
"""
import asyncio
from pathlib import Path
from omegaconf import OmegaConf

from manyagents.main import execute_workflow_chain


async def run_example_workflow():
    """
    Example: Run a simple PCA workflow programmatically.

    This demonstrates how to construct a workflow config in Python
    and execute it directly, bypassing Hydra CLI.
    """
    print("="*60)
    print("Running example workflow: PCA on SwissRoll")
    print("="*60)

    # Construct workflow config programmatically
    # This mimics the structure in configs/experiment/single_algorithm.yaml
    workflow_config = OmegaConf.create({
        "name": "programmatic_example",
        "workflow": {
            "steps": [
                {
                    "name": "pca_embedding",
                    "agent": "manylatents",
                    "config": {
                        "algorithm": "pca",
                        "data": "swissroll",
                        "n_components": 2,
                        "seed": 42
                    }
                }
            ]
        },
        "output_dir": "outputs/programmatic_example"
    })

    # Create output directory
    output_dir = Path(workflow_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute workflow
    result = await execute_workflow_chain(workflow_config, output_dir)

    # Print results
    print("\n" + "="*60)
    print("WORKFLOW RESULTS")
    print("="*60)

    if result["success"]:
        print(f"✓ Workflow completed successfully")
        print(f"✓ Total steps: {result['metadata']['total_steps']}")

        for step in result["steps"]:
            print(f"\n{step['name']}:")
            print(f"  Summary: {step['result']['summary']}")

            # Print embeddings info if available
            if 'embeddings' in step['result']['output_files']:
                embeddings = step['result']['output_files']['embeddings']
                print(f"  Embeddings shape: {embeddings.shape}")

            # Print metrics if available
            if 'scores' in step['result']['output_files']:
                scores = step['result']['output_files']['scores']
                print(f"  Metrics: {list(scores.keys())}")
    else:
        print(f"✗ Workflow failed")
        print(f"  Error: {result.get('metadata', {}).get('error', 'Unknown')}")

    print("\n" + "="*60)
    return result


async def run_multi_step_workflow():
    """
    Example: Run a multi-step PCA → UMAP pipeline.

    This demonstrates in-memory data passing between steps.
    """
    print("="*60)
    print("Running multi-step workflow: PCA → UMAP")
    print("="*60)

    workflow_config = OmegaConf.create({
        "name": "pca_umap_pipeline",
        "workflow": {
            "steps": [
                {
                    "name": "pca_preprocessing",
                    "agent": "manylatents",
                    "config": {
                        "algorithm": "pca",
                        "data": "swissroll",
                        "n_components": 50,
                        "seed": 42
                    }
                },
                {
                    "name": "umap_visualization",
                    "agent": "manylatents",
                    "config": {
                        "algorithm": "umap",
                        "n_components": 2,
                        "seed": 42
                    }
                }
            ]
        },
        "output_dir": "outputs/pca_umap_pipeline"
    })

    output_dir = Path(workflow_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = await execute_workflow_chain(workflow_config, output_dir)

    print("\n" + "="*60)
    print("MULTI-STEP WORKFLOW RESULTS")
    print("="*60)

    if result["success"]:
        print(f"✓ Pipeline completed successfully")

        for step in result["steps"]:
            print(f"\n{step['name']}:")
            print(f"  {step['result']['summary']}")
    else:
        print(f"✗ Pipeline failed")

    print("\n" + "="*60)
    return result


async def main():
    """Run example workflows."""
    print("\n" + "="*80)
    print("MANYAGENTS WORKFLOW EXAMPLES")
    print("="*80)

    # Run single-step workflow
    print("\n### Example 1: Single Algorithm ###\n")
    await run_example_workflow()

    # Run multi-step workflow
    print("\n### Example 2: Multi-Step Pipeline ###\n")
    await run_multi_step_workflow()

    print("\n" + "="*80)
    print("For production use, run via Hydra:")
    print("  uv run manyagents experiment=single_algorithm")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
