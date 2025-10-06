#!/usr/bin/env python
"""
Example workflows demonstrating how to use ManyLatentsAdapter with the new API.

This shows various patterns:
1. Single algorithm run
2. Sequential workflow with data passing
3. Direct API usage (bypassing adapter)
"""

import asyncio
import numpy as np
from manyagents.adapters import ManyLatentsAdapter
from manyagents.models import WorkflowState


async def example1_single_run():
    """Example 1: Simple single algorithm run - Direct API call."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Algorithm Run (PCA)")
    print("="*60)

    adapter = ManyLatentsAdapter()

    # Run PCA on SwissRoll - direct task_config approach
    result = await adapter.run(
        task_config={
            "algorithm": "pca",
            "data": "swissroll",
            "n_components": 2
        },
        input_files={}
    )

    print(f"Success: {result['success']}")
    print(f"Summary: {result['summary']}")
    print(f"Embeddings shape: {result['output_files']['embeddings'].shape}")
    print(f"Metrics: {result['output_files']['scores']}")

    return result


async def example2_sequential_workflow():
    """Example 2: Multi-step workflow with in-memory data passing."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Sequential Workflow (PCA → UMAP)")
    print("="*60)

    adapter = ManyLatentsAdapter()

    # Step 1: PCA to 50 components
    print("\n--- Step 1: PCA (reduce to 50 dimensions) ---")
    result1 = await adapter.run(
        task_config={
            "algorithm": "pca",
            "data": "swissroll",
            "n_components": 50
        },
        input_files={}
    )

    if not result1['success']:
        print(f"Step 1 failed: {result1['metadata'].get('error')}")
        return

    print(f"Step 1: {result1['summary']}")
    embeddings_pca = result1['output_files']['embeddings']

    # Step 2: UMAP using PCA embeddings as input
    print("\n--- Step 2: UMAP (50D → 2D) ---")
    result2 = await adapter.run(
        task_config={
            "algorithm": "umap",
            "n_components": 2
        },
        input_files={},
        input_data=embeddings_pca  # Pass PCA output directly
    )

    print(f"Step 2: {result2['summary']}")
    print(f"Final embeddings shape: {result2['output_files']['embeddings'].shape}")

    return result2


async def example3_multiple_algorithms():
    """Example 3: Run multiple algorithms on same dataset."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Multiple Algorithms (PCA, UMAP, TSNE)")
    print("="*60)

    adapter = ManyLatentsAdapter()
    algorithms = ["pca", "umap", "tsne"]
    results = {}

    for algo in algorithms:
        print(f"\n--- Running {algo.upper()} ---")
        result = await adapter.run(
            task_config={
                "algorithm": algo,
                "data": "swissroll",
                "n_components": 2
            },
            input_files={}
        )

        results[algo] = result
        print(f"{algo.upper()}: {result['summary']}")
        if result['output_files'].get('scores'):
            scores = result['output_files']['scores']
            print(f"  Metrics: {', '.join(f'{k}={v:.3f}' for k, v in list(scores.items())[:3])}")

    print("\n--- Comparison ---")
    for algo, result in results.items():
        shape = result['output_files']['embeddings'].shape
        print(f"{algo.upper()}: {shape}, success={result['success']}")

    return results


async def example4_using_config_file():
    """Example 4: Using an experiment config file."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Using Experiment Config File")
    print("="*60)

    from manylatents.api import run

    # Use the pca_phate_pipeline.yaml config
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: run(experiment='pca_phate_pipeline')
    )

    print(f"Config-driven pipeline complete!")
    print(f"Final embeddings shape: {result['embeddings'].shape}")

    return result


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("MANYAGENTS + MANYLATENTS API WORKFLOW EXAMPLES")
    print("="*80)

    # Example 1: Single run
    await example1_single_run()

    # Example 2: Sequential workflow
    await example2_sequential_workflow()

    # Example 3: Direct API pipeline
    await example3_direct_api_pipeline()

    # Example 4: Config file
    # await example4_using_config_file()  # Uncomment to test

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
