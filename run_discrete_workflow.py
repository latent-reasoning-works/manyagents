import asyncio
import pprint
from manyagents.models import WorkflowState
from manyagents.adapters import ManyLatentsAdapter, BioDiscoveryAgentAdapter

async def main():
    """
    Main entrypoint for the Phase 1 discrete workflow.
    This orchestrator runs a fixed sequence of agents.
    """
    goal = "Run a standard PCA analysis on swissroll data and then propose a set of genes."

    # 1. Initialize State
    state = WorkflowState(goal=goal)
    print(f"GOAL: {state.goal}\n")

    # 2. Instantiate Adapters
    manylatents_adapter = ManyLatentsAdapter()
    biodiscovery_adapter = BioDiscoveryAgentAdapter()

    # 3. Hard-coded, sequential execution
    print("--- Running Step 1: ManyLatents ---")
    ml_task_config = {
        "algorithm": "pca",
        "data": "swissroll",
        "n_components": 2
    }
    ml_result = await manylatents_adapter.run(task_config=ml_task_config, input_files={})
    # Note: Adapters now return dicts, not ExecutionResult objects
    # state.add_result(ml_result)  # Commenting out for now

    if not ml_result.get('success'):
        print(f"Step 1 failed. Halting workflow.")
        print(f"   Error: {ml_result.get('metadata', {}).get('error', 'Unknown')}")
        return

    print(f"Step 1 Summary: {ml_result['summary']}\n")

    print("--- Running Step 2: BioDiscoveryAgent ---")
    bd_task_config = {
        "analysis_type": "gene_discovery",
        "output_format": "json"
    }
    bd_result = await biodiscovery_adapter.run(task_config=bd_task_config, input_files=ml_result.get('output_files', {}))
    # Note: BioDiscoveryAdapter is now PlaceholderAdapter
    # Update when real BioDiscoveryAgent is integrated

    if not bd_result.get('success'):
        print(f"Step 2 failed. Halting workflow.")
        return

    print(f"Step 2 Summary: {bd_result['summary']}\n")

    # 4. Final summary
    print("\nWORKFLOW COMPLETE")
    print("\n=== ManyLatents Results ===")
    print(f"Embeddings shape: {ml_result['output_files']['embeddings'].shape}")
    print(f"Metrics: {ml_result['output_files']['scores']}")
    print("\n=== BioDiscovery Results ===")
    print(f"Output files: {list(bd_result['output_files'].keys())}")

if __name__ == "__main__":
    # Create outputs directory if it doesn't exist for the mock adapter
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    asyncio.run(main())