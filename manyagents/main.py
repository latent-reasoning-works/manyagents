"""
Smart CLI router for ManyAgents.

Routes to appropriate execution engine based on config:
- If 'scenarios' key present → invariance experiment
- If 'workflow' key present → workflow execution

Usage:
    # Workflow execution
    manyagents experiment=single_algorithm

    # Invariance experiment (auto-detected from scenarios key)
    manyagents experiment=invariance_golden

    # With wandb
    manyagents experiment=invariance_golden wandb.enabled=true
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig

# Register shop's custom Hydra launchers (enables cluster=mila_remote)
try:
    from shop.hydra.config_store import register_shop_launchers
    register_shop_launchers()
except ImportError:
    pass  # shop not installed, cluster launchers won't be available

from manyagents.adapters.manylatents_adapter import ManyLatentsAdapter
from manyagents.adapters.openai_adapter import OpenAIAdapter
from manyagents.adapters.cellforge_adapter import CellForgeAdapter
from manyagents.adapters.biomni_adapter import BiomniAdapter
from manyagents.adapters.kosmos_adapter import KosmosAdapter
from manyagents.adapters.claude_adapter import ClaudeAdapter
from manyagents.adapters.local_llm_adapter import LocalLLMAdapter
from manyagents.adapters.mock_adapter import MockAdapter
# from manyagents.adapters.biodiscovery_adapter import BioDiscoveryAgentAdapter  # Coming soon

log = logging.getLogger(__name__)

# Import LoggingContext if Geomancer is available
try:
    geomancer_path = Path(__file__).resolve().parents[2] / "Geomancer"
    if geomancer_path.exists() and str(geomancer_path) not in sys.path:
        sys.path.insert(0, str(geomancer_path))
    from geomancy.logging_context import LoggingContext
    GEOMANCER_AVAILABLE = True
except ImportError:
    GEOMANCER_AVAILABLE = False
    log.debug("LoggingContext not available - running in standalone mode")

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.debug("WandB not available")

# ============================================================================
# ADAPTER REGISTRY
# ============================================================================
# Maps agent names in config files to their corresponding adapter classes
# This is the bridge between declarative configs and Python implementations

ADAPTER_REGISTRY = {
    "manylatents": ManyLatentsAdapter,
    "openai": OpenAIAdapter,
    "cellforge": CellForgeAdapter,
    "biomni": BiomniAdapter,
    "kosmos": KosmosAdapter,
    "claude": ClaudeAdapter,
    "local_llm": LocalLLMAdapter,
    "mock": MockAdapter,
    # "biodiscovery": BioDiscoveryAgentAdapter,  # Placeholder for Phase 2
}


# ============================================================================
# ORCHESTRATION ENGINE
# ============================================================================

async def execute_workflow_chain(
    workflow_config: DictConfig,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Execute a workflow chain using the adapter pattern.

    This is the core orchestration function for Phase 1. It:
    1. Reads a declarative workflow configuration
    2. Executes each step using the appropriate adapter
    3. Manages state passing between steps
    4. Returns final results

    Args:
        workflow_config: Hydra config containing workflow.steps
        output_dir: Optional output directory for results

    Returns:
        Dictionary containing:
            - success: bool
            - steps: List of step results
            - final_state: Final workflow state
            - metadata: Execution metadata

    Example workflow_config structure:
        workflow:
          steps:
            - name: pca_analysis
              agent: manylatents
              config:
                algorithm: pca
                data: swissroll
                n_components: 2
    """
    log.info("Starting workflow execution")

    # Initialize workflow state
    state = {
        "steps_completed": [],
        "current_data": None,  # In-memory data passing between steps
        "output_files": {},    # Accumulated output files
        "metadata": {}
    }

    steps = workflow_config.workflow.steps
    log.info(f"Workflow has {len(steps)} steps")

    # Execute each step in sequence
    for i, step_config in enumerate(steps):
        step_name = step_config.get("name", f"step_{i}")
        agent_name = step_config.get("agent")
        task_config = dict(step_config.get("config", {}))

        log.info(f"\n{'='*60}")
        log.info(f"Executing Step {i+1}/{len(steps)}: {step_name}")
        log.info(f"Agent: {agent_name}")
        log.info(f"Config: {task_config}")
        log.info(f"{'='*60}")

        # Set step context for child processes (if under Geomancer orchestration)
        if GEOMANCER_AVAILABLE:
            LoggingContext.set_step_context(i, step_name)
            log.info(f"Set step context: step_{i}_{step_name}")

        # Get adapter class from registry
        if agent_name not in ADAPTER_REGISTRY:
            error_msg = f"Unknown agent '{agent_name}'. Available: {list(ADAPTER_REGISTRY.keys())}"
            log.error(error_msg)
            return {
                "success": False,
                "steps": state["steps_completed"],
                "final_state": state,
                "metadata": {"error": error_msg}
            }

        adapter_class = ADAPTER_REGISTRY[agent_name]

        # Instantiate adapter
        adapter = adapter_class()
        log.info(f"Using adapter: {adapter_class.__name__}")

        # Prepare input files from previous step outputs
        input_files = state.get("output_files", {})

        # Get input data from previous step (for in-memory passing)
        input_data = state.get("current_data")

        # Inject step name for logging/tracking if not already specified
        # This gets passed through to the agent's logging system (e.g., wandb)
        task_config_with_name = task_config.copy()
        if "name" not in task_config_with_name:
            task_config_with_name["name"] = f"step{i}_{step_name}"

        # Execute step
        try:
            result = await adapter.run(
                task_config=task_config_with_name,
                input_files=input_files,
                input_data=input_data
            )
        except Exception as e:
            log.error(f"Step {step_name} failed with exception: {e}", exc_info=True)
            return {
                "success": False,
                "steps": state["steps_completed"],
                "final_state": state,
                "metadata": {"error": str(e), "failed_step": step_name}
            }

        # Check if step succeeded
        if not result.get("success", False):
            log.error(f"Step {step_name} failed")
            log.error(f"Summary: {result.get('summary')}")
            return {
                "success": False,
                "steps": state["steps_completed"],
                "final_state": state,
                "metadata": result.get("metadata", {})
            }

        # Step succeeded - update state
        log.info(f"Step {step_name} completed successfully")
        log.info(f"Summary: {result['summary']}")

        # Log to WandB if available and initialized (by parent)
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                # Extract metrics from result
                metrics = result.get("output_files", {}).get("scores", {})
                embeddings = result.get("output_files", {}).get("embeddings")
                
                log_dict = {f"step_{i}/{step_name}/success": 1}
                
                # Add scalar metrics
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            log_dict[f"step_{i}/{step_name}/{key}"] = value
                
                # Add shape information
                if embeddings is not None and hasattr(embeddings, 'shape'):
                    log_dict[f"step_{i}/{step_name}/n_samples"] = embeddings.shape[0]
                    log_dict[f"step_{i}/{step_name}/n_components"] = embeddings.shape[1]
                
                wandb.log(log_dict, step=i)
                log.info(f"Logged metrics to WandB for step {i}")
            except Exception as e:
                log.warning(f"Failed to log to WandB: {e}")

        # Update state for next step
        state["steps_completed"].append({
            "name": step_name,
            "agent": agent_name,
            "result": result
        })

        # Pass data in-memory to next step (if available)
        if "embeddings" in result.get("output_files", {}):
            state["current_data"] = result["output_files"]["embeddings"]

        # Accumulate output files
        state["output_files"].update(result.get("output_files", {}))

        # Accumulate metadata
        if "metadata" in result:
            state["metadata"][step_name] = result["metadata"]

    log.info("\n" + "="*60)
    log.info("WORKFLOW COMPLETED SUCCESSFULLY")
    log.info(f"Total steps executed: {len(state['steps_completed'])}")
    log.info("="*60)

    return {
        "success": True,
        "steps": state["steps_completed"],
        "final_state": state,
        "metadata": {"total_steps": len(steps)}
    }


# ============================================================================
# HYDRA ENTRY POINT
# ============================================================================

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> Dict[str, Any]:
    """
    Smart CLI router that automatically detects and executes the appropriate mode.

    Inspects the configuration:
    - If 'scenarios' key present and non-empty → routes to run_experiment()
    - If 'workflow' key present and non-empty → routes to execute_workflow_chain()

    Usage:
        manyagents experiment=single_algorithm       # workflow mode
        manyagents experiment=invariance_golden      # experiment mode
    """
    log.info("Starting ManyAgents")
    log.info(f"Config name: {cfg.name}")

    # Smart routing: detect execution mode from config
    has_scenarios = hasattr(cfg, 'scenarios') and cfg.scenarios and len(cfg.scenarios) > 0
    has_workflow = hasattr(cfg, 'workflow') and cfg.workflow and cfg.workflow.get('steps')

    if has_scenarios:
        # Route to invariance experiment engine
        log.info("Detected scenarios config → running experiment mode")
        from manyagents.experiment.hydra_runner import run_experiment
        return asyncio.run(run_experiment(cfg))

    elif has_workflow:
        # Route to workflow engine
        log.info("Detected workflow config → running workflow mode")
        output_dir = Path(cfg.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        result = asyncio.run(execute_workflow_chain(cfg, output_dir))

        if result["success"]:
            log.info("\n" + "="*60)
            log.info("WORKFLOW RESULTS")
            log.info("="*60)
            for step in result["steps"]:
                log.info(f"\n{step['name']}:")
                log.info(f"  Summary: {step['result']['summary']}")
            log.info("\n" + "="*60)
        else:
            log.error("Workflow failed")
            log.error(f"Error: {result.get('metadata', {}).get('error', 'Unknown')}")

        return result

    else:
        log.error("No valid config detected.")
        log.info("Use experiment=invariance_golden for experiments")
        log.info("Use experiment=single_algorithm for workflows")
        return {"success": False, "error": "No valid config"}


if __name__ == "__main__":
    main()
