"""
Smart CLI router for ManyAgents.

Routes to appropriate execution engine based on config:
- If 'scenarios' key present → experiment mode
- If 'workflow' key present → workflow mode
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from manyagents.workflow import WorkflowState, log_step_to_wandb

# Register shop's Hydra launchers
try:
    from shop.hydra.config_store import register_shop_launchers
    register_shop_launchers()
except ImportError:
    pass

from manyagents.adapters.openai_adapter import OpenAIAdapter
from manyagents.adapters.cellforge_adapter import CellForgeAdapter
from manyagents.adapters.kosmos_adapter import KosmosAdapter
from manyagents.adapters.claude_adapter import ClaudeAdapter
from manyagents.adapters.local_llm_adapter import LocalLLMAdapter
from manyagents.adapters.mock_adapter import MockAdapter

log = logging.getLogger(__name__)

# Optional imports (TODO: make required when repos are public)
try:
    from manyagents.adapters.manylatents_adapter import ManyLatentsAdapter
    MANYLATENTS_AVAILABLE = True
except ImportError:
    ManyLatentsAdapter = None
    MANYLATENTS_AVAILABLE = False

try:
    from manyagents.adapters.biomni_adapter import BiomniAdapter
    BIOMNI_AVAILABLE = True
except ImportError:
    BiomniAdapter = None
    BIOMNI_AVAILABLE = False

# Geomancer integration
try:
    geomancer_path = Path(__file__).resolve().parents[2] / "Geomancer"
    if geomancer_path.exists() and str(geomancer_path) not in sys.path:
        sys.path.insert(0, str(geomancer_path))
    from geomancy.logging_context import LoggingContext
    GEOMANCER_AVAILABLE = True
except ImportError:
    GEOMANCER_AVAILABLE = False

# ============================================================================
# ADAPTER REGISTRY
# ============================================================================

ADAPTER_REGISTRY = {
    "openai": OpenAIAdapter,
    "cellforge": CellForgeAdapter,
    "kosmos": KosmosAdapter,
    "claude": ClaudeAdapter,
    "local_llm": LocalLLMAdapter,
    "mock": MockAdapter,
}

if MANYLATENTS_AVAILABLE:
    ADAPTER_REGISTRY["manylatents"] = ManyLatentsAdapter
if BIOMNI_AVAILABLE:
    ADAPTER_REGISTRY["biomni"] = BiomniAdapter


# ============================================================================
# ORCHESTRATION ENGINE
# ============================================================================

async def execute_workflow_chain(
    workflow_config: DictConfig,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Execute a workflow chain using the adapter pattern."""
    log.info("Starting workflow execution")

    state = WorkflowState()
    steps = workflow_config.workflow.steps
    log.info(f"Workflow has {len(steps)} steps")

    for i, step_config in enumerate(steps):
        step_name = step_config.get("name", f"step_{i}")
        agent_name = step_config.get("agent")
        task_config = dict(step_config.get("config", {}))

        log.info(f"\n{'='*60}\nStep {i+1}/{len(steps)}: {step_name} ({agent_name})\n{'='*60}")

        if GEOMANCER_AVAILABLE:
            LoggingContext.set_step_context(i, step_name)

        # Validate agent
        if agent_name not in ADAPTER_REGISTRY:
            return {
                "success": False,
                "steps": state.steps_completed,
                "final_state": state.to_dict(),
                "metadata": {"error": f"Unknown agent '{agent_name}'"},
            }

        # Execute step
        adapter = ADAPTER_REGISTRY[agent_name]()
        task_config["name"] = task_config.get("name", f"step{i}_{step_name}")

        try:
            result = await adapter.run(
                task_config=task_config,
                input_files=state.output_files,
                input_data=state.current_data,
            )
        except Exception as e:
            log.error(f"Step {step_name} failed: {e}", exc_info=True)
            return {
                "success": False,
                "steps": state.steps_completed,
                "final_state": state.to_dict(),
                "metadata": {"error": str(e), "failed_step": step_name},
            }

        if not result.get("success"):
            log.error(f"Step {step_name} failed: {result.get('summary')}")
            return {
                "success": False,
                "steps": state.steps_completed,
                "final_state": state.to_dict(),
                "metadata": result.get("metadata", {}),
            }

        log.info(f"Step {step_name} completed: {result['summary']}")
        state.add_step(step_name, agent_name, result)
        log_step_to_wandb(i, step_name, result)

    log.info(f"\n{'='*60}\nWORKFLOW COMPLETED ({len(state.steps_completed)} steps)\n{'='*60}")

    return {
        "success": True,
        "steps": state.steps_completed,
        "final_state": state.to_dict(),
        "metadata": {"total_steps": len(steps)},
    }


# ============================================================================
# HYDRA ENTRY POINT
# ============================================================================

def _has_scenarios(cfg: DictConfig) -> bool:
    """Check if config has scenarios for experiment mode."""
    return bool(OmegaConf.select(cfg, "scenarios"))


def _has_workflow(cfg: DictConfig) -> bool:
    """Check if config has workflow steps."""
    return bool(OmegaConf.select(cfg, "workflow.steps"))


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> Dict[str, Any]:
    """Smart CLI router - auto-detects experiment vs workflow mode."""
    log.info(f"Starting ManyAgents - {cfg.name}")

    match (_has_scenarios(cfg), _has_workflow(cfg)):
        case (True, _):
            log.info("Detected scenarios config → running experiment mode")
            from manyagents.experiment.hydra_runner import run_experiment
            return asyncio.run(run_experiment(cfg))

        case (_, True):
            log.info("Detected workflow config → running workflow mode")
            if output_dir := cfg.get("output_dir"):
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            result = asyncio.run(execute_workflow_chain(cfg, Path(output_dir) if output_dir else None))

            if result["success"]:
                for step in result["steps"]:
                    log.info(f"{step['name']}: {step['result']['summary']}")
            else:
                log.error(f"Workflow failed: {result.get('metadata', {}).get('error')}")

            return result

        case _:
            log.error("No valid config. Use experiment=X for experiments or workflows.")
            return {"success": False, "error": "No valid config"}


if __name__ == "__main__":
    main()
