"""
Hydra-based experiment runner for pipeline invariance testing.

Usage:
    manyagents-experiment experiment=invariance_golden
    manyagents-experiment experiment=invariance_full active_agents=[claude,openai]
    manyagents-experiment experiment=invariance_golden agents.local_llm.config.model=llama-3.3-70b
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from .metrics import compute_system_metrics
from ._shared import (
    extract_raw_response,
    build_success_result,
    build_error_result,
    add_ground_truth_matching,
    save_experiment_results,
    print_metrics_summary,
    generate_experiment_id,
)

log = logging.getLogger(__name__)


async def run_agent(
    agent_config: DictConfig,
    prompt: str,
    system_prompt: str
) -> Dict[str, Any]:
    """Run a single agent with the given prompt."""
    from manyagents.main import ADAPTER_REGISTRY

    adapter_name = agent_config.adapter
    if adapter_name not in ADAPTER_REGISTRY:
        return build_error_result(f"Unknown adapter: {adapter_name}")

    adapter = ADAPTER_REGISTRY[adapter_name]()

    task_config = dict(agent_config.config)
    task_config['prompt'] = prompt
    task_config['system_prompt'] = system_prompt

    try:
        result = await adapter.run(task_config, {})

        if result['success']:
            raw_response = extract_raw_response(result.get('output_files', {}))
            return build_success_result(raw_response, result.get('metadata', {}))
        return build_error_result(result.get('summary', 'Unknown error'))

    except Exception as e:
        log.error(f"Error running {adapter_name}: {e}", exc_info=True)
        return build_error_result(str(e))


def _get_agent_config(cfg: DictConfig, agent_name: str) -> DictConfig:
    """Extract agent config, handling nested 'agent' key from Hydra defaults."""
    agent_entry = cfg.agents[agent_name]
    return agent_entry.agent if hasattr(agent_entry, 'agent') else agent_entry


async def run_experiment(cfg: DictConfig) -> Dict[str, Any]:
    """Run the full experiment based on Hydra config."""
    experiment_id = generate_experiment_id(cfg.name)

    log.info(f"Starting experiment: {experiment_id}")
    log.info(f"Active agents: {cfg.active_agents}")
    log.info(f"Scenarios: {list(cfg.scenarios.keys())}")

    all_results = {agent: {} for agent in cfg.active_agents}
    scenarios_dict = {}

    for scenario_id, scenario in cfg.scenarios.items():
        scenario_dict = OmegaConf.to_container(scenario, resolve=True)
        scenarios_dict[scenario_id] = scenario_dict
        prompt = scenario_dict.get('text', scenario_dict.get('prompt', ''))

        log.info(f"Running scenario: {scenario_id}")

        # Build and run agent tasks
        tasks = []
        for agent_name in cfg.active_agents:
            if agent_name not in cfg.agents:
                log.warning(f"Agent {agent_name} not configured, skipping")
                continue
            agent_config = _get_agent_config(cfg, agent_name)
            tasks.append((agent_name, run_agent(agent_config, prompt, cfg.system_prompt)))

        gathered = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        for (agent_name, _), result in zip(tasks, gathered):
            if isinstance(result, Exception):
                result = build_error_result(str(result))

            add_ground_truth_matching(result, scenario_dict)
            all_results[agent_name][scenario_id] = result
            log.info(f"  {agent_name}: {'SUCCESS' if result.get('success') else 'FAILED'}")

    # Compute metrics
    metrics = {
        agent_name: compute_system_metrics(all_results[agent_name], scenarios_dict)
        for agent_name in cfg.active_agents
        if agent_name in all_results
    }

    experiment_results = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'scenarios': scenarios_dict,
        'results': all_results,
        'metrics': metrics
    }

    save_experiment_results(
        experiment_results,
        Path(cfg.output_dir),
        experiment_id
    )

    print_metrics_summary(metrics)

    return experiment_results


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for experiments."""
    if not hasattr(cfg, 'scenarios') or not cfg.scenarios:
        log.error("No scenarios defined. Use experiment=invariance_golden or similar.")
        return

    if not hasattr(cfg, 'active_agents') or not cfg.active_agents:
        log.error("No active_agents defined.")
        return

    asyncio.run(run_experiment(cfg))


if __name__ == "__main__":
    main()
