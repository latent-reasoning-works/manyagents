"""
Experiment runner for pipeline invariance testing.

Called by manyagents.main when scenarios config is detected.

Usage (via main CLI):
    manyagents experiment=invariance_golden
    manyagents experiment=invariance_full active_agents=[claude,openai]
    manyagents experiment=invariance_golden wandb.enabled=true
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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
from .logger import ExperimentLogger, NullLogger

log = logging.getLogger(__name__)


def _create_logger(cfg: DictConfig, experiment_id: str) -> ExperimentLogger:
    """Create an experiment logger based on config."""
    wandb_cfg = getattr(cfg, 'wandb', None)

    if wandb_cfg is None or not getattr(wandb_cfg, 'enabled', False):
        return NullLogger()

    tags = getattr(wandb_cfg, 'tags', [])
    tags = list(tags) if tags else None

    return ExperimentLogger(
        project=getattr(wandb_cfg, 'project', 'manyagents'),
        experiment_name=experiment_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        entity=getattr(wandb_cfg, 'entity', None),
        tags=tags,
        enabled=True,
    )


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

    # Create logger
    logger = _create_logger(cfg, experiment_id)

    log.info(f"Starting experiment: {experiment_id}")
    log.info(f"Active agents: {cfg.active_agents}")
    log.info(f"Scenarios: {list(cfg.scenarios.keys())}")

    all_results = {agent: {} for agent in cfg.active_agents}
    scenarios_dict = {}

    # Build scenarios dict first for logging
    for scenario_id, scenario in cfg.scenarios.items():
        scenarios_dict[scenario_id] = OmegaConf.to_container(scenario, resolve=True)

    # Log config
    logger.log_config(
        scenarios_dict,
        list(cfg.active_agents),
        cfg.system_prompt,
        None  # model_overrides from agent configs
    )

    for scenario_id in cfg.scenarios.keys():
        scenario_dict = scenarios_dict[scenario_id]
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

            # Log per-scenario result
            logger.log_scenario_result(agent_name, scenario_id, result)

    # Compute metrics
    metrics = {
        agent_name: compute_system_metrics(all_results[agent_name], scenarios_dict)
        for agent_name in cfg.active_agents
        if agent_name in all_results
    }

    # Log system metrics and summary table
    for agent_name, agent_metrics in metrics.items():
        logger.log_system_metrics(agent_name, agent_metrics)
    logger.log_summary_table(metrics)

    experiment_results = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'scenarios': scenarios_dict,
        'results': all_results,
        'metrics': metrics
    }

    output_dir = Path(cfg.output_dir)
    save_experiment_results(
        experiment_results,
        output_dir,
        experiment_id
    )

    # Log visualizations and artifacts
    logger.create_visualizations(experiment_results)
    logger.save_artifacts(experiment_results, output_dir, experiment_id)

    # Finish logging
    wandb_url = logger.finish()
    if wandb_url:
        experiment_results['wandb_url'] = wandb_url
        log.info(f"wandb run: {wandb_url}")

    print_metrics_summary(metrics)

    return experiment_results


# Note: Entry point is now manyagents.main which routes here when scenarios detected
