"""
Hydra-based experiment runner for pipeline invariance testing.

Usage:
    manyagents-experiment experiment=invariance_golden
    manyagents-experiment experiment=invariance_full active_agents=[claude,openai]
    manyagents-experiment experiment=invariance_golden agents.local_llm.config.model=llama-3.3-70b
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

from .extractor import extract_methods, check_ground_truth_match
from .metrics import compute_system_metrics, generate_summary_table

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
        return {
            'success': False,
            'error': f"Unknown adapter: {adapter_name}",
            'raw_response': None
        }

    adapter_class = ADAPTER_REGISTRY[adapter_name]
    adapter = adapter_class()

    # Build task config from agent config
    task_config = dict(agent_config.config)
    task_config['prompt'] = prompt
    task_config['system_prompt'] = system_prompt

    try:
        result = await adapter.run(task_config, {})

        if result['success']:
            raw_response = None
            if 'raw_response' in result.get('output_files', {}):
                response_path = result['output_files']['raw_response']
                if isinstance(response_path, Path):
                    raw_response = response_path.read_text()
                else:
                    raw_response = str(response_path)

            extraction = extract_methods(raw_response or '') if raw_response else {}

            return {
                'success': True,
                'raw_response': raw_response,
                'extracted_methods': extraction.get('extracted_methods', []),
                'mentions_clustering': extraction.get('mentions_clustering', False),
                'mentions_trajectory': extraction.get('mentions_trajectory', False),
                'mentions_data_inspection': extraction.get('mentions_data_inspection', False),
                'metadata': result.get('metadata', {})
            }
        else:
            return {
                'success': False,
                'error': result.get('summary', 'Unknown error'),
                'raw_response': None
            }

    except Exception as e:
        log.error(f"Error running {adapter_name}: {e}", exc_info=True)
        return {'success': False, 'error': str(e), 'raw_response': None}


async def run_experiment(cfg: DictConfig) -> Dict[str, Any]:
    """Run the full experiment based on Hydra config."""
    experiment_id = f"{cfg.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log.info(f"Starting experiment: {experiment_id}")
    log.info(f"Active agents: {cfg.active_agents}")
    log.info(f"Scenarios: {list(cfg.scenarios.keys())}")

    # Initialize results
    all_results = {agent: {} for agent in cfg.active_agents}
    scenarios_dict = {}

    # Run each scenario
    for scenario_id, scenario in cfg.scenarios.items():
        scenario_dict = OmegaConf.to_container(scenario, resolve=True)
        scenarios_dict[scenario_id] = scenario_dict
        prompt = scenario_dict.get('text', scenario_dict.get('prompt', ''))

        log.info(f"Running scenario: {scenario_id}")

        # Run each agent
        tasks = []
        for agent_name in cfg.active_agents:
            if agent_name not in cfg.agents:
                log.warning(f"Agent {agent_name} not configured, skipping")
                continue
            # Agent config may be nested under 'agent' key due to Hydra defaults
            agent_entry = cfg.agents[agent_name]
            agent_config = agent_entry.agent if hasattr(agent_entry, 'agent') else agent_entry
            task = run_agent(agent_config, prompt, cfg.system_prompt)
            tasks.append((agent_name, task))

        # Execute in parallel
        gathered = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        for (agent_name, _), result in zip(tasks, gathered):
            if isinstance(result, Exception):
                result = {'success': False, 'error': str(result), 'raw_response': None}

            # Add ground truth matching
            if result.get('success'):
                is_match, match_details = check_ground_truth_match(
                    result.get('extracted_methods', []),
                    scenario_dict.get('ground_truth_methods', []),
                    scenario_dict.get('failure_indicators', [])
                )
                result['matches_ground_truth'] = is_match
                result['ground_truth_details'] = match_details

            all_results[agent_name][scenario_id] = result
            log.info(f"  {agent_name}: {'SUCCESS' if result.get('success') else 'FAILED'}")

    # Compute metrics
    metrics = {}
    for agent_name in cfg.active_agents:
        if agent_name in all_results:
            metrics[agent_name] = compute_system_metrics(
                all_results[agent_name],
                scenarios_dict
            )

    # Assemble results
    experiment_results = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'scenarios': scenarios_dict,
        'results': all_results,
        'metrics': metrics
    }

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")

    # Generate summary
    summary_md = generate_summary_table(experiment_results, format='markdown')
    summary_path = output_dir / "summary.md"
    with open(summary_path, 'w') as f:
        f.write(f"# {cfg.name}\n\n")
        f.write(f"**Experiment ID:** {experiment_id}\n")
        f.write(f"**Timestamp:** {experiment_results['timestamp']}\n\n")
        f.write("## Results\n\n")
        f.write(summary_md)
        f.write("\n\n## Interpretation\n\n")
        f.write("- **Jaccard (Invariance):** Higher = same recommendations across prompts (BAD)\n")
        f.write("- **Ground Truth Match:** Higher = geometry-aware recommendations (GOOD)\n")
        f.write("- **Clustering-for-All:** Higher = always recommends clustering (BAD)\n")
    log.info(f"Summary saved to {summary_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    for agent_name, m in metrics.items():
        print(f"\n{agent_name}:")
        print(f"  Jaccard Similarity: {m.get('jaccard_similarity_across_prompts', 0):.2f}")
        print(f"  Ground Truth Match: {m.get('ground_truth_match_rate', 0):.1%}")
        print(f"  Clustering-for-All: {m.get('clustering_for_all_rate', 0):.1%}")

    return experiment_results


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for experiments."""
    # Validate experiment config
    if not hasattr(cfg, 'scenarios') or not cfg.scenarios:
        log.error("No scenarios defined. Use experiment=invariance_golden or similar.")
        return

    if not hasattr(cfg, 'active_agents') or not cfg.active_agents:
        log.error("No active_agents defined.")
        return

    asyncio.run(run_experiment(cfg))


if __name__ == "__main__":
    main()
