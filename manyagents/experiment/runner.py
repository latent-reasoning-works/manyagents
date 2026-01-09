"""
Experiment runner for pipeline invariance testing.

Handles parallel dispatch to multiple AI systems and results aggregation.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .extractor import extract_methods, check_ground_truth_match
from .metrics import compute_system_metrics, generate_summary_table
from ._shared import (
    extract_raw_response,
    build_success_result,
    build_error_result,
    add_ground_truth_matching,
    generate_experiment_id,
)

log = logging.getLogger(__name__)


async def run_single_agent(
    adapter_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a single agent with the given prompt.

    Args:
        adapter_name: Name of adapter ('claude', 'openai', 'biomni')
        prompt: The prompt to send
        system_prompt: Optional system prompt
        model: Optional model override

    Returns:
        Result dict with raw_response, extracted_methods, etc.
    """
    from manyagents.main import ADAPTER_REGISTRY

    if adapter_name not in ADAPTER_REGISTRY:
        return build_error_result(f"Unknown adapter: {adapter_name}")

    adapter = ADAPTER_REGISTRY[adapter_name]()

    task_config = {'prompt': prompt}
    if system_prompt:
        task_config['system_prompt'] = system_prompt
    if model:
        task_config['model'] = model

    try:
        result = await adapter.run(task_config, {})

        if result['success']:
            raw_response = extract_raw_response(result.get('output_files', {}))
            return build_success_result(raw_response, result.get('metadata', {}))
        return build_error_result(
            result.get('summary', 'Unknown error'),
            result.get('metadata', {})
        )

    except Exception as e:
        log.error(f"Error running {adapter_name}: {e}", exc_info=True)
        return build_error_result(str(e))


async def run_parallel_agents(
    prompt: str,
    target_systems: List[str],
    system_prompt: Optional[str] = None,
    models: Optional[Dict[str, str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run the same prompt against multiple AI systems in parallel.

    Args:
        prompt: The prompt to send to all systems
        target_systems: List of adapter names ('claude', 'openai', etc.)
        system_prompt: Optional system prompt for all
        models: Optional dict mapping system name to model override

    Returns:
        Dict mapping system name to result dict
    """
    models = models or {}

    tasks = [
        (system, run_single_agent(system, prompt, system_prompt, models.get(system)))
        for system in target_systems
    ]

    gathered = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

    return {
        system: build_error_result(str(result)) if isinstance(result, Exception) else result
        for (system, _), result in zip(tasks, gathered)
    }


async def run_invariance_experiment(
    scenarios: Dict[str, Dict[str, Any]],
    target_systems: List[str],
    system_prompt: Optional[str] = None,
    output_dir: Optional[Path] = None,
    model_overrides: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Run the full pipeline invariance experiment.

    Args:
        scenarios: Dict mapping scenario_id to scenario config
        target_systems: List of AI systems to query
        system_prompt: Optional system prompt for all queries
        output_dir: Optional directory to save results
        model_overrides: Optional dict mapping system name to model name

    Returns:
        Full experiment results with metrics
    """
    experiment_id = generate_experiment_id()

    log.info(f"Starting experiment {experiment_id}")
    log.info(f"Scenarios: {list(scenarios.keys())}")
    log.info(f"Target systems: {target_systems}")
    if model_overrides:
        log.info(f"Model overrides: {model_overrides}")

    # Initialize results structure
    all_results = {system: {} for system in target_systems}

    # Run all scenarios
    for scenario_id, scenario in scenarios.items():
        log.info(f"Running scenario {scenario_id}...")

        scenario_results = await run_parallel_agents(
            prompt=scenario['text'],
            target_systems=target_systems,
            system_prompt=system_prompt,
            models=model_overrides
        )

        for system, result in scenario_results.items():
            add_ground_truth_matching(result, scenario)
            all_results[system][scenario_id] = result

    # Compute metrics
    metrics = {
        system: compute_system_metrics(all_results[system], scenarios)
        for system in target_systems
    }

    # Assemble results
    experiment_results = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'prompts': {
            sid: {
                'text': s['text'],
                'expected_geometry': s.get('expected_geometry', 'unknown'),
                'ground_truth_methods': s.get('ground_truth_methods', [])
            }
            for sid, s in scenarios.items()
        },
        'results': all_results,
        'metrics': metrics
    }

    # Save results if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / f"{experiment_id}_results.json"
        with open(results_path, 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)
        log.info(f"Results saved to {results_path}")

        summary_md = generate_summary_table(experiment_results, format='markdown')
        summary_path = output_dir / f"{experiment_id}_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Pipeline Invariance Experiment Results\n\n")
            f.write(f"**Experiment ID:** {experiment_id}\n")
            f.write(f"**Timestamp:** {experiment_results['timestamp']}\n\n")
            f.write("## Summary Metrics\n\n")
            f.write(summary_md)
            f.write("\n\n## Interpretation\n\n")
            f.write("- **Jaccard (Invariance):** Higher = more similar recommendations across prompts (BAD for geometry-aware systems)\n")
            f.write("- **Ground Truth Match:** Higher = recommendations match expected methods for each geometry type (GOOD)\n")
            f.write("- **Clustering-for-All:** Higher = always recommends clustering regardless of data structure (BAD)\n")
        log.info(f"Summary saved to {summary_path}")

    return experiment_results


def run_experiment_sync(
    scenarios: Dict[str, Dict[str, Any]],
    target_systems: List[str],
    system_prompt: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Synchronous wrapper for run_invariance_experiment."""
    return asyncio.run(
        run_invariance_experiment(scenarios, target_systems, system_prompt, output_dir)
    )
