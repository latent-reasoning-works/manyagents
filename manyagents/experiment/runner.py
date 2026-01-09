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
    # Import here to avoid circular imports
    from manyagents.main import ADAPTER_REGISTRY

    if adapter_name not in ADAPTER_REGISTRY:
        return {
            'success': False,
            'error': f"Unknown adapter: {adapter_name}",
            'raw_response': None
        }

    adapter_class = ADAPTER_REGISTRY[adapter_name]
    adapter = adapter_class()

    task_config = {'prompt': prompt}
    if system_prompt:
        task_config['system_prompt'] = system_prompt
    if model:
        task_config['model'] = model

    try:
        result = await adapter.run(task_config, {})

        if result['success']:
            # Extract raw response text
            raw_response = None
            if 'raw_response' in result.get('output_files', {}):
                response_path = result['output_files']['raw_response']
                if isinstance(response_path, Path):
                    raw_response = response_path.read_text()
                else:
                    raw_response = str(response_path)

            # Extract methods from response
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
                'raw_response': None,
                'metadata': result.get('metadata', {})
            }

    except Exception as e:
        log.error(f"Error running {adapter_name}: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'raw_response': None
        }


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

    tasks = []
    for system in target_systems:
        model = models.get(system)
        task = run_single_agent(system, prompt, system_prompt, model)
        tasks.append((system, task))

    # Run all in parallel
    results = {}
    gathered = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

    for (system, _), result in zip(tasks, gathered):
        if isinstance(result, Exception):
            results[system] = {
                'success': False,
                'error': str(result),
                'raw_response': None
            }
        else:
            results[system] = result

    return results


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
        scenarios: Dict mapping scenario_id to scenario config:
            {
                "text": "prompt text",
                "expected_geometry": "trajectory|discrete|periodic|continuous",
                "ground_truth_methods": ["method1", "method2"],
                "failure_indicators": ["bad_method"]
            }
        target_systems: List of AI systems to query
        system_prompt: Optional system prompt for all queries
        output_dir: Optional directory to save results
        model_overrides: Optional dict mapping system name to model name

    Returns:
        Full experiment results with metrics
    """
    experiment_id = f"invariance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log.info(f"Starting experiment {experiment_id}")
    log.info(f"Scenarios: {list(scenarios.keys())}")
    log.info(f"Target systems: {target_systems}")
    if model_overrides:
        log.info(f"Model overrides: {model_overrides}")

    # Run all scenarios
    all_results = {}
    for system in target_systems:
        all_results[system] = {}

    for scenario_id, scenario in scenarios.items():
        prompt = scenario['text']
        log.info(f"Running scenario {scenario_id}...")

        # Run against all systems in parallel
        scenario_results = await run_parallel_agents(
            prompt=prompt,
            target_systems=target_systems,
            system_prompt=system_prompt,
            models=model_overrides
        )

        # Store results
        for system, result in scenario_results.items():
            # Add ground truth matching
            if result['success']:
                is_match, match_details = check_ground_truth_match(
                    result.get('extracted_methods', []),
                    scenario.get('ground_truth_methods', []),
                    scenario.get('failure_indicators', [])
                )
                result['matches_ground_truth'] = is_match
                result['ground_truth_details'] = match_details

            all_results[system][scenario_id] = result

    # Compute metrics for each system
    metrics = {}
    for system in target_systems:
        metrics[system] = compute_system_metrics(
            all_results[system],
            scenarios
        )

    # Assemble full results
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

        # Save full results
        results_path = output_dir / f"{experiment_id}_results.json"
        with open(results_path, 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)
        log.info(f"Results saved to {results_path}")

        # Save summary table
        summary_md = generate_summary_table(experiment_results, format='markdown')
        summary_path = output_dir / f"{experiment_id}_summary.md"
        with open(summary_path, 'w') as f:
            f.write(f"# Pipeline Invariance Experiment Results\n\n")
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
