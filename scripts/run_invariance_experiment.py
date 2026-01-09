#!/usr/bin/env python3
"""
Run pipeline invariance experiment.

Usage:
    # Run golden test only
    python scripts/run_invariance_experiment.py --golden

    # Run full experiment
    python scripts/run_invariance_experiment.py --full

    # Specify systems
    python scripts/run_invariance_experiment.py --golden --systems claude,openai
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from manyagents.experiment.runner import run_invariance_experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def load_test_config(config_path: Path) -> dict:
    """Load test configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


async def run_golden_test(systems: list[str]) -> dict:
    """Run the golden trajectory test."""
    config_path = project_root / "tests" / "golden_trajectory_test.json"
    config = load_test_config(config_path)

    log.info("=" * 60)
    log.info("RUNNING GOLDEN TRAJECTORY TEST")
    log.info("=" * 60)

    # Override systems if specified
    target_systems = systems if systems else config.get('target_systems', ['claude', 'openai'])
    model_overrides = config.get('model_overrides', {})

    results = await run_invariance_experiment(
        scenarios=config['scenarios'],
        target_systems=target_systems,
        system_prompt=config.get('system_prompt'),
        output_dir=project_root / "results",
        model_overrides=model_overrides
    )

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("GOLDEN TEST RESULTS")
    log.info("=" * 60)

    for system, metrics in results.get('metrics', {}).items():
        log.info(f"\n{system}:")
        log.info(f"  Ground Truth Match: {metrics.get('ground_truth_match_rate', 0):.1%}")
        log.info(f"  Clustering-for-All: {metrics.get('clustering_for_all_rate', 0):.1%}")

    return results


async def run_full_experiment(systems: list[str]) -> dict:
    """Run the full 4-scenario experiment."""
    config_path = project_root / "tests" / "experiment_config.json"
    config = load_test_config(config_path)

    log.info("=" * 60)
    log.info("RUNNING FULL PIPELINE INVARIANCE EXPERIMENT")
    log.info("=" * 60)

    # Override systems if specified
    target_systems = systems if systems else config.get('target_systems', ['claude', 'openai', 'biomni'])
    model_overrides = config.get('model_overrides', {})

    results = await run_invariance_experiment(
        scenarios=config['scenarios'],
        target_systems=target_systems,
        system_prompt=config.get('system_prompt'),
        output_dir=project_root / "results",
        model_overrides=model_overrides
    )

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("FULL EXPERIMENT RESULTS")
    log.info("=" * 60)

    for system, metrics in results.get('metrics', {}).items():
        log.info(f"\n{system}:")
        log.info(f"  Jaccard Similarity (Invariance): {metrics.get('jaccard_similarity_across_prompts', 0):.2f}")
        log.info(f"  Ground Truth Match Rate: {metrics.get('ground_truth_match_rate', 0):.1%}")
        log.info(f"  Clustering-for-All Rate: {metrics.get('clustering_for_all_rate', 0):.1%}")

    # Interpretation
    log.info("\n" + "=" * 60)
    log.info("INTERPRETATION")
    log.info("=" * 60)
    log.info("- High Jaccard (>0.7): System shows 'pipeline regurgitation' - same recommendations regardless of data")
    log.info("- Low Ground Truth Match: System is not geometry-aware")
    log.info("- High Clustering-for-All: System defaults to clustering even for non-cluster geometries")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run pipeline invariance experiment")
    parser.add_argument('--golden', action='store_true', help='Run golden trajectory test')
    parser.add_argument('--full', action='store_true', help='Run full 4-scenario experiment')
    parser.add_argument('--systems', type=str, help='Comma-separated list of systems (e.g., claude,openai)')

    args = parser.parse_args()

    # Parse systems
    systems = args.systems.split(',') if args.systems else []

    if not args.golden and not args.full:
        # Default to golden test
        args.golden = True

    if args.golden:
        asyncio.run(run_golden_test(systems))

    if args.full:
        asyncio.run(run_full_experiment(systems))


if __name__ == '__main__':
    main()
