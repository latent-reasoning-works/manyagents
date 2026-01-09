"""Loader for geometric reasoning prompt scenarios.

Reads YAML config files from prompts/geometric_reasoning/ and yields
structured prompt data for experiments.

Usage:
    from manyagents.experiment.prompt_loader import load_scenarios, iter_prompts

    # Load all scenarios
    scenarios = load_scenarios()

    # Iterate over all (scenario, condition, prompt) combinations
    for scenario_id, condition, prompt_data in iter_prompts():
        print(f"{scenario_id}/{condition}: {prompt_data['prompt'][:50]}...")

    # Get specific condition
    scenarios = load_scenarios()
    prompt_b = scenarios['immunology_discrete']['conditions']['B']['prompt']
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

log = logging.getLogger(__name__)

# Default path to geometric reasoning prompts
DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent / "configs" / "prompts" / "geometric_reasoning"

# Condition order for consistent iteration
CONDITION_ORDER = ["A", "B", "C"]


def load_scenario(filepath: Path) -> Dict[str, Any]:
    """Load a single scenario YAML file.

    Args:
        filepath: Path to the scenario YAML file

    Returns:
        Parsed scenario dict with metadata, geometry, ground_truth, conditions
    """
    with open(filepath) as f:
        scenario = yaml.safe_load(f)

    # Validate required fields
    required = ["scenario_name", "geometry", "ground_truth", "conditions"]
    missing = [f for f in required if f not in scenario]
    if missing:
        raise ValueError(f"Scenario {filepath.name} missing required fields: {missing}")

    # Validate conditions
    for cond in CONDITION_ORDER:
        if cond not in scenario["conditions"]:
            raise ValueError(f"Scenario {filepath.name} missing condition {cond}")

    return scenario


def load_scenarios(
    prompts_dir: Optional[Path] = None,
    scenarios: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load all scenario files from the prompts directory.

    Args:
        prompts_dir: Path to prompts directory (default: configs/prompts/geometric_reasoning)
        scenarios: Optional list of scenario names to load (default: all)

    Returns:
        Dict mapping scenario_name -> scenario dict
    """
    prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR

    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    result = {}

    for filepath in sorted(prompts_dir.glob("*.yaml")):
        # Skip schema file
        if filepath.name.startswith("_"):
            continue

        try:
            scenario = load_scenario(filepath)
            scenario_name = scenario["scenario_name"]

            # Filter if specific scenarios requested
            if scenarios and scenario_name not in scenarios:
                continue

            result[scenario_name] = scenario
            log.debug(f"Loaded scenario: {scenario_name}")

        except Exception as e:
            log.warning(f"Failed to load {filepath.name}: {e}")

    log.info(f"Loaded {len(result)} scenarios from {prompts_dir}")
    return result


def iter_prompts(
    prompts_dir: Optional[Path] = None,
    scenarios: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """Iterate over all (scenario, condition, prompt_data) combinations.

    Args:
        prompts_dir: Path to prompts directory
        scenarios: Optional list of scenario names to include
        conditions: Optional list of conditions to include (default: A, B, C)

    Yields:
        Tuples of (scenario_name, condition_letter, prompt_data)
        prompt_data includes: name, description, prompt, and parent scenario metadata
    """
    all_scenarios = load_scenarios(prompts_dir, scenarios)
    conditions = conditions or CONDITION_ORDER

    for scenario_name, scenario in all_scenarios.items():
        for cond in conditions:
            if cond not in scenario["conditions"]:
                continue

            cond_data = scenario["conditions"][cond]

            # Build prompt_data with full context
            prompt_data = {
                "prompt": cond_data["prompt"].strip(),
                "condition_name": cond_data.get("name", cond),
                "condition_description": cond_data.get("description", ""),
                # Include scenario metadata for evaluation
                "scenario_name": scenario_name,
                "domain": scenario.get("domain", ""),
                "geometry_type": scenario["geometry"]["type"],
                "expected_structure": scenario["geometry"]["expected_structure"],
                "ground_truth": scenario["ground_truth"],
            }

            yield scenario_name, cond, prompt_data


def scenarios_to_hydra_config(
    prompts_dir: Optional[Path] = None,
    scenarios: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Convert loaded scenarios to Hydra-compatible config format.

    Returns a dict that can be used as the 'scenarios' key in experiment config.

    Args:
        prompts_dir: Path to prompts directory
        scenarios: Optional list of scenario names to include
        conditions: Optional list of conditions to include

    Returns:
        Dict mapping "{scenario}_{condition}" -> scenario config for Hydra
    """
    result = {}

    for scenario_name, cond, prompt_data in iter_prompts(prompts_dir, scenarios, conditions):
        key = f"{scenario_name}_{cond}"

        result[key] = {
            "text": prompt_data["prompt"],
            "expected_geometry": prompt_data["geometry_type"],
            "ground_truth_methods": prompt_data["ground_truth"]["appropriate_methods"],
            "failure_indicators": prompt_data["ground_truth"].get("failure_indicators", []),
            # Extra metadata
            "scenario": scenario_name,
            "condition": cond,
            "domain": prompt_data["domain"],
        }

    return result


def get_scenario_matrix() -> Dict[str, List[str]]:
    """Get the scenario × condition matrix dimensions.

    Returns:
        Dict with 'scenarios' and 'conditions' lists
    """
    scenarios = load_scenarios()
    return {
        "scenarios": list(scenarios.keys()),
        "conditions": CONDITION_ORDER,
        "total_prompts": len(scenarios) * len(CONDITION_ORDER),
    }
